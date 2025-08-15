#!/usr/bin/env python3
"""
Local version of orchestrator - runs directly on head node
No Ray Client needed, avoids version mismatch issues
"""

import os
import itertools
import time
import numpy as np
import pandas as pd
import ray
import xgboost as xgb
from itertools import combinations
from typing import Dict, List, Tuple, Any

# ===================== CONFIGURATION =====================
DATA_PATHS = {
    "linux": "/home/monstrcow/mltownhouseeval/sales_2015_2025.csv",
    "wsl":   "/home/monstrcow/mltownhouseeval/sales_2015_2025.csv",
}

CONFIG = {
    # Data processing
    "rolling_window_days": 90,
    "min_fsa_samples": 15,
    "shrink_k": 10,
    "half_life_days": 180,
    "test_fraction": 0.20,
    "target_area_col": "TotFlArea",
    
    # Feature exploration
    "max_features": 4,                 # evaluate 1..max_features
    "batch_size": 300,                 # combos per worker batch
    "seed": 42,
    
    # XGBoost parameters
    "params": {
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "gpu_id": 0,                   # Ray sets CUDA_VISIBLE_DEVICES per actor
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "seed": 42,
        "verbosity": 0,
    },
    "num_rounds": 350,
    "early_stopping": 40,
    
    # Columns to exclude as features (leaky/post-listing)
    "exclude_features": set([
        "List Price", "Price", "Sold Price per SqFt", "SP/LP Ratio", "SP/OLP Ratio",
        "Sold Date", "Confirm Sold Date", "DOM", "Cumulative DOM", "Expiry Date",
        "Sold Price"  # target
    ]),
    
    # Optional hard-include features (pre-listing only). Leave empty to search all numeric safe features.
    "require_at_least_one_of": [],  # e.g., ["TotFlArea","Floor Area Fin - Main Flr"]
}

# ===================== RAY SETUP =====================
print(f"Initializing Ray locally (no client)...")
ray.init(address='auto')  # Connect to existing cluster
print("✓ Connected to Ray cluster")

# ===================== REMOTE WORKER =====================
@ray.remote(num_gpus=1)
class TrainWorker:
    """
    Ray actor that runs on a single GPU
    Loads data once and processes batches of feature combinations
    """
    
    def __init__(self, node_tag: str, data_path_map: Dict[str, str], config: Dict[str, Any]):
        self.node_tag = node_tag  # e.g., "linux" or "wsl"
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg["seed"])
        self.data_path = data_path_map[node_tag]
        self._load_and_prepare()

    def _load_and_prepare(self):
        """Load data and prepare features with causal baseline"""
        print(f"[{self.node_tag}] Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Basic cleaning - convert string numbers to numeric
        numeric_cols = [
            "Sold Price", "TotFlArea", "Floor Area Fin - Total", "MaintFee",
            "Floor Area - Unfinished", "Floor Area Fin - Main Flr", "Floor Area Fin - Abv Main",
            "Full Baths", "Half Baths", "Tot BR", "Tot Baths", "Bds In Bsmt", "Yr Blt",
            "Units in Development", "No. Floor Levels", "Fireplaces", "# of Kitchens",
            "Bath Ensuite # Of Pcs", "Price", "List Price", "Sold Price per SqFt",
            "DOM", "Cumulative DOM", "Tot Units in Strata Plan", "Age",
            "Floor Area Fin - Basement", "Floor Area Fin - BLW Main"
        ]
        
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == object:
                df[col] = (df[col].astype(str)
                          .str.replace('$', '', regex=False)
                          .str.replace(',', '', regex=False)
                          .str.replace(' ', '', regex=False))
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Filter valid sales
        df = df[df["Sold Price"].notna() & (df["Sold Price"] > 0)].copy()
        df["List Date"] = pd.to_datetime(df["List Date"], errors='coerce')
        df = df[df["List Date"].notna()].copy()
        
        # Create FSA (Forward Sortation Area)
        df["FSA"] = df["Postal Code"].astype(str).str.strip().str[:3].str.upper()

        # Sort by time (critical for causal operations)
        df = df.sort_values("List Date").reset_index(drop=True)

        # Determine area column for PPSF calculation
        area_col = self.cfg["target_area_col"] if self.cfg["target_area_col"] in df.columns else None
        if area_col is None:
            if "Floor Area Fin - Total" in df.columns:
                area_col = "Floor Area Fin - Total"
            else:
                raise RuntimeError("No suitable area column found for PPSF target.")

        # Calculate price per square foot
        df["ppsf"] = df["Sold Price"] / df[area_col].replace(0, np.nan)

        # Build causal 90-day baseline (FSA median with shrinkage to region)
        win = f'{self.cfg["rolling_window_days"]}D'
        s = df.set_index("List Date")
        grp = s.groupby("FSA")["ppsf"]
        
        # Rolling statistics per FSA
        fsa_median = grp.rolling(win, closed="left").median().reset_index(level=0, drop=True)
        fsa_count = grp.rolling(win, closed="left").count().reset_index(level=0, drop=True)
        
        # Regional (global) baseline
        reg_median = s["ppsf"].rolling(win, closed="left").median()

        # Shrinkage: blend FSA and regional based on sample size
        alpha = fsa_count / (fsa_count + self.cfg["shrink_k"])
        alpha = alpha.clip(lower=0, upper=1).fillna(0.0)
        baseline = alpha * fsa_median + (1 - alpha) * reg_median
        
        # Fallback to regional when FSA has too few samples
        baseline = np.where(
            (fsa_count.values < self.cfg["min_fsa_samples"]) | pd.isna(fsa_median.values),
            reg_median.values,
            baseline
        )
        df["base_ppsf_90d"] = baseline

        # Filter valid rows and compute log premium
        mask = df["ppsf"].notna() & pd.notna(df["base_ppsf_90d"])
        df = df[mask].copy()
        df["log_premium"] = np.log(df["ppsf"]) - np.log(df["base_ppsf_90d"])

        # Time-based train/test split
        split_idx = int((1 - self.cfg["test_fraction"]) * len(df))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # Time-decay weights for training
        ref_date = train_df["List Date"].max()
        age_days = (ref_date - train_df["List Date"]).dt.days.clip(lower=0)
        hl = float(self.cfg["half_life_days"])
        weights = (0.5 ** (age_days / hl)).astype("float32")
        weights = np.maximum(weights, 0.05)  # minimum weight floor

        # Build safe feature list (numeric & pre-listing)
        exclude = set(self.cfg["exclude_features"])
        numeric_feats = []
        
        for c in df.columns:
            if c in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_feats.append(c)

        # Optional: require at least one of a specific set
        req = set(self.cfg.get("require_at_least_one_of", []))
        if req:
            numeric_feats = [f for f in numeric_feats if f in req]

        # Remove target/derived columns
        for bad in ["ppsf", "base_ppsf_90d", "log_premium", "Sold Price"]:
            if bad in numeric_feats:
                numeric_feats.remove(bad)

        # Keep only necessary columns
        keep_cols = numeric_feats + ["log_premium", "List Date", "base_ppsf_90d", area_col]
        keep_cols = [c for c in keep_cols if c in df.columns]
        train_df = train_df[keep_cols].copy()
        test_df = test_df[keep_cols].copy()

        # Cache arrays for speed
        self.feature_names = numeric_feats
        self.area_col = area_col
        self.Xtr = train_df[self.feature_names].to_numpy(dtype="float32", copy=False)
        self.ytr = train_df["log_premium"].to_numpy(dtype="float32", copy=False)
        self.wtr = weights.to_numpy(dtype="float32", copy=False)
        self.Xte = test_df[self.feature_names].to_numpy(dtype="float32", copy=False)
        self.yte = test_df["log_premium"].to_numpy(dtype="float32", copy=False)
        self.base_te = test_df["base_ppsf_90d"].to_numpy(dtype="float32", copy=False)
        self.area_te = test_df[area_col].to_numpy(dtype="float32", copy=False)

        # Report summary
        self.train_rows = len(train_df)
        self.test_rows = len(test_df)
        print(f"[{self.node_tag}] Data loaded: {self.train_rows} train, {self.test_rows} test, {len(self.feature_names)} features")

    def run_combos(self, combos: List[Tuple[int, ...]], params: Dict, num_rounds: int, early_stopping: int) -> List[Dict]:
        """
        Train XGBoost models for each feature combination
        Returns list of results with metrics
        """
        results = []
        
        for combo in combos:
            try:
                idx = list(combo)
                Xtr_sub = self.Xtr[:, idx]
                Xte_sub = self.Xte[:, idx]

                # Create XGBoost DMatrix with weights
                dtrain = xgb.QuantileDMatrix(Xtr_sub, label=self.ytr, weight=self.wtr)
                dvalid = xgb.QuantileDMatrix(Xte_sub, label=self.yte)

                # Train model
                model = xgb.train(
                    params, dtrain,
                    num_boost_round=num_rounds,
                    evals=[(dvalid, "valid")],
                    early_stopping_rounds=early_stopping,
                    verbose_eval=False
                )

                # Predict log premium and convert back to price
                log_prem_hat = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
                prem_hat = np.exp(log_prem_hat).astype("float32")
                ppsf_hat = prem_hat * self.base_te
                price_hat = ppsf_hat * self.area_te

                # Convert true values to price space for metrics
                prem_true = np.exp(self.yte)
                ppsf_true = prem_true * self.base_te
                price_true = ppsf_true * self.area_te

                # Calculate metrics
                abs_err = np.abs(price_true - price_hat)
                mae = float(np.mean(abs_err))
                wape = float(np.sum(abs_err) / np.sum(np.abs(price_true)))
                
                # R² on log premium (more stable)
                ss_res = float(np.sum((self.yte - log_prem_hat) ** 2))
                ss_tot = float(np.sum((self.yte - np.mean(self.yte)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                results.append({
                    "features": [self.feature_names[j] for j in idx],
                    "n_features": len(idx),
                    "mae": mae,
                    "wape": wape,
                    "r2_logprem": r2,
                    "iters": int(model.best_iteration or 0)
                })
                
            except Exception as e:
                results.append({
                    "features": f"ERROR: {combo}",
                    "n_features": len(combo),
                    "mae": float("nan"),
                    "wape": float("nan"),
                    "r2_logprem": float("nan"),
                    "error": str(e)
                })
        
        return results

    def info(self) -> Dict[str, Any]:
        """Return worker info for debugging"""
        return {
            "node": self.node_tag,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "features": len(self.feature_names)
        }

# ===================== DISCOVER GPUs & SPAWN ACTORS =====================
print("\n" + "=" * 70)
print("DISCOVERING CLUSTER RESOURCES")
print("=" * 70)

cluster = ray.nodes()  # metadata about nodes/resources
num_gpus_total = int(sum(n["Resources"].get("GPU", 0) for n in cluster))
print(f"✓ Discovered {len(cluster)} nodes, total GPUs: {num_gpus_total}")

# Map IP addresses to node tags for data path selection
ip_to_tag = {
    "10.0.0.75": "linux",      # Linux VM with RTX 3090
    "192.168.0.233": "wsl"     # Windows WSL2 with RTX 4090
}

# Create one actor per detected GPU
actors = []
for n in cluster:
    gpus = int(n["Resources"].get("GPU", 0))
    ip = n["NodeManagerAddress"]
    tag = ip_to_tag.get(ip, "linux")  # default to linux if unknown
    print(f"  Node {ip} ({tag}): {gpus} GPU(s)")
    for gpu_idx in range(gpus):
        actors.append(TrainWorker.remote(tag, DATA_PATHS, CONFIG))

# Verify actors are alive and data loaded
print("\nInitializing workers...")
infos = ray.get([a.info.remote() for a in actors])
for i, info in enumerate(infos):
    print(f"  Worker {i}: {info['node']} - {info['train_rows']} train, {info['test_rows']} test, {info['features']} features")

# ===================== BUILD FEATURE COMBINATIONS =====================
print("\n" + "=" * 70)
print("GENERATING FEATURE COMBINATIONS")
print("=" * 70)

# Get feature count from first actor
feat_count = infos[0]["features"]
sizes = list(range(1, CONFIG["max_features"] + 1))
all_combos = []

for n in sizes:
    combos_n = list(combinations(range(feat_count), n))
    all_combos.extend(combos_n)
    print(f"  {n} features: {len(combos_n):,} combinations")

print(f"\n✓ Total combinations to evaluate: {len(all_combos):,}")

# ===================== DISPATCH TRAINING JOBS =====================
print("\n" + "=" * 70)
print("DISPATCHING TRAINING JOBS")
print("=" * 70)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Round-robin distribute batches across workers
futures = []
rr = itertools.cycle(actors)
batch_count = 0

for batch in chunks(all_combos, CONFIG["batch_size"]):
    worker = next(rr)
    futures.append(worker.run_combos.remote(
        batch,
        CONFIG["params"],
        CONFIG["num_rounds"],
        CONFIG["early_stopping"]
    ))
    batch_count += 1

print(f"✓ Dispatched {batch_count} batches across {len(actors)} workers")
print(f"  Batch size: {CONFIG['batch_size']} combinations")

# ===================== COLLECT RESULTS =====================
print("\n" + "=" * 70)
print("COLLECTING RESULTS")
print("=" * 70)

all_results = []
start_time = time.time()

for i, fut in enumerate(futures, 1):
    batch_results = ray.get(fut)
    all_results.extend(batch_results)
    
    # Progress update
    if i % 10 == 0 or i == len(futures):
        elapsed = time.time() - start_time
        rate = i / elapsed
        eta = (len(futures) - i) / rate if rate > 0 else 0
        print(f"  Collected {i}/{len(futures)} batches | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

# ===================== SAVE RESULTS =====================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Convert to DataFrame and clean
df_res = pd.DataFrame(all_results)
df_res = df_res.dropna(subset=["mae"])  # drop failures

# Sort by MAE (best first)
df_res.sort_values("mae", inplace=True)

# Save with timestamp
timestamp = int(time.time())
output_file = f"distributed_results_{timestamp}.csv"
df_res.to_csv(output_file, index=False)

print(f"✓ Saved {len(df_res):,} results to {output_file}")

# ===================== DISPLAY TOP RESULTS =====================
print("\n" + "=" * 70)
print("TOP 10 FEATURE COMBINATIONS")
print("=" * 70)

top10 = df_res.head(10)
for idx, row in top10.iterrows():
    features = row['features']
    if isinstance(features, str):
        # Parse string representation of list if needed
        try:
            import ast
            features = ast.literal_eval(features)
        except:
            pass
    
    print(f"\n#{list(df_res.index).index(idx) + 1}:")
    print(f"  Features: {features}")
    print(f"  MAE: ${row['mae']:,.2f}")
    print(f"  WAPE: {row['wape']:.4f}")
    print(f"  R² (log): {row['r2_logprem']:.4f}")
    print(f"  Iterations: {row['iters']}")

# ===================== SUMMARY STATISTICS =====================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

total_time = time.time() - start_time
print(f"Total runtime: {total_time:.1f} seconds")
print(f"Average time per combination: {total_time / len(all_combos):.3f} seconds")
print(f"Best MAE: ${df_res['mae'].min():,.2f}")
print(f"Worst MAE: ${df_res['mae'].max():,.2f}")
print(f"Mean MAE: ${df_res['mae'].mean():,.2f}")

# Shutdown Ray connection
ray.shutdown()
print("\n✓ Ray connection closed")
print("✓ Distributed training complete!")