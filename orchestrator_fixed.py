#!/usr/bin/env python3
"""
Fixed orchestrator with better data handling
"""

import os
import itertools
import time
import numpy as np
import pandas as pd
import ray
import xgboost as xgb
from itertools import combinations

# ===================== CONFIGURATION =====================
HEAD = "ray://10.0.0.75:10001"
DATA_PATHS = {
    "linux": "/root/mltownhouseeval/sales_2015_2025.csv",
    "wsl":   "/root/mltownhouseeval/sales_2015_2025.csv",
}

CONFIG = {
    "max_features": 2,  # Start small for testing
    "batch_size": 50,
    "seed": 42,
    "params": {
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "gpu_id": 0,
        "objective": "reg:squarederror",
        "max_depth": 4,
        "learning_rate": 0.1,
        "verbosity": 0,
    },
    "num_rounds": 100,
    "early_stopping": 20,
}

# Connect to Ray
print(f"Connecting to Ray cluster at {HEAD}...")
ray.init(address=HEAD)
print("✓ Connected to Ray cluster")

@ray.remote(num_gpus=1)
class TrainWorker:
    def __init__(self, node_tag, data_path_map, config):
        self.node_tag = node_tag
        self.cfg = config
        self.data_path = data_path_map[node_tag]
        self._load_and_prepare()

    def _clean_price(self, series):
        """Clean price columns with $ and commas"""
        if pd.api.types.is_object_dtype(series):
            return pd.to_numeric(
                series.astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip(),
                errors='coerce'
            )
        return series

    def _load_and_prepare(self):
        print(f"[{self.node_tag}] Loading data from {self.data_path}")
        
        # Load with proper dtype handling
        df = pd.read_csv(self.data_path, low_memory=False)
        
        # Clean price columns
        price_cols = ['Sold Price', 'List Price', 'Price']
        for col in price_cols:
            if col in df.columns:
                df[col] = self._clean_price(df[col])
        
        # Clean other numeric columns
        numeric_cols = ['TotFlArea', 'MaintFee', 'Tot BR', 'Tot Baths', 
                       'Full Baths', 'Half Baths', 'Yr Blt']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = self._clean_price(df[col])
        
        # Filter valid sales
        df = df[df['Sold Price'].notna() & (df['Sold Price'] > 0)].copy()
        
        # Get numeric features only
        feature_cols = []
        exclude = {'Sold Price', 'List Price', 'Price', 'SP/LP Ratio'}
        
        for col in df.columns:
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
                # Check if column has any non-null values
                if df[col].notna().sum() > 100:
                    feature_cols.append(col)
        
        # Limit features for testing
        feature_cols = feature_cols[:10]
        
        # Simple train/test split
        split_idx = int(0.8 * len(df))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Remove rows with missing values
        train_df = train_df[feature_cols + ['Sold Price']].dropna()
        test_df = test_df[feature_cols + ['Sold Price']].dropna()
        
        # Store as arrays
        self.feature_names = feature_cols
        self.Xtr = train_df[feature_cols].to_numpy(dtype='float32')
        self.ytr = train_df['Sold Price'].to_numpy(dtype='float32')
        self.Xte = test_df[feature_cols].to_numpy(dtype='float32')
        self.yte = test_df['Sold Price'].to_numpy(dtype='float32')
        
        self.train_rows = len(train_df)
        self.test_rows = len(test_df)
        
        print(f"[{self.node_tag}] Data loaded: {self.train_rows} train, {self.test_rows} test, {len(self.feature_names)} features")

    def run_combos(self, combos, params, num_rounds, early_stopping):
        results = []
        
        for combo in combos:
            try:
                idx = list(combo)
                Xtr_sub = self.Xtr[:, idx]
                Xte_sub = self.Xte[:, idx]
                
                dtrain = xgb.DMatrix(Xtr_sub, label=self.ytr)
                dvalid = xgb.DMatrix(Xte_sub, label=self.yte)
                
                model = xgb.train(
                    params, dtrain,
                    num_boost_round=num_rounds,
                    evals=[(dvalid, "valid")],
                    early_stopping_rounds=early_stopping,
                    verbose_eval=False
                )
                
                y_pred = model.predict(dvalid)
                mae = float(np.mean(np.abs(self.yte - y_pred)))
                
                results.append({
                    "features": [self.feature_names[j] for j in idx],
                    "n_features": len(idx),
                    "mae": mae,
                    "iters": model.best_iteration
                })
            except Exception as e:
                results.append({
                    "features": f"ERROR: {combo}",
                    "n_features": len(combo),
                    "mae": float("nan"),
                    "error": str(e)
                })
        
        return results

    def info(self):
        return {
            "node": self.node_tag,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "features": len(self.feature_names)
        }

# ===================== DISCOVER GPUs =====================
print("\n" + "=" * 70)
print("DISCOVERING CLUSTER RESOURCES")
print("=" * 70)

cluster = ray.nodes()
num_gpus_total = int(sum(n["Resources"].get("GPU", 0) for n in cluster))
print(f"✓ Discovered {len(cluster)} nodes, total GPUs: {num_gpus_total}")

# Create actors
actors = []
ip_to_tag = {"10.0.0.75": "linux", "192.168.0.233": "wsl"}

for n in cluster:
    gpus = int(n["Resources"].get("GPU", 0))
    if gpus > 0:
        ip = n["NodeManagerAddress"]
        tag = ip_to_tag.get(ip, "linux")
        print(f"  Creating worker on {ip} ({tag})")
        actors.append(TrainWorker.remote(tag, DATA_PATHS, CONFIG))

if not actors:
    print("No GPUs found! Exiting.")
    ray.shutdown()
    exit(1)

print(f"\nInitializing {len(actors)} workers...")
infos = ray.get([a.info.remote() for a in actors])
for info in infos:
    print(f"  {info['node']}: {info['train_rows']} train, {info['test_rows']} test, {info['features']} features")

# ===================== BUILD COMBINATIONS =====================
print("\n" + "=" * 70)
print("GENERATING FEATURE COMBINATIONS")
print("=" * 70)

feat_count = infos[0]["features"]
all_combos = []
for n in range(1, min(CONFIG["max_features"] + 1, feat_count + 1)):
    combos_n = list(combinations(range(feat_count), n))
    all_combos.extend(combos_n)
    print(f"  {n} features: {len(combos_n)} combinations")

print(f"\n✓ Total combinations: {len(all_combos)}")

# ===================== RUN TRAINING =====================
print("\n" + "=" * 70)
print("RUNNING DISTRIBUTED TRAINING")
print("=" * 70)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

futures = []
rr = itertools.cycle(actors)

for batch in chunks(all_combos, CONFIG["batch_size"]):
    worker = next(rr)
    futures.append(worker.run_combos.remote(
        batch,
        CONFIG["params"],
        CONFIG["num_rounds"],
        CONFIG["early_stopping"]
    ))

print(f"✓ Dispatched {len(futures)} batches")

# Collect results
all_results = []
start_time = time.time()

for i, fut in enumerate(futures, 1):
    batch_results = ray.get(fut)
    all_results.extend(batch_results)
    print(f"  Batch {i}/{len(futures)} complete")

# Save results
df_res = pd.DataFrame(all_results)
df_res = df_res.dropna(subset=["mae"])
df_res.sort_values("mae", inplace=True)

output_file = f"results_{int(time.time())}.csv"
df_res.to_csv(output_file, index=False)

print(f"\n✓ Saved {len(df_res)} results to {output_file}")

# Show top results
print("\n" + "=" * 70)
print("TOP 5 RESULTS")
print("=" * 70)

for idx, row in df_res.head(5).iterrows():
    print(f"\nRank {idx + 1}:")
    print(f"  Features: {row['features']}")
    print(f"  MAE: ${row['mae']:,.2f}")

total_time = time.time() - start_time
print(f"\n✓ Total time: {total_time:.1f}s")
print(f"✓ Combinations/sec: {len(all_combos) / total_time:.1f}")

ray.shutdown()
print("\n✓ Done!")