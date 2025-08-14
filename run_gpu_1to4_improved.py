#!/usr/bin/env python3
"""
GPU search testing 1, 2, 3, and 4 feature combinations
- Leakage detection and prevention
- Performance optimizations (beam search option)
- WAPE metric and sanity checks
- Auto-excludes problematic features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from itertools import combinations
from math import comb as ncr
import time, sys, warnings, os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ---------------- Logging to console + file ----------------
class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()
    def close(self):
        self.log.close()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'gpu_search_results_{timestamp}.log'
dual_output = DualOutput(log_filename)
sys.stdout = dual_output

print(f"📝 Logging output to: {log_filename}")
print("="*70)
print("GPU SEARCH - IMPROVED VERSION WITH LEAKAGE DETECTION")
print("="*70)

# ---------------- Requested features ----------------
REQUESTED_FEATURES = [
    "Status", "Address", "S/A", "Price", "List Date", "DOM", "Tot BR", "Tot Baths",
    "TotFlArea", "Yr Blt", "Age", "MaintFee", "TypeDwel", "Bylaw Restrictions",
    "# of Kitchens", "# of Pets", "Title to Land", "Bath Ensuite # Of Pcs",
    "Baths Concatenation", "Bds In Bsmt", "Complex/Subdivision Name", 
    "Confirm Sold Date", "Cumulative DOM", "Expiry Date", "Fireplaces",
    "Floor Area - Unfinished", "Floor Area Fin - Abv Main", 
    "Floor Area Fin - Basement", "Floor Area Fin - BLW Main", 
    "Floor Area Fin - Main Flr", "Floor Area Fin - Total",
    "Full Baths", "GST Incl", "Half Baths", "Land Lease Expiry Year", "List Price",
    "Measurement Type", "Member Board Affiliation", "No. Floor Levels", "Postal Code",
    "Price Per SQFT", "Restricted Age", "Room Type Search", "Sold Date", "Sold Price",
    "Sold Price per SqFt", "SP/LP Ratio", "SP/OLP Ratio", "Storeys in Building",
    "Tot Units in Strata Plan", "Units in Development", "Zoning"
]

# ---------------- Load data ----------------
print("\n📊 Loading and preparing data...")
csv_path = "/home/monstrcow/mltownhouseeval/Jan 1 2015_Aug 13 2025.csv"
df = pd.read_csv(csv_path)
print(f"✓ Loaded {len(df):,} records")

# ---------------- Light conversions ----------------
def to_numeric(series):
    if pd.api.types.is_object_dtype(series):
        s = (series.astype(str)
             .str.replace('$','',regex=False)
             .str.replace(',','',regex=False)
             .str.replace(' ','',regex=False))
        return pd.to_numeric(s, errors='coerce')
    return series

numeric_conversions = [
    'Price','List Price','Sold Price','TotFlArea','MaintFee',
    'Floor Area - Unfinished','Floor Area Fin - Abv Main','Floor Area Fin - Basement',
    'Floor Area Fin - BLW Main','Floor Area Fin - Main Flr','Floor Area Fin - Total',
    'Price Per SQFT','Sold Price per SqFt','# of Pets','Tot BR','Tot Baths',
    'Full Baths','Half Baths','No. Floor Levels','Storeys in Building','Fireplaces',
    'DOM', 'Cumulative DOM', 'Tot Units in Strata Plan', 'Units in Development'
]
for col in numeric_conversions:
    if col in df.columns:
        df[col] = to_numeric(df[col])

# Dates for time split
if 'List Date' in df.columns:
    df['List Date'] = pd.to_datetime(df['List Date'], errors='coerce')

# ---------------- Filter target ----------------
df = df[df['Sold Price'].notna() & (df['Sold Price'] > 0)].copy()
print(f"✓ {len(df):,} properties with valid sold prices")

# ---------------- LEAKAGE AUDIT ----------------
def audit_leakage(df, target='Sold Price'):
    import numpy as np, pandas as pd
    out = []
    if target not in df: 
        print(f"Target {target} not in df"); 
        return pd.DataFrame()
    y = pd.to_numeric(df[target], errors='coerce')

    # Helper to evaluate one candidate vs the target
    def eval_col(col, s):
        s = pd.to_numeric(s, errors='coerce')
        ok = s.notna() & y.notna()
        if ok.sum() < max(100, int(0.05*len(df))):
            return None
        diff = (s[ok] - y[ok]).abs()
        corr = s[ok].corr(y[ok])
        med_abs = float(diff.median())
        pct_w_5k = float((diff <= 5_000).mean())  # share within $5k
        return dict(column=col, corr=round(corr, 6), med_abs=round(med_abs, 2), 
                   pct_within_5k=round(100*pct_w_5k, 1))

    # 1) Direct numeric columns
    for col in df.columns:
        if col == target: 
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            m = eval_col(col, df[col])
            if m: out.append(m)

    # 2) If you have area, check "price per sqft" style columns
    if 'TotFlArea' in df.columns:
        area = pd.to_numeric(df['TotFlArea'], errors='coerce').replace(0, np.nan)
        ppsf_sold = y / area
        for col in [c for c in df.columns if 'per SQFT' in c.lower() or 'per sqft' in c.lower()]:
            m = eval_col(f"{col} vs SOLD PPSF", pd.to_numeric(df[col], errors='coerce'))
            if m:
                # compare to SOLD PPSF, not to Sold Price
                ok = ppsf_sold.notna() & pd.to_numeric(df[col], errors='coerce').notna()
                diff_ppsf = (pd.to_numeric(df[col], errors='coerce')[ok] - ppsf_sold[ok]).abs()
                m['ppsf_median_abs_diff'] = round(float(diff_ppsf.median()), 4)
                out.append(m)

    audit = pd.DataFrame(out).sort_values(['corr','pct_within_5k'], ascending=False)
    # Heuristic: high corr and very small median abs diff or big share within $5k → likely leakage
    audit['leak_flag'] = (audit['corr'] >= 0.995) & ((audit['med_abs'] <= 10_000) | (audit['pct_within_5k'] >= 50))
    print("\n🔎 Leakage audit (top 20 by suspicion):")
    print(audit.head(20).to_string(index=False))
    return audit

print("\n" + "="*70)
print("LEAKAGE DETECTION")
print("="*70)
audit = audit_leakage(df)

# ---------------- Auto-detect and exclude leaky features ----------------
# Hard rules
EXCLUDE_ALWAYS = {
    'Sold Price', 'Sold Price per SqFt', 'SP/LP Ratio', 'SP/OLP Ratio',
    'Sold Date', 'Confirm Sold Date'
}

# Start with requested list minus hard excludes
candidates = [f for f in REQUESTED_FEATURES if f in df.columns and f not in EXCLUDE_ALWAYS]

# Auto-detect suspicious columns
def auto_flag_leaky(df, candidates, target='Sold Price'):
    flagged = set()
    y = pd.to_numeric(df[target], errors='coerce')
    for col in candidates:
        s = pd.to_numeric(df[col], errors='coerce')
        ok = s.notna() & y.notna()
        if ok.sum() < max(100, int(0.05*len(df))): 
            continue
        diff = (s[ok] - y[ok]).abs()
        corr = s[ok].corr(y[ok])
        pct_w_5k = (diff <= 5_000).mean()
        med_abs = diff.median()
        if corr is not None and corr >= 0.995 and (med_abs <= 10_000 or pct_w_5k >= 0.50):
            flagged.add(col)
            print(f"⚠️  Auto-flagged '{col}' as leaky: corr={corr:.3f}, median_abs=${med_abs:,.0f}")
    
    # "Price Per SQFT" is leaky if it matches Sold PPSF
    if 'Price Per SQFT' in df.columns and 'TotFlArea' in df.columns:
        area = pd.to_numeric(df['TotFlArea'], errors='coerce').replace(0, np.nan)
        sold_ppsf = y / area
        ppsf = pd.to_numeric(df['Price Per SQFT'], errors='coerce')
        ok = ppsf.notna() & sold_ppsf.notna()
        if ok.sum() >= max(100, int(0.05*len(df))):
            if (ppsf[ok] - sold_ppsf[ok]).abs().median() <= 10:
                flagged.add('Price Per SQFT')
                print(f"⚠️  Auto-flagged 'Price Per SQFT' as leaky (matches sold PPSF)")
    return flagged

LEAKY_AUTO = auto_flag_leaky(df, candidates, target='Sold Price')

# Business knowledge: exclude features that don't exist at listing time
EXCLUDE_POST_LISTING = {'DOM', 'Cumulative DOM', 'Expiry Date'}  # These accumulate after listing
# Also exclude List Price as we want predictive model without knowing listing price
EXCLUDE_POST_LISTING.add('List Price')
print(f"\n📋 Excluding post-listing features: {EXCLUDE_POST_LISTING}")

# Combine all exclusions
ALL_EXCLUDE = EXCLUDE_ALWAYS | LEAKY_AUTO | EXCLUDE_POST_LISTING

SAFE_CANDIDATES = [f for f in candidates if f not in ALL_EXCLUDE]

# Keep only numeric among the safe set
available = [f for f in SAFE_CANDIDATES if pd.api.types.is_numeric_dtype(df[f])]

if not available:
    print("⚠️ No numeric features available after hygiene. Aborting.")
    sys.stdout = dual_output.terminal; dual_output.close(); sys.exit(1)

print(f"\n📊 Using {len(available)} safe numeric features:")
for i, feat in enumerate(available, 1):
    print(f"  {i:2}. {feat}")

# ---------------- Sanity checks ----------------
print("\n" + "="*70)
print("SANITY CHECKS")
print("="*70)

# Distribution of |Sold - List| to calibrate a naive baseline
if 'List Price' in df.columns:
    list_abs = (pd.to_numeric(df['List Price'], errors='coerce') - pd.to_numeric(df['Sold Price'], errors='coerce')).abs()
    print(f"Naive baseline (|List - Sold|):")
    print(f"  • Median: ${list_abs.median():,.0f}")
    print(f"  • Mean: ${list_abs.mean():,.0f}")
    print(f"  • 75th percentile: ${list_abs.quantile(0.75):,.0f}")

X_full = df[available]
y_full = df['Sold Price'].astype(float)

# ---------------- Time-based split if possible ----------------
if 'List Date' in df.columns and df['List Date'].notna().sum() > int(0.8*len(df)):
    ### Sort by List Date and use most recent 20% as test
    df_sorted = df.sort_values('List Date').reset_index(drop=True)
    X_full = df_sorted[available]
    y_full = df_sorted['Sold Price'].astype(float)
    split_idx = int(0.80 * len(df_sorted))
    X_train = X_full.iloc[:split_idx]
    X_test  = X_full.iloc[split_idx:]
    y_train = y_full.iloc[:split_idx]
    y_test  = y_full.iloc[split_idx:]
    print(f"\n🕒 Time-based split on List Date → Train: {len(X_train):,}, Test: {len(X_test):,}")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.20, random_state=42
    )
    print(f"\n🔀 Random split → Train: {len(X_train):,}, Test: {len(X_test):,}")

# Fixed validation split from training for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

# Targets: keep in price space for metrics
y_tr_np  = y_tr.values.astype(np.float32)
y_val_np = y_val.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

# To numpy (no scaling, keep NaNs)
X_tr_np   = X_tr.values.astype(np.float32)
X_val_np  = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

# ---------------- Helper: WAPE metric ----------------
def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error - scale-free metric"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

# ---------------- Combination accounting ----------------
k = len(available)
sizes = [1, 2, 3, 4]
total_tests = sum(ncr(k, n) for n in sizes)
print("\n📊 Generating combinations (1–4 features)...")
for n in sizes:
    print(f"  {n} feature(s): {ncr(k, n):,} combinations")
print(f"\n✅ Total combinations: {total_tests:,}")

# ---------------- XGBoost params (OPTIMIZED) ----------------
params = {
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0,
    'max_depth': 5,  # Reduced for speed
    'learning_rate': 0.10,  # Increased for faster convergence
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'min_child_weight': 1.0,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'seed': 42,
    'verbosity': 0
}
num_rounds = 350  # Reduced from 600
early_stopping = 40  # Increased from 30

results_by_size = {n: [] for n in sizes}
best_mae_overall = float('inf')

print("\n🚀 Starting GPU search (optimized settings)...")
print("-"*70)
start_time = time.time()
last_update = time.time()
completed = 0
skipped = 0

for n_features in sizes:
    print(f"\n\n📊 Testing {n_features}-feature combinations...")
    best_mae_size = float('inf')
    total_this_size = ncr(k, n_features)

    for combo_idx, combo in enumerate(combinations(range(k), n_features), 1):
        try:
            idx = list(combo)
            # Slice per-combo views (no scaling; GPU handles NaNs)
            Xtr = X_tr_np[:, idx]; Xval = X_val_np[:, idx]; Xtst = X_test_np[:, idx]

            dtrain = xgb.DMatrix(Xtr, label=y_tr_np)
            dvalid = xgb.DMatrix(Xval, label=y_val_np)
            dtest  = xgb.DMatrix(Xtst)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_rounds,
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=early_stopping,
                verbose_eval=False
            )

            # Predict
            y_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

            mae = mean_absolute_error(y_test_np, y_pred)
            r2  = r2_score(y_test_np, y_pred)
            wape_val = wape(y_test_np, y_pred)

            results_by_size[n_features].append({
                'combo': combo,
                'features': [available[j] for j in combo],
                'n_features': n_features,
                'mae': mae,
                'r2': r2,
                'wape': wape_val,
                'best_iteration': int(model.best_iteration or 0)
            })

            if mae < best_mae_size: best_mae_size = mae
            if mae < best_mae_overall: best_mae_overall = mae

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"\n⚠️ Skipped combo {combo} due to error: {e}")

        completed += 1

        # Progress
        now = time.time()
        if now - last_update >= 1.0 or completed == total_tests:
            elapsed = now - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = total_tests - completed
            eta = int(remaining / rate) if rate > 0 else 0
            print(
                f"\r  Progress: {combo_idx:,}/{total_this_size:,} | "
                f"Overall: {completed:,}/{total_tests:,} ({100*completed/total_tests:.1f}%) | "
                f"Rate: {rate:.0f}/sec | "
                f"Best MAE (size {n_features}): ${best_mae_size:,.0f} | "
                f"Skipped: {skipped:,} | ETA: {timedelta(seconds=eta)}",
                end='', flush=True
            )

print("\n\n" + "="*70)
print("✅ SEARCH COMPLETE!")
print("="*70)

elapsed = time.time() - start_time
rate = total_tests / elapsed if elapsed > 0 else 0
print(f"⏱️  Total time: {elapsed:.1f} seconds")
print(f"⚡ Average rate: {rate:.0f} models/sec")

# ---------------- Results by feature count ----------------
print("\n" + "="*70)
print("RESULTS BY FEATURE COUNT")
print("="*70)

for n_features in sizes:
    results = results_by_size[n_features]
    if not results:
        print(f"\n(No successful results for {n_features} features)")
        continue

    print(f"\n🏆 BEST {n_features}-FEATURE MODELS:")
    print("-"*70)
    results_sorted = sorted(results, key=lambda x: x['mae'])[:10]
    for i, r in enumerate(results_sorted, 1):
        features_str = ', '.join(r['features'])
        print(f"{i:2}. MAE: ${r['mae']:,.0f} | R²: {r['r2']:.3f} | WAPE: {r['wape']:.3f} | iters: {r['best_iteration']} | {features_str}")

    maes = [r['mae'] for r in results]
    wapes = [r['wape'] for r in results]
    print(f"\n  Summary for {n_features} features:")
    print(f"    • Best MAE:    ${min(maes):,.0f}")
    print(f"    • Worst MAE:   ${max(maes):,.0f}")
    print(f"    • Average MAE: ${np.mean(maes):,.0f}")
    print(f"    • Best WAPE:   {min(wapes):.3f}")
    print(f"    • Total tested: {len(results):,}")

# ---------------- Overall feature importance ----------------
print("\n" + "="*70)
print("📊 OVERALL FEATURE PRESENCE IN TOP MODELS")
print("="*70)

all_results = [r for n in sizes for r in results_by_size[n]]
if all_results:
    K = min(100, len(all_results))
    top_k = sorted(all_results, key=lambda x: x['mae'])[:K]
    counts = {}
    for r in top_k:
        for f in r['features']:
            counts[f] = counts.get(f, 0) + 1
    pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"\nMost common features in top {K} models:")
    for f, c in pairs:
        pct = 100.0 * c / K
        print(f"  {f:35} {pct:5.1f}% of top models")
else:
    print("No successful models to summarize.")

# ---------------- Final summary ----------------
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

if all_results:
    best_overall = min(all_results, key=lambda x: x['mae'])
    print(f"\n🥇 BEST MODEL OVERALL:")
    print(f"  Features: {', '.join(best_overall['features'])}")
    print(f"  MAE: ${best_overall['mae']:,.0f}")
    print(f"  R²: {best_overall['r2']:.3f}")
    print(f"  WAPE: {best_overall['wape']:.3f}")
    print(f"  Iterations: {best_overall['best_iteration']}")

# ---------------- Wrap up ----------------
sys.stdout = dual_output.terminal
dual_output.close()
print(f"\n📁 Results saved to: {log_filename}")