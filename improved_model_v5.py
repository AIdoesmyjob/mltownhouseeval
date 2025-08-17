#!/usr/bin/env python3
"""
Townhouse Price Prediction Model V5 - Distributed, Time-Aware, Calibrated

Major changes vs V4:
- True time-based rolling medians per group (90D/365D) with proper datetime index
- Removed hardcoded `gpu_id`; Ray sets CUDA_VISIBLE_DEVICES per actor
- Slower learning rate, many more rounds, stricter early stopping, deterministic hist
- TimeSeries target encoding with smoothing to avoid leakage and overfit
- Use NaNs natively in XGBoost (no fillna(0)); pass missing=np.nan
- Recency sample weights (half-life = 365 days)
- Predict at best iteration; ensemble average in log space
- Isotonic regression calibration in log space
"""

import ray
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
from datetime import datetime
import warnings
import sys
import time
import gc

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# Ray cluster configuration
# ---------------------------------------------------------------------
RAY_HEAD = "ray://10.0.0.198:10001"

def check_ray_cluster_gpus():
    """Check Ray cluster for available GPUs"""
    print("\n" + "="*60)
    print("RAY CLUSTER GPU CHECK")
    print("="*60)

    print(f"Connecting to Ray cluster at {RAY_HEAD}...")
    try:
        ray.init(address=RAY_HEAD, ignore_reinit_error=True, log_to_driver=True)
        print("Connected to Ray cluster")
    except Exception as e:
        print(f"Failed to connect to Ray cluster: {e}")
        return False

    resources = ray.cluster_resources()
    available = ray.available_resources()

    total_gpus = resources.get('GPU', 0)
    available_gpus = available.get('GPU', 0)

    print(f"\nCluster Resources:")
    print(f"  Total GPUs: {total_gpus}")
    print(f"  Available GPUs: {available_gpus}")

    nodes = ray.nodes()
    gpu_nodes = 0
    print("\nNode Details:")
    for node in nodes:
        if node.get('Alive', False):
            node_res = node.get('Resources', {})
            node_gpus = int(node_res.get('GPU', 0))
            if node_gpus > 0:
                gpu_nodes += 1
                print(f"  Node {node.get('NodeManagerAddress')}: {node_gpus} GPU(s)")

    if total_gpus < 2:
        print("\nERROR: Need 2 GPUs.")
        ray.shutdown()
        return False

    print("\nOK: Two or more GPUs detected.")
    return True

def progress_bar(current, total, prefix='', suffix='', length=50):
    """Display a progress bar"""
    if total == 0:
        return
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    sys.stdout.write(f'\r{prefix} |{bar}| {current}/{total} ({percent:.1%}) {suffix}')
    sys.stdout.flush()
    if current == total:
        print()

def to_numeric(series):
    """Convert series to numeric, handling $, commas and spaces"""
    if pd.api.types.is_object_dtype(series):
        series = (series.astype(str)
                       .str.replace('$', '', regex=False)
                       .str.replace(',', '', regex=False)
                       .str.replace(' ', '', regex=False))
    return pd.to_numeric(series, errors='coerce')

def wape(y_true, y_pred):
    """Calculate WAPE metric"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

# ---------------------------------------------------------------------
# Ray remote training actor
# ---------------------------------------------------------------------
@ray.remote(num_gpus=1)
class GPUTrainer:
    """GPU trainer actor to maintain model state on the assigned GPU"""
    def __init__(self, gpu_name):
        self.gpu_name = gpu_name
        self.model = None
        self.best_iteration = None

    def train(self, X_train, y_train, X_val, y_val, params, num_rounds=8000, early_stopping=300):
        """Train XGBoost model on GPU with early stopping"""
        dtrain = xgb.DMatrix(X_train['X'], label=X_train['y'], weight=X_train['w'], missing=np.nan)
        dval   = xgb.DMatrix(X_val['X'],   label=X_val['y'],   weight=X_val['w'],   missing=np.nan)

        print(f"[{self.gpu_name}] Training started")
        evals = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=200
        )
        self.best_iteration = self.model.best_iteration
        print(f"[{self.gpu_name}] Best iteration: {self.best_iteration}")
        return self.best_iteration

    def predict(self, X):
        """Predict using the best iteration"""
        dmatrix = xgb.DMatrix(X, missing=np.nan)
        it_lim = None
        if self.best_iteration is not None:
            it_lim = self.best_iteration + 1
        return self.model.predict(dmatrix, iteration_range=(0, it_lim) if it_lim else None)

    def get_importance(self, importance_type='gain'):
        """Get feature importance"""
        if self.model:
            return self.model.get_score(importance_type=importance_type)
        return {}

# ---------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------
print("="*60)
print("TOWNHOUSE MODEL V5 - DISTRIBUTED, TIME-AWARE, CALIBRATED")
print("="*60)

if not check_ray_cluster_gpus():
    print("\nExiting")
    sys.exit(1)

print("\n" + "="*60)
print("DATA LOADING AND PREPARATION")
print("="*60)

# Load data
print("\nLoading data...")
df = pd.read_csv('Jan 1 2015_Aug 13 2025.csv')
print(f"Initial shape: {df.shape}")

# Convert numeric columns
print("\nConverting numeric columns...")
num_cols = ['Price', 'Sold Price', 'List Price', 'TotFlArea', 'MaintFee', 'Yr Blt', 'Age',
            'Full Baths', 'Half Baths', 'Tot BR', 'Tot Baths', 'No. Floor Levels',
            'Storeys in Building', 'Fireplaces', 'Floor Area Fin - Total',
            'Floor Area Fin - Main Flr', 'Floor Area Fin - Abv Main',
            'Floor Area Fin - Basement', 'Floor Area - Unfinished',
            'Tot Units in Strata Plan', 'Units in Development',
            '# of Kitchens', '# of Pets', 'Bds In Bsmt', 'Bds Not In Bsmt',
            'TotalPrkng', '# or % of Rentals Allowed']

for i, col in enumerate(num_cols, 1):
    if col in df.columns:
        df[col] = to_numeric(df[col])
    progress_bar(i, len(num_cols), prefix='Converting', suffix=col[:22].ljust(22))

# Dates
print("\nParsing dates...")
DATE_COL = 'List Date'
for col in ['List Date', 'Sold Date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Filter valid rows
TARGET_COL = 'Sold Price' if 'Sold Price' in df.columns else 'Price'
AREA_COL = 'TotFlArea'
df = df[df[TARGET_COL].notna() & df[TARGET_COL].gt(0)].copy()
df = df[df[AREA_COL].notna() & df[AREA_COL].gt(0)].copy()
df = df[df[DATE_COL].notna()].copy()
df = df.sort_values(DATE_COL).reset_index(drop=True)
print(f"\nAfter filtering: {len(df)} valid sales")

# Derived
df['sold_ppsf'] = df[TARGET_COL] / df[AREA_COL]

# ---------------------------------------------------------------------
# Comparable baselines with TRUE time windows
# ---------------------------------------------------------------------
print("\n" + "="*60)
print("COMPARABLE BASELINE CONSTRUCTION (TIME-BASED)")
print("="*60)

def rolling_median_ppsf_time(df_in, group_col, window, min_periods):
    """
    Time-based rolling median per group using datetime index.
    Returns a DataFrame with [group_col, DATE_COL, new_col].
    """
    new_col = f'ppsf_{window}_{group_col}'
    print(f"  Computing {new_col}...")

    # Work on a copy sorted within group by date
    work = df_in[[group_col, DATE_COL, 'sold_ppsf']].copy()
    work = work.sort_values([group_col, DATE_COL])

    def _group_roll(g):
        # set index to date for offset window
        ser = g.set_index(DATE_COL)['sold_ppsf'] \
               .rolling(window=window, min_periods=min_periods).median().shift(1)
        ser = ser.reindex(g.set_index(DATE_COL).index)
        out = pd.DataFrame({
            group_col: g[group_col].values,
            DATE_COL: g[DATE_COL].values,
            new_col: ser.values
        })
        return out

    pieces = []
    for key, g in work.groupby(group_col, sort=False):
        pieces.append(_group_roll(g))
    res = pd.concat(pieces, axis=0, ignore_index=True)

    non_null = res[new_col].notna().sum()
    print(f"    computed {non_null} values")
    return res

# Location features
print("\nExtracting location features...")
df['FSA'] = df['Postal Code'].astype(str).str[:3] if 'Postal Code' in df.columns else ''
df['Region'] = df['S/A'].astype(str).str.extract(r'^([A-Z]+)')[0] if 'S/A' in df.columns else ''

# Build baselines
print("\nBuilding baseline features...")
candidates = []

if 'TypeDwel' in df.columns:
    print("\n1) TypeDwel baselines:")
    for window, min_p in [('90D', 8), ('365D', 12)]:
        try:
            r = rolling_median_ppsf_time(df, 'TypeDwel', window, min_p)
            df = df.merge(r, on=['TypeDwel', DATE_COL], how='left')
            candidates.append(r.columns[-1])
            gc.collect()
        except Exception as e:
            print(f"    failed: {str(e)[:80]}")

print("\n2) FSA baselines:")
for window, min_p in [('90D', 8), ('365D', 12)]:
    try:
        r = rolling_median_ppsf_time(df, 'FSA', window, min_p)
        df = df.merge(r, on=['FSA', DATE_COL], how='left')
        candidates.append(r.columns[-1])
        gc.collect()
    except Exception as e:
        print(f"    failed: {str(e)[:80]}")

if 'Region' in df.columns and df['Region'].notna().any():
    print("\n3) Region baselines:")
    for window, min_p in [('90D', 8), ('365D', 12)]:
        try:
            r = rolling_median_ppsf_time(df, 'Region', window, min_p)
            df = df.merge(r, on=['Region', DATE_COL], how='left')
            candidates.append(r.columns[-1])
            gc.collect()
        except Exception as e:
            print(f"    failed: {str(e)[:80]}")

print("\n4) Global baselines:")
tmp = df[[DATE_COL, 'sold_ppsf']].set_index(DATE_COL).sort_index()
df['ppsf_90D_global'] = tmp['sold_ppsf'].rolling('90D', min_periods=8).median().shift(1).values
df['ppsf_365D_global'] = tmp['sold_ppsf'].rolling('365D', min_periods=12).median().shift(1).values
candidates += ['ppsf_90D_global', 'ppsf_365D_global']

print("\n5) Select best baseline per row...")
df['ppsf_baseline'] = np.nan
for i, c in enumerate(candidates, 1):
    if c in df.columns:
        filled_before = df['ppsf_baseline'].notna().sum()
        df['ppsf_baseline'] = df['ppsf_baseline'].fillna(df[c])
        filled_after = df['ppsf_baseline'].notna().sum()
        if filled_after > filled_before:
            progress_bar(i, len(candidates), prefix='  Selecting', suffix=f'{c[:24]} ({filled_after}/{len(df)})')

df['ppsf_baseline'] = df['ppsf_baseline'].fillna(df['sold_ppsf'].median())
df['price_baseline'] = df['ppsf_baseline'] * df[AREA_COL]
df = df[df['price_baseline'].gt(0)].copy()
df['log_price_ratio'] = np.log(df[TARGET_COL]) - np.log(df['price_baseline'])

print(f"\nBaseline PPSF range: ${df['ppsf_baseline'].min():.0f}–${df['ppsf_baseline'].max():.0f}")
print(f"Records after baseline: {len(df)}")

# ---------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------
print("\n" + "="*60)
print("TEMPORAL SPLIT")
print("="*60)

n = len(df)
train_end = int(0.7 * n)
val_end = int(0.8 * n)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

print(f"Train: {len(train_df)} ({train_df[DATE_COL].min().date()} to {train_df[DATE_COL].max().date()})")
print(f"Val:   {len(val_df)} ({val_df[DATE_COL].min().date()} to {val_df[DATE_COL].max().date()})")
print(f"Test:  {len(test_df)} ({test_df[DATE_COL].min().date()} to {test_df[DATE_COL].max().date()})")

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

EXCLUDE_COLS = {
    TARGET_COL, 'log_price_ratio', 'Sold Price', 'Price', 'List Price',
    'Sold Date', 'sold_ppsf', 'price_baseline', 'ppsf_baseline',
    'DOM', 'Cumulative DOM', 'Expiry Date', 'SP/LP Ratio', 'SP/OLP Ratio',
    'Sold Price per SqFt', 'Price Per SQFT', DATE_COL,
    'Address', 'Postal Code', 'Complex/Subdivision Name',
    'Member Board Affiliation', 'GST Incl', 'Confirm Sold Date'
}

cat_cols = ['TypeDwel', 'S/A', 'Region', 'FSA', 'Bylaw Restrictions',
            'Title to Land', 'Room Type Search', 'Zoning', 'Status']
cat_cols = [c for c in cat_cols if c in df.columns]

print("\nTarget encoding categoricals (time-aware, smoothed)...")

def _smoothed_mean(group, target, prior, m):
    cnt = group[target].count()
    mean = group[target].mean()
    return (cnt * mean + m * prior) / (cnt + m)

def target_encode_timewise(train_df, val_df, test_df, col, target='log_price_ratio', n_splits=5, m=50):
    """
    TimeSeriesSplit target encoding with smoothing.
    - Fit encodings only on past folds for train_oof.
    - Then fit on full train for val/test mapping.
    """
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(train_df)//1000)) )
    train_encoded = np.zeros(len(train_df))
    global_prior = train_df[target].mean()

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train_df)):
        tr = train_df.iloc[tr_idx]
        va = train_df.iloc[va_idx]
        means = tr.groupby(col).apply(_smoothed_mean, target=target, prior=global_prior, m=m)
        train_encoded[va_idx] = va[col].map(means).fillna(global_prior).values

    train_df[f'{col}_te'] = train_encoded

    means_full = train_df.groupby(col).apply(_smoothed_mean, target=target, prior=global_prior, m=m)
    val_df[f'{col}_te']  = val_df[col].map(means_full).fillna(global_prior).values
    test_df[f'{col}_te'] = test_df[col].map(means_full).fillna(global_prior).values

for i, col in enumerate(cat_cols, 1):
    target_encode_timewise(train_df, val_df, test_df, col)
    progress_bar(i, len(cat_cols), prefix='  Encoding', suffix=col)

# Feature columns
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c not in cat_cols]
feature_cols = [c for c in feature_cols if c in train_df.columns]
feature_cols += [f'{c}_te' for c in cat_cols]

# Keep numeric only; let NaNs pass through
numeric_mask = train_df[feature_cols].dtypes.apply(lambda x: np.issubdtype(x, np.number))
feature_cols = [c for c, is_num in zip(feature_cols, numeric_mask) if is_num]

print(f"\nUsing {len(feature_cols)} features")
print("\nSample features:")
for i, feat in enumerate(feature_cols[:10], 1):
    print(f"  {i:2}. {feat}")
if len(feature_cols) > 10:
    print(f"  ... and {len(feature_cols) - 10} more")

# ---------------------------------------------------------------------
# Recency weights (half-life 365 days)
# ---------------------------------------------------------------------
print("\nComputing recency weights...")
max_date = train_df[DATE_COL].max()
def recency_weight(d, half_life_days=365.0):
    age_days = (max_date - d).days
    return 0.5 ** (age_days / half_life_days)

train_w = train_df[DATE_COL].apply(recency_weight).values
val_w   = val_df[DATE_COL].apply(lambda d: 1.0).values  # evaluate fairly
test_w  = test_df[DATE_COL].apply(lambda d: 1.0).values

# Matrices (no zero-impute; use NaN)
X_train = train_df[feature_cols].values
y_train = train_df['log_price_ratio'].values
X_val   = val_df[feature_cols].values
y_val   = val_df['log_price_ratio'].values
X_test  = test_df[feature_cols].values
y_test  = test_df['log_price_ratio'].values  # true log ratio, used for calibration checks

# ---------------------------------------------------------------------
# XGBoost parameters
# ---------------------------------------------------------------------
params = {
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'eval_metric': 'rmse',
    'learning_rate': 0.02,
    'grow_policy': 'lossguide',
    'max_depth': 0,          # ignored with lossguide
    'max_leaves': 255,
    'max_bin': 512,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'min_child_weight': 8,
    'reg_lambda': 2.0,
    'reg_alpha': 1.0,
    'base_score': 0.0,       # target mean ~ 0 in log space
    'seed': 42,
    'deterministic_histogram': True
}

print("\n" + "="*60)
print("DISTRIBUTED TRAINING ON 2 GPUs")
print("="*60)
print("Training two models in parallel with different seeds")

# Ray object store payloads
train_blob = {'X': X_train, 'y': y_train, 'w': train_w}
val_blob   = {'X': X_val,   'y': y_val,   'w': val_w}

X_train_ref = ray.put(train_blob)
X_val_ref   = ray.put(val_blob)
X_test_ref  = ray.put(X_test)

trainer1 = GPUTrainer.remote("GPU1")
trainer2 = GPUTrainer.remote("GPU2")

params1 = params.copy()
params2 = params.copy()
params2['seed'] = 123

start_time = time.time()
future1 = trainer1.train.remote(X_train_ref, y_train, X_val_ref, y_val, params1,
                                num_rounds=8000, early_stopping=300)
future2 = trainer2.train.remote(X_train_ref, y_train, X_val_ref, y_val, params2,
                                num_rounds=8000, early_stopping=300)

print("\nWaiting for training to complete...")
while True:
    ready, not_ready = ray.wait([future1, future2], num_returns=2, timeout=5)
    if len(ready) == 2:
        break
    elapsed = int(time.time() - start_time)
    sys.stdout.write(f'\r  Elapsed: {elapsed}s...')
    sys.stdout.flush()
print()

best_iter1, best_iter2 = ray.get([future1, future2])
training_time = time.time() - start_time

print(f"\nTraining time: {training_time:.1f}s")
print(f"GPU1 best iteration: {best_iter1}")
print(f"GPU2 best iteration: {best_iter2}")

# ---------------------------------------------------------------------
# Predictions and ensemble
# ---------------------------------------------------------------------
print("\n" + "="*60)
print("ENSEMBLE PREDICTIONS")
print("="*60)

pred1_future = trainer1.predict.remote(X_test_ref)
pred2_future = trainer2.predict.remote(X_test_ref)
test_pred_log1, test_pred_log2 = ray.get([pred1_future, pred2_future])

test_pred_log = (test_pred_log1 + test_pred_log2) / 2.0

test_df['pred_price_gpu1'] = np.exp(test_pred_log1) * test_df['price_baseline']
test_df['pred_price_gpu2'] = np.exp(test_pred_log2) * test_df['price_baseline']
test_df['pred_price_raw']  = np.exp(test_pred_log)  * test_df['price_baseline']

print("Ensemble formed in log space")

# ---------------------------------------------------------------------
# Calibration: isotonic regression in log space
# ---------------------------------------------------------------------
print("\n" + "="*60)
print("POST-FIT CALIBRATION (ISOTONIC)")
print("="*60)

# Validation predictions for calibration
val_p1 = ray.get(trainer1.predict.remote(X_val))
val_p2 = ray.get(trainer2.predict.remote(X_val))
val_pred_log = (val_p1 + val_p2) / 2.0

# True log ratio on validation
val_true_log = np.log(val_df[TARGET_COL]) - np.log(val_df['price_baseline'])

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(val_pred_log, val_true_log)

# Apply to test
test_pred_log_cal = iso.transform(test_pred_log)
test_df['pred_price_cal'] = np.exp(test_pred_log_cal) * test_df['price_baseline']

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
print("\n" + "="*60)
print("RESULTS")
print("="*60)

def print_metrics(name, y_true_price, y_pred_price):
    mae = mean_absolute_error(y_true_price, y_pred_price)
    mape = np.mean(np.abs((y_true_price - y_pred_price) / y_true_price)) * 100
    wape_val = wape(y_true_price, y_pred_price) * 100
    r2 = r2_score(y_true_price, y_pred_price)
    print(f"\n{name}:")
    print(f"  MAE:  ${mae:,.0f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  WAPE: {wape_val:.2f}%")
    print(f"  R²:   {r2:.4f}")
    return mae, mape, wape_val, r2

y_true_price_test = test_df[TARGET_COL].values

mae1 = mean_absolute_error(y_true_price_test, test_df['pred_price_gpu1'].values)
mae2 = mean_absolute_error(y_true_price_test, test_df['pred_price_gpu2'].values)
print(f"\nIndividual GPU MAE:")
print(f"  GPU1: ${mae1:,.0f}")
print(f"  GPU2: ${mae2:,.0f}")

mae_raw, mape_raw, wape_raw, r2_raw = print_metrics("Ensemble (raw)", y_true_price_test, test_df['pred_price_raw'].values)
mae_cal, mape_cal, wape_cal, r2_cal = print_metrics("Ensemble (calibrated)", y_true_price_test, test_df['pred_price_cal'].values)

print(f"\nEnsemble vs best single MAE delta: ${min(mae1, mae2) - mae_raw:,.0f}")
print(f"Calibration MAE delta: ${mae_raw - mae_cal:,.0f}")

# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
txt_file = f'full_distributed_results_V5_{timestamp}.txt'
pred_file = f'predictions_test_V5_{timestamp}.csv'

with open(txt_file, 'w') as f:
    f.write("TOWNHOUSE MODEL V5 - RESULTS\n")
    f.write("="*60 + "\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Features used: {len(feature_cols)}\n")
    f.write(f"Training time (s): {training_time:.1f}\n\n")
    f.write("Performance:\n")
    f.write(f"  Test MAE (raw ensemble): ${mae_raw:,.0f}\n")
    f.write(f"  Test MAPE (raw ensemble): {mape_raw:.2f}%\n")
    f.write(f"  Test WAPE (raw ensemble): {wape_raw:.2f}%\n")
    f.write(f"  Test R2   (raw ensemble): {r2_raw:.4f}\n")
    f.write(f"  Test MAE (calibrated): ${mae_cal:,.0f}\n")
    f.write(f"  Test MAPE (calibrated): {mape_cal:.2f}%\n")
    f.write(f"  Test WAPE (calibrated): {wape_cal:.2f}%\n")
    f.write(f"  Test R2   (calibrated): {r2_cal:.4f}\n")

test_export = test_df[[DATE_COL, TARGET_COL, 'price_baseline',
                       'pred_price_gpu1', 'pred_price_gpu2',
                       'pred_price_raw', 'pred_price_cal']].copy()
test_export.to_csv(pred_file, index=False)

print(f"\nSaved: {txt_file}")
print(f"Saved: {pred_file}")

# ---------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------
ray.shutdown()
print("\nDone.")