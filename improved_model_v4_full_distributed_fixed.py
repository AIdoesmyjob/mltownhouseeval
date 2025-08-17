#!/usr/bin/env python3
"""
Improved Townhouse Price Prediction Model V4 - Full Distributed Version (FIXED)
Complete feature engineering + Both GPUs required
Fixed memory issues and improved progress tracking
"""

import ray
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
from datetime import datetime
import warnings
import sys
import subprocess
import time
import gc  # For garbage collection
warnings.filterwarnings('ignore')

# Ray cluster configuration
RAY_HEAD = "ray://10.0.0.198:10001"

def check_ray_cluster_gpus():
    """Check Ray cluster for available GPUs"""
    print("\n" + "="*60)
    print("RAY CLUSTER GPU CHECK")
    print("="*60)
    
    print(f"Connecting to Ray cluster at {RAY_HEAD}...")
    try:
        ray.init(address=RAY_HEAD)
        print("✅ Connected to Ray cluster")
    except Exception as e:
        print(f"❌ Failed to connect to Ray cluster: {e}")
        return False
    
    resources = ray.cluster_resources()
    available = ray.available_resources()
    
    total_gpus = resources.get('GPU', 0)
    available_gpus = available.get('GPU', 0)
    
    print(f"\nCluster Resources:")
    print(f"  Total GPUs: {total_gpus}")
    print(f"  Available GPUs: {available_gpus}")
    
    nodes = ray.nodes()
    gpu_count = 0
    
    print("\nNode Details:")
    for node in nodes:
        if node['Alive']:
            node_resources = node.get('Resources', {})
            node_gpus = node_resources.get('GPU', 0)
            if node_gpus > 0:
                gpu_count += 1
                node_ip = node['NodeManagerAddress']
                print(f"  📍 Node {node_ip}: {node_gpus} GPU(s)")
    
    if total_gpus < 2:
        print(f"\n❌ ERROR: Need 2 GPUs but only {total_gpus} detected!")
        ray.shutdown()
        return False
    
    print(f"\n✅ Both GPUs detected and available!")
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

# Ray remote training functions
@ray.remote(num_gpus=1)
class GPUTrainer:
    """GPU trainer actor to maintain model state"""
    
    def __init__(self, gpu_name):
        self.gpu_name = gpu_name
        self.model = None
        
    def train(self, X_train, y_train, X_val, y_val, params, num_rounds=500):
        """Train XGBoost model on GPU"""
        import xgboost as xgb
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        print(f"[{self.gpu_name}] Starting training...")
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return self.model.best_iteration
    
    def predict(self, X):
        """Make predictions using trained model"""
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)
    
    def get_importance(self):
        """Get feature importance"""
        if self.model:
            return self.model.get_score(importance_type='gain')
        return {}

print("="*60)
print("FULL FEATURED MODEL V4 - DISTRIBUTED GPU VERSION (FIXED)")
print("="*60)

# Check for both GPUs before proceeding
if not check_ray_cluster_gpus():
    print("\n❌ Exiting: Both GPUs must be available")
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
    progress_bar(i, len(num_cols), prefix='Converting', suffix=col[:20].ljust(20))

# Convert dates
print("\nParsing dates...")
date_cols = ['List Date', 'Sold Date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Filter to valid sold properties
TARGET_COL = 'Sold Price'
AREA_COL = 'TotFlArea'
DATE_COL = 'List Date'

df = df[df[TARGET_COL].notna() & df[TARGET_COL].gt(0)].copy()
df = df[df[AREA_COL].notna() & df[AREA_COL].gt(0)].copy()
df = df[df[DATE_COL].notna()].copy()
df = df.sort_values(DATE_COL).reset_index(drop=True)
print(f"\nAfter filtering: {len(df)} valid sales")

# Create sold price per sqft
df['sold_ppsf'] = df[TARGET_COL] / df[AREA_COL]

print("\n" + "="*60)
print("COMPARABLE BASELINE CONSTRUCTION (OPTIMIZED)")
print("="*60)

def rolling_median_ppsf_optimized(df, group_col, window, min_periods):
    """Optimized rolling median calculation"""
    result_col_name = f'ppsf_{window}_{group_col}'
    
    print(f"  Computing {result_col_name}...")
    
    # Group and calculate rolling median
    df_sorted = df.sort_values([group_col, DATE_COL])
    
    # Calculate rolling median for each group
    df_sorted[result_col_name] = (df_sorted.groupby(group_col)['sold_ppsf']
                                   .transform(lambda x: x.rolling(window, min_periods=min_periods)
                                                        .median()
                                                        .shift(1)))
    
    # Only keep the new column and merge keys
    result = df_sorted[[group_col, DATE_COL, result_col_name]].copy()
    
    # Count non-null values
    non_null = result[result_col_name].notna().sum()
    print(f"    ✓ Computed {non_null} baseline values")
    
    return result

# Extract location features
print("\nExtracting location features...")
df['FSA'] = df['Postal Code'].astype(str).str[:3]
df['Region'] = df['S/A'].astype(str).str.extract(r'^([A-Z]+)')[0]

# Build baselines more efficiently
print("\nBuilding baseline features...")
candidates = []

# Only compute if TypeDwel exists
if 'TypeDwel' in df.columns:
    print("\n1. TypeDwel baselines:")
    for window, min_p in [('90D', 8), ('365D', 12)]:
        try:
            result = rolling_median_ppsf_optimized(df, 'TypeDwel', window, min_p)
            df = df.merge(result, on=['TypeDwel', DATE_COL], how='left')
            candidates.append(result.columns[-1])
            gc.collect()  # Force garbage collection
        except Exception as e:
            print(f"    ✗ Failed: {str(e)[:50]}")

# FSA baselines
print("\n2. FSA (postal code) baselines:")
for window, min_p in [('90D', 8), ('365D', 12)]:
    try:
        result = rolling_median_ppsf_optimized(df, 'FSA', window, min_p)
        df = df.merge(result, on=['FSA', DATE_COL], how='left')
        candidates.append(result.columns[-1])
        gc.collect()
    except Exception as e:
        print(f"    ✗ Failed: {str(e)[:50]}")

# Region baselines
if 'Region' in df.columns:
    print("\n3. Region baselines:")
    for window, min_p in [('90D', 8), ('365D', 12)]:
        try:
            result = rolling_median_ppsf_optimized(df, 'Region', window, min_p)
            df = df.merge(result, on=['Region', DATE_COL], how='left')
            candidates.append(result.columns[-1])
            gc.collect()
        except Exception as e:
            print(f"    ✗ Failed: {str(e)[:50]}")

# Global rolling medians
print("\n4. Global baselines:")
tmp = df[[DATE_COL, 'sold_ppsf']].set_index(DATE_COL).sort_index()
print("  Computing 90-day global median...")
df['ppsf_90D_global'] = tmp['sold_ppsf'].rolling('90D', min_periods=8).median().shift(1).values
print("    ✓ Added: ppsf_90D_global")
print("  Computing 365-day global median...")
df['ppsf_365D_global'] = tmp['sold_ppsf'].rolling('365D', min_periods=12).median().shift(1).values
print("    ✓ Added: ppsf_365D_global")
candidates += ['ppsf_90D_global', 'ppsf_365D_global']

# Choose best available baseline
print("\n5. Selecting best baseline for each record...")
df['ppsf_baseline'] = np.nan
for i, c in enumerate(candidates, 1):
    if c in df.columns:
        filled_before = df['ppsf_baseline'].notna().sum()
        df['ppsf_baseline'] = df['ppsf_baseline'].fillna(df[c])
        filled_after = df['ppsf_baseline'].notna().sum()
        if filled_after > filled_before:
            progress_bar(i, len(candidates), 
                        prefix='  Selecting', 
                        suffix=f'{c[:20]} ({filled_after}/{len(df)} filled)')

# Final fallback
df['ppsf_baseline'] = df['ppsf_baseline'].fillna(df['sold_ppsf'].median())
print(f"\n  ✓ All {len(df)} records have baseline")

# Baseline price & log residual target
df['price_baseline'] = df['ppsf_baseline'] * df[AREA_COL]
df = df[df['price_baseline'].gt(0)].copy()
df['log_price_ratio'] = np.log(df[TARGET_COL]) - np.log(df['price_baseline'])

print(f"\nBaseline PPSF range: ${df['ppsf_baseline'].min():.0f}–${df['ppsf_baseline'].max():.0f}")
print(f"Records after baseline: {len(df)}")

print("\n" + "="*60)
print("TEMPORAL SPLIT")
print("="*60)

# Time-based split
n = len(df)
train_end = int(0.7 * n)
val_end = int(0.8 * n)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print(f"Train: {len(train_df)} ({train_df[DATE_COL].min().date()} to {train_df[DATE_COL].max().date()})")
print(f"Val:   {len(val_df)} ({val_df[DATE_COL].min().date()} to {val_df[DATE_COL].max().date()})")
print(f"Test:  {len(test_df)} ({test_df[DATE_COL].min().date()} to {test_df[DATE_COL].max().date()})")

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Select features (no leakage)
EXCLUDE_COLS = {TARGET_COL, 'log_price_ratio', 'Sold Price', 'Price', 'List Price',
                'Sold Date', 'sold_ppsf', 'price_baseline', 'ppsf_baseline',
                'DOM', 'Cumulative DOM', 'Expiry Date', 'SP/LP Ratio', 'SP/OLP Ratio',
                'Sold Price per SqFt', 'Price Per SQFT', DATE_COL,
                'Address', 'Postal Code', 'Complex/Subdivision Name',
                'Member Board Affiliation', 'GST Incl', 'Confirm Sold Date'}

# Categorical columns for encoding
cat_cols = ['TypeDwel', 'S/A', 'Region', 'FSA', 'Bylaw Restrictions', 
            'Title to Land', 'Room Type Search', 'Zoning', 'Status']
cat_cols = [c for c in cat_cols if c in df.columns]

# Target encode categoricals
print("\nTarget encoding categorical features...")

def target_encode_ordered(train_df, val_df, test_df, col, target='log_price_ratio', n_splits=5):
    """Target encode with cross-validation on train set"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_encoded = np.zeros(len(train_df))
    
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        tr_data = train_df.iloc[tr_idx]
        val_data = train_df.iloc[val_idx]
        
        means = tr_data.groupby(col)[target].mean()
        global_mean = tr_data[target].mean()
        
        val_map = val_data[col].map(means).fillna(global_mean)
        train_encoded[val_idx] = val_map
    
    train_df[f'{col}_te'] = train_encoded
    
    means = train_df.groupby(col)[target].mean()
    global_mean = train_df[target].mean()
    
    val_df[f'{col}_te'] = val_df[col].map(means).fillna(global_mean)
    test_df[f'{col}_te'] = test_df[col].map(means).fillna(global_mean)

for i, col in enumerate(cat_cols, 1):
    target_encode_ordered(train_df, val_df, test_df, col)
    progress_bar(i, len(cat_cols), prefix='  Encoding', suffix=col)

# All features
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c not in cat_cols]
feature_cols = [c for c in feature_cols if c in train_df.columns]

# Add encoded features
feature_cols += [f'{c}_te' for c in cat_cols]

# Filter to numeric
numeric_mask = train_df[feature_cols].dtypes.apply(lambda x: np.issubdtype(x, np.number))
feature_cols = [c for c, is_num in zip(feature_cols, numeric_mask) if is_num]

print(f"\n✓ Using {len(feature_cols)} features")

# Show top features
print("\nSample features:")
for i, feat in enumerate(feature_cols[:10], 1):
    print(f"  {i:2}. {feat}")
if len(feature_cols) > 10:
    print(f"  ... and {len(feature_cols) - 10} more")

print("\n" + "="*60)
print("DISTRIBUTED TRAINING ON 2 GPUs")
print("="*60)

# Prepare data
X_train = train_df[feature_cols].fillna(0).values
y_train = train_df['log_price_ratio'].values

X_val = val_df[feature_cols].fillna(0).values
y_val = val_df['log_price_ratio'].values

X_test = test_df[feature_cols].fillna(0).values
y_test = test_df['log_price_ratio'].values

# XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.5,
    'min_child_weight': 3,
    'gpu_id': 0,
    'seed': 42
}

print("\n🚀 Training two models in parallel on separate GPUs...")
print("   Model 1: RTX 4090 (10.0.0.198) - seed 42")
print("   Model 2: RTX 3090 (10.0.0.75) - seed 123")

# Put data in Ray object store
X_train_ref = ray.put(X_train)
y_train_ref = ray.put(y_train)
X_val_ref = ray.put(X_val)
y_val_ref = ray.put(y_val)
X_test_ref = ray.put(X_test)

# Create GPU trainers
trainer1 = GPUTrainer.remote("GPU1-4090")
trainer2 = GPUTrainer.remote("GPU2-3090")

# Different seeds for ensemble diversity
params1 = params.copy()
params1['seed'] = 42

params2 = params.copy()
params2['seed'] = 123

# Start parallel training
print("\nTraining in progress on both GPUs...")
print("(This may take 1-2 minutes)")
start_time = time.time()

future1 = trainer1.train.remote(X_train_ref, y_train_ref, X_val_ref, y_val_ref, params1, num_rounds=500)
future2 = trainer2.train.remote(X_train_ref, y_train_ref, X_val_ref, y_val_ref, params2, num_rounds=500)

# Show progress while waiting
print("\nWaiting for training to complete...")
while True:
    ready, not_ready = ray.wait([future1, future2], num_returns=2, timeout=5)
    if len(ready) == 2:
        break
    elapsed = int(time.time() - start_time)
    sys.stdout.write(f'\r  Elapsed: {elapsed}s... Still training on both GPUs')
    sys.stdout.flush()

print()  # New line after progress

# Get results
best_iter1, best_iter2 = ray.get([future1, future2])
training_time = time.time() - start_time

print(f"\n✅ Training completed in {training_time:.1f} seconds!")
print(f"   GPU1 best iteration: {best_iter1}")
print(f"   GPU2 best iteration: {best_iter2}")

print("\n" + "="*60)
print("ENSEMBLE PREDICTIONS")
print("="*60)

# Get predictions from both models
print("Making predictions on test set...")
pred1_future = trainer1.predict.remote(X_test_ref)
pred2_future = trainer2.predict.remote(X_test_ref)

test_pred_log1, test_pred_log2 = ray.get([pred1_future, pred2_future])

# Ensemble predictions (average in log space)
test_pred_log = (test_pred_log1 + test_pred_log2) / 2

# Convert back to price domain
test_df['pred_price_gpu1'] = np.exp(test_pred_log1) * test_df['price_baseline']
test_df['pred_price_gpu2'] = np.exp(test_pred_log2) * test_df['price_baseline']
test_df['pred_price'] = np.exp(test_pred_log) * test_df['price_baseline']

print("✅ Ensemble created from both GPU models")

print("\n" + "="*60)
print("POST-FIT CALIBRATION")
print("="*60)

# Get validation predictions for calibration
print("Computing calibration on validation set...")
val_pred1 = ray.get(trainer1.predict.remote(ray.put(X_val)))
val_pred2 = ray.get(trainer2.predict.remote(ray.put(X_val)))
val_pred_log = (val_pred1 + val_pred2) / 2
val_df['pred_price'] = np.exp(val_pred_log) * val_df['price_baseline']

# Calibration on validation set
residuals = val_df[TARGET_COL] - val_df['pred_price']
bias = residuals.median()

print(f"Calibration bias: ${bias:,.0f}")

# Apply calibration
test_df['pred_calibrated'] = test_df['pred_price'] + bias

print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Metrics function
def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    wape_val = wape(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{name}:")
    print(f"  MAE:  ${mae:,.0f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  WAPE: {wape_val:.1f}%")
    print(f"  R²:   {r2:.3f}")
    
    return mae, mape, wape_val, r2

# Individual GPU performance
mae1 = mean_absolute_error(test_df[TARGET_COL], test_df['pred_price_gpu1'])
mae2 = mean_absolute_error(test_df[TARGET_COL], test_df['pred_price_gpu2'])

print(f"\nIndividual GPU Performance:")
print(f"  GPU 1 (4090) MAE: ${mae1:,.0f}")
print(f"  GPU 2 (3090) MAE: ${mae2:,.0f}")

# Ensemble performance
mae_raw, mape_raw, wape_raw, r2_raw = print_metrics("Ensemble (raw)", test_df[TARGET_COL], test_df['pred_price'])
mae_cal, mape_cal, wape_cal, r2_cal = print_metrics("Ensemble (calibrated)", test_df[TARGET_COL], test_df['pred_calibrated'])

print(f"\nImprovement from ensemble: ${min(mae1, mae2) - mae_raw:,.0f}")
print(f"Improvement from calibration: ${mae_raw - mae_cal:,.0f}")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'full_distributed_results_{timestamp}.txt'

with open(output_file, 'w') as f:
    f.write("FULL FEATURED DISTRIBUTED GPU MODEL RESULTS\n")
    f.write("="*60 + "\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Features used: {len(feature_cols)}\n")
    f.write(f"Training time: {training_time:.1f} seconds\n\n")
    
    f.write("Performance:\n")
    f.write(f"  Test MAE (calibrated): ${mae_cal:,.0f}\n")
    f.write(f"  Test MAPE (calibrated): {mape_cal:.1f}%\n")
    f.write(f"  Test WAPE (calibrated): {wape_cal:.1f}%\n")

print(f"\n✅ Results saved to {output_file}")

# Cleanup
ray.shutdown()
print("\n✅ Done! Ray cluster connection closed.")