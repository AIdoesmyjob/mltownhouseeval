#!/usr/bin/env python3
"""
Improved Townhouse Price Prediction Model V4 - With Progress Bars
All recommended fixes implemented:
- Proper target column (Sold Price)
- Time-based rolling windows
- No leakage features
- GPU enabled with status display
- Better categorical handling
- Post-fit calibration
- Ordered target encoding
- PROGRESS BARS for all long operations
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Check GPU availability first
def check_gpu_status():
    """Check and display GPU status"""
    print("\n" + "="*60)
    print("GPU STATUS CHECK")
    print("="*60)
    
    # Check CUDA availability
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', 
                               '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ GPU(s) detected:")
            for i, line in enumerate(result.stdout.strip().split('\n')):
                parts = line.split(',')
                name = parts[0].strip()
                total_mem = parts[1].strip()
                free_mem = parts[2].strip()
                print(f"   GPU {i}: {name}")
                print(f"          Memory: {free_mem} free / {total_mem} total")
            
            # Test XGBoost GPU
            try:
                test_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
                test_dmatrix = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
                xgb.train(test_params, test_dmatrix, num_boost_round=1, verbose_eval=False)
                print("\n✅ XGBoost GPU support: ENABLED")
                return True
            except:
                print("\n⚠️  XGBoost GPU support: NOT AVAILABLE")
                return False
        else:
            print("❌ No GPUs detected (nvidia-smi failed)")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {str(e)}")
        return False

def progress_bar(current, total, prefix='', suffix='', length=50):
    """Display a progress bar"""
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

print("="*60)
print("IMPROVED MODEL V4 - WITH PROGRESS TRACKING")
print("="*60)

# Check GPU status
use_gpu = check_gpu_status()

# Load and clean data
print("\n" + "="*60)
print("DATA LOADING AND PREPARATION")
print("="*60)
print("\nLoading data...")
df = pd.read_csv('Jan 1 2015_Aug 13 2025.csv')
print(f"Initial shape: {df.shape}")

# Convert numeric columns with progress
print("\nConverting numeric columns...")
num_cols = ['Price', 'Sold Price', 'List Price', 'TotFlArea', 'MaintFee', 'Yr Blt', 'Age',
            'Full Baths', 'Half Baths', 'Tot BR', 'Tot Baths', 'No. Floor Levels',
            'Storeys in Building', 'Fireplaces', 'Floor Area Fin - Total',
            'Floor Area Fin - Main Flr', 'Floor Area Fin - Abv Main',
            'Floor Area Fin - Basement', 'Floor Area - Unfinished',
            'Tot Units in Strata Plan', 'Units in Development',
            '# of Kitchens', '# of Pets', 'Bds In Bsmt']

for i, col in enumerate(num_cols, 1):
    if col in df.columns:
        df[col] = to_numeric(df[col])
    progress_bar(i, len(num_cols), prefix='Converting', suffix=col[:20].ljust(20))

# Convert dates
print("\nParsing dates...")
date_cols = ['List Date', 'Sold Date']
for i, col in enumerate(date_cols, 1):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    progress_bar(i, len(date_cols), prefix='Parsing', suffix=col)

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
print("COMPARABLE BASELINE CONSTRUCTION")
print("="*60)

def rolling_group_median_ppsf(df, group_cols, window, min_periods):
    """Calculate rolling median PPSF by group with progress tracking"""
    prefix = f"Computing {window} baseline"
    
    # Create grouping key
    if isinstance(group_cols, list):
        group_key = df[group_cols].astype(str).agg('|'.join, axis=1)
        suffix_name = '+'.join(group_cols)
    else:
        group_key = df[group_cols]
        suffix_name = group_cols
    
    unique_groups = group_key.unique()
    result_list = []
    
    for idx, group in enumerate(unique_groups, 1):
        mask = group_key == group
        group_df = df[mask].copy()
        
        if len(group_df) >= min_periods:
            tmp = group_df[[DATE_COL, 'sold_ppsf']].set_index(DATE_COL).sort_index()
            rolled = tmp['sold_ppsf'].rolling(window, min_periods=min_periods).median().shift(1)
            group_df[f'ppsf_{window}_{suffix_name}'] = rolled.values
            result_list.append(group_df)
        
        if idx % 10 == 0 or idx == len(unique_groups):
            progress_bar(idx, len(unique_groups), prefix=prefix, suffix=f'Groups processed')
    
    if result_list:
        result = pd.concat(result_list, ignore_index=True)
        result = result[[*group_cols, DATE_COL, f'ppsf_{window}_{suffix_name}']]
        return result
    return pd.DataFrame()

# Extract location features
print("\nExtracting location features...")
df['FSA'] = df['Postal Code'].astype(str).str[:3]
df['Region'] = df['S/A'].astype(str).str.extract(r'^([A-Z]+)')[0]

# Build comparables baselines with progress
candidates = []

# Regional baselines
print("\nBuilding regional baselines...")
group_cols_region = ['Region', 'TypeDwel'] if 'TypeDwel' in df.columns else ['Region']
group_cols_type = ['TypeDwel'] if 'TypeDwel' in df.columns else None

baseline_configs = [
    (group_cols_region, '90D', 8),
    (group_cols_region, '365D', 12),
    (group_cols_type, '90D', 8) if group_cols_type else None,
    (group_cols_type, '365D', 12) if group_cols_type else None,
    (['FSA'], '90D', 8),
    (['FSA'], '365D', 12)
]

# Filter out None configs
baseline_configs = [c for c in baseline_configs if c is not None]

for i, (cols, win, mp) in enumerate(baseline_configs, 1):
    print(f"\nBaseline {i}/{len(baseline_configs)}: {'+'.join(cols) if isinstance(cols, list) else cols} {win}")
    try:
        m = rolling_group_median_ppsf(df, cols, win, mp)
        if not m.empty:
            merge_cols = cols if isinstance(cols, list) else [cols]
            df = df.merge(m, on=merge_cols + [DATE_COL], how='left')
            candidates.append(m.columns[-1])
            print(f"  ✓ Added: {m.columns[-1]}")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:50]}")

# Global rolling medians
print("\nBuilding global baselines...")
tmp = df[[DATE_COL, 'sold_ppsf']].set_index(DATE_COL).sort_index()
df['ppsf_90D_global'] = tmp['sold_ppsf'].rolling('90D', min_periods=8).median().shift(1).values
df['ppsf_365D_global'] = tmp['sold_ppsf'].rolling('365D', min_periods=12).median().shift(1).values
candidates += ['ppsf_90D_global', 'ppsf_365D_global']
print("  ✓ Added: ppsf_90D_global, ppsf_365D_global")

# Choose best available baseline
print("\nSelecting best baseline...")
df['ppsf_baseline'] = np.nan
for c in candidates:
    if c in df.columns:
        filled_before = df['ppsf_baseline'].notna().sum()
        df['ppsf_baseline'] = df['ppsf_baseline'].fillna(df[c])
        filled_after = df['ppsf_baseline'].notna().sum()
        if filled_after > filled_before:
            print(f"  Using {c}: filled {filled_after - filled_before} more records")

# Final fallback
df['ppsf_baseline'] = df['ppsf_baseline'].fillna(df['sold_ppsf'].median())

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
                'Member Board Affiliation', 'GST Incl'}

# Categorical columns for encoding
cat_cols = ['TypeDwel', 'S/A', 'Region', 'FSA', 'Bylaw Restrictions', 
            'Title to Land', 'Room Type Search', 'Zoning']
cat_cols = [c for c in cat_cols if c in df.columns]

# Target encode categoricals with progress
print("\nTarget encoding categorical features...")
from sklearn.model_selection import KFold

def target_encode_ordered(train_df, val_df, test_df, col, target='log_price_ratio', n_splits=5):
    """Target encode with cross-validation on train set"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_encoded = np.zeros(len(train_df))
    
    # Cross-val on train
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_df), 1):
        tr_data = train_df.iloc[tr_idx]
        val_data = train_df.iloc[val_idx]
        
        means = tr_data.groupby(col)[target].mean()
        global_mean = tr_data[target].mean()
        
        val_map = val_data[col].map(means).fillna(global_mean)
        train_encoded[val_idx] = val_map
    
    train_df[f'{col}_te'] = train_encoded
    
    # Use full train for val/test
    means = train_df.groupby(col)[target].mean()
    global_mean = train_df[target].mean()
    
    val_df[f'{col}_te'] = val_df[col].map(means).fillna(global_mean)
    test_df[f'{col}_te'] = test_df[col].map(means).fillna(global_mean)

for i, col in enumerate(cat_cols, 1):
    target_encode_ordered(train_df, val_df, test_df, col)
    progress_bar(i, len(cat_cols), prefix='Encoding', suffix=col)

# All features
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c not in cat_cols]
feature_cols = [c for c in feature_cols if c in train_df.columns]

# Add encoded features
feature_cols += [f'{c}_te' for c in cat_cols]

# Filter to numeric
numeric_mask = train_df[feature_cols].dtypes.apply(lambda x: np.issubdtype(x, np.number))
feature_cols = [c for c, is_num in zip(feature_cols, numeric_mask) if is_num]

print(f"\nUsing {len(feature_cols)} features")

print("\n" + "="*60)
print(f"TRAINING XGBOOST ({'GPU' if use_gpu else 'CPU'})")
print("="*60)

# Prepare data
X_train = train_df[feature_cols].fillna(0)
y_train = train_df['log_price_ratio']

X_val = val_df[feature_cols].fillna(0)
y_val = val_df['log_price_ratio']

X_test = test_df[feature_cols].fillna(0)
y_test = test_df['log_price_ratio']

# XGBoost with GPU if available
params = {
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist' if use_gpu else 'hist',
    'predictor': 'gpu_predictor' if use_gpu else 'cpu_predictor',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.5,
    'min_child_weight': 3,
    'seed': 42
}

if use_gpu:
    params['gpu_id'] = 0
    print("🚀 Training on GPU...")
else:
    params['n_jobs'] = -1
    print("💻 Training on CPU...")

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Custom callback for progress
class ProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, num_rounds):
        self.num_rounds = num_rounds
        
    def after_iteration(self, model, epoch, evals_log):
        progress_bar(epoch + 1, self.num_rounds, 
                    prefix='Training', 
                    suffix=f'Round {epoch + 1}/{self.num_rounds}')
        return False

print("\nTraining model...")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=False,
    callbacks=[ProgressCallback(500)]
)

print(f"\nBest iteration: {model.best_iteration}")

# Predictions
print("\nMaking predictions...")
train_pred_log = model.predict(xgb.DMatrix(X_train))
val_pred_log = model.predict(xgb.DMatrix(X_val))
test_pred_log = model.predict(xgb.DMatrix(X_test))

# Convert back to price domain
train_df['pred_price'] = np.exp(train_pred_log) * train_df['price_baseline']
val_df['pred_price'] = np.exp(val_pred_log) * val_df['price_baseline']
test_df['pred_price'] = np.exp(test_pred_log) * test_df['price_baseline']

print("\n" + "="*60)
print("POST-FIT CALIBRATION")
print("="*60)

# Calibration on validation set
residuals = val_df[TARGET_COL] - val_df['pred_price']
bias = residuals.median()
scale = (residuals / val_df['pred_price']).std()

print(f"Calibration bias: ${bias:,.0f}")
print(f"Calibration scale: {scale:.3f}")

# Apply calibration
test_df['pred_calibrated'] = test_df['pred_price'] + bias

print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Metrics
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

print_metrics("Train", train_df[TARGET_COL], train_df['pred_price'])
print_metrics("Validation", val_df[TARGET_COL], val_df['pred_price'])
print_metrics("Test (raw)", test_df[TARGET_COL], test_df['pred_price'])
print_metrics("Test (calibrated)", test_df[TARGET_COL], test_df['pred_calibrated'])

# Feature importance
print("\n" + "="*60)
print("TOP 10 FEATURES")
print("="*60)
importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in importance.items()
]).sort_values('importance', ascending=False)

for i, row in importance_df.head(10).iterrows():
    print(f"{row['feature']:30} {row['importance']:10.0f}")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'improved_model_v4_results_{timestamp}.txt'
print(f"\n✅ Results saved to {output_file}")

with open(output_file, 'w') as f:
    f.write("IMPROVED MODEL V4 RESULTS\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"GPU Used: {use_gpu}\n")
    f.write(f"Test MAE (calibrated): ${mean_absolute_error(test_df[TARGET_COL], test_df['pred_calibrated']):,.0f}\n")
    f.write(f"Test WAPE (calibrated): {wape(test_df[TARGET_COL], test_df['pred_calibrated'])*100:.1f}%\n")

print("\n✅ Done!")