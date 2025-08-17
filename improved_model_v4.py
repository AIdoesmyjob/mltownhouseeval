#!/usr/bin/env python3
"""
Improved Townhouse Price Prediction Model V4
All recommended fixes implemented:
- Proper target column (Sold Price)
- Time-based rolling windows
- No leakage features
- GPU enabled
- Better categorical handling
- Post-fit calibration
- Ordered target encoding
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
print("IMPROVED MODEL V4 - WITH ALL FIXES")
print("="*60)

# Load and clean data
print("\nLoading data...")
df = pd.read_csv('Jan 1 2015_Aug 13 2025.csv')
print(f"Initial shape: {df.shape}")

# Convert numeric columns
num_cols = ['Price', 'Sold Price', 'List Price', 'TotFlArea', 'MaintFee', 'Yr Blt', 'Age',
            'Full Baths', 'Half Baths', 'Tot BR', 'Tot Baths', 'No. Floor Levels',
            'Storeys in Building', 'Fireplaces', 'Floor Area Fin - Total',
            'Floor Area Fin - Main Flr', 'Floor Area Fin - Abv Main',
            'Floor Area Fin - Basement', 'Floor Area - Unfinished',
            'Tot Units in Strata Plan', 'Units in Development',
            '# of Kitchens', '# of Pets']  # Removed DOM and Cumulative DOM

for col in num_cols:
    if col in df.columns:
        df[col] = to_numeric(df[col])

# Parse dates
df['List Date'] = pd.to_datetime(df['List Date'], errors='coerce')
if 'Sold Date' in df.columns:
    df['Sold Date'] = pd.to_datetime(df['Sold Date'], errors='coerce')

# === Canonical columns ===
TARGET_COL = 'Sold Price' if 'Sold Price' in df.columns else 'Price'
AREA_COL = 'TotFlArea'
DATE_COL = 'List Date'

print(f"Using target column: {TARGET_COL}")

# Keep valid rows
df = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0) &
        df[AREA_COL].notna() & (df[AREA_COL] > 0) &
        df[DATE_COL].notna()].copy()

# Sort by time
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Use SOLD PPSF for comps (not list price)
df['sold_ppsf'] = df[TARGET_COL] / df[AREA_COL]

print(f"After filtering: {len(df)} valid records")

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Time features
ref_date = pd.Timestamp('2015-01-01')
df['month_idx'] = (df[DATE_COL] - ref_date).dt.days / 30.0
df['year'] = df[DATE_COL].dt.year.astype('float32')
df['month'] = df[DATE_COL].dt.month.astype('float32')
df['quarter'] = ((df['month'] - 1) // 3 + 1).astype('float32')
df['day_of_year'] = df[DATE_COL].dt.dayofyear.astype('float32')
df['week_of_year'] = df[DATE_COL].dt.isocalendar().week.astype('float32')

# Age at listing
df['age_at_list'] = np.where(
    df['Yr Blt'].notna(),
    df['year'] - df['Yr Blt'].clip(lower=1900, upper=df['year']),
    df['Age']
)
df['age_at_list'] = df['age_at_list'].clip(lower=0).fillna(20)

# Location features
if 'Postal Code' in df.columns:
    df['FSA'] = df['Postal Code'].astype(str).str.extract(r'^([A-Za-z]\d[A-Za-z])', expand=False)
    df['FSA'] = df['FSA'].fillna('UNK')
else:
    df['FSA'] = 'UNK'

# Ratios (no arbitrary *1000)
df['rooms_per_sqft'] = (df['Tot BR'] / df[AREA_COL]).replace([np.inf, -np.inf], np.nan)
df['baths_equiv'] = (df['Full Baths'].fillna(0) + 0.5 * df['Half Baths'].fillna(0))
df['baths_per_br'] = df['baths_equiv'] / df['Tot BR'].clip(lower=1)
df['maint_per_sqft'] = (df['MaintFee'] / df[AREA_COL]).replace([np.inf, -np.inf], np.nan)

# Winsorize a few heavy tails
for c in ['sold_ppsf', 'maint_per_sqft', AREA_COL]:
    if c in df.columns:
        lo, hi = df[c].quantile([0.005, 0.995])
        df[c] = df[c].clip(lo, hi)

# Missingness indicators
for c in ['MaintFee', 'Floor Area Fin - Basement', 'Floor Area - Unfinished']:
    if c in df.columns:
        df[f'{c}_isna'] = df[c].isna().astype('float32')

print("\n" + "="*60)
print("BUILDING COMP BASELINE (time-based)")
print("="*60)

# Helper: rolling median PPSF over past window, time-safe (shifted)
def rolling_group_median_ppsf(df, group_cols, window='90D', min_periods=8):
    tmp = df[[DATE_COL] + group_cols + ['sold_ppsf']].dropna(subset=['sold_ppsf']).copy()
    tmp = tmp.sort_values(DATE_COL)
    med = (tmp.groupby(group_cols)
              .rolling(window=window, on=DATE_COL, min_periods=min_periods)
              .sold_ppsf.median()
              .shift(1))  # don't use current sale
    med.name = f'ppsf_{window}_{"_".join(group_cols)}_med'
    return med.reset_index()

group_cols = ['FSA']
if 'TypeDwel' in df.columns:  # finer comps if available
    group_cols_type = ['FSA', 'TypeDwel']
else:
    group_cols_type = None

# Primary (FSA + TypeDwel), fallback to FSA-only, then global
candidates = []

if group_cols_type:
    for win, mp in [('90D', 8), ('365D', 12)]:
        try:
            m = rolling_group_median_ppsf(df, group_cols_type, win, mp)
            df = df.merge(m, on=group_cols_type + [DATE_COL], how='left')
            candidates.append(m.columns[-1])
        except:
            pass

for win, mp in [('90D', 8), ('365D', 12)]:
    try:
        m = rolling_group_median_ppsf(df, ['FSA'], win, mp)
        df = df.merge(m, on=['FSA', DATE_COL], how='left')
        candidates.append(m.columns[-1])
    except:
        pass

# Global rolling medians
tmp = df[[DATE_COL, 'sold_ppsf']].set_index(DATE_COL).sort_index()
df['ppsf_90D_global'] = tmp['sold_ppsf'].rolling('90D', min_periods=8).median().shift(1).values
df['ppsf_365D_global'] = tmp['sold_ppsf'].rolling('365D', min_periods=12).median().shift(1).values
candidates += ['ppsf_90D_global', 'ppsf_365D_global']

# Choose best available baseline in priority order
df['ppsf_baseline'] = np.nan
for c in candidates:
    if c in df.columns:
        df['ppsf_baseline'] = df['ppsf_baseline'].fillna(df[c])

# Final fallback to long-run global median if needed
df['ppsf_baseline'] = df['ppsf_baseline'].fillna(df['sold_ppsf'].median())

# Baseline price & log residual target
df['price_baseline'] = df['ppsf_baseline'] * df[AREA_COL]
df = df[df['price_baseline'].gt(0)].copy()
df['log_price_ratio'] = np.log(df[TARGET_COL]) - np.log(df['price_baseline'])

print(f"Baseline PPSF range: ${df['ppsf_baseline'].min():.0f}–${df['ppsf_baseline'].max():.0f}")
print(f"Records after baseline: {len(df)}")

print("\n" + "="*60)
print("TEMPORAL SPLIT")
print("="*60)

# Time-based split
n = len(df)
train_end = int(0.7 * n)
val_end = int(0.8 * n)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f"Train: {len(df_train)} samples ({df_train[DATE_COL].min().date()} to {df_train[DATE_COL].max().date()})")
print(f"Val: {len(df_val)} samples ({df_val[DATE_COL].min().date()} to {df_val[DATE_COL].max().date()})")
print(f"Test: {len(df_test)} samples ({df_test[DATE_COL].min().date()} to {df_test[DATE_COL].max().date()})")

print("\n" + "="*60)
print("ORDERED TARGET ENCODING")
print("="*60)

def ordered_te(train_df, val_df, test_df, col, target_col='log_price_ratio'):
    """Time-safe ordered target encoding"""
    if col not in train_df.columns:
        return train_df, val_df, test_df
    
    # Compute expanding mean per category in TRAIN only (time-safe)
    t = train_df[[col, DATE_COL, target_col]].copy().sort_values(DATE_COL)
    t['cnt'] = 1
    t['cum_sum'] = t.groupby(col)[target_col].cumsum() - t[target_col]
    t['cum_cnt'] = t.groupby(col)['cnt'].cumsum() - 1
    global_mean = t[target_col].mean()
    t['te'] = (t['cum_sum'] / t['cum_cnt'].replace(0, np.nan)).fillna(global_mean)
    
    # Add to frames (val/test use train's last TE per cat)
    train_df[col + '_te'] = t['te'].values
    last_te = t.groupby(col)['te'].last()
    val_df[col + '_te'] = val_df[col].map(last_te).fillna(global_mean)
    test_df[col + '_te'] = test_df[col].map(last_te).fillna(global_mean)
    return train_df, val_df, test_df

# Apply to high-cardinality categoricals
for col in ['Complex/Subdivision Name', 'Address']:
    if col in df.columns:
        df_train, df_val, df_test = ordered_te(df_train, df_val, df_test, col)
        print(f"Added target encoding for {col}")

print("\n" + "="*60)
print("FEATURE PREPARATION")
print("="*60)

# Numeric features (removed DOM and Cumulative DOM)
num_features = [AREA_COL, 'MaintFee', 'age_at_list', 'Full Baths', 'Half Baths',
               'Tot BR', 'Tot Baths', 'No. Floor Levels', 'Storeys in Building',
               'Fireplaces', 'Floor Area Fin - Total', 'Floor Area Fin - Main Flr',
               'Floor Area Fin - Abv Main', 'Floor Area Fin - Basement',
               'Floor Area - Unfinished', 'Tot Units in Strata Plan',
               'Units in Development', 'month_idx', 'year', 'month', 'quarter',
               'day_of_year', 'week_of_year', 'rooms_per_sqft', 'baths_per_br',
               'maint_per_sqft', 'baths_equiv', 'ppsf_baseline']

# Add missingness indicators
num_features += [c for c in df.columns if c.endswith('_isna')]

# Add target encodings
num_features += [c for c in df.columns if c.endswith('_te')]

# Keep only existing columns
num_features = [f for f in num_features if f in df.columns]
print(f"Numeric features: {len(num_features)}")

# One-hot encoding with __OTHER__ bin
def ohe_with_other(train_s, val_s, test_s, min_freq=30):
    vc = train_s.fillna('NA').value_counts()
    keep = set(vc[vc >= min_freq].index)
    
    def transform(s):
        s = s.fillna('NA').where(s.fillna('NA').isin(keep), '__OTHER__')
        return pd.get_dummies(s, prefix=s.name)
    
    Xtr = transform(train_s)
    Xva = pd.get_dummies(val_s.fillna('NA').where(val_s.fillna('NA').isin(keep), '__OTHER__'),
                         prefix=val_s.name).reindex(columns=Xtr.columns, fill_value=0)
    Xte = pd.get_dummies(test_s.fillna('NA').where(test_s.fillna('NA').isin(keep), '__OTHER__'),
                         prefix=test_s.name).reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, Xva, Xte

X_train_list = [df_train[num_features].fillna(0)]
X_val_list = [df_val[num_features].fillna(0)]
X_test_list = [df_test[num_features].fillna(0)]

# Categorical features
cat_features = ['TypeDwel', 'FSA', 'Restricted Age', 'Title to Land', 'Zoning']
for cat in cat_features:
    if cat in df.columns:
        tr, va, te = ohe_with_other(df_train[cat], df_val[cat], df_test[cat], min_freq=30)
        X_train_list.append(tr)
        X_val_list.append(va)
        X_test_list.append(te)
        print(f"  {cat}: {tr.shape[1]} categories encoded")

# Combine all features
X_train = pd.concat(X_train_list, axis=1)
X_val = pd.concat(X_val_list, axis=1)
X_test = pd.concat(X_test_list, axis=1)

print(f"\nFinal feature count: {X_train.shape[1]}")

# Target variable
y_train = df_train['log_price_ratio'].values
y_val = df_val['log_price_ratio'].values
y_test = df_test['log_price_ratio'].values

print("\n" + "="*60)
print("SAMPLE WEIGHTS")
print("="*60)

def recency_weights(dates, half_life_days=180):
    max_date = dates.max()
    age = (max_date - dates).dt.days.clip(lower=0)
    return np.power(0.5, age / half_life_days)

w_rec = recency_weights(df_train[DATE_COL], half_life_days=365)
w_wape = 1.0 / np.maximum(df_train[TARGET_COL].values, 1_000.0)  # stabilize WAPE
sample_weights = (w_rec * w_wape) / np.mean(w_rec * w_wape)

print(f"Sample weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

print("\n" + "="*60)
print("TRAINING XGBOOST (GPU)")
print("="*60)

# Create DMatrix
dtrain = xgb.DMatrix(X_train.values, label=y_train, weight=sample_weights)
dval = xgb.DMatrix(X_val.values, label=y_val)
dtest = xgb.DMatrix(X_test.values)

# Try GPU first, fallback to CPU
try:
    # Test if GPU is available
    test_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
    test_dm = xgb.DMatrix(np.array([[1,2],[3,4]]), label=[1,0])
    xgb.train(test_params, test_dm, num_boost_round=1, verbose_eval=False)
    use_gpu = True
    print("GPU detected and will be used")
except:
    use_gpu = False
    print("GPU not available, using CPU")

# Set parameters based on availability
params = {
    'tree_method': 'gpu_hist' if use_gpu else 'hist',
    'predictor': 'gpu_predictor' if use_gpu else 'cpu_predictor',
    'max_depth': 8,
    'min_child_weight': 6.0,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'reg_lambda': 2.0,
    'reg_alpha': 0.0,
    'objective': 'reg:squarederror',  # optimizing log-residual
    'eval_metric': 'mae',
    'seed': 42
}

# Train model
model = xgb.train(
    params, dtrain,
    num_boost_round=2000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=50
)

print(f"\nBest iteration: {model.best_iteration}")

print("\n" + "="*60)
print("POST-FIT CALIBRATION")
print("="*60)

# Get predictions
train_pred_log_ratio = model.predict(dtrain, iteration_range=(0, model.best_iteration + 1))
val_pred_log_ratio = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
test_pred_log_ratio = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

# Calibrate residuals on validation (linear)
val_true_log_ratio = np.log(df_val[TARGET_COL]) - np.log(df_val['price_baseline'])
a, b = np.polyfit(val_pred_log_ratio, val_true_log_ratio, 1)
print(f"Calibration: y = {a:.3f} + {b:.3f} * x")

val_pred_log_ratio_cal = a + b * val_pred_log_ratio
test_pred_log_ratio_cal = a + b * test_pred_log_ratio

# Convert back to prices
df_train['pred_price'] = df_train['price_baseline'] * np.exp(train_pred_log_ratio)
df_val['pred_price'] = df_val['price_baseline'] * np.exp(val_pred_log_ratio_cal)
df_test['pred_price'] = df_test['price_baseline'] * np.exp(test_pred_log_ratio_cal)

print("\n" + "="*60)
print("EVALUATION: BASELINE vs MODEL")
print("="*60)

def evaluate(name, frame):
    base_mae = mean_absolute_error(frame[TARGET_COL], frame['price_baseline'])
    base_r2 = r2_score(frame[TARGET_COL], frame['price_baseline'])
    base_w = wape(frame[TARGET_COL].values, frame['price_baseline'].values)
    
    mdl_mae = mean_absolute_error(frame[TARGET_COL], frame['pred_price'])
    mdl_r2 = r2_score(frame[TARGET_COL], frame['pred_price'])
    mdl_w = wape(frame[TARGET_COL].values, frame['pred_price'].values)
    
    # Percentage within 10% of actual
    pct_error = np.abs(frame[TARGET_COL] - frame['pred_price']) / frame[TARGET_COL]
    within_10pct = (pct_error <= 0.10).mean() * 100
    
    print(f"\n{name} SET:")
    print(f"  Baseline → MAE ${base_mae:,.0f} | R² {base_r2:.3f} | WAPE {base_w:.3f}")
    print(f"  Model    → MAE ${mdl_mae:,.0f} | R² {mdl_r2:.3f} | WAPE {mdl_w:.3f} | Within 10%: {within_10pct:.1f}%")
    print(f"  Improvement: {(base_mae - mdl_mae)/base_mae*100:.1f}% reduction in MAE")
    
    return base_mae, base_w, mdl_mae, mdl_w

train_metrics = evaluate("TRAIN", df_train)
val_metrics = evaluate("VAL", df_val)
test_metrics = evaluate("TEST", df_test)

print("\n" + "="*60)
print("ERROR ANALYSIS BY FSA")
print("="*60)

# Error by FSA
df_test['abs_pct_err'] = np.abs(df_test[TARGET_COL] - df_test['pred_price']) / df_test[TARGET_COL]
print("\nTop 10 FSAs by error rate on TEST:")
fsa_errors = df_test.groupby('FSA').agg({
    'abs_pct_err': 'mean',
    TARGET_COL: 'count'
}).rename(columns={TARGET_COL: 'count'})
fsa_errors = fsa_errors.sort_values('abs_pct_err', ascending=False).head(10)
print(fsa_errors)

# Error by month
df_test['year_month'] = df_test[DATE_COL].dt.to_period('M')
month_errors = df_test.groupby('year_month')['abs_pct_err'].mean()
print("\nError by month (last 6 months):")
print(month_errors.tail(6))

print("\n" + "="*60)
print("FEATURE IMPORTANCE (Top 20)")
print("="*60)

# Get feature importance
importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in importance.items()
]).sort_values('importance', ascending=False)

# Map feature indices to names
feature_names = list(X_train.columns)
importance_df['feature'] = importance_df['feature'].apply(
    lambda x: feature_names[int(x[1:])] if x.startswith('f') else x
)

print(importance_df.head(20).to_string(index=False))

print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print("Previous model V3:")
print(f"  Test MAE: $56,928 | R² 0.732 | WAPE 0.085")
print("\nCurrent model V4 (with all fixes):")
print(f"  Test MAE: ${test_metrics[2]:,.0f} | R² {r2_score(df_test[TARGET_COL], df_test['pred_price']):.3f} | WAPE {test_metrics[3]:.3f}")
print("="*60)

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_file = f'improved_model_v4_results_{timestamp}.txt'
with open(results_file, 'w') as f:
    f.write(f"Test MAE: ${test_metrics[2]:,.0f}\n")
    f.write(f"Test R²: {r2_score(df_test[TARGET_COL], df_test['pred_price']):.3f}\n")
    f.write(f"Test WAPE: {test_metrics[3]:.3f}\n")
    f.write(f"\nBaseline MAE: ${test_metrics[0]:,.0f}\n")
    f.write(f"Baseline WAPE: {test_metrics[1]:.3f}\n")

print(f"\nResults saved to: {results_file}")