#!/usr/bin/env python3
"""
Improved Townhouse Price Prediction Model V3
With better diagnostics and feature engineering
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

# Load and clean data
print("="*60)
print("LOADING AND PREPROCESSING DATA")
print("="*60)
df = pd.read_csv('Jan 1 2015_Aug 13 2025.csv')
print(f"Initial shape: {df.shape}")

# Convert numeric columns
num_cols = ['Price', 'List Price', 'TotFlArea', 'MaintFee', 'Yr Blt', 'Age',
            'Full Baths', 'Half Baths', 'Tot BR', 'Tot Baths', 'No. Floor Levels',
            'Storeys in Building', 'Fireplaces', 'Floor Area Fin - Total',
            'Floor Area Fin - Main Flr', 'Floor Area Fin - Abv Main',
            'Floor Area Fin - Basement', 'Floor Area - Unfinished',
            'Tot Units in Strata Plan', 'Units in Development',
            '# of Kitchens', '# of Pets', 'DOM', 'Cumulative DOM']

for col in num_cols:
    if col in df.columns:
        df[col] = to_numeric(df[col])

# Parse dates
df['List Date'] = pd.to_datetime(df['List Date'], errors='coerce')
if 'Sold Date' in df.columns:
    df['Sold Date'] = pd.to_datetime(df['Sold Date'], errors='coerce')

# Keep only valid prices and dates
df = df[df['Price'].notna() & (df['Price'] > 0) & 
        df['List Date'].notna() & 
        df['TotFlArea'].notna() & (df['TotFlArea'] > 0)].copy()
print(f"After filtering: {len(df)} valid records")

# Sort by date
df = df.sort_values('List Date').reset_index(drop=True)

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Time features
ref_date = pd.Timestamp('2015-01-01')
df['month_idx'] = (df['List Date'] - ref_date).dt.days / 30.0
df['year'] = df['List Date'].dt.year.astype('float32')
df['month'] = df['List Date'].dt.month.astype('float32')
df['quarter'] = ((df['month'] - 1) // 3 + 1).astype('float32')
df['day_of_year'] = df['List Date'].dt.dayofyear.astype('float32')
df['week_of_year'] = df['List Date'].dt.isocalendar().week.astype('float32')

# Age at listing
df['age_at_list'] = np.where(
    df['Yr Blt'].notna(),
    df['year'] - df['Yr Blt'].clip(lower=1900, upper=df['year']),
    df['Age']
)
df['age_at_list'] = df['age_at_list'].clip(lower=0).fillna(20)  # Median age

# Location features
if 'Postal Code' in df.columns:
    df['FSA'] = df['Postal Code'].astype(str).str.extract(r'^([A-Za-z]\d[A-Za-z])', expand=False)
    df['FSA'] = df['FSA'].fillna('UNK')
else:
    df['FSA'] = 'UNK'

# Price per square foot
df['ppsf'] = df['Price'] / df['TotFlArea']

# Additional derived features
df['rooms_per_sqft'] = df['Tot BR'] / df['TotFlArea'] * 1000
df['baths_per_br'] = df['Tot Baths'] / df['Tot BR'].clip(lower=1)
df['maint_per_sqft'] = df['MaintFee'] / df['TotFlArea']

# Fill missing maintenance fees with median by area size
df['maint_per_sqft'] = df['maint_per_sqft'].fillna(
    df.groupby(pd.cut(df['TotFlArea'], bins=5))['maint_per_sqft'].transform('median')
)
df['maint_per_sqft'] = df['maint_per_sqft'].fillna(df['maint_per_sqft'].median())

print(f"Created {df.shape[1] - len(df.columns)} new features")

print("\n" + "="*60)
print("BUILDING COMP BASELINE")
print("="*60)

# Calculate rolling PPSF baseline
df['ppsf_baseline'] = np.nan

# By FSA with temporal component
for fsa in df['FSA'].unique():
    if pd.notna(fsa):
        mask = df['FSA'] == fsa
        if mask.sum() > 10:
            df.loc[mask, 'ppsf_fsa_90d'] = (
                df.loc[mask, 'ppsf']
                .rolling(window=90, min_periods=5, center=False)
                .median()
                .shift(1)
            )
            df.loc[mask, 'ppsf_fsa_365d'] = (
                df.loc[mask, 'ppsf']
                .rolling(window=365, min_periods=10, center=False)
                .median()
                .shift(1)
            )

# Global rolling median
df['ppsf_global_90d'] = df['ppsf'].rolling(window=90, min_periods=20).median().shift(1)
df['ppsf_global_365d'] = df['ppsf'].rolling(window=365, min_periods=50).median().shift(1)

# Combine baselines (prioritize local over global)
df['ppsf_baseline'] = (df['ppsf_fsa_90d']
                       .fillna(df['ppsf_fsa_365d'])
                       .fillna(df['ppsf_global_90d'])
                       .fillna(df['ppsf_global_365d'])
                       .fillna(df['ppsf'].median()))

# Calculate baseline price
df['price_baseline'] = df['ppsf_baseline'] * df['TotFlArea']

# Calculate residual (what we'll actually predict)
df['price_ratio'] = df['Price'] / df['price_baseline']
df['log_price_ratio'] = np.log(df['price_ratio'])

print(f"Baseline PPSF range: ${df['ppsf_baseline'].min():.0f} - ${df['ppsf_baseline'].max():.0f}")
print(f"Price ratio stats: mean={df['price_ratio'].mean():.3f}, std={df['price_ratio'].std():.3f}")

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

print(f"Train: {len(df_train)} samples ({df_train['List Date'].min().date()} to {df_train['List Date'].max().date()})")
print(f"Val: {len(df_val)} samples ({df_val['List Date'].min().date()} to {df_val['List Date'].max().date()})")
print(f"Test: {len(df_test)} samples ({df_test['List Date'].min().date()} to {df_test['List Date'].max().date()})")

print("\n" + "="*60)
print("FEATURE PREPARATION")
print("="*60)

# Numeric features
num_features = ['TotFlArea', 'MaintFee', 'age_at_list', 'Full Baths', 'Half Baths',
               'Tot BR', 'Tot Baths', 'No. Floor Levels', 'Storeys in Building',
               'Fireplaces', 'Floor Area Fin - Total', 'Floor Area Fin - Main Flr',
               'Floor Area Fin - Abv Main', 'Floor Area Fin - Basement',
               'Floor Area - Unfinished', 'Tot Units in Strata Plan',
               'Units in Development', 'month_idx', 'year', 'month', 'quarter',
               'day_of_year', 'week_of_year', 'rooms_per_sqft', 'baths_per_br',
               'maint_per_sqft', 'ppsf_baseline', 'DOM', 'Cumulative DOM']

# Keep only existing columns
num_features = [f for f in num_features if f in df.columns]
print(f"Numeric features: {len(num_features)}")

# Categorical features
cat_features = ['TypeDwel', 'FSA', 'Restricted Age', 'Title to Land', 'Zoning']
cat_features = [f for f in cat_features if f in df.columns]
print(f"Categorical features: {len(cat_features)}")

# Build feature matrices
X_train_list = [df_train[num_features].fillna(0)]
X_val_list = [df_val[num_features].fillna(0)]
X_test_list = [df_test[num_features].fillna(0)]

# One-hot encoding for high-frequency categorical values
for cat in cat_features:
    # Get value counts in training data
    value_counts = df_train[cat].value_counts()
    # Keep values that appear at least 30 times
    frequent_values = value_counts[value_counts >= 30].index
    
    print(f"  {cat}: encoding {len(frequent_values)} values")
    
    for val in frequent_values:
        col_name = f'{cat}_{str(val)[:20]}'  # Truncate long names
        X_train_list.append(pd.DataFrame({col_name: (df_train[cat] == val).astype(float)}))
        X_val_list.append(pd.DataFrame({col_name: (df_val[cat] == val).astype(float)}))
        X_test_list.append(pd.DataFrame({col_name: (df_test[cat] == val).astype(float)}))

# Combine all features
X_train = pd.concat(X_train_list, axis=1)
X_val = pd.concat(X_val_list, axis=1)
X_test = pd.concat(X_test_list, axis=1)

print(f"\nFinal feature count: {X_train.shape[1]}")

# Target variable (log ratio to baseline)
y_train = df_train['log_price_ratio'].values
y_val = df_val['log_price_ratio'].values
y_test = df_test['log_price_ratio'].values

print(f"Target stats - Train: mean={y_train.mean():.3f}, std={y_train.std():.3f}")

# Sample weights
def calc_recency_weights(dates, half_life_days=365):
    """Longer half-life for more stable training"""
    max_date = dates.max()
    days_old = (max_date - dates).dt.days.clip(lower=0)
    return np.power(0.5, days_old / half_life_days)

recency_weights = calc_recency_weights(df_train['List Date'])
price_weights = 1.0 / np.log(df_train['Price'].values + 1)  # Log scale for price weights
sample_weights = recency_weights * price_weights
sample_weights = sample_weights / sample_weights.mean()  # Normalize

print(f"Sample weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

print("\n" + "="*60)
print("TRAINING XGBOOST")
print("="*60)

# Create DMatrix
dtrain = xgb.DMatrix(X_train.values, label=y_train, weight=sample_weights)
dval = xgb.DMatrix(X_val.values, label=y_val)
dtest = xgb.DMatrix(X_test.values)

# XGBoost parameters
params = {
    'tree_method': 'hist',
    'max_depth': 8,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'learning_rate': 0.03,
    'reg_lambda': 2.0,
    'reg_alpha': 1.0,
    'gamma': 0.1,
    'objective': 'reg:squarederror',
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
print(f"Best validation score: {model.best_score:.4f}")

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

# Predictions
train_pred_log_ratio = model.predict(dtrain, iteration_range=(0, model.best_iteration + 1))
val_pred_log_ratio = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
test_pred_log_ratio = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

# Convert back to prices
df_train['pred_price'] = df_train['price_baseline'] * np.exp(train_pred_log_ratio)
df_val['pred_price'] = df_val['price_baseline'] * np.exp(val_pred_log_ratio)
df_test['pred_price'] = df_test['price_baseline'] * np.exp(test_pred_log_ratio)

# Calculate metrics
def evaluate_set(df, name):
    mae = mean_absolute_error(df['Price'], df['pred_price'])
    r2 = r2_score(df['Price'], df['pred_price'])
    wape_val = wape(df['Price'].values, df['pred_price'].values)
    
    # Percentage within 10% of actual
    pct_error = np.abs(df['Price'] - df['pred_price']) / df['Price']
    within_10pct = (pct_error <= 0.10).mean() * 100
    
    print(f"\n{name} SET:")
    print(f"  MAE: ${mae:,.0f}")
    print(f"  R²: {r2:.3f}")
    print(f"  WAPE: {wape_val:.3f}")
    print(f"  Within 10%: {within_10pct:.1f}%")
    
    return mae, r2, wape_val

print("="*60)
train_mae, train_r2, train_wape = evaluate_set(df_train, "TRAINING")
val_mae, val_r2, val_wape = evaluate_set(df_val, "VALIDATION")
test_mae, test_r2, test_wape = evaluate_set(df_test, "TEST")

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
print("COMPARISON TO BASELINE")
print("="*60)
print("Previous model (1-4 features):")
print("  MAE: ~$76,000")
print("  R²: ~0.45")
print("  WAPE: ~0.118")
print("\nImproved model (all features):")
print(f"  MAE: ${test_mae:,.0f}")
print(f"  R²: {test_r2:.3f}")
print(f"  WAPE: {test_wape:.3f}")
print(f"\nImprovement: {(76000 - test_mae)/76000*100:.1f}% reduction in MAE")
print("="*60)

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_file = f'improved_model_results_{timestamp}.txt'
with open(results_file, 'w') as f:
    f.write(f"Test MAE: ${test_mae:,.0f}\n")
    f.write(f"Test R²: {test_r2:.3f}\n")
    f.write(f"Test WAPE: {test_wape:.3f}\n")

print(f"\nResults saved to: {results_file}")