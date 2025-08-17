#!/usr/bin/env python3
"""
Improved Townhouse Price Prediction Model V2
Simplified version with better error handling
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
print("Loading data...")
df = pd.read_csv('Jan 1 2015_Aug 13 2025.csv')

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

# Keep only valid prices and dates
df = df[df['Price'].notna() & (df['Price'] > 0) & df['List Date'].notna()].copy()
print(f"Loaded {len(df)} valid sales records")

# Sort by date
df = df.sort_values('List Date').reset_index(drop=True)

# Add time features
print("Adding time features...")
ref_date = pd.Timestamp('2015-01-01')
df['month_idx'] = (df['List Date'] - ref_date).dt.days / 30.0
df['year'] = df['List Date'].dt.year.astype('float32')
df['month'] = df['List Date'].dt.month.astype('float32')
df['quarter'] = ((df['month'] - 1) // 3 + 1).astype('float32')

# Age at listing
df['age_at_list'] = np.where(
    df['Yr Blt'].notna(),
    df['year'] - df['Yr Blt'].clip(lower=1900, upper=df['year']),
    df['Age']
)
df['age_at_list'] = df['age_at_list'].clip(lower=0).fillna(df['Age'].median())

# Location features
print("Adding location features...")
if 'Postal Code' in df.columns:
    df['FSA'] = df['Postal Code'].astype(str).str.extract(r'^([A-Za-z]\d[A-Za-z])', expand=False)
else:
    df['FSA'] = 'UNK'

# Build comp baseline
print("Building comp-style baseline...")
area = df['TotFlArea'].replace(0, np.nan)
df['sold_ppsf'] = df['Price'] / area

# Simple rolling median PPSF by FSA
print("Calculating rolling PPSF medians...")
df['ppsf_baseline'] = np.nan

# Group by FSA for rolling median
for fsa in df['FSA'].unique():
    if pd.notna(fsa):
        mask = df['FSA'] == fsa
        df.loc[mask, 'ppsf_90d'] = (
            df.loc[mask, 'sold_ppsf']
            .rolling(window=30, min_periods=5, center=False)
            .median()
            .shift(1)
        )

# Global fallback
df['ppsf_global'] = df['sold_ppsf'].rolling(window=90, min_periods=10).median().shift(1)

# Use FSA-specific if available, else global
df['ppsf_baseline'] = df['ppsf_90d'].fillna(df['ppsf_global'])

# Fill remaining with overall median
df['ppsf_baseline'] = df['ppsf_baseline'].fillna(df['sold_ppsf'].median())

# Calculate baseline price and residual
df['price_baseline'] = df['ppsf_baseline'] * area

# Filter to valid baseline prices
mask = (df['price_baseline'].notna() & 
        (df['price_baseline'] > 0) & 
        area.notna())
df = df[mask].copy()
print(f"After baseline filter: {len(df)} records")

# Log residual target
df['y_log_resid'] = np.log(df['Price']) - np.log(df['price_baseline'])

# Time-based split
print("Creating temporal splits...")
n = len(df)
train_end = int(0.7 * n)
val_end = int(0.8 * n)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f"Train: {len(df_train)} samples")
print(f"Val: {len(df_val)} samples")
print(f"Test: {len(df_test)} samples")

# Prepare features
print("Preparing features...")
num_features = ['TotFlArea', 'MaintFee', 'age_at_list', 'Full Baths', 'Half Baths',
               'Tot BR', 'Tot Baths', 'No. Floor Levels', 'Storeys in Building',
               'Fireplaces', 'Floor Area Fin - Total', 'Floor Area Fin - Main Flr',
               'Floor Area Fin - Abv Main', 'Floor Area Fin - Basement',
               'Floor Area - Unfinished', 'Tot Units in Strata Plan',
               'Units in Development', 'month_idx', 'year', 'month', 'quarter',
               'ppsf_baseline']

# Keep only existing columns
num_features = [f for f in num_features if f in df.columns]

# Categorical features
cat_features = ['TypeDwel', 'FSA', 'Restricted Age', 'Title to Land']
cat_features = [f for f in cat_features if f in df.columns]

# Prepare training features
X_train_list = [df_train[num_features].fillna(0)]
X_val_list = [df_val[num_features].fillna(0)]
X_test_list = [df_test[num_features].fillna(0)]

# One-hot encoding for categoricals
for cat in cat_features:
    # Get top 20 most common values
    top_values = df_train[cat].value_counts().head(20).index
    
    for val in top_values:
        col_name = f'{cat}_{val}'
        X_train_list.append(pd.DataFrame({col_name: (df_train[cat] == val).astype(float)}))
        X_val_list.append(pd.DataFrame({col_name: (df_val[cat] == val).astype(float)}))
        X_test_list.append(pd.DataFrame({col_name: (df_test[cat] == val).astype(float)}))

# Combine features
X_train = pd.concat(X_train_list, axis=1)
X_val = pd.concat(X_val_list, axis=1)
X_test = pd.concat(X_test_list, axis=1)

print(f"Feature count: {X_train.shape[1]}")

# Targets
y_train = df_train['y_log_resid'].values
y_val = df_val['y_log_resid'].values
y_test = df_test['y_log_resid'].values

# Sample weights
print("Adding sample weights...")
def calc_recency_weights(dates, half_life_days=180):
    max_date = dates.max()
    days_old = (max_date - dates).dt.days
    return np.power(0.5, days_old / half_life_days)

recency_weights = calc_recency_weights(df_train['List Date'])
wape_weights = 1.0 / np.maximum(df_train['Price'].values, 1000.0)
sample_weights = recency_weights * wape_weights

# Train XGBoost
print("Training XGBoost model...")
dtrain = xgb.DMatrix(X_train.values, label=y_train, weight=sample_weights)
dval = xgb.DMatrix(X_val.values, label=y_val)
dtest = xgb.DMatrix(X_test.values)

params = {
    'tree_method': 'hist',  # Use CPU for now
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'reg_lambda': 1.0,
    'reg_alpha': 0.5,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'seed': 42
}

print("Training with early stopping...")
model = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=[(dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=100
)

# Evaluate
print("\nEvaluating model...")

# Predict
val_pred_resid = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
test_pred_resid = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

# Convert back to prices
df_val['pred_price'] = df_val['price_baseline'] * np.exp(val_pred_resid)
df_test['pred_price'] = df_test['price_baseline'] * np.exp(test_pred_resid)

# Metrics
val_mae = mean_absolute_error(df_val['Price'], df_val['pred_price'])
val_r2 = r2_score(df_val['Price'], df_val['pred_price'])
val_wape = wape(df_val['Price'].values, df_val['pred_price'].values)

test_mae = mean_absolute_error(df_test['Price'], df_test['pred_price'])
test_r2 = r2_score(df_test['Price'], df_test['pred_price'])
test_wape = wape(df_test['Price'].values, df_test['pred_price'].values)

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"\nVALIDATION SET (temporal middle 10%):")
print(f"  MAE: ${val_mae:,.0f}")
print(f"  R²: {val_r2:.3f}")
print(f"  WAPE: {val_wape:.3f}")

print(f"\nTEST SET (newest 20%):")
print(f"  MAE: ${test_mae:,.0f}")
print(f"  R²: {test_r2:.3f}")
print(f"  WAPE: {test_wape:.3f}")

print("\n" + "="*60)
print("COMPARISON TO BASELINE (1-4 features):")
print("="*60)
print("Baseline: MAE ~$76k, R² ~0.45, WAPE ~0.118")
print(f"Improved: MAE ${test_mae:,.0f}, R² {test_r2:.3f}, WAPE {test_wape:.3f}")
print(f"Improvement: {(76000 - test_mae)/76000*100:.1f}% reduction in MAE")
print("="*60)