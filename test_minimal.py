#!/usr/bin/env python3
"""
Minimal test of the no-list-price model
Quick verification that everything works
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("MINIMAL TEST - NO LIST PRICE MODEL")
print("="*50)

# Load data
csv_path = "/home/monstrcow/mltownhouseeval/Jan 1 2015_Aug 13 2025.csv"
df = pd.read_csv(csv_path)
print(f"✓ Loaded {len(df):,} records")

# Quick numeric conversion for key columns
def to_numeric(series):
    if pd.api.types.is_object_dtype(series):
        s = (series.astype(str)
             .str.replace('$','',regex=False)
             .str.replace(',','',regex=False)
             .str.replace(' ','',regex=False))
        return pd.to_numeric(s, errors='coerce')
    return series

for col in ['Sold Price', 'TotFlArea', 'Tot BR', 'Tot Baths', 'Age', 'MaintFee']:
    if col in df.columns:
        df[col] = to_numeric(df[col])

# Filter valid sales
df = df[df['Sold Price'].notna() & (df['Sold Price'] > 0)].copy()
print(f"✓ {len(df):,} valid sales")

# Simple features (NO LIST PRICE)
features = ['TotFlArea', 'Tot BR', 'Tot Baths', 'Age', 'MaintFee']
df = df.dropna(subset=features + ['Sold Price'])
print(f"✓ {len(df):,} rows with complete features")

# Quick train/test split (80/20)
split_idx = int(0.8 * len(df))
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

X_train = train[features].values.astype('float32')
X_test = test[features].values.astype('float32')
y_train = np.log(train['Sold Price'].values)  # Log transform target
y_test = np.log(test['Sold Price'].values)

print(f"\n📊 Training with {len(features)} features (NO List Price):")
for f in features:
    print(f"  - {f}")

# Simple XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'learning_rate': 0.1,
    'verbosity': 0
}

print(f"\n🚀 Training model...")
model = xgb.train(params, dtrain, num_boost_round=100, 
                  evals=[(dtest, 'test')], verbose_eval=False)

# Predictions
log_pred = model.predict(dtest)
pred = np.exp(log_pred)
actual = test['Sold Price'].values

# Metrics
mae = mean_absolute_error(actual, pred)
mape = np.mean(np.abs(actual - pred) / actual) * 100
r2 = r2_score(actual, pred)

print(f"\n📈 RESULTS (Without List Price):")
print(f"  MAE:  ${mae:,.0f}")
print(f"  MAPE: {mape:.1f}%")
print(f"  R²:   {r2:.3f}")

# Compare to naive baseline (median price)
baseline = np.full_like(actual, train['Sold Price'].median())
baseline_mae = mean_absolute_error(actual, baseline)
print(f"\n📊 Baseline (median): ${baseline_mae:,.0f}")
print(f"  Improvement: ${baseline_mae - mae:,.0f} ({(baseline_mae - mae)/baseline_mae*100:.1f}%)")

print(f"\n✅ Test complete!")