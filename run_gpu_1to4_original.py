#!/usr/bin/env python3
"""
GPU search testing 1, 2, 3, and 4 feature combinations
Shows best results for each feature count
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from itertools import combinations
import time
from datetime import datetime, timedelta
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Create a custom print function that outputs to both console and file
class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Set up dual output
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'gpu_search_results_{timestamp}.log'
dual_output = DualOutput(log_filename)
sys.stdout = dual_output

print(f"📝 Logging output to: {log_filename}")
print("="*70)
print("GPU SEARCH - TESTING 1, 2, 3, AND 4 FEATURE COMBINATIONS")
print("="*70)

# Your requested features
REQUESTED_FEATURES = [
    "Status", "Address", "S/A", "Price", "List Date", "DOM", "Tot BR", "Tot Baths",
    "TotFlArea", "Yr Blt", "Age", "MaintFee", "TypeDwel", "Bylaw Restrictions",
    "# of Kitchens", "# of Pets", "Title to Land", "Bath Ensuite # Of Pcs",
    "Baths Concatenation", "Bds In Bsmt", "Complex/Subdivision Name", 
    "Confirm Sold Date", "Cumulative DOM", "Expiry Date", "Fireplaces",
    "Floor Area - Unfinished", "Floor Area Fin - Abv Main", "Floor Area Fin - Basement",
    "Floor Area Fin - BLW Main", "Floor Area Fin - Main Flr", "Floor Area Fin - Total",
    "Full Baths", "GST Incl", "Half Baths", "Land Lease Expiry Year", "List Price",
    "Measurement Type", "Member Board Affiliation", "No. Floor Levels", "Postal Code",
    "Price Per SQFT", "Restricted Age", "Room Type Search", "Sold Date", "Sold Price",
    "Sold Price per SqFt", "SP/LP Ratio", "SP/OLP Ratio", "Storeys in Building",
    "Tot Units in Strata Plan", "Units in Development", "Zoning"
]

# Load data
print("\n📊 Loading and preparing data...")
df = pd.read_csv("/home/monstrcow/mltownhouseeval/Jan 1 2015_Aug 13 2025.csv")
print(f"✓ Loaded {len(df)} records")

# Function to convert price/monetary columns
def convert_to_numeric(series, name=''):
    """Convert monetary/numeric text columns to float"""
    if series.dtype == 'object':
        # Remove $, commas, and convert
        series = series.astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
        series = pd.to_numeric(series, errors='coerce')
    return series

# Convert all columns that should be numeric
numeric_conversions = [
    'Price', 'List Price', 'Sold Price', 'TotFlArea', 'MaintFee',
    'Floor Area - Unfinished', 'Floor Area Fin - Abv Main', 
    'Floor Area Fin - Basement', 'Floor Area Fin - BLW Main',
    'Floor Area Fin - Main Flr', 'Floor Area Fin - Total',
    'Price Per SQFT', 'Sold Price per SqFt', '# of Pets'
]

print("\n🔧 Converting text columns to numeric...")
for col in numeric_conversions:
    if col in df.columns:
        df[col] = convert_to_numeric(df[col], col)

# Use Sold Price as target
df_clean = df[df['Sold Price'] > 0].copy()
print(f"✓ {len(df_clean)} properties with valid sold prices")

# Get numeric features only (excluding targets and redundant)
numeric_features = []
for feature in REQUESTED_FEATURES:
    if feature in df_clean.columns:
        if feature in ['Price', 'List Price', 'Sold Price', 'Age']:  # Exclude targets and redundant
            continue
        elif df_clean[feature].dtype in ['int64', 'float64']:
            numeric_features.append(feature)

print(f"\n📊 Using {len(numeric_features)} numeric features:")
for i, feat in enumerate(numeric_features, 1):
    if i <= 10:  # Show first 10
        print(f"  {i:2}. {feat}")
    elif i == 11:
        print(f"  ... and {len(numeric_features) - 10} more")

# Prepare data
X = df_clean[numeric_features].fillna(0)
y = df_clean['Sold Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n✓ Train: {len(X_train)}, Test: {len(X_test)}")

# Convert to numpy
X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

# Generate combinations for 1, 2, 3, and 4 features
print("\n📊 Generating combinations (1-4 features)...")
all_combos_by_size = {}
total_tests = 0

for n in [1, 2, 3, 4]:
    combos = list(combinations(range(len(numeric_features)), n))
    all_combos_by_size[n] = combos
    total_tests += len(combos)
    print(f"  {n} feature(s): {len(combos):,} combinations")

print(f"\n✅ Total combinations: {total_tests:,}")
print(f"⏱️  Estimated time: {total_tests/50:.1f} seconds at 50 tests/sec")

# XGBoost parameters
params = {
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'max_depth': 5,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'verbosity': 0
}
num_rounds = 75

# Track results by feature count
results_by_size = {1: [], 2: [], 3: [], 4: []}
best_mae_overall = float('inf')

print("\n🚀 Starting GPU search...")
print("-"*70)
start_time = time.time()
last_update = time.time()
completed = 0

# Process each feature count separately
for n_features in [1, 2, 3, 4]:
    print(f"\n\n📊 Testing {n_features}-feature combinations...")
    combos = all_combos_by_size[n_features]
    best_mae_size = float('inf')
    
    for combo_idx, combo in enumerate(combos):
        try:
            # Select features
            indices = list(combo)
            X_train_sub = X_train_np[:, indices]
            X_test_sub = X_test_np[:, indices]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sub)
            X_test_scaled = scaler.transform(X_test_sub)
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train_np)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test_np)
            
            # Train
            model = xgb.train(params, dtrain, num_rounds, verbose_eval=False)
            
            # Predict
            y_pred = model.predict(dtest)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_np, y_pred)
            r2 = r2_score(y_test_np, y_pred)
            
            # Store result
            result = {
                'combo': combo,
                'features': [numeric_features[j] for j in combo],
                'n_features': len(combo),
                'mae': mae,
                'r2': r2
            }
            results_by_size[n_features].append(result)
            
            # Track best
            if mae < best_mae_size:
                best_mae_size = mae
            if mae < best_mae_overall:
                best_mae_overall = mae
        except:
            pass
        
        completed += 1
        
        # Progress update
        current_time = time.time()
        if current_time - last_update >= 1.0 or completed == total_tests:
            last_update = current_time
            elapsed = current_time - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_tests - completed) / rate if rate > 0 else 0
            
            print(f"\r  Progress: {combo_idx+1:,}/{len(combos):,} | "
                  f"Overall: {completed:,}/{total_tests:,} ({100*completed/total_tests:.1f}%) | "
                  f"Rate: {rate:.0f}/sec | "
                  f"Best MAE (this size): ${best_mae_size:,.0f} | "
                  f"ETA: {timedelta(seconds=int(eta))}", end='', flush=True)

# Done
print("\n\n" + "="*70)
print("✅ SEARCH COMPLETE!")
print("="*70)

elapsed = time.time() - start_time
print(f"⏱️  Total time: {elapsed:.1f} seconds")
print(f"⚡ Average rate: {total_tests/elapsed:.0f} models/sec")

# Show results for each feature count
print("\n" + "="*70)
print("RESULTS BY FEATURE COUNT")
print("="*70)

for n_features in [1, 2, 3, 4]:
    results = results_by_size[n_features]
    if not results:
        continue
        
    print(f"\n🏆 BEST {n_features}-FEATURE MODELS:")
    print("-"*70)
    
    # Sort and show top 10
    results_sorted = sorted(results, key=lambda x: x['mae'])[:10]
    for i, r in enumerate(results_sorted, 1):
        features_str = ', '.join(r['features'])
        print(f"{i:2}. MAE: ${r['mae']:,.0f} | R²: {r['r2']:.3f} | Features: {features_str}")
    
    # Summary stats
    all_maes = [r['mae'] for r in results]
    print(f"\n  Summary for {n_features} features:")
    print(f"    • Best MAE: ${min(all_maes):,.0f}")
    print(f"    • Worst MAE: ${max(all_maes):,.0f}")
    print(f"    • Average MAE: ${np.mean(all_maes):,.0f}")
    print(f"    • Total combinations tested: {len(results):,}")

# Overall feature importance
print("\n" + "="*70)
print("📊 OVERALL FEATURE IMPORTANCE")
print("="*70)

# Combine all results
all_results = []
for n in [1, 2, 3, 4]:
    all_results.extend(results_by_size[n])

# Get top 100 overall
top_100 = sorted(all_results, key=lambda x: x['mae'])[:100]
feature_counts = {}
for r in top_100:
    for f in r['features']:
        feature_counts[f] = feature_counts.get(f, 0) + 1

sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:20]
print("\nMost common features in top 100 models overall:")
for f, count in sorted_features:
    print(f"  {f:35} {count:3}% of top models")

print("\n✅ Analysis complete!")

# Close the dual output
sys.stdout = dual_output.terminal
dual_output.close()
print(f"\n📁 Results saved to: {log_filename}")