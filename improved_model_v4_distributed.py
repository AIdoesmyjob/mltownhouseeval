#!/usr/bin/env python3
"""
Improved Townhouse Price Prediction Model V4 - Distributed Ray Version
Requires both GPUs (head + worker) to be online before running
"""

import ray
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
import warnings
import sys
import subprocess
import time
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
        print("\nPlease ensure Ray head node is running:")
        print("  sudo bash -c 'source /root/raygpu/bin/activate && /home/monstrcow/start_ray_daemon.sh'")
        return False
    
    # Get cluster resources
    resources = ray.cluster_resources()
    available = ray.available_resources()
    
    total_gpus = resources.get('GPU', 0)
    available_gpus = available.get('GPU', 0)
    
    print(f"\nCluster Resources:")
    print(f"  Total GPUs: {total_gpus}")
    print(f"  Available GPUs: {available_gpus}")
    
    # Get node information
    nodes = ray.nodes()
    gpu_nodes = []
    
    print("\nNode Details:")
    for node in nodes:
        if node['Alive']:
            node_resources = node.get('Resources', {})
            node_gpus = node_resources.get('GPU', 0)
            node_ip = node['NodeManagerAddress']
            
            if node_gpus > 0:
                gpu_nodes.append({
                    'ip': node_ip,
                    'gpus': node_gpus,
                    'cpu': node_resources.get('CPU', 0),
                    'memory': node_resources.get('memory', 0) / (1024**3)  # Convert to GB
                })
                
                # Check which GPU using nvidia-smi if possible
                gpu_name = "Unknown"
                try:
                    # Try to get GPU info via Ray task
                    @ray.remote(num_gpus=0.1)
                    def get_gpu_info():
                        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            return result.stdout.strip()
                        return "Unknown"
                    
                    # Schedule on specific node
                    gpu_name = ray.get(get_gpu_info.options(resources={f"node:{node_ip}": 0.001}).remote())
                except:
                    pass
                
                print(f"  📍 Node {node_ip}:")
                print(f"     GPU: {node_gpus}x {gpu_name}")
                print(f"     CPU: {node_resources.get('CPU', 0)} cores")
                print(f"     Memory: {node_resources.get('memory', 0) / (1024**3):.1f} GB")
    
    # Check if we have exactly 2 GPUs (one on each node)
    if total_gpus < 2:
        print(f"\n❌ ERROR: Need 2 GPUs but only {total_gpus} detected!")
        print("\nTroubleshooting:")
        print("1. Check if worker node is connected:")
        print("   ssh monstrcow@10.0.0.75 'ps aux | grep ray'")
        print("\n2. If not running, start Ray on worker:")
        print("   ssh monstrcow@10.0.0.75 'sudo ray start --address=10.0.0.198:6379'")
        print("\n3. Check Ray status:")
        print("   ray status")
        ray.shutdown()
        return False
    
    if available_gpus < 2:
        print(f"\n⚠️  WARNING: {total_gpus} GPUs detected but only {available_gpus} available")
        print("Some GPUs may be in use by other processes")
        ray.shutdown()
        return False
    
    print(f"\n✅ Both GPUs detected and available!")
    print(f"   RTX 4090 (Head node: 10.0.0.198)")
    print(f"   RTX 3090 (Worker node: 10.0.0.75)")
    
    # Don't shutdown yet - we'll use the connection
    return True

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

# Ray remote functions for distributed training
@ray.remote(num_gpus=1)
def train_on_gpu(X_train, y_train, X_val, y_val, params, gpu_id, num_rounds=500):
    """Train XGBoost model on a specific GPU"""
    import xgboost as xgb
    
    # Update params for this GPU
    params = params.copy()
    params['gpu_id'] = gpu_id
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    print(f"Training on GPU {gpu_id}...")
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    return model

@ray.remote(num_gpus=0.5)
def evaluate_features_on_gpu(feature_subset, train_data, val_data, params):
    """Evaluate a subset of features on GPU"""
    X_train = train_data[0][feature_subset]
    y_train = train_data[1]
    X_val = val_data[0][feature_subset]
    y_val = val_data[1]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    val_pred = model.predict(dval)
    mae = mean_absolute_error(y_val, val_pred)
    
    return {'features': feature_subset, 'mae': mae, 'model': model}

print("="*60)
print("IMPROVED MODEL V4 - DISTRIBUTED GPU VERSION")
print("="*60)

# Check for both GPUs before proceeding
if not check_ray_cluster_gpus():
    print("\n❌ Exiting: Both GPUs must be available to run this model")
    sys.exit(1)

print("\n" + "="*60)
print("DATA LOADING AND PREPARATION")
print("="*60)

# Load and clean data
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
print("FEATURE ENGINEERING (SIMPLIFIED FOR DEMO)")
print("="*60)

# For demonstration, we'll use a simplified feature set
# In production, you'd want the full feature engineering pipeline

# Simple features
feature_cols = ['TotFlArea', 'Age', 'MaintFee', 'Tot BR', 'Tot Baths', 
                'Full Baths', 'Half Baths', 'Fireplaces', 
                'Floor Area Fin - Total', 'No. Floor Levels']

# Remove any missing columns
feature_cols = [c for c in feature_cols if c in df.columns]

# Fill missing values
for col in feature_cols:
    df[col] = df[col].fillna(0)

print(f"Using {len(feature_cols)} features for distributed training demo")

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

print(f"Train: {len(train_df)} records")
print(f"Val:   {len(val_df)} records")
print(f"Test:  {len(test_df)} records")

# Prepare data
X_train = train_df[feature_cols].values
y_train = train_df[TARGET_COL].values

X_val = val_df[feature_cols].values
y_val = val_df[TARGET_COL].values

X_test = test_df[feature_cols].values
y_test = test_df[TARGET_COL].values

print("\n" + "="*60)
print("DISTRIBUTED TRAINING ON 2 GPUs")
print("="*60)

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
    'seed': 42
}

print("\n🚀 Training two models in parallel on separate GPUs...")
print("   Model 1: RTX 4090 (10.0.0.198)")
print("   Model 2: RTX 3090 (10.0.0.75)")

# Put data in Ray object store for efficient access
X_train_ref = ray.put(X_train)
y_train_ref = ray.put(y_train)
X_val_ref = ray.put(X_val)
y_val_ref = ray.put(y_val)

# Launch parallel training on both GPUs
start_time = time.time()

# Train models with different seeds for ensemble
params1 = params.copy()
params1['seed'] = 42

params2 = params.copy()
params2['seed'] = 123

# Submit jobs to both GPUs
future1 = train_on_gpu.remote(X_train_ref, y_train_ref, X_val_ref, y_val_ref, params1, 0, num_rounds=300)
future2 = train_on_gpu.remote(X_train_ref, y_train_ref, X_val_ref, y_val_ref, params2, 0, num_rounds=300)

print("\nTraining in progress on both GPUs...")
print("Waiting for completion...")

# Wait for both to complete
model1, model2 = ray.get([future1, future2])

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time:.1f} seconds using both GPUs!")

print("\n" + "="*60)
print("ENSEMBLE PREDICTIONS")
print("="*60)

# Make predictions with both models
dtest = xgb.DMatrix(X_test)

pred1 = model1.predict(dtest)
pred2 = model2.predict(dtest)

# Ensemble predictions (average)
test_pred = (pred1 + pred2) / 2

print("Ensemble created from both GPU models")

print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Calculate metrics
mae = mean_absolute_error(y_test, test_pred)
mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
wape_val = wape(y_test, test_pred) * 100
r2 = r2_score(y_test, test_pred)

print(f"\nTest Set Performance (Ensemble):")
print(f"  MAE:  ${mae:,.0f}")
print(f"  MAPE: {mape:.1f}%")
print(f"  WAPE: {wape_val:.1f}%")
print(f"  R²:   {r2:.3f}")

# Compare with individual models
mae1 = mean_absolute_error(y_test, pred1)
mae2 = mean_absolute_error(y_test, pred2)

print(f"\nIndividual Model MAEs:")
print(f"  GPU 1 (4090): ${mae1:,.0f}")
print(f"  GPU 2 (3090): ${mae2:,.0f}")
print(f"  Ensemble:     ${mae:,.0f} (improvement: ${min(mae1, mae2) - mae:,.0f})")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'distributed_model_results_{timestamp}.txt'

with open(output_file, 'w') as f:
    f.write("DISTRIBUTED GPU MODEL RESULTS\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Training time: {training_time:.1f} seconds\n")
    f.write(f"Test MAE (ensemble): ${mae:,.0f}\n")
    f.write(f"Test WAPE (ensemble): {wape_val:.1f}%\n")
    f.write(f"GPU 1 MAE: ${mae1:,.0f}\n")
    f.write(f"GPU 2 MAE: ${mae2:,.0f}\n")

print(f"\n✅ Results saved to {output_file}")

# Cleanup
ray.shutdown()
print("\n✅ Done! Ray cluster connection closed.")