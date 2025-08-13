#!/bin/bash

# ============================================================
# RTX 3090 Linux VM Setup Script
# For GPU-Accelerated Townhouse Price Model Search
# ============================================================

echo "======================================================"
echo "RTX 3090 GPU SETUP FOR EXHAUSTIVE MODEL SEARCH"
echo "======================================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "⚠️  Please don't run as root. Use sudo where needed."
   exit 1
fi

# 1. System Update
echo "📦 Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# 2. Check NVIDIA Driver
echo ""
echo "🎮 Step 2: Checking NVIDIA drivers..."
if nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA drivers detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "❌ NVIDIA drivers not found!"
    echo "Install with: sudo apt install nvidia-driver-535"
    exit 1
fi

# 3. Check CUDA
echo ""
echo "🔧 Step 3: Checking CUDA installation..."
if nvcc --version &> /dev/null; then
    echo "✓ CUDA detected:"
    nvcc --version | grep release
else
    echo "⚠️  CUDA not found. Installing CUDA 12.2..."
    wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
    sudo sh cuda_12.2.0_535.54.03_linux.run --silent --toolkit
    echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
fi

# 4. Python Environment
echo ""
echo "🐍 Step 4: Setting up Python environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# 5. Install Python packages
echo ""
echo "📚 Step 5: Installing GPU-accelerated ML libraries..."

# Upgrade pip
pip install --upgrade pip

# Install core packages
pip install numpy pandas scikit-learn

# Install GPU-accelerated packages
echo "Installing XGBoost (GPU)..."
pip install xgboost

echo "Installing LightGBM (GPU)..."
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON

echo "Installing CatBoost (GPU)..."
pip install catboost

# Install additional utilities
pip install tqdm matplotlib seaborn jupyter

# 6. Verify GPU acceleration
echo ""
echo "✅ Step 6: Verifying GPU acceleration..."

python3 << EOF
import sys
print("\n🔍 Checking GPU libraries...")

try:
    import xgboost as xgb
    # Test XGBoost GPU
    dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[1, 2])
    params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
    bst = xgb.train(params, dtrain, num_boost_round=1)
    print("✓ XGBoost GPU: Working")
except Exception as e:
    print(f"✗ XGBoost GPU: {str(e)[:50]}")

try:
    import lightgbm as lgb
    print("✓ LightGBM: Installed (GPU support depends on build)")
except:
    print("✗ LightGBM: Not installed")

try:
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(task_type='GPU', devices='0', iterations=1, verbose=False)
    model.fit([[1, 2]], [1])
    print("✓ CatBoost GPU: Working")
except Exception as e:
    print(f"✗ CatBoost GPU: {str(e)[:50]}")
EOF

# 7. Create run script
echo ""
echo "📝 Step 7: Creating run scripts..."

cat > run_gpu_search.sh << 'SCRIPT'
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Default parameters
DATA_FILE="townhouse_data.csv"
MIN_FEATURES=3
MAX_FEATURES=15
OUTPUT_DIR="gpu_results"
BATCH_SIZE=1000
GPU_ID=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data) DATA_FILE="$2"; shift 2 ;;
        --min) MIN_FEATURES="$2"; shift 2 ;;
        --max) MAX_FEATURES="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --batch) BATCH_SIZE="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --sold-only) SOLD_ONLY="--sold-only"; shift ;;
        --resume) RESUME="--resume"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check data file
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Data file not found: $DATA_FILE"
    echo "Please upload your CSV data file"
    exit 1
fi

echo "======================================================"
echo "GPU EXHAUSTIVE SEARCH - RTX 3090"
echo "======================================================"
echo "Data: $DATA_FILE"
echo "Features: $MIN_FEATURES to $MAX_FEATURES"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "GPU ID: $GPU_ID"
echo "======================================================"

# Run the search
python3 gpu_exhaustive_search.py \
    --data "$DATA_FILE" \
    --min-features $MIN_FEATURES \
    --max-features $MAX_FEATURES \
    --output "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --gpu-id $GPU_ID \
    $SOLD_ONLY \
    $RESUME

echo ""
echo "✅ Search complete!"
echo "Results in: $OUTPUT_DIR/"
SCRIPT

chmod +x run_gpu_search.sh

# 8. Create monitoring script
cat > monitor_gpu.sh << 'MONITOR'
#!/bin/bash

# Monitor GPU usage during training
watch -n 1 nvidia-smi
MONITOR

chmod +x monitor_gpu.sh

# 9. Create analysis script
cat > analyze_results.py << 'ANALYZER'
#!/usr/bin/env python3
"""Analyze results from GPU search"""

import pickle
import pandas as pd
import sys
import os

if len(sys.argv) > 1:
    results_dir = sys.argv[1]
else:
    results_dir = "gpu_results"

# Load results
results_file = os.path.join(results_dir, "results.pkl")
best_file = os.path.join(results_dir, "best_models.pkl")
importance_file = os.path.join(results_dir, "feature_importance.pkl")

if not os.path.exists(results_file):
    print(f"❌ No results found in {results_dir}")
    sys.exit(1)

with open(results_file, 'rb') as f:
    results = pickle.load(f)

print(f"Loaded {len(results):,} model results")

# Convert to DataFrame
df = pd.DataFrame(results).sort_values('mae')

print("\n🏆 TOP 20 MODELS")
print("="*70)
for i in range(min(20, len(df))):
    row = df.iloc[i]
    print(f"{i+1:2}. MAE: ${row['mae']:,.0f} | R²: {row['r2']:.4f} | "
          f"MAPE: {row['mape']:.2f}% | {row['model_type']} | "
          f"{row['n_features']} features")

# Load feature importance
if os.path.exists(importance_file):
    with open(importance_file, 'rb') as f:
        importance = pickle.load(f)
    
    print("\n🔍 TOP FEATURES")
    print("="*70)
    for i, (feature, count) in enumerate(importance[:30], 1):
        print(f"{i:2}. {feature:40} {count}%")

# Best model details
best = df.iloc[0]
print("\n🥇 BEST MODEL DETAILS")
print("="*70)
print(f"Type: {best['model_type']}")
print(f"MAE: ${best['mae']:,.2f}")
print(f"RMSE: ${best['rmse']:,.2f}")
print(f"R²: {best['r2']:.5f}")
print(f"MAPE: {best['mape']:.3f}%")
print(f"Features ({best['n_features']}):")
for feature in best['features']:
    print(f"  - {feature}")
ANALYZER

chmod +x analyze_results.py

# 10. Final message
echo ""
echo "======================================================"
echo "✅ SETUP COMPLETE!"
echo "======================================================"
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. Upload your data file (CSV format)"
echo "   scp your_data.csv user@vm:/path/to/GPU_3090_Deploy/"
echo ""
echo "2. Run the exhaustive search:"
echo "   ./run_gpu_search.sh --data your_data.csv"
echo ""
echo "3. Monitor GPU usage (in another terminal):"
echo "   ./monitor_gpu.sh"
echo ""
echo "4. Analyze results after completion:"
echo "   ./analyze_results.py"
echo ""
echo "OPTIONS:"
echo "  --min N        Minimum features (default: 3)"
echo "  --max N        Maximum features (default: 15)"
echo "  --batch N      Batch size (default: 1000)"
echo "  --sold-only    Use only sold properties"
echo "  --resume       Resume from previous run"
echo ""
echo "Example for 2-day run:"
echo "  nohup ./run_gpu_search.sh --data full_data.csv --max 12 &"
echo ""
echo "======================================================"