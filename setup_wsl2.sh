#!/bin/bash
# Setup script for Windows WSL2 (192.168.0.233) with RTX 4090
# Run this inside WSL2 Ubuntu to prepare it as Ray worker node

set -e

echo "=========================================="
echo "WSL2 RAY WORKER NODE SETUP"
echo "=========================================="

# Check CUDA availability
echo -e "\n1. Checking CUDA/GPU via WSL2..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found in WSL2."
    echo "   Please ensure:"
    echo "   1. Latest NVIDIA Windows driver is installed"
    echo "   2. WSL2 is updated (wsl --update)"
    echo "   3. CUDA toolkit is installed in WSL2"
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "✓ GPU detected via WSL2"

# Install system dependencies
echo -e "\n2. Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential python3-venv python3-pip

# Setup Python environment
echo -e "\n3. Setting up Python environment..."
if [ ! -d "$HOME/raygpu" ]; then
    python3 -m venv $HOME/raygpu
    echo "✓ Created virtual environment at ~/raygpu"
else
    echo "✓ Virtual environment already exists at ~/raygpu"
fi

source $HOME/raygpu/bin/activate

# Install packages
echo -e "\n4. Installing Python packages..."
pip install --upgrade pip
pip install ray xgboost pandas numpy scikit-learn pyarrow

# Test XGBoost GPU support
echo -e "\n5. Testing XGBoost GPU support..."
python3 - <<'EOF'
import xgboost as xgb
import numpy as np

try:
    X = np.random.rand(1000, 20).astype('float32')
    y = (X @ np.arange(20)).astype('float32')
    d = xgb.DMatrix(X, label=y)
    params = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'objective': 'reg:squarederror',
        'verbosity': 0
    }
    model = xgb.train(params, d, 20)
    print("✓ XGBoost GPU support verified")
except Exception as e:
    print(f"❌ XGBoost GPU test failed: {e}")
    print("   Ensure CUDA toolkit is installed in WSL2")
    exit(1)
EOF

# Create data directory
echo -e "\n6. Setting up data directory..."
DATA_DIR="$HOME/mltownhouseeval"
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p $DATA_DIR
    echo "✓ Created data directory at $DATA_DIR"
else
    echo "✓ Data directory exists at $DATA_DIR"
fi

echo "⚠ Please copy sales_2015_2025.csv to $DATA_DIR/"
echo "  IMPORTANT: Keep file in WSL2 filesystem (not /mnt/c) for best performance"

# Test network connectivity to head node
echo -e "\n7. Testing network connectivity to head node..."
HEAD_IP="10.0.0.75"
if ping -c 1 $HEAD_IP &> /dev/null; then
    echo "✓ Can reach head node at $HEAD_IP"
else
    echo "⚠ Cannot reach $HEAD_IP - check network configuration"
fi

# Create Ray worker start script
echo -e "\n8. Creating Ray worker start script..."
cat > $HOME/start_ray_worker.sh <<'EOF'
#!/bin/bash
source $HOME/raygpu/bin/activate

# Stop any existing Ray instance
ray stop

# Start Ray worker node
echo "Connecting to Ray head at 10.0.0.75:6379..."
ray start --address='10.0.0.75:6379' --num-gpus=1

echo ""
echo "Ray worker node started!"
echo "Check head node dashboard at http://10.0.0.75:8265"
EOF

chmod +x $HOME/start_ray_worker.sh
echo "✓ Created ~/start_ray_worker.sh"

# Create Ray stop script
cat > $HOME/stop_ray.sh <<'EOF'
#!/bin/bash
source $HOME/raygpu/bin/activate
ray stop
echo "Ray stopped"
EOF

chmod +x $HOME/stop_ray.sh
echo "✓ Created ~/stop_ray.sh"

# Performance tips for WSL2
echo -e "\n9. WSL2 Performance Configuration..."
cat > $HOME/.wslconfig_recommended <<'EOF'
# Recommended .wslconfig for Windows user profile
# Copy this to C:\Users\YourUsername\.wslconfig

[wsl2]
memory=32GB  # Adjust based on your system RAM
processors=8  # Adjust based on your CPU cores
localhostForwarding=true
nestedVirtualization=true

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
EOF

echo "✓ Created ~/.wslconfig_recommended"
echo "  Copy to Windows user profile for better performance"

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy sales_2015_2025.csv to ~/mltownhouseeval/"
echo "   (Keep in WSL2 filesystem, not /mnt/c/)"
echo "2. Ensure Linux VM head node is running"
echo "3. Run: ~/start_ray_worker.sh"
echo "4. Check dashboard at http://10.0.0.75:8265"
echo "   You should see 2 nodes with 2 GPUs total"
echo ""
echo "To stop Ray: ~/stop_ray.sh"