#!/bin/bash
# Setup script for Linux VM (10.0.0.75) with RTX 3090
# Run this on the Linux VM to prepare it as Ray head node

set -e

echo "=========================================="
echo "LINUX VM RAY HEAD NODE SETUP"
echo "=========================================="

# Check CUDA availability
echo -e "\n1. Checking CUDA/GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "✓ GPU detected"

# Setup Python environment
echo -e "\n2. Setting up Python environment..."
if [ ! -d "$HOME/raygpu" ]; then
    python3 -m venv $HOME/raygpu
    echo "✓ Created virtual environment at ~/raygpu"
else
    echo "✓ Virtual environment already exists at ~/raygpu"
fi

source $HOME/raygpu/bin/activate

# Install packages
echo -e "\n3. Installing Python packages..."
pip install --upgrade pip
pip install ray xgboost pandas numpy scikit-learn pyarrow

# Test XGBoost GPU support
echo -e "\n4. Testing XGBoost GPU support..."
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
    exit(1)
EOF

# Configure firewall
echo -e "\n5. Configuring firewall..."
if command -v ufw &> /dev/null; then
    echo "Opening required ports..."
    sudo ufw allow 6379/tcp comment 'Ray GCS'
    sudo ufw allow 10001/tcp comment 'Ray Client'
    sudo ufw allow 8265/tcp comment 'Ray Dashboard'
    echo "✓ Firewall configured"
else
    echo "⚠ ufw not found. Please manually open ports 6379, 10001, 8265"
fi

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

# Create Ray start script
echo -e "\n7. Creating Ray start script..."
cat > $HOME/start_ray_head.sh <<'EOF'
#!/bin/bash
source $HOME/raygpu/bin/activate

# Stop any existing Ray instance
ray stop

# Start Ray head node
ray start --head \
  --port=6379 \
  --ray-client-server-port=10001 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-gpus=1

echo ""
echo "Ray head node started!"
echo "Dashboard: http://10.0.0.75:8265"
echo "Client address: ray://10.0.0.75:10001"
EOF

chmod +x $HOME/start_ray_head.sh
echo "✓ Created ~/start_ray_head.sh"

# Create Ray stop script
cat > $HOME/stop_ray.sh <<'EOF'
#!/bin/bash
source $HOME/raygpu/bin/activate
ray stop
echo "Ray stopped"
EOF

chmod +x $HOME/stop_ray.sh
echo "✓ Created ~/stop_ray.sh"

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy sales_2015_2025.csv to ~/mltownhouseeval/"
echo "2. Run: ~/start_ray_head.sh"
echo "3. Check dashboard at http://10.0.0.75:8265"
echo ""
echo "To stop Ray: ~/stop_ray.sh"