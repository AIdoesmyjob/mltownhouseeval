#!/bin/bash
# Setup script for MacBook as Ray client/orchestrator
# Run this on your Mac to prepare it to run the orchestrator

set -e

echo "=========================================="
echo "MAC RAY CLIENT SETUP"
echo "=========================================="

# Check Python version
echo -e "\n1. Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python $PYTHON_VERSION detected"

# Setup Python environment
echo -e "\n2. Setting up Python environment..."
VENV_PATH="$HOME/rayclient"

if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv $VENV_PATH
    echo "✓ Created virtual environment at ~/rayclient"
else
    echo "✓ Virtual environment already exists at ~/rayclient"
fi

source $VENV_PATH/bin/activate

# Install packages
echo -e "\n3. Installing Python packages..."
pip install --upgrade pip
pip install ray xgboost pandas numpy scikit-learn pyarrow

# Test Ray client
echo -e "\n4. Testing Ray client installation..."
python3 -c "import ray; print('✓ Ray client ready')"

# Test network connectivity
echo -e "\n5. Testing network connectivity..."
HEAD_IP="10.0.0.75"
if ping -c 1 $HEAD_IP &> /dev/null 2>&1; then
    echo "✓ Can reach Linux VM at $HEAD_IP"
else
    echo "⚠ Cannot reach $HEAD_IP - will test when running orchestrator"
fi

# Create run script
echo -e "\n6. Creating run script..."
cat > run_distributed.sh <<'EOF'
#!/bin/bash
# Run the distributed training orchestrator

source $HOME/rayclient/bin/activate

echo "Starting distributed training orchestrator..."
echo "Connecting to Ray cluster at 10.0.0.75:10001"
echo ""

python orchestrator.py

echo ""
echo "Training complete!"
EOF

chmod +x run_distributed.sh
echo "✓ Created run_distributed.sh"

# Create monitoring script
cat > monitor_cluster.sh <<'EOF'
#!/bin/bash
# Monitor the Ray cluster status

source $HOME/rayclient/bin/activate

python3 - <<'PY'
import ray
import time
import sys

try:
    print("Connecting to Ray cluster...")
    ray.init(address="ray://10.0.0.75:10001")
    
    nodes = ray.nodes()
    print(f"\n✓ Connected! Found {len(nodes)} nodes")
    
    total_gpus = sum(n["Resources"].get("GPU", 0) for n in nodes)
    total_cpus = sum(n["Resources"].get("CPU", 0) for n in nodes)
    
    print(f"\nCluster Resources:")
    print(f"  Total GPUs: {total_gpus}")
    print(f"  Total CPUs: {total_cpus}")
    
    print(f"\nNode Details:")
    for i, node in enumerate(nodes, 1):
        ip = node["NodeManagerAddress"]
        gpus = node["Resources"].get("GPU", 0)
        cpus = node["Resources"].get("CPU", 0)
        mem = node["Resources"].get("memory", 0) / (1024**3)  # Convert to GB
        alive = node["Alive"]
        status = "✓ Active" if alive else "✗ Inactive"
        
        print(f"  Node {i} ({ip}): {status}")
        print(f"    GPUs: {gpus}, CPUs: {cpus}, Memory: {mem:.1f} GB")
    
    print(f"\nDashboard: http://10.0.0.75:8265")
    
    ray.shutdown()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure the Ray cluster is running on the Linux VM")
    sys.exit(1)
PY
EOF

chmod +x monitor_cluster.sh
echo "✓ Created monitor_cluster.sh"

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Prerequisites before running:"
echo "1. Linux VM (10.0.0.75):"
echo "   - Run setup_linux_vm.sh"
echo "   - Copy data to ~/mltownhouseeval/"
echo "   - Start head: ~/start_ray_head.sh"
echo ""
echo "2. WSL2 (192.168.0.233):"
echo "   - Run setup_wsl2.sh"
echo "   - Copy data to ~/mltownhouseeval/"
echo "   - Start worker: ~/start_ray_worker.sh"
echo ""
echo "3. On this Mac:"
echo "   - Run: ./monitor_cluster.sh (to verify cluster)"
echo "   - Run: ./run_distributed.sh (to start training)"
echo ""
echo "Dashboard: http://10.0.0.75:8265"