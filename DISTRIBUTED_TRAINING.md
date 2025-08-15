# Distributed GPU Training with Ray

This guide explains how to run distributed XGBoost training across multiple GPUs using Ray.

## Architecture Overview

```
MacBook (Orchestrator)
    ↓
    Ray Client → ray://10.0.0.75:10001
    ↓
Linux VM (10.0.0.75) - Head Node
    - RTX 3090 GPU
    - Ray GCS Server
    - Ray Dashboard
    ↓
WSL2 (192.168.0.233) - Worker Node
    - RTX 4090 GPU
    - Ray Worker
```

## Quick Start

### Step 1: Setup Linux VM (Head Node)

SSH into your Linux VM (10.0.0.75) and run:

```bash
# Download and run setup script
bash setup_linux_vm.sh

# Copy your data file
cp /path/to/sales_2015_2025.csv ~/mltownhouseeval/

# Start Ray head node
~/start_ray_head.sh
```

### Step 2: Setup WSL2 (Worker Node)

In Windows WSL2 terminal (192.168.0.233):

```bash
# Download and run setup script
bash setup_wsl2.sh

# Copy data (keep in WSL2 filesystem for performance!)
cp /path/to/sales_2015_2025.csv ~/mltownhouseeval/

# Start Ray worker (after head is running)
~/start_ray_worker.sh
```

### Step 3: Run from MacBook

On your MacBook:

```bash
# Setup Mac client
bash setup_mac_client.sh

# Verify cluster is ready
./monitor_cluster.sh

# Start distributed training
./run_distributed.sh
```

## Detailed Setup Instructions

### Linux VM Requirements

1. **NVIDIA Driver**: Version 525+ for RTX 3090
2. **CUDA Toolkit**: 11.8 or 12.x
3. **Firewall Ports**:
   - TCP 6379 (Ray GCS)
   - TCP 10001 (Ray Client)
   - TCP 8265 (Dashboard)

```bash
# Open ports with ufw
sudo ufw allow 6379/tcp
sudo ufw allow 10001/tcp
sudo ufw allow 8265/tcp
```

### WSL2 Requirements

1. **Windows NVIDIA Driver**: Latest version with WSL2 support
2. **WSL2 Updates**: `wsl --update`
3. **Ubuntu in WSL2**: 20.04 or 22.04
4. **Performance Config**: Add to `C:\Users\YourName\.wslconfig`:

```ini
[wsl2]
memory=32GB
processors=8
localhostForwarding=true

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
```

### Data Placement

**CRITICAL**: Place data files at the same path on both machines:
- Linux VM: `/home/monstrcow/mltownhouseeval/sales_2015_2025.csv`
- WSL2: `/home/monstrcow/mltownhouseeval/sales_2015_2025.csv`

For WSL2, keep files in the Linux filesystem (not `/mnt/c/`) for 10x better I/O performance.

## Configuration

Edit `orchestrator.py` to customize:

```python
CONFIG = {
    "rolling_window_days": 90,      # Baseline window
    "min_fsa_samples": 15,           # Min samples for FSA baseline
    "half_life_days": 180,           # Time decay parameter
    "test_fraction": 0.20,           # Test set size
    "max_features": 4,               # Max feature combinations
    "batch_size": 300,               # Combos per batch
    
    "params": {
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        # ... other XGBoost params
    }
}
```

## Monitoring

### Ray Dashboard
Open http://10.0.0.75:8265 in your browser to see:
- Active nodes and GPUs
- Running tasks
- Resource utilization
- Job progress

### Check Cluster Status
```bash
# From Mac
./monitor_cluster.sh
```

### View Logs
```bash
# On Linux VM
ray status
tail -f /tmp/ray/session_latest/logs/gcs_server.out

# On WSL2
tail -f /tmp/ray/session_latest/logs/raylet.out
```

## Performance Tuning

### Batch Size
- Larger batches = less overhead, more memory
- Start with 300, adjust based on GPU memory
- RTX 4090 can handle larger batches than 3090

### GPU Memory
If you get CUDA OOM errors:
1. Reduce `batch_size`
2. Reduce `max_bin` in XGBoost params
3. Use `tree_method='hist'` instead of `gpu_hist`

### Network
- Ensure low latency between nodes (<5ms ideal)
- Use wired connections, not WiFi
- Consider increasing Ray's gRPC message size for large datasets

## Troubleshooting

### Cannot Connect to Ray Cluster
```bash
# Check head node is running
ssh user@10.0.0.75 "ray status"

# Check firewall
sudo ufw status

# Test connectivity
nc -zv 10.0.0.75 6379
```

### GPU Not Detected
```bash
# Verify CUDA
nvidia-smi

# Test XGBoost GPU
python3 -c "import xgboost; print(xgboost.__version__)"

# Check Ray sees GPU
ray status --address=10.0.0.75:6379
```

### WSL2 Slow Performance
- Move data from `/mnt/c/` to WSL2 filesystem
- Increase WSL2 memory in `.wslconfig`
- Disable Windows Defender scanning on WSL2 files

### XGBoost Falls Back to CPU
```python
# Force GPU in params
params = {
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0
}
```

## Scaling to More Nodes

To add more GPU nodes:

1. Run `setup_wsl2.sh` on new machine
2. Start worker: `ray start --address='10.0.0.75:6379'`
3. Update `ip_to_tag` mapping in `orchestrator.py`

## Results

Training outputs:
- `distributed_results_TIMESTAMP.csv`: All feature combinations with metrics
- Console: Top 10 best feature combinations
- Metrics: MAE, WAPE, R² on log premium

## Stopping the Cluster

```bash
# On each node
~/stop_ray.sh

# Or from head node
ray stop --force
```

## Advanced Features

### Custom Node Selection
```python
# Pin actor to specific node
@ray.remote(num_gpus=1, resources={"node:10.0.0.75": 1})
class TrainWorker:
    ...
```

### Fault Tolerance
```python
# Retry failed tasks
@ray.remote(num_gpus=1, max_retries=3)
class TrainWorker:
    ...
```

### Progress Tracking
```python
# Use Ray's progress bars
from ray.experimental import tqdm_ray
for batch in tqdm_ray.tqdm(batches):
    ...
```