# Ubuntu VM (4090) Ray Head Node Setup Instructions

## Prerequisites
- Ubuntu VM with RTX 4090 GPU
- Internet connection
- SSH access enabled

## Setup Instructions

### 1. System Update and Essential Tools
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y git curl wget build-essential python3-pip python3-venv openssh-server
```

### 2. Enable SSH Access (if not already enabled)
```bash
# Ensure SSH is installed and running
sudo systemctl enable ssh
sudo systemctl start ssh

# Get your IP address for remote access
ip addr show

# Note: Share this IP with Mac for SSH access
```

### 3. CUDA and GPU Drivers Setup
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA toolkit and drivers
sudo apt install -y cuda-toolkit-12-3 nvidia-driver-545

# Verify GPU is detected
nvidia-smi

# Add CUDA to PATH (add to ~/.bashrc)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Python Environment Setup
```bash
# Create Python virtual environment
python3 -m venv ~/ray_env

# Activate the environment
source ~/ray_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 5. Clone Repository from GitHub
```bash
# Configure git (replace with your details)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Clone the TownhouseAlgo repository
cd ~
git clone https://github.com/YOUR_USERNAME/TownhouseAlgo.git
cd TownhouseAlgo

# Pull latest changes
git pull origin main
```

### 6. Install Project Dependencies
```bash
# Make sure virtual environment is activated
source ~/ray_env/bin/activate

# Install Ray with GPU support
pip install "ray[default]" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib psutil

# Verify CUDA is available in PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 7. Configure Ray Head Node
```bash
# Start Ray head node with dashboard
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# Get the Ray cluster address (note this for worker nodes)
ray status

# The address will be something like: ray://192.168.x.x:10001
```

### 8. Configure Firewall for Ray
```bash
# Open necessary ports for Ray cluster
sudo ufw allow 6379/tcp   # Ray GCS
sudo ufw allow 8265/tcp   # Ray Dashboard
sudo ufw allow 10001/tcp  # Ray Client Server
sudo ufw allow 6380:6390/tcp  # Object Manager ports
sudo ufw allow 20000:30000/tcp  # Worker ports range

# Enable firewall if not already
sudo ufw enable
```

### 9. Create Orchestrator Launch Script
```bash
# Create a script to start everything
cat > ~/start_ray_head.sh << 'EOF'
#!/bin/bash
# Activate Python environment
source ~/ray_env/bin/activate

# Start Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# Display cluster info
echo "Ray cluster started. Dashboard available at: http://$(hostname -I | awk '{print $1}'):8265"
ray status

# Keep script running
echo "Ray head node is running. Press Ctrl+C to stop."
tail -f /dev/null
EOF

chmod +x ~/start_ray_head.sh
```

### 10. Running the Orchestrator
```bash
# Navigate to project directory
cd ~/TownhouseAlgo

# Activate environment
source ~/ray_env/bin/activate

# Run the orchestrator (after Ray is started)
python orchestrator.py
```

## For Mac/Worker Nodes to Connect

Share the following information with Mac/worker nodes:
1. **SSH Access**: `ssh username@<VM_IP_ADDRESS>`
2. **Ray Head Address**: `ray://<VM_IP_ADDRESS>:10001`
3. **Ray Dashboard**: `http://<VM_IP_ADDRESS>:8265`

### Mac Connection Commands:
```bash
# SSH into the head node
ssh username@<VM_IP_ADDRESS>

# Or, connect Mac as a Ray worker
ray start --address='<VM_IP_ADDRESS>:6379'
```

## Maintenance Commands

### Check Ray Status
```bash
ray status
```

### Stop Ray
```bash
ray stop
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Pull Latest Code Updates
```bash
cd ~/TownhouseAlgo
git pull origin main
```

### View Ray Dashboard
Open browser and navigate to: `http://<VM_IP_ADDRESS>:8265`

## Troubleshooting

### If Ray doesn't start:
```bash
# Kill any existing Ray processes
ray stop --force
pkill -f ray

# Restart
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

### If GPU not detected:
```bash
# Reboot after driver installation
sudo reboot

# Check driver status
nvidia-smi
```

### If connection refused from Mac:
```bash
# Check firewall rules
sudo ufw status

# Check Ray is listening
netstat -tlnp | grep ray
```

## Auto-start Ray on Boot (Optional)
```bash
# Create systemd service
sudo nano /etc/systemd/system/ray-head.service

# Add the following content (adjust username):
[Unit]
Description=Ray Head Node
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME
ExecStart=/home/YOUR_USERNAME/ray_env/bin/ray start --head --block --port=6379 --dashboard-host=0.0.0.0
Restart=on-failure

[Install]
WantedBy=multi-user.target

# Enable and start service
sudo systemctl enable ray-head.service
sudo systemctl start ray-head.service
```