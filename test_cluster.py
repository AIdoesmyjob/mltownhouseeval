#!/usr/bin/env python3
"""
Test script to verify Ray cluster setup
Checks connectivity, GPUs, and runs a small test
"""

import ray
import time
import sys

def test_cluster():
    print("=" * 70)
    print("RAY CLUSTER TEST")
    print("=" * 70)
    
    # Connect to Ray
    print("\n1. Connecting to Ray cluster...")
    try:
        ray.init(address="ray://10.0.0.75:10001")
        print("✓ Connected to Ray cluster")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("1. Is the Linux VM head node running? (~/start_ray_head.sh)")
        print("2. Is port 10001 open on the Linux VM?")
        print("3. Can you ping 10.0.0.75?")
        sys.exit(1)
    
    # Check nodes
    print("\n2. Checking cluster nodes...")
    nodes = ray.nodes()
    print(f"✓ Found {len(nodes)} nodes")
    
    # Display node info
    total_gpus = 0
    total_cpus = 0
    
    for i, node in enumerate(nodes, 1):
        ip = node["NodeManagerAddress"]
        gpus = node["Resources"].get("GPU", 0)
        cpus = node["Resources"].get("CPU", 0)
        mem = node["Resources"].get("memory", 0) / (1024**3)  # Convert to GB
        alive = node["Alive"]
        
        total_gpus += gpus
        total_cpus += cpus
        
        status = "✓ Active" if alive else "✗ Inactive"
        print(f"\n  Node {i} ({ip}): {status}")
        print(f"    GPUs: {gpus}")
        print(f"    CPUs: {cpus}")
        print(f"    Memory: {mem:.1f} GB")
        
        # Try to identify which machine
        if ip == "10.0.0.75":
            print(f"    → Linux VM (RTX 3090)")
        elif "192.168" in ip:
            print(f"    → WSL2 (RTX 4090)")
    
    print(f"\n  Total Resources:")
    print(f"    GPUs: {total_gpus}")
    print(f"    CPUs: {total_cpus}")
    
    # Check expected setup
    print("\n3. Verifying expected configuration...")
    if total_gpus == 2:
        print("✓ Found expected 2 GPUs")
    else:
        print(f"⚠ Expected 2 GPUs, found {total_gpus}")
        if total_gpus == 1:
            print("  One of the nodes might not have GPU access")
            print("  Check nvidia-smi on both machines")
    
    if len(nodes) == 2:
        print("✓ Found expected 2 nodes")
    else:
        print(f"⚠ Expected 2 nodes, found {len(nodes)}")
        if len(nodes) == 1:
            print("  WSL2 worker might not be connected")
            print("  Run ~/start_ray_worker.sh on WSL2")
    
    # Test GPU task
    print("\n4. Testing GPU task execution...")
    
    @ray.remote(num_gpus=1)
    def test_gpu():
        import socket
        import subprocess
        hostname = socket.gethostname()
        
        # Get GPU info
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            gpu_name = result.stdout.strip()
        except:
            gpu_name = "Unknown"
        
        return hostname, gpu_name
    
    try:
        # Run on all available GPUs
        futures = []
        for i in range(min(2, int(total_gpus))):
            futures.append(test_gpu.remote())
        
        results = ray.get(futures)
        print("✓ GPU tasks executed successfully")
        
        for hostname, gpu_name in results:
            print(f"  - {hostname}: {gpu_name}")
    
    except Exception as e:
        print(f"❌ GPU task failed: {e}")
    
    # Test data access
    print("\n5. Testing data file access...")
    
    @ray.remote
    def check_data_file():
        import os
        import socket
        hostname = socket.gethostname()
        path = "/home/monstrcow/mltownhouseeval/sales_2015_2025.csv"
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path) / (1024*1024)  # MB
            return hostname, True, f"{size:.1f} MB"
        return hostname, False, "Not found"
    
    # Check on both nodes
    futures = [check_data_file.remote() for _ in range(2)]
    results = ray.get(futures)
    
    all_good = True
    for hostname, exists, info in results:
        if exists:
            print(f"  ✓ {hostname}: {info}")
        else:
            print(f"  ❌ {hostname}: {info}")
            all_good = False
    
    if not all_good:
        print("\n  ⚠ Data file missing on some nodes")
        print("  Copy sales_2015_2025.csv to ~/mltownhouseeval/ on all machines")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if total_gpus == 2 and len(nodes) == 2 and all_good:
        print("✅ Cluster is ready for distributed training!")
        print("\nYou can now run:")
        print("  python orchestrator.py")
    else:
        print("⚠ Some issues detected. Please fix before running training.")
    
    print(f"\nRay Dashboard: http://10.0.0.75:8265")
    
    # Cleanup
    ray.shutdown()

if __name__ == "__main__":
    test_cluster()