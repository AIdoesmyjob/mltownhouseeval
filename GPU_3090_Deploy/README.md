# GPU Exhaustive Search - RTX 3090 Deployment

## 🚀 Quick Start

### 1. Transfer files to your Linux VM:
```bash
# From your Mac
scp -r GPU_3090_Deploy/ user@your-vm-ip:/home/user/
```

### 2. SSH into your VM:
```bash
ssh user@your-vm-ip
cd GPU_3090_Deploy
```

### 3. Run setup:
```bash
chmod +x setup_3090.sh
./setup_3090.sh
```

### 4. Upload your data:
```bash
# From your Mac
scp /path/to/your/full_townhouse_data.csv user@vm:/home/user/GPU_3090_Deploy/
```

### 5. Start the search:
```bash
# Quick test (1-2 hours)
./run_gpu_search.sh --data townhouse_data.csv --min 3 --max 5

# Full search (2-3 days)
nohup ./run_gpu_search.sh --data townhouse_data.csv --min 3 --max 12 &

# Monitor progress
tail -f nohup.out
```

## 📊 Expected Performance

### RTX 3090 Benchmarks:
- **XGBoost**: ~20,000-30,000 models/sec
- **LightGBM**: ~15,000-25,000 models/sec  
- **CatBoost**: ~10,000-20,000 models/sec

### Time Estimates:
| Features | Combinations | Time (approx) |
|----------|-------------|---------------|
| 3-5 | 1M | 1 hour |
| 3-8 | 100M | 1-2 hours |
| 3-10 | 1B | 10-15 hours |
| 3-12 | 10B | 2-3 days |
| 3-15 | 100B+ | 2+ weeks |

## 🎯 Recommended Settings

### For Initial Testing:
```bash
./run_gpu_search.sh --data your_data.csv --min 3 --max 8
```
- Takes 1-2 hours
- Finds good baseline models
- Tests feasibility

### For Production Results:
```bash
nohup ./run_gpu_search.sh --data your_data.csv --min 3 --max 12 --batch 2000 &
```
- Takes 2-3 days
- Comprehensive search
- Finds optimal model

### For Sold Properties Only:
```bash
./run_gpu_search.sh --data your_data.csv --min 3 --max 10 --sold-only
```
- Excludes active/pending listings
- Finds true value drivers
- No list price bias

## 📈 Monitor Progress

### Watch GPU usage:
```bash
# In another terminal
./monitor_gpu.sh
```

### Check progress:
```bash
# View progress file
cat gpu_results/progress.json

# Watch output
tail -f nohup.out
```

### Resume if interrupted:
```bash
./run_gpu_search.sh --data your_data.csv --resume
```

## 📊 Analyze Results

After completion:
```bash
./analyze_results.py

# Or specify directory
./analyze_results.py gpu_results
```

## 🔧 Troubleshooting

### GPU not detected:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Reinstall if needed
sudo apt install nvidia-driver-535
```

### Out of memory:
```bash
# Reduce batch size
./run_gpu_search.sh --batch 500
```

### Permission denied:
```bash
chmod +x *.sh
chmod +x *.py
```

## 📁 Output Files

After completion, you'll have:
- `gpu_results/results.pkl` - All model results
- `gpu_results/best_models.pkl` - Top 100 models
- `gpu_results/feature_importance.pkl` - Feature rankings
- `gpu_results/progress.json` - Progress tracking

## 🎯 What You'll Discover

1. **Optimal feature combination** from millions tested
2. **Best algorithm** (XGBoost vs LightGBM vs CatBoost)
3. **True value drivers** in Chilliwack market
4. **Feature importance rankings**
5. **Prediction accuracy** (MAE, R², MAPE)

## 💡 Tips

1. **Start small**: Test with `--max 5` first
2. **Use nohup**: For long runs that survive SSH disconnects
3. **Monitor temperature**: RTX 3090 can run hot
4. **Save results**: Copy pkl files back to your Mac regularly

## 🚨 Important Notes

- Ensure adequate cooling for RTX 3090
- Have at least 32GB RAM for large datasets
- Use SSD for faster data loading
- Keep VM running during multi-day searches

## 📞 Support

If issues arise:
1. Check `nvidia-smi` for GPU status
2. Review Python package versions
3. Verify CUDA installation
4. Check available disk space

Good luck with your search! The RTX 3090 will find the optimal model in 2-3 days instead of years!