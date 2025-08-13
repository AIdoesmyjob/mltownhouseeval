# RTX 3090 Deployment Instructions

## Files to Transfer

Transfer the `GPU_3090_Deploy/` directory to your Linux VM:

```bash
# From your Mac - Transfer deployment code and REAL 10-year dataset
cd /Users/monstrcow/projects/TownhouseAlgo
scp -r GPU_3090_Deploy/ user@your-vm-ip:/home/user/
scp "Jan 1 2015_Aug 13 2025.csv" user@vm:/home/user/GPU_3090_Deploy/townhouse_data.csv
```

## Quick Setup on Linux VM

```bash
ssh user@your-vm-ip
cd GPU_3090_Deploy
chmod +x setup_3090.sh
./setup_3090.sh
```

## Run Search

### Test Run (1-2 hours)
```bash
# Test with 7,435 properties from Jan 2015 - Aug 2025
./run_gpu_search.sh --data townhouse_data.csv --min 3 --max 8
```

### Full Run (2-3 days)
```bash
# Full exhaustive search on 10 years of Chilliwack data
nohup ./run_gpu_search.sh --data townhouse_data.csv --min 3 --max 12 &
tail -f nohup.out  # Monitor progress
```

### Sold Only Analysis
```bash
# Find true value drivers (without list price bias)
./run_gpu_search.sh --data townhouse_data.csv --min 3 --max 10 --sold-only
```

## Monitor & Analyze

```bash
# Watch GPU usage
./monitor_gpu.sh

# Check progress
cat gpu_results/progress.json

# Analyze results when done
./analyze_results.py
```

## Expected Performance

- RTX 3090: 20,000-30,000 models/sec
- Full dataset: **7,435 properties** (Jan 2015 - Aug 2025)
- 3-8 features: 1-2 hours
- 3-12 features: 2-3 days
- Memory needed: ~8GB GPU RAM, 32GB system RAM

## Files Created

- `gpu_results/results.pkl` - All model results
- `gpu_results/best_models.pkl` - Top 100 models
- `gpu_results/feature_importance.pkl` - Feature rankings
- `gpu_results/progress.json` - Progress tracking

Good luck! The RTX 3090 will find your optimal model in days instead of years!