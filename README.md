# ML Townhouse Price Evaluation System

GPU-accelerated exhaustive search system for finding the optimal townhouse/rowhouse pricing model using 10 years of real estate data from Chilliwack, BC, Canada.

## 🎯 Overview

This project combines 15 years of real estate expertise with advanced machine learning to identify the true drivers of townhouse and rowhouse values. It tests **millions of feature combinations** to find the optimal predictive model.

## 📊 Dataset

- **7,435 properties** from Chilliwack, BC
- **10 years of data** (January 2015 - August 2025)
- Includes townhouses and rowhouses
- Rich feature set: location, size, age, bedrooms, bathrooms, maintenance fees, etc.

## 🚀 Performance

Optimized for NVIDIA RTX 3090:
- **20,000-30,000 models/second**
- Tests that would take years on CPU complete in 2-3 days
- Exhaustive search up to 12-15 features

## 📁 Project Structure

```
├── GPU_3090_Deploy/           # GPU deployment package
│   ├── gpu_exhaustive_search.py  # Main GPU search engine
│   ├── setup_3090.sh             # Automated setup script
│   └── README.md                 # Deployment documentation
├── Jan 1 2015_Aug 13 2025.csv   # Full 10-year dataset
├── check_correlation.py          # Feature correlation analysis
├── investigate_single_story.py   # Simpson's Paradox investigation
└── DEPLOY_INSTRUCTIONS.md        # Quick deployment guide
```

## 🔧 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/AIdoesmyjob/mltownhouseeval.git
cd mltownhouseeval
```

### 2. Deploy to GPU Server
```bash
# Transfer to Linux VM with RTX 3090
scp -r GPU_3090_Deploy/ user@vm:/home/user/
scp "Jan 1 2015_Aug 13 2025.csv" user@vm:/home/user/GPU_3090_Deploy/townhouse_data.csv

# SSH and setup
ssh user@vm
cd GPU_3090_Deploy
chmod +x setup_3090.sh
./setup_3090.sh
```

### 3. Run Exhaustive Search
```bash
# Test run (1-2 hours)
./run_gpu_search.sh --data townhouse_data.csv --min 3 --max 8

# Full search (2-3 days)
nohup ./run_gpu_search.sh --data townhouse_data.csv --min 3 --max 12 &
```

## 🔍 Key Features

- **Exhaustive Feature Search**: Tests every possible combination
- **GPU Acceleration**: XGBoost, LightGBM, CatBoost with CUDA
- **Resume Capability**: Can restart interrupted searches
- **Feature Engineering**: 40+ engineered features including ratios, polynomials, interactions
- **Multiple Models**: Compares different algorithms simultaneously
- **Real Estate Specific**: Handles Canadian MLS data format

## 📈 Results

The system identifies:
- Optimal feature combinations from millions tested
- Best algorithm (XGBoost vs LightGBM vs CatBoost)
- True property value drivers
- Feature importance rankings
- Prediction accuracy metrics (MAE, R², MAPE)

## 💡 Insights Discovered

Through analysis of the sample data, we've already uncovered:
- **Simpson's Paradox**: Single-story units appear cheaper due to age confounding (31.5 vs 15.7 years average)
- **Age Restrictions**: 50% of single-story units are age-restricted (55+)
- **Feature Correlation**: Size and floor count have 0.632 correlation but capture different value signals

## 🛠️ Requirements

- **GPU**: NVIDIA RTX 3090 or better
- **CUDA**: 12.0+
- **RAM**: 32GB+ recommended
- **Storage**: SSD preferred for faster I/O
- **OS**: Linux (Ubuntu 20.04+ tested)

## 📊 Example Output

```
TOP 10 MODELS
==================================================
1. XGB | MAE: $3,441 | R²: 0.9874 | Features: 4
   List_Price_Clean, Status_Encoded, Bath_Per_BR, City_Encoded...
2. LGB | MAE: $3,558 | R²: 0.9865 | Features: 5
   List_Price_Clean, TotFlArea_Clean, Age, Location_Age...
```

## 🤝 Contributing

This project combines domain expertise with machine learning. Contributions welcome in:
- Feature engineering ideas
- Performance optimizations
- Canadian real estate data handling
- Visualization improvements

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Developed using Claude Code (Anthropic)
- Real estate data from Chilliwack & District Real Estate Board
- 15 years of local market expertise

---

**For detailed deployment instructions, see [DEPLOY_INSTRUCTIONS.md](DEPLOY_INSTRUCTIONS.md)**