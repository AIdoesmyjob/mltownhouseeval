#!/usr/bin/env python3
"""
GPU-ACCELERATED EXHAUSTIVE TOWNHOUSE PRICE MODEL SEARCH
Optimized for NVIDIA RTX 3090 on Linux
Tests all feature combinations to find optimal model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# GPU Libraries
try:
    import xgboost as xgb
    HAS_XGB = True
    print("✓ XGBoost available")
except:
    HAS_XGB = False
    print("✗ XGBoost not found - install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("✓ LightGBM available")
except:
    HAS_LGB = False
    print("✗ LightGBM not found - install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
    print("✓ CatBoost available")
except:
    HAS_CATBOOST = False
    print("✗ CatBoost not found - install with: pip install catboost")

# Standard libraries
import os
import sys
import time
import pickle
import json
from datetime import datetime, timedelta
from itertools import combinations
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import argparse


class GPUExhaustiveSearch:
    def __init__(self, data_path, output_dir='gpu_results', gpu_id=0):
        """Initialize GPU exhaustive search"""
        self.data_path = data_path
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        
        # Set GPU device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # File paths
        self.results_file = f"{output_dir}/results.pkl"
        self.best_models_file = f"{output_dir}/best_models.pkl"
        self.progress_file = f"{output_dir}/progress.json"
        self.feature_importance_file = f"{output_dir}/feature_importance.pkl"
        
        print("="*70)
        print("GPU EXHAUSTIVE SEARCH - RTX 3090 OPTIMIZED")
        print("="*70)
        self.check_gpu()
        
    def check_gpu(self):
        """Verify GPU is available"""
        print("\n🔍 Checking GPU Configuration...")
        
        # Check CUDA availability
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', 
                                   '--format=csv,noheader'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                print(f"✓ GPU Detected: {gpu_info}")
            else:
                print("⚠️ nvidia-smi not found - GPU may not be available")
        except:
            print("⚠️ Could not detect GPU")
        
        # Test XGBoost GPU
        if HAS_XGB:
            try:
                test_model = xgb.XGBRegressor(
                    tree_method='gpu_hist',
                    gpu_id=self.gpu_id,
                    n_estimators=1
                )
                test_model.fit([[1, 2]], [1])
                print("✓ XGBoost GPU acceleration confirmed")
            except Exception as e:
                print(f"✗ XGBoost GPU not working: {str(e)[:50]}")
        
        print("-"*70)
    
    def load_and_prepare_data(self, sold_only=False):
        """Load and prepare the full dataset"""
        print("\n📊 LOADING DATA...")
        
        # Load CSV
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df)} total records")
        
        # Filter to sold only if requested
        if sold_only:
            df = df[df['Status'] == 'F'].copy()
            print(f"✓ Filtered to {len(df)} sold properties")
        
        # Clean price (handle both Price and Sold Price columns)
        if 'Sold Price' in df.columns and sold_only:
            df['Price_Clean'] = pd.to_numeric(
                df['Sold Price'].astype(str).str.replace('$', '').str.replace(',', ''),
                errors='coerce'
            )
        else:
            df['Price_Clean'] = pd.to_numeric(
                df['Price'].astype(str).str.replace('$', '').str.replace(',', ''),
                errors='coerce'
            )
        
        # Remove missing prices
        df = df[df['Price_Clean'].notna() & (df['Price_Clean'] > 0)]
        print(f"✓ Valid prices: {len(df)} properties")
        
        # ========================================
        # COMPREHENSIVE FEATURE ENGINEERING
        # ========================================
        print("\n🔧 Engineering features...")
        features = {}
        
        # 1. Core numeric features
        numeric_cols = [
            'Tot BR', 'Tot Baths', 'Yr Blt', 'Age', 'DOM', 'TotalPrkng',
            '# of Kitchens', 'Full Baths', 'Half Baths', 'Bds In Bsmt',
            'Bds Not In Bsmt', 'No. Floor Levels', 'Cumulative DOM',
            'Floor Area - Unfinished', 'Floor Area Fin - Abv Main'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                features[col] = df[col].fillna(0)
        
        # 2. Parse string numeric features
        string_numeric = [
            'TotFlArea', 'MaintFee', 'List Price',
            'Floor Area Fin - Basement', 'Floor Area Fin - BLW Main',
            'Floor Area Fin - Main Flr', 'Floor Area Fin - Total'
        ]
        
        for col in string_numeric:
            if col in df.columns:
                # Skip List Price if doing sold_only analysis
                if sold_only and col == 'List Price':
                    continue
                clean_col = f'{col}_Clean'
                features[clean_col] = pd.to_numeric(
                    df[col].astype(str).str.replace('$', '').str.replace(',', ''),
                    errors='coerce'
                ).fillna(0)
        
        # 3. Categorical encodings
        categorical = ['S/A', 'TypeDwel', 'Status', 'Title to Land', 'City']
        
        for col in categorical:
            if col in df.columns:
                # Skip Status if doing sold_only
                if sold_only and col == 'Status':
                    continue
                le = LabelEncoder()
                features[f'{col}_Encoded'] = le.fit_transform(df[col].fillna('Unknown'))
        
        # 4. Date features
        if 'List Date' in df.columns:
            df['List_Date'] = pd.to_datetime(df['List Date'], errors='coerce')
            features['List_Month'] = df['List_Date'].dt.month.fillna(6)
            features['List_Quarter'] = df['List_Date'].dt.quarter.fillna(2)
            features['List_Year'] = df['List_Date'].dt.year.fillna(2020)
        
        # 5. Binary features
        features['Has_Basement'] = (df['Floor Area Fin - Basement'].notna()).astype(int)
        features['Is_Age_Restricted'] = df['Restricted Age'].notna().astype(int)
        
        # 6. Ratios and interactions
        if 'Tot BR' in features and features['Tot BR'] is not None:
            features['Bath_Per_BR'] = features.get('Tot Baths', 0) / np.maximum(features['Tot BR'], 1)
            features['Parking_Per_BR'] = features.get('TotalPrkng', 0) / np.maximum(features['Tot BR'], 1)
        
        if 'TotFlArea_Clean' in features:
            features['SqFt_Per_BR'] = features['TotFlArea_Clean'] / np.maximum(features.get('Tot BR', 1), 1)
            features['Price_Per_SqFt'] = df['Price_Clean'] / np.maximum(features['TotFlArea_Clean'], 1)
        
        # 7. Polynomial features
        if 'Age' in features:
            features['Age_Squared'] = features['Age'] ** 2
            features['Age_Log'] = np.log1p(features['Age'])
        
        if 'TotFlArea_Clean' in features:
            features['SqFt_Squared'] = features['TotFlArea_Clean'] ** 2
            features['SqFt_Log'] = np.log1p(features['TotFlArea_Clean'])
        
        # 8. Location × Size/Age interactions
        if 'S/A_Encoded' in features:
            if 'Age' in features:
                features['Location_Age'] = features['S/A_Encoded'] * features['Age']
            if 'TotFlArea_Clean' in features:
                features['Location_Size'] = features['S/A_Encoded'] * features['TotFlArea_Clean']
        
        # Convert to DataFrame
        X = pd.DataFrame(features)
        y = df['Price_Clean']
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Store metadata
        self.feature_names = list(X.columns)
        self.n_samples = len(X)
        self.price_mean = y.mean()
        self.price_std = y.std()
        
        print(f"✓ Created {len(self.feature_names)} features")
        print(f"✓ Dataset: {X.shape}")
        print(f"✓ Price range: ${y.min():,.0f} - ${y.max():,.0f}")
        print(f"✓ Average price: ${y.mean():,.0f}")
        
        return X, y
    
    def generate_combinations(self, min_features=3, max_features=15):
        """Generate feature combinations"""
        n_features = len(self.feature_names)
        max_features = min(max_features, n_features)
        
        total = 0
        combos = []
        
        for r in range(min_features, max_features + 1):
            n_combos = len(list(combinations(range(n_features), r)))
            total += n_combos
            print(f"  {r} features: {n_combos:,} combinations")
            
            for combo in combinations(range(n_features), r):
                combos.append(combo)
        
        print(f"\nTotal combinations: {total:,}")
        return combos
    
    def test_gpu_model(self, args):
        """Test a single model configuration on GPU"""
        combo, model_type, X_train, X_test, y_train, y_test, scaler = args
        
        try:
            # Select features
            X_train_sub = X_train[:, combo]
            X_test_sub = X_test[:, combo]
            
            # Scale
            X_train_scaled = scaler.fit_transform(X_train_sub)
            X_test_scaled = scaler.transform(X_test_sub)
            
            # Train model based on type
            if model_type == 'xgb' and HAS_XGB:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    tree_method='gpu_hist',
                    gpu_id=self.gpu_id,
                    random_state=42,
                    verbosity=0
                )
            elif model_type == 'lgb' and HAS_LGB:
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    device='gpu',
                    gpu_platform_id=0,
                    gpu_device_id=self.gpu_id,
                    random_state=42,
                    verbosity=-1
                )
            elif model_type == 'cat' and HAS_CATBOOST:
                model = CatBoostRegressor(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    task_type='GPU',
                    devices=str(self.gpu_id),
                    random_state=42,
                    verbose=False
                )
            else:
                return None
            
            # Fit and predict
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            return {
                'model_type': model_type,
                'n_features': len(combo),
                'feature_indices': combo,
                'features': [self.feature_names[i] for i in combo],
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }
            
        except Exception as e:
            return None
    
    def run_search(self, min_features=3, max_features=15, test_size=0.2, 
                   batch_size=1000, sold_only=False, resume=False):
        """Run exhaustive GPU search"""
        
        print("\n" + "="*70)
        print("🚀 STARTING GPU EXHAUSTIVE SEARCH")
        print("="*70)
        
        # Load data
        X, y = self.load_and_prepare_data(sold_only=sold_only)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Convert to numpy for faster GPU transfer
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values
        
        print(f"\n✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        # Generate combinations
        print("\n📊 Generating feature combinations...")
        combos = self.generate_combinations(min_features, max_features)
        
        # Model types to test
        model_types = []
        if HAS_XGB:
            model_types.append('xgb')
        if HAS_LGB:
            model_types.append('lgb')
        if HAS_CATBOOST:
            model_types.append('cat')
        
        if not model_types:
            print("❌ No GPU-accelerated libraries available!")
            return None
        
        total_tests = len(combos) * len(model_types)
        print(f"\n📈 Total tests: {total_tests:,}")
        print(f"   {len(combos):,} combinations × {len(model_types)} models")
        print(f"   Models: {model_types}")
        
        # Load previous results if resuming
        if resume and os.path.exists(self.results_file):
            with open(self.results_file, 'rb') as f:
                results = pickle.load(f)
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            completed = progress['completed']
            print(f"\n📂 Resuming from {completed:,} completed tests")
        else:
            results = []
            completed = 0
        
        # Create tasks
        tasks = []
        scaler = StandardScaler()
        
        for combo in combos:
            for model_type in model_types:
                if len(tasks) >= completed:  # Skip already completed
                    tasks.append((combo, model_type, X_train, X_test, 
                                y_train, y_test, scaler))
        
        # Process in batches
        print(f"\n⚡ Processing on GPU (batch size: {batch_size})...")
        print("-"*70)
        
        start_time = time.time()
        best_mae = float('inf')
        
        try:
            for i in range(completed, len(tasks), batch_size):
                batch = tasks[i:i+min(batch_size, len(tasks)-i)]
                batch_start = time.time()
                
                # Process batch on GPU
                batch_results = []
                for task in batch:
                    result = self.test_gpu_model(task)
                    if result:
                        batch_results.append(result)
                        if result['mae'] < best_mae:
                            best_mae = result['mae']
                            best_model = result
                
                results.extend(batch_results)
                completed += len(batch)
                
                # Calculate stats
                batch_time = time.time() - batch_start
                total_time = time.time() - start_time
                rate = completed / total_time
                eta = (total_tests - completed) / rate if rate > 0 else 0
                
                # Progress update
                print(f"\r⚡ Progress: {completed:,}/{total_tests:,} ({100*completed/total_tests:.1f}%) | "
                      f"Rate: {rate:.0f}/sec | "
                      f"Best MAE: ${best_mae:,.0f} | "
                      f"ETA: {timedelta(seconds=int(eta))}", end='', flush=True)
                
                # Save progress periodically
                if completed % 10000 == 0 or completed == total_tests:
                    self.save_progress(results, completed, total_tests)
                    print(f"\n💾 Saved {len(results)} results")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            self.save_progress(results, completed, total_tests)
        
        # Final save
        self.save_progress(results, completed, total_tests)
        
        # Analysis
        print("\n\n" + "="*70)
        print("✅ SEARCH COMPLETE")
        print("="*70)
        print(f"Total time: {timedelta(seconds=int(time.time() - start_time))}")
        print(f"Models tested: {len(results):,}")
        print(f"Average rate: {len(results)/(time.time() - start_time):.0f} models/sec")
        
        return self.analyze_results(results)
    
    def save_progress(self, results, completed, total):
        """Save current progress"""
        
        # Save results
        with open(self.results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save best models
        if results:
            df_results = pd.DataFrame(results).sort_values('mae')
            best = df_results.head(100)
            with open(self.best_models_file, 'wb') as f:
                pickle.dump(best, f)
        
        # Save progress
        progress = {
            'completed': completed,
            'total': total,
            'timestamp': datetime.now().isoformat(),
            'best_mae': min(r['mae'] for r in results) if results else None
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def analyze_results(self, results):
        """Analyze and display results"""
        
        if not results:
            return None
        
        df = pd.DataFrame(results).sort_values('mae')
        
        print("\n🏆 TOP 10 MODELS")
        print("-"*70)
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            print(f"{i+1}. {row['model_type'].upper()} | "
                  f"MAE: ${row['mae']:,.0f} | "
                  f"R²: {row['r2']:.4f} | "
                  f"Features: {row['n_features']}")
            print(f"   {', '.join(row['features'][:5])}...")
        
        # Feature importance
        print("\n🔍 TOP FEATURES (in top 100 models)")
        print("-"*70)
        
        feature_counts = {}
        for _, row in df.head(100).iterrows():
            for feature in row['features']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, count) in enumerate(sorted_features[:20], 1):
            print(f"{i:2}. {feature:40} {count}%")
        
        # Save feature importance
        with open(self.feature_importance_file, 'wb') as f:
            pickle.dump(sorted_features, f)
        
        return df


def main():
    parser = argparse.ArgumentParser(description='GPU Exhaustive Model Search')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--min-features', type=int, default=3)
    parser.add_argument('--max-features', type=int, default=15)
    parser.add_argument('--output', type=str, default='gpu_results')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--sold-only', action='store_true', help='Use only sold properties')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Run search
    searcher = GPUExhaustiveSearch(args.data, args.output, args.gpu_id)
    results = searcher.run_search(
        min_features=args.min_features,
        max_features=args.max_features,
        batch_size=args.batch_size,
        sold_only=args.sold_only,
        resume=args.resume
    )
    
    print(f"\n📁 Results saved to: {args.output}/")
    print("   - results.pkl: All results")
    print("   - best_models.pkl: Top 100 models")
    print("   - feature_importance.pkl: Feature rankings")


if __name__ == "__main__":
    main()