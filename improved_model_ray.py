#!/usr/bin/env python3
"""
Improved Townhouse Price Prediction Model - Ray Distributed Version
Leverages Ray for distributed training across multiple GPUs (4090 + 3090)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
import warnings
import ray
from ray import tune
from ray.train import ScalingConfig
from ray.train.xgboost import XGBoostTrainer
import os
warnings.filterwarnings('ignore')

def to_numeric(series):
    """Convert series to numeric, handling $, commas and spaces"""
    if pd.api.types.is_object_dtype(series):
        series = (series.astype(str)
                       .str.replace('$', '', regex=False)
                       .str.replace(',', '', regex=False)
                       .str.replace(' ', '', regex=False))
    return pd.to_numeric(series, errors='coerce')

def wape(y_true, y_pred):
    """Calculate WAPE metric"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

class DistributedTownhouseModel:
    def __init__(self, ray_address=None):
        """
        Initialize Ray cluster connection
        ray_address: e.g., 'ray://192.168.1.100:10001' for head node
        """
        if ray_address:
            ray.init(address=ray_address)
        else:
            ray.init(ignore_reinit_error=True)
        
        print("Ray cluster resources:")
        print(ray.cluster_resources())
        
    def load_and_prepare_data(self, csv_path='Jan 1 2015_Aug 13 2025.csv'):
        """Load and prepare data with all feature engineering"""
        
        print("="*60)
        print("DISTRIBUTED MODEL - RAY CLUSTER")
        print("="*60)
        
        # Load data
        print("\nLoading data...")
        df = pd.read_csv(csv_path)
        print(f"Initial shape: {df.shape}")
        
        # Convert numeric columns
        num_cols = ['Price', 'Sold Price', 'List Price', 'TotFlArea', 'MaintFee', 'Yr Blt', 'Age',
                    'Full Baths', 'Half Baths', 'Tot BR', 'Tot Baths', 'No. Floor Levels',
                    'Storeys in Building', 'Fireplaces', 'Floor Area Fin - Total',
                    'Floor Area Fin - Main Flr', 'Floor Area Fin - Abv Main',
                    'Floor Area Fin - Basement', 'Floor Area - Unfinished',
                    'Tot Units in Strata Plan', 'Units in Development',
                    '# of Kitchens', '# of Pets']
        
        for col in num_cols:
            if col in df.columns:
                df[col] = to_numeric(df[col])
        
        # Parse dates
        df['List Date'] = pd.to_datetime(df['List Date'], errors='coerce')
        if 'Sold Date' in df.columns:
            df['Sold Date'] = pd.to_datetime(df['Sold Date'], errors='coerce')
        
        # Canonical columns
        self.TARGET_COL = 'Sold Price' if 'Sold Price' in df.columns else 'Price'
        self.AREA_COL = 'TotFlArea'
        self.DATE_COL = 'List Date'
        
        print(f"Using target column: {self.TARGET_COL}")
        
        # Filter valid rows
        df = df[df[self.TARGET_COL].notna() & (df[self.TARGET_COL] > 0) &
                df[self.AREA_COL].notna() & (df[self.AREA_COL] > 0) &
                df[self.DATE_COL].notna()].copy()
        
        # Sort by time
        df = df.sort_values(self.DATE_COL).reset_index(drop=True)
        
        # PPSF for comps
        df['sold_ppsf'] = df[self.TARGET_COL] / df[self.AREA_COL]
        
        print(f"After filtering: {len(df)} valid records")
        
        # Feature engineering
        self._engineer_features(df)
        
        # Build comp baseline
        self._build_comp_baseline(df)
        
        # Create temporal splits
        self._create_splits(df)
        
        # Prepare features
        self._prepare_features()
        
        return self
    
    def _engineer_features(self, df):
        """Add all engineered features"""
        print("\nEngineering features...")
        
        # Time features
        ref_date = pd.Timestamp('2015-01-01')
        df['month_idx'] = (df[self.DATE_COL] - ref_date).dt.days / 30.0
        df['year'] = df[self.DATE_COL].dt.year.astype('float32')
        df['month'] = df[self.DATE_COL].dt.month.astype('float32')
        df['quarter'] = ((df['month'] - 1) // 3 + 1).astype('float32')
        df['day_of_year'] = df[self.DATE_COL].dt.dayofyear.astype('float32')
        df['week_of_year'] = df[self.DATE_COL].dt.isocalendar().week.astype('float32')
        
        # Age at listing
        df['age_at_list'] = np.where(
            df['Yr Blt'].notna(),
            df['year'] - df['Yr Blt'].clip(lower=1900, upper=df['year']),
            df['Age']
        )
        df['age_at_list'] = df['age_at_list'].clip(lower=0).fillna(20)
        
        # Location
        if 'Postal Code' in df.columns:
            df['FSA'] = df['Postal Code'].astype(str).str.extract(r'^([A-Za-z]\d[A-Za-z])', expand=False)
            df['FSA'] = df['FSA'].fillna('UNK')
        else:
            df['FSA'] = 'UNK'
        
        # Ratios
        df['rooms_per_sqft'] = (df['Tot BR'] / df[self.AREA_COL]).replace([np.inf, -np.inf], np.nan)
        df['baths_equiv'] = (df['Full Baths'].fillna(0) + 0.5 * df['Half Baths'].fillna(0))
        df['baths_per_br'] = df['baths_equiv'] / df['Tot BR'].clip(lower=1)
        df['maint_per_sqft'] = (df['MaintFee'] / df[self.AREA_COL]).replace([np.inf, -np.inf], np.nan)
        
        # Winsorize outliers
        for c in ['sold_ppsf', 'maint_per_sqft', self.AREA_COL]:
            if c in df.columns:
                lo, hi = df[c].quantile([0.005, 0.995])
                df[c] = df[c].clip(lo, hi)
        
        # Missingness indicators
        for c in ['MaintFee', 'Floor Area Fin - Basement', 'Floor Area - Unfinished']:
            if c in df.columns:
                df[f'{c}_isna'] = df[c].isna().astype('float32')
        
        self.df = df
    
    def _build_comp_baseline(self, df):
        """Build time-based comp baseline"""
        print("\nBuilding comp baseline...")
        
        # Simple global rolling median for now (can enhance with FSA grouping)
        tmp = df[[self.DATE_COL, 'sold_ppsf']].set_index(self.DATE_COL).sort_index()
        df['ppsf_90D_global'] = tmp['sold_ppsf'].rolling('90D', min_periods=8).median().shift(1).values
        df['ppsf_365D_global'] = tmp['sold_ppsf'].rolling('365D', min_periods=12).median().shift(1).values
        
        # Use best available
        df['ppsf_baseline'] = df['ppsf_90D_global'].fillna(df['ppsf_365D_global']).fillna(df['sold_ppsf'].median())
        
        # Calculate baseline and residual
        df['price_baseline'] = df['ppsf_baseline'] * df[self.AREA_COL]
        df = df[df['price_baseline'].gt(0)].copy()
        df['log_price_ratio'] = np.log(df[self.TARGET_COL]) - np.log(df['price_baseline'])
        
        self.df = df
    
    def _create_splits(self, df):
        """Create temporal train/val/test splits"""
        print("\nCreating temporal splits...")
        
        n = len(df)
        train_end = int(0.7 * n)
        val_end = int(0.8 * n)
        
        self.df_train = df.iloc[:train_end].copy()
        self.df_val = df.iloc[train_end:val_end].copy()
        self.df_test = df.iloc[val_end:].copy()
        
        print(f"Train: {len(self.df_train)} samples")
        print(f"Val: {len(self.df_val)} samples")
        print(f"Test: {len(self.df_test)} samples")
    
    def _prepare_features(self):
        """Prepare feature matrices"""
        print("\nPreparing features...")
        
        # Numeric features
        num_features = [self.AREA_COL, 'MaintFee', 'age_at_list', 'Full Baths', 'Half Baths',
                       'Tot BR', 'Tot Baths', 'No. Floor Levels', 'Storeys in Building',
                       'Fireplaces', 'Floor Area Fin - Total', 'Floor Area Fin - Main Flr',
                       'Floor Area Fin - Abv Main', 'Floor Area Fin - Basement',
                       'Floor Area - Unfinished', 'Tot Units in Strata Plan',
                       'Units in Development', 'month_idx', 'year', 'month', 'quarter',
                       'day_of_year', 'week_of_year', 'rooms_per_sqft', 'baths_per_br',
                       'maint_per_sqft', 'baths_equiv', 'ppsf_baseline']
        
        # Add missingness indicators
        num_features += [c for c in self.df.columns if c.endswith('_isna')]
        
        # Keep only existing
        num_features = [f for f in num_features if f in self.df.columns]
        
        # Simple one-hot for key categoricals
        cat_features = ['TypeDwel', 'FSA', 'Restricted Age', 'Title to Land']
        
        X_train_list = [self.df_train[num_features].fillna(0)]
        X_val_list = [self.df_val[num_features].fillna(0)]
        X_test_list = [self.df_test[num_features].fillna(0)]
        
        for cat in cat_features:
            if cat in self.df.columns:
                # Get frequent values
                top_values = self.df_train[cat].value_counts().head(20).index
                for val in top_values:
                    col_name = f'{cat}_{str(val)[:20]}'
                    X_train_list.append(pd.DataFrame({col_name: (self.df_train[cat] == val).astype(float)}))
                    X_val_list.append(pd.DataFrame({col_name: (self.df_val[cat] == val).astype(float)}))
                    X_test_list.append(pd.DataFrame({col_name: (self.df_test[cat] == val).astype(float)}))
        
        self.X_train = pd.concat(X_train_list, axis=1)
        self.X_val = pd.concat(X_val_list, axis=1)
        self.X_test = pd.concat(X_test_list, axis=1)
        
        self.y_train = self.df_train['log_price_ratio'].values
        self.y_val = self.df_val['log_price_ratio'].values
        self.y_test = self.df_test['log_price_ratio'].values
        
        print(f"Feature count: {self.X_train.shape[1]}")
        
        # Sample weights
        def recency_weights(dates, half_life_days=365):
            max_date = dates.max()
            age = (max_date - dates).dt.days.clip(lower=0)
            return np.power(0.5, age / half_life_days)
        
        w_rec = recency_weights(self.df_train[self.DATE_COL])
        w_wape = 1.0 / np.maximum(self.df_train[self.TARGET_COL].values, 1000.0)
        self.sample_weights = (w_rec * w_wape) / np.mean(w_rec * w_wape)
    
    def train_distributed(self, num_gpus=2):
        """Train using Ray's distributed XGBoost"""
        print("\n" + "="*60)
        print("DISTRIBUTED TRAINING WITH RAY")
        print("="*60)
        
        # Prepare Ray datasets
        train_dataset = ray.data.from_pandas(
            pd.concat([
                pd.DataFrame(self.X_train.values, columns=[f'f{i}' for i in range(self.X_train.shape[1])]),
                pd.DataFrame({'label': self.y_train, 'weight': self.sample_weights})
            ], axis=1)
        )
        
        val_dataset = ray.data.from_pandas(
            pd.concat([
                pd.DataFrame(self.X_val.values, columns=[f'f{i}' for i in range(self.X_val.shape[1])]),
                pd.DataFrame({'label': self.y_val})
            ], axis=1)
        )
        
        # XGBoost parameters optimized for GPUs
        params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'max_depth': 10,  # Can go deeper with GPUs
            'min_child_weight': 3.0,
            'subsample': 0.9,
            'colsample_bytree': 0.85,
            'learning_rate': 0.05,
            'reg_lambda': 1.5,
            'reg_alpha': 0.5,
            'gamma': 0.1,
            'objective': 'reg:squarederror',
            'eval_metric': ['mae', 'rmse'],
            'seed': 42
        }
        
        # Configure Ray trainer
        trainer = XGBoostTrainer(
            params=params,
            label_column='label',
            datasets={"train": train_dataset, "valid": val_dataset},
            num_boost_round=2000,
            scaling_config=ScalingConfig(
                num_workers=num_gpus,  # Use both GPUs
                use_gpu=True,
                resources_per_worker={"GPU": 1}
            ),
        )
        
        print(f"Training on {num_gpus} GPUs...")
        result = trainer.fit()
        
        # Get the trained model
        self.model = result.checkpoint.get_model()
        
        print(f"Training complete. Best iteration: {self.model.best_iteration}")
        
        return self
    
    def train_single_gpu(self):
        """Fallback to single GPU training if Ray not available"""
        print("\n" + "="*60)
        print("SINGLE GPU TRAINING")
        print("="*60)
        
        dtrain = xgb.DMatrix(self.X_train.values, label=self.y_train, weight=self.sample_weights)
        dval = xgb.DMatrix(self.X_val.values, label=self.y_val)
        
        params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0,
            'max_depth': 8,
            'min_child_weight': 6.0,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'learning_rate': 0.05,
            'reg_lambda': 2.0,
            'reg_alpha': 0.0,
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'seed': 42
        }
        
        self.model = xgb.train(
            params, dtrain,
            num_boost_round=2000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=50
        )
        
        return self
    
    def evaluate(self):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        # Make predictions
        dtest = xgb.DMatrix(self.X_test.values)
        test_pred_log_ratio = self.model.predict(dtest)
        
        # Apply calibration (fit on validation)
        dval = xgb.DMatrix(self.X_val.values)
        val_pred_log_ratio = self.model.predict(dval)
        val_true_log_ratio = self.y_val
        
        a, b = np.polyfit(val_pred_log_ratio, val_true_log_ratio, 1)
        test_pred_log_ratio_cal = a + b * test_pred_log_ratio
        
        # Convert to prices
        self.df_test['pred_price'] = self.df_test['price_baseline'] * np.exp(test_pred_log_ratio_cal)
        
        # Calculate metrics
        test_mae = mean_absolute_error(self.df_test[self.TARGET_COL], self.df_test['pred_price'])
        test_r2 = r2_score(self.df_test[self.TARGET_COL], self.df_test['pred_price'])
        test_wape = wape(self.df_test[self.TARGET_COL].values, self.df_test['pred_price'].values)
        
        # Baseline metrics
        base_mae = mean_absolute_error(self.df_test[self.TARGET_COL], self.df_test['price_baseline'])
        base_wape = wape(self.df_test[self.TARGET_COL].values, self.df_test['price_baseline'].values)
        
        print("\nTEST SET RESULTS:")
        print(f"Baseline → MAE ${base_mae:,.0f} | WAPE {base_wape:.3f}")
        print(f"Model    → MAE ${test_mae:,.0f} | R² {test_r2:.3f} | WAPE {test_wape:.3f}")
        print(f"Improvement: {(base_mae - test_mae)/base_mae*100:.1f}% reduction in MAE")
        
        return {
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_wape': test_wape,
            'base_mae': base_mae,
            'base_wape': base_wape
        }


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed Townhouse Price Model')
    parser.add_argument('--ray-address', type=str, default=None,
                       help='Ray cluster address (e.g., ray://192.168.1.100:10001)')
    parser.add_argument('--num-gpus', type=int, default=2,
                       help='Number of GPUs to use')
    parser.add_argument('--single-gpu', action='store_true',
                       help='Use single GPU training instead of distributed')
    
    args = parser.parse_args()
    
    # Initialize model
    model = DistributedTownhouseModel(ray_address=args.ray_address)
    
    # Load and prepare data
    model.load_and_prepare_data()
    
    # Train
    if args.single_gpu:
        model.train_single_gpu()
    else:
        try:
            model.train_distributed(num_gpus=args.num_gpus)
        except Exception as e:
            print(f"Distributed training failed: {e}")
            print("Falling back to single GPU...")
            model.train_single_gpu()
    
    # Evaluate
    metrics = model.evaluate()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'ray_model_results_{timestamp}.txt'
    with open(results_file, 'w') as f:
        f.write(f"Test MAE: ${metrics['test_mae']:,.0f}\n")
        f.write(f"Test R²: {metrics['test_r2']:.3f}\n")
        f.write(f"Test WAPE: {metrics['test_wape']:.3f}\n")
        f.write(f"Baseline MAE: ${metrics['base_mae']:,.0f}\n")
        f.write(f"Baseline WAPE: ${metrics['base_wape']:.3f}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()