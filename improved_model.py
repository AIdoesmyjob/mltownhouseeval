#!/usr/bin/env python3
"""
Improved Townhouse Price Prediction Model
- Single strong model instead of brute-force feature search
- Time and location features
- Comp-style rolling PPSF baseline
- Proper temporal validation
- Log-residual target for better WAPE alignment
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

class TownhousePriceModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.feature_cols = None
        
    def load_and_clean_data(self):
        """Load CSV and perform initial cleaning"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        
        # Convert numeric columns
        num_cols = ['Price', 'List Price', 'TotFlArea', 'MaintFee', 'Yr Blt', 'Age',
                    'Full Baths', 'Half Baths', 'Tot BR', 'Tot Baths', 'No. Floor Levels',
                    'Storeys in Building', 'Fireplaces', 'Floor Area Fin - Total',
                    'Floor Area Fin - Main Flr', 'Floor Area Fin - Abv Main',
                    'Floor Area Fin - Basement', 'Floor Area - Unfinished',
                    'Tot Units in Strata Plan', 'Units in Development',
                    '# of Kitchens', '# of Pets', 'DOM', 'Cumulative DOM']
        
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = self._to_numeric(self.df[col])
        
        # Parse dates
        if 'List Date' in self.df.columns:
            self.df['List Date'] = pd.to_datetime(self.df['List Date'], errors='coerce')
        if 'Sold Date' in self.df.columns:
            self.df['Sold Date'] = pd.to_datetime(self.df['Sold Date'], errors='coerce')
        
        # Keep only valid sold prices
        self.df = self.df[self.df['Price'].notna() & (self.df['Price'] > 0)].copy()
        print(f"Loaded {len(self.df)} valid sales records")
        
    def _to_numeric(self, series):
        """Convert series to numeric, handling $, commas and spaces"""
        if pd.api.types.is_object_dtype(series):
            series = (series.astype(str)
                           .str.replace('$', '', regex=False)
                           .str.replace(',', '', regex=False)
                           .str.replace(' ', '', regex=False))
        return pd.to_numeric(series, errors='coerce')
    
    def add_time_features(self):
        """Add time-based features"""
        print("Adding time features...")
        
        # Reference date for month_idx
        ref_date = pd.Timestamp('2015-01-01')
        self.df['month_idx'] = (self.df['List Date'] - ref_date).dt.days / 30.0
        self.df['year'] = self.df['List Date'].dt.year.astype('float32')
        self.df['month'] = self.df['List Date'].dt.month.astype('float32')
        self.df['quarter'] = ((self.df['month'] - 1) // 3 + 1).astype('float32')
        
        # Age at listing
        self.df['age_at_list'] = np.where(
            self.df['Yr Blt'].notna(),
            self.df['year'] - self.df['Yr Blt'].clip(lower=1900, upper=self.df['year']),
            self.df['Age']
        )
        self.df['age_at_list'] = self.df['age_at_list'].clip(lower=0)
        
    def add_location_features(self):
        """Extract location features from postal code"""
        print("Adding location features...")
        
        if 'Postal Code' in self.df.columns:
            # Extract FSA (Forward Sortation Area) - first 3 chars of postal code
            self.df['FSA'] = (self.df['Postal Code'].astype(str)
                             .str.extract(r'^([A-Za-z]\d[A-Za-z])', expand=False))
        else:
            self.df['FSA'] = np.nan
            
    def build_comp_baseline(self):
        """Build rolling PPSF (price per square foot) baseline"""
        print("Building comp-style baseline...")
        
        # Sort by date for rolling calculations
        self.df = self.df.sort_values('List Date').reset_index(drop=True)
        
        # Calculate PPSF
        area = self.df['TotFlArea'].replace(0, np.nan)
        self.df['sold_ppsf'] = self.df['Price'] / area
        
        # Group columns for rolling median
        type_col = 'TypeDwel' if 'TypeDwel' in self.df.columns else None
        group_cols = ['FSA'] + ([type_col] if type_col else [])
        
        # Calculate rolling medians at different granularities
        self._add_rolling_medians(group_cols, type_col)
        
        # Choose best available baseline
        self._select_baseline_ppsf()
        
        # Calculate baseline price and residual
        self.df['price_baseline'] = self.df['ppsf_baseline'] * area
        
        # Filter to valid baseline prices
        mask = (self.df['price_baseline'].notna() & 
                (self.df['price_baseline'] > 0) & 
                area.notna())
        self.df = self.df[mask].copy()
        
        # Log residual target
        self.df['y_log_resid'] = (np.log(self.df['Price']) - 
                                  np.log(self.df['price_baseline']))
        
    def _add_rolling_medians(self, group_cols, type_col):
        """Calculate rolling median PPSF at different levels"""
        
        # Helper function for grouped rolling median
        def calc_rolling_median(df, groups, window, min_periods):
            if not groups or not all(g in df.columns for g in groups):
                return None
                
            tmp = df[['List Date'] + groups + ['sold_ppsf']].copy()
            tmp = tmp.set_index('List Date')
            
            try:
                med = (tmp.groupby(groups)['sold_ppsf']
                      .apply(lambda s: s.rolling(window, min_periods=min_periods)
                             .median().shift(1)))
                med.name = f'ppsf_{window}_{"_".join(groups)}_med'
                result = med.reset_index()
                # Ensure List Date is in the result
                if 'List Date' not in result.columns:
                    result = result.reset_index()
                return result
            except Exception as e:
                print(f"Warning: Could not calculate rolling median for {groups}: {e}")
                return None
        
        # FSA + TypeDwel level
        if all(c in self.df.columns for c in group_cols):
            med_90 = calc_rolling_median(self.df, group_cols, '90D', 8)
            med_365 = calc_rolling_median(self.df, group_cols, '365D', 12)
            if med_90 is not None:
                self.df = self.df.merge(med_90, left_on=['List Date']+group_cols, 
                                       right_on=['List Date']+group_cols, how='left')
            if med_365 is not None:
                self.df = self.df.merge(med_365, left_on=['List Date']+group_cols,
                                       right_on=['List Date']+group_cols, how='left')
        
        # FSA only level (fallback)
        if 'FSA' in self.df.columns:
            med_90_fsa = calc_rolling_median(self.df, ['FSA'], '90D', 8)
            med_365_fsa = calc_rolling_median(self.df, ['FSA'], '365D', 12)
            if med_90_fsa is not None:
                self.df = self.df.merge(med_90_fsa, on=['List Date', 'FSA'], 
                                       how='left', suffixes=('', '_fsa'))
            if med_365_fsa is not None:
                self.df = self.df.merge(med_365_fsa, on=['List Date', 'FSA'],
                                       how='left', suffixes=('', '_fsa365'))
        
        # Global rolling medians (ultimate fallback)
        tmp = self.df[['List Date', 'sold_ppsf']].set_index('List Date').sort_index()
        self.df['ppsf_90D_global'] = (tmp['sold_ppsf'].rolling('90D', min_periods=8)
                                      .median().shift(1).values)
        self.df['ppsf_365D_global'] = (tmp['sold_ppsf'].rolling('365D', min_periods=12)
                                       .median().shift(1).values)
        
    def _select_baseline_ppsf(self):
        """Select best available PPSF baseline from candidates"""
        
        # Priority order: specific to general
        candidates = []
        
        # Add group-level candidates
        for col in self.df.columns:
            if 'ppsf_90D' in col and col.endswith('_med'):
                candidates.append(col)
        for col in self.df.columns:
            if 'ppsf_365D' in col and col.endswith('_med'):
                candidates.append(col)
                
        # Add global candidates
        candidates.extend(['ppsf_90D_global', 'ppsf_365D_global'])
        
        # Select first non-null baseline
        self.df['ppsf_baseline'] = np.nan
        for cand in candidates:
            if cand in self.df.columns:
                self.df['ppsf_baseline'] = self.df['ppsf_baseline'].fillna(self.df[cand])
                
    def create_temporal_split(self, train_ratio=0.7, val_ratio=0.1):
        """Create time-based train/val/test split"""
        print("Creating temporal splits...")
        
        n = len(self.df)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)
        
        self.df_train = self.df.iloc[:train_end].copy()
        self.df_val = self.df.iloc[train_end:val_end].copy()
        self.df_test = self.df.iloc[val_end:].copy()
        
        print(f"Train: {len(self.df_train)} samples (oldest {train_ratio*100:.0f}%)")
        print(f"Val: {len(self.df_val)} samples (middle {val_ratio*100:.0f}%)")
        print(f"Test: {len(self.df_test)} samples (newest {100-train_ratio*100-val_ratio*100:.0f}%)")
        
    def prepare_features(self):
        """Prepare feature matrices with numeric and categorical encoding"""
        print("Preparing features...")
        
        # Numeric features
        num_features = ['TotFlArea', 'MaintFee', 'age_at_list', 'Full Baths', 'Half Baths',
                       'Tot BR', 'Tot Baths', 'No. Floor Levels', 'Storeys in Building',
                       'Fireplaces', 'Floor Area Fin - Total', 'Floor Area Fin - Main Flr',
                       'Floor Area Fin - Abv Main', 'Floor Area Fin - Basement',
                       'Floor Area - Unfinished', 'Tot Units in Strata Plan',
                       'Units in Development', 'month_idx', 'year', 'month', 'quarter']
        
        # Keep only existing columns
        num_features = [f for f in num_features if f in self.df.columns]
        
        # One-hot encode categorical features
        cat_features = ['TypeDwel', 'FSA', 'Restricted Age', 'Title to Land', 'Zoning']
        cat_features = [f for f in cat_features if f in self.df.columns]
        
        # Prepare training features
        X_train_num = self.df_train[num_features].fillna(0)
        X_val_num = self.df_val[num_features].fillna(0)
        X_test_num = self.df_test[num_features].fillna(0)
        
        # One-hot encoding for categoricals
        X_train_cat_list = []
        X_val_cat_list = []
        X_test_cat_list = []
        
        for cat in cat_features:
            # Create one-hot encoding on train
            train_dummies = pd.get_dummies(self.df_train[cat].fillna('NA'), 
                                          prefix=cat, dtype=float)
            X_train_cat_list.append(train_dummies)
            
            # Apply same encoding to val/test
            val_dummies = pd.get_dummies(self.df_val[cat].fillna('NA'), 
                                        prefix=cat, dtype=float)
            val_dummies = val_dummies.reindex(columns=train_dummies.columns, fill_value=0)
            X_val_cat_list.append(val_dummies)
            
            test_dummies = pd.get_dummies(self.df_test[cat].fillna('NA'),
                                         prefix=cat, dtype=float)
            test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
            X_test_cat_list.append(test_dummies)
        
        # Combine numeric and categorical
        self.X_train = pd.concat([X_train_num] + X_train_cat_list, axis=1)
        self.X_val = pd.concat([X_val_num] + X_val_cat_list, axis=1)
        self.X_test = pd.concat([X_test_num] + X_test_cat_list, axis=1)
        
        # Targets
        self.y_train = self.df_train['y_log_resid'].values
        self.y_val = self.df_val['y_log_resid'].values
        self.y_test = self.df_test['y_log_resid'].values
        
        print(f"Feature count: {self.X_train.shape[1]}")
        
    def add_sample_weights(self):
        """Calculate recency and WAPE-aligned weights"""
        print("Adding sample weights...")
        
        # Recency weights (half-life = 180 days)
        def calc_recency_weights(dates, half_life_days=180):
            max_date = dates.max()
            days_old = (max_date - dates).dt.days
            return np.power(0.5, days_old / half_life_days)
        
        recency_weights = calc_recency_weights(self.df_train['List Date'])
        
        # WAPE-aligned weights (inverse of price)
        wape_weights = 1.0 / np.maximum(self.df_train['Price'].values, 1000.0)
        
        # Combined weights
        self.sample_weights = recency_weights * wape_weights
        
    def train_model(self):
        """Train XGBoost model with proper hyperparameters"""
        print("Training XGBoost model...")
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(self.X_train.values, label=self.y_train, 
                            weight=self.sample_weights)
        dval = xgb.DMatrix(self.X_val.values, label=self.y_val)
        dtest = xgb.DMatrix(self.X_test.values)
        
        # XGBoost parameters
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
            'seed': 42,
            'verbosity': 1
        }
        
        # Train with early stopping
        self.model = xgb.train(
            params, dtrain,
            num_boost_round=2000,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=100
        )
        
        # Store test predictions
        self.dtest = dtest
        
    def evaluate(self):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Predict on validation and test
        val_pred_resid = self.model.predict(
            xgb.DMatrix(self.X_val.values),
            iteration_range=(0, self.model.best_iteration + 1)
        )
        test_pred_resid = self.model.predict(
            self.dtest,
            iteration_range=(0, self.model.best_iteration + 1)
        )
        
        # Convert residuals back to prices
        self.df_val['pred_price'] = self.df_val['price_baseline'] * np.exp(val_pred_resid)
        self.df_test['pred_price'] = self.df_test['price_baseline'] * np.exp(test_pred_resid)
        
        # Calculate metrics
        def wape(y_true, y_pred):
            return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
        
        # Validation metrics
        val_mae = mean_absolute_error(self.df_val['Price'], self.df_val['pred_price'])
        val_r2 = r2_score(self.df_val['Price'], self.df_val['pred_price'])
        val_wape = wape(self.df_val['Price'].values, self.df_val['pred_price'].values)
        
        # Test metrics
        test_mae = mean_absolute_error(self.df_test['Price'], self.df_test['pred_price'])
        test_r2 = r2_score(self.df_test['Price'], self.df_test['pred_price'])
        test_wape = wape(self.df_test['Price'].values, self.df_test['pred_price'].values)
        
        print("\n" + "="*60)
        print("VALIDATION SET (temporal middle 10%):")
        print(f"  MAE: ${val_mae:,.0f}")
        print(f"  R²: {val_r2:.3f}")
        print(f"  WAPE: {val_wape:.3f}")
        
        print("\nTEST SET (newest 20%):")
        print(f"  MAE: ${test_mae:,.0f}")
        print(f"  R²: {test_r2:.3f}")
        print(f"  WAPE: {test_wape:.3f}")
        print("="*60)
        
        return {
            'val_mae': val_mae, 'val_r2': val_r2, 'val_wape': val_wape,
            'test_mae': test_mae, 'test_r2': test_r2, 'test_wape': test_wape
        }
    
    def run_full_pipeline(self):
        """Run the complete modeling pipeline"""
        print("Starting improved townhouse price model...")
        print("="*60)
        
        # Execute pipeline steps
        self.load_and_clean_data()
        self.add_time_features()
        self.add_location_features()
        self.build_comp_baseline()
        self.create_temporal_split()
        self.prepare_features()
        self.add_sample_weights()
        self.train_model()
        metrics = self.evaluate()
        
        print("\nPipeline complete!")
        return metrics


def main():
    # Path to data
    csv_path = "Jan 1 2015_Aug 13 2025.csv"
    
    # Create and run model
    model = TownhousePriceModel(csv_path)
    metrics = model.run_full_pipeline()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'improved_model_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("Improved Model Results\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Validation Metrics:\n")
        f.write(f"  MAE: ${metrics['val_mae']:,.0f}\n")
        f.write(f"  R²: {metrics['val_r2']:.3f}\n")
        f.write(f"  WAPE: {metrics['val_wape']:.3f}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"  MAE: ${metrics['test_mae']:,.0f}\n")
        f.write(f"  R²: {metrics['test_r2']:.3f}\n")
        f.write(f"  WAPE: {metrics['test_wape']:.3f}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()