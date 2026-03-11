import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from src.data import load_data, preprocess_data, get_logo_folds
from src.data_augmentation import augment_with_wgan
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_best_svr_grid(X_train, y_train):
    """Finds optimal SVR hyperparameters using Grid Search."""
    param_grid = {
        'C': [100, 500, 1000, 2000],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'epsilon': [0.01, 0.1],
        'kernel': ['rbf']
    }
    grid = GridSearchCV(SVR(), param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_best_rf(X_train, y_train):
    """Trains a Random Forest Regressor using Grid Search."""
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'random_state': [42]
    }
    grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def evaluate_logo_comparison(gan_epochs=1000, num_synthetic=500):
    """
    Performs 9-fold LOGO cross-validation for:
    SVR, RF (Baseline), and RF (GAN-Augmented).
    """
    print("Loading and preprocessing data...")
    df = load_data()
    X, y, groups, scaler = preprocess_data(df)
    logo = get_logo_folds(X, y, groups)
    
    results = []
    
    print(f"\n{'Fold':<5} | {'SVR MSE':<10} | {'RF MSE':<10} | {'RF+GAN MSE':<12}")
    print("-" * 55)
    
    for fold, (train_idx, test_idx) in enumerate(logo, 1):
        X_train_raw, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 1. Baseline SVR
        svr = train_best_svr_grid(X_train_raw, y_train_raw)
        mse_svr = mean_squared_error(y_test, svr.predict(X_test))
        
        # 2. Baseline Random Forest
        rf_base = train_best_rf(X_train_raw, y_train_raw)
        mse_rf_base = mean_squared_error(y_test, rf_base.predict(X_test))
        
        # 3. GAN-Augmented Random Forest
        # We train our WGAN-GP on the current training fold
        X_train_aug, y_train_aug = augment_with_wgan(
            X_train_raw, y_train_raw, 
            num_synthetic_samples=num_synthetic, 
            epochs=gan_epochs
        )
        rf_aug = train_best_rf(X_train_aug, y_train_aug)
        mse_rf_aug = mean_squared_error(y_test, rf_aug.predict(X_test))
        
        print(f"{fold:<5} | {mse_svr:<10.6f} | {mse_rf_base:<10.6f} | {mse_rf_aug:<12.6f}")
        results.append({
            'Fold': fold, 
            'SVR_MSE': mse_svr, 
            'RF_Baseline_MSE': mse_rf_base, 
            'RF_GAN_MSE': mse_rf_aug
        })
        
    res_df = pd.DataFrame(results)
    numeric_results = res_df.drop(columns=['Fold'])
    avg = numeric_results.mean()
    
    print("-" * 55)
    print(f"{'AVG':<5} | {avg['SVR_MSE']:<10.6f} | {avg['RF_Baseline_MSE']:<10.6f} | {avg['RF_GAN_MSE']:<12.6f}")
    
    res_df.to_csv('results_gan_rf_comparison.csv', index=False)
    print("\nResults saved to 'results_gan_rf_comparison.csv'")

if __name__ == '__main__':
    # Using 500 epochs for the test run to keep it reasonably fast
    evaluate_logo_comparison(gan_epochs=500, num_synthetic=500)
