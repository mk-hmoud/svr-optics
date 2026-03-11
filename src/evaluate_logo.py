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

def train_best_gpr(X_train, y_train):
    """Trains a Gaussian Process Regressor with an optimized kernel."""
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gpr.fit(X_train, y_train)
    return gpr

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

def train_best_xgb(X_train, y_train):
    """Trains an XGBoost Regressor using Bayesian Optimization."""
    search_spaces = {
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(1e-3, 0.3, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0)
    }
    opt = BayesSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        search_spaces,
        n_iter=20,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    opt.fit(X_train, y_train)
    return opt.best_estimator_

def evaluate_logo_comparison():
    """Performs 9-fold LOGO cross-validation for multiple models."""
    print("Loading and preprocessing data...")
    df = load_data()
    X, y, groups, scaler = preprocess_data(df)
    logo = get_logo_folds(X, y, groups)
    
    results = []
    
    print(f"\n{'Fold':<5} | {'SVR MSE':<10} | {'GPR MSE':<10} | {'RF MSE':<10} | {'XGB MSE':<10}")
    print("-" * 65)
    
    for fold, (train_idx, test_idx) in enumerate(logo, 1):
        X_train_raw, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 1. Baseline SVR
        svr = train_best_svr_grid(X_train_raw, y_train_raw)
        mse_svr = mean_squared_error(y_test, svr.predict(X_test))
        
        # 2. Gaussian Process Regression (GPR)
        gpr = train_best_gpr(X_train_raw, y_train_raw)
        mse_gpr = mean_squared_error(y_test, gpr.predict(X_test))
        
        # 3. Random Forest (RF)
        rf = train_best_rf(X_train_raw, y_train_raw)
        mse_rf = mean_squared_error(y_test, rf.predict(X_test))
        
        # 4. XGBoost (XGB)
        xg_model = train_best_xgb(X_train_raw, y_train_raw)
        mse_xgb = mean_squared_error(y_test, xg_model.predict(X_test))
        
        print(f"{fold:<5} | {mse_svr:<10.6f} | {mse_gpr:<10.6f} | {mse_rf:<10.6f} | {mse_xgb:<10.6f}")
        results.append({'Fold': fold, 'SVR_MSE': mse_svr, 'GPR_MSE': mse_gpr, 'RF_MSE': mse_rf, 'XGB_MSE': mse_xgb})
        
    res_df = pd.DataFrame(results)
    numeric_results = res_df.drop(columns=['Fold'])
    avg = numeric_results.mean()
    std = numeric_results.std()
    
    print("-" * 65)
    print(f"{'AVG':<5} | {avg['SVR_MSE']:<10.6f} | {avg['GPR_MSE']:<10.6f} | {avg['RF_MSE']:<10.6f} | {avg['XGB_MSE']:<10.6f}")
    
    res_df.loc[len(res_df)] = ['Average', avg['SVR_MSE'], avg['GPR_MSE'], avg['RF_MSE'], avg['XGB_MSE']]
    res_df.loc[len(res_df)] = ['Std Dev', std['SVR_MSE'], std['GPR_MSE'], std['RF_MSE'], std['XGB_MSE']]
    
    res_df.to_csv('results_comparison_final.csv', index=False)
    print("\nResults saved to 'results_comparison_final.csv'")

if __name__ == '__main__':
    evaluate_logo_comparison()
