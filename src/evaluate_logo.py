import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from src.data import load_data, preprocess_data, get_logo_folds
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_best_svr_bayesian(X_train, y_train):
    """Finds optimal SVR hyperparameters using Bayesian Optimization."""
    search_spaces = {
        'C': Real(1, 2000, prior='log-uniform'),
        'gamma': Real(1e-4, 1e0, prior='log-uniform'),
        'epsilon': Real(1e-4, 1e-1, prior='log-uniform'),
        'kernel': Categorical(['rbf'])
    }
    opt = BayesSearchCV(SVR(), search_spaces, n_iter=32, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', random_state=42)
    opt.fit(X_train, y_train)
    return opt.best_estimator_

def train_best_gpr(X_train, y_train):
    """Trains a Gaussian Process Regressor with an optimized kernel."""
    # ConstantKernel * RBF + WhiteKernel (to handle noise)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gpr.fit(X_train, y_train)
    return gpr

def evaluate_logo_comparison():
    """Performs 9-fold LOGO cross-validation for SVR and GPR."""
    print("Loading and preprocessing data...")
    df = load_data()
    X, y, groups, scaler = preprocess_data(df)
    logo = get_logo_folds(X, y, groups)
    
    results = []
    
    print(f"\n{'Fold':<5} | {'SVR MSE':<12} | {'GPR MSE':<12}")
    print("-" * 40)
    
    for fold, (train_idx, test_idx) in enumerate(logo, 1):
        X_train_raw, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 1. Baseline SVR (Bayesian Optimized)
        svr = train_best_svr_bayesian(X_train_raw, y_train_raw)
        mse_svr = mean_squared_error(y_test, svr.predict(X_test))
        
        # 2. Gaussian Process Regression (GPR)
        gpr = train_best_gpr(X_train_raw, y_train_raw)
        mse_gpr = mean_squared_error(y_test, gpr.predict(X_test))
        
        print(f"{fold:<5} | {mse_svr:<12.6f} | {mse_gpr:<12.6f}")
        results.append({'Fold': fold, 'SVR_MSE': mse_svr, 'GPR_MSE': mse_gpr})
        
    res_df = pd.DataFrame(results)
    
    # Calculate means and stds for the numeric columns only
    numeric_results = res_df.drop(columns=['Fold'])
    avg = numeric_results.mean()
    std = numeric_results.std()
    
    print("-" * 40)
    print(f"{'AVG':<5} | {avg['SVR_MSE']:<12.6f} | {avg['GPR_MSE']:<12.6f}")
    
    # Append summary rows
    res_df.loc[len(res_df)] = ['Average', avg['SVR_MSE'], avg['GPR_MSE']]
    res_df.loc[len(res_df)] = ['Std Dev', std['SVR_MSE'], std['GPR_MSE']]
    
    res_df.to_csv('results_comparison_svr_gpr.csv', index=False)
    print("\nResults saved to 'results_comparison_svr_gpr.csv'")

if __name__ == '__main__':
    evaluate_logo_comparison()
