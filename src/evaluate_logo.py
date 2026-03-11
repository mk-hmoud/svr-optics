import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from src.data import load_data, preprocess_data, get_logo_folds
from src.data_augmentation import augment_with_wgan, load_researcher_data
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_best_svr_bayesian(X_train, y_train):
    """
    Finds optimal SVR hyperparameters using Bayesian Optimization.
    Bayesian optimization is more efficient and intelligent than Grid Search.
    """
    search_spaces = {
        'C': Real(1, 2000, prior='log-uniform'),
        'gamma': Real(1e-4, 1e0, prior='log-uniform'),
        'epsilon': Real(1e-4, 1e-1, prior='log-uniform'),
        'kernel': Categorical(['rbf']) # Focus on RBF for consistency with paper
    }
    
    # 32 iterations is often enough for Bayesian Opt to find a global optimum
    opt = BayesSearchCV(
        SVR(),
        search_spaces,
        n_iter=32, 
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    
    opt.fit(X_train, y_train)
    return opt.best_estimator_

def evaluate_logo_comparison(use_researcher_data=True, num_synthetic_samples=1000, epochs=2000):
    """
    Performs 9-fold Leave-One-Group-Out cross-validation with Bayesian Optimization.
    """
    print("Loading and preprocessing data...")
    df = load_data()
    X, y, groups, scaler = preprocess_data(df)
    
    logo = get_logo_folds(X, y, groups)
    
    baseline_mses = []
    augmented_mses = []
    
    label = "Researcher-Gen" if use_researcher_data else "WGAN-GP"
    print(f"\n{'Fold':<5} | {'Baseline MSE':<15} | {f'Augmented MSE ({label})':<20}")
    print("-" * 65)
    
    # Pre-load researcher data if needed
    if use_researcher_data:
        X_res, y_res = load_researcher_data('data/gen_data.txt', feature_columns=X.columns)
        if X_res is None:
            print("Researcher data not found. Falling back to WGAN.")
            use_researcher_data = False
    
    # Iterate through each of the 9 configurations as the test set
    for fold, (train_idx, test_idx) in enumerate(logo, 1):
        X_train_raw, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 1. Baseline SVR (Bayesian Optimized)
        svr_baseline = train_best_svr_bayesian(X_train_raw, y_train_raw)
        mse_base = mean_squared_error(y_test, svr_baseline.predict(X_test))
        baseline_mses.append(mse_base)
        
        # 2. Augmented SVR (Bayesian Optimized)
        if use_researcher_data:
            X_train_aug = pd.concat([X_train_raw, X_res], ignore_index=True)
            y_train_aug = pd.concat([y_train_raw, y_res], ignore_index=True)
        else:
            X_train_aug, y_train_aug = augment_with_wgan(
                X_train_raw, y_train_raw, 
                num_synthetic_samples=num_synthetic_samples, 
                epochs=epochs
            )
        
        svr_augmented = train_best_svr_bayesian(X_train_aug, y_train_aug)
        mse_aug = mean_squared_error(y_test, svr_augmented.predict(X_test))
        augmented_mses.append(mse_aug)
        
        print(f"{fold:<5} | {mse_base:<15.6f} | {mse_aug:<20.6f}")
        
    avg_base = np.mean(baseline_mses)
    avg_aug = np.mean(augmented_mses)
    std_base = np.std(baseline_mses)
    std_aug = np.std(augmented_mses)
    
    print("-" * 65)
    print(f"{'AVG':<5} | {avg_base:<15.6f} | {avg_aug:<20.6f}")
    print(f"{'STD':<5} | {std_base:<15.6f} | {std_aug:<20.6f}")
    
    # Save results
    results_df = pd.DataFrame({
        'Fold': list(range(1, 10)) + ['Average', 'Std Dev'],
        'Baseline_MSE': baseline_mses + [avg_base, std_base],
        'Augmented_MSE': augmented_mses + [avg_aug, std_aug]
    })
    results_df.to_csv('results_comparison_bayesian.csv', index=False)
    print("\nResults saved to 'results_comparison_bayesian.csv'")

if __name__ == '__main__':
    evaluate_logo_comparison(use_researcher_data=True)
