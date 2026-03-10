import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.data import load_data, preprocess_data, get_logo_folds
from src.data_augmentation import augment_with_wgan
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_best_svr(X_train, y_train):
    """
    Finds optimal SVR hyperparameters for a specific training set.
    We use a focused grid to keep the 9-fold CV reasonably fast.
    """
    param_grid = {
        'kernel': ['rbf'], 
        'C': [1, 10, 100, 500],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.01, 0.1]
    }
    # Using 3-fold CV for the inner search to save time
    grid = GridSearchCV(SVR(), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def evaluate_logo_comparison(num_synthetic_samples=1000, epochs=2000):
    """
    Performs 9-fold Leave-One-Group-Out cross-validation for both 
    Baseline SVR and Augmented SVR (WGAN-GP), optimizing hyperparameters for each.
    """
    print("Loading and preprocessing data...")
    df = load_data()
    X, y, groups, scaler = preprocess_data(df)
    
    logo = get_logo_folds(X, y, groups)
    
    baseline_mses = []
    augmented_mses = []
    
    print(f"\n{'Fold':<5} | {'Baseline MSE':<15} | {'Augmented MSE (WGAN)':<15}")
    print("-" * 55)
    
    # Iterate through each of the 9 configurations as the test set
    for fold, (train_idx, test_idx) in enumerate(logo, 1):
        X_train_raw, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 1. Baseline SVR (Optimized on original training configurations)
        svr_baseline = train_best_svr(X_train_raw, y_train_raw)
        y_pred_base = svr_baseline.predict(X_test)
        mse_base = mean_squared_error(y_test, y_pred_base)
        baseline_mses.append(mse_base)
        
        # 2. Augmented SVR (Optimized on original + WGAN-GP synthetic data)
        X_train_aug, y_train_aug = augment_with_wgan(
            X_train_raw, y_train_raw, 
            num_synthetic_samples=num_synthetic_samples, 
            epochs=epochs
        )
        
        svr_augmented = train_best_svr(X_train_aug, y_train_aug)
        y_pred_aug = svr_augmented.predict(X_test)
        mse_aug = mean_squared_error(y_test, y_pred_aug)
        augmented_mses.append(mse_aug)
        
        print(f"{fold:<5} | {mse_base:<15.6f} | {mse_aug:<15.6f}")
        
    avg_base = np.mean(baseline_mses)
    avg_aug = np.mean(augmented_mses)
    std_base = np.std(baseline_mses)
    std_aug = np.std(augmented_mses)
    
    print("-" * 55)
    print(f"{'AVG':<5} | {avg_base:<15.6f} | {avg_aug:<15.6f}")
    print(f"{'STD':<5} | {std_base:<15.6f} | {std_aug:<15.6f}")
    
    # Save results for future plotting/paper
    results_df = pd.DataFrame({
        'Fold': list(range(1, 10)) + ['Average', 'Std Dev'],
        'Baseline_MSE': baseline_mses + [avg_base, std_base],
        'Augmented_MSE': augmented_mses + [avg_aug, std_aug]
    })
    results_df.to_csv('results_comparison.csv', index=False)
    print("\nFull results saved to 'results_comparison.csv'")

if __name__ == '__main__':
    # Using fewer epochs for the initial test to save time, 
    # but 2000+ is recommended for final results.
    evaluate_logo_comparison(num_synthetic_samples=1000, epochs=100)
