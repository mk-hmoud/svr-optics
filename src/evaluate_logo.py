import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.data import load_data, preprocess_data, get_logo_folds
from src.models.researcher_ann import build_researcher_ann, train_ann
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def evaluate_logo_ann_only():
    """
    Performs 9-fold LOGO cross-validation specifically for the Researcher's ANN.
    Reduced model scope to save memory.
    """
    print("Loading and preprocessing data...")
    df = load_data()
    X, y, groups, scaler = preprocess_data(df)
    logo = get_logo_folds(X, y, groups)
    
    results = []
    
    print(f"\n{'Fold':<5} | {'ANN MSE':<10}")
    print("-" * 20)
    
    for fold, (train_idx, test_idx) in enumerate(logo, 1):
        X_train_raw, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Split train into train and val for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_raw, y_train_raw, test_size=0.1, random_state=42)
        
        # Build and train ANN
        ann = build_researcher_ann(input_dim=X.shape[1])
        train_ann(ann, X_tr, y_tr, X_val, y_val, epochs=500) # Further reduced epochs for memory safety
        
        mse_ann = mean_squared_error(y_test, ann.predict(X_test, verbose=0).flatten())
        
        print(f"{fold:<5} | {mse_ann:<10.6f}")
        results.append({'Fold': fold, 'ANN_MSE': mse_ann})
        
        # Explicitly clear Keras session to free memory after each fold
        import tensorflow as tf
        tf.keras.backend.clear_session()
        
    res_df = pd.DataFrame(results)
    avg = res_df['ANN_MSE'].mean()
    
    print("-" * 20)
    print(f"{'AVG':<5} | {avg:<10.6f}")
    
    res_df.to_csv('results_comparison_researcher_ann_only.csv', index=False)
    print("\nResults saved to 'results_comparison_researcher_ann_only.csv'")

if __name__ == '__main__':
    evaluate_logo_ann_only()
