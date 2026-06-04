import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_data, get_logo_folds
from src.wgan_paper import train_wgan_paper, generate_samples_paper

def evaluate_logo_gpr_gan_once():
    print("Loading data...")
    df = load_data('data/data.xlsx')
    
    # 1. Standard Preparation (Matches paper features)
    feature_cols = ['Analyte', 'Re(eff)', 'lambda', 'Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    X = df[feature_cols]
    y = np.log10(np.clip(df['loss'] * 10**8, a_min=1e-10, a_max=None))
    
    # Identify geometric configuration for grouping
    config_cols = ['Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    df_configs = df[config_cols].drop_duplicates().reset_index(drop=True)
    df_configs['group_id'] = range(len(df_configs))
    df_merged = df.merge(df_configs, on=config_cols, how='left')
    groups = df_merged['group_id'].values
    
    # Scaling
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # 2. Train GAN ONCE on Designs 1-7
    print("\n--- Training GAN Once (Designs 1-7) ---")
    train_indices_gan = df_merged[df_merged['group_id'] < 7].index # Groups 0-6
    X_train_gan = X_scaled_df.iloc[train_indices_gan]
    y_train_gan = y.iloc[train_indices_gan]
    
    real_train_combined = np.hstack([X_train_gan.values, y_train_gan.values.reshape(-1, 1)])
    generator = train_wgan_paper(real_train_combined, epochs=2500)
    
    print("\nGenerating 1000 synthetic samples...")
    synthetic_data = generate_samples_paper(generator, num_samples=1000)
    # Clip to physical bounds
    synthetic_data = np.clip(synthetic_data, real_train_combined.min(axis=0), real_train_combined.max(axis=0))
    
    X_synth = synthetic_data[:, :-1]
    y_synth = synthetic_data[:, -1]
    
    # 3. 9-Fold LOGO Evaluation
    logo = get_logo_folds(X_scaled_df, y, groups)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
    
    results = []
    print(f"\n{'Fold':<5} | {'Test Design':<12} | {'GPR+GAN MSE':<12}")
    print("-" * 35)
    
    for fold, (train_idx, test_idx) in enumerate(logo, 1):
        X_train_real, X_test = X_scaled_df.iloc[train_idx], X_scaled_df.iloc[test_idx]
        y_train_real, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Augment this fold's training data with the SAME synthetic data
        X_aug = np.vstack([X_train_real.values, X_synth])
        y_aug = np.concatenate([y_train_real.values, y_synth])
        
        # Train GPR
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.0, random_state=42)
        
        try:
            gpr.fit(X_aug, y_aug)
            preds = gpr.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            print(f"{fold:<5} | Design {fold:<7} | {mse:<12.6f}")
            results.append(mse)
        except Exception as e:
            print(f"{fold:<5} | Design {fold:<7} | FAILED: {str(e)[:10]}")
            results.append(np.nan)

    print("-" * 35)
    print(f"{'AVG':<18} | {np.nanmean(results):<12.6f}")

if __name__ == "__main__":
    evaluate_logo_gpr_gan_once()
