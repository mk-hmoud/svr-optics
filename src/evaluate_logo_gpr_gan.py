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

from src.data import load_data, preprocess_data, get_logo_folds
from src.wgan_paper import train_wgan_paper, generate_samples_paper

def evaluate_logo_gpr_gan():
    """
    Performs 9-fold LOGO cross-validation for GPR + GAN.
    This replicates the paper's testing strategy but specifically for the augmented GPR.
    """
    print("Loading data...")
    df = load_data('data/data.xlsx')
    
    # We use all features used in the paper (7 features + 1 target = 8 columns)
    feature_cols = ['Analyte', 'Re(eff)', 'lambda', 'Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    X = df[feature_cols]
    y = np.log10(np.clip(df['loss'] * 10**8, a_min=1e-10, a_max=None))
    
    # Identify geometric configuration for grouping
    config_cols = ['Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    df_configs = df[config_cols].drop_duplicates().reset_index(drop=True)
    df_configs['group_id'] = range(len(df_configs))
    df = df.merge(df_configs, on=config_cols, how='left')
    groups = df['group_id'].values
    
    # Scaling
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    logo = get_logo_folds(X_scaled_df, y, groups)
    
    # GPR Kernel configuration from the paper
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
    
    results = []
    
    print(f"\n{'Fold':<5} | {'GPR+GAN MSE':<12}")
    print("-" * 22)
    
    for fold, (train_idx, test_idx) in enumerate(logo, 1):
        X_train, X_test = X_scaled_df.iloc[train_idx], X_scaled_df.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 1. Train GAN on this fold's training data (8 columns: 7 features + 1 target)
        real_train_combined = np.hstack([X_train.values, y_train.values.reshape(-1, 1)])
        # 1000 epochs is enough to see the GAN noise failure while saving time
        generator = train_wgan_paper(real_train_combined, epochs=1000) 
        
        # 2. Generate Augmented Data (1000 samples)
        synthetic_data = generate_samples_paper(generator, num_samples=1000)
        # Clip to physical bounds
        synthetic_data = np.clip(synthetic_data, real_train_combined.min(axis=0), real_train_combined.max(axis=0))
        
        X_aug = np.vstack([X_train.values, synthetic_data[:, :-1]])
        y_aug = np.concatenate([y_train, synthetic_data[:, -1]])
        
        # 3. Train GPR + GAN
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.0, random_state=42)
        
        try:
            gpr.fit(X_aug, y_aug)
            preds = gpr.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            print(f"{fold:<5} | {mse:<12.6f}")
            results.append({'Fold': fold, 'GPR_GAN_MSE': mse})
        except Exception as e:
            print(f"{fold:<5} | FAILED: {str(e)[:15]}")
            results.append({'Fold': fold, 'GPR_GAN_MSE': np.nan})

    res_df = pd.DataFrame(results)
    avg_mse = res_df['GPR_GAN_MSE'].mean()
    
    print("-" * 22)
    print(f"{'AVG':<5} | {avg_mse:<12.6f}")
    
    res_df.to_csv('results_logo_gpr_gan.csv', index=False)
    print("\nFull fold results saved to 'results_logo_gpr_gan.csv'")

if __name__ == "__main__":
    evaluate_logo_gpr_gan()
