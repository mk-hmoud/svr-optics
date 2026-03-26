import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_data
from src.wgan_paper import train_wgan_paper, generate_samples_paper
import torch

def evaluate_paper_gan():
    # 1. Load and Prepare Data (Using all 7 features + Target)
    df = load_data('data/data.xlsx')
    feature_cols = ['Analyte', 'Re(eff)', 'lambda', 'Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    X = df[feature_cols]
    y = np.log10(np.clip(df['loss'] * 10**8, a_min=1e-10, a_max=None))
    
    # 7-1-1 Split as per Methodology
    config_cols = ['Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    unique_configs = df[config_cols].drop_duplicates().values.tolist()
    
    def get_config_id(row):
        for i, config in enumerate(unique_configs):
            if all(row[config_cols] == config): return i + 1
        return None
    
    df['config_id'] = df.apply(get_config_id, axis=1)
    train_indices = df[df['config_id'] <= 7].index
    val_indices = df[df['config_id'] == 8].index
    test_indices = df[df['config_id'] == 9].index
    
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    
    # Scaling
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 2. Train Paper GAN
    print("--- Training GAN from Research Paper ---")
    real_train_combined = np.hstack([X_train_scaled, y_train.values.reshape(-1, 1)])
    generator = train_wgan_paper(real_train_combined, epochs=2500)
    
    # 3. Generate Augmented Data
    print("--- Generating Augmented Samples ---")
    synthetic_data = generate_samples_paper(generator, num_samples=1000)
    # Clip to physical bounds of training data
    synthetic_data = np.clip(synthetic_data, real_train_combined.min(axis=0), real_train_combined.max(axis=0))
    
    X_synth = synthetic_data[:, :-1]
    y_synth = synthetic_data[:, -1]
    
    X_aug = np.vstack([X_train_scaled, X_synth])
    y_aug = np.concatenate([y_train, y_synth])
    
    # 4. Evaluate Models
    results = {}
    
    # SVR Baseline
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    svr.fit(X_train_scaled, y_train)
    results['SVR_Baseline'] = mean_squared_error(y_test, svr.predict(X_test_scaled))
    
    # SVR + GAN
    svr_gan = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    svr_gan.fit(X_aug, y_aug)
    results['SVR_GAN'] = mean_squared_error(y_test, svr_gan.predict(X_test_scaled))
    
    # GPR Baseline (Improved Kernel)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0)
    gpr.fit(X_train_scaled, y_train)
    results['GPR_Baseline'] = mean_squared_error(y_test, gpr.predict(X_test_scaled))
    
    # GPR + GAN
    gpr_gan = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.0)
    gpr_gan.fit(X_aug, y_aug)
    results['GPR_GAN'] = mean_squared_error(y_test, gpr_gan.predict(X_test_scaled))
    
    print("\n" + "="*30)
    print("RESULTS (MSE on Test Config)")
    print("="*30)
    for k, v in results.items():
        print(f"{k:<15}: {v:.6f}")

if __name__ == "__main__":
    evaluate_paper_gan()
