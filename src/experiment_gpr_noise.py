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

from src.data import load_data
from src.wgan_paper import train_wgan_paper, generate_samples_paper

def run_gpr_noise_experiment():
    # 1. Load and Prepare Data
    print("Loading data...")
    df = load_data('data/data.xlsx')
    feature_cols = ['Analyte', 'Re(eff)', 'lambda', 'Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    X = df[feature_cols]
    y = np.log10(np.clip(df['loss'] * 10**8, a_min=1e-10, a_max=None))
    
    # 7-1-1 Split
    config_cols = ['Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    unique_configs = df[config_cols].drop_duplicates().values.tolist()
    
    def get_config_id(row):
        for i, config in enumerate(unique_configs):
            if all(row[config_cols] == config): return i + 1
        return None
    
    df['config_id'] = df.apply(get_config_id, axis=1)
    train_indices = df[df['config_id'] <= 7].index
    test_indices = df[df['config_id'] == 9].index
    
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 2. Train GAN and Generate Augmented Data
    print("\n--- Training GAN (WGAN-GP) ---")
    real_train_combined = np.hstack([X_train_scaled, y_train.values.reshape(-1, 1)])
    generator = train_wgan_paper(real_train_combined, epochs=2000)
    
    print("Generating 1000 synthetic samples...")
    synthetic_data = generate_samples_paper(generator, num_samples=1000)
    synthetic_data = np.clip(synthetic_data, real_train_combined.min(axis=0), real_train_combined.max(axis=0))
    
    X_aug = np.vstack([X_train_scaled, synthetic_data[:, :-1]])
    y_aug = np.concatenate([y_train, synthetic_data[:, -1]])
    
    # --- Running Experiments ---
    results = []
    
    # Define data sets
    data_sets = [
        {"name": "Real Data (No GAN)", "X": X_train_scaled, "y": y_train},
        {"name": "Augmented (With GAN)", "X": X_aug, "y": y_aug}
    ]
    
    # Define configurations
    base_kernel = C(1.0) * RBF(1.0)
    wk_kernel = base_kernel + WhiteKernel(noise_level=1e-3)
    noise_val = 1e-5 
    
    configs = [
        {"name": "No WK, No Alpha (alpha=0)", "kernel": base_kernel, "alpha": 0.0},
        {"name": "With WK, With Alpha", "kernel": wk_kernel, "alpha": noise_val},
        {"name": "With WK, No Alpha (alpha=0)", "kernel": wk_kernel, "alpha": 0.0},
        {"name": "No WK, With Alpha", "kernel": base_kernel, "alpha": noise_val},
    ]
    
    print("\n--- Running GPR Noise Experiments ---")
    header = f"{'Data Set':<20} | {'Configuration':<25} | {'Test MSE':<10}"
    print(header)
    print("-" * len(header))
    
    for ds in data_sets:
        for cfg in configs:
            # Set n_restarts_optimizer to 2 for speed
            gpr = GaussianProcessRegressor(kernel=cfg['kernel'], alpha=cfg['alpha'], n_restarts_optimizer=2, random_state=42)
            
            try:
                gpr.fit(ds['X'], ds['y'])
                preds = gpr.predict(X_test_scaled)
                mse = mean_squared_error(y_test, preds)
                print(f"{ds['name']:<20} | {cfg['name']:<25} | {mse:<10.6f}")
                results.append({"Data_Set": ds['name'], "Config": cfg['name'], "MSE": mse})
            except Exception as e:
                # Catch numerical errors (singular matrix etc)
                err_msg = str(e).split('\n')[0][:30]
                print(f"{ds['name']:<20} | {cfg['name']:<25} | FAILED: {err_msg}...")
                results.append({"Data_Set": ds['name'], "Config": cfg['name'], "MSE": np.nan})
            
    # Save results
    res_df = pd.DataFrame(results)
    res_df.to_csv('results_gpr_noise_experiment.csv', index=False)
    print("\nResults saved to 'results_gpr_noise_experiment.csv'")

if __name__ == "__main__":
    run_gpr_noise_experiment()
