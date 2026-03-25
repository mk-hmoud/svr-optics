import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    # 1. Load Data
    df = pd.read_excel('data/data.xlsx')
    
    # 2. Features and Target
    feature_cols = ['Analyte', 'Re(eff)', 'lambda', 'Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    X = df[feature_cols]
    
    # 3. Target Transformation: y = np.log10(np.clip(loss * 10**8, a_min=1e-10, a_max=None))
    loss = df['loss']
    y = np.log10(np.clip(loss * 10**8, a_min=1e-10, a_max=None))
    
    # Identify Configurations for 7-1-1 Split
    config_cols = ['Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    unique_configs = df[config_cols].drop_duplicates().values.tolist()
    
    def get_config_id(row):
        for i, config in enumerate(unique_configs):
            if all(row[config_cols] == config):
                return i + 1
        return None
    
    df['config_id'] = df.apply(get_config_id, axis=1)
    
    # Split into 7-1-1
    train_indices = df[df['config_id'] <= 7].index
    val_indices = df[df['config_id'] == 8].index
    test_indices = df[df['config_id'] == 9].index
    
    X_train_raw = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_val_raw = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    X_test_raw = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    # Scaling for SVR/GPR (Fit only on Train)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    results = {'ANN': [], 'SVR': [], 'GPR': []}
    num_runs = 10

    print(f"Starting {num_runs} robust runs of Researcher Methodology (7-1-1 Split)...")

    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...")
        
        # --- ANN (No Scaling) ---
        model_ann = models.Sequential([
            layers.Input(shape=(X_train_raw.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1)
        ])
        model_ann.compile(optimizer='adam', loss='mse')
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        model_ann.fit(X_train_raw, y_train, validation_data=(X_val_raw, y_val), 
                      epochs=500, batch_size=32, callbacks=[early_stop], verbose=0)
        
        mse_ann = mean_squared_error(y_test, model_ann.predict(X_test_raw, verbose=0).flatten())
        results['ANN'].append(mse_ann)

        # --- SVR (Scaled) ---
        # SVR is deterministic, but we run it for consistency
        svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
        svr.fit(X_train_scaled, y_train)
        mse_svr = mean_squared_error(y_test, svr.predict(X_test_scaled))
        results['SVR'].append(mse_svr)

        # --- GPR (Scaled, Fixed) ---
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.0)
        gpr.fit(X_train_scaled, y_train)
        mse_gpr = mean_squared_error(y_test, gpr.predict(X_test_scaled))
        results['GPR'].append(mse_gpr)
        
        tf.keras.backend.clear_session()

    # Summarize Results
    print("\n" + "="*40)
    print("ROBUST METHODOLOGY RESULTS (AVERAGE OF 10 RUNS)")
    print("="*40)
    print(f"{'Model':<10} | {'Avg MSE':<12} | {'Std Dev':<10}")
    print("-" * 40)
    for model_name, mses in results.items():
        avg_mse = np.mean(mses)
        std_mse = np.std(mses)
        print(f"{model_name:<10} | {avg_mse:<12.6f} | {std_mse:<10.6f}")
    print("="*40)

if __name__ == "__main__":
    main()
