import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.svm import SVR
from src.data import load_data, preprocess_data

def train_final_svr(X, y):
    """Trains the final SVR model with optimized hyperparameters."""
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    model.fit(X, y)
    return model

def find_resonance_peak(model, scaler, config_dict, wavelengths):
    """Predicts the resonance peak wavelength for a given configuration."""
    scan_data = []
    for wl in wavelengths:
        row = config_dict.copy()
        row['lambda'] = wl
        scan_data.append(row)
    
    scan_df = pd.DataFrame(scan_data)
    # Ensure correct column order for the scaler
    feature_order = ['Analyte', 'lambda', 'Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    scan_df = scan_df[feature_order]
    
    scan_scaled = scaler.transform(scan_df)
    preds = model.predict(scan_scaled)
    
    peak_idx = np.argmax(preds)
    return wavelengths[peak_idx], preds[peak_idx]

def run_robustness_analysis():
    """
    Simulates manufacturing tolerances by adding noise to geometric parameters
    and observing the shift (jitter) in the resonance peak.
    """
    print("Loading data and training model...")
    df = load_data()
    X_scaled, y, groups, scaler = preprocess_data(df)
    model = train_final_svr(X_scaled, y)
    
    # Baseline configuration (Design 1)
    base_config = {
        'Analyte': 1.33,
        'Pitch (um)': 2.0,
        'd1 (um)': 0.225,
        'd2 (um)': 0.375,
        'd3 (um)': 0.175
    }
    
    wl_min, wl_max = df['lambda'].min(), df['lambda'].max()
    wavelengths = np.linspace(wl_min, wl_max, 500)
    
    base_peak, _ = find_resonance_peak(model, scaler, base_config, wavelengths)
    print(f"Baseline Resonance Peak: {base_peak:.2f} nm")
    
    # Noise levels to test
    noise_levels = [0.01, 0.03, 0.05] # 1%, 3%, 5%
    geom_features = ['Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    
    num_simulations = 100
    results = []
    
    print("\nStarting Monte Carlo Simulations...")
    for level in noise_levels:
        print(f"Testing {level*100}% noise level...")
        peaks = []
        for i in range(num_simulations):
            noisy_config = base_config.copy()
            for feat in geom_features:
                # Add Gaussian noise relative to the feature value
                noise = np.random.normal(0, level * base_config[feat])
                noisy_config[feat] += noise
            
            peak, _ = find_resonance_peak(model, scaler, noisy_config, wavelengths)
            peaks.append(peak)
        
        results.append({
            'Noise_Level': f"{level*100}%",
            'Peaks': peaks,
            'Std_Dev': np.std(peaks),
            'Range': np.max(peaks) - np.min(peaks)
        })

    # --- Feature Sensitivity (Volatility) ---
    # Perturb ONE feature at a time to see which one is the "Volatility Driver"
    print("\nAnalyzing individual feature volatility (3% noise)...")
    volatility_results = []
    for feat in geom_features:
        peaks = []
        for i in range(num_simulations):
            noisy_config = base_config.copy()
            noise = np.random.normal(0, 0.03 * base_config[feat])
            noisy_config[feat] += noise
            
            peak, _ = find_resonance_peak(model, scaler, noisy_config, wavelengths)
            peaks.append(peak)
        
        volatility_results.append({
            'Feature': feat,
            'Volatility_nm': np.std(peaks)
        })

    # --- Visualization ---
    print("\nGenerating Robustness Plots...")
    
    # 1. Histogram of Peak Jitter
    plt.figure(figsize=(10, 6))
    for res in results:
        # Use simple histogram or basic kde if seaborn is missing
        plt.hist(res['Peaks'], bins=20, alpha=0.4, label=f"Noise {res['Noise_Level']} (std={res['Std_Dev']:.2f}nm)", density=True)
    
    plt.axvline(x=base_peak, color='red', linestyle='--', label='Theoretical Design')
    plt.title("Resonance Peak Jitter under Manufacturing Tolerances")
    plt.xlabel("Resonance Peak Wavelength (nm)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('robustness_jitter_histogram.png', bbox_inches='tight')
    
    # 2. Volatility Ranking Bar Chart
    vol_df = pd.DataFrame(volatility_results).sort_values(by='Volatility_nm', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(vol_df['Feature'], vol_df['Volatility_nm'], color='skyblue')
    plt.gca().invert_yaxis() # Highest volatility at the top
    plt.title("Manufacturing Volatility Drivers (3% Tolerance Noise)")
    plt.xlabel("Peak Jitter Standard Deviation (nm)")
    plt.ylabel("Geometric Feature")
    plt.grid(True, axis='x', alpha=0.3)
    plt.savefig('robustness_volatility_ranking.png', bbox_inches='tight')
    
    print("\n--- Manufacturing Robustness Summary ---")
    summary_df = pd.DataFrame(results).drop(columns=['Peaks'])
    print(summary_df.to_string(index=False))
    
    print("\n--- Volatility Ranking (Volatility SHAP) ---")
    print(vol_df.to_string(index=False))
    
    summary_df.to_csv('robustness_summary.csv', index=False)
    vol_df.to_csv('robustness_volatility_ranking.csv', index=False)
    print("\nResults saved to CSV and PNG files.")

if __name__ == '__main__':
    run_robustness_analysis()
