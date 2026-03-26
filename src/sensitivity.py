import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_data, preprocess_data
from src.evaluate_logo import train_best_svr_bayesian
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def calculate_spectral_sensitivity():
    """
    Calculates the Spectral Sensitivity (nm/RIU) of the PCF-SPR sensor.
    Sensitivity S = Δλ_peak / Δn_a.
    The SVR model is used to predict the loss curve and find the resonance peak.
    """
    # 1. Load Data
    print("Loading data...")
    df = load_data()
    # We need the raw df to get the original feature scales for scanning
    X_scaled, y, groups, scaler = preprocess_data(df)
    
    # 2. Train the Final Model
    print("Training the final SVR model for sensitivity calculation...")
    model = train_best_svr_bayesian(X_scaled, y)
    
    # 3. Analyze Peaks for different Analytes
    # In the dataset, we have Analyte RI: 1.33, 1.34, 1.35
    analytes = sorted(df['Analyte'].unique())
    peak_wavelengths = []
    
    # Use a specific geometric configuration (the first one) to perform the scan
    config_cols = ['Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    first_config = df[config_cols].iloc[0].to_dict()
    
    # Wavelength range from data
    wl_min, wl_max = df['lambda'].min(), df['lambda'].max()
    wavelengths = np.linspace(wl_min, wl_max, 1000) # High resolution scan
    
    plt.figure(figsize=(10, 6))
    
    print("\n--- Resonance Peak Detection ---")
    for analyte in analytes:
        scan_data = []
        for wl in wavelengths:
            row = {'Analyte': analyte, 'lambda': wl}
            row.update(first_config)
            scan_data.append(row)
            
        scan_df = pd.DataFrame(scan_data)
        # Ensure column order matches X_scaled
        scan_df = scan_df[X_scaled.columns]
        
        # Scale features using the fitted scaler from preprocess_data
        scan_scaled = scaler.transform(scan_df)
        
        # Predict Log10(Loss)
        preds = model.predict(scan_scaled)
        
        # Find peak (Confinement Loss is maximum at resonance)
        peak_idx = np.argmax(preds)
        peak_wl = wavelengths[peak_idx]
        peak_wavelengths.append(peak_wl)
        
        plt.plot(wavelengths, preds, label=f'Analyte RI = {analyte}')
        # Add markers at sparse intervals for better visibility
        plt.scatter(wavelengths[::50], preds[::50], marker='o', s=20)
        # Mark the actual peak with a cross
        plt.scatter([peak_wl], [preds[peak_idx]], marker='x', color='red', s=100, zorder=5)
        plt.axvline(x=peak_wl, linestyle='--', alpha=0.5)
        
        print(f"Analyte {analyte} RIU | Predicted Resonance Peak: {peak_wl:.4f} nm")
    
    # 4. Calculate Sensitivity: S = Δλ_peak / Δn_a
    print("\n--- Spectral Sensitivity Calculation ---")
    sens_values = []
    for i in range(len(analytes) - 1):
        delta_lambda = peak_wavelengths[i+1] - peak_wavelengths[i]
        delta_n = analytes[i+1] - analytes[i]
        s = delta_lambda / delta_n
        sens_values.append(s)
        print(f"Sensitivity (n={analytes[i]} to {analytes[i+1]}): {s:.2f} nm/RIU")
    
    avg_sensitivity = np.mean(sens_values)
    print(f"\nAverage Spectral Sensitivity: {avg_sensitivity:.2f} nm/RIU")
    
    # 5. Finalize Plot
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Predicted Log10(Confinement Loss * 10^8)")
    plt.title(f"Predicted Resonance Peaks\nAvg Sensitivity: {avg_sensitivity:.2f} nm/RIU")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('spectral_sensitivity_curves.png', bbox_inches='tight')
    print("\nSensitivity curves saved to 'spectral_sensitivity_curves.png'")
    
    # Save sensitivity results to CSV
    sens_df = pd.DataFrame({
        'Analyte_Transition': [f"{analytes[i]}->{analytes[i+1]}" for i in range(len(analytes)-1)],
        'Delta_Lambda_nm': [peak_wavelengths[i+1] - peak_wavelengths[i] for i in range(len(analytes)-1)],
        'Sensitivity_nm_RIU': sens_values
    })
    sens_df.to_csv('spectral_sensitivity_results.csv', index=False)

if __name__ == '__main__':
    calculate_spectral_sensitivity()
