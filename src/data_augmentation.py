import numpy as np
import pandas as pd
import os
from src.wgan import train_wgan, generate_samples

def load_researcher_data(filepath='data/gen_data.txt', feature_columns=None):
    """
    Loads pre-generated synthetic data provided by researchers.
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return None, None
        
    # Assuming CSV format without header: 7 features + 1 target
    data = pd.read_csv(filepath, header=None)
    
    # Separate features and target
    X_synth = data.iloc[:, :-1]
    y_synth = data.iloc[:, -1]
    
    if feature_columns is not None:
        X_synth.columns = feature_columns
        
    return X_synth, y_synth

def augment_with_wgan(X, y, num_synthetic_samples=1000, epochs=2000):
    """
    Augments tabular data using a Wasserstein GAN with Gradient Penalty.
    """
    # 1. Prepare data for GAN (Features + Target in one matrix)
    real_data = np.hstack([X.values, y.values.reshape(-1, 1)])
    
    # 2. Train the WGAN
    print(f"Training WGAN on {real_data.shape[0]} samples for {epochs} epochs...")
    generator = train_wgan(real_data, epochs=epochs)
    
    # 3. Generate synthetic samples
    synthetic_data = generate_samples(generator, num_samples=num_synthetic_samples)
    
    # 4. Filter and Clip to Physical Bounds
    min_bounds = real_data.min(axis=0)
    max_bounds = real_data.max(axis=0)
    synthetic_data = np.clip(synthetic_data, min_bounds, max_bounds)
    
    # 5. Separate features and target
    X_synth = pd.DataFrame(synthetic_data[:, :-1], columns=X.columns)
    y_synth = pd.Series(synthetic_data[:, -1])
    
    # 6. Combine original and synthetic
    X_combined = pd.concat([X, X_synth], ignore_index=True)
    y_combined = pd.concat([y, y_synth], ignore_index=True)
    
    return X_combined, y_combined

if __name__ == '__main__':
    from src.data import load_data, preprocess_data
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    df = load_data('data/data.xlsx')
    X, y, _, _ = preprocess_data(df)
    
    X_synth, y_synth = load_researcher_data('data/gen_data.txt', feature_columns=X.columns)
    if X_synth is not None:
        print(f"Loaded Researcher Data: {X_synth.shape}")
