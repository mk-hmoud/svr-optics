import numpy as np
import pandas as pd
from src.wgan import train_wgan, generate_samples

def augment_with_wgan(X, y, num_synthetic_samples=1000, epochs=2000):
    """
    Augments tabular data using a Wasserstein GAN with Gradient Penalty.
    
    Args:
    - X: DataFrame of features.
    - y: Series of target values (log10 loss).
    - num_synthetic_samples: Number of synthetic samples to generate.
    - epochs: Number of training epochs for the WGAN.
    
    Returns:
    - X_combined: Original + Synthetic features.
    - y_combined: Original + Synthetic targets.
    """
    # 1. Prepare data for GAN (Features + Target in one matrix)
    # Convert to numpy array
    real_data = np.hstack([X.values, y.values.reshape(-1, 1)])
    
    # 2. Train the WGAN
    print(f"Training WGAN on {real_data.shape[0]} samples for {epochs} epochs...")
    generator = train_wgan(real_data, epochs=epochs)
    
    # 3. Generate synthetic samples
    synthetic_data = generate_samples(generator, num_samples=num_synthetic_samples)
    
    # 4. Filter and Clip to Physical Bounds
    # Calculate physical bounds from original data
    min_bounds = real_data.min(axis=0)
    max_bounds = real_data.max(axis=0)
    
    # Clip all generated columns (7 features + 1 target) to their respective bounds
    # This ensures synthetic data remains "physically plausible".
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
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    df = load_data('data/data.xlsx')
    X, y, _, _ = preprocess_data(df)
    
    print(f"Original Data Shape: {X.shape}")
    X_aug, y_aug = augment_with_wgan(X, y, num_synthetic_samples=500, epochs=100)
    print(f"Augmented Data Shape: {X_aug.shape}")
