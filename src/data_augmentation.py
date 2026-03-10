import numpy as np
import pandas as pd

def augment_with_gaussian_noise(X, y, noise_level=0.05, num_synthetic_sets=1, random_state=42):
    """
    Augments tabular data by adding Gaussian noise to features, 
    with filtering to ensure physical plausibility.
    
    Args:
    - X: DataFrame of features.
    - y: Series of target values (log10 loss).
    - noise_level: The standard deviation of the noise relative to each feature's standard deviation.
    - num_synthetic_sets: How many times to duplicate and noisy the original dataset.
    
    Returns:
    - X_combined: Original + Synthetic features (clipped to physical bounds).
    - y_combined: Original + Synthetic targets.
    """
    np.random.seed(random_state)
    
    # Calculate physical bounds from original data
    # This ensures features like wavelength or pitch don't go outside the simulated range.
    min_bounds = X.min(axis=0)
    max_bounds = X.max(axis=0)
    
    X_synthetic = []
    y_synthetic = []
    
    # Calculate standard deviation for each column for proportional noise
    std_devs = X.std(axis=0)
    
    for _ in range(num_synthetic_sets):
        # Generate noise proportional to the standard deviation of each feature
        noise = np.random.normal(0, std_devs * noise_level, size=X.shape)
        X_aug = X + noise
        
        # CLIP to physical bounds: Matches the paper's filtering strategy to ensure 
        # that synthetic samples are within the valid "numeric range of the application".
        X_aug = X_aug.clip(lower=min_bounds, upper=max_bounds, axis=1)
        
        # Perturb target slightly based on feature noise
        y_noise = np.random.normal(0, y.std() * noise_level * 0.1, size=y.shape)
        y_aug = y + y_noise
        
        X_synthetic.append(X_aug)
        y_synthetic.append(y_aug)
        
    X_synth_df = pd.concat(X_synthetic)
    y_synth_series = pd.concat(y_synthetic)
    
    # Combine original and synthetic
    X_combined = pd.concat([X, X_synth_df], ignore_index=True)
    y_combined = pd.concat([y, y_synth_series], ignore_index=True)
    
    return X_combined, y_combined

if __name__ == '__main__':
    from src.data import load_data, preprocess_data
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    df = load_data('data/data.xlsx')
    X, y, _, _ = preprocess_data(df)
    
    print(f"Original Data Shape: {X.shape}")
    X_aug, y_aug = augment_with_gaussian_noise(X, y, noise_level=0.05, num_synthetic_sets=2)
    print(f"Augmented Data Shape: {X_aug.shape}")
    
    # Verify bounds
    print("\nBounds Verification (Original vs Augmented):")
    for col in X.columns:
        orig_min, orig_max = X[col].min(), X[col].max()
        aug_min, aug_max = X_aug[col].min(), X_aug[col].max()
        print(f"{col}: Orig [{orig_min:.2f}, {orig_max:.2f}] | Aug [{aug_min:.2f}, {aug_max:.2f}]")
