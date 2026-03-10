import os
import sys

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data import load_data, preprocess_data, get_train_test_split
from src.data_augmentation import augment_with_gaussian_noise
from src.models.baseline_svr import train_baseline_svr, evaluate_model

if __name__ == '__main__':
    print("Loading data for augmented training...")
    df = load_data()
    X, y, scaler = preprocess_data(df)
    
    # We split first, THEN augment only the training data to prevent data leakage!
    X_train_raw, X_test, y_train_raw, y_test = get_train_test_split(X, y)
    
    print(f"Original Train shape: {X_train_raw.shape}")
    X_train_aug, y_train_aug = augment_with_gaussian_noise(
        X_train_raw, y_train_raw, 
        noise_level=0.03, # 3% noise relative to std
        num_synthetic_sets=2 # Tripling the training set size
    )
    print(f"Augmented Train shape: {X_train_aug.shape}")
    
    # Train the SVR using the same GridSearch logic as the baseline
    print("\nTraining SVR on Augmented Data...")
    best_aug_svr_model = train_baseline_svr(X_train_aug, y_train_aug)
    
    print("\nEvaluating SVR on Original Unaugmented Test Set...")
    evaluate_model(best_aug_svr_model, X_test, y_test, "Augmented SVR")
