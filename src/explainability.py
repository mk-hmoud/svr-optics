import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.data import load_data, preprocess_data
from src.evaluate_logo import train_best_svr_bayesian
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def analyze_feature_importance():
    """
    Uses SHAP (SHapley Additive exPlanations) to interpret the SVR model.
    This reveals exactly which PCF features drive the confinement loss prediction.
    """
    # 1. Load and prep all data
    print("Loading data...")
    df = load_data()
    X, y, _, _ = preprocess_data(df)
    
    # 2. Train the model on the full dataset for global importance
    print("Training the final SVR model for explainability analysis (using Bayesian Opt)...")
    model = train_best_svr_bayesian(X, y)
    
    # 3. Create SHAP explainer
    # KernelExplainer is model-agnostic and works for SVR.
    # We use a summarized background dataset (kmeans) to speed up calculation.
    print("Initializing SHAP Explainer...")
    X_summary = shap.kmeans(X, 20) 
    explainer = shap.KernelExplainer(model.predict, X_summary)
    
    # 4. Calculate SHAP values
    # For the summary plot, we don't need every single point if calculation is slow.
    # Using 100 points provides a very strong statistical representation.
    X_test_explain = X.sample(min(100, len(X)), random_state=42)
    print(f"Calculating SHAP values for {len(X_test_explain)} samples...")
    shap_values = explainer.shap_values(X_test_explain)
    
    # 5. Visualize and Save
    print("Generating Importance Plots...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_explain, show=False, plot_type="bar")
    plt.title("Global Feature Importance (Mean |SHAP Value|)")
    plt.savefig('feature_importance_bar.png', bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_explain, show=False)
    plt.title("SHAP Summary Plot: Impact of Feature Values on Loss")
    plt.savefig('feature_importance_summary.png', bbox_inches='tight')
    plt.close()
    
    print("Plots saved: 'feature_importance_bar.png' and 'feature_importance_summary.png'")
    
    # 6. Output numerical ranking
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean_Absolute_SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)
    
    print("\n--- Physical Feature Importance Ranking ---")
    print(importance_df.to_string(index=False))
    importance_df.to_csv('feature_importance_ranking.csv', index=False)
    print("\nRanking saved to 'feature_importance_ranking.csv'")

if __name__ == '__main__':
    analyze_feature_importance()
