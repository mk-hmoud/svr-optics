import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import os
import sys
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_data
from src.wgan_paper import train_wgan_paper, generate_samples_paper

def find_gan_contradictions():
    print("Loading real data and training GAN...")
    df = load_data('data/data.xlsx')
    
    # Standard features used in the paper GAN
    feature_cols = ['Analyte', 'Re(eff)', 'lambda', 'Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    X = df[feature_cols]
    y = np.log10(np.clip(df['loss'] * 10**8, a_min=1e-10, a_max=None))
    
    # Combined for GAN training (9-column scale if Target is included, or scale separately)
    # The original script scaled X and y together by hstacking then scaling?
    # No, it scaled X and then stacked y. Let's see:
    # X_train_scaled = scaler_X.fit_transform(X_train)
    # real_train_combined = np.hstack([X_train_scaled, y_train.values.reshape(-1, 1)])
    
    # Wait, if y is not scaled, it ranges from 5 to 9.7, while X is 0 to 1.
    # This might be why GAN favors the larger y values.
    # Let's try scaling Y as well for the GAN training to see if it learns better.
    
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    real_train_combined = np.hstack([X_scaled, y_scaled])
    
    # Train GAN (WGAN-GP) or load if exists
    model_path = 'src/researcher_code/trained_gan.pth'
    if os.path.exists(model_path):
        print("Loading pre-trained GAN...")
        generator = torch.load(model_path, weights_only=False)
    else:
        print("Training GAN...")
        generator = train_wgan_paper(real_train_combined, epochs=2000)
        torch.save(generator, model_path)
    
    # Generate a large synthetic dataset to find collisions
    num_synth = 50000
    print(f"\nGenerating {num_synth} synthetic samples...")
    synthetic_data = generate_samples_paper(generator, num_samples=num_synth)
    
    # Unscale y for analysis
    X_synth_scaled = synthetic_data[:, :-1]
    y_synth_scaled = synthetic_data[:, -1]
    y_synth = scaler_y.inverse_transform(y_synth_scaled.reshape(-1, 1)).flatten()
    
    # --- Finding the Contradictions ---
    print("Searching for Geometric Contradictions (x_i approx x_j, y_i != y_j)...")
    
    # Find 2 nearest neighbors for each point in the synthetic set
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=-1)
    nn.fit(X_synth_scaled)
    distances, indices = nn.kneighbors(X_synth_scaled)
    
    # Calculate target difference for the nearest neighbor pairs
    y_diffs = np.abs(y_synth[indices[:, 0]] - y_synth[indices[:, 1]])
    
    # Always take the worst ratios for the 'smoking gun'
    print("Finding the most mathematically impossible vertical jumps (High Delta Y / Low Dist)...")
    safe_dist = np.maximum(distances[:, 1], 1e-6)
    ratio = y_diffs / safe_dist
    top_indices = np.argsort(ratio)[::-1][:10]

    # --- Display Results ---
    print("\n" + "="*80)
    print("TOP GAN MATHEMATICAL IMPOSSIBILITIES (Vertical Jumps)")
    print("="*80)
    print(f"{'Dist (Scaled)':<15} | {'y1 (Log Loss)':<15} | {'y2 (Log Loss)':<15} | {'Delta y':<10} | {'Ratio'}")
    print("-" * 80)
    
    contradictions = []
    for idx in top_indices:
        neighbor_idx = indices[idx, 1]
        dist = distances[idx, 1]
        y1 = y_synth[idx]
        y2 = y_synth[neighbor_idx]
        dy = y_diffs[idx]
        r = ratio[idx]
        
        print(f"{dist:<15.8f} | {y1:<15.6f} | {y2:<15.6f} | {dy:<10.6f} | {r:.2f}")
        
        # Save one for the paper
        contradictions.append({
            'Distance': dist,
            'y1': y1,
            'y2': y2,
            'Delta_y': dy,
            'Sample1_Features': X_synth_scaled[idx],
            'Sample2_Features': X_synth_scaled[neighbor_idx]
        })

    # Save the worst one to a text file for inclusion in paper
    worst = contradictions[0]
    with open('gan_contradiction_proof.txt', 'w') as f:
        f.write(f"Empirical Proof of GPR Collapse:\n")
        f.write(f"Two GAN-generated samples were found with a Euclidean distance of {worst['Distance']:.6f} in scaled feature space.\n")
        f.write(f"Sample 1 Target (Log10 Loss): {worst['y1']:.6f}\n")
        f.write(f"Sample 2 Target (Log10 Loss): {worst['y2']:.6f}\n")
        f.write(f"Target Discrepancy (Delta Y): {worst['Delta_y']:.6f}\n")
        f.write(f"\nFeature Vectors (Scaled):\n")
        f.write(f"S1: {worst['Sample1_Features']}\n")
        f.write(f"S2: {worst['Sample2_Features']}\n")

    print("\nWorst contradiction saved to 'gan_contradiction_proof.txt'")

if __name__ == '__main__':
    find_gan_contradictions()
