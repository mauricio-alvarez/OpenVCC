# analyze_separability_pls.py
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

def calculate_robust_kl_with_pls(X, y):
    """
    Calculates KL using PLS for dimensionality reduction.
    PLS finds the subspace that maximally separates the classes.
    """
    n_samples = len(X)
    # We need at least 2 samples per class to calculate variance
    if np.sum(y==0) < 2 or np.sum(y==1) < 2:
        return float('nan'), float('nan')

    # 1. Use PLS to project data into the most discriminative subspace
    # We use 2 components if possible, or 1 if data is very scarce
    n_components = min(n_samples - 1, 2)
    
    # PLS requires Y to be numeric. We have 0s and 1s, which is perfect.
    pls = PLSRegression(n_components=n_components)
    try:
        X_pls = pls.fit_transform(X, y)[0] # PLS returns tuple (X_scores, Y_scores)
    except Exception as e:
        print(f"PLS failed (likely too few samples): {e}")
        return float('nan'), float('nan')

    # 2. Split back into classes
    X1_proj = X_pls[y == 0]
    X2_proj = X_pls[y == 1]
    
    # 3. Calculate Statistics
    mean1 = np.mean(X1_proj, axis=0)
    cov1 = np.cov(X1_proj, rowvar=False)
    mean2 = np.mean(X2_proj, axis=0)
    cov2 = np.cov(X2_proj, rowvar=False)
    
    # Handle 1D case
    if n_components == 1:
        cov1 = np.array([[cov1]]) if np.ndim(cov1)==0 else cov1
        cov2 = np.array([[cov2]]) if np.ndim(cov2)==0 else cov2

    # 4. Regularize
    cov1 += np.eye(cov1.shape[0]) * 1e-4
    cov2 += np.eye(cov2.shape[0]) * 1e-4
    
    kl_1_2 = kl_divergence_gaussians(mean1, cov1, mean2, cov2)
    kl_2_1 = kl_divergence_gaussians(mean2, cov2, mean1, cov1)
    
    return kl_1_2, kl_2_1, X_pls

def kl_divergence_gaussians(mean1, cov1, mean2, cov2):
    try:
        cov2_inv = np.linalg.inv(cov2)
        term1 = np.trace(cov2_inv @ cov1)
        term2 = (mean2 - mean1).T @ cov2_inv @ (mean2 - mean1)
        sign2, logdet2 = np.linalg.slogdet(cov2)
        sign1, logdet1 = np.linalg.slogdet(cov1)
        if sign1 != 1 or sign2 != 1: return float('nan')
        term3 = logdet2 - logdet1
        k = len(mean1)
        return max(0.0, 0.5 * (term1 + term2 - k + term3))
    except:
        return float('nan')

def analyze(args):
    print(f"--- Analyzing data from: {args.data_dir} using PLS ---")
    X = np.load(os.path.join(args.data_dir, "all_vectors.npy"))
    y = np.load(os.path.join(args.data_dir, "all_labels.npy"))
    class_names = args.class_names

    print(f"Total Samples: {len(X)}")
    
    # --- PLS Calculation & Visualization ---
    if len(class_names) == 2:
        print("Performing PLS-DA (Discriminant Analysis)...")
        kl_1_2, kl_2_1, X_pls = calculate_robust_kl_with_pls(X, y)
        
        # Visualization
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            if X_pls.shape[1] == 1:
                plt.scatter(X_pls[y == i, 0], np.zeros_like(X_pls[y == i, 0]), label=class_name, alpha=0.7)
            else:
                plt.scatter(X_pls[y == i, 0], X_pls[y == i, 1], label=class_name, alpha=0.7)
        
        plt.title(f"PLS Concept Separation: {args.data_dir.split('/')[-1]}")
        plt.xlabel("PLS Component 1 (Max separation)")
        plt.ylabel("PLS Component 2")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(args.data_dir, "concept_separability_pls.png")
        plt.savefig(plot_path)
        print(f"Saved PLS visualization to: {plot_path}")
        plt.close()
        
        print(f"\n--- PLS Metrics ---")
        print(f"KL Divergence (KL(Class1 || Class2)): {kl_1_2:.4f}")
        print(f"KL Divergence (KL(Class2 || Class1)): {kl_2_1:.4f}")
    else:
        print("PLS requires exactly 2 classes for binary discriminant analysis in this script.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--class_names', nargs='+', default=['pickup', 'garbage_truck'])
    args = parser.parse_args()
    analyze(args)
