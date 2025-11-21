# analyze_raw_pls.py
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import silhouette_score

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
    print(f"--- Analyzing Raw Activations from: {args.data_dir} ---")
    X = np.load(os.path.join(args.data_dir, "all_vectors.npy"))
    y = np.load(os.path.join(args.data_dir, "all_labels.npy"))
    class_names = args.class_names

    print(f"Total Samples: {len(X)}")
    print(f"Dimensions: {X.shape[1]}")
    
    # PLS-DA
    print("Performing PLS-DA...")
    n_components = 2
    pls = PLSRegression(n_components=n_components)
    X_pls = pls.fit_transform(X, y)[0]
    
    # Visualization
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        plt.scatter(X_pls[y == i, 0], X_pls[y == i, 1], label=name, alpha=0.6)
    
    plt.title(f"PLS Separation (Raw Activations): {class_names[0]} vs {class_names[1]}")
    plt.xlabel("PLS Component 1")
    plt.ylabel("PLS Component 2")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(args.data_dir, "pls_separation.png")
    plt.savefig(plot_path)
    print(f"Saved plot to: {plot_path}")

    # Metrics (KL and Silhouette)
    X1 = X_pls[y==0]
    X2 = X_pls[y==1]
    
    mean1, cov1 = np.mean(X1, axis=0), np.cov(X1, rowvar=False)
    mean2, cov2 = np.mean(X2, axis=0), np.cov(X2, rowvar=False)
    
    # Regularize slightly
    cov1 += np.eye(n_components) * 1e-4
    cov2 += np.eye(n_components) * 1e-4
    
    kl1 = kl_divergence_gaussians(mean1, cov1, mean2, cov2)
    kl2 = kl_divergence_gaussians(mean2, cov2, mean1, cov1)
    sil = silhouette_score(X_pls, y)
    
    print(f"\n--- Results ---")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"KL(Class1 || Class2): {kl1:.4f}")
    print(f"KL(Class2 || Class1): {kl2:.4f}")
    
    with open(os.path.join(args.data_dir, 'results.txt'), 'w') as f:
        f.write(f"Silhouette: {sil:.4f}\nKL1: {kl1:.4f}\nKL2: {kl2:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--class_names', nargs='+', required=True)
    args = parser.parse_args()
    analyze(args)
