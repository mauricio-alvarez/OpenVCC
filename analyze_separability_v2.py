# analyze_separability.py
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

def calculate_robust_kl(X1, X2):
    """
    Calculates KL divergence robustly, even with few samples in high dimensions.
    It applies PCA to project data into a valid subspace before calculation.
    """
    # 1. Determine valid dimensions based on sample count
    n_samples_1 = len(X1)
    n_samples_2 = len(X2)
    
    if n_samples_1 < 2 or n_samples_2 < 2:
        return float('nan') # Cannot calculate variance with 1 point

    # We need fewer dimensions than samples to get a valid covariance matrix
    # We pick the minimum of samples-1 or a cap (e.g., 5 dimensions)
    max_possible_dim = min(n_samples_1, n_samples_2) - 1
    target_dim = min(max_possible_dim, 5) # 5 dimensions is usually enough for separation info
    
    if target_dim < 1: target_dim = 1

    # 2. Project both classes into the SAME common subspace using PCA
    X_combined = np.vstack([X1, X2])
    pca = PCA(n_components=target_dim)
    X_combined_pca = pca.fit_transform(X_combined)
    
    X1_proj = X_combined_pca[:n_samples_1]
    X2_proj = X_combined_pca[n_samples_1:]
    
    # 3. Calculate Statistics in this reduced space
    mean1 = np.mean(X1_proj, axis=0)
    cov1 = np.cov(X1_proj, rowvar=False)
    
    mean2 = np.mean(X2_proj, axis=0)
    cov2 = np.cov(X2_proj, rowvar=False)
    
    # Handle 1D case (numpy returns scalar for cov)
    if target_dim == 1:
        cov1 = np.array([[cov1]]) if np.ndim(cov1)==0 else cov1
        cov2 = np.array([[cov2]]) if np.ndim(cov2)==0 else cov2

    # 4. Regularize (Add small noise to diagonal to prevent singular matrices)
    cov1 += np.eye(cov1.shape[0]) * 1e-4
    cov2 += np.eye(cov2.shape[0]) * 1e-4
    
    return kl_divergence_gaussians(mean1, cov1, mean2, cov2)

def kl_divergence_gaussians(mean1, cov1, mean2, cov2):
    """Standard Multivariate Normal KL Divergence."""
    try:
        cov2_inv = np.linalg.inv(cov2)
        term1 = np.trace(cov2_inv @ cov1)
        term2 = (mean2 - mean1).T @ cov2_inv @ (mean2 - mean1)
        
        # Use slogdet for better numerical stability
        sign2, logdet2 = np.linalg.slogdet(cov2)
        sign1, logdet1 = np.linalg.slogdet(cov1)
        
        if sign1 != 1 or sign2 != 1:
            return float('nan') # Invalid covariance

        term3 = logdet2 - logdet1
        k = len(mean1)
        result = 0.5 * (term1 + term2 - k + term3)
        return max(0.0, result) # KL cannot be negative
    except Exception as e:
        print(f"Warning: KL calculation failed: {e}")
        return float('nan')

def analyze(args):
    print(f"--- Analyzing data from: {args.data_dir} ---")
    X = np.load(os.path.join(args.data_dir, "all_vectors.npy"))
    y = np.load(os.path.join(args.data_dir, "all_labels.npy"))
    class_names = args.class_names

    print(f"Total Samples: {len(X)}")
    print(f"Dimensions: {X.shape[1]}")
    for i, name in enumerate(class_names):
        print(f"  Class '{name}': {np.sum(y==i)} samples")

    # 1. Dimensionality Reduction and Visualization (PCA)
    print("Performing PCA for visualization...")
    # Handle case with very few samples for visualization
    n_viz_components = min(len(X), 2)
    pca = PCA(n_components=n_viz_components)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        if n_viz_components == 1:
            # 1D scatter plot (points on a line)
            plt.scatter(X_2d[y == i, 0], np.zeros_like(X_2d[y == i, 0]), label=class_name, alpha=0.7)
        else:
            plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=class_name, alpha=0.7)
    
    plt.title(f"Concept Separability for {args.data_dir.split('/')[-1]}")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(args.data_dir, "concept_separability_pca.png")
    plt.savefig(plot_path)
    print(f"Saved PCA visualization to: {plot_path}")
    plt.close()

    # 2. Clustering for Quantitative Separability
    print("Performing KMeans clustering...")
    if len(X) >= len(class_names):
        kmeans = KMeans(n_clusters=len(class_names), random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        ari_score = adjusted_rand_score(y, cluster_labels)
        
        # Silhouette requires at least 2 labels and > 2 samples
        if len(np.unique(cluster_labels)) > 1 and len(X) > 2:
            sil_score = silhouette_score(X, y)
        else:
            sil_score = 0.0
    else:
        ari_score = 0.0
        sil_score = 0.0
    
    print(f"\n--- Quantitative Metrics ---")
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    print(f"Silhouette Score: {sil_score:.4f}")

    # 3. Robust KL Divergence
    print("Calculating Robust KL Divergence...")
    if len(class_names) == 2:
        concepts1 = X[y == 0]
        concepts2 = X[y == 1]
        
        kl_1_2 = calculate_robust_kl(concepts1, concepts2)
        kl_2_1 = calculate_robust_kl(concepts2, concepts1)
        
        print(f"KL Divergence (KL(Class1 || Class2)): {kl_1_2:.4f}")
        print(f"KL Divergence (KL(Class2 || Class1)): {kl_2_1:.4f}")
    else:
        kl_1_2 = 0.0
        kl_2_1 = 0.0
        print("Skipping KL for > 2 classes.")

    # Save metrics
    with open(os.path.join(args.data_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Total Samples: {len(X)}\n")
        f.write(f"Adjusted Rand Index: {ari_score:.4f}\n")
        f.write(f"Silhouette Score: {sil_score:.4f}\n")
        f.write(f"KL(Class1 || Class2): {kl_1_2:.4f}\n")
        f.write(f"KL(Class2 || Class1): {kl_2_1:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the aggregated data directory.")
    parser.add_argument('--class_names', nargs='+', default=['pickup', 'garbage_truck'])
    args = parser.parse_args()
    analyze(args)
