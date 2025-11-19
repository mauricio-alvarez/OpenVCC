# analyze_separability.py
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.linalg import sqrtm

def kl_divergence_gaussians(mean1, cov1, mean2, cov2):
    """Calculate KL divergence between two multivariate Gaussians."""
    cov2_inv = np.linalg.inv(cov2)
    term1 = np.trace(cov2_inv @ cov1)
    term2 = (mean2 - mean1).T @ cov2_inv @ (mean2 - mean1)
    term3 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    return 0.5 * (term1 + term2 - len(mean1) + term3)

def analyze(args):
    print(f"--- Analyzing data from: {args.data_dir} ---")
    X = np.load(os.path.join(args.data_dir, "all_vectors.npy"))
    y = np.load(os.path.join(args.data_dir, "all_labels.npy"))
    class_names = args.class_names

    # 1. Dimensionality Reduction and Visualization (using PCA)
    print("Performing PCA for visualization...")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=class_name, alpha=0.7)
    
    plt.title(f"Concept Separability for {args.data_dir.split('/')[1]}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(args.data_dir, "concept_separability_pca.png")
    plt.savefig(plot_path)
    print(f"Saved PCA visualization to: {plot_path}")
    plt.close()

    # 2. Clustering for Quantitative Separability
    print("Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=len(class_names), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    ari_score = adjusted_rand_score(y, cluster_labels)
    sil_score = silhouette_score(X, y) # Using true labels to measure separation
    
    print(f"\n--- Quantitative Metrics ---")
    print(f"Adjusted Rand Index (Clustering vs. True Labels): {ari_score:.4f}")
    print(f"Silhouette Score (Based on True Labels): {sil_score:.4f}")

    # 3. KL Divergence
    print("Calculating KL Divergence...")
    concepts1 = X[y == 0]
    concepts2 = X[y == 1]
    
    mean1, cov1 = np.mean(concepts1, axis=0), np.cov(concepts1, rowvar=False)
    mean2, cov2 = np.mean(concepts2, axis=0), np.cov(concepts2, rowvar=False)
    
    # Add a small identity matrix to covariance to prevent singularity
    cov1 += np.identity(cov1.shape[0]) * 1e-6
    cov2 += np.identity(cov2.shape[0]) * 1e-6

    kl_1_2 = kl_divergence_gaussians(mean1, cov1, mean2, cov2)
    kl_2_1 = kl_divergence_gaussians(mean2, cov2, mean1, cov1)
    
    print(f"KL Divergence (KL(Class1 || Class2)): {kl_1_2:.4f}")
    print(f"KL Divergence (KL(Class2 || Class1)): {kl_2_1:.4f}")

    # Save metrics to a file
    with open(os.path.join(args.data_dir, 'metrics.txt'), 'w') as f:
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
