import numpy as np
import os
import argparse
from scipy.stats import entropy
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_data(vectors_path, labels_path):
    print(f"Loading data from {vectors_path}...")
    X = np.load(vectors_path)
    y = np.load(labels_path)
    return X, y

def kl_divergence_gaussians(mean1, cov1, mean2, cov2):
    """
    Computes KL(P || Q) where P ~ N(mean1, cov1) and Q ~ N(mean2, cov2).
    """
    n = len(mean1)
    
    # Add small epsilon to diagonal for numerical stability (regularization)
    cov1_reg = cov1 + np.eye(n) * 1e-5
    cov2_reg = cov2 + np.eye(n) * 1e-5

    try:
        cov2_inv = np.linalg.inv(cov2_reg)
        term1 = np.trace(cov2_inv @ cov1_reg)
        term2 = (mean2 - mean1).T @ cov2_inv @ (mean2 - mean1)
        
        sign2, logdet2 = np.linalg.slogdet(cov2_reg)
        sign1, logdet1 = np.linalg.slogdet(cov1_reg)
        
        term3 = logdet2 - logdet1
        
        kl = 0.5 * (term1 + term2 - n + term3)
        return max(0.0, kl) # Clamp negative precision errors to 0
    except np.linalg.LinAlgError:
        return np.nan

def generate_kl_matrix(X, y, class_count=16):
    print(f"Calculating KL Divergence Matrix on {X.shape[1]} dim vectors...")
    
    # 1. Estimate Gaussian parameters (Mean, Covariance) for each class
    means = []
    covs = []
    
    for i in range(class_count):
        data_i = X[y == i]
        if len(data_i) < 2:
            print(f"Warning: Class {i} has insufficient samples.")
            means.append(np.zeros(X.shape[1]))
            covs.append(np.eye(X.shape[1]))
            continue
            
        mu = np.mean(data_i, axis=0)
        sigma = np.cov(data_i, rowvar=False)
        
        # Handle 1D case (unlikely here but good practice)
        if X.shape[1] == 1:
            sigma = np.array([[np.var(data_i)]])
            
        means.append(mu)
        covs.append(sigma)

    # 2. Compute Pairwise KL Matrix
    kl_matrix = np.zeros((class_count, class_count))
    
    for i in range(class_count):
        for j in range(class_count):
            if i == j:
                kl_matrix[i, j] = 0.0
            else:
                # KL(Class i || Class j)
                kl_matrix[i, j] = kl_divergence_gaussians(means[i], covs[i], means[j], covs[j])

    return kl_matrix

def save_numerical_results(kl_matrix, save_dir):
    # Print to Console
    print("\n--- Pairwise KL Divergence Matrix ---")
    #print(np.array2string(kl_matrix, precision=4, suppress_small=True, linewidth=200))

    # Save to CSV
    save_path = os.path.join(save_dir, "kl_divergence_matrix.csv")
    np.savetxt(save_path, kl_matrix, delimiter=",", fmt="%.6f")
    print(f"\nMatrix saved to: {save_path}")

def main():
    # --- CONFIG ---
    VECTORS_FILE = '/home/mauricio.alvarez/tesis/VCC/analysis/global_activations_double_doubleshvit_L3/all_vectors.npy'
    LABELS_FILE = '/home/mauricio.alvarez/tesis/VCC/analysis/global_activations_double_doubleshvit_L3/all_labels.npy'
    OUTPUT_DIR = '.' # Current directory
    # --------------

    if not os.path.exists(VECTORS_FILE):
        print("Files not found. Please ensure .npy files are in the folder.")
        return

    X, y = load_data(VECTORS_FILE, LABELS_FILE)
    num_classes = len(np.unique(y))
    print(f"Data Shape: {X.shape}, Classes: {num_classes}")

    # ==========================================
    # ANALYSIS: RAW VECTORS ONLY
    # ==========================================
    
    # 1. Compute KL Matrix using full high-dimensional data
    kl_raw = generate_kl_matrix(X, y, num_classes)

    # 2. Save and Print Results
    save_numerical_results(kl_raw, OUTPUT_DIR)

if __name__ == "__main__":
    main()
