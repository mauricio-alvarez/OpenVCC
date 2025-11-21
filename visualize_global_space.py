import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap
from sklearn.manifold import TSNE

def visualize(args):
    print(f"--- Loading data from: {args.data_dir} ---")
    X = np.load(os.path.join(args.data_dir, "all_vectors.npy"))
    y = np.load(os.path.join(args.data_dir, "all_labels.npy"))
    
    # We need to reconstruct class names. 
    # Assuming the generation script processed them in sorted order (standard behavior)
    # If you passed them via command line, we need that list. 
    # For now, we will label them Class 0-15 to allow running without the explicit list,
    # but you can pass --class_names to label them correctly.
    if args.class_names:
        class_names = args.class_names
    else:
        class_names = [f"Class {i}" for i in range(len(np.unique(y)))]

    print(f"Total Samples: {len(X)}")
    print(f"Feature Dimension: {X.shape[1]}")
    print(f"Number of Classes: {len(class_names)}")

    # --- 1. UMAP Visualization ---
    print("Running UMAP (this may take a moment)...")
    # metric='cosine' is usually better for deep learning embeddings than euclidean
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_umap = reducer.fit_transform(X)
    
    plot_scatter(embedding_umap, y, class_names, "UMAP Projection", 
                 os.path.join(args.data_dir, "global_umap.png"))

    # --- 2. t-SNE Visualization ---
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    embedding_tsne = tsne.fit_transform(X)
    
    plot_scatter(embedding_tsne, y, class_names, "t-SNE Projection", 
                 os.path.join(args.data_dir, "global_tsne.png"))

def plot_scatter(X_2d, y, class_names, title, save_path):
    plt.figure(figsize=(14, 10))
    
    # Use a distinct colormap (tab20 is good for up to 20 classes)
    unique_y = np.unique(y)
    colors = cm.tab20(np.linspace(0, 1, len(unique_y)))
    
    for i, class_idx in enumerate(unique_y):
        # Select points for this class
        points = X_2d[y == class_idx]
        
        # Plot
        plt.scatter(points[:, 0], points[:, 1], 
                    c=[colors[i]], 
                    label=class_names[int(class_idx)], 
                    s=10, alpha=0.7) # s is size, alpha is transparency
    
    plt.title(title, fontsize=18)
    # Place legend outside the plot so it doesn't cover data
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, markerscale=3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"Saved {title} to: {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to folder with .npy files")
    # Optional: Pass actual names if you want the legend to be readable
    parser.add_argument('--class_names', nargs='+', default=None, help="List of class names in correct order")
    args = parser.parse_args()
    visualize(args)
