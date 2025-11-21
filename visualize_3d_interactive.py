import numpy as np
import os
import argparse
import pandas as pd
import plotly.express as px
import umap
from sklearn.decomposition import PCA

def visualize_3d(args):
    print(f"--- Loading data from: {args.data_dir} ---")
    X = np.load(os.path.join(args.data_dir, "all_vectors.npy"))
    y = np.load(os.path.join(args.data_dir, "all_labels.npy"))
    
    # Handle class names
    if args.class_names:
        class_names = args.class_names
        # Map numeric labels to string names for the plot
        labels_str = [class_names[int(i)] for i in y]
    else:
        labels_str = [f"Class {int(i)}" for i in y]

    print(f"Total Samples: {len(X)}")
    
    # Create a DataFrame for Plotly (makes handling data easier)
    df = pd.DataFrame({'label': labels_str})

    # --- 1. 3D PCA ---
    print("Calculating 3D PCA...")
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)
    
    df['pca_x'] = pca_result[:, 0]
    df['pca_y'] = pca_result[:, 1]
    df['pca_z'] = pca_result[:, 2]
    
    print("Generating Interactive PCA Plot...")
    fig_pca = px.scatter_3d(
        df, x='pca_x', y='pca_y', z='pca_z',
        color='label',
        title=f"3D PCA: {args.data_dir.split('/')[-1]}",
        opacity=0.7,
        size_max=5  # Adjust dot size
    )
    # Make the dots smaller so we can see structure
    fig_pca.update_traces(marker=dict(size=3))
    
    output_pca = os.path.join(args.data_dir, "interactive_3d_pca.html")
    fig_pca.write_html(output_pca)
    print(f"Saved interactive PCA to: {output_pca}")

    # --- 2. 3D UMAP ---
    print("Calculating 3D UMAP (Metric: Cosine)...")
    # n_neighbors: larger values (e.g. 30-50) preserve more global structure
    # min_dist: smaller values (0.1) make tighter clusters
    reducer = umap.UMAP(
        n_components=3, 
        n_neighbors=30, 
        min_dist=0.1, 
        metric='cosine', # Important for deep learning embeddings
        random_state=42
    )
    umap_result = reducer.fit_transform(X)
    
    df['umap_x'] = umap_result[:, 0]
    df['umap_y'] = umap_result[:, 1]
    df['umap_z'] = umap_result[:, 2]

    print("Generating Interactive UMAP Plot...")
    fig_umap = px.scatter_3d(
        df, x='umap_x', y='umap_y', z='umap_z',
        color='label',
        title=f"3D UMAP: {args.data_dir.split('/')[-1]}",
        opacity=0.7
    )
    fig_umap.update_traces(marker=dict(size=3))
    
    output_umap = os.path.join(args.data_dir, "interactive_3d_umap.html")
    fig_umap.write_html(output_umap)
    print(f"Saved interactive UMAP to: {output_umap}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to folder with .npy files")
    parser.add_argument('--class_names', nargs='+', default=None, help="List of class names")
    args = parser.parse_args()
    visualize_3d(args)
