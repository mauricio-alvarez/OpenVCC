# aggregate_concepts_v2.py
import numpy as np
import os
import argparse

def aggregate(args):
    all_vectors = []
    all_labels = []

    print(f"--- Aggregating Concepts ---")
    for i, (data_dir, class_name) in enumerate(zip(args.input_dirs, args.class_names)):
        centers_file = os.path.join(data_dir, f"centers_{args.layer}.npy")

        if os.path.exists(centers_file):
            vectors = np.load(centers_file)
            print(f"Class '{class_name}': Found {len(vectors)} concept vectors.")
            
            all_vectors.append(vectors)
            all_labels.append(np.full(len(vectors), i))
        else:
            print(f"FATAL ERROR: File not found: {centers_file}")
            return

    if not all_vectors:
        print("No data found.")
        return

    # Combine
    X = np.concatenate(all_vectors, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "all_vectors.npy"), X)
    np.save(os.path.join(args.output_dir, "all_labels.npy"), y)

    print(f"\n--- Summary ---")
    print(f"Total vectors aggregated: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Saved to: {args.output_dir}")
    
    if len(X) < 10:
        print("\n!!! WARNING !!!")
        print(f"You only have {len(X)} vectors total.")
        print("The analysis script will attempt to handle this using dimensionality reduction,")
        print("but results may be statistically weak.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+', required=True, help="List of directories containing the centers.npy files.")
    parser.add_argument('--class_names', nargs='+', required=True, help="List of class names.")
    parser.add_argument('--layer', type=int, required=True, help="Layer number (e.g., 23).")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory.")
    args = parser.parse_args()
    aggregate(args)
