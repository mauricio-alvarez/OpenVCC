# aggregate_concepts_v2.py
import numpy as np
import os
import argparse

def aggregate(args):
    all_vectors = []
    all_labels = []

    # Loop through the provided directories and class names
    for i, (data_dir, class_name) in enumerate(zip(args.input_dirs, args.class_names)):
        centers_file = os.path.join(data_dir, f"centers_{args.layer}.npy")

        if os.path.exists(centers_file):
            print(f"Loading concepts for class '{class_name}' from: {centers_file}")
            vectors = np.load(centers_file)
            all_vectors.append(vectors)
            # Assign a numeric label: 0 for the first class, 1 for the second
            all_labels.append(np.full(len(vectors), i))
        else:
            print(f"FATAL ERROR: File not found, cannot proceed: {centers_file}")
            return

    # Combine all vectors and labels into single arrays
    X = np.concatenate(all_vectors, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Create the output directory and save the aggregated data
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "all_vectors.npy"), X)
    np.save(os.path.join(args.output_dir, "all_labels.npy"), y)

    print(f"\nSuccessfully aggregated {len(X)} concept vectors.")
    print(f"Saved aggregated data to: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+', required=True, help="List of directories containing the centers.npy files.")
    parser.add_argument('--class_names', nargs='+', required=True, help="List of class names corresponding to the input directories.")
    parser.add_argument('--layer', type=int, required=True, help="The layer number that was analyzed (e.g., 23).")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the aggregated results.")
    args = parser.parse_args()
    aggregate(args)
