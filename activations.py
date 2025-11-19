# generate_raw_activations.py
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from vcc import make_model, features_blobs, cls_token_blobs
from vcc_helpers import load_images_from_files

def generate_activations(args):
    print(f"--- Loading model '{args.model_to_run}' ---")
    model_args = argparse.Namespace(
        model_to_run=args.model_to_run,
        model_path=args.model_path,
        pretrained=True, # Will be ignored if model_path is used
        num_classes=args.num_classes,
        feature_names=[args.layer], # We only need to hook one layer
        target_layer=args.layer,
        target_dataset='custom',
        imagenet_path=args.dataset_path,
        labels_path='./imagenet_class_index.json' # Dummy path
    )
    model = make_model(model_args, hook=True)

    # Create the main output directory
    main_output_dir = f"analysis/raw_activations_{args.model_to_run}_L{args.layer}"
    os.makedirs(main_output_dir, exist_ok=True)
    
    all_vectors = []
    all_labels = []

    for i, class_name in enumerate(args.class_names):
        print(f"\n--- Processing class: {class_name} ---")
        class_dir = os.path.join(args.dataset_path, class_name)
        
        # Load all images for the class
        images = load_images_from_files(
            [os.path.join(class_dir, f) for f in os.listdir(class_dir)],
            max_imgs=args.max_imgs,
            shape=[224, 224]
        )
        
        class_activations = []
        batch_size = args.batch_size

        for j in tqdm(range(0, len(images), batch_size), desc=f"Getting activations for {class_name}"):
            img_batch = images[j : j + batch_size]
            img_tensor = torch.tensor(img_batch).cuda().permute(0, 3, 1, 2)
            img_tensor = torch.nn.functional.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).float()

            with torch.no_grad():
                _ = model(img_tensor)
            
            # The hook populates features_blobs
            raw_acts = features_blobs.pop(0)
            if 'vit' in args.model_to_run:
                cls_token_blobs.pop(0)

            # Average over all spatial/patch dimensions to get one vector per image
            avg_acts = np.mean(raw_acts, axis=tuple(range(2, raw_acts.ndim)))
            class_activations.append(avg_acts)
        
        # Aggregate and save
        class_vectors = np.concatenate(class_activations, axis=0)
        all_vectors.append(class_vectors)
        all_labels.append(np.full(len(class_vectors), i))

    # Save the final aggregated dataset
    X = np.concatenate(all_vectors, axis=0)
    y = np.concatenate(all_labels, axis=0)
    np.save(os.path.join(main_output_dir, "all_vectors.npy"), X)
    np.save(os.path.join(main_output_dir, "all_labels.npy"), y)

    print(f"\n--- Successfully saved {len(X)} raw activation vectors to: {main_output_dir} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_to_run', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--class_names', nargs='+', required=True)
    parser.add_argument('--num_classes', type=int, default=16)
    parser.add_argument('--max_imgs', type=int, default=200, help="Number of images per class to process.")
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    generate_activations(args)
