# generate_raw_activations.py (FIXED DIMENSIONS)
import os
import sys
import argparse
import numpy as np
import torch
import glob
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from vcc import make_model, features_blobs, cls_token_blobs

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def generate_activations(args):
    print(f"--- Setup: Model {args.model_to_run} on Layer {args.layer} ---")
    
    model_args = argparse.Namespace(
        model_to_run=args.model_to_run,
        model_path=args.model_path,
        pretrained=True,
        num_classes=args.num_classes,
        feature_names=[str(args.layer)],
        target_layer=args.layer,
        target_dataset='custom',
        imagenet_path=args.dataset_path,
        labels_path='./imagenet_class_index.json'
    )
    
    model = make_model(model_args, hook=True)
    transform = get_image_transform()
    
    all_vectors = []
    all_labels = []
    
    for class_idx, class_name in enumerate(args.class_names):
        class_dir = os.path.join(args.dataset_path, class_name)
        image_files = glob.glob(os.path.join(class_dir, "*"))
        print(f"Processing {len(image_files)} images for class '{class_name}'...")
        
        for i in tqdm(range(0, len(image_files), args.batch_size)):
            batch_files = image_files[i : i + args.batch_size]
            batch_tensors = []
            
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_tensors.append(transform(img))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

            if not batch_tensors: continue
            input_tensor = torch.stack(batch_tensors).cuda()
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            # raw_acts shape is typically (Batch, Channels, Height, Width) for PyTorch
            raw_acts = features_blobs.pop(0)
            
            if len(cls_token_blobs) > 0:
                cls_token_blobs.pop(0)
            
            # <<< --- THE CRITICAL FIX --- >>>
            # We must average over the spatial dimensions (Height and Width).
            # In PyTorch format (B, C, H, W), these are the last two dimensions.
            if raw_acts.ndim == 4:
                # Shape: (B, C, H, W) -> Average axes 2 and 3
                avg_acts = np.mean(raw_acts, axis=(2, 3))
            elif raw_acts.ndim == 3:
                # Shape: (B, N, C) -> ViT Format -> Average axis 1 (patches)
                avg_acts = np.mean(raw_acts, axis=1)
            else:
                # Already flattened or unexpected shape
                avg_acts = raw_acts
            # <<< --- END FIX --- >>>

            all_vectors.append(avg_acts)
            all_labels.append(np.full(len(avg_acts), class_idx))
            
    X = np.concatenate(all_vectors, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "all_vectors.npy"), X)
    np.save(os.path.join(output_dir, "all_labels.npy"), y)
    
    print(f"--- Done! Saved {len(X)} vectors to {output_dir} ---")
    print(f"--- VECTOR DIMENSION: {X.shape[1]} (Should be 1024, 768, etc., NOT 14) ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_to_run', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--class_names', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    generate_activations(args)
