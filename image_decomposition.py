# image_decomposition.py (Corrected Version 2)
import os
import sys
import argparse
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from vcc import ConceptDiscovery, make_model, features_blobs, cls_token_blobs
from vcc_helpers import load_image_from_file

def debug_single_image(args):
    print(f"--- Loading model '{args.model_to_run}' ---")
    cd_args = argparse.Namespace(
        model_to_run=args.model_to_run,
        pretrained=True,
        feature_names=args.feature_names,
        target_dataset='imagenet',
        labels_path='./imagenet_class_index.json',
        use_elbow=False,
        sp_method='KM',
        cluster_parallel_workers=0,
        average_image_value=117
    )
    model = make_model(cd_args, hook=True)
    cd = ConceptDiscovery(cd_args, model, '', '', [], '', '', '', '')

    print(f"--- Loading and processing image: {args.image_path} ---")
    image = load_image_from_file(args.image_path, shape=[224, 224])
    image_tensor = torch.tensor(image).unsqueeze(0).cuda().permute(0, 3, 1, 2)
    image_tensor = torchvision.transforms.functional.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).float()

    with torch.no_grad():
        _ = model(image_tensor)

    activations = {}
    for bn in args.feature_names:
        activations[bn] = features_blobs[0].transpose(0, 2, 3, 1)
        features_blobs.pop(0)
        if 'vit' in args.model_to_run:
            cls_token_blobs.pop(0)
    
    single_img_acts = {layer: act[0] for layer, act in activations.items()}
    
    # <<< --- FIX 1: CALCULATE THE MISSING BN_SHAPES DICTIONARY --- >>>
    print("--- Calculating bottleneck shapes... ---")
    bn_shapes = {}
    for bn, act in activations.items():
        # Shape is (batch, H, W, C), so we want (H, W, C)
        bn_shapes[bn] = act.shape[1:]

    print("--- Generating top-down segmentation... ---")
    bn_superpixels, _, _, _ = cd._return_top_down_img_segments(
        img=image,
        single_img_acts=single_img_acts,
        n_top_clusters=args.num_clusters,
        bn_shapes=bn_shapes  # <<< --- FIX 2: PASS THE DICTIONARY TO THE FUNCTION --- >>>
    )

    print(f"--- Saving segmentation results to: {args.output_dir} ---")
    os.makedirs(args.output_dir, exist_ok=True)
    
    plt.imsave(os.path.join(args.output_dir, 'original_image.png'), image)

    for bn, patches in bn_superpixels.items():
        bn_dir = os.path.join(args.output_dir, f'layer_{bn}')
        os.makedirs(bn_dir, exist_ok=True)
        for i, patch_img in enumerate(patches):
            save_path = os.path.join(bn_dir, f'concept_{i}.png')
            plt.imsave(save_path, patch_img)

    print("\n--- Debugging complete! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate top-down segmentation for a single image.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the misclassified image.")
    parser.add_argument('--model_to_run', type=str, default='vit_b', help="The model to debug.")
    parser.add_argument('--feature_names', nargs='+', default=['2', '5', '8', '10'], help="Feature layers for the model.")
    parser.add_argument('--output_dir', type=str, default='outputs/decomposition', help="Directory to save the segmented images.")
    parser.add_argument('--num_clusters', type=int, default=3, help="Number of concepts to segment at the top layer.")

    args = parser.parse_args()
    debug_single_image(args)
