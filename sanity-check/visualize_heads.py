import torch
import torch.nn as nn
import timm
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================
# 1. Required Classes (Must match the training script exactly)
# =========================================================================
class L0Mask(nn.Module):
    def __init__(self, num_heads, temp=2./3., limit_l=-0.1, limit_r=1.1):
        super().__init__()
        self.num_heads = num_heads
        self.log_alpha = nn.Parameter(torch.Tensor(num_heads))

    def forward(self, training=True):
        # Deterministic Evaluation Logic
        s = torch.sigmoid(self.log_alpha)
        # Standard DSP logic: stretch and clamp
        s_bar = s * (1.1 - (-0.1)) + (-0.1)
        z = torch.clamp(s_bar, min=0.0, max=1.0)
        # Hard threshold
        z = (z > 0.0).float()
        return z

class PrunableAttention(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.l0_mask = L0Mask(self.num_heads)
    def forward(self, x): return x 

# =========================================================================
# 2. Visualization Logic
# =========================================================================
def visualize_sparsity(args):
    print(f"Loading architecture: {args.model}")
    
    # A. Create the base architecture
    model = timm.create_model(args.model, pretrained=False)
    
    # B. Patch
    for block in model.blocks:
        block.attn = PrunableAttention(block.attn)
        
    # C. Load Checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading pruned weights from: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in state_dict: state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # D. Extract Counts per Layer
    layer_counts = []
    total_possible_heads_per_layer = model.blocks[0].attn.num_heads
    
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            mask = block.attn.l0_mask(training=False)
            active_count = mask.sum().item()
            layer_counts.append(int(active_count))

    # E. Construct the "Bar Graph" Heatmap Matrix
    # Shape: [Max_Heads, Num_Layers]
    # We fill it from bottom (0) to top (active_count)
    num_layers = len(layer_counts)
    grid = np.zeros((total_possible_heads_per_layer, num_layers))
    
    for layer_idx, count in enumerate(layer_counts):
        # Fill the bottom 'count' rows with a value (e.g., the count itself or just 1)
        # We use the count value to create a nice color gradient per layer
        if count > 0:
            grid[0:count, layer_idx] = count 

    # F. Plotting
    plt.figure(figsize=(12, 7))
    
    # Use a sequential colormap (e.g., 'viridis', 'magma', 'Blues')
    # Valid values (>0) get color, 0 gets the background color
    ax = sns.heatmap(
        grid,
        cmap="viridis", 
        annot=False,       # Show numbers inside cells? (Optional, remove if too cluttered)
        fmt=".0f",        # Format for annotation (integers)
        cbar=True,
        cbar_kws={'label': 'Active Heads Count'},
        mask=(grid==0),   # Hide the empty cells (make them white/transparent)
        linewidths=0.5,
        linecolor='lightgray'
    )
    
    # Set the background for masked (empty) values to distinct color
    ax.set_facecolor('#f0f0f0') # Light gray for empty space

    # Formatting
    # Invert Y-axis so 0 is at bottom (Bar graph style)
    ax.invert_yaxis()
    
    plt.title(f"Layer-wise Head Density\nModel: {args.model}", fontsize=16)
    plt.xlabel("Transformer Layer (Depth)", fontsize=14)
    plt.ylabel("Number of Active Heads", fontsize=14)
    
    # X-Ticks: Layer 0, Layer 1...
    plt.xticks(np.arange(num_layers) + 0.5, [f"L{i}" for i in range(num_layers)], rotation=0)
    
    # Y-Ticks: 1, 2, ... Max
    plt.yticks(np.arange(total_possible_heads_per_layer) + 0.5, np.arange(1, total_possible_heads_per_layer + 1))

    # Add text summary on top of bars
    for i, count in enumerate(layer_counts):
        # Place text slightly above the top-most filled block
        plt.text(i + 0.5, count + 0.2, str(count), 
                 ha='center', va='bottom', fontweight='bold', color='black')

    filename = f"density_{os.path.basename(args.checkpoint).replace('.pth', '.png')}"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    visualize_sparsity(args)