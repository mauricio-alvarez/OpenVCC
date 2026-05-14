import torch
import torch.nn.functional as F
import sys
import argparse

def compute_metrics(w1, w2):
    # Flatten weights to 1D vectors
    v1 = w1.view(-1)
    v2 = w2.view(-1)
    
    # Euclidean distance
    dist = torch.norm(v1 - v2, p=2).item()
    
    # Cosine Similarity
    if torch.norm(v1) == 0 or torch.norm(v2) == 0:
        cos_sim = 0.0
    else:
        cos_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        
    # Max Absolute Difference
    max_diff = torch.max(torch.abs(v1 - v2)).item()
    
    return dist, cos_sim, max_diff

def compare_checkpoints(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Check for Double or Triple head
    has_C = any('blocks2_C' in k for k in state_dict.keys())
    
    print("\n=========================================")
    print(f"Model Type: {'Triple-Head' if has_C else 'Double-Head'} SHViT")
    print("=========================================\n")

    layers_to_check = [
        '0.conv.m.c.weight', # First Conv
        '1.conv.m.c.weight', # Second Block Conv
    ]
    
    stages = ['blocks2', 'blocks3']

    for stage in stages:
        print(f"--- Analyzing {stage.upper()} ---")
        for layer_suffix in layers_to_check:
            key_A = f"{stage}_A.{layer_suffix}"
            key_B = f"{stage}_B.{layer_suffix}"
            key_C = f"{stage}_C.{layer_suffix}"
            
            # Find the actual keys matching this pattern
            actual_A = [k for k in state_dict.keys() if key_A in k]
            actual_B = [k for k in state_dict.keys() if key_B in k]
            
            if not actual_A or not actual_B:
                continue
                
            wA = state_dict[actual_A[0]]
            wB = state_dict[actual_B[0]]
            
            print(f"\nLayer: {actual_A[0].replace('_A.', '.*.')}")
            
            d_AB, c_AB, m_AB = compute_metrics(wA, wB)
            print(f"  Branch A vs B -> Cosine Sim: {c_AB:.4f} | Max Diff: {m_AB:.4f} | L2 Dist: {d_AB:.4f}")
            
            if has_C:
                actual_C = [k for k in state_dict.keys() if key_C in k]
                if actual_C:
                    wC = state_dict[actual_C[0]]
                    d_AC, c_AC, m_AC = compute_metrics(wA, wC)
                    d_BC, c_BC, m_BC = compute_metrics(wB, wC)
                    
                    print(f"  Branch A vs C -> Cosine Sim: {c_AC:.4f} | Max Diff: {m_AC:.4f} | L2 Dist: {d_AC:.4f}")
                    print(f"  Branch B vs C -> Cosine Sim: {c_BC:.4f} | Max Diff: {m_BC:.4f} | L2 Dist: {d_BC:.4f}")

    print("\n=========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare weights of different branches in SHViT models to check for symmetry breaking.")
    parser.add_argument("checkpoint", type=str, help="Path to the .pth checkpoint file")
    args = parser.parse_args()
    
    compare_checkpoints(args.checkpoint)
