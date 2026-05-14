import torch
import sys

def check_symmetry(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print("\n--- Checking Symmetry in blocks2 ---")
    
    # We will sample a few weights from blocks2_A, blocks2_B, blocks2_C if they exist
    # Let's check the first Conv layer in the first BasicBlock
    # e.g., blocks2_A.0.conv.m.c.weight
    
    keys_A = [k for k in state_dict.keys() if 'blocks2_A.0.conv.m.c.weight' in k]
    keys_B = [k for k in state_dict.keys() if 'blocks2_B.0.conv.m.c.weight' in k]
    keys_C = [k for k in state_dict.keys() if 'blocks2_C.0.conv.m.c.weight' in k]

    if not keys_A or not keys_B:
        print("Could not find expected keys for blocks2_A and blocks2_B.")
        return

    wA = state_dict[keys_A[0]]
    wB = state_dict[keys_B[0]]
    
    print(f"Weight A ({keys_A[0]}) shape: {wA.shape}")
    print(f"Weight B ({keys_B[0]}) shape: {wB.shape}")
    
    # Print some random values
    print(f"wA[0, 0, 0, :5]:\n{wA[0, 0, 0, :5]}")
    print(f"wB[0, 0, 0, :5]:\n{wB[0, 0, 0, :5]}")
    
    diff_AB = torch.max(torch.abs(wA - wB)).item()
    print(f"\nMax absolute difference between A and B: {diff_AB}")
    
    if keys_C:
        wC = state_dict[keys_C[0]]
        diff_AC = torch.max(torch.abs(wA - wC)).item()
        print(f"Max absolute difference between A and C: {diff_AC}")
        print(f"wC[0, 0, 0, :5]:\n{wC[0, 0, 0, :5]}")

    print("\n--- Checking Symmetry in ds_1_to_2 ---")
    k_ds_A = [k for k in state_dict.keys() if 'ds_1_to_2_A.0.m.c.weight' in k][0]
    k_ds_B = [k for k in state_dict.keys() if 'ds_1_to_2_B.0.m.c.weight' in k][0]
    w_ds_A = state_dict[k_ds_A]
    w_ds_B = state_dict[k_ds_B]
    print(f"Max diff in downsample A vs B: {torch.max(torch.abs(w_ds_A - w_ds_B)).item()}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_symmetry(sys.argv[1])
    else:
        print("Please provide a checkpoint path.")
