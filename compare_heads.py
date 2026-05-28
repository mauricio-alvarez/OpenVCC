import torch
import torch.nn.functional as F
import sys

ckpt_path = sys.argv[1]
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

def get_cosine_sim(tensor_a, tensor_b):
    return F.cosine_similarity(tensor_a.flatten().unsqueeze(0), tensor_b.flatten().unsqueeze(0)).item()

print(f"--- Analyzing Checkpoint: {ckpt_path} ---")

if 'head_A.l.weight' in state_dict and 'head_B.l.weight' in state_dict:
    wA = state_dict['head_A.l.weight']
    wB = state_dict['head_B.l.weight']
    print(f"Head_A vs Head_B Weight Cosine Sim: {get_cosine_sim(wA, wB):.4f}")

b3a = state_dict.get('blocks3_A.4.conv.m.c.weight')
b3b = state_dict.get('blocks3_B.4.conv.m.c.weight')
if b3a is not None and b3b is not None:
    print(f"Blocks3.4.conv.m.c Weight Cosine Sim: {get_cosine_sim(b3a, b3b):.4f}")
