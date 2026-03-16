import argparse
import torch
import torch.nn as nn
import timm
from timm.data import create_dataset, create_loader, resolve_data_config
import math
from tqdm import tqdm
import sys
import os

# --- HELPER FUNCTIONS (Added locally to fix ImportErrors) ---
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# --- 1. CONFIGURATION & ARGS ---
def get_args():
    parser = argparse.ArgumentParser(description='ViT Iterative Head Pruning')
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, 
                        help='Model name (e.g., vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224)')
    parser.add_argument('--data_dir', default='/path/to/imagenet/val', type=str,
                        help='Path to validation dataset')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--iterations', default=5, type=int,
                        help='Number of iterations for pruning')
    parser.add_argument('--calibration_batches', default=20, type=int,
                        help='Number of batches to use for calculating Head Importance')
    return parser.parse_args()

# --- 2. THE IMPORTANCE TRACKER ---
class HeadImportanceTracker:
    def __init__(self, model):
        self.model = model
        self.importance_scores = {}
        self.imp_hooks = []
        self.pruning_hooks = []
        self.masks = {}
        
        # 1. Get the Attention block from the first layer
        first_attn = model.blocks[0].attn
        
        # 2. Get number of heads
        self.num_heads = first_attn.num_heads
        
        # 3. Calculate head_dim manually (Total Embedding Dim / Num Heads)
        self.embed_dim = first_attn.qkv.in_features 
        self.head_dim = self.embed_dim // self.num_heads

        self.attn_layers = []
        for i, block in enumerate(model.blocks):
            # We hook the projection layer: block.attn.proj
            self.attn_layers.append((i, block.attn.proj))
            self.importance_scores[i] = torch.zeros(self.num_heads).cuda()
            # Initialize mask to all ones (active)
            self.masks[i] = torch.ones(1, 1, self.num_heads, 1).cuda()

    def reset_scores(self):
        """Reset importance scores to zero."""
        for i in self.importance_scores:
            self.importance_scores[i].zero_()

    def register_importance_hooks(self):
        """Register hooks to capture gradients flowing into the projection layer."""
        for layer_idx, module in self.attn_layers:
            # We use register_full_backward_hook to capture gradients
            h = module.register_full_backward_hook(self._get_grad_hook(layer_idx))
            self.imp_hooks.append(h)

    def _get_grad_hook(self, layer_idx):
        def hook(module, grad_input, grad_output):
            # grad_input usually corresponds to (input_grad, weight_grad, bias_grad)
            g = grad_input[0] 
            
            # Safety check for shape
            if g is None or g.dim() != 3:
                 for tensor in grad_input:
                    if tensor is not None and tensor.dim() == 3:
                        g = tensor
                        break
            
            if g is None: return

            # g shape is [Batch, Tokens, Hidden_Dim]
            # Reshape to [Batch, Tokens, Num_Heads, Head_Dim]
            B, N, C = g.shape
            g_reshaped = g.view(B, N, self.num_heads, self.head_dim)
            
            # Importance = Sum of absolute gradients
            score = g_reshaped.abs().sum(dim=(0, 1, 3))
            self.importance_scores[layer_idx] += score
        return hook

    def remove_importance_hooks(self):
        """Remove only importance calculation hooks."""
        for h in self.imp_hooks:
            h.remove()
        self.imp_hooks = []

    def prune_least_important(self, n_remove=2):
        """Finds the least important active heads and updates masks."""
        print(f"\n--- Pruning {n_remove} Least Important Heads per Layer ---")
        for layer_idx, scores in self.importance_scores.items():
            # Get current mask (1D)
            current_mask = self.masks[layer_idx].squeeze().view(-1)
            
            # Create candidates tensor on the same device
            candidates = scores.clone()
            
            # Set scores of already pruned heads to infinity so they are not picked
            # We ensure current_mask is on correct device for boolean indexing
            if current_mask.device != candidates.device:
                current_mask = current_mask.to(candidates.device)
                
            candidates[current_mask == 0] = float('inf')
            
            # Check if we have enough heads to prune
            num_active = (current_mask == 1).sum().item()
            actual_remove = min(n_remove, int(num_active))
            
            if actual_remove == 0:
                print(f"Layer {layer_idx}: No active heads left.")
                continue
                
            # Find indices of heads with smallest importance scores
            _, indices = torch.topk(candidates, actual_remove, largest=False)
            
            # Update mask
            for idx in indices:
                self.masks[layer_idx][:, :, idx, :] = 0.0
                print(f"Layer {layer_idx}: Pruning Head {idx.item()} (Score: {scores[idx].item():.2e})")

    def register_pruning_hooks(self):
        """Registers forward hooks that apply the mask during inference."""
        # Only register once
        if len(self.pruning_hooks) > 0:
            return

        for layer_idx, module in self.attn_layers:
            h = module.register_forward_pre_hook(self._get_mask_hook(layer_idx))
            self.pruning_hooks.append(h)

    def _get_mask_hook(self, layer_idx):
        def hook(module, args):
            x = args[0] # Input to projection layer: (Batch, Tokens, Hidden)
            B, N, C = x.shape
            
            # Reshape -> Mask -> Reshape
            x = x.view(B, N, self.num_heads, self.head_dim)
            x = x * self.masks[layer_idx]
            x = x.view(B, N, C)
            
            return (x,)
        return hook

# --- 3. MAIN EXECUTION ---
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device} with model: {args.model}")

    # Load Model
    model = timm.create_model(args.model, pretrained=True)
    model.to(device)
    model.eval()

    # Load Data
    data_config = resolve_data_config(vars(args), model=model)
    loader = create_loader(
        create_dataset('', root=args.data_dir, split='test'),
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        is_training=False,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers
    )

    tracker = HeadImportanceTracker(model)
    # Register pruning hooks immediately (masks are all 1s initially)
    tracker.register_pruning_hooks()
    
    criterion = nn.CrossEntropyLoss()
    
    # Pruning Loop: 5 Iterations
    # In each iteration: Calibrate -> Prune 2 heads -> Evaluate
    
    total_iterations = args.iterations
    heads_to_remove = 1
    print(f"Phase 0: Full Architecture Evaluation...")
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for input, target in tqdm(loader, desc="Eval"):
            input, target = input.to(device), target.to(device)
            output = model(input)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

    print(f"\nRESULTS FULL ARCHITECTURE")
    print(f"Top-1 Accuracy: {top1.avg:.2f}%")
    print(f"Top-5 Accuracy: {top5.avg:.2f}%")
    
    for iteration in range(1, total_iterations + 1):
        print(f"\n{'='*40}")
        print(f"ITERATION {iteration}/{total_iterations}")
        print(f"{'='*40}")
        
        # --- PHASE 1: CALIBRATION ---
        print(f"Phase 1: Calculating Head Importance...")
        tracker.reset_scores()
        tracker.register_importance_hooks()
        
        for i, (input, target) in enumerate(tqdm(loader, total=args.calibration_batches, desc="Calibrating")):
            if i >= args.calibration_batches: break
            
            input, target = input.to(device), target.to(device)
            model.zero_grad()
            
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

        tracker.remove_importance_hooks()

        # --- PHASE 2: PRUNING ---
        tracker.prune_least_important(heads_to_remove)

        # --- PHASE 3: EVALUATION ---
        print(f"Phase 3: Evaluating...")
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        with torch.no_grad():
            for input, target in tqdm(loader, desc="Eval"):
                input, target = input.to(device), target.to(device)
                output = model(input)
                
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

        print(f"\nRESULTS FOR ITERATION {iteration}")
        print(f"Top-1 Accuracy: {top1.avg:.2f}%")
        print(f"Top-5 Accuracy: {top5.avg:.2f}%")

    print("\n" + "="*30)
    print("Done.")

if __name__ == '__main__':
    main()