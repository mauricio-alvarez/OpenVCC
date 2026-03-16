import torch
import torch.nn as nn
import torch.optim as optim
import timm
import math
import argparse
import os
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

# =========================================================================
# 1. The Hard Concrete Mask
# =========================================================================
class L0Mask(nn.Module):
    def __init__(self, num_heads, temp=2./3., limit_l=-0.1, limit_r=1.1):
        super().__init__()
        self.num_heads = num_heads
        self.temp = temp
        self.limit_l = limit_l
        self.limit_r = limit_r
        self.log_alpha = nn.Parameter(torch.Tensor(num_heads))
        nn.init.constant_(self.log_alpha, 3.0) 

    def forward(self, training=True):
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((self.log_alpha + torch.log(u) - torch.log(1 - u)) / self.temp)
            s_bar = s * (self.limit_r - self.limit_l) + self.limit_l
            z = torch.clamp(s_bar, min=0.0, max=1.0)
        else:
            s = torch.sigmoid(self.log_alpha)
            s_bar = s * (self.limit_r - self.limit_l) + self.limit_l
            z = torch.clamp(s_bar, min=0.0, max=1.0)
            z = (z > 0.0).float()
        return z

    def get_expected_active_heads(self):
        term = self.log_alpha - self.temp * math.log(-self.limit_l / self.limit_r)
        return torch.sigmoid(term).sum()

class PrunableAttention(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.scale = original_attn.scale
        self.qkv = original_attn.qkv
        self.attn_drop = original_attn.attn_drop
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop
        self.l0_mask = L0Mask(self.num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        mask = self.l0_mask(training=self.training).view(1, self.num_heads, 1, 1)
        x = x * mask
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# =========================================================================
# 2. Main Execution
# =========================================================================
def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # A. Data
    print(f"Loading data from {args.data_path}...")
    transform_train = timm.data.create_transform(224, is_training=True, auto_augment='rand-m9-mstd0.5')
    full_dataset = ImageFolder(args.data_path, transform=transform_train)
    target_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label < 100]
    train_loader = DataLoader(Subset(full_dataset, target_indices), batch_size=args.batch_size, shuffle=True, num_workers=4)

    # B. Model
    print(f"Loading Model Skeleton: {args.model}")
    model = timm.create_model(args.model, pretrained=True)
    model.to(device)

    total_model_heads = 0
    print("Injecting L0 Masks...")
    for block in model.blocks:
        total_model_heads += block.attn.num_heads
        block.attn = PrunableAttention(block.attn).to(device)

    # Load Previous Checkpoint
    if args.initial_checkpoint and os.path.exists(args.initial_checkpoint):
        print(f"--> LOADING PREVIOUS STATE from: {args.initial_checkpoint}")
        state_dict = torch.load(args.initial_checkpoint, map_location=device)
        if 'model' in state_dict: state_dict = state_dict['model']
        model.load_state_dict(state_dict)

    print(f"Total Heads: {total_model_heads} | Target: {args.target_heads} (+/- {args.tolerance})")

    # C. Optimizer
    # FIX: We allow weights to train slightly (lr=1e-5) to recover accuracy
    params_masks = [p for n, p in model.named_parameters() if "l0_mask" in n]
    params_weights = [p for n, p in model.named_parameters() if "l0_mask" not in n]

    optimizer = optim.AdamW([
        {'params': params_weights, 'lr': 1e-5, 'weight_decay': 1e-4}, # Low LR for healing
        {'params': params_masks,   'lr': 0.1,  'weight_decay': 0.0}   # High LR for pruning
    ])
    
    criterion = nn.CrossEntropyLoss()

    # D. Training Loop
    initial_temp = 2.0
    final_temp = 0.1
    
    success_counter = 0 # How many epochs have we been in the target zone?
    
    print("Starting Training...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_penalty = 0
        correct = 0
        total = 0
        
        current_temp = initial_temp - (initial_temp - final_temp) * (epoch / args.epochs)
        for block in model.blocks:
            block.attn.l0_mask.temp = current_temp
            
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            
            loss_task = criterion(output, labels)
            
            # Barrier Penalty
            expected_heads = sum([b.attn.l0_mask.get_expected_active_heads() for b in model.blocks])
            excess = expected_heads - args.target_heads
            
            loss_sparsity = args.lambda_l0 * (excess ** 2)
            
            loss = loss_task + loss_sparsity
            loss.backward()
            optimizer.step()
            
            total_penalty += loss_sparsity.item()
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        
        # --- EARLY STOPPING CHECK ---
        active_heads_hard = 0
        with torch.no_grad():
            for block in model.blocks:
                mask = block.attn.l0_mask(training=False)
                active_heads_hard += mask.sum().item()
        
        active_heads_hard = int(active_heads_hard)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Temp: {current_temp:.2f} | Acc: {train_acc:.2f}% | Heads: {active_heads_hard}/{total_model_heads}")

        lower_bound = args.target_heads - args.tolerance
        upper_bound = args.target_heads + args.tolerance

        if lower_bound <= active_heads_hard <= upper_bound:
            success_counter += 1
            print(f"   >>> Target Reached ({success_counter}/{args.patience})")
            
            # Save strictly whenever we hit the target
            torch.save(model.state_dict(), args.output_path)
            
            if success_counter >= args.patience:
                print("   >>> Early Stopping triggered. Optimization Complete.")
                break
        else:
            success_counter = 0 # Reset if we drift out (unlikely with barrier loss)

    if success_counter < args.patience:
        print("Warning: Max epochs reached without stable convergence. Saving final state.")
        torch.save(model.state_dict(), args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_tiny_patch16_224.augreg2_in21k_ft_in1k")
    parser.add_argument("--epochs", type=int, default=10) # Max limit
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_heads", type=int, required=True)
    parser.add_argument("--lambda_l0", type=float, default=0.2)
    parser.add_argument("--tolerance", type=int, default=2, help="Acceptable deviation (+/- heads)")
    parser.add_argument("--patience", type=int, default=2, help="Epochs to hold target before stopping")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--initial_checkpoint", type=str, default="")
    args = parser.parse_args()
    
    run_experiment(args)