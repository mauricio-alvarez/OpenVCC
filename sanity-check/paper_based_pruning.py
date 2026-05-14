import torch
import torch.nn as nn
import torch.optim as optim
import timm
import math
import argparse
import os
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

# 1. The Hard Concrete Mask (Standard DSP Logic)
class L0Mask(nn.Module):
    def __init__(self, num_heads, temp=2./3., limit_l=-0.1, limit_r=1.1):
        super().__init__()
        self.num_heads = num_heads
        self.temp = temp
        self.limit_l = limit_l
        self.limit_r = limit_r
        self.log_alpha = nn.Parameter(torch.Tensor(num_heads))
        nn.init.constant_(self.log_alpha, 3.0) # Start fully open

    def forward(self, training=True):
        if training:
            u = torch.rand_like(self.log_alpha).cuda()
            s = torch.sigmoid((self.log_alpha + torch.log(u) - torch.log(1 - u)) / self.temp)
            s_bar = s * (self.limit_r - self.limit_l) + self.limit_l
            z = torch.clamp(s_bar, min=0.0, max=1.0)
        else:
            s = torch.sigmoid(self.log_alpha)
            s_bar = s * (self.limit_r - self.limit_l) + self.limit_l
            z = torch.clamp(s_bar, min=0.0, max=1.0)
            z = (z > 0.0).float()
        return z

    def get_l0_penalty(self):
        term = self.log_alpha - self.temp * math.log(-self.limit_l / self.limit_r)
        return torch.sigmoid(term).sum()

# 2. Patching ViT Attention
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

        # Apply Mask
        mask = self.l0_mask(training=self.training)
        mask = mask.view(1, self.num_heads, 1, 1)
        x = x * mask

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 3. Main Execution
def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # --- A. Data Loading (Subset of ImageNet) ---
    print(f"Loading data from {args.data_path}...")
    transform_train = timm.data.create_transform(224, is_training=True, auto_augment='rand-m9-mstd0.5')
    
    # We load the FULL folder structure to ensure labels 0..999 are preserved
    full_dataset = ImageFolder(args.data_path, transform=transform_train)
    
    # We select specific classes (e.g., indices 0 to 99)
    # Subset simply filters the list; it does NOT re-map labels.
    # So if we pick image 10 from class 5, it still has label '5'.
    target_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label < 100]
    
    # Split Train/Val (80/20) for sanity check
    split = int(len(target_indices) * 0.8)
    train_idx = target_indices[:split]
    val_idx = target_indices[split:]
    
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train Size: {len(train_ds)} | Val Size: {len(val_ds)} | Classes: 100")
    print("NOTE: Using original 1000-class head. Target labels are in range [0, 99].")

    # --- B. Model Setup ---
    print(f"Loading Model: {args.model}")
    model = timm.create_model(args.model, pretrained=True)
    model.to(device)

    # Inject Masks
    print("Injecting L0 Masks into Attention Layers...")
    for block in model.blocks:
        block.attn = PrunableAttention(block.attn).to(device)

    # --- C. Optimizer ---
    # We freeze weights and ONLY train masks to ensure "Sanity Check" is pure pruning
    # (Or use very low LR for weights)
    params_masks = [p for n, p in model.named_parameters() if "l0_mask" in n]
    params_weights = [p for n, p in model.named_parameters() if "l0_mask" not in n]

    optimizer = optim.AdamW([
        {'params': params_weights, 'lr': 1e-5, 'weight_decay': 1e-4}, # Very slow fine-tuning
        {'params': params_masks,   'lr': 0.1,  'weight_decay': 0.0}   # Fast mask learning
    ])
    
    criterion = nn.CrossEntropyLoss()

    # --- D. Training Loop ---
    initial_temp = 2.0
    final_temp = 0.1
    
    print("Starting Pruning...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_l0 = 0
        correct = 0
        total = 0
        
        # Anneal Temperature
        current_temp = initial_temp - (initial_temp - final_temp) * (epoch / args.epochs)
        for block in model.blocks:
            block.attn.l0_mask.temp = current_temp
            
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(imgs) # Output shape: [Batch, 1000]
            
            loss_task = criterion(output, labels)
            loss_sparsity = sum([b.attn.l0_mask.get_l0_penalty() for b in model.blocks])
            
            loss = loss_task + (args.lambda_l0 * loss_sparsity)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss_task.item()
            total_l0 += loss_sparsity.item()
            
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        active_heads = 0
        total_heads_count = 0
        
        with torch.no_grad():
            for block in model.blocks:
                mask = block.attn.l0_mask(training=False)
                active_heads += mask.sum().item()
                total_heads_count += mask.numel()

        sparsity_rate = 100 * (1 - active_heads/total_heads_count)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Temp: {current_temp:.2f}")
        print(f"   Train Acc: {train_acc:.2f}% (on subset)")
        print(f"   Heads Active: {int(active_heads)}/{total_heads_count} (Pruned: {sparsity_rate:.2f}%)")
        print(f"   L0 Penalty: {total_l0/len(train_loader):.4f}")

    print("Saving pruned model state dict...")
    # Save the state dict (compatible with original architecture, just has extra mask params)
    torch.save(model.state_dict(), args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_small_patch16_224")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lambda_l0", type=float, default=0.05)
    parser.add_argument("--output_path", type=str, default="pruned_vit_small.pth")
    args = parser.parse_args()
    
    run_experiment(args)