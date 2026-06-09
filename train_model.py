import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import timm
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from tqdm import tqdm
import os
import copy
import random
import numpy as np
from SHViT import BN_Linear, shvit_s1, shvit_s2, shvit_s3, shvit_s4, DoubleHeadSHViT, TripleHeadSHViT, SHViT_s1, SHViT_s2, SHViT_s3, SHViT_s4
import torchvision
from functools import partial
from algorithms import logit_only, training_based

class L0Mask(nn.Module):
    def __init__(self, num_heads, temp=2./3., limit_l=-0.1, limit_r=1.1):
        super().__init__()
        self.num_heads = num_heads
        self.temp = temp
        self.limit_l = limit_l
        self.limit_r = limit_r
        self.log_alpha = nn.Parameter(torch.Tensor(num_heads))

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

    def forward(self, x, attn_mask=None):
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

class ApplyTransform(Dataset):
  def __init__(self, subset, transform=None):
    self.subset = subset
    self.transform = transform

  def __getitem__(self, index):
    x, y = self.subset[index]
    if self.transform:
      x = self.transform(x)
    return x, y

  def __len__(self):
    return len(self.subset)

def set_seed(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # kept False for reproducibility (seed is set)

    # Para DataLoader workers
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker

def get_imagenet_labels():
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        class_index = json.loads(response.text)

        idx_to_label = {int(k): v[1] for k, v in class_index.items()}
        return idx_to_label

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the labels: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def freeze_stage_1(model):
    """
    Freezes the Shared Stem (Patch Embed) and Stage 1 (Blocks1).
    """
    print("Freezing Stage 1 (Stem + Blocks1)...")
    
    # Freeze Patch Embedding (Stem)
    for param in model.patch_embed.parameters():
        param.requires_grad = False
        
    # Freeze Block 1
    for param in model.blocks1.parameters():
        param.requires_grad = False
        
    # Verification print
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Frozen. Trainable Params: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M")

def build_and_load_monster(original_shvit, num_classes=1000, model_cfg=None):
    if model_cfg is None: model_cfg = {}
    print("Building DoubleHeadSHViT (Diversified Ensemble)...")
    monster = DoubleHeadSHViT(num_classes=num_classes, **model_cfg)
    
    print(" - Copying Shared Stem and Stage 1...")
    monster.patch_embed.load_state_dict(original_shvit.patch_embed.state_dict())
    monster.blocks1.load_state_dict(original_shvit.blocks1.state_dict())
    
    # Copy pretrained head to BOTH branch heads
    print(" - Copying classification head to both head_A and head_B...")
    monster.head_A.load_state_dict(original_shvit.head.state_dict())
    monster.head_B.load_state_dict(original_shvit.head.state_dict())
    
    print(" - Duplicating Stage 2 and 3 to both heads...")

    def copy_params(src_layers, dest_module):
        # We wrap source layers in a temporary Sequential.
        # This re-indexes keys to '0', '1', '2' matching the destination Sequential.
        temp_seq = torch.nn.Sequential(*src_layers)
        dest_module.load_state_dict(temp_seq.state_dict())

    # --- Stage 2 ---
    # Original structure: [Down1(0), PatchMerge(1), Down2(2), Block(3), Block(4)...]
    src_ds_2 = [original_shvit.blocks2[i] for i in range(3)] 
    src_blks_2 = [original_shvit.blocks2[i] for i in range(3, len(original_shvit.blocks2))]

    copy_params(src_ds_2, monster.ds_1_to_2_A)
    copy_params(src_blks_2, monster.blocks2_A)
    
    copy_params(src_ds_2, monster.ds_1_to_2_B)
    copy_params(src_blks_2, monster.blocks2_B)
    # Stronger noise (1e-2) for meaningful symmetry breaking
    for param in monster.blocks2_B.parameters(): param.data += torch.randn_like(param) * 1e-2

    # --- Stage 3 ---
    src_ds_3 = [original_shvit.blocks3[i] for i in range(3)]
    src_blks_3 = [original_shvit.blocks3[i] for i in range(3, len(original_shvit.blocks3))]

    copy_params(src_ds_3, monster.ds_2_to_3_A)
    copy_params(src_blks_3, monster.blocks3_A)
    
    copy_params(src_ds_3, monster.ds_2_to_3_B)
    copy_params(src_blks_3, monster.blocks3_B)
    # Stronger noise (1e-2) for meaningful symmetry breaking
    for param in monster.blocks3_B.parameters(): param.data += torch.randn_like(param) * 1e-2
    
    print("Monster initialized successfully!")
    return monster

def load_finetuned_monster(checkpoint_path, num_classes=1000, device='cuda', model_cfg=None):
    print(f"Loading finetuned monster from {checkpoint_path}...")
    if model_cfg is None:
        if 's4' in checkpoint_path:
            model_cfg = SHViT_s4
        elif 's3' in checkpoint_path:
            model_cfg = SHViT_s3
        elif 's2' in checkpoint_path:
            model_cfg = SHViT_s2
        else:
            model_cfg = SHViT_s1
            
    monster = DoubleHeadSHViT(num_classes=num_classes, **model_cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Filter out mismatching parameters (e.g. classifier head from a different number of classes)
    model_state_dict = monster.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"  Skipping parameter '{k}' due to size mismatch: model shape {model_state_dict[k].shape} vs checkpoint shape {v.shape}")
        else:
            filtered_state_dict[k] = v

    # Handle both old (router+head) and new (head_A+head_B) checkpoint formats
    missing, unexpected = monster.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} keys")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} keys")
        
    monster.to(device)
    monster.eval()
    print("Monster loaded and ready!")
    return monster

def build_and_load_triple_monster(original_shvit, num_classes=1000, model_cfg=None):
    if model_cfg is None: model_cfg = {}
    print("Building TripleHeadSHViT...")
    monster = TripleHeadSHViT(num_classes=num_classes, **model_cfg)
    
    print(" - Copying Shared Stem and Stage 1...")
    monster.patch_embed.load_state_dict(original_shvit.patch_embed.state_dict())
    monster.blocks1.load_state_dict(original_shvit.blocks1.state_dict())
    monster.head.load_state_dict(original_shvit.head.state_dict())
    
    print(" - Duplicating Stage 2 and 3 to all three heads...")

    def copy_params(src_layers, dest_module):
        # We wrap source layers in a temporary Sequential.
        # This re-indexes keys to '0', '1', '2' matching the destination Sequential.
        temp_seq = torch.nn.Sequential(*src_layers)
        dest_module.load_state_dict(temp_seq.state_dict())

    # --- Stage 2 ---
    src_ds_2 = [original_shvit.blocks2[i] for i in range(3)] 
    src_blks_2 = [original_shvit.blocks2[i] for i in range(3, len(original_shvit.blocks2))]

    copy_params(src_ds_2, monster.ds_1_to_2_A)
    copy_params(src_blks_2, monster.blocks2_A)
    
    copy_params(src_ds_2, monster.ds_1_to_2_B)
    copy_params(src_blks_2, monster.blocks2_B)
    for param in monster.blocks2_B.parameters(): param.data += torch.randn_like(param) * 1e-4

    copy_params(src_ds_2, monster.ds_1_to_2_C)
    copy_params(src_blks_2, monster.blocks2_C)
    for param in monster.blocks2_C.parameters(): param.data += torch.randn_like(param) * 1e-4

    # --- Stage 3 ---
    src_ds_3 = [original_shvit.blocks3[i] for i in range(3)]
    src_blks_3 = [original_shvit.blocks3[i] for i in range(3, len(original_shvit.blocks3))]

    copy_params(src_ds_3, monster.ds_2_to_3_A)
    copy_params(src_blks_3, monster.blocks3_A)
    
    copy_params(src_ds_3, monster.ds_2_to_3_B)
    copy_params(src_blks_3, monster.blocks3_B)
    for param in monster.blocks3_B.parameters(): param.data += torch.randn_like(param) * 1e-4

    copy_params(src_ds_3, monster.ds_2_to_3_C)
    copy_params(src_blks_3, monster.blocks3_C)
    for param in monster.blocks3_C.parameters(): param.data += torch.randn_like(param) * 1e-4
    
    print("Triple Monster initialized successfully!")
    return monster

def load_finetuned_triple_monster(checkpoint_path, num_classes=1000, device='cuda', model_cfg=None):
    print(f"Loading finetuned triple monster from {checkpoint_path}...")
    if model_cfg is None:
        if 's4' in checkpoint_path:
            model_cfg = SHViT_s4
        elif 's3' in checkpoint_path:
            model_cfg = SHViT_s3
        elif 's2' in checkpoint_path:
            model_cfg = SHViT_s2
        else:
            model_cfg = SHViT_s1
            
    monster = TripleHeadSHViT(num_classes=num_classes, **model_cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
        
    # Filter out mismatching parameters
    model_state_dict = monster.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"  Skipping parameter '{k}' due to size mismatch: model shape {model_state_dict[k].shape} vs checkpoint shape {v.shape}")
        else:
            filtered_state_dict[k] = v

    monster.load_state_dict(filtered_state_dict, strict=False)
    
    monster.to(device)
    monster.eval()
    print("Triple Monster loaded and ready!")
    return monster

def build_shvit(shvit_name, location, classes_output=1000):
    if shvit_name == 's1':
      shvit = shvit_s1(num_classes=1000)
    elif shvit_name == 's2':
      shvit = shvit_s2(num_classes=classes_output)
    elif shvit_name == 's3':
      shvit = shvit_s3(num_classes=classes_output)
    elif shvit_name == 's4':
      shvit = shvit_s4(num_classes=classes_output)
    else:
      print("ADD A VALID MODEL NAME: [s1,s2,s3,s4] ")
      return
    checkpoint = torch.load(location, map_location="cuda")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
      state_dict = checkpoint["model"]
    else:
      state_dict = checkpoint
    shvit.load_state_dict(state_dict)
    if classes_output != 1000:
      in_features_for_new_head = shvit.head.l.in_features
      shvit.head.l = nn.Linear(in_features_for_new_head, classes_output, bias=True)

    return shvit

    
def build_timm(model_name, classes_output=1000):
    model = timm.create_model(model_name, pretrained=True, num_classes=1000)
    print(f"[DEBUG] Loaded model: {model_name}")
    if hasattr(model, 'default_cfg'):
        url = model.default_cfg.get('url')
        if url:
            print(f"[DEBUG] Model Weights URL: {url}")
            # Check local cache for this file
            hub_dir = torch.hub.get_dir()
            filename = os.path.basename(url)
            cached_path = os.path.join(hub_dir, 'checkpoints', filename)
            if os.path.exists(cached_path):
                 print(f"[DEBUG] Verified: Using cached weights from {cached_path}")
            else:
                 print(f"[DEBUG] WARNING: Expected cache file not found at {cached_path}")
        else:
            print("[DEBUG] No URL found in model.default_cfg")
    if classes_output != 1000:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, classes_output)
    return model

def build_base(classes_output=1000):
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    if classes_output != 1000:
      model.heads.head = nn.Linear(in_features, classes_output)

    return model

def build_pruned(model_name, checkpoint_path):
    kwargs = {}

    model = timm.create_model(model_name, pretrained=False)
    print("Patching ViT with PrunableAttention layers...")
    for block in model.blocks:
        block.attn = PrunableAttention(block.attn)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading pruned weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(state_dict)

        return model.cuda().eval()
    else:
        raise ValueError("IS_PRUNED is true but CHECKPOINT_PATH is invalid.")
    
    

def data_loader(PATH, BATCH_SIZE, seed_worker, train_len=0.2):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    print("Loading Stylized Images dataset...")
    full_dataset = datasets.ImageFolder(PATH)
    targets = full_dataset.targets

    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        train_size=train_len,
        shuffle=True,
        stratify=targets, 
        random_state=42 
    )
    train_dataset = ApplyTransform(Subset(full_dataset, train_idx), transform=data_transforms['train'])
    val_dataset = ApplyTransform(Subset(full_dataset, val_idx), transform=data_transforms['val'])
    

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, worker_init_fn=seed_worker,
        shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker,
        num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=False
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of classes in Stylized Images: {len(full_dataset.classes)}")
    return train_loader, val_loader

def train_model_timm(model, train_loader, val_loader, model_name, SAVE_DIR="~/tesis/VCC/model_weights/large", num_epochs=10, freeze_strategy='head_only', learning_rate=1e-3, DEVICE='cuda'):

    print(f"\n--- Starting fine-tuning for {model_name} with strategy: {freeze_strategy} ---")

    # 1. Freeze/Unfreeze Layers based on strategy
    if freeze_strategy == 'head_only':
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the specific head parameters
        if isinstance(model.head, BN_Linear): # For SHViT
            for param in model.head.parameters():
                param.requires_grad = True
        elif isinstance(model.head, nn.Linear): # For TinyViT (and general ViTs)
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown head type for {model_name}")

        print(f"[{model_name}] Freezing all layers except the classification head.")

     # 2. Define Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            dataloader = train_loader if phase == 'train' else val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc=f"{phase}ing {model_name} Epoch {epoch}"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'imagenet_r_{model_name}_{best_acc:-4f}acc.pth'))
                print(f"Saved best model for {model_name} ({freeze_strategy}) with accuracy: {best_acc:.4f}")

    print(f'Fine-tuning complete for {model_name} ({freeze_strategy})')
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def train_model_base(model, train_loader, val_loader, model_name, SAVE_DIR,num_epochs=10,
                freeze_strategy='head_only', DEVICE='cuda',
                learning_rate=1e-3):

    print(f"\n--- Starting fine-tuning for {model_name} with strategy: {freeze_strategy} ---")
    criterion = nn.CrossEntropyLoss()
    # Pre-compute a set of head parameters for efficient lookup
    # Adjust for torchvision ViT which has .heads.head
    if model_name == "ViT-B16": # Assuming this name is used for the torchvision model
        head_params_set = set(model.heads.head.parameters())
    else: # For SHViT or other models with .head
        head_params_set = set(model.head.parameters())

    # 1. Freeze/Unfreeze Layers based on strategy
    if freeze_strategy == 'head_only':
        # Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the specific head parameters
        # Adjust for torchvision ViT which has .heads.head
        if model_name == "ViT-B16":
            for param in model.heads.head.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'): # For SHViT
            for param in model.head.parameters():
                param.requires_grad = True
                if isinstance(model.head, BN_Linear):
                    model.head.train()
            else:
                 print(f"Warning: Unknown head type for {model_name}. Ensure it's trainable.")
        else:
            raise ValueError(f"Could not find a standard 'head' attribute for {model_name} to unfreeze. Check model structure.")

        # Set all BatchNorm/LayerNorm layers in the *body* to eval mode
        # to freeze their running statistics updates.
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                # Only apply to BatchNorm/LayerNorms NOT in the head
                is_in_head = False
                for p in m.parameters():
                    if p in head_params_set: # Using the pre-computed set
                        is_in_head = True
                        break
                if not is_in_head:
                    m.eval() # Freeze BatchNorm/LayerNorm statistics for the backbone

        print(f"[{model_name}] Freezing all layers except the classification head. BatchNorm/LayerNorms in body set to eval mode.")
    else:
        raise ValueError(f"Unknown freeze_strategy: {freeze_strategy}")

    # 2. Define Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in [ 'train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            dataloader = train_loader if phase == 'train' else val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc=f"{phase}ing {model_name} Epoch {epoch}"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{model_name}_{freeze_strategy}_best_finetuned.pth'))
                print(f"Saved best model for {model_name} ({freeze_strategy}) with accuracy: {best_acc:.4f}")

    print(f'Fine-tuning complete for {model_name} ({freeze_strategy})')
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def train_monster(use_shvit, num_heads, model_location, loader_train, loader_eval, NUM_CLASS, NUM_EPOCHS, output_file_name, resume_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shvit_name = 's1'
    model_cfg = SHViT_s1
    if 's4' in model_location:
        shvit_name = 's4'
        model_cfg = SHViT_s4
    elif 's3' in model_location:
        shvit_name = 's3'
        model_cfg = SHViT_s3
    elif 's2' in model_location:
        shvit_name = 's2'
        model_cfg = SHViT_s2

    # 1. Load Standard Pretrained SHViT
    if use_shvit:
      print(f"Loading original SHViT {shvit_name}...")
      original_model = build_shvit(shvit_name, model_location, classes_output=1000)
      if num_heads == 2:
        print("Constructing Double-Head Monster...")
        model = build_and_load_monster(original_model, num_classes=NUM_CLASS, model_cfg=model_cfg)
      elif num_heads == 3:
        print("Constructing Triple-Head Monster...")
        model = build_and_load_triple_monster(original_model, num_classes=NUM_CLASS, model_cfg=model_cfg)
      else:
        raise ValueError(f"Invalid number of heads: {num_heads}. Must be 2 or 3.")
    else:
      if num_heads == 2:
        model = load_finetuned_monster(model_location, num_classes=NUM_CLASS, device=device, model_cfg=model_cfg)
      elif num_heads == 3:
        model = load_finetuned_triple_monster(model_location, num_classes=NUM_CLASS, device=device, model_cfg=model_cfg)
      else:
        raise ValueError(f"Invalid number of heads: {num_heads}. Must be 2 or 3.")
    
    freeze_stage_1(model)
    model.to(device)

    # 4. Setup Training
    param_groups = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(param_groups, lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = NativeScaler()
    diversity_lambda = 0.5  # Weight for diversity loss

    start_epoch = 0
    best_acc = 0.0

    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
        else:
            model.load_state_dict(checkpoint)
            print("Warning: Checkpoint did not contain optimizer/scheduler state. Only model weights loaded.")

    print("Starting Fine-tuning (Diversified Ensemble)...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_diversity = 0
        num_steps = 0

        for batch_idx, (input, target) in enumerate(loader_train):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            # AMP Context
            with torch.cuda.amp.autocast():
                output_tuple = model(input)
                if isinstance(output_tuple, tuple) and len(output_tuple) == 4:
                    # Diversified Ensemble: (logits_A, logits_B, feat_A, feat_B)
                    logits_A, logits_B, feat_A, feat_B = output_tuple
                    loss_A = criterion(logits_A, target)
                    loss_B = criterion(logits_B, target)
                    # Diversity loss: push features to be ORTHOGONAL (minimize squared cosine similarity)
                    diversity = (F.cosine_similarity(feat_A, feat_B, dim=1) ** 2).mean()
                    loss = loss_A + loss_B + diversity_lambda * diversity
                    total_diversity += diversity.item()
                elif isinstance(output_tuple, tuple) and len(output_tuple) == 2:
                    # Legacy MoE format fallback
                    output, routing_weights = output_tuple
                    loss = criterion(output, target)
                else:
                    output = output_tuple
                    loss = criterion(output, target)

            scaler(loss, optimizer)

            total_loss += loss.item()
            num_steps += 1

            if batch_idx % 100 == 0:
                div_str = f" | Diversity: {diversity.item():.4f}" if isinstance(output_tuple, tuple) and len(output_tuple) == 4 else ""
                print(f"Epoch {epoch}: Step {batch_idx} Loss {loss.item():.4f}{div_str}")
        scheduler.step()
        # Validation
        acc = validate(model, loader_eval, device)
        avg_div = total_diversity / max(num_steps, 1)
        print(f"Epoch {epoch} Done. Avg Loss: {total_loss/num_steps:.4f} | Avg Diversity: {avg_div:.4f} | Val Acc: {acc:.2f}%")

        # Save Checkpoint
        if acc > best_acc or epoch % 20 == 0:
            best_acc = acc
            checkpoint_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint_dict, str(epoch) + "_" + output_file_name)
            print(f"Saved model with acc {acc:.2f}%")
            
    return model

def train_monster_improved(use_shvit, num_heads, model_location, loader_train, loader_eval, NUM_CLASS, NUM_EPOCHS, output_file_name, resume_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shvit_name = 's1'
    model_cfg = SHViT_s1
    if 's4' in model_location:
        shvit_name = 's4'
        model_cfg = SHViT_s4
    elif 's3' in model_location:
        shvit_name = 's3'
        model_cfg = SHViT_s3
    elif 's2' in model_location:
        shvit_name = 's2'
        model_cfg = SHViT_s2

    # 1. Load Standard Pretrained SHViT
    if use_shvit:
      print(f"Loading original SHViT {shvit_name}...")
      original_model = build_shvit(shvit_name, model_location, classes_output=1000)
      if num_heads == 2:
        print("Constructing Double-Head Monster...")
        model = build_and_load_monster(original_model, num_classes=NUM_CLASS, model_cfg=model_cfg)
        # We no longer initialize fusion_conv as the model now uses an MoE Router.
      elif num_heads == 3:
        print("Constructing Triple-Head Monster...")
        model = build_and_load_triple_monster(original_model, num_classes=NUM_CLASS, model_cfg=model_cfg)
        
        # We no longer initialize fusion_conv as the model now uses an MoE Router.
      else:
        raise ValueError(f"Invalid number of heads: {num_heads}. Must be 2 or 3.")
    else:
      if num_heads == 2:
        model = load_finetuned_monster(model_location, num_classes=NUM_CLASS, device=device, model_cfg=model_cfg)
      elif num_heads == 3:
        model = load_finetuned_triple_monster(model_location, num_classes=NUM_CLASS, device=device, model_cfg=model_cfg)
      else:
        raise ValueError(f"Invalid number of heads: {num_heads}. Must be 2 or 3.")
    
    freeze_stage_1(model)
    model.to(device)

    # FIX: Progressive Unfreezing
    print("Progressive Unfreezing: Freezing Stage 2 and Stage 3 for the first 3 epochs...")
    for name, param in model.named_parameters():
        if 'blocks2' in name or 'blocks3' in name or 'ds_1_to_2' in name or 'ds_2_to_3' in name:
            param.requires_grad = False

    # FIX: Lower weight_decay to 0.01
    param_groups = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(param_groups, lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = NativeScaler() 

    start_epoch = 0
    best_acc = 0.0

    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Check if we are resuming after epoch 3 to set requires_grad properly
            if 'epoch' in checkpoint and checkpoint['epoch'] >= 3:
                for name, param in model.named_parameters():
                    if 'blocks2' in name or 'blocks3' in name or 'ds_1_to_2' in name or 'ds_2_to_3' in name:
                        param.requires_grad = True
                
                # Recreate the optimizer structure so load_state_dict matches the saved state
                param_groups = [
                    {'params': [p for n, p in model.named_parameters() if ('blocks2' in n or 'blocks3' in n or 'ds_1_to_2' in n or 'ds_2_to_3' in n) and p.requires_grad], 'lr': 1e-5},
                    {'params': [p for n, p in model.named_parameters() if not ('blocks2' in n or 'blocks3' in n or 'ds_1_to_2' in n or 'ds_2_to_3' in n) and p.requires_grad], 'lr': 1e-4}
                ]
                optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - 3)

            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
        else:
            model.load_state_dict(checkpoint)
            print("Warning: Checkpoint did not contain optimizer/scheduler state. Only model weights loaded.")

    print("Starting Fine-tuning...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        if epoch == 3:
            print("Unfreezing Stage 2 and Stage 3...")
            for name, param in model.named_parameters():
                if 'blocks2' in name or 'blocks3' in name or 'ds_1_to_2' in name or 'ds_2_to_3' in name:
                    param.requires_grad = True
            # Update optimizer groups and set a lower LR (1e-5) for the newly unfrozen layers
            param_groups = [
                {'params': [p for n, p in model.named_parameters() if ('blocks2' in n or 'blocks3' in n or 'ds_1_to_2' in n or 'ds_2_to_3' in n) and p.requires_grad], 'lr': 1e-5},
                {'params': [p for n, p in model.named_parameters() if not ('blocks2' in n or 'blocks3' in n or 'ds_1_to_2' in n or 'ds_2_to_3' in n) and p.requires_grad], 'lr': 1e-4}
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
            # Recreate scheduler to match remaining epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - 3)

        model.train()
        total_loss = 0
        num_steps = 0

        for batch_idx, (input, target) in enumerate(loader_train):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            # AMP Context
            with torch.cuda.amp.autocast():
                output_tuple = model(input)
                if isinstance(output_tuple, tuple):
                    output, routing_weights = output_tuple
                    loss_ce = criterion(output, target)
                    # Load balancing loss: variance of the mean routing probabilities
                    mean_routing = routing_weights.mean(dim=0)
                    # multiply by number of branches (mean_routing.size(0)) to scale variance properly
                    loss_bal = torch.var(mean_routing) * mean_routing.size(0)
                    loss = loss_ce + 0.1 * loss_bal
                else:
                    output = output_tuple
                    loss = criterion(output, target)

            scaler(loss, optimizer)

            total_loss += loss.item()
            num_steps += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}: Step {batch_idx} Loss {loss.item():.4f}")
        
        scheduler.step()
        
        # Validation
        acc = validate(model, loader_eval, device)
        print(f"Epoch {epoch} Done. Avg Loss: {total_loss/num_steps:.4f} | Val Acc: {acc:.2f}%")

        # Save Checkpoint
        if acc > best_acc or epoch % 20 == 0:
            best_acc = acc
            checkpoint_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint_dict, str(epoch)+"_"+output_file_name)
            print(f"Saved model with acc {acc:.2f}%")
            
    return model

VAL_PATH = "/home/mauricio.alvarez/tesis/archive/imagenet-val/imagenet-val"
TEST_PATH = "/home/mauricio.alvarez/tesis/archive/imagenet-test"

class ImageNetTestSet(Dataset):
    """
    Custom Dataset for the ImageNet Test set (100k images, no labels, flat directory).
    It sorts images by name to ensure the submission file order is correct.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith('.jpeg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, img_name

def test_imagenet_1k(model, set_name="val", batch_size=64, num_workers=4):
    """
    Evaluates the model on ImageNet-1K.
    
    Args:
        model: Loaded PyTorch model (Standard or Pruned).
        set_name: 'val' (calculate accuracy) or 'test' (generate submission file).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Standard ImageNet Transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"--- Running ImageNet Evaluation on '{set_name}' set ---")

    if set_name == 'val':
        # --- VALIDATION MODE (Local Accuracy) ---
        if not os.path.exists(VAL_PATH):
            raise ValueError(f"Validation path not found: {VAL_PATH}")

        dataset = datasets.ImageFolder(VAL_PATH, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        correct_1 = 0
        correct_5 = 0
        total = 0
        
        print(f"Dataset: {len(dataset)} images.")
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                # Top-k accuracy
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                
                correct_1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                correct_5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                total += inputs.size(0)

        top1 = 100 * correct_1 / total
        top5 = 100 * correct_5 / total
        print(f"\nResults for {set_name}:")
        print(f"Top-1 Accuracy: {top1:.2f}%")
        print(f"Top-5 Accuracy: {top5:.2f}%")
        
        return top1, top5

    elif set_name == 'test':
        # --- TEST MODE (Submission File) ---
        if not os.path.exists(TEST_PATH):
            raise ValueError(f"Test path not found: {TEST_PATH}")

        dataset = ImageNetTestSet(TEST_PATH, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        predictions = []
        
        print(f"Dataset: {len(dataset)} images (Unlabeled).")
        print("Generating predictions...")
        
        with torch.no_grad():
            for inputs, filenames in tqdm(dataloader, desc="Inferencing"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                # Get Top-1 Prediction
                _, pred_idx = outputs.max(1)
                predictions.extend(pred_idx.cpu().tolist())

        # Save to text file
        output_file = "imagenet_test_submission.txt"
        print(f"\nSaving predictions to {output_file}...")
        
        with open(output_file, "w") as f:
            for idx in predictions:
                f.write(f"{idx}\n")
                
        print("Done. File is ready for upload.")
        return output_file

    else:
        raise ValueError("set_name must be 'val' or 'test'")


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input, target in tqdm(loader, desc="Validation"):
            input, target = input.to(device), target.to(device)
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def calib_err(confidence, correct, p='2', beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr

def aurra(confidence, correct):
    conf_ranks = np.argsort(confidence)[::-1]  # indices from greatest to least confidence
    rra_curve = np.cumsum(np.asarray(correct)[conf_ranks])
    rra_curve = rra_curve / np.arange(1, len(rra_curve) + 1)  # accuracy at each response rate
    return np.mean(rra_curve)


def soft_f1(confidence, correct):
    wrong = 1 - correct

    # # the incorrectly classified samples are our interest
    # # so they make the positive class
    # tp_soft = np.sum((1 - confidence) * wrong)
    # fp_soft = np.sum((1 - confidence) * correct)
    # fn_soft = np.sum(confidence * wrong)

    # return 2 * tp_soft / (2 * tp_soft + fn_soft + fp_soft)
    return 2 * ((1 - confidence) * wrong).sum()/(1 - confidence + wrong).sum()

def show_calibration_results(confidence, correct):

    print('RMS Calib Error (%): \t{:.2f}'.format(
        100 * calib_err(confidence, correct, p='2')))

    print('AURRA (%): \t\t{:.2f}'.format(
        100 * aurra(confidence, correct)))

    # print('MAD Calib Error (%): \t\t{:.2f}'.format(
    #     100 * calib_err(confidence, correct, p='1')))

    # print('Soft F1-Score (%): \t\t{:.2f}'.format(
    #     100 * soft_f1(confidence, correct))

def test_imagenet_r(model):
    imagenet_val_location = "/home/mauricio.alvarez/tesis/archive/imagenet-val/imagenet-val"

    all_wnids = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267', 'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02699494', 'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516', 'n02823428', 'n02823750', 'n02825657', 'n02834397', 'n02835271', 'n02837789', 'n02840245', 'n02841315', 'n02843684', 'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02871525', 'n02877765', 'n02879718', 'n02883205', 'n02892201', 'n02892767', 'n02894605', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936', 'n02917067', 'n02927161', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159', 'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003', 'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632', 'n03042490', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03160309', 'n03179701', 'n03180011', 'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941', 'n03208938', 'n03216828', 'n03218198', 'n03220513', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574', 'n03272010', 'n03272562', 'n03290653', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051', 'n03384352', 'n03388043', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03457902', 'n03459775', 'n03461385', 'n03467068', 'n03476684', 'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03529860', 'n03530642', 'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480', 'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03637318', 'n03642806', 'n03649909', 'n03657121', 'n03658185', 'n03661043', 'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03697007', 'n03706229', 'n03709823', 'n03710193', 'n03710637', 'n03710721', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733281', 'n03733805', 'n03742115', 'n03743016', 'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681', 'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863', 'n03871628', 'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397', 'n03887697', 'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03899768', 'n03902125', 'n03903868', 'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711', 'n03967562', 'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04008634', 'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434', 'n04070727', 'n04074963', 'n04081281', 'n04086273', 'n04090263', 'n04099969', 'n04111531', 'n04116512', 'n04118538', 'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04152593', 'n04153751', 'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04200800', 'n04201297', 'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04239074', 'n04243546', 'n04251144', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257', 'n04264628', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04296562', 'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04356056', 'n04357314', 'n04366367', 'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04372370', 'n04376876', 'n04380533', 'n04389033', 'n04392985', 'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04443257', 'n04447861', 'n04456115', 'n04458633', 'n04461696', 'n04462240', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04493381', 'n04501370', 'n04505470', 'n04507155', 'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04523525', 'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04592741', 'n04596742', 'n04597913', 'n04599235', 'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06359193', 'n06596364', 'n06785654', 'n06794110', 'n06874185', 'n07248320', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07590611', 'n07613480', 'n07614500', 'n07615774', 'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146', 'n07836838', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n07892512', 'n07920052', 'n07930864', 'n07932039', 'n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635', 'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141']

    imagenet_r_wnids = {'n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677'}

    imagenet_r_mask = [wnid in imagenet_r_wnids for wnid in all_wnids]
    # imagenet_r_indices = [i for i in range(1000) if imagenet_r_mask[i] is True]
    # [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transform = trn.Compose(
        [trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])

    imagenet_r = dset.ImageFolder(root="/home/mauricio.alvarez/tesis/archive/imagenet-r", transform=test_transform)
    imagenet_r_loader = torch.utils.data.DataLoader(imagenet_r, batch_size=128, shuffle=False,
                                            num_workers=4, pin_memory=True)

    def create_symlinks_to_imagenet(imagenet_folder):
        if not os.path.exists(imagenet_folder):
            os.makedirs(imagenet_folder)
            folders_of_interest = imagenet_r_wnids  # os.listdir(folder_to_scan)
            for folder in folders_of_interest:
                os.symlink(os.path.join(imagenet_val_location, folder), os.path.join(imagenet_folder, folder), target_is_directory=True)
        else:
            print('Folder containing IID validation images already exists')

    imagenet_200_folder = "./imagenet_val_for_imagenet_r/"
    create_symlinks_to_imagenet(imagenet_200_folder)

    iid_examples = dset.ImageFolder(root=imagenet_200_folder, transform=test_transform)
    iid_loader = torch.utils.data.DataLoader(iid_examples, batch_size=256, shuffle=False,
                                            num_workers=4, pin_memory=True)

    if hasattr(model, 'module'):
        net = model.module
    else:
        net = model
        
    net = model
    net.cuda()
    net.eval()

    temp_features = {}
    def generic_hook(module, input, output, name):
        temp_features[name] = input[0]

    def get_last_layer(model):
        if hasattr(model, 'module'):
            base_model = model.module
        else:
            base_model = model
            
        if hasattr(base_model, 'head'):
            return base_model.head
        elif hasattr(base_model, 'fc'):
            return base_model.fc
        elif hasattr(base_model, 'classifier'):
            return base_model.classifier
        else:
            return list(base_model.children())[-1]

    last_layer = get_last_layer(net)
    hook_handle = last_layer.register_forward_hook(partial(generic_hook, name="feat"))
    
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.to('cpu').numpy()

    def get_predictions(loader):
        eval_features =[]
        eval_logits = []
        correct_list =[]
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                
                # Forward pass
                output = net(data)

                if "feat" not in temp_features:
                    raise RuntimeError(f"Hook failed to trigger! Attached to: {last_layer}")
                
                eval_logits.append(output.detach().cpu())
                eval_features.append(temp_features["feat"].detach().cpu())
                correct_list.extend(target.cpu().numpy())
                
                # Clear the dictionary to be safe for the next batch
                del temp_features["feat"]

        eval_logits = torch.cat(eval_logits)
        eval_features = torch.cat(eval_features)
        targets = np.array(correct_list)

        # Restrict logits to the 200 ImageNet-R classes
        logits_200 = eval_logits[:, imagenet_r_mask]

        # Calculate Accuracy (1.0 for correct, 0.0 for incorrect)
        preds = logits_200.max(1)[1].numpy()
        correct = (preds == targets).astype(float)
        num_correct = correct.sum()

        # Calculate all Confidence Scores
        scores = {}
        
        probs_200 = F.softmax(logits_200, dim=1)
        
        # 1. MSP (on 200 classes)
        scores['msp'] = probs_200.max(1)[0].numpy()
        
        # Predictive Entropy (negative entropy so higher = more confident)
        scores['neg_entropy'] = torch.sum(probs_200 * torch.log(probs_200 + 1e-12), dim=1).numpy()
        
        # 2. Max Logits (on 200 classes)
        scores['ml'] = logits_200.max(1)[0].numpy()
        
        # 3. Energy (on 200 classes)
        scores['energy'] = torch.logsumexp(logits_200, dim=1).numpy()
        
        # 4. Maximum Cosine (Requires full features/weights)
        try:
            scores['Maximum Cosine'] = logit_only["Maximum Cosine"](eval_features, last_layer).numpy()
        except Exception as e:
            pass
            
        # 5. ASH
        try:
            if "ash" in training_based:
                scores['ash_b'] = training_based["ash"](eval_features, last_layer, 90, version='b').numpy()
        except Exception as e:
            pass

        return scores, correct, num_correct, preds, targets


    def get_acc(loader, dataset_name):
        scores, correct, num_correct, preds, targets = get_predictions(loader)
        acc = num_correct / len(correct)
        
        print(f'{dataset_name} Results')
        print(f'Accuracy (%):\t\t {round(100 * acc, 2)}')
        
        if 'msp' in scores:
            correct_mask = (correct == 1.0)
            msp_correct = scores['msp'][correct_mask].mean() if correct_mask.sum() > 0 else 0
            msp_incorrect = scores['msp'][~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
            print(f'Mean MSP (Correct):\t {msp_correct:.4f}')
            print(f'Mean MSP (Incorrect):\t {msp_incorrect:.4f}')
            
        # Save Confusion Matrix
        cm = confusion_matrix(targets, preds)
        cm_filename = f'confusion_matrix_{dataset_name.replace(" ", "_")}.npy'
        np.save(cm_filename, cm)
        print(f'Saved Confusion Matrix array to: {cm_filename}')
        
        # Save as PNG
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, cmap='Blues')
        plt.title(f'Confusion Matrix: {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        png_filename = f'confusion_matrix_{dataset_name.replace(" ", "_")}.png'
        plt.savefig(png_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved Confusion Matrix image to: {png_filename}')
        
        print('\nAURRA (%):')
        for metric_name, confidence in scores.items():
            aurra_val = aurra(confidence, correct) * 100
            print(f'  {metric_name:<15}: {aurra_val:.2f}')
            
        # Only MSP is bounded [0, 1] for a valid RMSCE calculation
        if 'msp' in scores:
            rmsce = calib_err(scores['msp'], correct, p='2') * 100
            ece = calib_err(scores['msp'], correct, p='1') * 100
            print(f'\nRMS Calib Error (RMSCE) [%]: {rmsce:.2f}')
            print(f'Expected Calib Error (ECE) [%]: {ece:.2f}')
            
        return acc

    print('='*40)
    acc_orig = get_acc(iid_loader, 'ImageNet-200')
    acc_r = get_acc(imagenet_r_loader,'ImageNet-R')
    print('\nDelta Acc (%):\t\t', round(100*(acc_orig - acc_r), 2))
    hook_handle.remove()


def build_and_load_typhon(original_shvit, num_classes=1000, model_cfg=None, num_mixers=2):
    from SHViT import Typhon, TyphonBasicBlock
    import torch
    
    if model_cfg is None:
        model_cfg = {}
    print(f"Building Typhon with {num_mixers} mixers...")
    typhon = Typhon(num_classes=num_classes, num_mixers=num_mixers, **model_cfg)
    
    print(" - Copying Shared Stem...")
    typhon.patch_embed.load_state_dict(original_shvit.patch_embed.state_dict())
    
    print(" - Copying Classification Head...")
    if typhon.head.l.out_features == original_shvit.head.l.out_features:
        typhon.head.load_state_dict(original_shvit.head.state_dict())
    else:
        print(f"Skipping head state copy due to class output mismatch: {original_shvit.head.l.out_features} (src) vs {typhon.head.l.out_features} (dest)")
    
    print(" - Copying blocks and duplicating mixers...")
    
    def copy_stage(src_seq, dest_seq):
        for src_blk, dest_blk in zip(src_seq, dest_seq):
            if isinstance(dest_blk, TyphonBasicBlock):
                if dest_blk.type == 's':
                    # Copy conv & ffn
                    dest_blk.conv.load_state_dict(src_blk.conv.state_dict())
                    dest_blk.ffn.load_state_dict(src_blk.ffn.state_dict())
                    # Copy shared projection layer
                    dest_blk.proj.load_state_dict(src_blk.mixer.m.proj.state_dict())
                    # Copy the single mixer to all mixers in Typhon block (excl. proj)
                    src_mixer_state = {k: v for k, v in src_blk.mixer.m.state_dict().items() if not k.startswith("proj")}
                    for k in range(dest_blk.num_mixers):
                        dest_blk.mixers[k].load_state_dict(src_mixer_state)
                        # Add tiny noise for symmetry breaking
                        if k > 0:
                            for param in dest_blk.mixers[k].parameters():
                                param.data += torch.randn_like(param.data) * 1e-4
                else:
                    # Early stage type 'i' block
                    dest_blk.load_state_dict(src_blk.state_dict())
            else:
                # Downsample blocks (nn.Sequential or PatchMerging)
                dest_blk.load_state_dict(src_blk.state_dict())

    copy_stage(original_shvit.blocks1, typhon.blocks1)
    copy_stage(original_shvit.blocks2, typhon.blocks2)
    copy_stage(original_shvit.blocks3, typhon.blocks3)
    
    print("Typhon initialized successfully from SHViT weights!")
    return typhon


def load_finetuned_typhon(checkpoint_path, num_classes=1000, device='cuda', num_mixers=2, model_cfg=None):
    from SHViT import Typhon, SHViT_s1, SHViT_s2, SHViT_s3, SHViT_s4
    import torch
    import os
    
    print(f"Loading finetuned Typhon from {checkpoint_path}...")
    if model_cfg is None:
        if 's4' in checkpoint_path:
            model_cfg = SHViT_s4
        elif 's3' in checkpoint_path:
            model_cfg = SHViT_s3
        elif 's2' in checkpoint_path:
            model_cfg = SHViT_s2
        else:
            model_cfg = SHViT_s1
            
    model = Typhon(num_classes=num_classes, num_mixers=num_mixers, **model_cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Filter out mismatching parameters
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"  Skipping parameter '{k}' due to size mismatch: model shape {model_state_dict[k].shape} vs checkpoint shape {v.shape}")
        else:
            filtered_state_dict[k] = v

    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Typhon loaded successfully!")
    return model


def train_typhon(use_shvit, model_location, loader_train, loader_eval, NUM_CLASS, NUM_EPOCHS, output_file_name, num_mixers=2, resume_checkpoint=None):
    from SHViT import SHViT_s1, SHViT_s2, SHViT_s3, SHViT_s4
    import torch
    import os
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shvit_name = 's1'
    model_cfg = SHViT_s1
    if 's4' in model_location:
        shvit_name = 's4'
        model_cfg = SHViT_s4
    elif 's3' in model_location:
        shvit_name = 's3'
        model_cfg = SHViT_s3
    elif 's2' in model_location:
        shvit_name = 's2'
        model_cfg = SHViT_s2

    # 1. Initialize or Load Typhon Model
    if use_shvit:
        print(f"Loading original SHViT {shvit_name} to initialize Typhon...")
        original_model = build_shvit(shvit_name, model_location, classes_output=1000)
        model = build_and_load_typhon(original_model, num_classes=NUM_CLASS, model_cfg=model_cfg, num_mixers=num_mixers)
    else:
        model = load_finetuned_typhon(model_location, num_classes=NUM_CLASS, device=device, num_mixers=num_mixers, model_cfg=model_cfg)
    
    freeze_stage_1(model)
    model.to(device)

    # 2. Setup Training
    param_groups = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(param_groups, lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = NativeScaler()

    start_epoch = 0
    best_acc = 0.0

    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
        else:
            model.load_state_dict(checkpoint)
            print("Warning: Checkpoint did not contain optimizer/scheduler state. Only model weights loaded.")

    print("Starting Fine-tuning (Typhon)...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        num_steps = 0

        for batch_idx, (input, target) in enumerate(loader_train):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            # AMP Context
            with torch.cuda.amp.autocast():
                output = model(input)
                loss = criterion(output, target)

            scaler(loss, optimizer)

            total_loss += loss.item()
            num_steps += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}: Step {batch_idx} Loss {loss.item():.4f}")
        scheduler.step()
        
        # Validation
        acc = validate(model, loader_eval, device)
        print(f"Epoch {epoch} Done. Avg Loss: {total_loss/num_steps:.4f} | Val Acc: {acc:.2f}%")

        # Save Checkpoint
        if acc > best_acc or epoch % 20 == 0:
            best_acc = acc
            checkpoint_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint_dict, str(epoch) + "_" + output_file_name)
            print(f"Saved model with acc {acc:.2f}%")
            
    return model


def train_typhon_imagenet_1k(use_shvit, model_location, NUM_EPOCHS, output_file_name, batch_size=256, num_mixers=2, resume_checkpoint=None):
    from SHViT import SHViT_s1, SHViT_s2, SHViT_s3, SHViT_s4
    import torch
    import os
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    shvit_name = 's1'
    model_cfg = SHViT_s1
    if 's4' in model_location:
        shvit_name = 's4'
        model_cfg = SHViT_s4
    elif 's3' in model_location:
        shvit_name = 's3'
        model_cfg = SHViT_s3
    elif 's2' in model_location:
        shvit_name = 's2'
        model_cfg = SHViT_s2

    # 1. Initialize or Load Typhon Model
    NUM_CLASS = 1000
    if use_shvit:
        print(f"Loading original SHViT {shvit_name} to initialize Typhon...")
        original_model = build_shvit(shvit_name, model_location, classes_output=1000)
        model = build_and_load_typhon(original_model, num_classes=NUM_CLASS, model_cfg=model_cfg, num_mixers=num_mixers)
    else:
        model = load_finetuned_typhon(model_location, num_classes=NUM_CLASS, device=device, num_mixers=num_mixers, model_cfg=model_cfg)
    
    freeze_stage_1(model)
    model.to(device)

    # 2. Data Loading for ImageNet-1K
    print("Preparing ImageNet-1K Dataloaders...")
    train_dir = "/home/mauricio.alvarez/tesis/archive/imagenet_train/"
    val_dir = "/home/mauricio.alvarez/tesis/archive/imagenet-val/imagenet-val"

    if not os.path.exists(train_dir):
        raise ValueError(f"ImageNet train directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"ImageNet validation directory not found: {val_dir}")

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, prefetch_factor=2
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # 3. Setup Training
    param_groups = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(param_groups, lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = NativeScaler()

    start_epoch = 0
    best_acc = 0.0

    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
        else:
            model.load_state_dict(checkpoint)
            print("Warning: Checkpoint did not contain optimizer/scheduler state. Only model weights loaded.")

    print("Starting Fine-tuning (Typhon on ImageNet-1K)...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        num_steps = 0

        for batch_idx, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(input)
                loss = criterion(output, target)

            scaler(loss, optimizer)

            total_loss += loss.item()
            num_steps += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}: Step {batch_idx}/{len(train_loader)} Loss {loss.item():.4f}")
        scheduler.step()
        
        # Validation
        acc = validate(model, val_loader, device)
        print(f"Epoch {epoch} Done. Avg Loss: {total_loss/num_steps:.4f} | Val Acc: {acc:.2f}%")

        # Save Checkpoint
        if acc > best_acc or epoch % 10 == 0:
            best_acc = acc
            checkpoint_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint_dict, str(epoch) + "_" + output_file_name)
            print(f"Saved model with acc {acc:.2f}%")
    test_imagenet_r(model)
    test_imagenet_1k(model, set_name='val')       
    return model


def test_fashion_mnist(model, num_epochs=2, use_subset=True):
    from SHViT import BN_Linear
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from functools import partial
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model

    # 1. Check/Replace Head for 10 classes
    if hasattr(base_model, 'head_A') and hasattr(base_model, 'head_B'):
        # Double-head model
        for head_name in ['head_A', 'head_B']:
            head_module = getattr(base_model, head_name)
            if hasattr(head_module, 'l') and isinstance(head_module.l, nn.Linear):
                if head_module.l.out_features != 10:
                    print(f"Replacing BN_Linear {head_name}: {head_module.l.out_features} -> 10 classes")
                    in_features = head_module.l.in_features
                    setattr(base_model, head_name, BN_Linear(in_features, 10))
            elif isinstance(head_module, nn.Linear):
                if head_module.out_features != 10:
                    print(f"Replacing Linear {head_name}: {head_module.out_features} -> 10 classes")
                    in_features = head_module.in_features
                    setattr(base_model, head_name, nn.Linear(in_features, 10))
    elif hasattr(base_model, 'head'):
        if hasattr(base_model.head, 'l'):
            # BN_Linear (SHViT / Typhon style)
            if base_model.head.l.out_features != 10:
                print(f"Replacing BN_Linear head: {base_model.head.l.out_features} -> 10 classes")
                in_features = base_model.head.l.in_features
                base_model.head = BN_Linear(in_features, 10)
        elif isinstance(base_model.head, nn.Linear):
            if base_model.head.out_features != 10:
                print(f"Replacing Linear head: {base_model.head.out_features} -> 10 classes")
                in_features = base_model.head.in_features
                base_model.head = nn.Linear(in_features, 10)
    elif hasattr(base_model, 'heads') and hasattr(base_model.heads, 'head'):
        # torchvision ViT style
        if base_model.heads.head.out_features != 10:
            print(f"Replacing heads.head layer: {base_model.heads.head.out_features} -> 10 classes")
            in_features = base_model.heads.head.in_features
            base_model.heads.head = nn.Linear(in_features, 10)
    
    model.to(device)

    # 2. Data Loading
    print("Preparing Fashion MNIST Dataset...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.FashionMNIST(root='./data_mnist', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='./data_mnist', train=False, download=True, transform=transform)

    if use_subset:
        train_subset = Subset(train_set, range(2000))
        test_subset = Subset(test_set, range(500))
    else:
        train_subset = train_set
        test_subset = test_set

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)

    # 3. Fine-tuning
    print(f"Fine-tuning model on Fashion MNIST for {num_epochs} epochs...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if isinstance(outputs, tuple) and len(outputs) == 4:
                    logits_A, logits_B, feat_A, feat_B = outputs
                    loss_A = criterion(logits_A, targets)
                    loss_B = criterion(logits_B, targets)
                    # Diversity loss to push branches apart
                    diversity = (F.cosine_similarity(feat_A, feat_B, dim=1) ** 2).mean()
                    loss = loss_A + loss_B + 0.5 * diversity
                    outputs_for_acc = (logits_A + logits_B) / 2
                else:
                    loss = criterion(outputs, targets)
                    outputs_for_acc = outputs

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs_for_acc.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct_train / total_train) * 100
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

    
    print("Evaluating model...")
    model.eval()

    is_double_head = hasattr(base_model, 'head_A') and hasattr(base_model, 'head_B')
    temp_features = {}

    if is_double_head:
        def generic_hook_a(module, input, output):
            temp_features["feat_A"] = input[0]
        def generic_hook_b(module, input, output):
            temp_features["feat_B"] = input[0]

        hook_handle_a = base_model.head_A.register_forward_hook(generic_hook_a)
        hook_handle_b = base_model.head_B.register_forward_hook(generic_hook_b)
    else:
        def generic_hook(module, input, output, name):
            temp_features[name] = input[0]

        # Helper function to find last layer
        def get_last_layer(model):
            if hasattr(model, 'module'):
                base_model = model.module
            else:
                base_model = model
                
            if hasattr(base_model, 'head'):
                return base_model.head
            elif hasattr(base_model, 'fc'):
                return base_model.fc
            elif hasattr(base_model, 'classifier'):
                return base_model.classifier
            else:
                return list(base_model.children())[-1]

        last_layer = get_last_layer(model)
        hook_handle = last_layer.register_forward_hook(partial(generic_hook, name="feat"))

    eval_features = []
    eval_logits = []
    correct_list = []
    preds_list = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if is_double_head:
                if "feat_A" not in temp_features or "feat_B" not in temp_features:
                    raise RuntimeError("Hooks failed to trigger for double head!")
                # Ensemble feature averaging
                feat = (temp_features["feat_A"] + temp_features["feat_B"]) / 2
                del temp_features["feat_A"]
                del temp_features["feat_B"]
            else:
                if "feat" not in temp_features:
                    raise RuntimeError(f"Hook failed to trigger! Attached to: {last_layer}")
                feat = temp_features["feat"]
                del temp_features["feat"]

            eval_logits.append(outputs.detach().cpu())
            eval_features.append(feat.detach().cpu())
            correct_list.extend(targets.cpu().numpy())

            probs = F.softmax(outputs, dim=1)
            _, predicted = probs.max(1)
            preds_list.extend(predicted.cpu().numpy())

    eval_logits = torch.cat(eval_logits)
    eval_features = torch.cat(eval_features)
    targets = np.array(correct_list)
    preds = np.array(preds_list)

    correct = (preds == targets).astype(float)
    num_correct = correct.sum()
    acc = num_correct / len(correct)

    # Compute all confidence scores
    scores = {}
    probs = F.softmax(eval_logits, dim=1)
    scores['msp'] = probs.max(1)[0].numpy()
    scores['neg_entropy'] = torch.sum(probs * torch.log(probs + 1e-12), dim=1).numpy()
    scores['ml'] = eval_logits.max(1)[0].numpy()
    scores['energy'] = torch.logsumexp(eval_logits, dim=1).numpy()

    dataset_name = 'Fashion MNIST'
    print(f'\n========================================')
    print(f'{dataset_name} Results')
    print(f'Accuracy (%):\t\t {round(100 * acc, 2)}')
    
    correct_mask = (correct == 1.0)
    msp_correct = scores['msp'][correct_mask].mean() if correct_mask.sum() > 0 else 0
    msp_incorrect = scores['msp'][~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
    print(f'Mean MSP (Correct):\t {msp_correct:.4f}')
    print(f'Mean MSP (Incorrect):\t {msp_incorrect:.4f}')
        
    # Save Confusion Matrix
    cm = confusion_matrix(targets, preds)
    cm_filename = f'confusion_matrix_{dataset_name.replace(" ", "_")}.npy'
    np.save(cm_filename, cm)
    print(f'Saved Confusion Matrix array to: {cm_filename}')
    
    # Save as PNG
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    png_filename = f'confusion_matrix_{dataset_name.replace(" ", "_")}.png'
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved Confusion Matrix image to: {png_filename}')
    
    print('\nAURRA (%):')
    for metric_name, confidence in scores.items():
        aurra_val = aurra(confidence, correct) * 100
        print(f'  {metric_name:<15}: {aurra_val:.2f}')
        
    rmsce = calib_err(scores['msp'], correct, p='2') * 100
    ece = calib_err(scores['msp'], correct, p='1') * 100
    print(f'\nRMS Calib Error (RMSCE) [%]: {rmsce:.2f}')
    print(f'Expected Calib Error (ECE) [%]: {ece:.2f}')
    print(f'========================================')

    if is_double_head:
        hook_handle_a.remove()
        hook_handle_b.remove()
    else:
        hook_handle.remove()
    return acc