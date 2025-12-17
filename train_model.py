import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import os
import copy
import random
import numpy as np
from SHViT import BN_Linear, shvit_s1, shvit_s2, shvit_s3, shvit_s4
import torchvision

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def build_and_load_monster(original_shvit, num_classes=1000):
    """
    Takes a standard pre-trained SHViT instance and transfers weights 
    to a new DoubleHeadSHViT.
    """
    # Create the monster
    orthrus = DoubleHeadSHViT(num_classes=num_classes) 
    
    # Transfer Shared weights
    orthrus.patch_embed.load_state_dict(original_shvit.patch_embed.state_dict())
    orthrus.blocks1.load_state_dict(original_shvit.blocks1.state_dict())
    orthrus.head.load_state_dict(original_shvit.head.state_dict())
    
    # --- Helper to copy parameters ---
    def copy_params(src_module, dest_module):
        dest_module.load_state_dict(src_module.state_dict())
    
    ds_layers_2 = torch.nn.Sequential(original_shvit.blocks2[0], original_shvit.blocks2[1], original_shvit.blocks2[2])
    blk_layers_2 = torch.nn.Sequential(*[original_shvit.blocks2[i] for i in range(3, len(original_shvit.blocks2))])
    
    # Branch A
    copy_params(ds_layers_2, orthrus.ds_1_to_2_A)
    copy_params(blk_layers_2, orthrus.blocks2_A)
    
    # Branch B (The Duplication!)
    copy_params(ds_layers_2, orthrus.ds_1_to_2_B)
    copy_params(blk_layers_2, orthrus.blocks2_B)

    # --- Copying Stage 3 (Original blocks3) ---
    ds_layers_3 = torch.nn.Sequential(original_shvit.blocks3[0], original_shvit.blocks3[1], original_shvit.blocks3[2])
    blk_layers_3 = torch.nn.Sequential(*[original_shvit.blocks3[i] for i in range(3, len(original_shvit.blocks3))])
    
    # Branch A
    copy_params(ds_layers_3, orthrus.ds_2_to_3_A)
    copy_params(blk_layers_3, orthrus.blocks3_A)
    
    # Branch B
    copy_params(ds_layers_3, orthrus.ds_2_to_3_B)
    copy_params(blk_layers_3, orthrus.blocks3_B)
    
    print("Orthrus initialized with duplicated Siamese heads.")
    return orthrus

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

    #train_size = int(train_len * len(full_dataset))
    #val_size = len(full_dataset) - train_size
    #train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    #train_dataset.dataset.transform = data_transforms['train']
    #val_dataset.dataset.transform = data_transforms['val']
    train_dataset = ApplyTransform(Subset(full_dataset, train_idx), transform=data_transforms['train'])
    val_dataset = ApplyTransform(Subset(full_dataset, val_idx), transform=data_transforms['val'])
    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, worker_init_fn=seed_worker, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, num_workers=2)

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

def train_monster(shvit_location, loader_train, loader_eval, NUM_EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Standard Pretrained SHViT
    print("Loading original SHViT...")
    # Make sure 'shvit_s1' is available in your registry or instantiated manually
    original_model = build_shvit('s1', shvit_location, classes_output=1000)

    # 2. Build the Monster
    print("Constructing Double-Head Monster...")
    model = build_and_load_monster(original_model, num_classes=1000)
    freeze_stage_1(model)
    model.to(device)

    # 4. Setup Training
    # We want to freeze Stage 1? Optional. Let's train everything but with low LR.
    param_groups = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params_groups, lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = NativeScaler() # For AMP if needed

    print("Starting Fine-tuning...")
    for epoch in range(NUM_EPOCHS):
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
        torch.save(model.state_dict(), f"shvit_doublehead_epoch_{epoch}.pth")

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input, target in loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total
