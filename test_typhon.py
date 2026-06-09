import sys
import os
sys.path.append("/home/mauricio.alvarez/tesis/VCC")

import torch
from SHViT import shvit_s1, SHViT_s1
from train_model import build_and_load_typhon, test_fashion_mnist, load_finetuned_monster

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize original SHViT model with 1000 classes
    orig_shvit = shvit_s1(num_classes=1000)
    
    # Load local pretrained SHViT weights if they exist
    weights_path = '/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s1.pth'
    if os.path.exists(weights_path):
        print(f"Loading pretrained weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location='cpu')
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        orig_shvit.load_state_dict(state_dict)
    else:
        print("Warning: Pretrained SHViT weights not found. Using randomly initialized weights.")

    # 2. Build Typhon model from SHViT weights (classes count is initially 10, head is initialized/replaced)
    model = build_and_load_typhon(orig_shvit, num_classes=10, model_cfg=SHViT_s1, num_mixers=2)
    model.to(device)

    # 3. Call the shared Fashion MNIST testing and fine-tuning suite
    test_fashion_mnist(model, num_epochs=10, use_subset=False)

    model = load_finetuned_monster("/home/mauricio.alvarez/tesis/VCC/model_weights/experiments/42_shvit_s1_doublehead_1805_100epochs.pth", num_classes=10, device=device)
    model.to(device)
    test_fashion_mnist(model, num_epochs=10, use_subset=False)    

if __name__ == "__main__":
    main()
