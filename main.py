from train_model import set_seed, data_loader, build_shvit, build_timm, build_base, train_model_base, train_model_timm
import torch
import os

if __name__ == '__main__':
    seed = 42
    seed_worker = set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Initialize Variables
    batch_size = 32
    vit_large = 'vit_large_patch16_224'
    vit_tiny = 'vit_tiny_patch16_224'
    shvit_s1 = '/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s1.pth'
    SAVE_DIR = '/home/mauricio.alvarez/tesis/VCC/model_weights'
    IMAGES_DIR = '/home/mauricio.alvarez/tesis/archive/session-1'
    NUM_CLASS = 16
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")
    # Train a model: Vit Base and Large
    train_loader, val_loader = data_loader(IMAGES_DIR, batch_size, seed_worker)
    model_large = build_shvit('s1', shvit_s1, NUM_CLASS)
    model_large.to(device)
    print(model_large)
    large_finetuned = train_model_timm(model_large, train_loader, val_loader, 'SH-ViT', SAVE_DIR=SAVE_DIR,20)
    



