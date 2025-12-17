from train_model import set_seed, data_loader, build_shvit, build_timm, build_base, train_model_base, train_model_timm, train_monster, build_and_load_monster
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
    shvit_s4 = '/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s4.pth'
    SAVE_DIR = '/home/mauricio.alvarez/tesis/VCC/model_weights'
    #IMAGES_DIR = '/home/mauricio.alvarez/tesis/archive/session-1'
    IMAGES_DIR = '/home/mauricio.alvarez/tesis/imagenet_val/imagenet_val'
    #IMAGES_DIR = '/home/mauricio.alvarez/tesis/archive/imagenet-r'
    #NUM_CLASS = 16
    #NUM_CLASS = 200
    NUM_CLASS = 1000
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")
    # Train a model: Vit Base and Large
    train_loader, val_loader = data_loader(IMAGES_DIR, batch_size, seed_worker, 0.7)
    #model = build_shvit('s1', shvit_s1, NUM_CLASS)
    #model = build_timm(vit_large, NUM_CLASS)
    #model = build_base(NUM_CLASS)
    #model = build_timm(vit_tiny, NUM_CLASS)
    model = build_and_load_monster(shvit_s1)
    model.to(device)
    print(model)
    #model_finetuned = train_model_base(model, train_loader, val_loader, 'ViT-B16', SAVE_DIR=SAVE_DIR,num_epochs=15)
    #model_finetuned = train_model_timm(model, train_loader, val_loader, 'ViT-Large', SAVE_DIR=SAVE_DIR,num_epochs=15)
    model_dinetuned = train_monster(model, train_loader, val_loader)



