from train_model import set_seed, data_loader, build_shvit, build_timm, build_base, train_model_base, train_model_timm, train_monster, build_and_load_monster, load_finetuned_monster, validate, test_imagenet_r, test_imagenet_1k, build_pruned
import torch
import os

if __name__ == '__main__':
    os.environ['TORCH_HOME'] = 'model_cache' 
    seed = 42
    seed_worker = set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Initialize Variables
    batch_size = 512
    vit_large = 'vit_large_patch16_224'
    vit_tiny = 'vit_tiny_patch16_224'
    vit_small = 'vit_small_patch16_224'
    vit_base = 'vit_base_patch16_224'
    # PRUNED MODELS
    vit_tiny_pruned = '/home/mauricio.alvarez/tesis/sanity-check/pruned_vit_tiny.pth'
    vit_small_pruned = '/home/mauricio.alvarez/tesis/sanity-check/pruned_vit_small_02.pth'
    vit_base_pruned = '/home/mauricio.alvarez/tesis/sanity-check/pruned_vit_base_02.pth'
    # SHViT
    shvit_s1 = '/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s1.pth'
    shvit_s4 = '/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s4.pth'
    SAVE_DIR = '/home/mauricio.alvarez/tesis/VCC/model_weights'
    #IMAGES_DIR = '/home/mauricio.alvarez/tesis/archive/session-1'
    #IMAGES_DIR = '/home/mauricio.alvarez/tesis/archive/imagenet-val/imagenet-val'
    IMAGES_DIR = '/home/mauricio.alvarez/tesis/archive/imagenet_train/'
    #IMAGES_DIR = '/home/mauricio.alvarez/tesis/archive/imagenet-r'
    #NUM_CLASS = 16
    #NUM_CLASS = 200
    NUM_CLASS = 1000
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")
    # Train a model: Vit Base and Large
    #train_loader, val_loader = data_loader(IMAGES_DIR, batch_size, seed_worker, 0.7)
    #print("Model Using: {} pruned 30%".format(vit_small))
    #model = build_shvit('s1', shvit_s1, NUM_CLASS)
    #model = build_timm(vit_large, NUM_CLASS)
    #model = build_pruned(vit_small, vit_small_pruned)
    #model.to(device)
    #print(model)
    #model_finetuned = train_model_base(model, train_loader, val_loader, 'ViT-B16', SAVE_DIR=SAVE_DIR,num_epochs=5)
    #model_finetuned = train_model_timm(model, train_loader, val_loader, 'ViT-Large', SAVE_DIR=SAVE_DIR,num_epochs=5)
    #test_imagenet_r(model)
    #test_imagenet_1k(model, set_name='val')
    #model_finetuned = train_monster(shvit_s1, train_loader, val_loader, 50)
    #monster = train_monster(True, shvit_s1, train_loader, val_loader, NUM_CLASS, 300)
    
    print("Testing load_finetuned_monster...")
    checkpoint_path = "/home/mauricio.alvarez/tesis/VCC/shvit_s1_doublehead_0103_300epochs_pretrained.pth"
    # checkpoint_path = "/home/mauricio.alvarez/tesis/VCC/shvit_s1_doublehead_0103_300epochs_pretrained.pth"
    if os.path.exists(checkpoint_path):
        monster = load_finetuned_monster(checkpoint_path, num_classes=NUM_CLASS, device=device)
        test_imagenet_r(monster)
        test_imagenet_1k(monster, set_name='val')
        #monster = train_monster(False, checkpoint_path, train_loader, val_loader, NUM_CLASS, 30)
        #print("Running validation on loaded model...")
        #acc = validate(monster, val_loader, device)
        #print(f"Validation Accuracy: {acc:.2f}%")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Skipping validation.")
    '''
    '''


