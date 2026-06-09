from train_model import set_seed, data_loader, build_shvit, build_timm, build_base, train_model_base, train_model_timm, train_monster, train_monster_improved, build_and_load_monster, load_finetuned_monster, validate, test_imagenet_r, test_imagenet_1k, build_pruned, load_finetuned_triple_monster, train_typhon_imagenet_1k
import torch
import os
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate pruned models")
    parser.add_argument('--model_name', type=str, default='vit_small_patch16_224', help='ViT model name')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to pruned checkpoint')
    args, unknown = parser.parse_known_args()

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
    vit_tiny_pruned = '/home/mauricio.alvarez/tesis/sanity-check/weights/tiny_30heads.pth'
    vit_small_pruned = '/home/mauricio.alvarez/tesis/sanity-check/weights/small_60heads.pth'
    vit_base_pruned = '/home/mauricio.alvarez/tesis/sanity-check/weights/base_120heads.pth'
    # SHViT
    shvit_s1 = '/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s1.pth'
    shvit_s2 = '/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s2.pth'
    shvit_s3 = '/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s3.pth'
    shvit_s4 = '/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s4.pth'
    thvit_s1 = "/home/mauricio.alvarez/tesis/VCC/model_weights/experiments/4_shvit_s1_triplehead_0405_150epochs.pth"
    #thvit_s4 = "/home/mauricio.alvarez/tesis/VCC/model_weights/DoubleHeadViT/shvit_s1_triplehead_2704_150epochs.pth"
    #dhvit_s1 = "/home/mauricio.alvarez/tesis/VCC/model_weights/DoubleHeadViT/shvit_s1_doublehead_0103_300epochs_pretrained.pth"
    dhvit_s1 = "/home/mauricio.alvarez/tesis/VCC/model_weights/experiments/42_shvit_s1_doublehead_1805_100epochs.pth"
    #dhvit_s4 = "/home/mauricio.alvarez/tesis/VCC/model_weights/experiments/20_shvit_s4_doublehead_0405_150epochs.pth"
    dhvit_s4 = "/home/mauricio.alvarez/tesis/VCC/model_weights/experiments/3_shvit_s4_doublehead_0405_150epochs.pth"
    
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
    if args.checkpoint_path is not None:
        model_name_to_use = args.model_name
        checkpoint_path_to_use = args.checkpoint_path
    else:
        model_name_to_use = vit_small
        checkpoint_path_to_use = vit_small_pruned

    print(f"Model Using: {checkpoint_path_to_use} Pruned")
    #model = build_shvit('s4', shvit_s4, NUM_CLASS)
    #model = build_timm(vit_small, NUM_CLASS)
    #model = build_pruned(model_name_to_use, checkpoint_path_to_use)
    #model = load_finetuned_monster(dhvit_s1, num_classes=NUM_CLASS, device=device)
    #model = load_finetuned_triple_monster(thvit_s1, num_classes=NUM_CLASS, device=device)
    #model.to(device)
    #print(model)
    #model_finetuned = train_model_base(model, train_loader, val_loader, 'ViT-B16', SAVE_DIR=SAVE_DIR,num_epochs=5)
    #model_finetuned = train_model_timm(model, train_loader, val_loader, 'ViT-Large', SAVE_DIR=SAVE_DIR,num_epochs=5)
    #model = train_monster(True,2, shvit_s1, train_loader, val_loader, NUM_CLASS, 50, 'shvit_s1_doublehead_1805_100epochs.pth', resume_checkpoint=dhvit_s1)

    #test_imagenet_r(model)
    #test_imagenet_1k(model, set_name='val')
    #model_finetuned = train_monster(shvit_s1, train_loader, val_loader, 50)

    train_typhon_imagenet_1k(
        use_shvit=True,
        model_location="/home/mauricio.alvarez/tesis/VCC/model_weights/SHViT/shvit_s1.pth",
        NUM_EPOCHS=12,
        output_file_name="typhon_s1_imagenet1k.pth",
        batch_size=256,
        num_mixers=2,
        resume_checkpoint="4_typhon_s1_imagenet1k.pth"
    )
    
    '''
    ckpt = '/home/mauricio.alvarez/tesis/VCC/13_shvit_s1_improved_0905_150epochs.pth'
    model = train_monster_improved(
    use_shvit=True, 
    num_heads=2, 
    model_location=shvit_s1, 
    loader_train=train_loader, 
    loader_eval=val_loader, 
    NUM_CLASS=NUM_CLASS, 
    NUM_EPOCHS=150, 
    output_file_name='shvit_s1_improved_0905_150epochs.pth',
    resume_checkpoint=None)
    
    print("Testing load_finetuned_monster...")
    
    if os.path.exists(checkpoint_path):
        #monster = load_finetuned_monster(checkpoint_path, num_classes=NUM_CLASS, device=device)
        #print("Testing load_finetuned_triple_monster...")
        #monster_triple = build_and_load_triple_monster(monster)
        #test_imagenet_r(monster_triple)
        #test_imagenet_1k(monster_triple, set_name='val')
        
        #print("Running validation on loaded model...")
        #acc = validate(monster, val_loader, device)
        #print(f"Validation Accuracy: {acc:.2f}%")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Skipping validation.")
    '''
    


