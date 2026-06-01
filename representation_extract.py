import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from SHViT import (
    DoubleHeadSHViT,
    SHViT_s1,
    SHViT_s2,
    SHViT_s3,
    SHViT_s4,
    shvit_s1,
    shvit_s2,
    shvit_s3,
    shvit_s4,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_imagenet_index(path):
    with open(path, "r", encoding="utf-8") as f:
        class_index = json.load(f)
    wnid_to_idx = {}
    wnid_to_name = {}
    for idx, value in class_index.items():
        wnid, name = value
        wnid_to_idx[wnid] = int(idx)
        wnid_to_name[wnid] = name
    return wnid_to_idx, wnid_to_name


def read_class_names(args, dataset_root):
    if args.class_file:
        with open(args.class_file, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    elif args.class_names:
        class_names = args.class_names
    else:
        class_names = sorted([p.name for p in Path(dataset_root).iterdir() if p.is_dir()])

    if args.max_classes is not None:
        class_names = class_names[: args.max_classes]
    return class_names


class SelectedImageFolder(Dataset):
    def __init__(self, root, class_names, wnid_to_idx, max_images_per_class, seed):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.wnid_to_idx = wnid_to_idx
        self.samples = []

        rng = random.Random(seed)
        for local_label, class_name in enumerate(self.class_names):
            class_dir = self.root / class_name
            if not class_dir.exists():
                print(f"Warning: missing class directory {class_dir}")
                continue

            files = [
                p
                for p in sorted(class_dir.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            rng.shuffle(files)
            if max_images_per_class is not None:
                files = files[:max_images_per_class]

            imagenet_target = wnid_to_idx.get(class_name, -1)
            for path in files:
                self.samples.append((path, local_label, imagenet_target, class_name))

        if not self.samples:
            raise ValueError(f"No images found under {root} for classes: {self.class_names}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, local_label, imagenet_target, class_name = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), local_label, imagenet_target, str(path), class_name


def collate_batch(batch):
    images, local_labels, imagenet_targets, paths, class_names = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.tensor(local_labels, dtype=torch.long),
        torch.tensor(imagenet_targets, dtype=torch.long),
        list(paths),
        list(class_names),
    )


def get_model_cfg(model_size):
    cfgs = {"s1": SHViT_s1, "s2": SHViT_s2, "s3": SHViT_s3, "s4": SHViT_s4}
    return cfgs[model_size]


def build_shvit_model(model_size, num_classes):
    builders = {"s1": shvit_s1, "s2": shvit_s2, "s3": shvit_s3, "s4": shvit_s4}
    return builders[model_size](num_classes=num_classes)


def load_state_dict(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_model(args, device):
    if args.model_kind == "shvit":
        model = build_shvit_model(args.model_size, args.num_classes)
    elif args.model_kind == "dhvit":
        model = DoubleHeadSHViT(num_classes=args.num_classes, **get_model_cfg(args.model_size))
    else:
        raise ValueError(f"Unsupported model kind: {args.model_kind}")

    state_dict = load_state_dict(args.checkpoint, device)
    model_state = model.state_dict()
    adjusted_state = {}
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape != value.shape:
            if model_state[key].ndim == 4 and value.ndim == 2:
                value = value[:, :, None, None]
            else:
                print(f"Skipping shape-mismatched key {key}: checkpoint {tuple(value.shape)} vs model {tuple(model_state[key].shape)}")
                continue
        adjusted_state[key] = value
    state_dict = adjusted_state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys while loading {args.model_kind}: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading {args.model_kind}: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def pool_feature(x):
    if x.ndim == 4:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)
    if x.ndim == 3:
        return x.mean(dim=1)
    return x.flatten(1)


def forward_shvit(model, images):
    out = {}
    x = model.patch_embed(images)
    out["patch_embed"] = pool_feature(x)
    x = model.blocks1(x)
    out["blocks1"] = pool_feature(x)
    x = model.blocks2(x)
    out["blocks2"] = pool_feature(x)
    x = model.blocks3(x)
    out["blocks3"] = pool_feature(x)
    pooled = pool_feature(x)
    out["pooled"] = pooled
    out["logits"] = model.head(pooled)
    return out


def forward_dhvit(model, images):
    out = {}
    x = model.patch_embed(images)
    out["patch_embed"] = pool_feature(x)
    x = model.blocks1(x)
    out["blocks1"] = pool_feature(x)

    x_a = model.ds_1_to_2_A(x)
    x_a = model.blocks2_A(x_a)
    out["blocks2_A"] = pool_feature(x_a)
    x_a = model.ds_2_to_3_A(x_a)
    x_a = model.blocks3_A(x_a)
    out["blocks3_A"] = pool_feature(x_a)
    feat_a = pool_feature(x_a)

    x_b = model.ds_1_to_2_B(x)
    x_b = model.blocks2_B(x_b)
    out["blocks2_B"] = pool_feature(x_b)
    x_b = model.ds_2_to_3_B(x_b)
    x_b = model.blocks3_B(x_b)
    out["blocks3_B"] = pool_feature(x_b)
    feat_b = pool_feature(x_b)

    logits_a = model.head_A(feat_a)
    logits_b = model.head_B(feat_b)
    out["feat_A"] = feat_a
    out["feat_B"] = feat_b
    out["feat_mean"] = (feat_a + feat_b) / 2
    out["logits_A"] = logits_a
    out["logits_B"] = logits_b
    out["logits_ensemble"] = (logits_a + logits_b) / 2
    return out


def append_outputs(storage, batch_outputs):
    for key, value in batch_outputs.items():
        storage.setdefault(key, []).append(value.detach().cpu().float().numpy())


def save_outputs(output_dir, storage, local_labels, imagenet_targets, paths, class_names, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, arrays in storage.items():
        np.save(output_dir / f"{key}.npy", np.concatenate(arrays, axis=0).astype(np.float32))

    np.save(output_dir / "labels_local.npy", np.asarray(local_labels, dtype=np.int64))
    np.save(output_dir / "targets_imagenet.npy", np.asarray(imagenet_targets, dtype=np.int64))
    with open(output_dir / "paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(paths))
    with open(output_dir / "class_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    class_order = []
    for _, class_name in sorted(set(zip(local_labels, class_names))):
        class_order.append(class_name)
    with open(output_dir / "class_order.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_order))

    metadata = {
        "model_kind": args.model_kind,
        "model_size": args.model_size,
        "checkpoint": args.checkpoint,
        "dataset_root": args.dataset_root,
        "dataset_name": args.dataset_name,
        "num_samples": len(local_labels),
        "num_classes": len(set(class_names)),
        "max_images_per_class": args.max_images_per_class,
        "class_order": class_order,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract SHViT/DHViT representation vectors.")
    parser.add_argument("--model-kind", choices=["shvit", "dhvit"], required=True)
    parser.add_argument("--model-size", choices=["s1", "s2", "s3", "s4"], default="s1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imagenet-index", default="imagenet_class_index.json")
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument("--class-file", default=None)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    wnid_to_idx, _ = load_imagenet_index(args.imagenet_index)
    selected_classes = read_class_names(args, args.dataset_root)
    dataset = SelectedImageFolder(
        args.dataset_root,
        selected_classes,
        wnid_to_idx,
        args.max_images_per_class,
        args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    print(f"Loaded {len(dataset)} images across {len(selected_classes)} requested classes.")
    model = load_model(args, device)
    forward_fn = forward_shvit if args.model_kind == "shvit" else forward_dhvit

    storage = {}
    all_local_labels = []
    all_imagenet_targets = []
    all_paths = []
    all_class_names = []

    with torch.no_grad():
        for images, labels_local, targets_imagenet, paths, class_names in tqdm(loader, desc="Extracting"):
            images = images.to(device, non_blocking=True)
            batch_outputs = forward_fn(model, images)
            append_outputs(storage, batch_outputs)
            all_local_labels.extend(labels_local.numpy().tolist())
            all_imagenet_targets.extend(targets_imagenet.numpy().tolist())
            all_paths.extend(paths)
            all_class_names.extend(class_names)

    output_dir = Path(args.output_dir) / args.model_kind / args.dataset_name
    save_outputs(output_dir, storage, all_local_labels, all_imagenet_targets, all_paths, all_class_names, args)
    print(f"Saved representation files to {output_dir}")


if __name__ == "__main__":
    main()
