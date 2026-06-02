import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from representation_extract import (
    SelectedImageFolder,
    append_outputs,
    collate_batch,
    forward_dhvit,
    get_model_cfg,
    load_imagenet_index,
    read_class_names,
    build_shvit_model,
    forward_shvit,
)
from SHViT import DoubleHeadSHViT
from torch.utils.data import DataLoader


STAGE1_PREFIXES = ("patch_embed.", "blocks1.")


def load_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint, checkpoint["model"], True
    return checkpoint, checkpoint, False


def maybe_expand_conv_weight(source, target):
    if target.ndim == 4 and source.ndim == 2:
        return source[:, :, None, None]
    return source


def copy_stage1_state(shvit_state, dhvit_state):
    repaired = copy.deepcopy(dhvit_state)
    copied = []
    skipped = []

    for key, source_value in shvit_state.items():
        if not key.startswith(STAGE1_PREFIXES):
            continue
        if key not in repaired:
            skipped.append((key, "missing_in_dhvit"))
            continue

        target_value = repaired[key]
        source_value = maybe_expand_conv_weight(source_value, target_value)
        if tuple(source_value.shape) != tuple(target_value.shape):
            skipped.append((key, f"shape {tuple(source_value.shape)} != {tuple(target_value.shape)}"))
            continue

        repaired[key] = source_value.detach().clone()
        copied.append(key)

    return repaired, copied, skipped


def save_repaired_checkpoint(original_checkpoint, repaired_state, checkpoint_has_model, output_path, copied_keys):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_has_model:
        output_checkpoint = copy.deepcopy(original_checkpoint)
        output_checkpoint["model"] = repaired_state
        output_checkpoint["stage1_repair"] = {
            "copied_prefixes": list(STAGE1_PREFIXES),
            "num_copied_keys": len(copied_keys),
        }
        torch.save(output_checkpoint, output_path)
    else:
        torch.save(repaired_state, output_path)


def repair_checkpoint(args):
    print(f"Loading SHViT checkpoint: {args.shvit_checkpoint}")
    _, shvit_state, _ = load_checkpoint(args.shvit_checkpoint, map_location="cpu")
    print(f"Loading DHViT checkpoint: {args.dhvit_checkpoint}")
    dhvit_checkpoint, dhvit_state, checkpoint_has_model = load_checkpoint(args.dhvit_checkpoint, map_location="cpu")

    repaired_state, copied, skipped = copy_stage1_state(shvit_state, dhvit_state)
    save_repaired_checkpoint(dhvit_checkpoint, repaired_state, checkpoint_has_model, args.output_checkpoint, copied)

    print(f"Copied {len(copied)} stage-1/shared-stem tensors into DHViT.")
    if skipped:
        print(f"Skipped {len(skipped)} keys:")
        for key, reason in skipped[:20]:
            print(f"  {key}: {reason}")
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")

    report = {
        "shvit_checkpoint": args.shvit_checkpoint,
        "dhvit_checkpoint": args.dhvit_checkpoint,
        "output_checkpoint": args.output_checkpoint,
        "copied_prefixes": list(STAGE1_PREFIXES),
        "num_copied_keys": len(copied),
        "copied_keys": copied,
        "skipped": skipped,
    }
    report_path = Path(args.output_checkpoint).with_suffix(".stage1_repair.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Repair report saved to {report_path}")


def load_dhvit_model(checkpoint_path, model_size, num_classes, device):
    model = DoubleHeadSHViT(num_classes=num_classes, **get_model_cfg(model_size))
    _, state_dict, _ = load_checkpoint(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    adjusted = {}
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape != value.shape:
            value = maybe_expand_conv_weight(value, model_state[key])
            if model_state[key].shape != value.shape:
                print(f"Skipping shape-mismatched key {key}: {tuple(value.shape)} vs {tuple(model_state[key].shape)}")
                continue
        adjusted[key] = value
    missing, unexpected = model.load_state_dict(adjusted, strict=False)
    if missing:
        print(f"Missing keys while loading repaired DHViT: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading repaired DHViT: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def load_shvit_model(checkpoint_path, model_size, num_classes, device):
    model = build_shvit_model(model_size, num_classes)
    _, state_dict, _ = load_checkpoint(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    adjusted = {}
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape != value.shape:
            value = maybe_expand_conv_weight(value, model_state[key])
            if model_state[key].shape != value.shape:
                print(f"Skipping shape-mismatched key {key}: {tuple(value.shape)} vs {tuple(model_state[key].shape)}")
                continue
        adjusted[key] = value
    missing, unexpected = model.load_state_dict(adjusted, strict=False)
    if missing:
        print(f"Missing keys while loading SHViT: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading SHViT: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def extract_features(args, checkpoint_path, model_kind="dhvit"):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    wnid_to_idx, _ = load_imagenet_index(args.imagenet_index)
    class_names = read_class_names(args, args.dataset_root)

    dataset = SelectedImageFolder(
        args.dataset_root,
        class_names,
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

    print(f"Extracting {model_kind.upper()} activations from {len(dataset)} images on {device}.")
    if model_kind == "dhvit":
        model = load_dhvit_model(checkpoint_path, args.model_size, args.num_classes, device)
        forward_fn = forward_dhvit
    else:
        model = load_shvit_model(checkpoint_path, args.model_size, args.num_classes, device)
        forward_fn = forward_shvit

    storage = {}
    labels = []
    targets = []
    paths = []
    sample_classes = []

    with torch.no_grad():
        for images, labels_local, targets_imagenet, batch_paths, batch_classes in tqdm(loader, desc=f"Extracting {model_kind.upper()}"):
            images = images.to(device, non_blocking=True)
            outputs = forward_fn(model, images)
            append_outputs(storage, outputs)
            labels.extend(labels_local.numpy().tolist())
            targets.extend(targets_imagenet.numpy().tolist())
            paths.extend(batch_paths)
            sample_classes.extend(batch_classes)

    output_dir = Path(args.activation_output_dir) / model_kind
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, arrays in storage.items():
        np.save(output_dir / f"{key}.npy", np.concatenate(arrays, axis=0).astype(np.float32))
    np.save(output_dir / "labels_local.npy", np.asarray(labels, dtype=np.int64))
    np.save(output_dir / "targets_imagenet.npy", np.asarray(targets, dtype=np.int64))
    with open(output_dir / "class_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sample_classes))
    with open(output_dir / "paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(paths))
    print(f"Saved extracted activation files for {model_kind} to {output_dir}")
    return output_dir


def load_plot_inputs(activation_dir, feature_key):
    activation_dir = Path(activation_dir)
    feature_path = activation_dir / f"{feature_key}.npy"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    X = np.load(feature_path)
    y = np.load(activation_dir / "labels_local.npy")
    class_names_path = activation_dir / "class_names.txt"
    if class_names_path.exists():
        with open(class_names_path, "r", encoding="utf-8") as f:
            per_sample_names = [line.strip() for line in f if line.strip()]
    else:
        per_sample_names = [f"class_{int(label)}" for label in y]
    return X, y, per_sample_names


def subsample_for_umap(X, y, class_names, max_points, seed):
    if max_points is None or len(X) <= max_points:
        return X, y, class_names
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=max_points, replace=False)
    return X[indices], y[indices], [class_names[i] for i in indices]


def plot_umap_3d(args, activation_dir, feature_key, title, output_path):
    import pandas as pd
    import plotly.express as px
    import umap

    X, y, per_sample_names = load_plot_inputs(activation_dir, feature_key)
    X, y, per_sample_names = subsample_for_umap(X, y, per_sample_names, args.max_umap_points, args.seed)

    print(f"Running 3D UMAP for {len(X)} points from feature '{feature_key}'.")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    )
    embedding = reducer.fit_transform(X)

    df = pd.DataFrame(
        {
            "umap_x": embedding[:, 0],
            "umap_y": embedding[:, 1],
            "umap_z": embedding[:, 2],
            "label_id": y.astype(str),
            "class_name": per_sample_names,
        }
    )
    fig = px.scatter_3d(
        df,
        x="umap_x",
        y="umap_y",
        z="umap_z",
        color="class_name",
        hover_data=["label_id", "class_name"],
        title=title,
        opacity=0.72,
    )
    fig.update_traces(marker=dict(size=args.marker_size))

    output_html = Path(output_path)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html)
    np.save(output_html.with_suffix(".embedding.npy"), embedding.astype(np.float32))
    print(f"Saved 3D UMAP HTML to {output_html}")
    print(f"Saved UMAP embedding to {output_html.with_suffix('.embedding.npy')}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy SHViT-S1 shared stem/stage1 tensors into a DHViT checkpoint and plot 3D UMAP last-layer activations."
    )
    parser.add_argument("--shvit-checkpoint", required=True)
    parser.add_argument("--dhvit-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--model-size", choices=["s1", "s2", "s3", "s4"], default="s1")
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--plot-umap", action="store_true")
    parser.add_argument("--plot-shvit-umap", action="store_true", help="Also plot the UMAP of the original SHViT architecture.")
    parser.add_argument("--activation-dir", default=None, help="Existing activation directory to plot from.")
    parser.add_argument("--activation-output-dir", default="analysis/repaired_stage1_umap")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset-name", default="imagenetr")
    parser.add_argument("--imagenet-index", default="imagenet_class_index.json")
    parser.add_argument("--class-file", default=None)
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--feature-key", default="feat_mean", choices=["feat_A", "feat_B", "feat_mean", "logits_ensemble"])
    parser.add_argument("--shvit-feature-key", default="pooled", choices=["patch_embed", "blocks1", "blocks2", "blocks3", "pooled", "logits"])
    parser.add_argument("--umap-output", default="analysis/repaired_stage1_umap/dhvit_repaired_feat_mean_umap3d.html")
    parser.add_argument("--shvit-umap-output", default="analysis/repaired_stage1_umap/shvit_pooled_umap3d.html")
    parser.add_argument("--max-umap-points", type=int, default=10000)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", default="cosine")
    parser.add_argument("--marker-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-repaired", action="store_true", help="Run ImageNet-1K and ImageNet-R evaluation functions from train_model.py")
    args = parser.parse_args()

    repair_checkpoint(args)

    if args.plot_umap:
        # Plot DHViT UMAP
        if args.activation_dir:
            dhvit_activation_dir = Path(args.activation_dir)
            if (dhvit_activation_dir / "dhvit").exists():
                dhvit_activation_dir = dhvit_activation_dir / "dhvit"
        else:
            if not args.dataset_root:
                raise ValueError("--dataset-root is required when --plot-umap is used without --activation-dir")
            dhvit_activation_dir = extract_features(args, args.output_checkpoint, model_kind="dhvit")
        
        plot_umap_3d(
            args,
            dhvit_activation_dir,
            feature_key=args.feature_key,
            title=f"3D UMAP of DHViT {args.feature_key}",
            output_path=args.umap_output
        )

        # Plot SHViT UMAP if requested
        if args.plot_shvit_umap:
            if args.activation_dir:
                shvit_activation_dir = Path(args.activation_dir)
                if (shvit_activation_dir / "shvit").exists():
                    shvit_activation_dir = shvit_activation_dir / "shvit"
            else:
                shvit_activation_dir = extract_features(args, args.shvit_checkpoint, model_kind="shvit")
            
            plot_umap_3d(
                args,
                shvit_activation_dir,
                feature_key=args.shvit_feature_key,
                title=f"3D UMAP of SHViT {args.shvit_feature_key}",
                output_path=args.shvit_umap_output
            )

    if args.eval_repaired:
        print("\n--- Running Evaluation of Repaired Model on ImageNet-1K and ImageNet-R ---")
        from train_model import test_imagenet_1k, test_imagenet_r
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        repaired_model = load_dhvit_model(args.output_checkpoint, args.model_size, args.num_classes, device)
        
        print("Running test_imagenet_1k...")
        test_imagenet_1k(repaired_model, set_name="val", batch_size=args.batch_size, num_workers=args.num_workers)
        
        print("\nRunning test_imagenet_r...")
        test_imagenet_r(repaired_model)


if __name__ == "__main__":
    main()
