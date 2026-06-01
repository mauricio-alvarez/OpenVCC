import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, StandardScaler


def load_array(directory, name):
    path = Path(directory) / f"{name}.npy"
    if not path.exists():
        return None
    return np.load(path)


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_rep_dir(directory):
    directory = Path(directory)
    data = {
        "labels_local": load_array(directory, "labels_local"),
        "targets_imagenet": load_array(directory, "targets_imagenet"),
        "class_names": load_lines(directory / "class_names.txt"),
    }
    for path in directory.glob("*.npy"):
        key = path.stem
        if key not in data:
            data[key] = np.load(path)
    return data


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv(path, fieldnames, rows):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sample_rows(X, y, class_names, max_samples, seed):
    if max_samples is None or len(X) <= max_samples:
        return X, y, class_names
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], y[idx], [class_names[i] for i in idx]


def center_columns(X):
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y):
    X = center_columns(np.asarray(X, dtype=np.float64))
    Y = center_columns(np.asarray(Y, dtype=np.float64))
    hsic = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, ord="fro")
    norm_y = np.linalg.norm(Y.T @ Y, ord="fro")
    denom = norm_x * norm_y
    if denom == 0:
        return np.nan
    return float(hsic / denom)


def mean_pairwise_cosine(X, Y):
    if X.shape != Y.shape:
        return np.nan
    Xn = normalize(X)
    Yn = normalize(Y)
    return float(np.mean(np.sum(Xn * Yn, axis=1)))


def mse(X, Y):
    if X.shape != Y.shape:
        return np.nan
    return float(np.mean((X - Y) ** 2))


def fisher_ratio(X, y):
    classes = np.unique(y)
    global_mean = X.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in classes:
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        mu = Xc.mean(axis=0)
        between += len(Xc) * float(np.sum((mu - global_mean) ** 2))
        within += float(np.sum((Xc - mu) ** 2))
    if within == 0:
        return np.nan
    return between / within


def nearest_centroid_accuracy(X, y):
    classes = np.unique(y)
    centroids = {}
    for c in classes:
        centroids[c] = X[y == c].mean(axis=0)
    centroid_matrix = np.stack([centroids[c] for c in classes], axis=0)
    Xn = normalize(X)
    Cn = normalize(centroid_matrix)
    pred = classes[np.argmax(Xn @ Cn.T, axis=1)]
    return float(accuracy_score(y, pred))


def separability_metrics(X, y, max_samples, seed, linear_probe):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    valid = y >= 0
    X = X[valid]
    y = y[valid]
    if len(np.unique(y)) < 2 or len(X) < 4:
        return {
            "samples": len(X),
            "classes": len(np.unique(y)),
            "silhouette": np.nan,
            "fisher_ratio": np.nan,
            "nearest_centroid_acc": np.nan,
            "knn_acc": np.nan,
            "linear_probe_acc": np.nan,
        }

    X_sample, y_sample, _ = sample_rows(X, y, [""] * len(X), max_samples, seed)
    X_norm = normalize(X_sample)

    if len(X_sample) > len(np.unique(y_sample)):
        sil = float(silhouette_score(X_norm, y_sample, metric="cosine"))
    else:
        sil = np.nan

    fisher = float(fisher_ratio(X_norm, y_sample))
    centroid_acc = nearest_centroid_accuracy(X_norm, y_sample)

    stratify = y_sample if np.min(np.bincount(y_sample.astype(int))) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm,
        y_sample,
        test_size=0.3,
        random_state=seed,
        stratify=stratify,
    )

    n_neighbors = min(5, len(X_train))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(X_train, y_train)
    knn_acc = float(knn.score(X_test, y_test))

    if linear_probe:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
        clf.fit(X_train_s, y_train)
        linear_acc = float(clf.score(X_test_s, y_test))
    else:
        linear_acc = np.nan

    return {
        "samples": len(X_sample),
        "classes": len(np.unique(y_sample)),
        "silhouette": sil,
        "fisher_ratio": fisher,
        "nearest_centroid_acc": centroid_acc,
        "knn_acc": knn_acc,
        "linear_probe_acc": linear_acc,
    }


def compute_cka_rows(shvit, dhvit, dataset_name, max_samples, seed):
    rows = []
    pairs = [
        ("patch_embed", "patch_embed"),
        ("blocks1", "blocks1"),
        ("pooled", "feat_A"),
        ("pooled", "feat_B"),
        ("pooled", "feat_mean"),
        ("logits", "logits_ensemble"),
    ]
    y = shvit["labels_local"]
    class_names = shvit["class_names"]
    for sh_key, dh_key in pairs:
        if sh_key not in shvit or dh_key not in dhvit:
            continue
        X, _, _ = sample_rows(shvit[sh_key], y, class_names, max_samples, seed)
        Y, _, _ = sample_rows(dhvit[dh_key], y, class_names, max_samples, seed)
        rows.append(
            {
                "dataset": dataset_name,
                "shvit_feature": sh_key,
                "dhvit_feature": dh_key,
                "linear_cka": linear_cka(X, Y),
                "mean_pairwise_cosine": mean_pairwise_cosine(X, Y),
                "mse": mse(X, Y),
            }
        )
    return rows


def compute_branch_rows(dhvit, dataset_name):
    rows = []
    if "feat_A" in dhvit and "feat_B" in dhvit:
        rows.append(
            {
                "dataset": dataset_name,
                "comparison": "feat_A_vs_feat_B",
                "linear_cka": linear_cka(dhvit["feat_A"], dhvit["feat_B"]),
                "mean_pairwise_cosine": mean_pairwise_cosine(dhvit["feat_A"], dhvit["feat_B"]),
                "mse": mse(dhvit["feat_A"], dhvit["feat_B"]),
            }
        )
    if "logits_A" in dhvit and "logits_B" in dhvit:
        pred_a = dhvit["logits_A"].argmax(axis=1)
        pred_b = dhvit["logits_B"].argmax(axis=1)
        pred_e = dhvit["logits_ensemble"].argmax(axis=1)
        targets = dhvit["targets_imagenet"]
        valid = targets >= 0
        correct_a = pred_a[valid] == targets[valid]
        correct_b = pred_b[valid] == targets[valid]
        correct_e = pred_e[valid] == targets[valid]
        agree = pred_a[valid] == pred_b[valid]
        rows.append(
            {
                "dataset": dataset_name,
                "comparison": "predictions",
                "branch_a_acc": float(correct_a.mean()) if len(correct_a) else np.nan,
                "branch_b_acc": float(correct_b.mean()) if len(correct_b) else np.nan,
                "ensemble_acc": float(correct_e.mean()) if len(correct_e) else np.nan,
                "agreement_rate": float(agree.mean()) if len(agree) else np.nan,
                "oracle_branch_acc": float((correct_a | correct_b).mean()) if len(correct_a) else np.nan,
                "ensemble_fixes_branch_error": float((correct_e & ~(correct_a & correct_b)).mean())
                if len(correct_e)
                else np.nan,
            }
        )
    return rows


def compute_separability_rows(model_name, dataset_name, data, feature_keys, max_samples, seed, linear_probe):
    rows = []
    y = data["labels_local"]
    for key in feature_keys:
        if key not in data:
            continue
        metrics = separability_metrics(data[key], y, max_samples, seed, linear_probe)
        row = {"model": model_name, "dataset": dataset_name, "feature": key}
        row.update(metrics)
        rows.append(row)
    return rows


def centroid_shift_rows(source_data, shift_data, model_name, source_name, shift_name, feature_keys):
    rows = []
    source_classes = np.asarray(source_data["class_names"])
    shift_classes = np.asarray(shift_data["class_names"])
    common = sorted(set(source_classes).intersection(set(shift_classes)))
    if not common:
        return rows

    for key in feature_keys:
        if key not in source_data or key not in shift_data:
            continue
        Xs = normalize(source_data[key])
        Xt = normalize(shift_data[key])

        centroids = []
        centroid_classes = []
        shifts = []
        for class_name in common:
            s_mask = source_classes == class_name
            t_mask = shift_classes == class_name
            if not np.any(s_mask) or not np.any(t_mask):
                continue
            cs = Xs[s_mask].mean(axis=0)
            ct = Xt[t_mask].mean(axis=0)
            centroids.append(cs)
            centroid_classes.append(class_name)
            shifts.append(float(np.linalg.norm(cs - ct)))

        if not centroids:
            continue
        C = normalize(np.stack(centroids, axis=0))
        pred_idx = np.argmax(Xt @ C.T, axis=1)
        pred_classes = np.asarray(centroid_classes)[pred_idx]
        keep = np.isin(shift_classes, centroid_classes)
        cross_acc = float(np.mean(pred_classes[keep] == shift_classes[keep])) if np.any(keep) else np.nan

        rows.append(
            {
                "model": model_name,
                "source_dataset": source_name,
                "shift_dataset": shift_name,
                "feature": key,
                "classes": len(centroid_classes),
                "mean_centroid_shift_l2": float(np.mean(shifts)),
                "median_centroid_shift_l2": float(np.median(shifts)),
                "cross_domain_nearest_centroid_acc": cross_acc,
            }
        )
    return rows


def plot_pca(path, title, X, y, class_names, max_samples, seed):
    ensure_dir(Path(path).parent)
    Xs, ys, names = sample_rows(X, y, class_names, max_samples, seed)
    X2 = PCA(n_components=2, random_state=seed).fit_transform(normalize(Xs))
    unique = np.unique(ys)
    plt.figure(figsize=(10, 8))
    for label in unique:
        mask = ys == label
        label_names = sorted(set(np.asarray(names)[mask]))
        shown = label_names[0] if label_names else f"class_{label}"
        plt.scatter(X2[mask, 0], X2[mask, 1], s=10, alpha=0.65, label=shown)
    if len(unique) <= 20:
        plt.legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze saved SHViT/DHViT representation vectors.")
    parser.add_argument("--input-root", default="analysis/representation")
    parser.add_argument("--source-dataset", default="imagenet200")
    parser.add_argument("--shift-dataset", default="imagenetr")
    parser.add_argument("--output-dir", default="analysis/representation/results")
    parser.add_argument("--max-metric-samples", type=int, default=8000)
    parser.add_argument("--max-plot-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-probe", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(figure_dir)

    datasets = {}
    for dataset_name in [args.source_dataset, args.shift_dataset]:
        shvit_dir = input_root / "shvit" / dataset_name
        dhvit_dir = input_root / "dhvit" / dataset_name
        if shvit_dir.exists() and dhvit_dir.exists():
            datasets[dataset_name] = {
                "shvit": load_rep_dir(shvit_dir),
                "dhvit": load_rep_dir(dhvit_dir),
            }
        else:
            print(f"Skipping {dataset_name}; missing {shvit_dir} or {dhvit_dir}")

    cka_rows = []
    branch_similarity_rows = []
    separability_rows = []
    for dataset_name, pair in datasets.items():
        shvit = pair["shvit"]
        dhvit = pair["dhvit"]
        cka_rows.extend(compute_cka_rows(shvit, dhvit, dataset_name, args.max_metric_samples, args.seed))
        branch_similarity_rows.extend(compute_branch_rows(dhvit, dataset_name))
        separability_rows.extend(
            compute_separability_rows(
                "shvit",
                dataset_name,
                shvit,
                ["patch_embed", "blocks1", "blocks2", "blocks3", "pooled", "logits"],
                args.max_metric_samples,
                args.seed,
                args.linear_probe,
            )
        )
        separability_rows.extend(
            compute_separability_rows(
                "dhvit",
                dataset_name,
                dhvit,
                ["patch_embed", "blocks1", "blocks2_A", "blocks2_B", "blocks3_A", "blocks3_B", "feat_A", "feat_B", "feat_mean", "logits_ensemble"],
                args.max_metric_samples,
                args.seed,
                args.linear_probe,
            )
        )

        if "pooled" in shvit:
            plot_pca(
                figure_dir / f"pca_shvit_{dataset_name}_pooled.png",
                f"SHViT pooled features - {dataset_name}",
                shvit["pooled"],
                shvit["labels_local"],
                shvit["class_names"],
                args.max_plot_samples,
                args.seed,
            )
        if "feat_mean" in dhvit:
            plot_pca(
                figure_dir / f"pca_dhvit_{dataset_name}_feat_mean.png",
                f"DHViT mean branch features - {dataset_name}",
                dhvit["feat_mean"],
                dhvit["labels_local"],
                dhvit["class_names"],
                args.max_plot_samples,
                args.seed,
            )

    if cka_rows:
        write_csv(output_dir / "cka_shvit_vs_dhvit.csv", list(cka_rows[0].keys()), cka_rows)
    if branch_similarity_rows:
        keys = sorted({key for row in branch_similarity_rows for key in row.keys()})
        write_csv(output_dir / "branch_similarity.csv", keys, branch_similarity_rows)
    if separability_rows:
        write_csv(output_dir / "separability.csv", list(separability_rows[0].keys()), separability_rows)

    centroid_rows = []
    if args.source_dataset in datasets and args.shift_dataset in datasets:
        source = datasets[args.source_dataset]
        shift = datasets[args.shift_dataset]
        centroid_rows.extend(
            centroid_shift_rows(
                source["shvit"],
                shift["shvit"],
                "shvit",
                args.source_dataset,
                args.shift_dataset,
                ["patch_embed", "blocks1", "blocks3", "pooled", "logits"],
            )
        )
        centroid_rows.extend(
            centroid_shift_rows(
                source["dhvit"],
                shift["dhvit"],
                "dhvit",
                args.source_dataset,
                args.shift_dataset,
                ["patch_embed", "blocks1", "blocks3_A", "blocks3_B", "feat_A", "feat_B", "feat_mean", "logits_ensemble"],
            )
        )
    if centroid_rows:
        write_csv(output_dir / "centroid_shift.csv", list(centroid_rows[0].keys()), centroid_rows)

    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
