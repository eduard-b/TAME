import os
import csv
import math
import numpy as np
import torch

from utils.utils import ensure_dir
from data.prepare_database import prepare_db, DATASET_REGISTRY


def safe_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def class_stats(y: torch.Tensor, num_classes: int):
    """
    Returns counts, probs, and imbalance metrics.
    """
    y_cpu = y.detach().cpu()
    counts = torch.bincount(y_cpu, minlength=num_classes).double()
    n = counts.sum().item()
    probs = (counts / max(n, 1.0)).clamp(min=0)

    # Basic
    majority_frac = probs.max().item() if n > 0 else float("nan")
    minority_frac = probs[probs > 0].min().item() if (probs > 0).any() else float("nan")

    # Ratios
    min_count = counts[counts > 0].min().item() if (counts > 0).any() else 0.0
    max_count = counts.max().item() if n > 0 else 0.0
    max_min_ratio = (max_count / min_count) if min_count > 0 else float("inf")

    # Entropy (normalized)
    eps = 1e-12
    ent = -(probs * (probs + eps).log()).sum().item()
    ent_norm = ent / math.log(max(num_classes, 2))

    # Gini impurity (higher = more balanced up to max)
    gini = 1.0 - (probs * probs).sum().item()

    # Effective number of classes (perplexity)
    eff_classes = math.exp(ent) if n > 0 else float("nan")

    # Balancedness index: min/mean count
    mean_count = (n / max(num_classes, 1)) if n > 0 else float("nan")
    min_over_mean = (min_count / mean_count) if mean_count and mean_count > 0 else float("nan")

    return {
        "n_train": int(n),
        "min_class_count": float(min_count),
        "max_class_count": float(max_count),
        "max_min_ratio": float(max_min_ratio),
        "majority_frac": float(majority_frac),
        "minority_frac": float(minority_frac),
        "class_entropy": float(ent),
        "class_entropy_norm": float(ent_norm),
        "gini_impurity": float(gini),
        "effective_num_classes": float(eff_classes),
        "min_over_mean_count": float(min_over_mean),
    }


def feature_shape_stats(
    X: torch.Tensor,
    max_rows: int = 50000,
    max_dims: int = 200,
    seed: int = 0,
):
    """
    Computes feature distribution proxies with subsampling:
      - frac_zeros
      - mean_abs, mean_std
      - avg_abs_skew
      - avg_kurtosis (non-excess)
      - outlier_rate_3std : fraction of |z|>3 (approx tailness)
    """
    # Work on CPU for stability
    X = X.detach()
    if X.is_cuda:
        X = X.cpu()

    n, d = X.shape
    rng = np.random.default_rng(seed)

    # Subsample rows
    if n > max_rows:
        idx = rng.choice(n, size=max_rows, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    # Subsample dims
    if d > max_dims:
        jdx = rng.choice(d, size=max_dims, replace=False)
        Xs = Xs[:, jdx]
    else:
        Xs = Xs

    # Basic scale & sparsity
    frac_zeros = (Xs == 0).float().mean().item()
    mean_abs = Xs.abs().mean().item()

    mu = Xs.mean(dim=0, keepdim=True)
    xc = Xs - mu
    var = (xc * xc).mean(dim=0)
    std = torch.sqrt(var + 1e-12)
    mean_std = std.mean().item()

    # Standardized moments (stable across scales)
    z = xc / (std + 1e-12)
    m3 = (z ** 3).mean(dim=0)
    m4 = (z ** 4).mean(dim=0)

    avg_abs_skew = m3.abs().mean().item()
    avg_kurtosis = m4.mean().item()  # not excess (i.e., Gaussian ~3)

    outlier_rate_3std = (z.abs() > 3.0).float().mean().item()

    return {
        "frac_zeros": float(frac_zeros),
        "mean_abs": float(mean_abs),
        "mean_std": float(mean_std),
        "avg_abs_skew": float(avg_abs_skew),
        "avg_kurtosis": float(avg_kurtosis),
        "outlier_rate_3std": float(outlier_rate_3std),
        "shape_rows_used": int(Xs.shape[0]),
        "shape_dims_used": int(Xs.shape[1]),
    }


def compute_dataset_stats(
    dataset_name: str,
    device: str = "cpu",
    max_rows: int = 50000,
    max_dims: int = 200,
    seed: int = 0,
):
    """
    Loads dataset via prepare_db and returns one flat dict of stats.
    """
    # Minimal config — add flags here only if prepare_db requires them
    config = {"dataset_name": dataset_name, "device": device, "random_seed":42}
    data = prepare_db(config, name=dataset_name)

    X_train = data["X_train"].float()
    y_train = data["y_train"].long()

    input_dim = int(data.get("input_dim", X_train.shape[1]))
    num_classes = int(data.get("num_classes", int(y_train.max().item()) + 1))

    n_train = int(X_train.shape[0])
    n_val = int(data["X_val"].shape[0]) if "X_val" in data else 0
    n_test = int(data["X_test"].shape[0]) if "X_test" in data else 0

    # Size/dim derived
    stats = {
        "dataset": dataset_name,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_total": int(n_train + n_val + n_test),
        "d": int(input_dim),
        "num_classes": int(num_classes),
        "n_over_d": float(n_train / max(input_dim, 1)),
        "n_per_class_avg": float(n_train / max(num_classes, 1)),
    }

    # Class imbalance
    stats.update(class_stats(y_train, num_classes))

    # Feature distribution proxies (subsampled)
    stats.update(feature_shape_stats(X_train, max_rows=max_rows, max_dims=max_dims, seed=seed))

    return stats


def save_all_dataset_stats_csv(
    out_csv: str,
    device: str = "cpu",
    max_rows: int = 50000,
    max_dims: int = 200,
    seed: int = 0,
):
    ensure_dir(os.path.dirname(out_csv) or ".")

    db_list = list(DATASET_REGISTRY.keys())
    rows = []
    for db in db_list:
        try:
            s = compute_dataset_stats(
                db,
                device=device,
                max_rows=max_rows,
                max_dims=max_dims,
                seed=seed,
            )
            rows.append(s)
            print(f"[OK] {db} | n_train={s['n_train']} d={s['d']} C={s['num_classes']}")
        except Exception as e:
            print(f"[SKIP] {db} failed: {repr(e)}")

    if not rows:
        raise RuntimeError("No datasets processed successfully.")

    # Stable column order: union of keys, with a preferred front block
    preferred = [
        "dataset",
        "n_train", "n_val", "n_test", "n_total",
        "d", "num_classes", "n_over_d", "n_per_class_avg",
        "min_class_count", "max_class_count", "max_min_ratio",
        "majority_frac", "minority_frac",
        "class_entropy", "class_entropy_norm", "gini_impurity", "effective_num_classes", "min_over_mean_count",
        "frac_zeros", "mean_abs", "mean_std",
        "avg_abs_skew", "avg_kurtosis", "outlier_rate_3std",
        "shape_rows_used", "shape_dims_used",
    ]
    all_keys = sorted({k for r in rows for k in r.keys()})
    fieldnames = preferred + [k for k in all_keys if k not in preferred]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSaved dataset stats CSV to: {out_csv}")


if __name__ == "__main__":
    # Output location
    OUT_CSV = os.path.join("analysis_outputs", "dataset_stats.csv")

    # For stats-only computation, CPU is fine and usually safer.
    # If prepare_db needs GPU tensors for some reason, switch to "cuda".
    save_all_dataset_stats_csv(
        out_csv=OUT_CSV,
        device="cpu",
        max_rows=50000,   # reduce if slow
        max_dims=200,     # reduce if slow
        seed=0,
    )
