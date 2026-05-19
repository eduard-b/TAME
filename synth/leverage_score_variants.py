"""
Leverage-score variants — adaptations of PCA-norm leverage sampling to the
class-balanced IPC setting.

All four use the same leverage formulation as the existing leverage_score_synth.py
(PCA projection norms, ℓ_i = ‖z_i‖² where z_i = (x_i - mean) @ V_k) so the
comparison isolates the *adaptation strategy*, not the leverage formula itself.

Strategies:
  B   leverage_global_strat   : global PCA, IPC per class weighted by global leverage
  C   leverage_global_unrestr : global PCA, K*IPC global sample (closest to reference)
  D   leverage_aug            : PCA on [X, scale·one-hot(y)], K*IPC global sample
  Db  leverage_aug_balanced   : PCA on [X, scale·one-hot(y)], IPC per class (D ablation)
  G   leverage_topk           : global PCA, deterministic top-IPC per class (no rng)

All accept the same data dict and config as the other synth functions and
return (X_syn_tensor, y_syn_tensor) on config["device"].
"""

import numpy as np
import torch
from sklearn.decomposition import PCA

from utils.utils import set_seed


def _global_pca_leverage(X, n_components_max, seed):
    """Fit one PCA on full X, return leverage = ||z_i||^2 (PCA-norm style)."""
    n, d = X.shape
    n_components = min(int(n_components_max), d, n)
    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(X)
    return np.sum(Z ** 2, axis=1)


def _safe_probs(weights):
    """Normalize a non-negative weight vector to a probability vector;
    fall back to uniform on degenerate inputs."""
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def _global_sample_with_class_floor(y, leverage, budget, num_classes, rng):
    """Sample `budget` indices weighted by leverage, but guarantee at least
    one index per class so the downstream classifier sees every class.

    Reserves the highest-leverage point from each class, then draws the
    remaining `budget - num_classes` indices globally from the leftover pool
    weighted by leverage. Returns the index array of length `budget`.

    A note on purity: this is a small departure from the literal "sample N
    points purely by leverage" reference. We do it because XGBoost (and the
    eval pipeline's argmax-based prediction) require contiguous 0..K-1 labels
    in y_syn. If you want the pure version with no class floor, set
    config["lev_min_per_class"]=0 in the calling function.
    """
    n = len(y)
    classes = np.unique(y)
    K_present = len(classes)

    if budget <= K_present:
        # Degenerate: budget too small to fit one per class. Fall back to
        # picking the top-leverage point per class until budget is hit.
        order = np.argsort(-np.array([leverage[y == c].max() for c in classes]))
        chosen = []
        for c in classes[order][:budget]:
            pool = np.where(y == c)[0]
            chosen.append(pool[int(np.argmax(leverage[pool]))])
        return np.array(chosen, dtype=int)

    # Reserve highest-leverage point from each class
    reserved = np.empty(K_present, dtype=int)
    for i, c in enumerate(classes):
        pool = np.where(y == c)[0]
        reserved[i] = pool[int(np.argmax(leverage[pool]))]

    # Remaining budget drawn from the rest, weighted by leverage
    remaining_budget = budget - K_present
    mask = np.ones(n, dtype=bool)
    mask[reserved] = False
    rest_pool = np.where(mask)[0]

    if remaining_budget >= len(rest_pool):
        extra = rest_pool
    else:
        rest_probs = _safe_probs(leverage[rest_pool])
        extra = rng.choice(rest_pool, size=remaining_budget,
                           replace=False, p=rest_probs)

    return np.concatenate([reserved, extra])


# -----------------------------------------------------------------------------
# B — Global PCA, class-stratified IPC sampling
# -----------------------------------------------------------------------------
def leverage_global_strat_synthesize(data, config):
    """B: Global PCA, IPC-per-class sampling weighted by global leverage.

    One PCA on all training data → leverage scores reflect importance in the
    GLOBAL row space. Then for each class, draw IPC points from that class
    with probability ∝ each point's global leverage. Class-balanced output.
    """
    set_seed(config["random_seed"])
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    device = config["device"]
    ipc = int(config["ipc"])

    n_components_max = int(config.get("lev_pca_components", 20))
    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)

    leverage = _global_pca_leverage(X, n_components_max, seed)

    X_out, y_out = [], []
    for cls in np.unique(y):
        idx_cls = np.where(y == cls)[0]
        nc = len(idx_cls)
        if nc <= ipc:
            X_out.append(X[idx_cls])
            y_out.append(np.full(nc, cls))
            continue

        probs = _safe_probs(leverage[idx_cls])
        chosen = rng.choice(nc, size=ipc, replace=False, p=probs)
        X_out.append(X[idx_cls[chosen]])
        y_out.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_out), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_out), device=device, dtype=torch.long),
    )


# -----------------------------------------------------------------------------
# C — Global PCA, single global K*IPC sample
# -----------------------------------------------------------------------------
def leverage_global_unrestr_synthesize(data, config):
    """C: Global PCA, K*IPC global sampling. Closest to the reference paper.

    Sampling is leverage-weighted globally — class proportions reflect leverage
    rather than a fixed per-class budget. To keep the output usable by the eval
    pipeline (XGBoost requires contiguous 0..K-1 labels; argmax-based eval
    requires all classes seen during training), we guarantee at least one
    sample per class via _global_sample_with_class_floor. Set
    config["lev_min_per_class"]=0 to disable that floor and get the pure
    unrestricted version.
    """
    set_seed(config["random_seed"])
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    device = config["device"]
    ipc = int(config["ipc"])
    num_classes = int(data["num_classes"])

    n_components_max = int(config.get("lev_pca_components", 20))
    enforce_floor = bool(config.get("lev_min_per_class", 1))
    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)

    n = len(X)
    budget = num_classes * ipc

    leverage = _global_pca_leverage(X, n_components_max, seed)

    if budget >= n:
        idx = np.arange(n)
    elif enforce_floor:
        idx = _global_sample_with_class_floor(y, leverage, budget, num_classes, rng)
    else:
        probs = _safe_probs(leverage)
        idx = rng.choice(n, size=budget, replace=False, p=probs)
        classes_present = np.unique(y[idx])
        if len(classes_present) < num_classes:
            missing = sorted(set(np.unique(y).tolist()) - set(classes_present.tolist()))
            print(
                f"[leverage_global_unrestr] WARNING: classes {missing} missing "
                f"from sample. Classifier will train on "
                f"{len(classes_present)}/{num_classes} classes."
            )

    return (
        torch.tensor(X[idx], device=device, dtype=torch.float32),
        torch.tensor(y[idx], device=device, dtype=torch.long),
    )


# -----------------------------------------------------------------------------
# D — Augmented PCA on [X, scale·one-hot(y)] + global K*IPC sample
# -----------------------------------------------------------------------------
def leverage_aug_synthesize(data, config):
    """D: PCA on [X, scale·one-hot(y)], K*IPC global sampling.

    Labels become explicit columns of the matrix PCA sees, so the leverage
    scores are class-aware automatically — no per-class loop. Only the X
    portion is returned; the augmentation is purely for computing leverage.

    Config:
      lev_aug_label_scale : float
        Scale applied to the one-hot block before PCA. Defaults to
        sqrt(num_classes), which makes each label column have variance ≈ 1
        for class-balanced data — comparable to a z-scored feature column.
        Larger values make leverage more class-aware; smaller values make it
        behave closer to plain leverage on X.
    """
    set_seed(config["random_seed"])
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    device = config["device"]
    ipc = int(config["ipc"])
    num_classes = int(data["num_classes"])

    n_components_max = int(config.get("lev_pca_components", 20))
    label_scale = float(
        config.get("lev_aug_label_scale", float(np.sqrt(num_classes)))
    )
    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)

    n, d = X.shape
    budget = num_classes * ipc
    enforce_floor = bool(config.get("lev_min_per_class", 1))

    # Build augmented matrix [X | scale * one_hot(y)]
    Y_onehot = np.zeros((n, num_classes), dtype=X.dtype)
    Y_onehot[np.arange(n), y.astype(int)] = 1.0
    X_aug = np.concatenate([X, label_scale * Y_onehot], axis=1)

    leverage = _global_pca_leverage(X_aug, n_components_max, seed)

    if budget >= n:
        idx = np.arange(n)
    elif enforce_floor:
        idx = _global_sample_with_class_floor(y, leverage, budget, num_classes, rng)
    else:
        probs = _safe_probs(leverage)
        idx = rng.choice(n, size=budget, replace=False, p=probs)
        classes_present = np.unique(y[idx])
        if len(classes_present) < num_classes:
            missing = sorted(set(np.unique(y).tolist()) - set(classes_present.tolist()))
            print(
                f"[leverage_aug] WARNING: classes {missing} missing from sample "
                f"(label_scale={label_scale:.3f}). Consider increasing label_scale."
            )

    # Return only the original X (not the augmented columns) and the labels.
    return (
        torch.tensor(X[idx], device=device, dtype=torch.float32),
        torch.tensor(y[idx], device=device, dtype=torch.long),
    )


# -----------------------------------------------------------------------------
# D-balanced — Augmented PCA + class-balanced IPC sampling
# -----------------------------------------------------------------------------
def leverage_aug_balanced_synthesize(data, config):
    """D-balanced: Augmented PCA on [X, scale·one-hot(y)] + IPC per class.

    Same leverage formulation as leverage_aug (labels as explicit PCA columns)
    but with B-style class-balanced sampling instead of global. Designed to
    isolate the "augmented subspace" effect from the "class-skew" effect.

    Compared against B and the unbalanced D, this triangulates which factor
    drives D's accuracy:
        D ≈ D-balanced  → augmented subspace is doing the work
        D-balanced ≈ B  → D's lift was class-skew exploitation
    """
    set_seed(config["random_seed"])
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    device = config["device"]
    ipc = int(config["ipc"])
    num_classes = int(data["num_classes"])

    n_components_max = int(config.get("lev_pca_components", 20))
    label_scale = float(
        config.get("lev_aug_label_scale", float(np.sqrt(num_classes)))
    )
    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)

    n = len(X)

    # Build augmented matrix [X | scale * one_hot(y)]
    Y_onehot = np.zeros((n, num_classes), dtype=X.dtype)
    Y_onehot[np.arange(n), y.astype(int)] = 1.0
    X_aug = np.concatenate([X, label_scale * Y_onehot], axis=1)

    leverage = _global_pca_leverage(X_aug, n_components_max, seed)

    X_out, y_out = [], []
    for cls in np.unique(y):
        idx_cls = np.where(y == cls)[0]
        nc = len(idx_cls)
        if nc <= ipc:
            X_out.append(X[idx_cls])
            y_out.append(np.full(nc, cls))
            continue

        probs = _safe_probs(leverage[idx_cls])
        chosen = rng.choice(nc, size=ipc, replace=False, p=probs)
        X_out.append(X[idx_cls[chosen]])
        y_out.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_out), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_out), device=device, dtype=torch.long),
    )


# -----------------------------------------------------------------------------
# G — Global PCA, deterministic top-IPC by leverage per class
# -----------------------------------------------------------------------------
def leverage_topk_synthesize(data, config):
    """G: Global PCA, take top-IPC highest-leverage points per class.

    No random sampling — selection is deterministic given the PCA seed.
    Often stronger than B on small IPC budgets because it commits to the
    obvious "most informative" points instead of probabilistically missing
    them. Class-balanced output.
    """
    set_seed(config["random_seed"])
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    device = config["device"]
    ipc = int(config["ipc"])

    n_components_max = int(config.get("lev_pca_components", 20))
    seed = int(config["random_seed"])

    leverage = _global_pca_leverage(X, n_components_max, seed)

    X_out, y_out = [], []
    for cls in np.unique(y):
        idx_cls = np.where(y == cls)[0]
        nc = len(idx_cls)
        if nc <= ipc:
            X_out.append(X[idx_cls])
            y_out.append(np.full(nc, cls))
            continue

        lev_cls = leverage[idx_cls]
        # Top IPC by leverage (descending). argpartition is O(n) and we don't
        # need full sort.
        top_pos = np.argpartition(-lev_cls, ipc)[:ipc]
        chosen = idx_cls[top_pos]
        X_out.append(X[chosen])
        y_out.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_out), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_out), device=device, dtype=torch.long),
    )
