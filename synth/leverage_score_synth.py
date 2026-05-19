"""
Leverage-score coreset selection.

Adapted from:
  Tabular-Data-Distillation (github.com/... — see distill_coreset_leverage_scores)

Method
------
For each class, project the real samples to a low-dimensional subspace using
PCA, compute per-sample leverage scores as squared Euclidean norms in that
subspace, normalize to a probability distribution, and sample IPC points
(weighted by leverage, without replacement).

Leverage score ℓ_i = ||z_i||^2 where z_i is the PCA projection of x_i.
High-leverage points are those with large directional influence on the
row-space of the data — classical statistical importance sampling.

This is a real-sample instance-selection method (not synthetic generation),
analogous to Voronoi and Gonzalez in your reference_synth.py.

Config keys:
  ipc                   : samples per class
  random_seed           : seed
  lev_pca_components    : PCA components (default: min(20, d))
  lev_replace_if_needed : if a class has fewer samples than IPC, fill with
                          replacement (default: True; matches your other refs)
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from utils.utils import set_seed


def leverage_score_synthesize(data, config):
    set_seed(config["random_seed"])
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    device = config["device"]
    ipc = int(config["ipc"])

    n_components_max = int(config.get("lev_pca_components", 20))
    seed = int(config["random_seed"])

    rng = np.random.default_rng(seed)

    X_out, y_out = [], []

    for cls in np.unique(y):
        Xc = X[y == cls]
        nc, d = Xc.shape

        # if class has fewer samples than IPC budget, keep them all
        if nc <= ipc:
            X_out.append(Xc)
            y_out.append(np.full(nc, cls))
            continue

        # PCA projection
        n_components = min(n_components_max, d, nc)
        pca = PCA(n_components=n_components, random_state=seed)
        Z = pca.fit_transform(Xc)

        # leverage scores = squared norms in projected space
        leverage = np.sum(Z ** 2, axis=1)

        # numerical safety: avoid degenerate zero-sum
        total = leverage.sum()
        if total <= 0 or not np.isfinite(total):
            # fall back to uniform sampling for this class
            probs = np.full(nc, 1.0 / nc)
        else:
            probs = leverage / total

        # weighted sample without replacement
        idx = rng.choice(nc, size=ipc, replace=False, p=probs)
        X_out.append(Xc[idx])
        y_out.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_out), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_out), device=device, dtype=torch.long),
    )
