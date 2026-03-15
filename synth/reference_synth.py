"""
Reference / baseline synthesizers: random, k-means (VQ), voronoi, gonzalez, full.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from utils.utils import set_seed


def full_synthesize(data, config):
    """Return the entire training set. For uniform eval pipelines."""
    return data["X_train"], data["y_train"]


def random_ipc_synthesize(data, config):
    set_seed(config["random_seed"])
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    device = config["device"]
    ipc = config["ipc"]

    X_sel, y_sel = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        chosen = np.random.choice(idx, ipc, replace=len(idx) < ipc)
        X_sel.append(X[chosen])
        y_sel.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_sel), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_sel), device=device, dtype=torch.long),
    )


def vq_synthesize(data, config):
    set_seed(config["random_seed"])
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    device = config["device"]
    ipc = config["ipc"]

    X_out, y_out = [], []
    for cls in np.unique(y):
        Xc = X[y == cls]
        if len(Xc) <= ipc:
            X_out.append(Xc)
            y_out.append(np.full(len(Xc), cls))
            continue
        km = KMeans(n_clusters=ipc, random_state=config["random_seed"])
        km.fit(Xc)
        X_out.append(km.cluster_centers_)
        y_out.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_out), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_out), device=device, dtype=torch.long),
    )


def voronoi_synthesize(data, config):
    set_seed(config["random_seed"])
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    device = config["device"]
    ipc = config["ipc"]

    X_out, y_out = [], []
    for cls in np.unique(y):
        Xc = X[y == cls]
        if len(Xc) <= ipc:
            X_out.append(Xc)
            y_out.append(np.full(len(Xc), cls))
            continue
        km = KMeans(n_clusters=ipc, random_state=config["random_seed"])
        labels = km.fit_predict(Xc)
        centers = km.cluster_centers_
        for k in range(ipc):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue
            dists = np.linalg.norm(Xc[idx] - centers[k], axis=1)
            chosen = idx[np.argmin(dists)]
            X_out.append(Xc[chosen:chosen + 1])
            y_out.append(np.array([cls]))

    return (
        torch.tensor(np.vstack(X_out), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_out), device=device, dtype=torch.long),
    )


def gonzalez_synthesize(data, config):
    set_seed(config["random_seed"])
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    device = config["device"]
    ipc = config["ipc"]

    X_out, y_out = [], []
    for cls in np.unique(y):
        Xc = X[y == cls]
        n = len(Xc)
        if n <= ipc:
            X_out.append(Xc)
            y_out.append(np.full(n, cls))
            continue
        sel = [np.random.randint(0, n)]
        dist = np.full(n, np.inf)
        for _ in range(ipc - 1):
            dist = np.minimum(dist, np.linalg.norm(Xc - Xc[sel[-1]], axis=1))
            sel.append(np.argmax(dist))
        X_out.append(Xc[sel])
        y_out.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_out), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_out), device=device, dtype=torch.long),
    )
