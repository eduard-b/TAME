import os
import json
import csv
import random
import numpy as np
import torch

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import umap

# ---- your project imports ----
from utils.utils import ensure_dir
from data.prepare_database import prepare_db
from models.embedders import sample_random_embedder  # must accept embedder_size (or default)
# If your distillation function lives elsewhere, import it:
# from synth.dm_moments_cov2 import dm_moment_synthesize_cov2
from synth.registry import synthesize  # if you want to call through registry


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Helpers: init syn0
# =========================
@torch.no_grad()
def make_syn_init_from_real(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    ipc: int,
    num_classes: int,
    device: str,
    seed: int = 0,
):
    """
    Mimics your dm init: per-class samples from real DB.
    Returns:
      syn0: (C*ipc, D) float tensor on device
      y0:   (C*ipc,) long tensor on device
    """
    set_seed(seed)
    X_train = X_train.to(device).float()
    y_train = y_train.to(device).long()

    y_np = y_train.detach().cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    def get_real_batch(c, n):
        idx = indices_class[c]
        idx_sel = np.random.choice(idx, n, replace=len(idx) < n)
        return X_train[idx_sel]

    input_dim = X_train.shape[1]
    syn0 = torch.empty((num_classes * ipc, input_dim), device=device, dtype=torch.float32)
    y0 = torch.arange(num_classes, device=device).repeat_interleave(ipc)

    for c in range(num_classes):
        syn0[c * ipc : (c + 1) * ipc] = get_real_batch(c, ipc)

    return syn0, y0


# =========================
# Helpers: UMAP pipeline
# =========================
def fit_umap_and_transform(
    X_real: np.ndarray,
    X_a: np.ndarray,
    X_b: np.ndarray,
    *,
    pca_dim: int = 50,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    metric: str = "euclidean",
    seed: int = 42,
):
    """
    Fits scaler + PCA + UMAP ON REAL ONLY, transforms A and B into same 2D space.
    Returns:
      Z_real, Z_a, Z_b
      plus fitted objects (scaler, pca, umap) for reproducibility/plotting
    """
    scaler = StandardScaler()
    Xr = scaler.fit_transform(X_real)
    Xa = scaler.transform(X_a)
    Xb = scaler.transform(X_b)

    # PCA for stability/speed (especially helpful for embedder space, still fine for raw)
    pca_dim = min(pca_dim, Xr.shape[1])
    pca = PCA(n_components=pca_dim, random_state=seed)
    Xr_p = pca.fit_transform(Xr)
    Xa_p = pca.transform(Xa)
    Xb_p = pca.transform(Xb)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )

    Zr = reducer.fit_transform(Xr_p)
    Za = reducer.transform(Xa_p)
    Zb = reducer.transform(Xb_p)

    return (Zr, Za, Zb), dict(scaler=scaler, pca=pca, umap=reducer)


def save_umap_artifact_pt(path: str, payload: dict):
    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)


def save_umap_artifact_csv(path: str, Z: np.ndarray, y: np.ndarray, source: str):
    """
    Simple CSV: x, y, label, source
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "label", "source"])
        for i in range(Z.shape[0]):
            w.writerow([float(Z[i, 0]), float(Z[i, 1]), int(y[i]), source])


# =========================
# Helpers: embedder features
# =========================
@torch.no_grad()
def embed_features(
    X: torch.Tensor,
    *,
    embedder_type: str,
    embedder_size: str,
    input_dim: int,
    hidden: int,
    embed_dim: int,
    device: str,
    batch_size: int = 2048,
):
    """
    Computes embedder(X) in batches and returns numpy (N, embed_dim).
    Embedder is frozen & eval by sample_random_embedder.
    """
    net = sample_random_embedder(
        embedder_type=embedder_type,
        embedder_size=embedder_size,
        input_dim=input_dim,
        hidden=hidden,
        embed_dim=embed_dim,
        device=device,
    )

    X = X.to(device).float()
    outs = []

    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size]
        outs.append(net(xb).detach().cpu())

    return torch.cat(outs, dim=0).numpy()


# =========================
# Main experiment
# =========================
def run_magic_umap_experiment(config: dict):
    """
    Artifacts produced in config["save_dir"]:
      - syn0.pt, synT.pt
      - umap_raw.pt (and optional CSVs)
      - umap_embed.pt (and optional CSVs)
      - config.json
    """
    ensure_dir(config["save_dir"])
    with open(os.path.join(config["save_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config.get("random_seed", 42))
    set_seed(seed)

    # ---------------- Load MAGIC ----------------
    data = prepare_db(config, name="magic")
    X_train = data["X_train"]
    y_train = data["y_train"]
    input_dim = int(data["input_dim"])
    num_classes = int(data["num_classes"])

    # Optionally use a stratified subset of real for fitting UMAP
    # (recommended so real doesn't dominate density and so it’s fast)
    real_umap_n = int(config.get("real_umap_n", 8000))
    y_np = y_train.cpu().numpy()
    if real_umap_n < len(y_np):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=real_umap_n, random_state=seed)
        idx, _ = next(sss.split(np.zeros_like(y_np), y_np))
        X_real_umap = X_train[idx]
        y_real_umap = y_train[idx]
    else:
        X_real_umap = X_train
        y_real_umap = y_train

    # ---------------- Create syn0 (pre-distill) ----------------
    ipc = int(config["ipc"])
    syn0, y0 = make_syn_init_from_real(
        X_train=X_train,
        y_train=y_train,
        ipc=ipc,
        num_classes=num_classes,
        device=device,
        seed=seed,
    )

    torch.save({"X_syn0": syn0.detach().cpu(), "y_syn0": y0.detach().cpu()},
               os.path.join(config["save_dir"], "syn0.pt"))

    # ---------------- Distill to synT ----------------
    # You can either:
    # A) Call your function directly (if imported), or
    # B) Go through your registry synthesize() using config["synth_type"].
    #
    # Here: go through synthesize() to match your main pipeline style.
# ---------------- syn0 (pre-distill) ----------------
    syn0_path = config.get("syn0_path", os.path.join(config["save_dir"], "syn0.pt"))

    if os.path.exists(syn0_path):
        pack0 = torch.load(syn0_path, map_location="cpu")
        syn0 = pack0["X_syn0"].to(device).float()
        y0   = pack0["y_syn0"].to(device).long()
    else:
        ipc = int(config["ipc"])
        syn0, y0 = make_syn_init_from_real(
            X_train=X_train,
            y_train=y_train,
            ipc=ipc,
            num_classes=num_classes,
            device=device,
            seed=seed,
        )
        torch.save({"X_syn0": syn0.detach().cpu(), "y_syn0": y0.detach().cpu()}, syn0_path)

    # ---------------- synT (distilled) ----------------
    use_precomputed = bool(config.get("use_precomputed_synT", False))
    synT_path = config.get("synT_path", os.path.join(config["save_dir"], "synT_best.pt"))

    if use_precomputed:
        if not os.path.exists(synT_path):
            raise FileNotFoundError(f"synT_path not found: {synT_path}")
        packT = torch.load(synT_path, map_location="cpu")

        # Accept either naming convention: X_synT/y_synT or X/y or syn_data/label_syn
        if "X_synT" in packT and "y_synT" in packT:
            X_synT = packT["X_synT"]
            y_synT = packT["y_synT"]

        elif "X_syn" in packT and "y_syn" in packT:   # ✅ your file
            X_synT = packT["X_syn"]
            y_synT = packT["y_syn"]

        elif "X" in packT and "y" in packT:
            X_synT = packT["X"]
            y_synT = packT["y"]

        elif "syn_data" in packT and "label_syn" in packT:
            X_synT = packT["syn_data"]
            y_synT = packT["label_syn"]

        else:
            raise KeyError(f"Unknown keys in {synT_path}: {list(packT.keys())}")


        X_synT = X_synT.to(device).float()
        y_synT = y_synT.to(device).long()

    else:
        X_synT, y_synT = synthesize(
            synth_type=config["synth_type"],
            data=data,
            config=config,
        )
        torch.save({"X_synT": X_synT.detach().cpu(), "y_synT": y_synT.detach().cpu()},
                os.path.join(config["save_dir"], "synT.pt"))


    # Make sure we’re using the same labels as syn0 (should be identical if ipc & classes match)
    X_syn0_cpu = syn0.detach().cpu()
    y_syn0_cpu = y0.detach().cpu()
    X_synT_cpu = X_synT.detach().cpu()
    y_synT_cpu = y_synT.detach().cpu()

    # ---------------- UMAP: RAW space ----------------
    Zs = {}
    (Z_real, Z_0, Z_T), fitted_raw = fit_umap_and_transform(
        X_real=X_real_umap.cpu().numpy(),
        X_a=X_syn0_cpu.numpy(),
        X_b=X_synT_cpu.numpy(),
        pca_dim=int(config.get("umap_pca_dim_raw", 10)),   # MAGIC raw is 10D; keep small
        n_neighbors=int(config.get("umap_n_neighbors", 30)),
        min_dist=float(config.get("umap_min_dist", 0.05)),
        metric=str(config.get("umap_metric", "euclidean")),
        seed=seed,
    )

    raw_payload = {
        "Z_real": torch.from_numpy(Z_real).float(),
        "y_real": y_real_umap.cpu().long(),
        "Z_syn0": torch.from_numpy(Z_0).float(),
        "y_syn0": y_syn0_cpu.long(),
        "Z_synT": torch.from_numpy(Z_T).float(),
        "y_synT": y_synT_cpu.long(),
        "meta": {
            "space": "raw",
            "pca_dim": int(config.get("umap_pca_dim_raw", 10)),
            "umap": {
                "n_neighbors": int(config.get("umap_n_neighbors", 30)),
                "min_dist": float(config.get("umap_min_dist", 0.05)),
                "metric": str(config.get("umap_metric", "euclidean")),
                "seed": seed,
            },
        },
    }

    save_umap_artifact_pt(os.path.join(config["save_dir"], "umap_raw.pt"), raw_payload)

    if bool(config.get("save_csv", False)):
        save_umap_artifact_csv(os.path.join(config["save_dir"], "umap_raw_real.csv"), Z_real, y_real_umap.cpu().numpy(), "real")
        save_umap_artifact_csv(os.path.join(config["save_dir"], "umap_raw_syn0.csv"), Z_0, y_syn0_cpu.numpy(), "syn0")
        save_umap_artifact_csv(os.path.join(config["save_dir"], "umap_raw_synT.csv"), Z_T, y_synT_cpu.numpy(), "synT")

    # ---------------- UMAP: EMBEDDER space ----------------
    emb_type = config["dm_embedder_type"]
    emb_size = config.get("dm_embedder_size", "base")
    emb_hidden = int(config["dm_embed_hidden"])
    emb_dim = int(config["dm_embed_dim"])

    # Compute embeddings for each set (fit UMAP on real embeddings only)
    Xr_feat = embed_features(
        X_real_umap,
        embedder_type=emb_type,
        embedder_size=emb_size,
        input_dim=input_dim,
        hidden=emb_hidden,
        embed_dim=emb_dim,
        device=device,
        batch_size=int(config.get("embed_batch_size", 4096)),
    )
    X0_feat = embed_features(
        X_syn0_cpu,
        embedder_type=emb_type,
        embedder_size=emb_size,
        input_dim=input_dim,
        hidden=emb_hidden,
        embed_dim=emb_dim,
        device=device,
        batch_size=int(config.get("embed_batch_size", 4096)),
    )
    XT_feat = embed_features(
        X_synT_cpu,
        embedder_type=emb_type,
        embedder_size=emb_size,
        input_dim=input_dim,
        hidden=emb_hidden,
        embed_dim=emb_dim,
        device=device,
        batch_size=int(config.get("embed_batch_size", 4096)),
    )

    (Zr_e, Z0_e, ZT_e), fitted_emb = fit_umap_and_transform(
        X_real=Xr_feat,
        X_a=X0_feat,
        X_b=XT_feat,
        pca_dim=int(config.get("umap_pca_dim_embed", 50)),
        n_neighbors=int(config.get("umap_n_neighbors", 30)),
        min_dist=float(config.get("umap_min_dist", 0.05)),
        metric=str(config.get("umap_metric", "euclidean")),
        seed=seed,
    )

    emb_payload = {
        "Z_real": torch.from_numpy(Zr_e).float(),
        "y_real": y_real_umap.cpu().long(),
        "Z_syn0": torch.from_numpy(Z0_e).float(),
        "y_syn0": y_syn0_cpu.long(),
        "Z_synT": torch.from_numpy(ZT_e).float(),
        "y_synT": y_synT_cpu.long(),
        "meta": {
            "space": "embedder",
            "embedder": {
                "type": emb_type,
                "size": emb_size,
                "hidden": emb_hidden,
                "embed_dim": emb_dim,
            },
            "pca_dim": int(config.get("umap_pca_dim_embed", 50)),
            "umap": {
                "n_neighbors": int(config.get("umap_n_neighbors", 30)),
                "min_dist": float(config.get("umap_min_dist", 0.05)),
                "metric": str(config.get("umap_metric", "euclidean")),
                "seed": seed,
            },
        },
    }

    torch.save(emb_payload, os.path.join(config["save_dir"], "umap_embed.pt"))


    save_umap_artifact_pt(os.path.join(config["save_dir"], "umap_embed.pt"), emb_payload)

    if bool(config.get("save_csv", False)):
        save_umap_artifact_csv(os.path.join(config["save_dir"], "umap_embed_real.csv"), Zr_e, y_real_umap.cpu().numpy(), "real")
        save_umap_artifact_csv(os.path.join(config["save_dir"], "umap_embed_syn0.csv"), Z0_e, y_syn0_cpu.numpy(), "syn0")
        save_umap_artifact_csv(os.path.join(config["save_dir"], "umap_embed_synT.csv"), ZT_e, y_synT_cpu.numpy(), "synT")

    print(f"[DONE] saved artifacts to: {config['save_dir']}")


if __name__ == "__main__":
    # Example config for ONE run (you can wrap this in your sweeps)
    config = {
        "dataset_name": "magic",
        "save_dir": "stageU/magic_umap_demo",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": 42,

        "ipc": 50,

        # Don't distill; load best
        "use_precomputed_synT": True,
        "synT_path": "stageU/magic_umap_demo/synT_best.pt",
        "syn0_path": "stageU/magic_umap_demo/syn0.pt",

        # Embedder (only used for embed-space UMAP now)
        "dm_embedder_type": "ln_res_l",
        "dm_embedder_size": "base",
        "dm_embed_hidden": 256,
        "dm_embed_dim": 48,

        # UMAP knobs
        "real_umap_n": 8000,
        "umap_n_neighbors": 30,
        "umap_min_dist": 0.05,
        "umap_metric": "euclidean",
        "umap_pca_dim_raw": 10,
        "umap_pca_dim_embed": 48,
        "embed_batch_size": 4096,

        "save_csv": False,
    }


    run_magic_umap_experiment(config)
