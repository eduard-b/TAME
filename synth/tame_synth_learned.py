"""
TAME with a *learned* embedder — ablation experiment.

Instead of sampling a fresh random frozen embedder each iteration,
we first pre-train an embedder on the real data (supervised classification),
strip the classification head, freeze the feature backbone, and then run
the same moment-matching distillation in that learned space.

This isolates one variable: is the random embedder the bottleneck,
or is moment matching itself the ceiling?

Supports the same moment orders as tame_synth_orders:
  order 1: mean
  order 2: mean + variance (per-dim)
  order 3: mean + variance + skewness
  order 4: mean + variance + skewness + kurtosis
  order 5: mean + full covariance (Frobenius)

Register as "tame_learned" in synth/registry.py.
Config keys:
  dm_moment_order  : 1-5  (default 5)
  dm_pretrain_epochs: how many epochs to train the embedder (default 50)
  dm_pretrain_lr    : learning rate for pre-training (default 1e-3)
  (all other dm_* keys same as base TAME)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from models.embedders import build_embedder, SIZE_LADDERS, _round
from .tame_synth import cov_matrix
from .tame_synth_orders import _moments_up_to, _moment_loss


# ------------------------------------------------------------------
# Step 1: build an embedder + classification head, train on real data
# ------------------------------------------------------------------

def _build_embedder_for_pretraining(embedder_type, embedder_size, input_dim,
                                     hidden, embed_dim, device):
    """
    Build the same embedder architecture used in base TAME,
    but WITHOUT freezing — we want to train it.
    """
    name = embedder_type.lower()
    size = embedder_size.lower()

    if name not in SIZE_LADDERS:
        raise ValueError(f"Unknown embedder_type={name}")
    if size not in SIZE_LADDERS[name]:
        raise ValueError(f"Unknown size={size} for {name}")

    ladder = dict(SIZE_LADDERS[name][size])

    if name == "ln_res_l":
        h = _round(hidden * ladder["hidden_mul"])
        kwargs = dict(
            input_dim=input_dim, hidden=h, embed_dim=embed_dim,
            depth=ladder["depth"], expansion=ladder["expansion"], dropout=0.0,
        )
    elif name == "dcnv2_base":
        h = _round(hidden * ladder["hidden_mul"])
        kwargs = dict(input_dim=input_dim, hidden=h, embed_dim=embed_dim)
    elif name == "node":
        tree_dim = _round((embed_dim // 2) * ladder["tree_dim_mul"],
                          multiple=8, min_val=16)
        kwargs = dict(
            input_dim=input_dim, hidden=hidden, embed_dim=embed_dim,
            num_layers=ladder["num_layers"], num_trees=ladder["num_trees"],
            depth=ladder["depth"], tree_dim=tree_dim, dropout=0.0,
        )
    else:
        kwargs = dict(input_dim=input_dim, hidden=hidden, embed_dim=embed_dim)

    net = build_embedder(name, **kwargs).to(device)
    return net


class _EmbedderWithHead(nn.Module):
    """Wraps an embedder backbone + a linear classification head."""
    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        out = 1 if num_classes == 2 else num_classes
        self.head = nn.Linear(embed_dim, out)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)


def _pretrain_embedder(backbone, embed_dim, X_train, y_train, num_classes,
                       device, epochs=50, lr=1e-3, batch_size=256):
    """
    Train backbone + head on real data. Return the trained backbone (no head).
    """
    model = _EmbedderWithHead(backbone, embed_dim, num_classes).to(device)

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            if num_classes == 2:
                loss = criterion(logits.view(-1), yb.float())
            else:
                loss = criterion(logits, yb.long())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  [pretrain] epoch {ep:3d}/{epochs} | "
                  f"loss {total_loss / n_batches:.4f}")

    # strip head, freeze backbone
    trained_backbone = model.backbone
    for p in trained_backbone.parameters():
        p.requires_grad_(False)
    trained_backbone.eval()

    return trained_backbone


# ------------------------------------------------------------------
# Step 2: run moment-matching distillation with the learned embedder
# ------------------------------------------------------------------

def tame_learned_synthesize(data, config):
    device = config["device"]
    X_train = data["X_train"].to(device).float()
    y_train = data["y_train"].to(device).long()

    ipc = int(config["ipc"])
    iters = int(config["dm_iters"])
    lr = float(config["dm_lr"])
    batch_real = int(config["dm_batch_real"])
    input_dim = int(data["input_dim"])
    num_classes = int(data["num_classes"])

    embed_hidden = int(config["dm_embed_hidden"])
    embed_dim = int(config["dm_embed_dim"])
    embedder_type = config["dm_embedder_type"]
    embedder_size = config.get("dm_embedder_size", "base")

    moment_order = max(1, min(5, int(config.get("dm_moment_order", 5))))
    grad_clip = float(config.get("grad_clip", 10.0))
    eps = float(config.get("moment_eps", 1e-6))
    cov_weight = float(config.get("cov_weight", 1.0))
    save_dir = config.get("save_dir", None)

    pretrain_epochs = int(config.get("dm_pretrain_epochs", 50))
    pretrain_lr = float(config.get("dm_pretrain_lr", 1e-3))

    # ---- pre-train the embedder on real data ----
    print(f"[TAME-learned] Pre-training {embedder_type} embedder "
          f"for {pretrain_epochs} epochs ...")

    backbone = _build_embedder_for_pretraining(
        embedder_type, embedder_size, input_dim,
        embed_hidden, embed_dim, device,
    )
    learned_embed = _pretrain_embedder(
        backbone, embed_dim, X_train, y_train, num_classes,
        device, epochs=pretrain_epochs, lr=pretrain_lr,
    )
    print(f"[TAME-learned] Embedder pre-training done.")

    # ---- standard TAME distillation, but using the single learned embedder ----
    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    def get_real_batch(c, n):
        idx = indices_class[c]
        return X_train[np.random.choice(idx, n, replace=len(idx) < n)]

    syn_data = torch.randn((num_classes * ipc, input_dim),
                           device=device, requires_grad=True)
    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc)

    with torch.no_grad():
        for c in range(num_classes):
            syn_data[c * ipc:(c + 1) * ipc] = get_real_batch(c, ipc)

    optimizer = torch.optim.SGD([syn_data], lr=lr, momentum=0.5)

    best_loss = float("inf")
    best_it = -1
    best_syn = syn_data.detach().clone()

    for it in range(iters + 1):
        # KEY DIFFERENCE: we reuse the same learned embedder every iteration
        # (no sampling, no randomness in the projection)
        optimizer.zero_grad(set_to_none=True)

        loss_total = torch.zeros((), device=device)

        for c in range(num_classes):
            real_b = get_real_batch(c, batch_real)
            syn_b = syn_data[c * ipc:(c + 1) * ipc]

            feat_real = learned_embed(real_b).detach()
            feat_syn = learned_embed(syn_b)

            if moment_order == 5:
                mu_r, cov_r = cov_matrix(feat_real, eps)
                mu_s, cov_s = cov_matrix(feat_syn, eps)
                loss_total = loss_total + ((mu_r - mu_s) ** 2).sum()
                diff = cov_r - cov_s
                loss_total = loss_total + cov_weight * (diff * diff).sum()
            else:
                m_real = _moments_up_to(feat_real, moment_order, eps)
                m_syn = _moments_up_to(feat_syn, moment_order, eps)
                loss_total = loss_total + _moment_loss(m_real, m_syn,
                                                       moment_order)

        cur = float((loss_total / num_classes).detach().item())
        if np.isfinite(cur) and cur < best_loss:
            best_loss = cur
            best_it = it
            best_syn = syn_data.detach().clone()

        if torch.isfinite(loss_total):
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_([syn_data], grad_clip)
            optimizer.step()

        if it % 100 == 0:
            tag = (f"Order{moment_order}" if moment_order != 5
                   else "Mean+FullCov")
            print(f"[TAME-learned-{tag}] iter {it:04d} | "
                  f"loss {cur:.6f} | best {best_loss:.6f}@{best_it:04d}")

    # ---- save ----
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = (f"order{moment_order}" if moment_order != 5
                  else "mean_fullcov")
        torch.save(
            {"X_syn": best_syn.cpu(), "y_syn": label_syn.cpu(),
             "best_loss": best_loss, "best_it": best_it,
             "dm_moment_order": moment_order,
             "embedder_mode": "learned"},
            os.path.join(save_dir, f"best_syn_learned_{suffix}.pt"),
        )

    return best_syn, label_syn.detach()
