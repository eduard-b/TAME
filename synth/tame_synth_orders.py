"""
TAME order study: moment matching up to order k in frozen embedder space.

  order 1: mean
  order 2: mean + variance (per-dim)
  order 3: mean + variance + skewness
  order 4: mean + variance + skewness + kurtosis
  order 5: mean + full covariance (Frobenius)

Register as e.g. "tame_orders" in synth/registry.py.
Set config["dm_moment_order"] to select.
"""

import os
import torch
import numpy as np
from models.embedders import sample_random_embedder
from .tame_synth import cov_matrix


def _moments_up_to(z, order, eps=1e-6):
    mu = z.mean(0)
    out = {"m1": mu}
    if order >= 2:
        xc = z - mu
        m2 = (xc * xc).mean(0)
        out["m2"] = m2
    if order >= 3:
        xc = z - mu
        std = torch.sqrt(out["m2"] + eps)
        out["m3"] = (xc ** 3).mean(0) / (std ** 3 + eps)
    if order >= 4:
        xc = z - mu
        std = torch.sqrt(out["m2"] + eps)
        out["m4"] = (xc ** 4).mean(0) / (std ** 4 + eps)
    return out


def _moment_loss(m_real, m_syn, order):
    loss = ((m_real["m1"] - m_syn["m1"]) ** 2).sum()
    if order >= 2:
        loss = loss + ((m_real["m2"] - m_syn["m2"]) ** 2).sum()
    if order >= 3:
        loss = loss + ((m_real["m3"] - m_syn["m3"]) ** 2).sum()
    if order >= 4:
        loss = loss + ((m_real["m4"] - m_syn["m4"]) ** 2).sum()
    return loss


def tame_orders_synthesize(data, config):
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

    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    def get_real_batch(c, n):
        idx = indices_class[c]
        return X_train[np.random.choice(idx, n, replace=len(idx) < n)]

    syn_data = torch.randn((num_classes * ipc, input_dim), device=device, requires_grad=True)
    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc)

    with torch.no_grad():
        for c in range(num_classes):
            syn_data[c * ipc:(c + 1) * ipc] = get_real_batch(c, ipc)

    optimizer = torch.optim.SGD([syn_data], lr=lr, momentum=0.5)

    best_loss = float("inf")
    best_it = -1
    best_syn = syn_data.detach().clone()

    for it in range(iters + 1):
        embed_net = sample_random_embedder(
            embedder_type, embedder_size, input_dim, embed_hidden, embed_dim, device
        )
        embed_net.eval()
        optimizer.zero_grad(set_to_none=True)

        loss_total = torch.zeros((), device=device)

        for c in range(num_classes):
            real_b = get_real_batch(c, batch_real)
            syn_b = syn_data[c * ipc:(c + 1) * ipc]

            feat_real = embed_net(real_b).detach()
            feat_syn = embed_net(syn_b)

            if moment_order == 5:
                mu_r, cov_r = cov_matrix(feat_real, eps)
                mu_s, cov_s = cov_matrix(feat_syn, eps)
                loss_total = loss_total + ((mu_r - mu_s) ** 2).sum()
                diff = cov_r - cov_s
                loss_total = loss_total + cov_weight * (diff * diff).sum()
            else:
                m_real = _moments_up_to(feat_real, moment_order, eps)
                m_syn = _moments_up_to(feat_syn, moment_order, eps)
                loss_total = loss_total + _moment_loss(m_real, m_syn, moment_order)

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
            tag = f"Order{moment_order}" if moment_order != 5 else "Mean+FullCov"
            print(f"[TAME-{tag}] iter {it:04d} | loss {cur:.6f} | best {best_loss:.6f}@{best_it:04d}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = f"order{moment_order}" if moment_order != 5 else "mean_fullcov"
        torch.save(
            {"X_syn": best_syn.cpu(), "y_syn": label_syn.cpu(),
             "best_loss": best_loss, "best_it": best_it,
             "dm_moment_order": moment_order},
            os.path.join(save_dir, f"best_syn_{suffix}.pt"),
        )

    return best_syn, label_syn.detach()
