"""
TAME + mixed embedders: mean+cov2 distribution matching with embedder
drawn from a pool (random each iter) or fused (concat + compress).

Register as "tame_fusion" in synth/registry.py.

Config keys (beyond base TAME):
  dm_embedder_mode: "single" | "pool_random" | "fusion"
  dm_embedder_pool: list of embedder names (for pool_random / fusion)
  dm_fusion_per_dim: output dim per sub-embedder before concat
"""

import os
import torch
import torch.nn as nn
import numpy as np
from models.embedders import sample_random_embedder
from .tame_synth import cov_matrix


class FusionEmbedder(nn.Module):
    def __init__(self, embedders, per_dims, out_dim):
        super().__init__()
        self.embedders = nn.ModuleList(embedders)
        in_dim = sum(per_dims)
        self.compress = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    @torch.no_grad()
    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.compress(torch.cat([e(x) for e in self.embedders], dim=1))


def _sample_embedder(config, input_dim, embed_hidden, embed_dim, device):
    mode = str(config.get("dm_embedder_mode", "single")).lower()
    embedder_size = config.get("dm_embedder_size", "base")

    if mode == "single":
        name = config["dm_embedder_type"]
        net = sample_random_embedder(name, embedder_size, input_dim, embed_hidden, embed_dim, device)
        return net, f"single:{name}"

    pool = config.get("dm_embedder_pool", ["ln_res_l", "node", "dcnv2_base"])

    if mode == "pool_random":
        name = pool[np.random.randint(0, len(pool))]
        net = sample_random_embedder(name, embedder_size, input_dim, embed_hidden, embed_dim, device)
        return net, f"pool:{name}"

    if mode == "fusion":
        per_dim = int(config.get("dm_fusion_per_dim", max(8, embed_dim // len(pool))))
        embedders, per_dims = [], []
        for name in pool:
            e = sample_random_embedder(name, embedder_size, input_dim, embed_hidden, per_dim, device)
            e.eval()
            embedders.append(e)
            per_dims.append(per_dim)
        net = FusionEmbedder(embedders, per_dims, embed_dim).to(device)
        net.eval()
        net.freeze()
        return net, f"fusion:{'+'.join(pool)}"

    raise ValueError(f"Unknown dm_embedder_mode '{mode}'")


def tame_fusion_synthesize(data, config):
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

    # for fusion mode, optionally build once
    mode = str(config.get("dm_embedder_mode", "single")).lower()
    fixed_net, fixed_tag = None, None
    if mode == "fusion" and config.get("dm_fusion_build_once", True):
        fixed_net, fixed_tag = _sample_embedder(config, input_dim, embed_hidden, embed_dim, device)

    for it in range(iters + 1):
        if fixed_net is not None:
            embed_net, tag = fixed_net, fixed_tag
        else:
            embed_net, tag = _sample_embedder(config, input_dim, embed_hidden, embed_dim, device)
        embed_net.eval()

        optimizer.zero_grad(set_to_none=True)

        loss_mean = torch.zeros((), device=device)
        loss_cov = torch.zeros((), device=device)

        for c in range(num_classes):
            real_b = get_real_batch(c, batch_real)
            syn_b = syn_data[c * ipc:(c + 1) * ipc]

            feat_real = embed_net(real_b).detach()
            feat_syn = embed_net(syn_b)

            mu_r, cov_r = cov_matrix(feat_real, eps)
            mu_s, cov_s = cov_matrix(feat_syn, eps)

            loss_mean = loss_mean + ((mu_r - mu_s) ** 2).sum()
            diff = cov_r - cov_s
            loss_cov = loss_cov + cov_weight * (diff * diff).sum()

        loss_total = loss_mean + loss_cov

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
            print(f"[TAME-Fusion | {tag}] iter {it:04d} | loss {cur:.6f} | best {best_loss:.6f}@{best_it:04d}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {"X_syn": best_syn.cpu(), "y_syn": label_syn.cpu(),
             "best_loss": best_loss, "best_it": best_it},
            os.path.join(save_dir, "best_syn.pt"),
        )

    return best_syn, label_syn.detach()
