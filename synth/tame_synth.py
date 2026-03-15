"""
TAME: Tabular Alignment via Moment Embeddings

Core synthesizer — mean + full covariance matching in frozen embedder space.
One random embedder per iteration, best-loss checkpoint, saves .pt output.
"""

import os
import torch
import numpy as np
from models.embedders import sample_random_embedder


def cov_matrix(z, eps=0.0):
    n = z.shape[0]
    mu = z.mean(0, keepdim=True)
    zc = z - mu
    cov = (zc.T @ zc) / max(n, 1)
    if eps > 0:
        cov = cov + eps * torch.eye(cov.shape[0], device=z.device)
    return mu.squeeze(0), cov


def tame_synthesize(data, config):
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

    grad_clip = float(config.get("grad_clip", 10.0))
    eps = float(config.get("moment_eps", 1e-6))
    cov_weight = float(config.get("cov_weight", 1.0))

    save_dir = config.get("save_dir", None)

    # group indices by class
    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    def get_real_batch(c, n):
        idx = indices_class[c]
        return X_train[np.random.choice(idx, n, replace=len(idx) < n)]

    # init synthetic data from real samples
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
            print(
                f"[TAME] iter {it:04d} | "
                f"loss {cur:.6f} | best {best_loss:.6f}@{best_it:04d}"
            )

    # save best checkpoint
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {"X_syn": best_syn.cpu(), "y_syn": label_syn.cpu(),
             "best_loss": best_loss, "best_it": best_it},
            os.path.join(save_dir, "best_syn.pt"),
        )

    return best_syn, label_syn.detach()
