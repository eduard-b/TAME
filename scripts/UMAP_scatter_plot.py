import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def plot_umap_real_vs_syn_colormap_ax(
    ax,
    art,
    title=None,
):
    import numpy as np
    import torch

    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    Zr, yr = _np(art["Z_real"]), _np(art["y_real"]).astype(int)
    Z0, y0 = _np(art["Z_syn0"]), _np(art["y_syn0"]).astype(int)
    ZT, yT = _np(art["Z_synT"]), _np(art["y_synT"]).astype(int)

    classes = sorted(set(yr.tolist()) | set(y0.tolist()) | set(yT.tolist()))

    real_colors = {0: "tab:blue", 1: "tab:red"}
    syn_colors  = {0: "tab:blue", 1: "tab:red"}

    # --- Real ---
    for c in classes:
        m = (yr == c)
        if m.any():
            ax.scatter(
                Zr[m, 0], Zr[m, 1],
                s=6, alpha=0.12,
                c=real_colors.get(c),
                label=f"real class {c}",
            )

    # --- Syn0 ---
    for c in classes:
        m = (y0 == c)
        if m.any():
            ax.scatter(
                Z0[m, 0], Z0[m, 1],
                s=80, marker="x", linewidths=2,
                c=syn_colors.get(c),
                label=f"syn0 class {c}",
            )

    # --- SynT ---
    for c in classes:
        m = (yT == c)
        if m.any():
            ax.scatter(
                ZT[m, 0], ZT[m, 1],
                s=80, marker="o",
                edgecolors="k", linewidths=0.5,
                c=syn_colors.get(c),
                label=f"synT class {c}",
            )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")


run_dir = "stageU/magic_umap_demo"

raw_art   = torch.load(os.path.join(run_dir, "umap_raw.pt"),   map_location="cpu")
embed_art = torch.load(os.path.join(run_dir, "umap_embed.pt"), map_location="cpu")

# 1. Initialize without constrained_layout to gain manual control
fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6)
)

plot_umap_real_vs_syn_colormap_ax(axes[0], raw_art, title="RAW space")
plot_umap_real_vs_syn_colormap_ax(axes[1], embed_art, title="EMBEDDER space")

# 2. Extract unique handles for the legend
handles, labels = axes[0].get_legend_handles_labels()
seen = set()
uniq_h, uniq_l = [], []
for h, l in zip(handles, labels):
    if l not in seen:
        uniq_h.append(h)
        uniq_l.append(l)
        seen.add(l)

# 3. Create room at the bottom (0.2 is roughly 20% of figure height)
# This prevents 'UMAP-1' from hitting the legend.
plt.tight_layout(rect=[0, 0.12, 1, 1]) 

# 4. Place the legend in that specific empty 12% space at the bottom
# loc="upper center" paired with a low y-coordinate (0.1) keeps it clear of labels
fig.legend(
    uniq_h, uniq_l,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.1), 
    ncol=6,
    fontsize=9,
    frameon=False,
)

out_pdf = os.path.join(run_dir, "umap_raw_vs_embed.pdf")

# 5. bbox_inches='tight' ensures the legend is included in the saved PDF
plt.savefig(out_pdf, bbox_inches='tight')
plt.close(fig)

print("saved:", out_pdf)