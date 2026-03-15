import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import re

# ---------- helpers ----------

def append_mean_std_rows(df):
    """
    Append Mean and Std rows computed across datasets (rows) for each method (column).
    """
    mean_row = df.mean(axis=0)
    std_row  = df.std(axis=0)

    df_ext = pd.concat(
        [
            df,
            pd.DataFrame([mean_row], index=["Mean"]),
            pd.DataFrame([std_row], index=["Std"]),
        ],
        axis=0,
    )
    return df_ext

def load_table(
    path,
    *,
    drop_full=True,
    index_col="dataset",
    drop_extra_regex=r"^(global_|.*_global_).*|^(global_mean|global_std)$"
):
    df = pd.read_csv(path)

    # Robust index handling (some CSVs store the index as Unnamed: 0)
    if index_col in df.columns:
        df = df.set_index(index_col)
    elif "Unnamed: 0" in df.columns:
        df = df.set_index("Unnamed: 0")

    # Normalize method column names across files
    df = df.rename(columns={
        "K-means": "k-means",
        "k-medoid": "k-means",
        "ln_res_l": "ln_residual",
        "ln_res": "ln_residual",
        "dcnv2_base": "dcnv2",
    })

    # Drop extra/global summary columns if present
    if drop_extra_regex:
        pat = re.compile(drop_extra_regex)
        drop_cols = [c for c in df.columns if isinstance(c, str) and pat.match(c)]
        df = df.drop(columns=drop_cols, errors="ignore")

    # Optionally drop full column
    if drop_full:
        df = df.drop(columns=[c for c in df.columns if isinstance(c, str) and c.lower() == "full"], errors="ignore")

    # Keep only numeric columns (guards against stray text columns)
    df = df[[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]]

    return df


def discrete_blues(k: int):
    """
    k discrete colors from matplotlib 'Blues', with rank=1 darkest.
    """
    if k < 2:
        raise ValueError("Need at least 2 columns/methods for a rank heatmap.")
    # Avoid the very light end; sample from mid->dark, then reverse so [0]=darkest.
    vals = np.linspace(0.25, 0.95, k)
    colors = plt.cm.Blues(vals)[::-1]  # reverse: darkest first
    return ListedColormap(colors)

def rank_heatmap_with_values(ax, df, title, *, lower_is_better: bool, fmt="{:.3f}"):
    methods = df.columns.tolist()
    k = len(methods)

    cmap = discrete_blues(k)

    # Rank per row: rank 1 = best
    ranks = df[methods].rank(axis=1, ascending=lower_is_better, method="average")

    bounds = np.arange(0.5, k + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(ranks.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)

    # Gridlines
    ax.set_xticks(np.arange(k + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df.index) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate with VALUES (not ranks)
    values = df[methods].values.astype(float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            s = "" if np.isnan(v) else fmt.format(v)
            r = ranks.values[i, j]
            # rank 1..k => dark-to-light; use white text on darker half
            text_color = "white" if r <= (k / 2) else "black"
            ax.text(j, i, s, ha="center", va="center", fontsize=8, color=text_color)

    return im, k

def maybe_rename_kmedoid(df):
    return df.rename(columns={"k-medoid": "k-means"})

def strip_rl_prefix(df):
    """
    If RL csv has columns like RL_random, RL_node..., turn them into random, node...
    """
    rename = {}
    for c in df.columns:
        if isinstance(c, str) and c.startswith("RL_"):
            rename[c] = c[len("RL_"):]
    return df.rename(columns=rename)

# ---------- figure 1: single ACC heatmap including random + full ----------

def save_acc_full_heatmap(
    acc_path,
    out_pdf="acc_mlp_full_values.pdf",
    *,
    fmt="{:.3f}",
    rename_kmedoid=True,
):
    # Keep full + random (i.e., do NOT drop full)
    df_acc = load_table(acc_path, drop_full=False)

    if rename_kmedoid:
        df_acc = maybe_rename_kmedoid(df_acc)

    # Optional: you may want a stable column order, if present
    preferred = ["random", "k-means", "dcnv2", "ln_residual", "node", "full"]
    cols = [c for c in preferred if c in df_acc.columns] + [c for c in df_acc.columns if c not in preferred]
    df_acc = df_acc[cols]

    fig = plt.figure(figsize=(7.2, 4.6))  # good for LaTeX single/medium width; tweak as needed
    ax = fig.add_subplot(111)

    im, k = rank_heatmap_with_values(ax, df_acc, "ACC (MLP) — incl. random & full", lower_is_better=False, fmt=fmt)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=list(range(1, k + 1)))
    cbar.set_label("Rank (1 = best)")

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")

# ---------- figure 2: side-by-side RL and RR ----------

def save_rl_and_rr_pair(
    rl_path,
    rr_path,
    out_pdf="pair_rl_rr_mlp_values.pdf",
    *,
    fmt="{:.3f}",
    rename_kmedoid=True,
):
    df_rl = load_table(rl_path, drop_full=False)  # RL csv shouldn't have full anyway, but keep safe
    df_rr = load_table(rr_path, drop_full=True)   # you said RR missing random/full; keep as-is

    if rename_kmedoid:
        df_rl = maybe_rename_kmedoid(df_rl)
        df_rr = maybe_rename_kmedoid(df_rr)

    # Make RL columns match RR column names (RL_random -> random, etc.)
    df_rl = strip_rl_prefix(df_rl)

    # Align methods shown in both panels
    common = [c for c in df_rr.columns if c in df_rl.columns]
    df_rl = df_rl[common]
    df_rr = df_rr[common]
    # Append Mean / Std rows (aggregated across datasets)
    df_rl = append_mean_std_rows(df_rl)
    df_rr = append_mean_std_rows(df_rr)



    fig = plt.figure(figsize=(14.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.08)

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], sharey=axL)
    cax = fig.add_subplot(gs[0, 2])

    # RL: lower is better (smaller loss)
    imL, kL = rank_heatmap_with_values(axL, df_rl, "Relative Loss (ACC full − method)", lower_is_better=True, fmt=fmt)
    # RR: lower is better
    imR, kR = rank_heatmap_with_values(axR, df_rr, "RR (ACC), MLP", lower_is_better=True, fmt=fmt)

    if kL != kR:
        raise ValueError(f"RL has {kL} methods, RR has {kR}. Make them match.")

    # Hide duplicate y labels on right
    plt.setp(axR.get_yticklabels(), visible=False)
    axR.tick_params(axis="y", length=0)

    cbar = fig.colorbar(imR, cax=cax, ticks=list(range(1, kR + 1)))
    cbar.set_label("Rank (1 = best)")

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")

# ---------- example calls ----------

if __name__ == "__main__":
    base = "final_results/K_MEANS_REPEAT"

    # Figure 1: ACC full (includes random + full)
    save_acc_full_heatmap(
        acc_path=f"{base}/acc_mlp_IPC50.csv",
        out_pdf="final_results/K_MEANS_REPEAT/acc_mlp_IPC50.pdf",
        fmt="{:.3f}",
        rename_kmedoid=True,
    )

    # Figure 2: RL + RR side-by-side
    save_rl_and_rr_pair(
        rl_path=f"{base}/acc_mlp_IPC50_RL.csv",
        rr_path=f"{base}/acc_mlp_IPC50_RR.csv",
        out_pdf="final_results/K_MEANS_REPEAT/acc_mlp_IPC50_rl_rr_values.pdf",
        fmt="{:.3f}",
        rename_kmedoid=True,
    )
