#!/usr/bin/env python3
"""
Normality diagnostic: does the TAME pipeline push data toward Gaussianity?

For each dataset we compute normality statistics on SIX data versions:

  1. raw            — original feature space
  2. embedded       — raw data pushed through random frozen embedders
                       (averaged across 20 random draws)
  3. random_low     — distilled with random embedder, mean+cov loss
  4. random_high    — distilled with random embedder, higher-order loss
  5. learned_low    — distilled with learned embedder, mean+cov loss
  6. learned_high   — distilled with learned embedder, higher-order loss

For each version we measure (per class, then averaged):

  Univariate: Shapiro-Wilk per dimension → pass rate at p>0.05
              avg |skewness|, avg kurtosis (Gaussian = 3)
  Multivariate: Mardia's skewness (pass rate) and kurtosis statistic
                (Gaussian expects p(p+2), where p = # dimensions)

The claim tested:
  raw → embedded → distilled becomes progressively MORE Gaussian.

Usage:
  # quick test on one dataset:
  python main_normality_test.py --datasets adult

  # full overnight run:
  python main_normality_test.py
"""

import os
import sys
import glob
import argparse
import random
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.prepare_database import prepare_db, DATASET_REGISTRY
from models.embedders import sample_random_embedder


# ============================================================
# Multivariate normality — Mardia's test
# ============================================================

def mardia_test(X, max_n=3000):
    """
    Mardia's test for multivariate normality.

    Subsamples to max_n rows before computing the O(n^2) Mahalanobis matrix,
    otherwise it explodes on large datasets (airlines has 500k samples ->
    300+ GB of memory).  Mardia's asymptotic distribution is accurate for
    n > a few hundred, so 3000 is comfortably sufficient.

    Returns dict with:
      skewness_stat     : Mardia's b_{1,p}
      skewness_chi2     : n * b_{1,p} / 6 (~ chi-sq under H0)
      skewness_p        : p-value (pass if > 0.05)
      kurtosis_stat     : Mardia's b_{2,p}
      kurtosis_expected : p(p+2) under Gaussian
      kurtosis_z        : standardized statistic
      kurtosis_p        : p-value (pass if > 0.05)
    """
    n_full, p = X.shape

    if n_full < p + 10:
        return None

    if n_full > max_n:
        idx = np.random.choice(n_full, max_n, replace=False)
        X = X[idx]

    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = np.cov(Xc, rowvar=False, bias=False)

    try:
        S_inv = np.linalg.inv(S + 1e-8 * np.eye(p))
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    D = Xc @ S_inv @ Xc.T

    b1p = (D ** 3).mean()
    skew_chi2 = n * b1p / 6.0
    skew_df = p * (p + 1) * (p + 2) / 6.0
    skew_p = 1.0 - stats.chi2.cdf(skew_chi2, df=skew_df)

    b2p = (np.diag(D) ** 2).mean()
    kurt_expected = p * (p + 2)
    kurt_var = 8.0 * p * (p + 2) / n
    kurt_z = (b2p - kurt_expected) / np.sqrt(kurt_var)
    kurt_p = 2.0 * (1.0 - stats.norm.cdf(abs(kurt_z)))

    return dict(
        skewness_stat=float(b1p),
        skewness_chi2=float(skew_chi2),
        skewness_p=float(skew_p),
        kurtosis_stat=float(b2p),
        kurtosis_expected=float(kurt_expected),
        kurtosis_z=float(kurt_z),
        kurtosis_p=float(kurt_p),
    )


# ============================================================
# Univariate normality — Shapiro-Wilk per dimension
# ============================================================

def univariate_stats(X, max_samples=5000):
    """
    Returns dict of per-dimension statistics averaged across dimensions:
      shapiro_pass_rate  : fraction of dims with Shapiro-Wilk p > 0.05
      avg_abs_skewness   : mean |skewness| across dims
      avg_kurtosis       : mean raw kurtosis (Gaussian = 3)
    """
    n, p = X.shape
    if n < 3:
        return None

    X_sub = X[:max_samples] if n > max_samples else X

    shapiro_pvals = []
    skews = []
    kurts = []

    for d in range(p):
        col = X_sub[:, d]

        # skip constant columns: Shapiro-Wilk returns W=1, p=1 trivially
        # which inflates the pass rate. Common cause: one-hot features
        # that become constant within a class after the class split.
        if np.std(col) < 1e-10:
            continue

        try:
            if len(col) >= 3:
                _, sw_p = stats.shapiro(col)
                shapiro_pvals.append(sw_p)
        except Exception:
            pass

        try:
            skews.append(float(stats.skew(col)))
            kurts.append(float(stats.kurtosis(col, fisher=False)))
        except Exception:
            pass

    if not shapiro_pvals:
        return None

    return dict(
        shapiro_pass_rate=float(np.mean(np.array(shapiro_pvals) > 0.05)),
        avg_abs_skewness=float(np.mean(np.abs(skews))),
        avg_kurtosis=float(np.mean(kurts)),
    )


# ============================================================
# Helpers
# ============================================================

def analyze_matrix(X, y):
    """
    Run both univariate and multivariate normality on X, averaged across classes.
    Returns flat dict of metrics.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    classes = np.unique(y)

    uni_results = []
    mar_results = []

    for c in classes:
        Xc = X[y == c]
        if Xc.shape[0] < 10:
            continue

        u = univariate_stats(Xc)
        if u is not None:
            uni_results.append(u)

        m = mardia_test(Xc)
        if m is not None:
            mar_results.append(m)

    out = {}

    if uni_results:
        out["shapiro_pass_rate"] = float(np.mean([r["shapiro_pass_rate"] for r in uni_results]))
        out["avg_abs_skewness"] = float(np.mean([r["avg_abs_skewness"] for r in uni_results]))
        out["avg_kurtosis"] = float(np.mean([r["avg_kurtosis"] for r in uni_results]))
    else:
        out["shapiro_pass_rate"] = float("nan")
        out["avg_abs_skewness"] = float("nan")
        out["avg_kurtosis"] = float("nan")

    if mar_results:
        out["mardia_skew_pass_rate"] = float(np.mean([r["skewness_p"] > 0.05 for r in mar_results]))
        out["mardia_kurt_pass_rate"] = float(np.mean([r["kurtosis_p"] > 0.05 for r in mar_results]))
        out["mardia_kurt_stat"] = float(np.mean([r["kurtosis_stat"] for r in mar_results]))
        out["mardia_kurt_expected"] = float(np.mean([r["kurtosis_expected"] for r in mar_results]))
        out["mardia_kurt_deviation"] = out["mardia_kurt_stat"] - out["mardia_kurt_expected"]
    else:
        out["mardia_skew_pass_rate"] = float("nan")
        out["mardia_kurt_pass_rate"] = float("nan")
        out["mardia_kurt_stat"] = float("nan")
        out["mardia_kurt_expected"] = float("nan")
        out["mardia_kurt_deviation"] = float("nan")

    return out


def build_embedded_version(X, y, input_dim, device, embedder_type, embedder_size,
                           embed_hidden, embed_dim, n_embedders=20,
                           max_samples=2000):
    """
    Push raw data through n_embedders random frozen embedders, stack outputs,
    return a single "average embedding behavior" matrix.

    We can't average the embeddings themselves (different bases), so instead
    we report statistics averaged across the n_embedders runs.
    """
    X_t = X.to(device).float() if isinstance(X, torch.Tensor) else torch.tensor(X, device=device, dtype=torch.float32)
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

    n = X_t.shape[0]
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        X_t = X_t[idx]
        y_np = y_np[idx]

    all_stats = []
    for ei in range(n_embedders):
        embed_net = sample_random_embedder(
            embedder_type, embedder_size, input_dim,
            embed_hidden, embed_dim, device,
        )
        embed_net.eval()
        with torch.no_grad():
            Z = embed_net(X_t).cpu().numpy()

        s = analyze_matrix(Z, y_np)
        all_stats.append(s)

    keys = all_stats[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in all_stats if np.isfinite(s[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def load_distilled(synth_dir, dataset, condition, ipc=50, max_runs=5,
                   embedder_filter=None):
    """
    Load distilled data from .pt files.

    Expects files named:
      {dataset}__{condition}__{embedder}__ipc{ipc}__run{NN}.pt

    If embedder_filter is provided (e.g. 'dcnv2_base', 'ln_res_l', 'node'),
    only files where the embedder slot matches are loaded.  This lets us
    separate `tame_dcnv2_base` from `tame_node` etc. when both have
    condition='tame' in the filename.

    Returns a list of (X, y) tuples — one per run.
    """
    if embedder_filter:
        pattern = os.path.join(
            synth_dir,
            f"{dataset}__{condition}__{embedder_filter}__ipc{ipc}__run*.pt",
        )
    else:
        pattern = os.path.join(
            synth_dir,
            f"{dataset}__{condition}__*__ipc{ipc}__run*.pt",
        )
    files = sorted(glob.glob(pattern))[:max_runs]

    out = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        X = d["X_syn"] if "X_syn" in d else d.get("X")
        y = d["y_syn"] if "y_syn" in d else d.get("y")
        if X is not None and y is not None:
            out.append((X, y))
    return out


# ============================================================
# Main
# ============================================================

def run(args):
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = list(DATASET_REGISTRY.keys())

    conditions = args.conditions.split(",")

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    config_base = dict(
        random_seed=args.seed,
        device=device,
    )

    rows = []

    for db_name in datasets:
        print(f"\n{'='*70}")
        print(f"  DATASET: {db_name}")
        print(f"{'='*70}")

        data = prepare_db(config_base, name=db_name)
        X_train = data["X_train"]
        y_train = data["y_train"]
        input_dim = int(data["input_dim"])

        # ---- 1. raw ----
        print("\n  [1] raw feature space")
        raw_stats = analyze_matrix(X_train, y_train)
        print_stats(raw_stats, "      ")
        rows.append({"dataset": db_name, "version": "raw", **raw_stats})

        # ---- 2. embedded (averaged across N random embedders) ----
        print(f"\n  [2] embedded (avg over {args.n_embedders} random embedders)")
        emb_stats = build_embedded_version(
            X_train, y_train, input_dim, device,
            args.embedder, args.embedder_size,
            args.embed_hidden, args.embed_dim,
            n_embedders=args.n_embedders,
            max_samples=args.max_samples,
        )
        print_stats(emb_stats, "      ")
        rows.append({"dataset": db_name, "version": "embedded", **emb_stats})

        # ---- 3-6. distilled conditions ----
        # conditions can be specified as either:
        #   "tame_orders"           -> matches all embedders for that condition
        #   "tame:dcnv2_base"       -> only files with that specific embedder
        # We use ':' as a separator between condition tag and embedder filter.
        for cond_spec in conditions:
            if ":" in cond_spec:
                cond, emb_filter = cond_spec.split(":", 1)
                display_name = f"{cond}:{emb_filter}"
            else:
                cond, emb_filter = cond_spec, None
                display_name = cond_spec

            print(f"\n  [3+] distilled — {display_name}")
            runs = load_distilled(args.synth_dir, db_name, cond,
                                  ipc=args.ipc, max_runs=args.max_runs,
                                  embedder_filter=emb_filter)
            if not runs:
                print(f"      (no .pt files found for {display_name})")
                rows.append({"dataset": db_name, "version": display_name})
                continue

            run_stats = [analyze_matrix(X, y) for X, y in runs]
            keys = run_stats[0].keys()
            cond_stats = {}
            for k in keys:
                vals = [s[k] for s in run_stats if np.isfinite(s[k])]
                cond_stats[k] = float(np.mean(vals)) if vals else float("nan")

            print_stats(cond_stats, "      ")
            rows.append({"dataset": db_name, "version": display_name, **cond_stats})

    # ---- save ----
    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, "normality_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ---- summary table ----
    print("\n" + "="*100)
    print("  SUMMARY")
    print("="*100)
    print(f"\n{'Dataset':<14} {'Version':<16} {'SW pass':>8} "
          f"{'|Skew|':>7} {'Kurt':>7} {'M-Sk%':>7} {'M-Kt%':>7} "
          f"{'M-Kt stat':>11} {'M-Kt exp':>10}")
    print("-"*100)
    for _, r in df.iterrows():
        fmt_or_dash = lambda v, p=3: f"{v:.{p}f}" if (isinstance(v, float) and np.isfinite(v)) else "—"
        print(f"{r['dataset']:<14} {r['version']:<16} "
              f"{fmt_or_dash(r.get('shapiro_pass_rate'), 3):>8} "
              f"{fmt_or_dash(r.get('avg_abs_skewness'), 3):>7} "
              f"{fmt_or_dash(r.get('avg_kurtosis'), 3):>7} "
              f"{fmt_or_dash(r.get('mardia_skew_pass_rate'), 3):>7} "
              f"{fmt_or_dash(r.get('mardia_kurt_pass_rate'), 3):>7} "
              f"{fmt_or_dash(r.get('mardia_kurt_stat'), 1):>11} "
              f"{fmt_or_dash(r.get('mardia_kurt_expected'), 1):>10}")
    print("-"*100)


def print_stats(s, prefix=""):
    print(f"{prefix}Shapiro-Wilk pass rate: {s['shapiro_pass_rate']:.1%}")
    print(f"{prefix}Avg |skewness|:         {s['avg_abs_skewness']:.3f}  "
          f"(Gaussian = 0)")
    print(f"{prefix}Avg kurtosis:           {s['avg_kurtosis']:.3f}  "
          f"(Gaussian = 3)")
    if np.isfinite(s.get('mardia_kurt_stat', float('nan'))):
        print(f"{prefix}Mardia skew pass:       {s['mardia_skew_pass_rate']:.1%}")
        print(f"{prefix}Mardia kurt pass:       {s['mardia_kurt_pass_rate']:.1%}")
        print(f"{prefix}Mardia kurt stat:       {s['mardia_kurt_stat']:.1f}  "
              f"(Gaussian expects {s['mardia_kurt_expected']:.1f})")
        print(f"{prefix}Mardia kurt deviation:  "
              f"{s['mardia_kurt_deviation']:+.1f}")
    else:
        print(f"{prefix}Mardia test skipped (n too small for embedding dim)")


def main():
    p = argparse.ArgumentParser(
        description="Normality diagnostic for raw, embedded, and distilled data")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Datasets to test (default: all 18)")
    p.add_argument("--conditions", type=str,
                   default="random_low,random_high,learned_low,learned_high",
                   help="Comma-separated distillation conditions to test")
    p.add_argument("--synth-dir", default="synth_outputs_learned_ablation",
                   help="Directory containing distilled .pt files")
    p.add_argument("--ipc", type=int, default=50)
    p.add_argument("--max-runs", type=int, default=5,
                   help="Number of run files to average across")
    p.add_argument("--embedder", default="ln_res_l",
                   help="Embedder type for 'embedded' version")
    p.add_argument("--embedder-size", default="base")
    p.add_argument("--embed-hidden", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=48)
    p.add_argument("--n-embedders", type=int, default=20,
                   help="Random embedders to average for 'embedded' version")
    p.add_argument("--max-samples", type=int, default=2000,
                   help="Max samples for embedded-version analysis")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="results_normality_IPC50_levage",)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run(args)


if __name__ == "__main__":
    main()
