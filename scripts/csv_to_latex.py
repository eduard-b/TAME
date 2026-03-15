#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

FILES = [
    "acc_mlp_mean.csv",
    "acc_rf_mean.csv",
    "auc_mlp_mean.csv",
    "auc_rf_mean.csv"
]

FILES = [
    "rr_acc_mlp_mean.csv",
    "rr_acc_rf_mean.csv",
    "rr_auc_mlp_mean.csv",
    "rr_auc_rf_mean.csv"
]

# Columns to consider for "best" (exclude 'full')
BOLD_POOL = ["random", "k-medoid", "ln_residual", "dcnv2", "node"]
BOLD_POOL = ["k-medoid", "ln_residual", "dcnv2", "node"]
FULL_COL = "full"


def fmt(x, ndigits=4):
    if pd.isna(x):
        return ""
    return f"{x:.{ndigits}f}"


def bold_best_per_row(df: pd.DataFrame, ndigits=4) -> pd.DataFrame:
    """
    Returns a string DataFrame with best value (excluding FULL_COL) bolded per row.
    """
    out = df.copy()

    # ensure numeric for comparisons
    for c in out.columns:
        if c != "dataset":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # build formatted string version
    str_df = pd.DataFrame({"dataset": out["dataset"].astype(str)})

    # compute row-wise argmax over BOLD_POOL (ignore NaN)
    pool = [c for c in BOLD_POOL if c in out.columns]

    if not pool:
        raise ValueError(f"None of BOLD_POOL columns found. Present: {list(out.columns)}")

    # idx of max per row
    max_col = out[pool].idxmax(axis=1)

    for c in out.columns:
        if c == "dataset":
            continue
        col_vals = out[c]
        s = col_vals.map(lambda v: fmt(v, ndigits))
        # bold if this column is row-best and not 'full'
        if c in pool:
            s = np.where(max_col == c, r"\textbf{" + s + "}", s)
        str_df[c] = s

    return str_df


def to_latex_table(df_str: pd.DataFrame, caption: str, label: str) -> str:
    # Use escape=False so \textbf{} is preserved
    latex = df_str.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="l" + "c" * (df_str.shape[1] - 1),
    )
    return latex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory with the 8 CSVs")
    ap.add_argument("--out_dir", required=True, help="Directory to write .tex files")
    ap.add_argument("--digits", type=int, default=4, help="Decimal digits in LaTeX")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        p = in_dir / fname
        if not p.exists():
            raise FileNotFoundError(p)

        df = pd.read_csv(p)
        if "dataset" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "dataset"})

        # Keep a consistent column order if present
        cols_order = ["dataset"] + [c for c in BOLD_POOL if c in df.columns] + ([FULL_COL] if FULL_COL in df.columns else [])
        df = df[[c for c in cols_order if c in df.columns]]

        df_str = bold_best_per_row(df, ndigits=args.digits)

        base = fname.replace(".csv", "")
        caption = base.replace("_", " ").upper()
        label = f"tab:{base}"

        latex = to_latex_table(df_str, caption=caption, label=label)

        out_path = out_dir / f"{base}.tex"
        out_path.write_text(latex, encoding="utf-8")
        print(f"Wrote {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
