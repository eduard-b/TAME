#!/usr/bin/env python3
"""
Generate from results.csv:
    - 6 per-classifier wide CSVs (acc_{clf}.csv, rr_{clf}.csv)         [3 each]
    - 2 median-summary CSVs (acc_median_by_classifier, rr_median_...)  [new]
    - 1 LaTeX file with all 8 tables (summaries first, then per-classifier)

Conventions:
    method_label = "{method}_{embedder}" if embedder set, else "{method}"
    ACC tables include `full` and `random` as reference columns.
    RR tables exclude them (full -> RR=0, random -> RR=1 by definition).
    Top-k counts use rank(method='min'): ties share the rank.

Usage:
    python make_all_tables.py results.csv -o tables/

LaTeX preamble required:
    \\usepackage{booktabs}
    \\usepackage{graphicx}   % for \\resizebox on wide tables
"""
import argparse
import os
import pandas as pd


# Display order — methods absent from the CSV are silently dropped
METHOD_ORDER = [
    "tame_ln_res_l", "tame_node", "tame_dcnv2_base",
    "leverage_score", "vq", "voronoi", "gonzalez", "ctgan", "tvae",
    "full", "random",
]

# Pretty LaTeX names
TEX_NAME = {
    "tame_ln_res_l":   r"TAME$\times$ln\_res\_l",
    "tame_node":       r"TAME$\times$NODE",
    "tame_dcnv2_base": r"TAME$\times$DCNv2",
    "leverage_score":  r"Leverage",
    "vq":              r"VQ",
    "voronoi":         r"Voronoi",
    "gonzalez":        r"Gonzalez",
    "ctgan":           r"CTGAN",
    "tvae":            r"TVAE",
    "full":            r"Full",
    "random":          r"Random",
}
CLF_NAME = {"mlp": "MLP", "rf": "RF", "xgboost": "XGBoost"}


# ---------- core data prep ----------

def method_label(row):
    if pd.isna(row["embedder"]) or row["embedder"] == "":
        return row["method"]
    return f"{row['method']}_{row['embedder']}"


def compute_long_with_rr(df):
    df = df.copy()
    df["method_label"] = df.apply(method_label, axis=1)
    fr = (df[df["method_label"].isin(["full", "random"])]
          .pivot_table(index=["dataset", "classifier"],
                       columns="method_label", values="test_acc_mean")
          .reset_index())
    fr.columns.name = None
    df = df.merge(fr, on=["dataset", "classifier"], how="left")
    df["rr"] = (df["full"] - df["test_acc_mean"]) / (df["full"] - df["random"])
    return df


# ---------- per-classifier wide tables ----------

def append_summary_rows(wide, higher_is_better):
    mean_row   = wide.mean(axis=0)
    median_row = wide.median(axis=0)
    ranks = wide.rank(axis=1, ascending=not higher_is_better, method="min")
    top1 = (ranks <= 1).sum(axis=0).astype(int)
    top3 = (ranks <= 3).sum(axis=0).astype(int)
    summary = pd.DataFrame(
        [mean_row, median_row, top1, top3],
        index=["mean", "median", "top1_count", "top3_count"],
    )
    return pd.concat([wide, summary])


def build_per_classifier(df_long, classifier, ds_order):
    sub = df_long[df_long["classifier"] == classifier]
    acc_w = sub.pivot_table(index="dataset", columns="method_label",
                            values="test_acc_mean")
    cols = [m for m in METHOD_ORDER if m in acc_w.columns]
    acc_w = acc_w.reindex(index=ds_order, columns=cols)
    acc_full = append_summary_rows(acc_w, higher_is_better=True)

    rr_cols = [c for c in cols if c not in ("full", "random")]
    rr_w = sub.pivot_table(index="dataset", columns="method_label",
                           values="rr").reindex(index=ds_order, columns=rr_cols)
    rr_full = append_summary_rows(rr_w, higher_is_better=False)

    return acc_full, rr_full


# ---------- median-by-classifier summary ----------

def median_by_classifier(df_long, metric, classifiers, exclude=()):
    """Rows = method, cols = classifier; value = median across datasets."""
    sub = df_long[~df_long["method_label"].isin(exclude)]
    pivot = sub.pivot_table(
        index="method_label", columns="classifier",
        values=metric, aggfunc="median",
    )
    methods = [m for m in METHOD_ORDER if m in pivot.index and m not in exclude]
    pivot = pivot.reindex(index=methods, columns=classifiers)
    return pivot


# ---------- LaTeX rendering ----------

def fmt_cell(value, is_int=False):
    if pd.isna(value):
        return "--"
    if is_int:
        return f"{int(round(value))}"
    return f"{value:.4f}"


def fmt_row_with_bold(values, best_idx, is_int=False):
    cells = []
    for i, v in enumerate(values):
        c = fmt_cell(v, is_int=is_int)
        if i == best_idx:
            c = r"\textbf{" + c + "}"
        cells.append(c)
    return cells


def latex_summary_table(pivot, caption, label, higher_is_better):
    """Render the median-by-classifier summary (rows=method, cols=classifier)."""
    cols = list(pivot.columns)
    align = "l" + "c" * len(cols)
    lines = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule",
        " & ".join(["Method"] + [CLF_NAME.get(c, c) for c in cols]) + r" \\",
        r"\midrule",
    ]
    # Best per column: bold once per column
    if higher_is_better:
        best_per_col = pivot.idxmax(axis=0)
    else:
        best_per_col = pivot.idxmin(axis=0)
    for m in pivot.index:
        cells = [TEX_NAME.get(m, m)]
        for c in cols:
            v = pivot.loc[m, c]
            s = fmt_cell(v)
            if best_per_col[c] == m:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_wide_table(wide, caption, label):
    """Render a per-classifier wide table.
    Datasets in dataset rows, then mean/median/top1/top3 in summary rows.
    No bolding to keep tables clean — the user can add highlighting later."""
    methods = list(wide.columns)
    align = "l" + "c" * len(methods)
    summary_idx = {"mean", "median", "top1_count", "top3_count"}
    SUMMARY_PRETTY = {
        "mean": "Mean", "median": "Median",
        "top1_count": "Top-1 count", "top3_count": "Top-3 count",
    }
    head = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule",
        " & ".join(["Dataset"] + [TEX_NAME.get(m, m) for m in methods]) + r" \\",
        r"\midrule",
    ]
    body = []
    body_done = False
    for idx in wide.index:
        if idx in summary_idx and not body_done:
            body.append(r"\midrule")
            body_done = True
        is_int = idx in ("top1_count", "top3_count")
        label_cell = SUMMARY_PRETTY.get(idx, str(idx))
        cells = [label_cell] + [fmt_cell(wide.loc[idx, m], is_int=is_int)
                                for m in methods]
        body.append(" & ".join(cells) + r" \\")
    tail = [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(head + body + tail)


# ---------- driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("-o", "--out-dir", default="tables")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_raw = pd.read_csv(args.input_csv)
    df = compute_long_with_rr(df_raw)

    classifiers = sorted(df["classifier"].unique())
    print(f"Classifiers: {classifiers}")

    # Dataset order = order of first appearance in the file
    ds_order = df_raw.drop_duplicates("dataset")["dataset"].tolist()

    # ---- per-classifier wide CSVs ----
    wide_tables = {}
    for clf in classifiers:
        acc_w, rr_w = build_per_classifier(df, clf, ds_order)
        wide_tables[("acc", clf)] = acc_w
        wide_tables[("rr",  clf)] = rr_w
        acc_w.round(4).to_csv(os.path.join(args.out_dir, f"acc_{clf}.csv"),
                              index_label="dataset")
        rr_w.round(4).to_csv(os.path.join(args.out_dir, f"rr_{clf}.csv"),
                             index_label="dataset")

    # ---- new median-by-classifier summaries ----
    acc_med = median_by_classifier(df, "test_acc_mean", classifiers)
    rr_med  = median_by_classifier(df, "rr", classifiers,
                                   exclude=("full", "random"))
    acc_med.round(4).to_csv(os.path.join(args.out_dir,
                                         "acc_median_by_classifier.csv"),
                            index_label="method")
    rr_med.round(4).to_csv(os.path.join(args.out_dir,
                                        "rr_median_by_classifier.csv"),
                           index_label="method")

    # ---- LaTeX ----
    parts = [
        "% Requires: \\usepackage{booktabs} \\usepackage{graphicx}",
        "",
        latex_summary_table(
            acc_med,
            "Median test accuracy across 18 datasets, per classifier.",
            "tab:acc-median-by-classifier",
            higher_is_better=True,
        ),
        "",
        latex_summary_table(
            rr_med,
            "Median Relative Regret across 18 datasets, per classifier "
            "(lower is better).",
            "tab:rr-median-by-classifier",
            higher_is_better=False,
        ),
        "",
    ]
    for clf in classifiers:
        parts.append(latex_wide_table(
            wide_tables[("acc", clf)].round(4),
            f"Per-dataset test accuracy with the {CLF_NAME[clf]} classifier.",
            f"tab:acc-{clf}",
        ))
        parts.append("")
    for clf in classifiers:
        parts.append(latex_wide_table(
            wide_tables[("rr", clf)].round(4),
            f"Per-dataset Relative Regret with the {CLF_NAME[clf]} classifier "
            f"(lower is better).",
            f"tab:rr-{clf}",
        ))
        parts.append("")
    with open(os.path.join(args.out_dir, "all_tables.tex"), "w") as f:
        f.write("\n".join(parts))

    print("\nSaved to", args.out_dir)
    for name in ["acc_median_by_classifier.csv", "rr_median_by_classifier.csv",
                 "all_tables.tex"]:
        print(f"  {name}")
    print("  (plus the 6 per-classifier acc_*/rr_*.csv)")


if __name__ == "__main__":
    main()