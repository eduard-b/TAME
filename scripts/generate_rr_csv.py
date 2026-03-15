import os
import pandas as pd
import numpy as np

IN_PATH  = r"E:/doctorat/Tabular-DD/final_results/K_MEANS_REPEAT/acc_mlp_IPC50.csv"
OUT_PATH = r"E:/doctorat/Tabular-DD/final_results/K_MEANS_REPEAT/acc_mlp_IPC50_RR.csv"

GLOBAL_METHODS = ["node", "dcnv2", "ln_res_l"]  # case-insensitive targets

def main():
    df = pd.read_csv(IN_PATH)

    # Use "dataset" column as index if present
    if "dataset" in df.columns:
        df = df.set_index("dataset")

    # Normalize column names for robust lookup
    cols_lower = {c.lower(): c for c in df.columns}

    if "full" not in cols_lower:
        raise ValueError(f"Missing 'full' column in {IN_PATH}. Found columns: {list(df.columns)}")
    if "random" not in cols_lower:
        raise ValueError(f"Missing 'random' column in {IN_PATH}. Found columns: {list(df.columns)}")

    full_col = cols_lower["full"]
    rand_col = cols_lower["random"]

    # Force numeric conversion
    df = df.apply(pd.to_numeric, errors="coerce")

    A_full = df[full_col]
    A_rand = df[rand_col]
    denom  = (A_full - A_rand)

    # Methods to compute RR for: all except full and random
    method_cols = [c for c in df.columns if c not in {full_col, rand_col}]
    rr = pd.DataFrame(index=df.index, columns=method_cols, dtype=float)

    # Compute RR per method
    for m in method_cols:
        rr[m] = (A_full - df[m]) / denom

    # Undefined when A_full == A_rand
    bad_rows = np.isclose(denom.to_numpy(), 0.0)
    rr.loc[bad_rows, :] = np.nan

    # ---------- Per-method summary rows ----------
    mean_row   = rr.mean(axis=0, skipna=True)
    std_row    = rr.std(axis=0, skipna=True)
    median_row = rr.median(axis=0, skipna=True)

    # ---------- Global summaries (only selected methods) ----------
    # Find which columns match the requested global set (case-insensitive)
    rr_cols_lower = {c.lower(): c for c in rr.columns}
    global_cols = [rr_cols_lower[m] for m in GLOBAL_METHODS if m in rr_cols_lower]

    # Compute global mean/median over all dataset values pooled across those columns
    global_mean = np.nan
    global_median = np.nan
    if len(global_cols) > 0:
        pooled = rr[global_cols].to_numpy().ravel()
        pooled = pooled[~np.isnan(pooled)]
        if pooled.size > 0:
            global_mean = float(np.mean(pooled))
            global_median = float(np.median(pooled))

    # Build rows (NaN everywhere, fill only the desired columns for globals)
    global_mean_row = pd.Series(np.nan, index=rr.columns, dtype=float)
    global_median_row = pd.Series(np.nan, index=rr.columns, dtype=float)

    for c in global_cols:
        global_mean_row[c] = global_mean
        global_median_row[c] = global_median

    # Append rows in the order you requested
    rr_out = pd.concat(
        [
            rr,
            pd.DataFrame([mean_row], index=["MEAN"]),
            pd.DataFrame([std_row], index=["STD"]),
            pd.DataFrame([median_row], index=["MEDIAN"]),
            pd.DataFrame([global_mean_row], index=["GLOBAL_MEAN"]),
            pd.DataFrame([global_median_row], index=["GLOBAL_MEDIAN"]),
        ],
        axis=0,
    )

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rr_out.to_csv(OUT_PATH, index=True)

    print(f"Saved RR CSV: {OUT_PATH}")
    print(f"Computed RR for methods: {method_cols}")
    print(f"Global methods found: {global_cols}")
    print(f"GLOBAL_MEAN={global_mean:.4f}, GLOBAL_MEDIAN={global_median:.4f}")

if __name__ == "__main__":
    main()
