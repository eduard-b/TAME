import os
import pandas as pd
import numpy as np

IN_PATH  = r"E:/doctorat/Tabular-DD/final_results/K_MEANS_REPEAT/acc_mlp_IPC50.csv"
OUT_PATH = r"E:/doctorat/Tabular-DD/final_results/K_MEANS_REPEAT/acc_mlp_IPC50_RL.csv"

GLOBAL_METHODS = ["node", "dcnv2", "ln_res_l"]  # case-insensitive

def main():
    df = pd.read_csv(IN_PATH)

    # Use dataset as index if present
    if "dataset" in df.columns:
        df = df.set_index("dataset")

    # Normalize column names
    cols_lower = {c.lower(): c for c in df.columns}

    if "full" not in cols_lower:
        raise ValueError(f"Missing 'full' column in {IN_PATH}. Found: {list(df.columns)}")

    full_col = cols_lower["full"]

    # Force numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Methods = everything except full
    method_cols = [c for c in df.columns if c != full_col]

    # ---------- Compute RL ----------
    rl = pd.DataFrame(index=df.index, columns=method_cols, dtype=float)
    for m in method_cols:
        rl[m] = df[full_col] - df[m]

    # ---------- Per-method summary rows ----------
    mean_row   = rl.mean(axis=0, skipna=True)
    std_row    = rl.std(axis=0, skipna=True)
    median_row = rl.median(axis=0, skipna=True)

    # ---------- Global summaries (NODE / DCNv2 / LN_RES_L) ----------
    rl_cols_lower = {c.lower(): c for c in rl.columns}
    global_cols = [rl_cols_lower[m] for m in GLOBAL_METHODS if m in rl_cols_lower]

    global_mean = np.nan
    global_median = np.nan
    if len(global_cols) > 0:
        pooled = rl[global_cols].to_numpy().ravel()
        pooled = pooled[~np.isnan(pooled)]
        if pooled.size > 0:
            global_mean = float(np.mean(pooled))
            global_median = float(np.median(pooled))

    global_mean_row = pd.Series(np.nan, index=rl.columns, dtype=float)
    global_median_row = pd.Series(np.nan, index=rl.columns, dtype=float)

    for c in global_cols:
        global_mean_row[c] = global_mean
        global_median_row[c] = global_median

    # ---------- Concatenate ----------
    rl_out = pd.concat(
        [
            rl,
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
    rl_out.to_csv(OUT_PATH, index=True)

    print(f"Saved RL CSV: {OUT_PATH}")
    print(f"Methods: {method_cols}")
    print(f"Global methods found: {global_cols}")
    print(f"GLOBAL_MEAN={global_mean:.4f}, GLOBAL_MEDIAN={global_median:.4f}")

if __name__ == "__main__":
    main()
