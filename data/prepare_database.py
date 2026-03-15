import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from scipy import sparse

def debug_dataset_snapshot(name, X, y, max_uniques=20):
    import numpy as np

    print(f"\n=== DEBUG SNAPSHOT: {name} ===")

    # X shape / type
    try:
        x_shape = X.shape
    except Exception:
        x_shape = None
    print("X type:", type(X), "| shape:", x_shape)

    # If pandas DataFrame
    if hasattr(X, "dtypes"):
        print("X dtypes counts:\n", X.dtypes.value_counts())
        # show problematic columns
        obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["category"]).columns.tolist()
        bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
        print("object cols:", obj_cols[:10], ("(+more)" if len(obj_cols) > 10 else ""))
        print("category cols:", cat_cols[:10], ("(+more)" if len(cat_cols) > 10 else ""))
        print("bool cols:", bool_cols[:10], ("(+more)" if len(bool_cols) > 10 else ""))

        # quick peek into first few values of object/category cols
        peek_cols = (obj_cols + cat_cols)[:5]
        for c in peek_cols:
            vals = X[c].astype(str).str.lower().value_counts().head(5)
            print(f"Top values for col '{c}':\n{vals}")

    # y summary
    if hasattr(y, "to_numpy"):
        y_np = y.to_numpy()
    else:
        y_np = np.asarray(y)

    print("y type:", type(y), "| y dtype:", getattr(y_np, "dtype", None))
    uniques = np.unique(y_np)
    print("y uniques (head):", uniques[:max_uniques], "| count:", len(uniques))

    # y counts (works for strings too)
    # robust bincount for numeric only; otherwise use dict counts
    try:
        y_int = y_np.astype(np.int64)
        counts = np.bincount(y_int - y_int.min())  # safe shift
        print("y numeric counts (shifted):", counts, "| min count:", int(counts.min()))
    except Exception:
        from collections import Counter
        c = Counter(y_np.tolist())
        smallest = sorted(c.items(), key=lambda kv: kv[1])[:10]
        print("y smallest classes:", smallest)

def openml_sanity(name, X, y, expected=None, show_cols=8):
    """
    expected: dict like:
      {
        "n_rows_min": 10000,
        "n_features_min": 10,
        "n_classes_min": 2,
        "n_classes_max": 30,
        "must_have_cols": ["..."],  # optional
      }
    """
    import numpy as np

    print(f"\n=== SANITY: {name} ===")

    # X summary
    if hasattr(X, "shape"):
        print("X shape:", X.shape, "| X type:", type(X))
    else:
        print("X type:", type(X))

    # Column names peek if DataFrame
    if hasattr(X, "columns"):
        cols = list(X.columns)
        print("X columns (head):", cols[:show_cols], ("(+more)" if len(cols) > show_cols else ""))
        print("X dtypes counts:\n", X.dtypes.value_counts())

    # y summary
    if hasattr(y, "to_numpy"):
        y_np = y.to_numpy()
    else:
        y_np = np.asarray(y)

    uniq = np.unique(y_np)
    print("y dtype:", getattr(y_np, "dtype", None), "| #classes:", len(uniq), "| uniques(head):", uniq[:10])

    # class counts (robust)
    try:
        y_int = y_np.astype(np.int64)
        # if labels aren't 0..K-1, shift for bincount
        y_shift = y_int - y_int.min()
        counts = np.bincount(y_shift)
        print("class counts (shifted):", counts, "| min:", int(counts.min()), "max:", int(counts.max()))
    except Exception:
        from collections import Counter
        c = Counter(y_np.tolist())
        smallest = sorted(c.items(), key=lambda kv: kv[1])[:5]
        largest = sorted(c.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("smallest classes:", smallest)
        print("largest classes:", largest)

    # expectations
    if expected:
        ok = True
        if "n_rows_min" in expected and hasattr(X, "shape"):
            if X.shape[0] < expected["n_rows_min"]:
                print(f"[WARN] rows {X.shape[0]} < expected min {expected['n_rows_min']}")
                ok = False
        if "n_features_min" in expected and hasattr(X, "shape"):
            if X.shape[1] < expected["n_features_min"]:
                print(f"[WARN] feats {X.shape[1]} < expected min {expected['n_features_min']}")
                ok = False
        if "n_classes_min" in expected:
            if len(uniq) < expected["n_classes_min"]:
                print(f"[WARN] classes {len(uniq)} < expected min {expected['n_classes_min']}")
                ok = False
        if "n_classes_max" in expected:
            if len(uniq) > expected["n_classes_max"]:
                print(f"[WARN] classes {len(uniq)} > expected max {expected['n_classes_max']}")
                ok = False
        if "must_have_cols" in expected and hasattr(X, "columns"):
            missing = [c for c in expected["must_have_cols"] if c not in X.columns]
            if missing:
                print(f"[WARN] missing expected columns: {missing}")
                ok = False

        if ok:
            print("[OK] sanity checks passed.")

def print_class_distribution(name, y):
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    print(f"\n=== Class Distribution: {name} ===")
    for c in np.unique(y_np):
        print(f"  Class {c}: {np.sum(y_np == c)} samples")

def stratified_train_val_test_split(X, y, test_size=0.3, val_size=0.5, seed=42, max_tries=20):
    rng = np.random.RandomState(seed)

    for attempt in range(max_tries):
        # first: train vs temp
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rng.randint(0, 10**6))
        train_idx, temp_idx = next(sss1.split(X, y))

        # then: val vs test from temp
        y_temp = y[temp_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=rng.randint(0, 10**6))
        val_idx, test_idx = next(sss2.split(X[temp_idx], y_temp))

        y_train, y_val, y_test = y[train_idx], y_temp[val_idx], y_temp[test_idx]

        # ensure all splits contain both classes
        if (np.unique(y_train).shape[0] == 2 and
            np.unique(y_val).shape[0] == 2 and
            np.unique(y_test).shape[0] == 2):
            X_train = X[train_idx]
            X_val   = X[temp_idx][val_idx]
            X_test  = X[temp_idx][test_idx]
            return X_train, X_val, X_test, y_train, y_val, y_test

    # If we fail max_tries times, just fall back to the last attempt
    print("[prepare_bank] WARNING: could not guarantee both classes in all splits.")
    X_train = X[train_idx]
    X_val   = X[temp_idx][val_idx]
    X_test  = X[temp_idx][test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_credit_default(random_seed=42, device="cpu"):
    print("Loading Credit Default dataset...")

    credit = fetch_openml("default-of-credit-card-clients", version=1, as_frame=True)
    df = credit.frame

    # Target is simply "y"
    y = df["y"].astype(np.float32).values
    X_df = df.drop(columns=["y"])

    # No real categorical columns preserved, so no one-hot:
    # OpenML version is already numeric-encoded!
    X = X_df.values.astype(np.float32)
    feature_dim = X.shape[1]

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_seed, stratify=y_temp
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device),
        "X_val":   torch.tensor(X_val,   device=device),
        "y_val":   torch.tensor(y_val,   device=device),
        "X_test":  torch.tensor(X_test,  device=device),
        "y_test":  torch.tensor(y_test,  device=device),
        "input_dim": feature_dim,
        "num_classes": 2,
    }

def prepare_bank_marketing(random_seed=42, device="cpu"):
    """
    Loads the Bank Marketing dataset from OpenML.
    Handles both textual and numeric label formats.
    Returns normalized tensors ready for the DM pipeline.
    """

    print("Loading Bank Marketing dataset ...")
    bank = fetch_openml("bank-marketing", version=1, as_frame=True)

    X_df = bank.data
    y_series = bank.target

    # -----------------------------------------------------
    # FIX: Robust label parser (handles yes/no OR 1/2)
    # -----------------------------------------------------
    raw = y_series.astype(str)
    uniques = set(np.unique(raw))
    print("Raw label values:", uniques)

    # Text labels ("yes"/"no")
    if uniques <= {"yes", "no"}:
        print("Detected yes/no → mapping: yes=1, no=0")
        y = (raw == "yes").astype(int).values

    # Numeric labels ("1"/"2")
    elif uniques <= {"1", "2"}:
        print("Detected numeric labels 1/2 → mapping: 2=1, 1=0")
        y = (raw == "2").astype(int).values

    else:
        raise ValueError(f"Unknown bank-marketing label format: {uniques}")

    # -----------------------------------------------------
    # Handle categorical features → One-hot encode
    # -----------------------------------------------------
    print("One-hot encoding categorical features...")
    X_df = pd.get_dummies(X_df, drop_first=False)
    X = X_df.values.astype(np.float32)
    input_dim = X.shape[1]
    num_classes = 2

    print(f"Bank Marketing: {len(X)} samples | dim = {input_dim}")

    # -----------------------------------------------------
    # Train/Val/Test split (stratified)
    # -----------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_seed, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
    )

    # Show class balance
    print("Train class distribution:", np.bincount(y_train))
    print("Val class distribution:  ", np.bincount(y_val))
    print("Test class distribution: ", np.bincount(y_test))

    # -----------------------------------------------------
    # Standardize features
    # -----------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # -----------------------------------------------------
    # Convert to tensors
    # -----------------------------------------------------
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)

    X_val_t   = torch.tensor(X_val,   device=device)
    y_val_t   = torch.tensor(y_val,   device=device)

    X_test_t  = torch.tensor(X_test,  device=device)
    y_test_t  = torch.tensor(y_test,  device=device)

    # -----------------------------------------------------
    # Output standardized dictionary
    # -----------------------------------------------------
    return {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_val":   X_val_t,
        "y_val":   y_val_t,
        "X_test":  X_test_t,
        "y_test":  y_test_t,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": ["no", "yes"],  # your canonical mapping
    }

def prepare_adult(random_seed=42, device="cpu"):
    print("Loading Adult dataset (OpenML)...")
    adult = fetch_openml("adult", version=2, as_frame=True)

    X_df: pd.DataFrame = adult.data
    y_series: pd.Series = adult.target

    y = (y_series.astype(str).str.contains(">50K")).astype(np.float32).values

    print("One-hot encoding...")
    X_df = pd.get_dummies(X_df, drop_first=False)
    X = X_df.values.astype(np.float32)

    feature_dim = X.shape[1]
    print(f"Feature dim after one-hot = {feature_dim}")

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_seed, stratify=y_temp
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # To tensors
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_val_t   = torch.tensor(X_val,   device=device)
    y_val_t   = torch.tensor(y_val,   device=device)
    X_test_t  = torch.tensor(X_test,  device=device)
    y_test_t  = torch.tensor(y_test,  device=device)

    return {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_test": X_test_t,
        "y_test": y_test_t,
        "input_dim": feature_dim,
        "num_classes": 2,
    }

def prepare_airlines_optimized(random_seed=42, device="cpu"):
    print("Loading Airlines dataset (OpenML id=1169)...")
    ds = fetch_openml(data_id=1169, as_frame=True)
    df = ds.frame.copy()

    # TARGET
    y = (df["Delay"].astype(str) == "1").astype(np.int64).values
    df = df.drop(columns=["Delay"])

    # CATEGORICAL SPLIT
    low_card = ["Airline", "DayOfWeek"]
    high_card = ["Flight", "AirportFrom", "AirportTo"]

    # LOW CARD = ONE-HOT
    df = pd.get_dummies(df, columns=low_card, drop_first=False)

    # HIGH CARD = CATEGORY CODES
    for col in high_card:
        df[col] = df[col].astype("category").cat.codes.astype(np.int32)

    # All remaining columns numeric
    X = df.values.astype(np.float32)

    print("Final feature dim:", X.shape[1])

    # SPLIT + SCALE
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # TORCH
    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": X_train.shape[1],
        "num_classes": 2,
    }

def prepare_higgs(random_seed=42, device="cpu"):
    """
    Loads the HIGGS dataset (OpenML ID 23512).
    Drops NaN rows because KMeans (herding) cannot handle NaN.
    """
    print("Loading HIGGS dataset (OpenML id=23512)...")

    ds = fetch_openml(data_id=23512, as_frame=True)

    df = ds.frame.copy()

    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Drop NaNs early
    df = df.dropna()
    print("After dropna:", df.shape)

    # ---------------------------------------
    # Features and target
    # ---------------------------------------
    y = df["class"].astype(int).values
    X = df.drop(columns=["class"]).values.astype(np.float32)

    print(f"HIGGS: {len(X)} samples | dim={X.shape[1]}")
    print("Target distribution:", np.bincount(y))

    # ---------------------------------------
    # Train/val/test split
    # ---------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # ---------------------------------------
    # Standardize
    # ---------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ---------------------------------------
    # Torch tensors
    # ---------------------------------------
    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.float32),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.float32),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.float32),
        "input_dim": X_train.shape[1],
        "num_classes": 2,
    }

def prepare_spambase(random_seed=42, device="cpu"):
    """
    Spambase (OpenML id=44)
    Binary classification. Mostly clean numeric features.
    Robust to sparse matrices and label dtypes.
    """
    print("Loading Spambase dataset (OpenML id=44)...")

    # parser='auto' helps across sklearn versions; as_frame='auto' avoids sparse-frame issues
    ds = fetch_openml(data_id=44, as_frame="auto", parser="auto")

    X = ds.data
    y = ds.target

    # Convert X to numpy float32 (handle sparse)
    if sparse.issparse(X):
        X = X.toarray()
    else:
        # if X is a pandas DataFrame
        X = np.asarray(X)

    X = X.astype(np.float32)

    # Convert y robustly (could be strings or numeric)
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()

    y = np.asarray(y)

    # Common cases: {0,1} already, or strings like "0"/"1"
    try:
        y = y.astype(np.int64)
    except Exception:
        y = np.array([int(str(v)) for v in y], dtype=np.int64)

    input_dim = X.shape[1]
    num_classes = 2

    print(f"Spambase: {len(X)} samples | dim={input_dim} | y uniques={np.unique(y)}")

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": ["ham", "spam"],
    }

def prepare_credit_g(random_seed=42, device="cpu"):
    """
    German Credit (OpenML: credit-g, data_id=31)
    Binary classification (good/bad).
    One-hot encodes categoricals, standardizes features.
    """
    print("Loading German Credit dataset (OpenML credit-g, id=31)...")

    # Use as_frame=True here; this dataset is not the sparse-arff headache.
    ds = fetch_openml(data_id=31, as_frame=True)

    X_df = ds.data.copy()
    y_series = ds.target

    # --- robust label handling ---
    raw = y_series.astype(str).str.strip().str.lower()
    uniques = set(np.unique(raw))
    print("Raw label values:", uniques)

    if uniques <= {"good", "bad"}:
        y = (raw == "good").astype(np.int64).values
        class_names = ["bad", "good"]
    elif uniques <= {"1", "2"}:
        # Occasionally appears as 1/2 (convention varies, but 1 is typically "good")
        y = (raw == "1").astype(np.int64).values
        class_names = ["other", "one"]
    else:
        raise ValueError(f"Unknown label format for credit-g: {uniques}")

    # --- one-hot encode categoricals ---
    X_df = pd.get_dummies(X_df, drop_first=False)
    X = X_df.values.astype(np.float32)

    input_dim = X.shape[1]
    num_classes = 2
    print(f"credit-g: {len(X)} samples | dim={input_dim}")

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": class_names,
    }

def prepare_magic_telescope(random_seed=42, device="cpu"):
    """
    Magic Gamma Telescope (OpenML: MagicTelescope, data_id=1120)
    Binary classification. Numeric features, label usually {g,h}.
    Robust to sparse matrices and label dtype quirks.
    """
    print("Loading MagicTelescope dataset (OpenML id=1120)...")

    X, y = fetch_openml(
        data_id=1120,
        return_X_y=True,
        as_frame="auto",
        parser="auto",
    )

    # X -> numpy float32
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    X = X.astype(np.float32)

    # y -> int64 (usually 'g'/'h')
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    y = np.asarray(y).astype(str)

    uniques = set(np.unique(y))
    print("Raw label values:", uniques)

    # common: 'g' (gamma) and 'h' (hadron)
    if uniques <= {"g", "h"}:
        y = (y == "g").astype(np.int64)
        class_names = ["hadron", "gamma"]
    # sometimes uppercase or other string forms
    elif uniques <= {"G", "H"}:
        y = (y == "G").astype(np.int64)
        class_names = ["hadron", "gamma"]
    else:
        # try numeric cast fallback
        try:
            y = y.astype(np.int64)
            class_names = [str(v) for v in sorted(np.unique(y))]
        except Exception as e:
            raise ValueError(f"Unknown MagicTelescope label format: {uniques}") from e

    input_dim = X.shape[1]
    num_classes = 2

    print(f"MagicTelescope: {len(X)} samples | dim={input_dim}")

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": class_names,
    }

def prepare_phishing_websites(random_seed=42, device="cpu"):
    """
    PhishingWebsites (OpenML data_id=4534)
    Binary classification.
    Robust: handles label formats and categorical/numeric mix.
    """

    print("Loading PhishingWebsites dataset (OpenML id=4534)...")

    ds = fetch_openml(data_id=4534, as_frame=True)

    X_df = ds.data.copy()
    y = ds.target

    # --- y robust conversion ---
    y = y.astype(str).str.strip()
    uniques = set(np.unique(y))
    print("Raw label values:", uniques)

    # Common variants: {-1, 1} or {0, 1} or strings
    if uniques <= {"-1", "1"}:
        y = (y == "1").astype(np.int64).values
        class_names = ["phishing(-1)", "legit(1)"]
    elif uniques <= {"0", "1"}:
        y = (y == "1").astype(np.int64).values
        class_names = ["0", "1"]
    else:
        # fallback: try numeric cast
        try:
            y_num = y.astype(float)
            # map to {0,1} if it's {-1,1}
            if set(np.unique(y_num)) <= {-1.0, 1.0}:
                y = (y_num == 1.0).astype(np.int64)
                class_names = ["phishing(-1)", "legit(1)"]
            else:
                # if already 0/1-ish
                y = (y_num > 0).astype(np.int64)
                class_names = ["nonpos", "pos"]
        except Exception as e:
            raise ValueError(f"Unknown label format for PhishingWebsites: {uniques}") from e

    # --- X handling ---
    # If OpenML gave object/category columns, one-hot them; if already numeric, get_dummies is harmless.
    X_df = pd.get_dummies(X_df, drop_first=False)

    X = X_df.values
    if sparse.issparse(X):
        X = X.toarray()

    X = np.asarray(X).astype(np.float32)

    input_dim = X.shape[1]
    num_classes = 2
    print(f"PhishingWebsites: {len(X)} samples | dim={input_dim}")

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": class_names,
    }

def prepare_letter_recognition(random_seed=42, device="cpu"):
    """
    Letter Recognition (OpenML: 'letter', data_id=6)
    Multiclass classification with 26 classes (A-Z), 16 numeric features.
    """

    print("Loading Letter Recognition dataset (OpenML id=6)...")

    X, y = fetch_openml(
        data_id=6,
        return_X_y=True,
        as_frame="auto",
        parser="auto",
    )

    # X -> numpy float32
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    X = X.astype(np.float32)

    # y -> 0..25 (A..Z)
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    y = np.asarray(y).astype(str)

    # Common: 'A'..'Z'
    uniques = np.unique(y)
    print("Raw label sample:", uniques[:10], "(count:", len(uniques), ")")

    # Map sorted unique labels -> 0..K-1
    classes_sorted = sorted(list(set(uniques)))
    cls_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    y_idx = np.array([cls_to_idx[v] for v in y], dtype=np.int64)

    num_classes = len(classes_sorted)
    input_dim = X.shape[1]

    print(f"Letter: {len(X)} samples | dim={input_dim} | classes={num_classes}")

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_idx, test_size=0.30, stratify=y_idx, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "class_names": classes_sorted,  # should be A..Z
        "class_mapping": cls_to_idx,
    }

def prepare_shuttle(random_seed=42, device="cpu"):
    """
    Shuttle (OpenML data_id=62)
    Multiclass classification. Robust to boolean-like strings and categorical dtypes.
    """

    print("Loading Shuttle dataset (OpenML id=62)...")

    X, y = fetch_openml(name="shuttle", version=1, return_X_y=True, as_frame="auto", parser="auto")

    X_df = X.copy()

    # --- handle boolean dtype ---
    bool_cols = X_df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X_df[bool_cols] = X_df[bool_cols].astype(np.float32)

    # --- handle object + category columns (where 'true'/'false' may live) ---
    cat_cols = X_df.select_dtypes(include=["category"]).columns
    obj_cols = X_df.select_dtypes(include=["object"]).columns
    cols_to_fix = list(cat_cols) + list(obj_cols)

    bool_map = {"true": 1.0, "false": 0.0, "t": 1.0, "f": 0.0, "yes": 1.0, "no": 0.0}

    for c in cols_to_fix:
        # Work in string space first
        s = X_df[c].astype(str).str.strip().str.lower()

        # Map common boolean-like tokens
        s2 = s.map(bool_map).where(s.isin(bool_map.keys()), s)

        # Try numeric coercion
        num = pd.to_numeric(s2, errors="coerce")

        if num.notna().all():
            # Fully numeric after coercion
            X_df[c] = num.astype(np.float32)
        else:
            # Not fully numeric: fall back to categorical codes
            X_df[c] = s.astype("category").cat.codes.astype(np.float32)

    # Any remaining NaNs? Fill with column means
    if X_df.isna().any().any():
        X_df = X_df.fillna(X_df.mean(numeric_only=True))

    # Now safe
    X_np = X_df.to_numpy(dtype=np.float32)

    # --- y handling ---
    y = y.astype(str).str.strip()
    try:
        y_num = y.astype(np.int64).to_numpy()
    except Exception:
        # fallback: categorical encoding
        y_num = y.astype("category").cat.codes.to_numpy(dtype=np.int64)

    # Remap to 0..K-1 (safe for DM)
    classes_sorted = sorted(np.unique(y_num).tolist())
    cls_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    y_np = np.array([cls_to_idx[v] for v in y_num], dtype=np.int64)

    num_classes = len(classes_sorted)
    input_dim = X_np.shape[1]

    print(f"Shuttle: {len(X_np)} samples | dim={input_dim} | classes={num_classes}")
    binc = np.bincount(y_np)
    print("Class counts:", binc, "| min:", int(binc.min()), "max:", int(binc.max()))

    # --- split ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_np, y_np, test_size=0.30, stratify=y_np, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # --- scale ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "class_names": [f"class_{i}" for i in range(num_classes)],
        "class_mapping": cls_to_idx,
    }

def prepare_electricity(random_seed=42, device="cpu", debug=True):
    """
    Electricity dataset (OpenML id=151)
    Binary classification.
    """

    print("Loading Electricity dataset (OpenML id=151)...")

    ds = fetch_openml(
        data_id=151,
        as_frame=True,
        parser="auto",
    )

    if debug:
        print("[OpenML]", ds.details["id"], ds.details["name"], ds.details["version"])
        print("Target name:", ds.target.name)
        print("X shape:", ds.data.shape)
        print("y value counts:\n", ds.target.value_counts().head())

    X_df = ds.data.copy()
    y = ds.target.copy()

        # --- label handling ---
    y = y.astype(str).str.strip().str.lower()
    uniques = set(y.unique())
    print("Raw label values:", uniques)

    # Common labels: "up" / "down"
    if uniques <= {"up", "down"}:
        y = (y == "up").astype(np.int64).to_numpy()
        class_names = ["down", "up"]
    else:
        # fallback numeric
        y = y.astype(np.int64).to_numpy()
        class_names = ["class_0", "class_1"]

    # --- feature handling ---
    X_df = pd.get_dummies(X_df, drop_first=False)

    X = X_df.to_numpy(dtype=np.float32)

    input_dim = X.shape[1]
    num_classes = 2

    print(f"Electricity: {len(X)} samples | dim={input_dim}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "class_names": class_names,
    }

def prepare_madelon(random_seed=42, device="cpu", debug=True):
    """
    Madelon (OpenML id=1485)
    Binary classification, wide features (~500).
    Labels typically in {-1, +1}. We map +1->1, -1->0.
    """

    print("Loading Madelon dataset (OpenML id=1485)...")

    ds = fetch_openml(
        data_id=1485,
        as_frame="auto",
        parser="auto",
    )

    if debug:
        print("[OpenML]", ds.details["id"], ds.details["name"], ds.details["version"])
        try:
            print("Target name:", ds.target.name)
        except Exception:
            pass
        try:
            print("X shape:", ds.data.shape)
            # works if Series
            vc = ds.target.astype(str).value_counts()
            print("y value counts (head):\n", vc.head(10))
        except Exception:
            pass

    X = ds.data
    y = ds.target

    # X -> numpy float32
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    X = X.astype(np.float32)

    # y -> numpy int64 in {0,1}
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    y = np.asarray(y)

    # robust cast
    try:
        y = y.astype(np.int64)
    except Exception:
        y = np.array([int(str(v)) for v in y], dtype=np.int64)

    uniques = sorted(np.unique(y).tolist())
    print("Raw label uniques:", uniques)

    # common: {-1, +1}
    if set(uniques) <= {-1, 1}:
        y = (y == 1).astype(np.int64)
        class_names = ["-1", "+1"]
    # common: {0, 1}
    elif set(uniques) <= {0, 1}:
        y = y.astype(np.int64)
        class_names = ["0", "1"]
    else:
        # fallback: remap arbitrary binary labels
        if len(uniques) != 2:
            raise ValueError(f"Madelon expected binary labels, got uniques={uniques}")
        mapping = {uniques[0]: 0, uniques[1]: 1}
        y = np.array([mapping[v] for v in y], dtype=np.int64)
        class_names = [str(uniques[0]), str(uniques[1])]

    input_dim = X.shape[1]
    num_classes = 2
    print(f"Madelon: {len(X)} samples | dim={input_dim}")

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "class_names": class_names,
    }

def prepare_satimage(random_seed=42, device="cpu", debug=True):
    """
    satimage (OpenML id=182)
    Multiclass classification (typically 6 classes).
    """

    print("Loading satimage dataset (OpenML id=182)...")

    ds = fetch_openml(
        data_id=182,
        as_frame=True,
        parser="auto",
    )

    if debug:
        print("[OpenML]", ds.details["id"], ds.details["name"], ds.details["version"])
        print("Target name:", getattr(ds.target, "name", "unknown"))
        print("X shape:", ds.data.shape)
        print("y value counts (head):\n", ds.target.astype(str).value_counts().head(10))

    X_df = ds.data.copy()
    y = ds.target.copy()

    # --- X: one-hot if needed, then numpy float32 ---
    X_df = pd.get_dummies(X_df, drop_first=False)

    X = X_df.to_numpy(dtype=np.float32)
    input_dim = X.shape[1]

    # --- y: convert to numpy and ALWAYS remap to 0..K-1 ---
    y = y.astype(str).str.strip()
    y_np = y.to_numpy()

    classes_sorted = sorted(pd.unique(y_np).tolist())
    cls_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    y_idx = np.array([cls_to_idx[v] for v in y_np], dtype=np.int64)

    num_classes = len(classes_sorted)

    print(f"satimage: {len(X)} samples | dim={input_dim} | classes={num_classes}")

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_idx, test_size=0.30, stratify=y_idx, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "class_names": classes_sorted,
        "class_mapping": cls_to_idx,
    }

def prepare_pendigits(random_seed=42, device="cpu", debug=True):
    print("Loading pendigits dataset (OpenML id=32)...")

    ds = fetch_openml(data_id=32, as_frame="auto", parser="auto")
    if debug:
        print("[OpenML]", ds.details["id"], ds.details["name"], ds.details["version"])

    X = ds.data
    y = ds.target

    # X -> numpy float32
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    X = X.astype(np.float32)

    # y -> numpy
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    y = np.asarray(y)

    # make labels stable 0..K-1
    y = y.astype(str).astype(str)
    classes_sorted = sorted(np.unique(y).tolist())
    cls_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    y = np.array([cls_to_idx[v] for v in y], dtype=np.int64)

    input_dim = X.shape[1]
    num_classes = len(classes_sorted)
    print(f"pendigits: {len(X)} samples | dim={input_dim} | classes={num_classes}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "class_names": classes_sorted,
        "class_mapping": cls_to_idx,
    }

def prepare_pageblocks(random_seed=42, device="cpu", debug=True):
    print("Loading page-blocks dataset (OpenML id=30)...")

    ds = fetch_openml(data_id=30, as_frame=True, parser="auto")
    if debug:
        print("[OpenML]", ds.details["id"], ds.details["name"], ds.details["version"])
        print("Target:", getattr(ds.target, "name", "unknown"))

    X_df = ds.data.copy()
    y = ds.target.copy()

    # X: one-hot any categoricals (usually already numeric, but harmless)
    X_df = pd.get_dummies(X_df, drop_first=False)
    X = X_df.to_numpy(dtype=np.float32)

    # y: remap to 0..K-1 (string-safe)
    y = y.astype(str).str.strip()
    y_np = y.to_numpy()
    classes_sorted = sorted(pd.unique(y_np).tolist())
    cls_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    y_idx = np.array([cls_to_idx[v] for v in y_np], dtype=np.int64)

    input_dim = X.shape[1]
    num_classes = len(classes_sorted)
    print(f"page-blocks: {len(X)} samples | dim={input_dim} | classes={num_classes}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_idx, test_size=0.30, stratify=y_idx, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "class_names": classes_sorted,
        "class_mapping": cls_to_idx,
    }

def prepare_segment(random_seed=42, device="cpu", debug=True):
    print("Loading segment dataset (OpenML id=36)...")

    ds = fetch_openml(data_id=36, as_frame=True, parser="auto")
    if debug:
        print("[OpenML]", ds.details["id"], ds.details["name"], ds.details["version"])
        print("Target:", getattr(ds.target, "name", "unknown"))

    X_df = ds.data.copy()
    y = ds.target.copy()

    # X: one-hot if needed
    X_df = pd.get_dummies(X_df, drop_first=False)
    X = X_df.to_numpy(dtype=np.float32)

    # y: remap to 0..K-1
    y = y.astype(str).str.strip()
    y_np = y.to_numpy()
    classes_sorted = sorted(pd.unique(y_np).tolist())
    cls_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    y_idx = np.array([cls_to_idx[v] for v in y_np], dtype=np.int64)

    input_dim = X.shape[1]
    num_classes = len(classes_sorted)
    print(f"segment: {len(X)} samples | dim={input_dim} | classes={num_classes}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_idx, test_size=0.30, stratify=y_idx, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "class_names": classes_sorted,
        "class_mapping": cls_to_idx,
    }

def prepare_climate_model_simulation_crashes(random_seed=42, device="cpu", debug=True):
    """
    climate-model-simulation-crashes (OpenML id=1467)
    Binary classification: crash/fail vs success.
    ARFF-safe via as_frame='auto'.
    """

    print("Loading climate-model-simulation-crashes dataset (OpenML id=1467)...")

    ds = fetch_openml(
        data_id=1467,
        as_frame="auto",
        parser="auto",
    )

    if debug:
        det = getattr(ds, "details", {}) or {}
        print("[OpenML]", det.get("id", "?"), det.get("name", "?"), det.get("version", "?"))
        try:
            print("X shape:", ds.data.shape, "| target:", getattr(ds.target, "name", "unknown"))
        except Exception:
            pass

    X = ds.data
    y = ds.target

    # If DataFrame, optionally drop ID columns
    if hasattr(X, "columns"):
        X_df = X.copy()

        # Common structure: 20 cols = 2 identifiers + 18 params
        if X_df.shape[1] == 20:
            if debug:
                print("[INFO] Dropping first two columns as likely identifiers:", list(X_df.columns[:2]))
            X_df = X_df.iloc[:, 2:]

        # Force numeric
        for c in X_df.columns:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

        # Fill any NaNs (should be none per spec, but stay robust)
        if X_df.isna().any().any():
            if debug:
                print("[WARN] NaNs found in X; imputing with column means.")
            X_df = X_df.fillna(X_df.mean(numeric_only=True))

        X_np = X_df.to_numpy(dtype=np.float32)

    else:
        # ndarray / sparse
        if sparse.issparse(X):
            X_np = X.toarray().astype(np.float32)
        else:
            X_np = np.asarray(X).astype(np.float32)

    # y -> numpy int64 in {0,1}
    if hasattr(y, "to_numpy"):
        y_np = y.to_numpy()
    else:
        y_np = np.asarray(y)

    # Handle typical {0,1} or strings
    try:
        y_np = y_np.astype(np.int64)
    except Exception:
        y_str = np.asarray(y_np).astype(str).astype(str)
        # try numeric string
        y_np = np.array([int(v) for v in y_str], dtype=np.int64)

    uniq = sorted(np.unique(y_np).tolist())
    if set(uniq) <= {0, 1}:
        pass
    elif set(uniq) <= {-1, 1}:
        y_np = (y_np == 1).astype(np.int64)
    else:
        # remap any 2 unique labels to 0/1
        if len(uniq) != 2:
            raise ValueError(f"Expected binary labels, got {uniq}")
        mapping = {uniq[0]: 0, uniq[1]: 1}
        y_np = np.array([mapping[v] for v in y_np], dtype=np.int64)

    if debug:
        counts = np.bincount(y_np)
        print("y counts:", counts, "| classes:", np.unique(y_np).tolist())

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_np, y_np, test_size=0.30, stratify=y_np, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": int(X_train.shape[1]),
        "num_classes": 2,
        "class_names": ["fail", "success"],
    }


DATASET_REGISTRY = {}

DATASET_REGISTRY["phishing"] = prepare_phishing_websites
DATASET_REGISTRY["pendigits"] = prepare_pendigits
DATASET_REGISTRY["spambase"] = prepare_spambase
DATASET_REGISTRY["segment"] = prepare_segment
DATASET_REGISTRY["satimage"] = prepare_satimage
DATASET_REGISTRY["bank"] = prepare_bank_marketing
DATASET_REGISTRY["climate"] = prepare_climate_model_simulation_crashes
DATASET_REGISTRY["magic"] = prepare_magic_telescope
DATASET_REGISTRY["pageblocks"] = prepare_pageblocks
DATASET_REGISTRY["electricity"] = prepare_electricity
DATASET_REGISTRY["adult"] = prepare_adult
DATASET_REGISTRY["letter"] = prepare_letter_recognition
DATASET_REGISTRY["shuttle"] = prepare_shuttle
DATASET_REGISTRY["credit"] = prepare_credit_default
DATASET_REGISTRY["german"] = prepare_credit_g
DATASET_REGISTRY["airlines"] = prepare_airlines_optimized
DATASET_REGISTRY["higgs"] = prepare_higgs
DATASET_REGISTRY["madelon"] = prepare_madelon

def prepare_db(config, name):
    seed = config["random_seed"]
    device = config["device"]

    key = name.lower()

    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    prepare_fn = DATASET_REGISTRY[key]

    # Normalize call signature if needed
    try:
        return prepare_fn(random_seed=seed, device=device)
    except TypeError:
        # fallback for functions that use positional args
        return prepare_fn(seed, device)
