"""
TDBench dataset loaders.

Adds the 16 datasets from TDBench (Kang et al. 2024) that aren't already in
your benchmark, so we can do a head-to-head comparison on their full set
of 23 datasets.

Datasets added:
  two_d_planes, amazon_employee_access, click_prediction_small,
  diabetes (Diabetes130US), credit_default, elevators, hcdr (home_equity),
  house (house_16H), jannis, law_school_admissions, mini_boo_ne,
  numer_ai, nursery, pol, road_safety, medical_appointments

Already in your benchmark (just maps to existing prepare functions):
  adult, bank_marketing, credit, electricity, higgs, magic_telescope,
  phishing_websites

Usage:
  from data.tdbench_datasets import register_tdbench_datasets, TDBENCH_NAMES
  register_tdbench_datasets()  # adds them to DATASET_REGISTRY

  # Then prepare_db works for any of them:
  data = prepare_db(config, name="jannis")
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.prepare_database import DATASET_REGISTRY


# We use the openml package directly because sklearn's fetch_openml hits
# 301 redirect loops with newer OpenML domains.
try:
    import openml
    _HAS_OPENML = True
except ImportError:
    _HAS_OPENML = False
    from sklearn.datasets import fetch_openml


# --- the 23 datasets ---
TDBENCH_NAMES = [
    "two_d_planes", "amazon_employee_access", "bank_marketing",
    "click_prediction_small", "diabetes_130us", "credit_default",
    "magic_telescope", "medical_appointments", "mini_boo_ne",
    "phishing_websites", "adult", "credit", "electricity",
    "elevators", "hcdr", "higgs", "house", "jannis",
    "law_school_admissions", "numer_ai", "nursery", "pol", "road_safety",
]


# --- shared preprocessing (mirrors your existing pipeline) ---

def _preprocess_and_split(X_df, y, random_seed, device,
                          test_size=0.15, val_size=0.15,
                          one_hot_threshold=10):
    """Same preprocessing pipeline as your other datasets:
       - one-hot low-cardinality categoricals
       - integer-code high-cardinality categoricals
       - standardize all features
       - 70/15/15 stratified split
       - map labels to {0,...,C-1}
    """
    # encode categoricals
    X_df = X_df.copy()
    for col in X_df.columns:
        if X_df[col].dtype.name in ("category", "object", "bool"):
            X_df[col] = X_df[col].astype("category")
            n_unique = X_df[col].nunique()
            if n_unique <= one_hot_threshold:
                # one-hot
                dummies = pd.get_dummies(X_df[col], prefix=col, dummy_na=False)
                X_df = X_df.drop(columns=[col])
                X_df = pd.concat([X_df, dummies], axis=1)
            else:
                # integer codes
                X_df[col] = X_df[col].cat.codes.astype(np.float32)
        else:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = X_df.values.astype(np.float32)

    # encode labels to 0..C-1
    if y.dtype.name in ("category", "object", "bool"):
        y = pd.Categorical(y).codes
    y = np.asarray(y, dtype=np.int64)

    # 70/15/15 stratified split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=test_size + val_size,
        stratify=y, random_state=random_seed,
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=test_size / (test_size + val_size),
        stratify=y_tmp, random_state=random_seed,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    feature_dim = X_tr.shape[1]
    n_classes = int(len(np.unique(y)))

    return {
        "X_train": torch.tensor(X_tr, device=device, dtype=torch.float32),
        "y_train": torch.tensor(y_tr, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device, dtype=torch.float32),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_te, device=device, dtype=torch.float32),
        "y_test":  torch.tensor(y_te, device=device, dtype=torch.long),
        "input_dim": feature_dim,
        "num_classes": n_classes,
    }


def _fetch_openml_safe(openml_id):
    """Fetch from OpenML, returns (X_df, y_series).

    Tries the `openml` package first (more reliable with current API),
    falls back to sklearn.datasets.fetch_openml.
    """
    if _HAS_OPENML:
        ds = openml.datasets.get_dataset(
            openml_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        X_df, y, _, _ = ds.get_data(
            target=ds.default_target_attribute,
            dataset_format="dataframe",
        )
        return X_df, y
    else:
        d = fetch_openml(data_id=openml_id, as_frame=True)
        return d.data, d.target


# --- the 16 new TDBench-only datasets ---

def prepare_two_d_planes(random_seed=42, device="cpu"):
    print("Loading TwoDPlanes (OpenML id=727) ...")
    X_df, y = _fetch_openml_safe(727)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_amazon_employee_access(random_seed=42, device="cpu"):
    print("Loading AmazonEmployeeAccess (OpenML id=4135) ...")
    X_df, y = _fetch_openml_safe(4135)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_click_prediction_small(random_seed=42, device="cpu"):
    print("Loading ClickPredictionSmall (OpenML id=1220) ...")
    X_df, y = _fetch_openml_safe(1220)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_diabetes_130us(random_seed=42, device="cpu"):
    print("Loading Diabetes130US (OpenML id=45022) ...")
    X_df, y = _fetch_openml_safe(45022)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_credit_default(random_seed=42, device="cpu"):
    print("Loading CreditDefault (OpenML id=45020) ...")
    X_df, y = _fetch_openml_safe(45020)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_medical_appointments(random_seed=42, device="cpu"):
    print("Loading MedicalAppointments (OpenML id=43439) ...")
    X_df, y = _fetch_openml_safe(43439)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_mini_boo_ne(random_seed=42, device="cpu"):
    print("Loading MiniBooNE (OpenML id=44088) ...")
    X_df, y = _fetch_openml_safe(44088)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_elevators(random_seed=42, device="cpu"):
    print("Loading Elevators (OpenML id=846) ...")
    X_df, y = _fetch_openml_safe(846)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_hcdr(random_seed=42, device="cpu"):
    print("Loading HomeEquityCredit (OpenML id=45071) ...")
    X_df, y = _fetch_openml_safe(45071)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_house(random_seed=42, device="cpu"):
    print("Loading House16H (OpenML id=821) ...")
    X_df, y = _fetch_openml_safe(821)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_jannis(random_seed=42, device="cpu"):
    print("Loading Jannis (OpenML id=45021) ...")
    X_df, y = _fetch_openml_safe(45021)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_law_school_admissions(random_seed=42, device="cpu"):
    print("Loading LawSchoolAdmissions (OpenML id=43890) ...")
    X_df, y = _fetch_openml_safe(43890)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_numer_ai(random_seed=42, device="cpu"):
    print("Loading NumerAI (OpenML id=23517) ...")
    X_df, y = _fetch_openml_safe(23517)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_nursery(random_seed=42, device="cpu"):
    print("Loading Nursery (OpenML id=959) ...")
    X_df, y = _fetch_openml_safe(959)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_pol(random_seed=42, device="cpu"):
    print("Loading Pol (OpenML id=722) ...")
    X_df, y = _fetch_openml_safe(722)
    return _preprocess_and_split(X_df, y, random_seed, device)


def prepare_road_safety(random_seed=42, device="cpu"):
    print("Loading RoadSafety (OpenML id=44161) ...")
    X_df, y = _fetch_openml_safe(44161)
    return _preprocess_and_split(X_df, y, random_seed, device)


# --- registration ---

NEW_TDBENCH_LOADERS = {
    "two_d_planes":          prepare_two_d_planes,
    "amazon_employee_access": prepare_amazon_employee_access,
    "click_prediction_small": prepare_click_prediction_small,
    "diabetes_130us":         prepare_diabetes_130us,
    "credit_default":         prepare_credit_default,
    "medical_appointments":   prepare_medical_appointments,
    "mini_boo_ne":            prepare_mini_boo_ne,
    "elevators":              prepare_elevators,
    "hcdr":                   prepare_hcdr,
    "house":                  prepare_house,
    "jannis":                 prepare_jannis,
    "law_school_admissions":  prepare_law_school_admissions,
    "numer_ai":               prepare_numer_ai,
    "nursery":                prepare_nursery,
    "pol":                    prepare_pol,
    "road_safety":            prepare_road_safety,
}

# Maps from TDBench dataset names to the names already in your registry
EXISTING_TDBENCH_ALIASES = {
    "bank_marketing": "bank",
    "magic_telescope": "magic",
    "phishing_websites": "phishing",
}


def register_tdbench_datasets():
    """Register the 16 new datasets and aliases into your DATASET_REGISTRY."""
    for name, fn in NEW_TDBENCH_LOADERS.items():
        DATASET_REGISTRY[name] = fn

    # add aliases pointing to your existing functions
    for alias, existing in EXISTING_TDBENCH_ALIASES.items():
        if existing in DATASET_REGISTRY and alias not in DATASET_REGISTRY:
            DATASET_REGISTRY[alias] = DATASET_REGISTRY[existing]
