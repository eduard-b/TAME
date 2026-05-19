#!/usr/bin/env python3
"""
Head-to-head comparison: TAME vs TDColER on TDBench's 23 datasets.

Runs all three TAME embedder variants (LnRes, DCNv2, NODE) on TDBench's
23 datasets at IPC=10, evaluates with multiple downstream classifiers
(MLP, RF, KNN, LR, NB, XGBoost), and computes Relative Regret using
TDColER's exact formula:

    RR(IPC) = (A_full - A_distilled) / (A_full - A_random@IPC=10)

Note the denominator is FIXED at random@IPC=10, regardless of the distilled
IPC. This is different from TAME's paper where the denominator scales with
IPC.

TDColER's reported best (Table 18, IPC=10, aggregated over classifiers):
    KM (k-means in latent space) + TF-SFT: median RR = 0.4056

Output: results_tdbench_comparison/tdbench_comparison_results.csv
"""

import os
import sys
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.prepare_database import prepare_db, DATASET_REGISTRY
from data.tdbench_datasets import register_tdbench_datasets, TDBENCH_NAMES
from synth.registry import synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier


# ============================================================
# Config
# ============================================================

IPC_DISTILL = 10                  # TDColER's main reporting IPC
IPC_RANDOM_REF = 10               # their RR denominator anchor
NUM_RUNS = 5                      # match their 5-repetition averaging

EMBEDDERS = ["dcnv2_base"]   # all three TAME variants
SYNTH_TYPE = "tame"                          # base TAME, mean+covariance loss

CLASSIFIERS = ["mlp_sci", "rf", "knn", "lr", "nb", "xgboost"]
# Their 7 are: MLP, KNN, LR, NB, XGBoost, ResNet, FT-Transformer.
# We skip ResNet and FT-Transformer (they require per-dataset HPO that
# would be its own engineering project). The 5 above are their non-deep
# classifiers and cover the substance of their comparison.

RESULTS_DIR = "results_tdbench_comparison_dcnv2"
SYNTH_DIR = "synth_outputs_tdbench_dcnv2"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_full_data_acc(data, classifiers, config, device):
    """Train on full training set, evaluate on test set."""
    train_data = {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "input_dim": data["input_dim"],
        "num_classes": data["num_classes"],
    }
    out = {}
    for clf in classifiers:
        clf_config = dict(config)
        clf_config["classifier"] = clf
        model = train_classifier(train_data, clf_config)
        # use balanced accuracy to match TDColER's metric
        try:
            X_test = data["X_test"]
            y_test = data["y_test"].cpu().numpy()
            preds = _predict(model, X_test, clf, device)
            acc = balanced_accuracy_score(y_test, preds)
        except Exception:
            acc, _ = evaluate_classifier(model, data, device)
        out[clf] = float(acc)
    return out


def get_random_ipc_acc(data, ipc, classifiers, config, device, num_runs=5):
    """Train on a randomly-sampled IPC subset, average over num_runs."""
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()
    classes = np.unique(y)

    accs = {clf: [] for clf in classifiers}

    for run_id in range(num_runs):
        rng = np.random.default_rng(42 + run_id)
        idxs = []
        for c in classes:
            class_idx = np.where(y == c)[0]
            n_take = min(ipc, len(class_idx))
            idxs.extend(rng.choice(class_idx, n_take, replace=False))
        idxs = np.array(idxs)

        X_sub = torch.tensor(X[idxs], device=device, dtype=torch.float32)
        y_sub = torch.tensor(y[idxs], device=device, dtype=torch.long)

        train_data = {
            "X_train": X_sub, "y_train": y_sub,
            "X_val": data["X_val"], "y_val": data["y_val"],
            "input_dim": data["input_dim"],
            "num_classes": data["num_classes"],
        }

        for clf in classifiers:
            clf_config = dict(config)
            clf_config["classifier"] = clf
            try:
                model = train_classifier(train_data, clf_config)
                X_test = data["X_test"]
                y_test = data["y_test"].cpu().numpy()
                preds = _predict(model, X_test, clf, device)
                acc = balanced_accuracy_score(y_test, preds)
            except Exception as e:
                print(f"      random {clf} failed: {e}")
                acc = float("nan")
            accs[clf].append(acc)

    return {clf: float(np.nanmean(accs[clf])) for clf in classifiers}


def _predict(model, X_test, clf, device):
    """Get predictions in a classifier-agnostic way (returns numpy class labels)."""
    if hasattr(model, "predict"):
        # sklearn-style
        X_np = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        return model.predict(X_np)
    # torch model
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device).float())
        if logits.dim() == 1 or logits.shape[-1] == 1:
            preds = (torch.sigmoid(logits.view(-1)) >= 0.5).long()
        else:
            preds = logits.argmax(dim=-1)
    return preds.cpu().numpy()


# ============================================================
# Main
# ============================================================

def main():
    register_tdbench_datasets()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_template = {
        "device": device,
        "ipc": IPC_DISTILL,
        "synth_type": SYNTH_TYPE,
        "dm_iters": 1000,
        "dm_lr": 0.5,
        "dm_batch_real": 128,
        "dm_embedder_size": "base",
        "dm_embed_hidden": 256,
        "dm_embed_dim": 48,
        "classifier_hidden": [128, 64],
        "classifier_epochs": 20,
        "random_seed": 42,
        "synth_save_dir": SYNTH_DIR,
    }

    all_rows = []

    for db_name in TDBENCH_NAMES:
        if db_name not in DATASET_REGISTRY:
            print(f"  SKIP {db_name} (not registered)")
            continue

        print(f"\n{'='*70}")
        print(f"  DATASET: {db_name}")
        print(f"{'='*70}")

        config = dict(config_template)
        config["dataset_name"] = db_name

        # --- 1. full and random@IPC=10 baselines (once per dataset) ---
        print(f"\n  [Baselines] full + random@IPC={IPC_RANDOM_REF}")
        try:
            data0 = prepare_db(config, name=db_name)
        except Exception as e:
            print(f"  FAILED to load {db_name}: {e}")
            continue

        full_accs = get_full_data_acc(data0, CLASSIFIERS, config, device)
        rand_accs = get_random_ipc_acc(data0, IPC_RANDOM_REF, CLASSIFIERS,
                                        config, device, num_runs=NUM_RUNS)
        for clf in CLASSIFIERS:
            print(f"    {clf}: full={full_accs[clf]:.4f}  "
                  f"random@{IPC_RANDOM_REF}={rand_accs[clf]:.4f}")

        # --- 2. for each embedder, distill and evaluate ---
        for embedder in EMBEDDERS:
            print(f"\n  [TAME-{embedder}] distilling at IPC={IPC_DISTILL}, "
                  f"{NUM_RUNS} runs ...")

            distilled_accs = {clf: [] for clf in CLASSIFIERS}
            synth_times = []

            for run_id in range(NUM_RUNS):
                run_seed = 42 + run_id
                set_seed(run_seed)
                cfg_run = dict(config)
                cfg_run["random_seed"] = run_seed
                cfg_run["dm_embedder_type"] = embedder

                data = prepare_db(cfg_run, name=db_name)

                t0 = time.time()
                try:
                    X_syn, y_syn = synthesize(synth_type=SYNTH_TYPE,
                                               data=data, config=cfg_run)
                except Exception as e:
                    print(f"      run{run_id} distillation failed: {e}")
                    for clf in CLASSIFIERS:
                        distilled_accs[clf].append(float("nan"))
                    continue
                synth_times.append(time.time() - t0)

                train_data = {
                    "X_train": X_syn, "y_train": y_syn,
                    "X_val": data["X_val"], "y_val": data["y_val"],
                    "input_dim": data["input_dim"],
                    "num_classes": data["num_classes"],
                }
                for clf in CLASSIFIERS:
                    clf_config = dict(cfg_run)
                    clf_config["classifier"] = clf
                    try:
                        model = train_classifier(train_data, clf_config)
                        X_test = data["X_test"]
                        y_test = data["y_test"].cpu().numpy()
                        preds = _predict(model, X_test, clf, device)
                        acc = balanced_accuracy_score(y_test, preds)
                    except Exception as e:
                        print(f"      run{run_id} {clf} failed: {e}")
                        acc = float("nan")
                    distilled_accs[clf].append(acc)
                    print(f"      run{run_id} {clf}: acc={acc:.4f}")

            # --- 3. compute RR for this (dataset, embedder) and save ---
            for clf in CLASSIFIERS:
                if not any(np.isfinite(a) for a in distilled_accs[clf]):
                    continue
                mean_acc = float(np.nanmean(distilled_accs[clf]))
                std_acc = float(np.nanstd(distilled_accs[clf]))

                full = full_accs[clf]
                rand10 = rand_accs[clf]

                denom = full - rand10
                rr_tdcoler = ((full - mean_acc) / denom
                              if abs(denom) > 1e-6 else float("nan"))

                row = {
                    "dataset": db_name,
                    "embedder": embedder,
                    "classifier": clf,
                    "ipc_distill": IPC_DISTILL,
                    "ipc_random_ref": IPC_RANDOM_REF,
                    "num_runs": NUM_RUNS,
                    "acc_full": full,
                    "acc_random_ipc10": rand10,
                    "acc_distilled_mean": mean_acc,
                    "acc_distilled_std": std_acc,
                    "rr_tdcoler_formula": rr_tdcoler,
                    "rl": full - mean_acc,
                    "synth_time_mean": (float(np.mean(synth_times))
                                         if synth_times else float("nan")),
                }
                all_rows.append(row)
                print(f"    SUMMARY [{embedder}] {clf}: "
                      f"acc={mean_acc:.4f}±{std_acc:.4f}  "
                      f"RR(TDColER)={rr_tdcoler:.4f}  RL={row['rl']:.4f}")

            # incremental save after every (dataset, embedder)
            pd.DataFrame(all_rows).to_csv(
                os.path.join(RESULTS_DIR, "tdbench_comparison_partial.csv"),
                index=False,
            )

    # --- final save and summary ---
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "tdbench_comparison_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n" + "="*70)
    print("  AGGREGATE SUMMARY BY EMBEDDER (TDColER's RR formula)")
    print("="*70)
    for embedder in EMBEDDERS:
        print(f"\n  ===== EMBEDDER: {embedder} =====")
        sub_emb = df[df["embedder"] == embedder]
        for clf in CLASSIFIERS:
            sub = sub_emb[sub_emb["classifier"] == clf]
            if sub.empty:
                continue
            rr = sub["rr_tdcoler_formula"].dropna()
            print(f"    {clf:<10}: n={len(rr):2d}  "
                  f"mean={rr.mean():.4f}  median={rr.median():.4f}")

        rr_emb = sub_emb["rr_tdcoler_formula"].dropna()
        print(f"    OVERALL   : n={len(rr_emb)}  "
              f"mean={rr_emb.mean():.4f}  median={rr_emb.median():.4f}")

    print("\n" + "="*70)
    print(f"  TDColER reference: median RR = 0.4056 (their best, IPC=10)")
    print("="*70)


if __name__ == "__main__":
    main()