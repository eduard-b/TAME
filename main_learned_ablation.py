#!/usr/bin/env python3
"""
Ablation experiment: Random vs Learned embedder × Low vs High moment order.

Four conditions:
  1. random_low   — random frozen embedder, mean + full covariance  (order=5)
  2. random_high  — random frozen embedder, mean+var+skew+kurtosis  (order=4)
  3. learned_low  — pre-trained embedder,   mean + full covariance  (order=5)
  4. learned_high — pre-trained embedder,   mean+var+skew+kurtosis  (order=4)

All conditions use the same datasets, IPC, embedder architecture, and
downstream classifiers, so differences are attributable to the two variables.

Output: CSV with per-dataset, per-condition, per-classifier accuracy.
"""

import os
import random
import time
import numpy as np
import pandas as pd
import torch

from data.prepare_database import prepare_db, DATASET_REGISTRY
from synth.registry import synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_synth_data(X_syn, y_syn, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {"X_syn": X_syn.cpu(), "y_syn": y_syn.cpu()},
        os.path.join(out_dir, f"{tag}.pt"),
    )


def run_single(config, num_runs, condition_label):
    """Run one condition (synth + eval) for num_runs, return result rows."""
    classifiers = config.get("classifiers", ["mlp"])
    device = config["device"]
    synth_dir = config.get("synth_save_dir", "synth_outputs")

    accs = {clf: [] for clf in classifiers}
    synth_times = []

    for run_id in range(num_runs):
        run_seed = config.get("random_seed", 42) + run_id
        set_seed(run_seed)

        data = prepare_db(config, name=config["dataset_name"])

        t0 = time.time()
        X_syn, y_syn = synthesize(
            synth_type=config["synth_type"], data=data, config=config,
        )
        synth_times.append(time.time() - t0)

        tag = (f"{config['dataset_name']}__{condition_label}"
               f"__{config['dm_embedder_type']}__ipc{config['ipc']}"
               f"__run{run_id:02d}")
        save_synth_data(X_syn, y_syn, synth_dir, tag)

        train_data = {
            "X_train": X_syn,
            "y_train": y_syn,
            "X_val": data["X_val"],
            "y_val": data["y_val"],
            "input_dim": data["input_dim"],
            "num_classes": data["num_classes"],
        }

        for clf in classifiers:
            clf_config = dict(config)
            clf_config["classifier"] = clf
            model = train_classifier(train_data, clf_config)
            acc, _ = evaluate_classifier(model, data, device)
            accs[clf].append(float(acc))
            print(
                f"  [{config['dataset_name']} | {condition_label} | "
                f"run {run_id:02d} | {clf}] acc={acc:.4f}"
            )

    rows = []
    for clf in classifiers:
        row = {
            "dataset": config["dataset_name"],
            "condition": condition_label,
            "synth_type": config["synth_type"],
            "embedder": config.get("dm_embedder_type", ""),
            "moment_order": config.get("dm_moment_order", ""),
            "ipc": config["ipc"],
            "classifier": clf,
            "num_runs": num_runs,
            "test_acc_mean": float(np.mean(accs[clf])),
            "test_acc_std": float(np.std(accs[clf])),
            "synth_time_mean": float(np.mean(synth_times)),
        }
        rows.append(row)
        print(
            f"  [SUMMARY | {row['dataset']} | {condition_label} | {clf}] "
            f"acc={row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f} | "
            f"time={row['synth_time_mean']:.1f}s"
        )
    return rows


def main():
    # ------------------------------------------------------------------
    # Configuration — edit these as needed
    # ------------------------------------------------------------------
    DB_LIST = ['adult', 'electricity', 'madelon', 'magic', 'phishing', 'satimage']#list(DATASET_REGISTRY.keys())
    # For a quick test, uncomment the line below:
    # DB_LIST = ["adult", "magic", "phishing", "electricity"]

    EMBEDDER = "ln_res_l"          # single embedder for clean comparison
    IPC = 50
    NUM_RUNS = 3                   # match your paper's protocol
    CLASSIFIERS = ["mlp", "rf"]    # MLP + RF to track the bias gap

    RESULTS_DIR = "results_learned_ablation_adult"
    SYNTH_DIR = "synth_outputs_learned_ablation_adult"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # The four experimental conditions
    # ------------------------------------------------------------------
    # Each condition is: (label, synth_type, moment_order)
    #
    # order=5 in your code means: mean + full covariance (Frobenius)
    # order=4 in your code means: mean + per-dim variance + skewness + kurtosis
    #
    CONDITIONS = [
        ("random_low",  "tame_orders",  5),   # random embedder, mean+cov
        ("random_high", "tame_orders",  4),   # random embedder, mean+var+skew+kurt
        ("learned_low",  "tame_learned", 5),  # learned embedder, mean+cov
        ("learned_high", "tame_learned", 4),  # learned embedder, mean+var+skew+kurt
    ]

    # ------------------------------------------------------------------
    # Shared config template
    # ------------------------------------------------------------------
    config_template = {
        "device": DEVICE,
        "ipc": IPC,

        # distillation
        "dm_iters": 1000,
        "dm_lr": 0.5,
        "dm_batch_real": 128,
        "dm_embedder_type": EMBEDDER,
        "dm_embedder_size": "base",
        "dm_embed_hidden": 256,
        "dm_embed_dim": 48,

        # learned embedder pre-training
        "dm_pretrain_epochs": 50,
        "dm_pretrain_lr": 1e-3,

        # downstream classifier
        "classifiers": CLASSIFIERS,
        "classifier_hidden": [128, 64],
        "classifier_epochs": 20,
        "random_seed": 42,
        "synth_save_dir": SYNTH_DIR,
    }

    # ------------------------------------------------------------------
    # Run all conditions × all datasets
    # ------------------------------------------------------------------
    all_rows = []

    for db_name in DB_LIST:
        print(f"\n{'='*70}")
        print(f"  DATASET: {db_name}")
        print(f"{'='*70}")

        for cond_label, synth_type, moment_order in CONDITIONS:
            print(f"\n  --- Condition: {cond_label} "
                  f"(synth={synth_type}, order={moment_order}) ---")

            config = dict(config_template)
            config["dataset_name"] = db_name
            config["synth_type"] = synth_type
            config["dm_moment_order"] = moment_order

            try:
                rows = run_single(config, NUM_RUNS, cond_label)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  *** FAILED: {db_name} / {cond_label}: {e}")
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Save and print summary
    # ------------------------------------------------------------------
    master = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "learned_ablation_results.csv")
    master.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(master)} rows)")

    # Pivot table for quick comparison
    print("\n\n" + "="*70)
    print("  RESULTS PIVOT: mean test accuracy")
    print("="*70)

    for clf in CLASSIFIERS:
        sub = master[master["classifier"] == clf]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="dataset",
            columns="condition",
            values="test_acc_mean",
            aggfunc="first",
        )
        # reorder columns
        col_order = [c for c in ["random_low", "random_high",
                                  "learned_low", "learned_high"]
                     if c in pivot.columns]
        pivot = pivot[col_order]
        print(f"\n  Classifier: {clf.upper()}")
        print(pivot.round(4).to_string())
        print()

        # Per-condition mean across datasets
        print(f"  Mean across datasets ({clf.upper()}):")
        for col in col_order:
            print(f"    {col:<16}: {pivot[col].mean():.4f}")
        print()


if __name__ == "__main__":
    main()
