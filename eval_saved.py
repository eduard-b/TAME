#!/usr/bin/env python3
"""
Evaluate saved .pt synth files and write results to CSV.

Expected file layout (from save_synth_data in main.py):
  {synth_dir}/{dataset}__{method}__{embedder}__ipc{ipc}__run{run_id:02d}.pt

Each .pt contains: {"X_syn": Tensor, "y_syn": Tensor}
"""

import os
import glob
import random
import re
import csv
import numpy as np
import torch

from data.prepare_database import prepare_db
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_pt_filename(fname):
    """Parse dataset__method__embedder__ipc{N}__run{NN}.pt into a dict."""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("__")
    if len(parts) != 5:
        return None
    m = re.match(r"ipc(\d+)", parts[3])
    r = re.match(r"run(\d+)", parts[4])
    if not m or not r:
        return None
    return {
        "dataset": parts[0],
        "method": parts[1],
        "embedder": parts[2],
        "ipc": int(m.group(1)),
        "run_id": int(r.group(1)),
    }


def discover_pt_files(synth_dir):
    """Scan synth_dir for all .pt files and group by (dataset, method, embedder, ipc)."""
    files = sorted(glob.glob(os.path.join(synth_dir, "*.pt")))
    groups = {}
    for f in files:
        info = parse_pt_filename(f)
        if info is None:
            print(f"[SKIP] Cannot parse: {f}")
            continue
        key = (info["dataset"], info["method"], info["embedder"], info["ipc"])
        groups.setdefault(key, []).append((info["run_id"], f))
    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups


def eval_synth_dir(
    synth_dir,
    classifiers=("mlp",),
    classifier_hidden=(128, 64),
    classifier_epochs=20,
    random_seed=42,
    device=None,
    output_dir=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if output_dir is None:
        output_dir = synth_dir

    groups = discover_pt_files(synth_dir)
    print(f"Found {len(groups)} settings, {sum(len(v) for v in groups.values())} total .pt files\n")

    runs_path = os.path.join(output_dir, "eval_runs.csv")
    summary_path = os.path.join(output_dir, "eval_summary.csv")

    run_fields = ["dataset", "method", "embedder", "ipc", "classifier", "run_id", "seed", "test_acc", "test_auc"]
    sum_fields = ["dataset", "method", "embedder", "ipc", "classifier", "num_runs",
                  "test_acc_mean", "test_acc_std", "test_auc_mean", "test_auc_std"]

    data_cache = {}
    all_runs = []
    all_summaries = []

    for (dataset, method, embedder, ipc), run_list in sorted(groups.items()):

        if dataset not in data_cache:
            cfg = {"random_seed": random_seed, "device": device}
            data_cache[dataset] = prepare_db(cfg, name=dataset)
        data = data_cache[dataset]

        for clf in classifiers:
            accs, aucs = [], []

            for run_id, pt_path in run_list:
                seed = random_seed + run_id
                set_seed(seed)

                ckpt = torch.load(pt_path, map_location="cpu")
                X_syn = ckpt["X_syn"].to(device).float()
                y_syn = ckpt["y_syn"].to(device).long()

                train_data = {
                    "X_train": X_syn,
                    "y_train": y_syn,
                    "X_val": data["X_val"],
                    "y_val": data["y_val"],
                    "input_dim": data["input_dim"],
                    "num_classes": data["num_classes"],
                }

                clf_config = {
                    "device": device,
                    "classifier": clf,
                    "classifier_hidden": list(classifier_hidden),
                    "classifier_epochs": classifier_epochs,
                    "random_seed": seed,
                }

                model = train_classifier(train_data, clf_config)
                acc, auc = evaluate_classifier(model, data, device)
                accs.append(float(acc))
                aucs.append(float(auc))

                all_runs.append({
                    "dataset": dataset, "method": method, "embedder": embedder,
                    "ipc": ipc, "classifier": clf, "run_id": run_id,
                    "seed": seed, "test_acc": acc, "test_auc": auc,
                })

                print(f"[{dataset} | {method} | {embedder} | ipc={ipc} | {clf} | run {run_id:02d}] "
                      f"acc={acc:.4f} auc={auc:.4f}")

            all_summaries.append({
                "dataset": dataset, "method": method, "embedder": embedder,
                "ipc": ipc, "classifier": clf, "num_runs": len(accs),
                "test_acc_mean": float(np.mean(accs)),
                "test_acc_std": float(np.std(accs)),
                "test_auc_mean": float(np.mean(aucs)),
                "test_auc_std": float(np.std(aucs)),
            })

    os.makedirs(output_dir, exist_ok=True)

    with open(runs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader()
        w.writerows(all_runs)

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(all_summaries)

    print(f"\nSaved:\n  {runs_path}\n  {summary_path}")


if __name__ == "__main__":
    eval_synth_dir(
        synth_dir="E:\doctorat\TAME\smoke_test_synth/",
        classifiers=["mlp"],
        classifier_hidden=(128, 64),
        classifier_epochs=20,
        random_seed=42,
        output_dir="eval_results",
    )
