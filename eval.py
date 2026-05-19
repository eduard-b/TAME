#!/usr/bin/env python3
"""
Reproduce the evaluation CSV from saved .pt synth files.

Filename conventions handled:
    TAME      adult__tame__ln_res_l__ipc50__run04.pt
    non-TAME  airlines__full____ipc50__run09.pt   (embedder slot is empty)

Both split into 5 parts on "__"; for non-TAME the embedder becomes "".

Outputs (in --output-dir):
    eval_runs.csv      per-run accuracy/AUC
    eval_summary.csv   aggregated per (dataset, method, embedder, ipc, classifier)

Summary columns include both mean and MEDIAN across runs so you can compare.
synth_time_mean is NOT reproduced — it is measured at synthesis time, not eval.

Quick start:
    # smoke test: one (dataset, method, embedder) group, MLP only
    python reproduce_csv.py --synth-dir /path/to/synths \
        --dataset adult --method tame --embedder ln_res_l \
        --classifiers mlp --output-dir eval_smoke

    # full sweep
    python reproduce_csv.py --synth-dir /path/to/synths \
        --classifiers mlp rf xgboost --output-dir eval_full
"""
import os
import glob
import re
import csv
import random
import argparse

import numpy as np
import torch

from data.prepare_database import prepare_db
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier


# ---------- utilities ----------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_pt_filename(fname):
    """Return dict for a parsable filename, else None.
    Handles both 'a__b__c__ipcN__runNN' and 'a__b____ipcN__runNN'."""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("__")
    if len(parts) != 5:
        return None
    m_ipc = re.fullmatch(r"ipc(\d+)", parts[3])
    m_run = re.fullmatch(r"run(\d+)", parts[4])
    if not m_ipc or not m_run:
        return None
    return {
        "dataset":  parts[0],
        "method":   parts[1],
        "embedder": parts[2],  # may be "" for non-TAME methods
        "ipc":      int(m_ipc.group(1)),
        "run_id":   int(m_run.group(1)),
    }


def discover_pt_files(synth_dir):
    files = sorted(glob.glob(os.path.join(synth_dir, "*.pt")))
    groups, skipped = {}, []
    for f in files:
        info = parse_pt_filename(f)
        if info is None:
            skipped.append(f)
            continue
        key = (info["dataset"], info["method"], info["embedder"], info["ipc"])
        groups.setdefault(key, []).append((info["run_id"], f))
    for k in groups:
        groups[k].sort(key=lambda x: x[0])
    return groups, skipped


def filter_groups(groups, dataset=None, method=None, embedder=None, ipc=None):
    """Keep only groups matching the given filters. None = wildcard."""
    out = {}
    for (d, m, e, p), runs in groups.items():
        if dataset  is not None and d != dataset:  continue
        if method   is not None and m != method:   continue
        if embedder is not None and e != embedder: continue
        if ipc      is not None and p != ipc:      continue
        out[(d, m, e, p)] = runs
    return out


# ---------- evaluation ----------

def evaluate_one_run(pt_path, data, clf, clf_hidden, clf_epochs, seed, device):
    set_seed(seed)
    ckpt = torch.load(pt_path, map_location="cpu")
    X_syn = ckpt["X_syn"].to(device).float()
    y_syn = ckpt["y_syn"].to(device).long()

    train_data = {
        "X_train": X_syn,
        "y_train": y_syn,
        "X_val":   data["X_val"],
        "y_val":   data["y_val"],
        "input_dim":   data["input_dim"],
        "num_classes": data["num_classes"],
    }
    clf_config = {
        "device":             device,
        "classifier":         clf,
        "classifier_hidden":  list(clf_hidden),
        "classifier_epochs":  clf_epochs,
        "random_seed":        seed,
    }
    model = train_classifier(train_data, clf_config)
    acc, auc = evaluate_classifier(model, data, device)
    return float(acc), float(auc)


def evaluate_group(key, run_list, data, classifiers, clf_hidden, clf_epochs,
                   random_seed, device):
    dataset, method, embedder, ipc = key
    per_run, summaries = [], []
    for clf in classifiers:
        accs, aucs = [], []
        for run_id, pt_path in run_list:
            seed = random_seed + run_id
            try:
                acc, auc = evaluate_one_run(
                    pt_path, data, clf, clf_hidden, clf_epochs, seed, device,
                )
            except Exception as e:
                print(f"  [FAIL] run {run_id:02d} {clf}: {e}")
                continue
            accs.append(acc); aucs.append(auc)
            per_run.append({
                "dataset": dataset, "method": method, "embedder": embedder,
                "ipc": ipc, "classifier": clf, "run_id": run_id, "seed": seed,
                "test_acc": acc, "test_auc": auc,
            })
            print(f"  [{dataset}|{method}|{embedder or '-'}|ipc{ipc}|{clf}|run{run_id:02d}] "
                  f"acc={acc:.4f} auc={auc:.4f}")
        if not accs:
            print(f"  [SKIP] no successful runs for {key}/{clf}")
            continue
        summaries.append({
            "dataset": dataset, "method": method, "embedder": embedder,
            "ipc": ipc, "classifier": clf, "num_runs": len(accs),
            "test_acc_mean":   float(np.mean(accs)),
            "test_acc_std":    float(np.std(accs)),
            "test_acc_median": float(np.median(accs)),
            "test_auc_mean":   float(np.mean(aucs)),
            "test_auc_std":    float(np.std(aucs)),
            "test_auc_median": float(np.median(aucs)),
        })
    return per_run, summaries


# ---------- IO ----------

def write_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth-dir", required=True,
                   help="Directory containing the .pt synth files")
    p.add_argument("--output-dir", default="eval_results")
    p.add_argument("--classifiers", nargs="+", default=["mlp", "rf", "xgboost"])
    p.add_argument("--classifier-hidden", nargs="+", type=int, default=[128, 64])
    p.add_argument("--classifier-epochs", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="Default: cuda if available else cpu")
    p.add_argument("--deterministic", action="store_true",
                   help="Force cudnn deterministic (slower, more reproducible)")
    # Filters — leave any of these unset to wildcard. Use --embedder "" for non-TAME.
    p.add_argument("--dataset",  default=None)
    p.add_argument("--method",   default=None)
    p.add_argument("--embedder", default=None,
                   help="Empty string '' selects non-TAME methods")
    p.add_argument("--ipc", type=int, default=None)
    args = p.parse_args()

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"[WARN] could not enable deterministic algorithms: {e}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    groups, skipped = discover_pt_files(args.synth_dir)
    n_files = sum(len(v) for v in groups.values())
    print(f"Discovered {len(groups)} groups, {n_files} .pt files. "
          f"Skipped {len(skipped)} unparseable.")
    if skipped:
        for s in skipped[:10]:
            print(f"  unparseable: {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    groups = filter_groups(groups, args.dataset, args.method, args.embedder, args.ipc)
    if not groups:
        print("No groups match the filter. Exiting.")
        return
    print(f"After filtering: {len(groups)} group(s) to process.\n")

    data_cache = {}
    all_runs, all_summaries = [], []
    for key in sorted(groups.keys()):
        dataset, method, embedder, ipc = key
        run_list = groups[key]
        print(f"=== {dataset} | {method} | {embedder or '-'} | ipc={ipc} | {len(run_list)} runs ===")
        if dataset not in data_cache:
            cfg = {"random_seed": args.random_seed, "device": device}
            data_cache[dataset] = prepare_db(cfg, name=dataset)
        per_run, summaries = evaluate_group(
            key, run_list, data_cache[dataset],
            args.classifiers, args.classifier_hidden,
            args.classifier_epochs, args.random_seed, device,
        )
        all_runs.extend(per_run)
        all_summaries.extend(summaries)
        for s in summaries:
            print(f"  -> {s['classifier']:8s} n={s['num_runs']:2d}  "
                  f"acc mean={s['test_acc_mean']:.4f} median={s['test_acc_median']:.4f}  "
                  f"auc mean={s['test_auc_mean']:.4f} median={s['test_auc_median']:.4f}")
        print()

    run_fields = ["dataset", "method", "embedder", "ipc", "classifier",
                  "run_id", "seed", "test_acc", "test_auc"]
    sum_fields = ["dataset", "method", "embedder", "ipc", "classifier", "num_runs",
                  "test_acc_mean", "test_acc_std", "test_acc_median",
                  "test_auc_mean", "test_auc_std", "test_auc_median"]

    runs_path    = os.path.join(args.output_dir, "eval_runs.csv")
    summary_path = os.path.join(args.output_dir, "eval_summary.csv")
    write_csv(all_runs,      runs_path,    run_fields)
    write_csv(all_summaries, summary_path, sum_fields)
    print(f"Saved:\n  {runs_path}\n  {summary_path}")


if __name__ == "__main__":
    main()