#!/usr/bin/env python3
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


def save_synth_data(X_syn, y_syn, out_dir, dataset, method, embedder, ipc, run_id):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{dataset}__{method}__{embedder}__ipc{ipc}__run{run_id:02d}"
    torch.save({"X_syn": X_syn.cpu(), "y_syn": y_syn.cpu()}, os.path.join(out_dir, f"{tag}.pt"))


def run_experiment(config, num_runs=10):
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
            synth_type=config["synth_type"], data=data, config=config
        )
        synth_times.append(time.time() - t0)

        save_synth_data(
            X_syn, y_syn, synth_dir,
            config["dataset_name"], config["synth_type"],
            config.get("dm_embedder_type", "none"), config["ipc"], run_id,
        )

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
                f"[{config['dataset_name']} | {config['synth_type']} | "
                f"{config.get('dm_embedder_type', '')} | run {run_id:02d} | "
                f"{clf}] acc={acc:.4f}"
            )

    rows = []
    for clf in classifiers:
        row = {
            "dataset": config["dataset_name"],
            "method": config["synth_type"],
            "embedder": config.get("dm_embedder_type", ""),
            "ipc": config["ipc"],
            "classifier": clf,
            "num_runs": num_runs,
            "test_acc_mean": float(np.mean(accs[clf])),
            "test_acc_std": float(np.std(accs[clf])),
            "synth_time_mean": float(np.mean(synth_times)),
        }
        rows.append(row)
        print(
            f"[SUMMARY | {row['dataset']} | {row['method']} | {clf}] "
            f"acc={row['test_acc_mean']:.4f}±{row['test_acc_std']:.4f} | "
            f"time={row['synth_time_mean']:.2f}s"
        )
    return rows


def main():
    #DB_LIST = ['adult', 'electricity', 'madelon', 'magic', 'phishing', 'satimage']#list(DATASET_REGISTRY.keys())
    DB_LIST = list(DATASET_REGISTRY.keys())
    SYNTH_TYPES = ["leverage_score"]#, "ctgan", "tvae"]
    IPCs = [50]
    EMBEDDERS = ["ln_res_l"]#, "dcnv2_base", "node"]
    CLASSIFIERS = ["mlp", "rf"]
    NUM_RUNS = 5

    RESULTS_DIR = "leverage_results"
    SYNTH_DIR = "levarage_synth_outputs"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_rows = []

    no_embedder = {"ctgan", "tvae", "full", "random", "vq", "voronoi", "gonzalez"}

    for db in DB_LIST:
        for synth_type in SYNTH_TYPES:
            embedder_iter = EMBEDDERS if synth_type not in no_embedder else [""]

            for embedder in embedder_iter:
                for ipc in IPCs:
                    config = {
                        "dataset_name": db,
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        "synth_type": synth_type,
                        "ipc": ipc,

                        "dm_iters": 1000,
                        "dm_lr": 0.5,
                        "dm_batch_real": 128,
                        "dm_embedder_type": embedder,
                        "dm_embedder_size": "base",
                        "dm_embed_hidden": 256,
                        "dm_embed_dim": 48,

                        "ctgan_epochs": 100,
                        "tvae_epochs": 100,

                        "classifiers": CLASSIFIERS,
                        "classifier_hidden": [128, 64],
                        "classifier_epochs": 20,
                        "random_seed": 42,
                        "synth_save_dir": SYNTH_DIR,
                    }

                    rows = run_experiment(config, num_runs=NUM_RUNS)
                    all_rows.extend(rows)

    master = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "results.csv")
    master.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(master)} rows)")


if __name__ == "__main__":
    main()
