# TAME — Tabular Alignment via Moment Embeddings

Dataset distillation for tabular data through distribution matching. TAME optimizes a compact synthetic dataset by aligning first- and second-order statistics (mean + full covariance) with the real data in a frozen random embedder space.

## Method

At each iteration, a randomly initialized frozen neural network embeds both real and synthetic samples. The synthetic data is updated via SGD to minimize the moment discrepancy between the two distributions, per class. By resampling the embedder each iteration, the synthetic data generalizes across representations rather than overfitting to a single projection.

The loss for each class:

```
L_c = ||μ_real - μ_syn||² + λ ||Σ_real - Σ_syn||²_F
```

## Project structure

```
TAME/
├── main.py                    # sweep driver: synthesize → train → evaluate → CSV
├── eval_saved.py              # re-evaluate saved .pt synth files
├── data/
│   └── prepare_database.py    # 18 OpenML datasets, standardized pipeline
├── synth/
│   ├── registry.py            # dispatch: synthesize("tame", data, config)
│   ├── tame_synth.py          # core TAME: mean + full covariance
│   ├── tame_synth_orders.py   # ablation: moment orders 1–4 + full cov
│   ├── tame_synth_critic.py   # extension: adversarial (WGAN-GP) regularizer
│   ├── tame_synth_fusion.py   # extension: pool / fusion embedder modes
│   ├── ctgan_tvae_synth.py    # CTGAN and TVAE baselines
│   └── reference_synth.py     # random, k-means (VQ), voronoi, gonzalez, full
├── models/
│   ├── embedders.py           # LN-ResL, DCNv2, NODE + size ladders
│   └── classifiers.py         # MLP (torch), MLP (sklearn), RF, SVM
├── eval/
│   └── eval_classifiers.py    # accuracy + AUC evaluation
├── scripts/                   # visualization and analysis utilities
└── utils/
    └── utils.py
```

## Benchmark

18 tabular classification datasets from OpenML, covering binary and multi-class settings (540 to 539k samples, 9 to 500 features, 2 to 26 classes). All datasets use stratified 70/15/15 splits and StandardScaler fitted on train.

## Synthesizer methods

| Method | Type | Description |
|--------|------|-------------|
| `tame` | Condensation | Mean + full covariance matching (core method) |
| `tame_orders` | Condensation | Moment order ablation (1–4 + full cov) |
| `tame_critic` | Condensation | TAME + adversarial critic regularizer |
| `tame_fusion` | Condensation | TAME + pool/fusion embedder selection |
| `ctgan` | Generative | Conditional GAN for tabular data |
| `tvae` | Generative | Tabular VAE |
| `vq` | Reference | Per-class k-means centroids |
| `voronoi` | Reference | Nearest real sample to k-means centroids |
| `gonzalez` | Reference | Farthest-first traversal |
| `random` | Reference | Random per-class sampling |
| `full` | Reference | Full training set (upper bound) |

## Embedder architectures

Three frozen random embedder families are evaluated:

- **LN-ResL**: Residual MLP with LayerNorm and GELU activations
- **DCNv2**: Deep & Cross Network v2 with low-rank cross layers + MLP
- **NODE**: Neural Oblivious Decision Ensembles (differentiable soft trees)

Each has a size ladder (tiny → XL) controlled via `dm_embedder_size`.

## Quick start

```bash
pip install -r requirements.txt

# run a real experiment
python main.py
```

Edit the `DB_LIST`, `SYNTH_TYPES`, `EMBEDDERS`, `IPCs` and `CLASSIFIERS` variables in `main.py` to configure your sweep.

## Evaluation

Results are saved to `final_results/results.csv`. Synthetic datasets are saved as `.pt` files under `synth_outputs/` and can be re-evaluated with different classifiers:

```bash
python eval_saved.py
```
