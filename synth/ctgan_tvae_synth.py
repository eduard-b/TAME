import numpy as np
import pandas as pd
import torch

from ctgan import CTGAN, TVAE


# ------------------------------------------------------------------ #
#  helpers
# ------------------------------------------------------------------ #

def _tensors_to_dataframe(X: torch.Tensor, y: torch.Tensor) -> pd.DataFrame:
    """Convert (X, y) tensors → pandas DataFrame with a '__label__' column."""
    X_np = X.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy().astype(int)

    df = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
    df["__label__"] = y_np.astype(str)  # string so CTGAN treats it as discrete
    return df


def _enforce_ipc(df_syn: pd.DataFrame, num_classes: int, ipc: int, input_dim: int, device: str):
    """
    From a generated DataFrame, extract exactly `ipc` rows per class.
    Returns (X_syn, y_syn) as tensors on `device`.

    If a class is under-represented in the raw generation, we resample
    with replacement to hit ipc.  If over-represented, we subsample.
    """
    parts_X, parts_y = [], []

    for c in range(num_classes):
        mask = df_syn["__label__"].astype(int) == c
        sub = df_syn.loc[mask]

        if len(sub) == 0:
            # CTGAN occasionally misses a class → fill with noise around 0
            # (this is honest: the baseline simply failed for this class)
            print(f"[WARN] CTGAN/TVAE generated 0 samples for class {c}, padding with noise")
            X_c = np.random.randn(ipc, input_dim).astype(np.float32) * 0.01
        elif len(sub) < ipc:
            X_c = sub.drop(columns=["__label__"]).values.astype(np.float32)
            # resample with replacement to reach ipc
            idx = np.random.choice(len(X_c), size=ipc, replace=True)
            X_c = X_c[idx]
        else:
            X_c = sub.drop(columns=["__label__"]).values.astype(np.float32)
            idx = np.random.choice(len(X_c), size=ipc, replace=False)
            X_c = X_c[idx]

        parts_X.append(X_c)
        parts_y.append(np.full(ipc, c, dtype=np.int64))

    X_syn = torch.tensor(np.concatenate(parts_X, axis=0), device=device, dtype=torch.float32)
    y_syn = torch.tensor(np.concatenate(parts_y, axis=0), device=device, dtype=torch.long)
    return X_syn, y_syn


# ------------------------------------------------------------------ #
#  CTGAN wrapper
# ------------------------------------------------------------------ #

def ctgan_synthesize(data: dict, config: dict):
    """
    CTGAN baseline for the distillation benchmark.

    Relevant config keys (all optional, sensible defaults provided):
        ctgan_epochs          (int)   : training epochs          [300]
        ctgan_batch_size      (int)   : batch size (must be even) [500]
        ctgan_generator_dim   (tuple) : generator layer sizes    [(256,256)]
        ctgan_discriminator_dim (tuple): discriminator layers     [(256,256)]
        ctgan_embedding_dim   (int)   : noise vector dim          [128]
        ctgan_generator_lr    (float) : generator LR             [2e-4]
        ctgan_discriminator_lr(float) : discriminator LR         [2e-4]
        ctgan_discriminator_steps (int): D steps per G step      [1]
        ctgan_pac             (int)   : PacGAN grouping          [10]
        ctgan_gen_samples_mul (int)   : generate this * total_ipc samples,
                                        then trim to ipc/class  [5]
    """
    device = config["device"]
    ipc = config["ipc"]
    num_classes = data["num_classes"]
    input_dim = data["input_dim"]

    # --- build training dataframe ---
    df_train = _tensors_to_dataframe(data["X_train"], data["y_train"])

    # --- hyperparams ---
    epochs    = config.get("ctgan_epochs", 300)
    batch_size = config.get("ctgan_batch_size", 500)
    gen_dim   = config.get("ctgan_generator_dim", (256, 256))
    disc_dim  = config.get("ctgan_discriminator_dim", (256, 256))
    emb_dim   = config.get("ctgan_embedding_dim", 128)
    gen_lr    = config.get("ctgan_generator_lr", 2e-4)
    disc_lr   = config.get("ctgan_discriminator_lr", 2e-4)
    disc_steps = config.get("ctgan_discriminator_steps", 1)
    pac       = config.get("ctgan_pac", 10)
    gen_mul   = config.get("ctgan_gen_samples_mul", 5)

    # batch_size must be even and divisible by pac
    batch_size = max(batch_size, pac * 2)
    batch_size = batch_size - (batch_size % (pac * 2))

    # Use GPU only if the user's device is cuda
    use_gpu = "cuda" in str(device)

    model = CTGAN(
        embedding_dim=emb_dim,
        generator_dim=gen_dim,
        discriminator_dim=disc_dim,
        generator_lr=gen_lr,
        discriminator_lr=disc_lr,
        discriminator_steps=disc_steps,
        batch_size=batch_size,
        epochs=epochs,
        pac=pac,
        enable_gpu=use_gpu,
        verbose=True,
    )

    model.fit(df_train, discrete_columns=["__label__"])

    # --- generate & enforce IPC ---
    total_needed = num_classes * ipc * gen_mul
    df_syn = model.sample(total_needed)

    # CTGAN returns a DataFrame with same columns as training data
    # Remap __label__ back to int (CTGAN may return strings)
    df_syn["__label__"] = pd.to_numeric(df_syn["__label__"], errors="coerce")
    df_syn = df_syn.dropna(subset=["__label__"])
    df_syn["__label__"] = df_syn["__label__"].round().astype(int)

    # Clamp labels to valid range (CTGAN can hallucinate invalid classes)
    df_syn = df_syn[df_syn["__label__"].between(0, num_classes - 1)]

    X_syn, y_syn = _enforce_ipc(df_syn, num_classes, ipc, input_dim, device)

    return X_syn, y_syn


# ------------------------------------------------------------------ #
#  TVAE wrapper
# ------------------------------------------------------------------ #

def tvae_synthesize(data: dict, config: dict):
    """
    TVAE baseline for the distillation benchmark.

    Relevant config keys (all optional, sensible defaults provided):
        tvae_epochs           (int)   : training epochs           [300]
        tvae_batch_size       (int)   : batch size                [500]
        tvae_compress_dims    (tuple) : encoder layer sizes       [(128,128)]
        tvae_decompress_dims  (tuple) : decoder layer sizes       [(128,128)]
        tvae_embedding_dim    (int)   : latent dim                [128]
        tvae_l2scale          (float) : weight decay              [1e-5]
        tvae_loss_factor      (int)   : reconstruction loss scale [2]
        tvae_gen_samples_mul  (int)   : oversample multiplier     [5]
    """
    device = config["device"]
    ipc = config["ipc"]
    num_classes = data["num_classes"]
    input_dim = data["input_dim"]

    # --- build training dataframe ---
    df_train = _tensors_to_dataframe(data["X_train"], data["y_train"])

    # --- hyperparams ---
    epochs    = config.get("tvae_epochs", 300)
    batch_size = config.get("tvae_batch_size", 500)
    comp_dims = config.get("tvae_compress_dims", (128, 128))
    decomp_dims = config.get("tvae_decompress_dims", (128, 128))
    emb_dim   = config.get("tvae_embedding_dim", 128)
    l2scale   = config.get("tvae_l2scale", 1e-5)
    loss_factor = config.get("tvae_loss_factor", 2)
    gen_mul   = config.get("tvae_gen_samples_mul", 5)

    use_gpu = "cuda" in str(device)

    model = TVAE(
        embedding_dim=emb_dim,
        compress_dims=comp_dims,
        decompress_dims=decomp_dims,
        l2scale=l2scale,
        batch_size=batch_size,
        epochs=epochs,
        loss_factor=loss_factor,
        enable_gpu=use_gpu,
        verbose=True,
    )

    model.fit(df_train, discrete_columns=["__label__"])

    # --- generate & enforce IPC ---
    total_needed = num_classes * ipc * gen_mul
    df_syn = model.sample(total_needed)

    df_syn["__label__"] = pd.to_numeric(df_syn["__label__"], errors="coerce")
    df_syn = df_syn.dropna(subset=["__label__"])
    df_syn["__label__"] = df_syn["__label__"].round().astype(int)
    df_syn = df_syn[df_syn["__label__"].between(0, num_classes - 1)]

    X_syn, y_syn = _enforce_ipc(df_syn, num_classes, ipc, input_dim, device)

    return X_syn, y_syn
