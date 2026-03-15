import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

# def evaluate_classifier(model, data, device):
#     X_test = data["X_test"]
#     y_test = data["y_test"]
#     num_classes = data["num_classes"]

#     if hasattr(model, "predict_proba"):
#         probs = model.predict_proba(X_test.cpu().numpy())
#     else:
#         with torch.no_grad():
#             model.eval()
#             logits = model(X_test.to(device).float())
#             if num_classes == 2:
#                 probs = torch.sigmoid(logits).cpu().numpy()
#             else:
#                 probs = torch.softmax(logits, dim=1).cpu().numpy()

#     y_true = y_test.cpu().numpy()

#     if num_classes == 2:
#         auc = roc_auc_score(y_true, probs)
#         acc = ((probs > 0.5).astype(int) == y_true).mean()
#     else:
#         auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
#         acc = (probs.argmax(1) == y_true).mean()

#     return acc, auc


import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def evaluate_classifier(model, data, device):
    """
    Evaluates model on test split.

    Supports:
      - torch nn.Module: outputs logits (binary: [N] or [N,1], multiclass: [N,C])
      - sklearn estimators with predict_proba
    """
    X_test = data["X_test"]
    y_test = data["y_test"]

    y_true = _to_numpy(y_test).astype(int)

    # ---- get probabilities/scores ----
    probs = None

    # sklearn-style
    if hasattr(model, "predict_proba"):
        Xn = _to_numpy(X_test)
        probs = model.predict_proba(Xn)  # (N,2) or (N,C)
        probs = _to_numpy(probs)

    # torch-style
    else:
        model.eval()
        with torch.no_grad():
            Xt = X_test.to(device).float()
            logits = model(Xt)

            if isinstance(logits, tuple):
                logits = logits[0]

            logits = logits.detach()

            # binary: logits can be [N] or [N,1]
            if logits.ndim == 1:
                probs = torch.sigmoid(logits).cpu().numpy()  # (N,)
            elif logits.ndim == 2 and logits.shape[1] == 1:
                probs = torch.sigmoid(logits[:, 0]).cpu().numpy()  # (N,)
            else:
                # multiclass: (N,C)
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N,C)

    # ---- predictions for accuracy ----
    if probs.ndim == 1:
        # binary
        y_pred = (probs >= 0.5).astype(int)
    else:
        # multiclass OR binary probs as (N,2)
        y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))

    # ---- AUC ----
    # binary: need 1D scores for positive class
    try:
        if probs.ndim == 1:
            auc = float(roc_auc_score(y_true, probs))
        else:
            # probs is (N, C)
            if probs.shape[1] == 2:
                auc = float(roc_auc_score(y_true, probs[:, 1]))
            else:
                # multiclass
                auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        # Happens if y_true contains only one class in test split
        auc = float("nan")

    return acc, auc
