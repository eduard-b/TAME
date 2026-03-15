import torch.nn as nn
from utils.utils import set_seed
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone

def train_classifier(data, config):
    """
    Unified classifier training entry point.

    Expects:
      config["classifier"] in CLASSIFIER_REGISTRY
    """
    clf_name = config.get("classifier", "mlp")

    if clf_name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{clf_name}'. "
            f"Available: {list(CLASSIFIER_REGISTRY.keys())}"
        )

    train_fn = CLASSIFIER_REGISTRY[clf_name]
    return train_fn(data, config)

def train_rf(data, config):
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()

    model = RandomForestClassifier(
        n_estimators=config.get("rf_n_estimators", 200),
        random_state=config["random_seed"],
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

def train_svm(data, config):
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=config["random_seed"],
    )
    model.fit(X, y)
    return model

class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden=[128, 64], num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h

        # Output logits only
        if num_classes == 2:
            layers.append(nn.Linear(prev, 1))
        else:
            layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        logits = self.net(x)
        if self.num_classes == 2:
            return logits.view(-1)  # shape [N]
        else:
            return logits            # shape [N, C]
        
def train_mlp_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    hidden,
    epochs,
    seed,
    device,
    num_classes,
):
    set_seed(seed)

    X_train = X_train.to(device).float()
    X_val   = X_val.to(device).float()
    y_train = y_train.to(device)
    y_val   = y_val.to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=1024, shuffle=False)

    model = ClassifierMLP(input_dim, hidden, num_classes).to(device)

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_auc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):

        # -------------- TRAIN ----------------
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)

            opt.zero_grad()

            logits = model(xb)
            if num_classes == 2:
                loss = criterion(logits, yb.float())
            else:
                loss = criterion(logits, yb.long())

            loss.backward()
            opt.step()

        # -------------- VALID ----------------
        model.eval()
        probs_all = []
        trues_all = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device)

                logits = model(xb)
                if num_classes == 2:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)

                probs_all.append(probs.cpu().numpy())
                trues_all.append(yb.cpu().numpy())

        probs_all = np.concatenate(probs_all)
        trues_all = np.concatenate(trues_all)

        if num_classes == 2:
            val_auc = roc_auc_score(trues_all, probs_all)
        else:
            val_auc = roc_auc_score(trues_all, probs_all, multi_class="ovr", average="macro")

        print(f"[Classifier] Epoch {ep:02d} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model, best_auc        

def train_sklearn_mlp_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    hidden,
    epochs,
    seed,
    device,
    num_classes
):
    """
    Train an sklearn MLPClassifier and select the best model by validation AUC.

    Parameters
    ----------
    X_train, y_train, X_val, y_val:
        Can be torch tensors or numpy arrays.
    hidden:
        int or tuple/list of ints for hidden layer sizes.
        (e.g., 100 or (100,) or (256, 256))
    epochs:
        Interpreted as max_iter for sklearn. We'll fit multiple times with warm_start
        and snapshot the best by *external* validation AUC (X_val/y_val).
    seed:
        Random seed for reproducibility.
    num_classes:
        2 for binary, >2 for multiclass.

    Returns
    -------
    best_model, best_auc
    """
    # ---- convert torch -> numpy if needed ----
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        if hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.cpu().numpy()
        return np.asarray(x)

    Xtr = to_numpy(X_train).astype(np.float32)
    Xva = to_numpy(X_val).astype(np.float32)
    ytr = to_numpy(y_train)
    yva = to_numpy(y_val)

    # sklearn expects 1D y
    ytr = ytr.reshape(-1)
    yva = yva.reshape(-1)

    # normalize hidden spec
    if isinstance(hidden, int):
        hidden_layer_sizes = (hidden,)
    else:
        hidden_layer_sizes = tuple(hidden)

    # Base model close to sklearn defaults, with reproducibility and warm_start
    base = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=256,          # close to your torch batch size
        learning_rate_init=1e-3,  # matches your Adam lr
        max_iter=1,              # we will step epochs manually via warm_start
        warm_start=True,
        shuffle=True,
        random_state=seed,
        early_stopping=False,    # we use external val set, so disable internal split
        n_iter_no_change=10,
        tol=1e-4,
        verbose=False,
    )

    best_auc = -1.0
    best_model = None

    # We'll "simulate epochs" by fitting for 1 iteration at a time using warm_start
    model = clone(base)

    for ep in range(1, epochs + 1):
        model.fit(Xtr, ytr)

        # ---- Validation AUC ----
        if num_classes == 2:
            # sklearn returns probs for class 1 in [:, 1]
            p = model.predict_proba(Xva)[:, 1]
            val_auc = roc_auc_score(yva, p)
        else:
            P = model.predict_proba(Xva)
            val_auc = roc_auc_score(yva, P, multi_class="ovr", average="macro")

        print(f"[sklearn MLP] Epoch {ep:02d} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            # clone+set_params won't copy learned weights, so use deepcopy
            import copy
            best_model = copy.deepcopy(model)

    return best_model, best_auc

def train_mlp(data, config):
    model, _ = train_mlp_classifier(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        input_dim=data["input_dim"],
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=config["device"],
        num_classes=data["num_classes"],
    )
    return model

def train_mlp_scikit(data, config):
    model, _ = train_sklearn_mlp_classifier(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        input_dim=data["input_dim"],
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=config["device"],
        num_classes=data["num_classes"],
    )
    return model

CLASSIFIER_REGISTRY = {
    "mlp": train_mlp,
    "mlp_sci": train_mlp_scikit,
    "rf": train_rf,
    "svm": train_svm,
}
