"""Layer- and dimension-level analysis of proxy hidden states.

Given ``(N, L+1, D)`` proxy features and per-sample labels we answer:

* Which layer carries the most authorship signal? (linear probe accuracy per layer)
* Which dimensions inside that layer fire differentially across models?
  (L1 logistic coefficients → top-k neurons; class-conditioned mean matrix)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class LayerProbeResult:
    layer_index: int
    accuracy: float
    macro_f1: float


def _torch_probe_one_layer(
    x_layer: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    device: str,
    c_value: float,
    max_iter: int,
) -> tuple[float, float]:
    """Full-batch L2-LR via Adam on GPU. Small data → converges fast, no DataLoader."""
    classes = np.unique(y)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.asarray([class_to_idx[c] for c in y], dtype=np.int64)

    fold_acc: list[float] = []
    fold_f1: list[float] = []
    x_full = torch.as_tensor(x_layer, dtype=torch.float32, device=device)
    y_full = torch.as_tensor(y_idx, device=device)

    weight_decay = 1.0 / (2.0 * c_value * max(len(splits[0][0]), 1))
    for train_idx, test_idx in splits:
        tr_i = torch.as_tensor(train_idx, device=device, dtype=torch.long)
        te_i = torch.as_tensor(test_idx, device=device, dtype=torch.long)
        x_tr = x_full.index_select(0, tr_i)
        x_te = x_full.index_select(0, te_i)
        # StandardScaler-equivalent on train stats.
        mean = x_tr.mean(dim=0, keepdim=True)
        std = x_tr.std(dim=0, keepdim=True) + 1e-6
        x_tr = (x_tr - mean) / std
        x_te = (x_te - mean) / std
        y_tr = y_full.index_select(0, tr_i)
        y_te = y_full.index_select(0, te_i)

        model = torch.nn.Linear(x_tr.shape[1], len(classes)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(max_iter):
            opt.zero_grad(set_to_none=True)
            logits = model(x_tr)
            loss = loss_fn(logits, y_tr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = model(x_te).argmax(dim=1).cpu().numpy()
        y_te_np = y_te.cpu().numpy()
        fold_acc.append(accuracy_score(y_te_np, pred))
        fold_f1.append(f1_score(y_te_np, pred, average="macro"))
    return float(np.mean(fold_acc)), float(np.mean(fold_f1))


def layerwise_probe_accuracy(
    features: np.ndarray,
    labels: list[str],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    c_value: float = 1.0,
    max_iter: int = 200,
    device: str | None = None,
) -> list[LayerProbeResult]:
    """Run a stratified K-fold linear probe independently for every layer.

    ``features`` has shape ``(N, L, D)``. Uses GPU torch LR if CUDA available
    (fast for the ``L ≈ 37`` layers we sweep); otherwise falls back to sklearn
    lbfgs.
    """
    if features.ndim != 3:
        raise ValueError(f"expected (N, L, D); got {features.shape}")
    _, n_layers, _ = features.shape
    y = np.asarray(labels)
    use_cuda = (device == "cuda") or (device is None and torch.cuda.is_available())
    dev = "cuda" if use_cuda else "cpu"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]

    results: list[LayerProbeResult] = []
    for layer in range(n_layers):
        x_layer = features[:, layer, :]
        if use_cuda:
            acc, f1 = _torch_probe_one_layer(
                x_layer, y, splits, device=dev, c_value=c_value, max_iter=max_iter
            )
        else:
            fold_acc: list[float] = []
            fold_f1: list[float] = []
            for train_idx, test_idx in splits:
                scaler = StandardScaler()
                x_tr = scaler.fit_transform(x_layer[train_idx])
                x_te = scaler.transform(x_layer[test_idx])
                clf = LogisticRegression(C=c_value, max_iter=max_iter, solver="lbfgs")
                clf.fit(x_tr, y[train_idx])
                pred = clf.predict(x_te)
                fold_acc.append(accuracy_score(y[test_idx], pred))
                fold_f1.append(f1_score(y[test_idx], pred, average="macro"))
            acc, f1 = float(np.mean(fold_acc)), float(np.mean(fold_f1))
        results.append(LayerProbeResult(layer_index=layer, accuracy=acc, macro_f1=f1))
    return results


def top_discriminative_dims(
    layer_features: np.ndarray,
    labels: list[str],
    *,
    top_k: int = 32,
    method: str = "l1",
    random_state: int = 42,
) -> np.ndarray:
    """Return dimension indices most predictive of the label at one layer."""
    scaler = StandardScaler()
    x = scaler.fit_transform(layer_features)
    if method == "l1":
        base = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.25, max_iter=300, tol=1e-3
        )
        clf = OneVsRestClassifier(base, n_jobs=9)
        clf.fit(x, labels)
        coef = np.abs(
            np.vstack([est.coef_.ravel() for est in clf.estimators_])
        ).sum(axis=0)
    elif method == "mi":
        coef = mutual_info_classif(x, labels, random_state=random_state)
    else:
        raise ValueError(f"unknown method {method!r}")
    order = np.argsort(coef)[::-1]
    return order[:top_k]


def class_conditioned_activation_matrix(
    layer_features: np.ndarray,
    labels: list[str],
    dim_indices: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Return (classes x len(dim_indices)) mean-activation matrix, with class order."""
    classes = sorted(set(labels))
    y = np.asarray(labels)
    matrix = np.zeros((len(classes), len(dim_indices)), dtype=np.float32)
    for i, cls in enumerate(classes):
        matrix[i] = layer_features[y == cls][:, dim_indices].mean(axis=0)
    # Z-score across classes per dim to make the heatmap readable.
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True) + 1e-6
    matrix = (matrix - mean) / std
    return matrix, classes
