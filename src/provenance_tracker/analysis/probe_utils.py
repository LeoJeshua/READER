"""Fit a linear probe on best-layer features and serialize it as a bundle."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class ProbeBundle:
    class_names: list[str]
    weights: np.ndarray   # (C, D)
    bias: np.ndarray      # (C,)
    mean: np.ndarray      # (D,)
    std: np.ndarray       # (D,)
    train_accuracy: float


def fit_probe(features: np.ndarray, labels: list[str], *, C: float = 1.0) -> ProbeBundle:
    """Fit ``StandardScaler + multinomial LR`` on ``(N, D)`` features."""
    classes = sorted(set(labels))
    y = np.asarray(labels)

    scaler = StandardScaler().fit(features)
    x = scaler.transform(features)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    clf.fit(x, y)
    # Align weight rows to the sorted class order.
    order = [list(clf.classes_).index(c) for c in classes]
    weights = clf.coef_[order]
    bias = clf.intercept_[order]
    return ProbeBundle(
        class_names=classes,
        weights=weights.astype(np.float32),
        bias=bias.astype(np.float32),
        mean=scaler.mean_.astype(np.float32),
        std=(scaler.scale_ + 1e-6).astype(np.float32),
        train_accuracy=float(clf.score(x, y)),
    )


def save_probe(bundle: ProbeBundle, path: str) -> None:
    np.savez(
        path,
        class_names=np.asarray(bundle.class_names),
        weights=bundle.weights,
        bias=bundle.bias,
        mean=bundle.mean,
        std=bundle.std,
        train_accuracy=np.asarray(bundle.train_accuracy),
    )


def load_probe(path: str) -> ProbeBundle:
    data = np.load(path, allow_pickle=True)
    return ProbeBundle(
        class_names=list(map(str, data["class_names"].tolist())),
        weights=data["weights"],
        bias=data["bias"],
        mean=data["mean"],
        std=data["std"],
        train_accuracy=float(data["train_accuracy"]),
    )
