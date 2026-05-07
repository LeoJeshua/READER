"""Unified multi-metric evaluation for per-sample provenance features.

All methods (mpnet baseline, proxy last-layer, proxy best-layer, proxy multi-layer)
produce ``(N, D)`` feature matrices with aligned labels, and are scored with the
same four metric families:

* ``classification``  — stratified K-fold LR accuracy / macro-F1
* ``pairwise_auc``    — mean of per-pair binary LR AUC
* ``retrieval``       — leave-one-out cosine retrieval mAP@k
* ``clustering``      — KMeans(k=#classes) ARI / NMI on normalized features
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _make_clf(clf: str):
    """Return a fresh classifier. ``clf`` is one of {'lr', 'svm'}."""
    if clf == "lr":
        return LogisticRegression(max_iter=3000, solver="lbfgs")
    if clf == "svm":
        # Linear SVM matches LLM-DNA's hinge-loss head; no Platt scaling needed
        # since pairwise AUC consumes decision_function (any monotonic score).
        return LinearSVC(C=1.0, dual="auto", max_iter=5000)
    raise ValueError(f"unknown clf={clf!r}")


def _score_binary(model, x_te: np.ndarray) -> np.ndarray:
    """Return a 1D positive-class score for binary AUC (works for LR & SVM)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_te)[:, 1]
    return model.decision_function(x_te)


@dataclass(slots=True)
class EvaluationReport:
    method: str
    n_samples: int
    n_classes: int
    classification_accuracy: float
    classification_accuracy_std: float
    classification_macro_f1: float
    classification_macro_f1_std: float
    mean_pairwise_auc: float
    mean_pairwise_auc_std: float
    retrieval_map_at_5: float
    retrieval_map_at_10: float
    clustering_ari: float
    clustering_nmi: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    # Math (OOD) classification metrics — populated only when math features are
    # supplied to evaluate_method. The same fold-trained LR predicts on math,
    # so these share fold + scaler with the in-distribution metrics above.
    math_n_samples: int = 0
    math_classification_accuracy: float = float("nan")
    math_classification_accuracy_std: float = float("nan")
    math_classification_macro_f1: float = float("nan")
    math_classification_macro_f1_std: float = float("nan")
    math_per_class_f1: dict[str, float] = field(default_factory=dict)

    def as_flat_dict(self) -> dict[str, float | str | int]:
        payload = {k: getattr(self, k) for k in (
            "method",
            "n_samples",
            "n_classes",
            "classification_accuracy",
            "classification_accuracy_std",
            "classification_macro_f1",
            "classification_macro_f1_std",
            "mean_pairwise_auc",
            "mean_pairwise_auc_std",
            "retrieval_map_at_5",
            "retrieval_map_at_10",
            "clustering_ari",
            "clustering_nmi",
            "math_n_samples",
            "math_classification_accuracy",
            "math_classification_accuracy_std",
            "math_classification_macro_f1",
            "math_classification_macro_f1_std",
        )}
        payload["per_class_f1"] = dict(self.per_class_f1)
        payload["math_per_class_f1"] = dict(self.math_per_class_f1)
        return payload


def _standardize(x_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(x_tr), scaler.transform(x_te)


def _per_class_f1(true: list, pred: list, classes: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(true, pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true, pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true, pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[cls] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return out


def _classification_fold(
    features: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    clf: str,
    math_features: np.ndarray | None = None,
    math_y: np.ndarray | None = None,
) -> tuple:
    """Single fold worker. Fits scaler+clf on agent[train], predicts on agent[test]
    and (optionally) on math (full). Returns flat tuple so joblib can serialize.
    """
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(features[tr_idx])
    x_te = scaler.transform(features[te_idx])
    model = _make_clf(clf)
    model.fit(x_tr, y[tr_idx])
    pred = model.predict(x_te)
    in_acc = float(accuracy_score(y[te_idx], pred))
    in_f1 = float(f1_score(y[te_idx], pred, average="macro"))

    if math_features is None or math_y is None:
        return (in_acc, in_f1, y[te_idx].tolist(), pred.tolist(),
                None, None, None, None)

    math_x = scaler.transform(math_features)
    math_pred = model.predict(math_x)
    classes = sorted(set(y.tolist()) | set(math_y.tolist()))
    math_acc = float(accuracy_score(math_y, math_pred))
    math_f1 = float(f1_score(math_y, math_pred, average="macro",
                             labels=classes, zero_division=0))
    return (in_acc, in_f1, y[te_idx].tolist(), pred.tolist(),
            math_acc, math_f1, math_y.tolist(), math_pred.tolist())


def classification_cv(
    features: np.ndarray,
    labels: list[str],
    *,
    math_features: np.ndarray | None = None,
    math_labels: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    clf: str = "lr",
    n_jobs: int = -1,
) -> tuple[float, float, float, float, dict[str, float], dict | None]:
    """Returns (mean_acc, std_acc, mean_macro_f1, std_macro_f1, per_class_f1, math).

    ``math`` is None unless math_features+math_labels are supplied; otherwise a
    dict {n_samples, accuracy, accuracy_std, macro_f1, macro_f1_std, per_class_f1}
    computed by predicting on math with each fold's trained LR (fold-shared,
    scaler-aligned with the agent train fold).

    Folds are fit in parallel via joblib (loky backend); features matrix is
    shared across workers via memmap.
    """
    y = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(features, y))
    features_c = np.ascontiguousarray(features)

    has_math = math_features is not None and math_labels is not None
    math_features_c: np.ndarray | None = None
    math_y: np.ndarray | None = None
    if has_math:
        math_features_c = np.ascontiguousarray(math_features)
        math_y = np.asarray(math_labels)

    fold_results = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes="1M")(
        delayed(_classification_fold)(
            features_c, y, tr_idx, te_idx, clf,
            math_features_c, math_y,
        )
        for tr_idx, te_idx in splits
    )

    acc = [r[0] for r in fold_results]
    f1 = [r[1] for r in fold_results]
    all_true: list[str] = []
    all_pred: list[str] = []
    for r in fold_results:
        all_true.extend(r[2])
        all_pred.extend(r[3])

    classes = sorted(set(labels))
    per_class = _per_class_f1(all_true, all_pred, classes)

    math_payload: dict | None = None
    if has_math:
        m_acc = [r[4] for r in fold_results]
        m_f1 = [r[5] for r in fold_results]
        m_true_all: list[str] = []
        m_pred_all: list[str] = []
        for r in fold_results:
            m_true_all.extend(r[6])
            m_pred_all.extend(r[7])
        m_classes = sorted(set(math_labels))
        m_per_class = _per_class_f1(m_true_all, m_pred_all, m_classes)
        math_payload = {
            "n_samples": int(len(math_labels)),
            "accuracy": float(np.mean(m_acc)),
            "accuracy_std": float(np.std(m_acc, ddof=0)),
            "macro_f1": float(np.mean(m_f1)),
            "macro_f1_std": float(np.std(m_f1, ddof=0)),
            "per_class_f1": m_per_class,
        }

    return (
        float(np.mean(acc)), float(np.std(acc, ddof=0)),
        float(np.mean(f1)), float(np.std(f1, ddof=0)),
        per_class, math_payload,
    )


def _pair_auc_cv(
    features: np.ndarray,
    pair_idx: np.ndarray,
    y_pair: np.ndarray,
    n_splits: int,
    random_state: int,
    clf: str,
) -> float | None:
    """One-pair K-fold mean AUC. Returns None if pair is degenerate.

    ``features`` is the full (N, D) matrix shared via joblib memmap; the
    worker slices it locally with ``pair_idx`` (small int array) so we don't
    pickle a per-pair copy of the data.
    """
    if y_pair.sum() == 0 or y_pair.sum() == len(y_pair):
        return None
    x_pair = features[pair_idx]
    skf = StratifiedKFold(
        n_splits=min(n_splits, int(y_pair.sum()), int(len(y_pair) - y_pair.sum())),
        shuffle=True,
        random_state=random_state,
    )
    fold_auc: list[float] = []
    for tr_idx, te_idx in skf.split(x_pair, y_pair):
        x_tr, x_te = _standardize(x_pair[tr_idx], x_pair[te_idx])
        model = _make_clf(clf)
        model.fit(x_tr, y_pair[tr_idx])
        if len(set(y_pair[te_idx])) < 2:
            continue
        score = _score_binary(model, x_te)
        fold_auc.append(roc_auc_score(y_pair[te_idx], score))
    if not fold_auc:
        return None
    return float(np.mean(fold_auc))


def pairwise_auc(
    features: np.ndarray,
    labels: list[str],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    clf: str = "lr",
    n_jobs: int = -1,
) -> tuple[float, float]:
    """Mean ± std (across class pairs) of one-vs-one binary AUC.

    Outer pair loop is parallelized via joblib (loky backend, process-level).
    The full ``features`` matrix is shared across workers via joblib memmap
    (``max_nbytes=1M``); each pair only ships a small int index array. Per-pair
    fits stay single-threaded so total CPU usage is bounded by ``n_jobs``.
    Std is taken across class pairs (each pair mean-pooled across folds).
    """
    y = np.asarray(labels)
    classes = sorted(set(labels))
    features_c = np.ascontiguousarray(features)
    pair_specs: list[tuple[np.ndarray, np.ndarray]] = []
    for a, b in combinations(classes, 2):
        mask = (y == a) | (y == b)
        pair_idx = np.flatnonzero(mask)
        pair_specs.append((pair_idx, (y[pair_idx] == a).astype(np.int8)))

    results = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes="1M")(
        delayed(_pair_auc_cv)(features_c, pair_idx, y_pair, n_splits, random_state, clf)
        for pair_idx, y_pair in pair_specs
    )
    aucs = [a for a in results if a is not None]
    if not aucs:
        return 0.0, 0.0
    return float(np.mean(aucs)), float(np.std(aucs, ddof=0))


def retrieval_map(
    features: np.ndarray,
    labels: list[str],
    *,
    k_values: tuple[int, ...] = (5, 10),
) -> dict[int, float]:
    """Leave-one-out cosine retrieval mean average precision at k."""
    y = np.asarray(labels)
    x = features.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x_n = x / norms
    sims = x_n @ x_n.T
    np.fill_diagonal(sims, -np.inf)
    maps: dict[int, float] = {}
    n = len(y)
    for k in k_values:
        ap_list: list[float] = []
        for i in range(n):
            top = np.argpartition(-sims[i], min(k, n - 1))[:k]
            # sort within top by similarity
            top = top[np.argsort(-sims[i][top])]
            rel = (y[top] == y[i]).astype(int)
            if rel.sum() == 0:
                ap_list.append(0.0)
                continue
            cum = np.cumsum(rel)
            precision_at = cum / (np.arange(len(rel)) + 1)
            ap = float((precision_at * rel).sum() / rel.sum())
            ap_list.append(ap)
        maps[k] = float(np.mean(ap_list))
    return maps


def clustering_quality(
    features: np.ndarray,
    labels: list[str],
    *,
    random_state: int = 42,
    n_init: int = 10,
) -> tuple[float, float]:
    y = np.asarray(labels)
    classes = sorted(set(labels))
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    km = KMeans(n_clusters=len(classes), random_state=random_state, n_init=n_init)
    pred = km.fit_predict(x)
    return float(adjusted_rand_score(y, pred)), float(normalized_mutual_info_score(y, pred))


def evaluate_method(
    method: str,
    features: np.ndarray,
    labels: list[str],
    *,
    math_features: np.ndarray | None = None,
    math_labels: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    clf: str = "lr",
) -> EvaluationReport:
    """Multi-metric evaluation. If ``math_features`` + ``math_labels`` are given,
    each fold's LR also predicts on math and the report's ``math_*`` fields are
    populated with classification accuracy / macro-F1 / per-class F1. Pairwise
    AUC, retrieval, and clustering remain in-distribution only (semantics on a
    fold-shared LR don't transfer cleanly to OOD)."""
    if features.ndim != 2:
        raise ValueError(f"expected 2D features; got {features.shape}")
    if math_features is not None and math_features.ndim != 2:
        raise ValueError(f"expected 2D math features; got {math_features.shape}")
    acc, acc_std, macro_f1, macro_f1_std, per_class, math_payload = classification_cv(
        features, labels,
        math_features=math_features, math_labels=math_labels,
        n_splits=n_splits, random_state=random_state, clf=clf,
    )
    auc, auc_std = pairwise_auc(features, labels, n_splits=n_splits, random_state=random_state, clf=clf)
    map_scores = retrieval_map(features, labels, k_values=(5, 10))
    ari, nmi = clustering_quality(features, labels, random_state=random_state)
    report = EvaluationReport(
        method=method,
        n_samples=features.shape[0],
        n_classes=len(set(labels)),
        classification_accuracy=acc,
        classification_accuracy_std=acc_std,
        classification_macro_f1=macro_f1,
        classification_macro_f1_std=macro_f1_std,
        mean_pairwise_auc=auc,
        mean_pairwise_auc_std=auc_std,
        retrieval_map_at_5=map_scores[5],
        retrieval_map_at_10=map_scores[10],
        clustering_ari=ari,
        clustering_nmi=nmi,
        per_class_f1=per_class,
    )
    if math_payload is not None:
        report.math_n_samples = math_payload["n_samples"]
        report.math_classification_accuracy = math_payload["accuracy"]
        report.math_classification_accuracy_std = math_payload["accuracy_std"]
        report.math_classification_macro_f1 = math_payload["macro_f1"]
        report.math_classification_macro_f1_std = math_payload["macro_f1_std"]
        report.math_per_class_f1 = math_payload["per_class_f1"]
    return report
