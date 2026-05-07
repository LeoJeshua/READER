"""9-metric scoring for log-posterior K-sweep outputs.

Combines:

* **Classifier-head metrics** — Acc / Macro-F1 / NLL / ECE / per-class F1
  computed on the per-fold ``(y, logp)`` pairs returned by
  ``run_logposterior_*_sweep_full``. Two flavours of Acc/F1 are emitted:

    - ``classification_accuracy``: concat-fold one-shot scoring (bit-matches
      ``_score_log_scores`` inside multi_traj.py — same numbers as the
      ``aggregators[].accuracy`` field that lives in
      ``reports_multi_traj_intra``).
    - ``classification_accuracy_fold_mean`` ± ``..._fold_std``: per-fold
      stratified Acc/F1 (matches the ``classification_accuracy`` field
      in ``reports_intra/`` produced by ``evaluate.py``).

* **Geometric metrics** — Pair-AUC ± std / mAP@5 / mAP@10 / ARI / NMI
  computed on K-pooled mean-pool fingerprint features. These do not depend
  on the classifier head, so they are constant across ``meanpool_lr``,
  ``logposterior``, and ``gaussian_disc`` aggregators (given the same
  fingerprint partition). This matches the ``reports_intra`` protocol
  exactly so cross-aggregator comparisons stay apples-to-apples on the
  geometric metrics.

The fingerprint features used for the geometric metrics are built
independently from the LR sweep — the LR sweep uses per-fold-rotated
fingerprints whereas geometric metrics need a single fixed
``(N_fp, D)`` matrix to feed ``pairwise_auc`` / ``retrieval_map`` /
``clustering_quality``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

from provenance_tracker.evaluation.metrics import (
    clustering_quality,
    pairwise_auc,
    retrieval_map,
)
from provenance_tracker.evaluation.multi_traj import (
    LogPostRawSweep,
    _normalise_logps,
    expected_calibration_error,
)


def _per_class_f1_int(y_true: np.ndarray, y_pred: np.ndarray,
                      classes: list[str]) -> dict[str, float]:
    """Per-class F1 keyed by class string. ``y_true``/``y_pred`` are int idx."""
    out: dict[str, float] = {}
    for ci, cls in enumerate(classes):
        tp = int(((y_true == ci) & (y_pred == ci)).sum())
        fp = int(((y_true != ci) & (y_pred == ci)).sum())
        fn = int(((y_true == ci) & (y_pred != ci)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[cls] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return out


def _build_fp_features_intra(
    x_intra: np.ndarray,
    labels: list[str],
    K: int,
    M: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Build (N_fp, D) mean-pool fingerprint features from intra (N, M_max, D).

    Per-class shuffle (seed=``seed + 4242 + K``) → chunk into K-groups → for
    each group mean-pool over K samples × M intra positions (uniform sampling
    matching ``_intra_positions``). Returns ``(fp_x, fp_labels)``.
    """
    from provenance_tracker.evaluation.multi_traj import _intra_positions

    pos = _intra_positions(M, x_intra.shape[1])
    D = x_intra.shape[2]
    rng = np.random.default_rng(seed + 4242 + K)
    by_cls: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_cls.setdefault(lab, []).append(i)
    fp_x: list[np.ndarray] = []
    fp_y: list[str] = []
    for cls, idxs in by_cls.items():
        arr = np.asarray(idxs)
        rng.shuffle(arr)
        n_g = len(arr) // K
        if n_g == 0:
            continue
        chunked = arr[: n_g * K].reshape(n_g, K)
        for row in chunked:
            tokens = x_intra[np.ix_(row, pos)].reshape(-1, D)
            fp_x.append(tokens.mean(axis=0))
            fp_y.append(cls)
    return np.stack(fp_x, axis=0), fp_y


def _build_fp_features_flat(
    x: np.ndarray,
    labels: list[str],
    K: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Flat-feature analogue: mean-pool K trajectories per fingerprint."""
    rng = np.random.default_rng(seed + 4242 + K)
    by_cls: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_cls.setdefault(lab, []).append(i)
    fp_x: list[np.ndarray] = []
    fp_y: list[str] = []
    for cls, idxs in by_cls.items():
        arr = np.asarray(idxs)
        rng.shuffle(arr)
        n_g = len(arr) // K
        if n_g == 0:
            continue
        chunked = arr[: n_g * K].reshape(n_g, K)
        for row in chunked:
            fp_x.append(x[row].mean(axis=0))
            fp_y.append(cls)
    return np.stack(fp_x, axis=0), fp_y


def score_lp_classifier(raw: LogPostRawSweep) -> dict:
    """Classifier-head metrics from a :class:`LogPostRawSweep`.

    Returns a dict with both fold-mean ± std (parallel to ``evaluate.py``'s
    ``classification_*`` fields) AND concat-fold one-shot scoring (parallel
    to ``aggregators[].accuracy`` in ``reports_multi_traj_intra``). NLL/ECE
    are concat-fold only since per-fold ECE is too noisy for K=1 (small N).
    """
    classes = list(raw.classes)
    C = len(classes)
    label_range = list(range(C))

    fold_acc: list[float] = []
    fold_f1: list[float] = []
    for y, logp in zip(raw.y_per_fold, raw.logp_per_fold):
        if len(y) == 0:
            continue
        p = _normalise_logps(logp)
        pred = p.argmax(axis=1)
        fold_acc.append(float(accuracy_score(y, pred)))
        fold_f1.append(float(f1_score(y, pred, average="macro",
                                      labels=label_range, zero_division=0)))

    y_concat = np.concatenate(raw.y_per_fold) if raw.y_per_fold else np.array([], int)
    logp_concat = (np.concatenate(raw.logp_per_fold, axis=0)
                   if raw.logp_per_fold else np.zeros((0, C)))
    if y_concat.size == 0:
        return {
            "classification_accuracy": float("nan"),
            "classification_accuracy_fold_mean": float("nan"),
            "classification_accuracy_fold_std": float("nan"),
            "classification_macro_f1": float("nan"),
            "classification_macro_f1_fold_mean": float("nan"),
            "classification_macro_f1_fold_std": float("nan"),
            "nll": float("nan"),
            "ece": float("nan"),
            "alpha_calib": raw.alpha,
            "per_class_f1": {c: 0.0 for c in classes},
            "n_fingerprints": 0,
        }
    p_concat = _normalise_logps(logp_concat)
    pred_concat = p_concat.argmax(axis=1)
    concat_acc = float(accuracy_score(y_concat, pred_concat))
    concat_f1 = float(f1_score(y_concat, pred_concat, average="macro",
                               labels=label_range, zero_division=0))
    nll = float(log_loss(y_concat, p_concat, labels=label_range))
    ece = float(expected_calibration_error(y_concat, p_concat))
    per_class = _per_class_f1_int(y_concat, pred_concat, classes)

    return {
        "classification_accuracy": concat_acc,
        "classification_accuracy_fold_mean": float(np.mean(fold_acc)) if fold_acc else float("nan"),
        "classification_accuracy_fold_std": float(np.std(fold_acc, ddof=0)) if fold_acc else float("nan"),
        "classification_macro_f1": concat_f1,
        "classification_macro_f1_fold_mean": float(np.mean(fold_f1)) if fold_f1 else float("nan"),
        "classification_macro_f1_fold_std": float(np.std(fold_f1, ddof=0)) if fold_f1 else float("nan"),
        "nll": nll,
        "ece": ece,
        "alpha_calib": raw.alpha,
        "per_class_f1": per_class,
        "n_fingerprints": int(y_concat.size),
    }


def score_lp_geometric(
    fp_features: np.ndarray,
    fp_labels: list[str],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    clf: str = "lr",
    n_jobs: int = -1,
) -> dict:
    """Geometric metrics on (N_fp, D) mean-pool fingerprint features."""
    if fp_features.shape[0] == 0:
        return {
            "mean_pairwise_auc": float("nan"),
            "mean_pairwise_auc_std": float("nan"),
            "retrieval_map_at_5": float("nan"),
            "retrieval_map_at_10": float("nan"),
            "clustering_ari": float("nan"),
            "clustering_nmi": float("nan"),
            "n_fp_geom": 0,
        }
    pa, pa_std = pairwise_auc(
        fp_features, fp_labels,
        n_splits=n_splits, random_state=random_state, clf=clf, n_jobs=n_jobs,
    )
    maps = retrieval_map(fp_features, fp_labels, k_values=(5, 10))
    ari, nmi = clustering_quality(fp_features, fp_labels, random_state=random_state)
    return {
        "mean_pairwise_auc": float(pa),
        "mean_pairwise_auc_std": float(pa_std),
        "retrieval_map_at_5": float(maps[5]),
        "retrieval_map_at_10": float(maps[10]),
        "clustering_ari": float(ari),
        "clustering_nmi": float(nmi),
        "n_fp_geom": int(fp_features.shape[0]),
    }


def assemble_lp_report(
    method: str,
    aggregator: str,
    K: int,
    M: int | None,
    raw: LogPostRawSweep,
    fp_features: np.ndarray,
    fp_labels: list[str],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    clf: str = "lr",
    n_jobs: int = -1,
    n_samples: int | None = None,
) -> dict:
    """Combine classifier + geometric metrics into a single 9-metric record.

    Output schema mirrors ``reports_intra/<tag>_*.json`` so downstream
    plotting code keeps working unchanged. Adds ``aggregator``, ``M``, ``K``,
    ``n_fp_geom`` for traceability.
    """
    cls_metrics = score_lp_classifier(raw)
    geom_metrics = score_lp_geometric(
        fp_features, fp_labels,
        n_splits=n_splits, random_state=random_state, clf=clf, n_jobs=n_jobs,
    )
    out: dict = {
        "method": method,
        "aggregator": aggregator,
        "K": int(K),
        "M": int(M) if M is not None else None,
        "n_samples": int(n_samples if n_samples is not None
                         else cls_metrics["n_fingerprints"]),
        "n_classes": len(raw.classes),
    }
    out.update({k: v for k, v in cls_metrics.items() if k != "per_class_f1"})
    out.update(geom_metrics)
    out["per_class_f1"] = cls_metrics["per_class_f1"]
    return out


__all__ = [
    "score_lp_classifier",
    "score_lp_geometric",
    "assemble_lp_report",
    "_build_fp_features_intra",
    "_build_fp_features_flat",
]
