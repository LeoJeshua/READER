"""Cross-domain evaluation: train on agent prompts, test on agent (in-dist) + math (OOD).

For each fold of a stratified K-fold on the agent feature batch:
  * fit StandardScaler + LR on the train rows
  * predict on the held-out agent test rows  → in-distribution metrics
  * predict on the *entire* math feature batch → out-of-distribution metrics

Cross-fold mean and std are reported for accuracy / macro-F1 on both views,
plus per-class F1 from concatenated predictions.

Both feature batches must use the same label set (script will intersect and
warn if not). Mirrors ``evaluate.py``'s view system so any feature shape
(``flat`` / ``last_layer`` / ``layer=i`` / ``best_layer`` / ``intra_m=m``)
is supported.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from provenance_tracker.analysis.circuit import layerwise_probe_accuracy
from provenance_tracker.utils.io import ensure_parent_dir, load_feature_batch


def _resolve_view(features: np.ndarray, labels: list[str], view: str) -> np.ndarray:
    """Subset of evaluate.py:_resolve_view (kept local to avoid import cycle)."""
    if view == "flat":
        if features.ndim != 2:
            raise ValueError(f"flat view requires 2D; got {features.shape}")
        return features
    if features.ndim != 3:
        raise ValueError(f"view {view!r} requires 3D (N,L,D); got {features.shape}")
    if view == "last_layer":
        return features[:, -1, :]
    if view.startswith("layer="):
        idx = int(view.split("=", 1)[1])
        return features[:, idx, :]
    if view == "best_layer":
        probes = layerwise_probe_accuracy(features, labels, n_splits=5)
        best = max(probes, key=lambda r: r.accuracy)
        print(f"[crossdomain] best_layer chosen on agent: {best.layer_index} (acc {best.accuracy:.3f})")
        return features[:, best.layer_index, :]
    if view.startswith("intra_m="):
        m = int(view.split("=", 1)[1])
        m_max = features.shape[1]
        if m < 1 or m > m_max:
            raise ValueError(f"intra_m={m} must be in [1, {m_max}]")
        if m == 1:
            idx = np.array([m_max - 1], dtype=np.int64)
        else:
            idx = np.unique(np.round(np.linspace(0, m_max - 1, m)).astype(np.int64))
        return features[:, idx, :].mean(axis=1)
    raise ValueError(f"unknown view {view!r}")


def _per_class_f1(true: list[str], pred: list[str], classes: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(true, pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true, pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true, pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[cls] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return out


@dataclass(slots=True)
class DomainScores:
    n_samples: int
    accuracy: float
    accuracy_std: float
    macro_f1: float
    macro_f1_std: float
    per_class_f1: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "accuracy_std": self.accuracy_std,
            "macro_f1": self.macro_f1,
            "macro_f1_std": self.macro_f1_std,
            "per_class_f1": dict(self.per_class_f1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-domain agent→math evaluation")
    parser.add_argument("--agent-features-path", required=True)
    parser.add_argument("--math-features-path", required=True)
    parser.add_argument("--view", default="last_layer")
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"[crossdomain] loading agent features: {args.agent_features_path}")
    agent = load_feature_batch(args.agent_features_path)
    print(f"[crossdomain] loading math features: {args.math_features_path}")
    math = load_feature_batch(args.math_features_path)

    # Restrict to common labels.
    agent_classes = set(agent.labels)
    math_classes = set(math.labels)
    common = sorted(agent_classes & math_classes)
    if not common:
        raise SystemExit("no common labels between agent and math feature batches")
    if agent_classes != math_classes:
        missing_agent = math_classes - agent_classes
        missing_math = agent_classes - math_classes
        if missing_agent:
            print(f"[crossdomain] WARN labels in math but not agent: {sorted(missing_agent)}")
        if missing_math:
            print(f"[crossdomain] WARN labels in agent but not math: {sorted(missing_math)}")

    common_set = set(common)
    a_keep = np.array([l in common_set for l in agent.labels])
    m_keep = np.array([l in common_set for l in math.labels])
    agent_feats = agent.features[a_keep]
    agent_labels = [l for l, k in zip(agent.labels, a_keep) if k]
    math_feats = math.features[m_keep]
    math_labels = [l for l, k in zip(math.labels, m_keep) if k]

    a_view = _resolve_view(agent_feats, agent_labels, args.view)
    m_view = _resolve_view(math_feats, math_labels, args.view)
    print(f"[crossdomain] agent view shape={a_view.shape} math view shape={m_view.shape}")

    y_a = np.asarray(agent_labels)
    y_m = np.asarray(math_labels)
    classes = sorted(common)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    a_acc: list[float] = []
    a_f1: list[float] = []
    m_acc: list[float] = []
    m_f1: list[float] = []
    a_true_all: list[str] = []
    a_pred_all: list[str] = []
    m_true_all: list[str] = []
    m_pred_all: list[str] = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(a_view, y_a)):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(a_view[tr_idx])
        x_te_a = scaler.transform(a_view[te_idx])
        x_te_m = scaler.transform(m_view)
        clf = LogisticRegression(max_iter=3000, solver="lbfgs")
        clf.fit(x_tr, y_a[tr_idx])
        pred_a = clf.predict(x_te_a)
        pred_m = clf.predict(x_te_m)
        acc_a = accuracy_score(y_a[te_idx], pred_a)
        f1_a = f1_score(y_a[te_idx], pred_a, average="macro", labels=classes, zero_division=0)
        acc_m = accuracy_score(y_m, pred_m)
        f1_m = f1_score(y_m, pred_m, average="macro", labels=classes, zero_division=0)
        print(f"[crossdomain] fold {fold}: agent_acc={acc_a:.4f} math_acc={acc_m:.4f}  agent_f1={f1_a:.4f} math_f1={f1_m:.4f}")
        a_acc.append(acc_a); a_f1.append(f1_a)
        m_acc.append(acc_m); m_f1.append(f1_m)
        a_true_all.extend(y_a[te_idx].tolist()); a_pred_all.extend(pred_a.tolist())
        m_true_all.extend(y_m.tolist()); m_pred_all.extend(pred_m.tolist())

    in_dist = DomainScores(
        n_samples=int(a_view.shape[0]),
        accuracy=float(np.mean(a_acc)),
        accuracy_std=float(np.std(a_acc, ddof=0)),
        macro_f1=float(np.mean(a_f1)),
        macro_f1_std=float(np.std(a_f1, ddof=0)),
        per_class_f1=_per_class_f1(a_true_all, a_pred_all, classes),
    )
    ood = DomainScores(
        n_samples=int(m_view.shape[0]),
        accuracy=float(np.mean(m_acc)),
        accuracy_std=float(np.std(m_acc, ddof=0)),
        macro_f1=float(np.mean(m_f1)),
        macro_f1_std=float(np.std(m_f1, ddof=0)),
        per_class_f1=_per_class_f1(m_true_all, m_pred_all, classes),
    )
    payload = {
        "method": args.method_name,
        "view": args.view,
        "n_splits": args.n_splits,
        "n_classes": len(classes),
        "in_distribution_agent": in_dist.to_dict(),
        "out_of_distribution_math": ood.to_dict(),
    }
    out_path = ensure_parent_dir(args.output_json)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in payload.items() if k != "in_distribution_agent" and k != "out_of_distribution_math"}, ensure_ascii=False))
    print(f"in_dist: acc={in_dist.accuracy:.4f}±{in_dist.accuracy_std:.4f}  f1={in_dist.macro_f1:.4f}±{in_dist.macro_f1_std:.4f}")
    print(f"OOD math: acc={ood.accuracy:.4f}±{ood.accuracy_std:.4f}  f1={ood.macro_f1:.4f}±{ood.macro_f1_std:.4f}")


if __name__ == "__main__":
    main()
