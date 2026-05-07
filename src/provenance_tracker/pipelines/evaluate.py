"""Unified evaluation harness.

Reads one or more feature .npz files (mpnet, proxy multi-layer, etc.), projects
each into a 2D (N, D) representation via a declared ``--view``, and reports a
common set of metrics so methods are directly comparable.

Views:
  - ``flat``         : use features as-is (must be 2D already; mpnet)
  - ``last_layer``   : pick layer_index=-1 from (N, L, D)
  - ``layer=<i>``    : pick layer i from (N, L, D)
  - ``best_layer``   : pick layer with best probe accuracy (requires labels)
  - ``concat_layers`` : concat layers [start, end) from (N, L, D)
  - ``intra_m=<m>``  : from (N, M_max, D) intra-position features, uniformly
                       subsample m positions (last-endpoint preserved) and
                       mean-pool to get (N, D). m=1 reproduces the last sampled
                       position (≈ last-token baseline).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from provenance_tracker.analysis.circuit import layerwise_probe_accuracy
from provenance_tracker.evaluation.metrics import evaluate_method
from provenance_tracker.utils.io import ensure_parent_dir, load_feature_batch


def _aggregate(x: np.ndarray, labels: list[str], k: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """Per-class shuffle then mean-pool every K consecutive samples.

    Drops the tail remainder if a class size isn't divisible by K so every
    aggregated vector averages exactly K samples.
    """
    rng = np.random.default_rng(seed)
    by_cls: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_cls.setdefault(lab, []).append(i)
    agg_x: list[np.ndarray] = []
    agg_y: list[str] = []
    for lab, idxs in by_cls.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_groups = len(idxs) // k
        if n_groups == 0:
            continue
        trimmed = idxs[: n_groups * k].reshape(n_groups, k)
        for row in trimmed:
            agg_x.append(x[row].mean(axis=0))
            agg_y.append(lab)
    return np.stack(agg_x, axis=0), agg_y


def _resolve_view(features: np.ndarray, labels: list[str], view: str) -> np.ndarray:
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
        print(f"[evaluate] best_layer chosen: {best.layer_index} (acc {best.accuracy:.3f})")
        return features[:, best.layer_index, :]
    if view.startswith("concat_layers"):
        # concat_layers=start:end
        body = view.split("=", 1)[1] if "=" in view else "0:-1"
        start_s, end_s = body.split(":")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else features.shape[1]
        if end < 0:
            end = features.shape[1] + end
        return features[:, start:end, :].reshape(features.shape[0], -1)
    if view.startswith("intra_m="):
        m = int(view.split("=", 1)[1])
        m_max = features.shape[1]
        if m < 1 or m > m_max:
            raise ValueError(f"intra_m={m} must be in [1, {m_max}]")
        if m == 1:
            # last stored position — matches the old last-token baseline
            idx = np.array([m_max - 1], dtype=np.int64)
        else:
            idx = np.round(np.linspace(0, m_max - 1, m)).astype(np.int64)
            idx = np.unique(idx)
        print(f"[evaluate] intra_m={m} positions={idx.tolist()} (of M_max={m_max})")
        return features[:, idx, :].mean(axis=1)
    raise ValueError(f"unknown view {view!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified provenance evaluation")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--math-features-path", default=None,
                        help="Optional OOD math features. Same view + aggregate-k "
                             "applied; classification metrics computed by predicting "
                             "on math with each fold's trained LR (fold + scaler "
                             "shared with the in-dist eval).")
    parser.add_argument("--view", default="flat")
    parser.add_argument("--method-name", required=True,
                        help="Label used in the output (e.g. 'mpnet', 'proxy_last', 'proxy_best')")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--aggregate-k", type=int, default=1,
                        help="If >1, mean-pool every K per-class samples before metric eval")
    parser.add_argument("--clf", choices=["lr", "svm"], default="lr",
                        help="Classification head: 'lr' (default, LogisticRegression) or "
                             "'svm' (LinearSVC, matches LLM-DNA's hinge-loss head)")
    args = parser.parse_args()

    batch = load_feature_batch(args.features_path)
    # If view is best_layer + math is provided, freeze the layer choice on agent
    # so math uses the same layer (otherwise math would pick its own best_layer
    # and the OOD comparison would be apples-to-oranges).
    resolved_view = args.view
    if args.view == "best_layer" and batch.features.ndim == 3 and args.math_features_path:
        probes = layerwise_probe_accuracy(batch.features, batch.labels, n_splits=5)
        best = max(probes, key=lambda r: r.accuracy)
        resolved_view = f"layer={best.layer_index}"
        print(f"[evaluate] best_layer pinned to layer={best.layer_index} for math alignment "
              f"(agent probe acc {best.accuracy:.3f})")
    x = _resolve_view(batch.features, batch.labels, resolved_view)
    eval_labels = batch.labels

    math_x: np.ndarray | None = None
    math_eval_labels: list[str] | None = None
    if args.math_features_path:
        math_batch = load_feature_batch(args.math_features_path)
        # Restrict math to labels also present in agent (otherwise OOD predictions
        # for unknown classes are ill-defined under the agent-trained LR).
        agent_classes = set(batch.labels)
        m_keep = np.array([l in agent_classes for l in math_batch.labels])
        if not m_keep.any():
            raise SystemExit("no common labels between agent and math feature batches")
        math_features_kept = math_batch.features[m_keep]
        math_labels_kept = [l for l, k in zip(math_batch.labels, m_keep) if k]
        dropped = len(math_batch.labels) - int(m_keep.sum())
        if dropped:
            print(f"[evaluate] math dropped {dropped} rows whose labels are not in agent")
        math_x = _resolve_view(math_features_kept, math_labels_kept, resolved_view)
        math_eval_labels = math_labels_kept

    if args.aggregate_k and args.aggregate_k > 1:
        x, eval_labels = _aggregate(x, batch.labels, args.aggregate_k, args.random_state)
        if math_x is not None and math_eval_labels is not None:
            math_x, math_eval_labels = _aggregate(
                math_x, math_eval_labels, args.aggregate_k, args.random_state
            )
        print(f"[evaluate] aggregate_k={args.aggregate_k} -> agent {x.shape}"
              f"{f' math {math_x.shape}' if math_x is not None else ''}")
    print(f"[evaluate] method={args.method_name} view={args.view} x.shape={x.shape} clf={args.clf}"
          f"{f' math.shape={math_x.shape}' if math_x is not None else ''}")
    n_splits = min(args.n_splits, min(eval_labels.count(c) for c in set(eval_labels)))
    report = evaluate_method(
        args.method_name, x, eval_labels,
        math_features=math_x, math_labels=math_eval_labels,
        n_splits=n_splits, random_state=args.random_state, clf=args.clf,
    )
    output_path = ensure_parent_dir(args.output_json)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report.as_flat_dict(), f, ensure_ascii=False, indent=2)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
