from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from provenance_tracker.analysis.circuit import (
    class_conditioned_activation_matrix,
    layerwise_probe_accuracy,
    top_discriminative_dims,
)
from provenance_tracker.utils.io import ensure_parent_dir, load_feature_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-wise probing and dimension importance over proxy hidden states"
    )
    parser.add_argument("--proxy-features", required=True,
                        help=".npz from extract_proxy (multi-layer)")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k-dims", type=int, default=32)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    batch = load_feature_batch(args.proxy_features)
    features = batch.features
    if features.ndim != 3:
        raise SystemExit(f"expected (N, L, D) multi-layer features; got {features.shape}")

    print(f"[circuit] features={features.shape} classes={sorted(set(batch.labels))}")
    probes = layerwise_probe_accuracy(
        features,
        batch.labels,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    best = max(probes, key=lambda r: r.accuracy)
    print(f"[circuit] best layer={best.layer_index} acc={best.accuracy:.4f}")

    top_dims = top_discriminative_dims(
        features[:, best.layer_index, :], batch.labels, top_k=args.top_k_dims
    )
    matrix, classes = class_conditioned_activation_matrix(
        features[:, best.layer_index, :], batch.labels, top_dims
    )

    payload = {
        "layer_probes": [
            {"layer_index": r.layer_index, "accuracy": r.accuracy, "macro_f1": r.macro_f1}
            for r in probes
        ],
        "best_layer": best.layer_index,
        "best_accuracy": best.accuracy,
        "best_macro_f1": best.macro_f1,
        "top_dims": top_dims.tolist(),
        "class_order": classes,
        "activation_matrix": matrix.tolist(),
        "feature_shape": list(features.shape),
    }
    output_path = ensure_parent_dir(args.output_json)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    best_layer_feats = features[:, best.layer_index, :].astype(np.float32)
    np.save(output_path.with_suffix(".best_layer.npy"), best_layer_feats)
    print(f"[circuit] wrote {output_path}")


if __name__ == "__main__":
    main()
