"""Evaluate a proxy fingerprint under a fixed-stride token subsampling.

For each sample, take positions ``[end-1, end-1-Δ, end-1-2Δ, ...]`` (anchored
at the last response token, step backwards by ``Δ``) clipped to the valid
span, then mean-pool into a single D-dim vector. Numbers of positions vary per
sample; CV still uses N=720.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from provenance_tracker.evaluation.metrics import evaluate_method
from provenance_tracker.pipelines.evaluate import _aggregate
from provenance_tracker.proxy.all_positions import load_all_positions
from provenance_tracker.utils.io import ensure_parent_dir


def stride_mean_pool(
    features: np.ndarray,          # (N, T_max, D), zero-padded
    valid_counts: np.ndarray,      # (N,) int32
    stride: int,
    anchor: str = "end",           # "end" | "start"
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pooled (N, D), positions_per_sample (N,))``.

    ``anchor="end"``: last token always included, step backwards by `stride`.
    ``anchor="start"``: first token always included, step forwards.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    N, _, D = features.shape
    pooled = np.zeros((N, D), dtype=np.float32)
    per_sample = np.zeros(N, dtype=np.int32)
    for i in range(N):
        vc = int(valid_counts[i])
        if vc <= 0:
            continue
        if anchor == "end":
            pos = np.arange(vc - 1, -1, -stride, dtype=np.int64)[::-1]
        elif anchor == "start":
            pos = np.arange(0, vc, stride, dtype=np.int64)
        else:
            raise ValueError(f"anchor must be 'end' or 'start', got {anchor!r}")
        pooled[i] = features[i, pos, :].mean(axis=0)
        per_sample[i] = len(pos)
    return pooled, per_sample


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stride-based intra-position evaluation for a proxy feature batch"
    )
    p.add_argument("--features-path", required=True,
                   help="NPZ produced by extract_all_positions (has valid_counts)")
    p.add_argument("--stride", type=int, required=True,
                   help="Token stride Δ. stride=1 means pool over every response token.")
    p.add_argument("--anchor", choices=("end", "start"), default="end")
    p.add_argument("--method-name", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--aggregate-k", type=int, default=1,
                   help="Optional cross-sample mean-pool on top of stride pool")
    args = p.parse_args()

    batch, counts = load_all_positions(args.features_path)
    x, per_sample = stride_mean_pool(batch.features, counts, args.stride, args.anchor)
    print(
        f"[stride] stride={args.stride} anchor={args.anchor} "
        f"positions/sample: min={per_sample.min()} mean={per_sample.mean():.1f} "
        f"max={per_sample.max()}"
    )

    labels = batch.labels
    if args.aggregate_k and args.aggregate_k > 1:
        x, labels = _aggregate(x, batch.labels, args.aggregate_k, args.random_state)
        print(f"[stride] aggregate_k={args.aggregate_k} -> {x.shape}")
    print(f"[stride] x.shape={x.shape}")

    n_splits = min(args.n_splits, min(labels.count(c) for c in set(labels)))
    report = evaluate_method(
        args.method_name, x, labels, n_splits=n_splits, random_state=args.random_state
    )
    output_path = ensure_parent_dir(args.output_json)
    payload = report.as_flat_dict()
    payload["stride"] = args.stride
    payload["anchor"] = args.anchor
    payload["aggregate_k"] = args.aggregate_k
    payload["mean_positions_per_sample"] = float(per_sample.mean())
    payload["min_positions_per_sample"] = int(per_sample.min())
    payload["max_positions_per_sample"] = int(per_sample.max())
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
