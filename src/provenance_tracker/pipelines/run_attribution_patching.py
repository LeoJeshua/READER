"""Run attribution patching across samples and save a (L, last_k_positions) heatmap."""
from __future__ import annotations

import argparse
import json

import numpy as np

from provenance_tracker.analysis.attribution_patching import (
    AttributionConfig,
    AttributionPatcher,
)
from provenance_tracker.analysis.probe_utils import load_probe
from provenance_tracker.datasets.schemas import ResponseRecord
from provenance_tracker.utils.io import ensure_parent_dir, read_jsonl


def _records(paths: list[str]) -> list[ResponseRecord]:
    out: list[ResponseRecord] = []
    for p in paths:
        out.extend(ResponseRecord(**d) for d in read_jsonl(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribution patching heatmap")
    parser.add_argument("--records-path", action="append", required=True)
    parser.add_argument("--probe-bundle", required=True)
    parser.add_argument("--proxy-model-name", required=True)
    parser.add_argument("--best-layer", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-heatmap", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--last-k-positions", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    probe = load_probe(args.probe_bundle)
    records = _records(args.records_path)
    if args.max_samples and len(records) > args.max_samples:
        # Evenly subsample while preserving class balance.
        import random
        random.Random(42).shuffle(records)
        records = records[: args.max_samples]
    print(f"[attr] records={len(records)} classes={probe.class_names}")

    cfg = AttributionConfig(
        proxy_model_name_or_path=args.proxy_model_name,
        best_layer=args.best_layer,
        probe_weights=probe.weights,
        probe_bias=probe.bias,
        probe_mean=probe.mean,
        probe_std=probe.std,
        class_names=probe.class_names,
        device=args.device,
        dtype=args.dtype,
        last_k_positions=args.last_k_positions,
    )
    patcher = AttributionPatcher(cfg)
    try:
        result = patcher.run(records)
    finally:
        patcher.close()

    payload = {
        "layer_indices": result.layer_indices,
        "per_layer_total": result.per_layer_total.tolist(),
        "class_names": result.class_names,
        "num_samples": result.num_samples,
        "best_layer": args.best_layer,
        "last_k_positions": args.last_k_positions,
    }
    out = ensure_parent_dir(args.output_json)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    np.save(args.output_heatmap, result.heatmap)
    print(f"[attr] wrote {out} heatmap={result.heatmap.shape}")


if __name__ == "__main__":
    main()
