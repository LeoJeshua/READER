"""Run last-token activation patching and save importance-per-layer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from provenance_tracker.analysis.activation_patching import (
    ActivationPatcher,
    PatchingConfig,
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
    parser = argparse.ArgumentParser(description="Activation patching sweep over proxy layers")
    parser.add_argument("--records-path", action="append", required=True)
    parser.add_argument("--probe-bundle", required=True)
    parser.add_argument("--proxy-model-name", required=True)
    parser.add_argument("--best-layer", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    probe = load_probe(args.probe_bundle)
    records = _records(args.records_path)
    print(f"[patch] records={len(records)} classes={probe.class_names}")

    cfg = PatchingConfig(
        proxy_model_name_or_path=args.proxy_model_name,
        best_layer=args.best_layer,
        probe_weights=probe.weights,
        probe_bias=probe.bias,
        probe_mean=probe.mean,
        probe_std=probe.std,
        class_names=probe.class_names,
        device=args.device,
        dtype=args.dtype,
    )
    patcher = ActivationPatcher(cfg)
    try:
        result = patcher.run(
            records, num_pairs=args.num_pairs, layer_stride=args.layer_stride
        )
    finally:
        patcher.close()

    payload = {
        "layer_indices": result.layer_indices,
        "importance": result.importance.tolist(),
        "clean_target_prob": result.clean_target_prob,
        "clean_source_prob": result.clean_source_prob,
        "class_names": result.class_names,
        "num_pairs": result.num_pairs,
        "best_layer": args.best_layer,
    }
    out = ensure_parent_dir(args.output_json)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[patch] wrote {out}  clean={result.clean_target_prob:.3f}  src={result.clean_source_prob:.3f}")


if __name__ == "__main__":
    main()
