"""Construct the original LLM-DNA concat--fixed-GRP model vectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reader_provenance.baselines.pairwise import fixed_gaussian_projection
from reader_provenance.features.io import load_features


def build_vectors(
    feature_path: Path,
    output: Path,
    *,
    n_components: int,
    seed: int,
    device: str,
) -> None:
    batch = load_features(feature_path)
    if batch.features.ndim != 2:
        raise ValueError("LLM-DNA expects flat response embeddings")
    grouped: dict[str, list[tuple[str, np.ndarray]]] = {}
    for sample_id, label, feature in zip(
        batch.sample_ids,
        batch.labels,
        batch.features,
        strict=True,
    ):
        grouped.setdefault(label, []).append((sample_id, feature))
    arrays = {}
    expected_ids = None
    for label, rows in grouped.items():
        rows.sort(key=lambda row: row[0])
        sample_ids = [sample_id for sample_id, _ in rows]
        if expected_ids is None:
            expected_ids = sample_ids
        elif sample_ids != expected_ids:
            raise ValueError(f"unaligned response embeddings for {label}")
        arrays[label] = np.stack([feature for _, feature in rows])
    vectors = fixed_gaussian_projection(
        arrays,
        n_components=n_components,
        seed=seed,
        device=device,
    )
    names = sorted(vectors)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        model_names=np.asarray(names, dtype=object),
        vectors=np.stack([vectors[name] for name in names]),
        metadata=np.asarray(
            json.dumps(
                {
                    "protocol": "llm_dna_concat_fixed_grp_v1",
                    "source": str(feature_path),
                    "input_shape_per_model": list(arrays[names[0]].shape),
                    "n_components": n_components,
                    "seed": seed,
                    "standardize_before_projection": False,
                },
                sort_keys=True,
            ),
            dtype=object,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    build_vectors(
        args.features,
        args.output,
        n_components=args.components,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
