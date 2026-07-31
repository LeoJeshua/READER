"""Cross-fitted source-to-prompt variance decomposition."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from reader_provenance.evaluation.protocol import prompt_grouped_folds
from reader_provenance.features.io import load_features

PROTOCOL = "reader_source_prompt_variance_v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def two_way_variance(
    features: np.ndarray,
    source_labels: np.ndarray,
    prompt_ids: np.ndarray,
) -> dict[str, float | int]:
    """Decompose a balanced source-by-prompt panel into crossed main effects."""
    values = np.asarray(features, dtype=np.float64)
    sources, source_inverse = np.unique(source_labels, return_inverse=True)
    prompts, prompt_inverse = np.unique(prompt_ids, return_inverse=True)
    if values.ndim != 2 or len(values) != len(source_labels):
        raise ValueError("features and labels must be aligned 2-D arrays")
    if len(values) != len(sources) * len(prompts):
        raise ValueError("held-out panel is not a complete source-by-prompt grid")
    pair_ids = source_inverse * len(prompts) + prompt_inverse
    if not np.all(
        np.bincount(pair_ids, minlength=len(sources) * len(prompts)) == 1
    ):
        raise ValueError("each source/prompt cell must occur exactly once")

    grand_mean = values.mean(axis=0)
    source_sums = np.zeros((len(sources), values.shape[1]), dtype=np.float64)
    prompt_sums = np.zeros((len(prompts), values.shape[1]), dtype=np.float64)
    np.add.at(source_sums, source_inverse, values)
    np.add.at(prompt_sums, prompt_inverse, values)
    source_effects = source_sums / len(prompts) - grand_mean
    prompt_effects = prompt_sums / len(sources) - grand_mean
    source_ss = float(
        len(prompts) * np.einsum("ij,ij->", source_effects, source_effects)
    )
    prompt_ss = float(
        len(sources) * np.einsum("ij,ij->", prompt_effects, prompt_effects)
    )
    centered = values - grand_mean
    total_ss = float(np.einsum("ij,ij->", centered, centered))
    residual_ss = max(total_ss - source_ss - prompt_ss, 0.0)
    normalizer = float(values.size)
    source_variance = source_ss / normalizer
    prompt_variance = prompt_ss / normalizer
    residual_variance = residual_ss / normalizer
    epsilon = np.finfo(np.float64).eps
    return {
        "n_samples": len(values),
        "n_sources": len(sources),
        "n_prompts": len(prompts),
        "n_dimensions": values.shape[1],
        "source_variance": source_variance,
        "prompt_variance": prompt_variance,
        "residual_variance": residual_variance,
        "source_prompt_ratio": source_variance / max(prompt_variance, epsilon),
        "source_fraction": source_ss / max(total_ss, epsilon),
        "prompt_fraction": prompt_ss / max(total_ss, epsilon),
        "residual_fraction": residual_ss / max(total_ss, epsilon),
    }


def _aggregate(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "n_folds": len(rows),
        "n_dimensions": int(rows[0]["n_dimensions"]),
    }
    for metric in ("source_variance", "prompt_variance", "residual_variance"):
        values = np.asarray([float(row[metric]) for row in rows])
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_std"] = float(values.std(ddof=0))
    ratios = np.asarray([float(row["source_prompt_ratio"]) for row in rows])
    source_sum = sum(float(row["source_variance"]) for row in rows)
    prompt_sum = sum(float(row["prompt_variance"]) for row in rows)
    output.update(
        {
            "source_prompt_ratio_pooled": source_sum / prompt_sum,
            "source_prompt_ratio_fold_mean": float(ratios.mean()),
            "source_prompt_ratio_fold_std": float(ratios.std(ddof=0)),
            "source_prompt_ratio_by_fold": ratios.tolist(),
        }
    )
    return output


def evaluate_variance(
    feature_path: Path,
    output: Path,
    *,
    n_splits: int = 5,
    split_seed: int = 42,
) -> dict[str, Any]:
    batch = load_features(feature_path)
    features = np.asarray(batch.features, dtype=np.float32)
    if features.ndim != 3 or features.shape[1] != 2:
        raise ValueError("source/prompt analysis requires N x 2 x D DC-AC features")
    representations = {
        "dc": features[:, 0],
        "ac": features[:, 1],
        "dc-ac": features.reshape(len(features), -1),
    }
    labels = np.asarray(batch.labels, dtype=object)
    prompt_ids = np.asarray(batch.sample_ids, dtype=object)
    folds = list(prompt_grouped_folds(batch.sample_ids, n_splits, split_seed))
    fold_rows = []
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        metrics = {}
        for name, representation in representations.items():
            scaler = StandardScaler().fit(representation[train_indices])
            held_out = scaler.transform(representation[test_indices])
            metrics[name] = two_way_variance(
                held_out,
                labels[test_indices],
                prompt_ids[test_indices],
            )
        fold_rows.append({"fold": fold_index, "metrics": metrics})
    summary = {
        name: _aggregate([row["metrics"][name] for row in fold_rows])
        for name in representations
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "feature_source": str(feature_path),
        "n_samples": len(features),
        "n_sources": len(set(batch.labels)),
        "n_prompts": len(set(batch.sample_ids)),
        "split_seed": split_seed,
        "n_splits": len(folds),
        "preprocessing": "fold-local per-dimension standardization",
        "fit_scope": "outer-training prompts only",
        "summary": summary,
        "folds": fold_rows,
        "complete": True,
    }
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    evaluate_variance(
        args.features,
        args.output,
        n_splits=args.n_splits,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
