"""No-retraining evaluation on prompt-aligned and unseen-prompt panels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from reader_provenance.evaluation.grouped import (
    grouped_fold_model_metrics,
    grouped_oof_metrics_detailed,
    grouped_panel_metrics,
)
from reader_provenance.features.io import load_features
from reader_provenance.training.probe import LinearSourceProbe


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_probes(directory: Path) -> list[tuple[LinearSourceProbe, list[str]]]:
    paths = sorted(directory.glob("fold-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no fold probes under {directory}")
    probes = [LinearSourceProbe.load(path) for path in paths]
    classes = [probe.classes for probe, _ in probes]
    if any(values != classes[0] for values in classes[1:]):
        raise ValueError("fold probes have different class rosters")
    return probes


def _prompt_matched_log_posteriors(
    features: np.ndarray,
    sample_ids: list[str],
    probes: list[tuple[LinearSourceProbe, list[str]]],
) -> tuple[np.ndarray, np.ndarray]:
    flat = features.reshape(len(features), -1)
    output = np.full((len(flat), len(probes[0][0].classes)), np.nan, dtype=np.float32)
    counts = np.zeros(len(flat), dtype=np.int8)
    fold_assignments = np.full(len(flat), -1, dtype=np.int16)
    ids = np.asarray(sample_ids, dtype=object)
    for fold_index, (probe, held_out_prompts) in enumerate(probes):
        indices = np.flatnonzero(np.isin(ids, held_out_prompts))
        output[indices] = probe.log_probabilities(flat[indices])
        counts[indices] += 1
        fold_assignments[indices] = fold_index
    if not np.all(counts == 1):
        raise ValueError(
            "prompt-matched evaluation requires exactly one fold prediction "
            f"per response; observed {counts.min()}..{counts.max()}"
        )
    return output, fold_assignments


def _fold_model_log_posteriors(
    features: np.ndarray,
    probes: list[tuple[LinearSourceProbe, list[str]]],
) -> np.ndarray:
    flat = features.reshape(len(features), -1)
    return np.stack(
        [probe.log_probabilities(flat) for probe, _prompts in probes]
    ).astype(np.float32)


def evaluate_external(
    feature_path: Path,
    probes_dir: Path,
    output: Path,
    *,
    protocol: str,
    budgets: tuple[int, ...],
    grouping_seeds: tuple[int, ...],
) -> dict[str, Any]:
    batch = load_features(feature_path)
    probes = _load_probes(probes_dir)
    classes = probes[0][0].classes
    if set(batch.labels) != set(classes):
        raise ValueError("external panel and enrollment class rosters differ")
    protocol_details: dict[str, Any]
    if protocol == "prompt_matched_out_of_fold":
        row_logp, fold_assignments = _prompt_matched_log_posteriors(
            batch.features, batch.sample_ids, probes
        )
        # This global grouping is the paper's reporting protocol: every row is
        # OOF-scored first, after which fingerprints are sampled by source.
        by_seed, summary = grouped_panel_metrics(
            labels=batch.labels,
            classes=classes,
            log_posteriors=row_logp,
            budgets=budgets,
            grouping_seeds=grouping_seeds,
        )
        strict_by_seed, strict_summary, by_fold, fold_summary = (
            grouped_oof_metrics_detailed(
                labels=batch.labels,
                classes=classes,
                fold_assignments=fold_assignments,
                log_posteriors=row_logp,
                budgets=budgets,
                grouping_seeds=grouping_seeds,
            )
        )
        protocol_details = {
            "paper_reporting": "metrics_across_grouping_seeds",
            "paper_grouping_scope": (
                "source-wise groups after row-level prompt-matched OOF scoring"
            ),
            "strict_fold_local_metrics_by_grouping_seed": strict_by_seed,
            "strict_fold_local_metrics_across_grouping_seeds": strict_summary,
            "metrics_by_fold": by_fold,
            "metrics_across_folds": fold_summary,
        }
    elif protocol == "fold_model_ensemble":
        fold_logp = _fold_model_log_posteriors(batch.features, probes)
        by_seed, summary, by_fold, fold_summary = grouped_fold_model_metrics(
            labels=batch.labels,
            classes=classes,
            fold_log_posteriors=fold_logp,
            budgets=budgets,
            grouping_seeds=grouping_seeds,
        )
        protocol_details = {
            "paper_reporting": "metrics_across_folds",
            "fold_model_aggregation": "mean class log-posterior",
            "metrics_by_fold": by_fold,
            "metrics_across_folds": fold_summary,
            "fold_logposterior_ensemble": {
                "metrics_by_grouping_seed": by_seed,
                "metrics_across_grouping_seeds": summary,
            },
        }
    else:
        raise ValueError(f"unknown external evaluation protocol: {protocol}")
    report = {
        "schema_version": 1,
        "protocol": "reader_no_retraining_external_v1",
        "feature_source": str(feature_path),
        "fold_models": str(probes_dir),
        "evaluation_protocol": protocol,
        "n_responses": len(batch.labels),
        "n_sources": len(classes),
        "n_prompts": len(set(batch.sample_ids)),
        "budgets": list(budgets),
        "grouping_seeds": list(grouping_seeds),
        "metrics_by_grouping_seed": by_seed,
        "metrics_across_grouping_seeds": summary,
        **protocol_details,
        "retrained": False,
        "complete": True,
    }
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--probes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("prompt_matched_out_of_fold", "fold_model_ensemble"),
        required=True,
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=(1, 5, 10, 20, 50, 100)
    )
    parser.add_argument("--grouping-seeds", type=int, nargs="+", default=(42, 43, 44))
    args = parser.parse_args()
    evaluate_external(
        args.features,
        args.probes_dir,
        args.output,
        protocol=args.protocol,
        budgets=tuple(sorted(set(args.budgets))),
        grouping_seeds=tuple(dict.fromkeys(args.grouping_seeds)),
    )


if __name__ == "__main__":
    main()
