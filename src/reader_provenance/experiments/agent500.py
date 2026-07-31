"""Canonical prompt-grouped Agent500 evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from reader_provenance.evaluation.evidence import accumulate_log_posteriors
from reader_provenance.evaluation.grouped import grouped_oof_metrics_detailed
from reader_provenance.evaluation.metrics import classification_metrics
from reader_provenance.evaluation.protocol import (
    prompt_grouped_folds,
    source_groups,
)
from reader_provenance.features.io import load_features
from reader_provenance.training.probe import fit_source_probe

DEFAULT_BUDGETS = (1, 5, 10, 20, 50, 100)
DEFAULT_GROUPING_SEEDS = (42, 43, 44)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _representation(features: np.ndarray, component: str) -> np.ndarray:
    if component == "raw":
        if features.ndim != 2:
            raise ValueError("raw components require an (N, D) feature archive")
        return features
    if features.ndim != 3 or features.shape[1] != 2:
        raise ValueError("DC/AC components require an (N, 2, D) feature archive")
    if component == "dc":
        return features[:, 0]
    if component == "ac":
        return features[:, 1]
    if component == "dc-ac":
        return features.reshape(len(features), -1)
    raise ValueError(f"unknown component: {component}")


def _group_metrics(
    *,
    test_indices: np.ndarray,
    labels: list[str],
    class_to_index: dict[str, int],
    row_log_posteriors: np.ndarray,
    budget: int,
    grouping_seed: int,
    fold_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = source_groups(
        test_indices,
        labels,
        budget,
        grouping_seed + 100 + budget * 7919 + fold_index * 31,
    )
    row_lookup = np.full(len(labels), -1, dtype=np.int64)
    row_lookup[test_indices] = np.arange(len(test_indices))
    targets = np.asarray(
        [class_to_index[label] for label, _ in groups], dtype=np.int64
    )
    combined = np.stack(
        [
            accumulate_log_posteriors(
                row_log_posteriors[row_lookup[group_indices]]
            )
            for _, group_indices in groups
        ]
    )
    return targets, combined


def evaluate(
    feature_path: Path,
    output: Path,
    artifacts_dir: Path,
    *,
    component: str,
    budgets: tuple[int, ...],
    grouping_seeds: tuple[int, ...],
    n_splits: int,
    split_seed: int,
    device: str,
) -> dict[str, Any]:
    batch = load_features(feature_path)
    x = _representation(batch.features, component)
    labels = list(batch.labels)
    classes = sorted(set(labels))
    class_to_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    folds = list(prompt_grouped_folds(batch.sample_ids, n_splits, split_seed))
    oof_logp = np.full((len(x), len(classes)), np.nan, dtype=np.float32)
    fold_assignments = np.full(len(x), -1, dtype=np.int16)
    grouped: dict[int, dict[int, list[tuple[np.ndarray, np.ndarray]]]] = {
        seed: {budget: [] for budget in budgets} for seed in grouping_seeds
    }
    fold_reports = []

    for fold_index, (train_indices, test_indices) in enumerate(folds):
        probe, diagnostics = fit_source_probe(
            x[train_indices], y[train_indices], classes, device=device
        )
        test_logp = probe.log_probabilities(x[test_indices]).astype(np.float32)
        oof_logp[test_indices] = test_logp
        fold_assignments[test_indices] = fold_index
        held_out_prompts = sorted(
            {batch.sample_ids[index] for index in test_indices}
        )
        probe.save(
            artifacts_dir / "probes" / f"fold-{fold_index}.npz",
            held_out_prompts=held_out_prompts,
        )
        for grouping_seed in grouping_seeds:
            for budget in budgets:
                grouped[grouping_seed][budget].append(
                    _group_metrics(
                        test_indices=test_indices,
                        labels=labels,
                        class_to_index=class_to_index,
                        row_log_posteriors=test_logp,
                        budget=budget,
                        grouping_seed=grouping_seed,
                        fold_index=fold_index,
                    )
                )
        fold_reports.append(
            {
                "fold": fold_index,
                "train_responses": int(len(train_indices)),
                "test_responses": int(len(test_indices)),
                "held_out_prompts": len(held_out_prompts),
                "fit": diagnostics,
            }
        )

    if np.isnan(oof_logp).any() or np.any(fold_assignments < 0):
        raise AssertionError("out-of-fold predictions are incomplete")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts_dir / "oof_log_posteriors.npz",
        log_posteriors=oof_logp,
        labels=y,
        classes=np.asarray(classes, dtype=object),
        sample_ids=np.asarray(batch.sample_ids, dtype=object),
        fold_assignments=fold_assignments,
    )

    by_seed: dict[str, Any] = {}
    for seed in grouping_seeds:
        by_seed[str(seed)] = {}
        for budget in budgets:
            labels_for_budget = np.concatenate(
                [value[0] for value in grouped[seed][budget]]
            )
            logp_for_budget = np.concatenate(
                [value[1] for value in grouped[seed][budget]]
            )
            by_seed[str(seed)][str(budget)] = classification_metrics(
                labels_for_budget, logp_for_budget
            )

    summary: dict[str, Any] = {}
    for budget in budgets:
        row: dict[str, float] = {}
        for metric in ("accuracy", "macro_f1", "nll"):
            values = np.asarray(
                [by_seed[str(seed)][str(budget)][metric] for seed in grouping_seeds]
            )
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        summary[str(budget)] = row

    detailed_by_seed, detailed_summary, by_fold, fold_summary = (
        grouped_oof_metrics_detailed(
            labels=labels,
            classes=classes,
            fold_assignments=fold_assignments,
            log_posteriors=oof_logp,
            budgets=budgets,
            grouping_seeds=grouping_seeds,
        )
    )
    by_seed = detailed_by_seed
    summary = detailed_summary

    report = {
        "schema_version": 1,
        "protocol": "reader_agent500_prompt_grouped_v1",
        "feature_source": str(feature_path),
        "component": component,
        "n_responses": len(labels),
        "n_sources": len(classes),
        "n_prompts": len(set(batch.sample_ids)),
        "split_seed": split_seed,
        "grouping_seeds": list(grouping_seeds),
        "budgets": list(budgets),
        "probe": {
            "implementation": "full-batch PyTorch multinomial linear probe",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "steps": 40,
            "schedule": "cosine, horizon=100, eta_min=1e-5",
            "c_value": 1.0,
        },
        "evidence": "prior-corrected sum of per-response log posteriors",
        "folds": fold_reports,
        "metrics_by_grouping_seed": by_seed,
        "metrics_across_grouping_seeds": summary,
        "metrics_by_fold": by_fold,
        "metrics_across_folds": fold_summary,
        "complete": True,
    }
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=("dc", "ac", "dc-ac", "raw"),
        default="dc-ac",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--grouping-seeds", type=int, nargs="+", default=DEFAULT_GROUPING_SEEDS
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(
        args.features,
        args.output,
        args.artifacts_dir,
        component=args.component,
        budgets=tuple(sorted(set(args.budgets))),
        grouping_seeds=tuple(dict.fromkeys(args.grouping_seeds)),
        n_splits=args.n_splits,
        split_seed=args.split_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
