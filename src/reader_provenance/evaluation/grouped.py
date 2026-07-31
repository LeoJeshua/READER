from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from reader_provenance.evaluation.evidence import accumulate_log_posteriors
from reader_provenance.evaluation.metrics import classification_metrics
from reader_provenance.evaluation.protocol import source_groups


def _summarize_grouping_seeds(
    by_seed: dict[str, Any],
    budgets: Sequence[int],
    grouping_seeds: Sequence[int],
) -> dict[str, Any]:
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
    return summary


def grouped_panel_metrics(
    *,
    labels: Sequence[str],
    classes: Sequence[str],
    log_posteriors: np.ndarray,
    budgets: Sequence[int],
    grouping_seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Group one scored response panel without a train/test fold boundary."""
    class_to_index = {label: index for index, label in enumerate(classes)}
    logp = np.asarray(log_posteriors, dtype=np.float64)
    if logp.shape != (len(labels), len(classes)):
        raise ValueError("log-posterior shape differs from labels/classes")
    if set(labels) != set(classes):
        raise ValueError("response labels and class roster differ")
    indices = np.arange(len(labels), dtype=np.int64)
    by_seed: dict[str, Any] = {}
    for seed in grouping_seeds:
        by_seed[str(seed)] = {}
        for budget in budgets:
            groups = source_groups(
                indices,
                list(labels),
                int(budget),
                int(seed) + 100 + int(budget) * 7919,
            )
            if not groups:
                raise ValueError(
                    f"budget {budget} exceeds the available responses per source"
                )
            targets = np.asarray(
                [class_to_index[label] for label, _group in groups],
                dtype=np.int64,
            )
            accumulated = np.stack(
                [accumulate_log_posteriors(logp[group]) for _label, group in groups]
            )
            by_seed[str(seed)][str(budget)] = classification_metrics(
                targets, accumulated
            )
    return by_seed, _summarize_grouping_seeds(
        by_seed, budgets, grouping_seeds
    )


def grouped_fold_model_metrics(
    *,
    labels: Sequence[str],
    classes: Sequence[str],
    fold_log_posteriors: np.ndarray,
    budgets: Sequence[int],
    grouping_seeds: Sequence[int],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Score each frozen fold model and their mean-log-posterior ensemble."""
    fold_logp = np.asarray(fold_log_posteriors, dtype=np.float64)
    if fold_logp.ndim != 3 or fold_logp.shape[1:] != (
        len(labels),
        len(classes),
    ):
        raise ValueError("fold predictions must have shape (F, N, C)")
    if fold_logp.shape[0] < 2:
        raise ValueError("fold-model statistics require at least two models")

    ensemble_by_seed, ensemble_summary = grouped_panel_metrics(
        labels=labels,
        classes=classes,
        log_posteriors=fold_logp.mean(axis=0),
        budgets=budgets,
        grouping_seeds=grouping_seeds,
    )
    by_fold: dict[str, Any] = {}
    for fold_index, predictions in enumerate(fold_logp):
        fold_by_seed, _fold_summary = grouped_panel_metrics(
            labels=labels,
            classes=classes,
            log_posteriors=predictions,
            budgets=budgets,
            grouping_seeds=grouping_seeds,
        )
        by_fold[str(fold_index)] = fold_by_seed

    across_folds: dict[str, Any] = {}
    for budget in budgets:
        row: dict[str, Any] = {}
        for metric in ("accuracy", "macro_f1", "nll"):
            per_fold = np.asarray(
                [
                    np.mean(
                        [
                            by_fold[str(fold)][str(seed)][str(budget)][metric]
                            for seed in grouping_seeds
                        ]
                    )
                    for fold in range(len(fold_logp))
                ],
                dtype=np.float64,
            )
            row[f"{metric}_mean"] = float(per_fold.mean())
            row[f"{metric}_std"] = float(per_fold.std(ddof=0))
            row[f"{metric}_by_fold"] = per_fold.tolist()
        across_folds[str(budget)] = row
    return ensemble_by_seed, ensemble_summary, by_fold, across_folds


def grouped_oof_metrics(
    *,
    labels: Sequence[str],
    classes: Sequence[str],
    fold_assignments: np.ndarray,
    log_posteriors: np.ndarray,
    budgets: Sequence[int],
    grouping_seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_seed, summary, _by_fold, _fold_summary = grouped_oof_metrics_detailed(
        labels=labels,
        classes=classes,
        fold_assignments=fold_assignments,
        log_posteriors=log_posteriors,
        budgets=budgets,
        grouping_seeds=grouping_seeds,
    )
    return by_seed, summary


def grouped_oof_metrics_detailed(
    *,
    labels: Sequence[str],
    classes: Sequence[str],
    fold_assignments: np.ndarray,
    log_posteriors: np.ndarray,
    budgets: Sequence[int],
    grouping_seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return pooled curves and the fold statistics reported in the paper."""
    class_to_index = {label: index for index, label in enumerate(classes)}
    folds = np.asarray(fold_assignments, dtype=np.int64)
    logp = np.asarray(log_posteriors, dtype=np.float64)
    if logp.shape != (len(labels), len(classes)):
        raise ValueError("OOF log-posterior shape differs from labels/classes")
    if np.any(folds < 0):
        raise ValueError("every row must have one held-out fold assignment")
    fold_values = sorted(set(folds.tolist()))
    by_seed: dict[str, Any] = {}
    by_fold: dict[str, Any] = {str(fold): {} for fold in fold_values}
    for seed in grouping_seeds:
        by_seed[str(seed)] = {}
        for budget in budgets:
            targets = []
            accumulated = []
            for fold_index in fold_values:
                indices = np.flatnonzero(folds == fold_index)
                groups = source_groups(
                    indices,
                    list(labels),
                    int(budget),
                    int(seed) + 100 + int(budget) * 7919 + fold_index * 31,
                )
                fold_targets = np.asarray(
                    [class_to_index[label] for label, _group in groups],
                    dtype=np.int64,
                )
                fold_accumulated = np.stack(
                    [accumulate_log_posteriors(logp[group]) for _label, group in groups]
                )
                by_fold[str(fold_index)].setdefault(str(seed), {})[
                    str(budget)
                ] = classification_metrics(fold_targets, fold_accumulated)
                targets.extend(fold_targets.tolist())
                accumulated.extend(fold_accumulated)
            by_seed[str(seed)][str(budget)] = classification_metrics(
                np.asarray(targets, dtype=np.int64),
                np.stack(accumulated),
            )
    summary = _summarize_grouping_seeds(by_seed, budgets, grouping_seeds)
    fold_summary: dict[str, Any] = {}
    for budget in budgets:
        row = {}
        for metric in ("accuracy", "macro_f1", "nll"):
            per_fold = np.asarray(
                [
                    np.mean(
                        [
                            by_fold[str(fold)][str(seed)][str(budget)][metric]
                            for seed in grouping_seeds
                        ]
                    )
                    for fold in fold_values
                ]
            )
            row[f"{metric}_mean"] = float(per_fold.mean())
            row[f"{metric}_std"] = float(per_fold.std(ddof=0))
            row[f"{metric}_by_fold"] = per_fold.tolist()
        fold_summary[str(budget)] = row
    return by_seed, summary, by_fold, fold_summary
