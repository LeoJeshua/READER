from __future__ import annotations

import numpy as np

from reader_provenance.evaluation.grouped import (
    grouped_fold_model_metrics,
    grouped_oof_metrics,
    grouped_oof_metrics_detailed,
    grouped_panel_metrics,
)


def test_grouped_oof_metrics_preserve_perfect_predictions() -> None:
    labels = ["a", "b"] * 4
    folds = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    logp = np.asarray(
        [[-0.01, -5.0] if label == "a" else [-5.0, -0.01] for label in labels]
    )
    by_seed, summary = grouped_oof_metrics(
        labels=labels,
        classes=["a", "b"],
        fold_assignments=folds,
        log_posteriors=logp,
        budgets=(1, 2),
        grouping_seeds=(42, 43),
    )
    assert by_seed["42"]["1"]["accuracy"] == 1.0
    assert summary["2"]["accuracy_mean"] == 1.0

    _, _, by_fold, fold_summary = grouped_oof_metrics_detailed(
        labels=labels,
        classes=["a", "b"],
        fold_assignments=folds,
        log_posteriors=logp,
        budgets=(1, 2),
        grouping_seeds=(42, 43),
    )
    assert by_fold["0"]["42"]["2"]["accuracy"] == 1.0
    assert fold_summary["1"]["accuracy_by_fold"] == [1.0, 1.0]


def test_panel_and_fold_model_metrics_preserve_perfect_predictions() -> None:
    labels = ["a"] * 4 + ["b"] * 4
    logp = np.asarray(
        [[-0.01, -5.0] if label == "a" else [-5.0, -0.01] for label in labels]
    )
    by_seed, summary = grouped_panel_metrics(
        labels=labels,
        classes=["a", "b"],
        log_posteriors=logp,
        budgets=(1, 2),
        grouping_seeds=(42, 43),
    )
    assert by_seed["42"]["2"]["accuracy"] == 1.0
    assert summary["1"]["macro_f1_mean"] == 1.0

    ensemble, _, by_fold, across_folds = grouped_fold_model_metrics(
        labels=labels,
        classes=["a", "b"],
        fold_log_posteriors=np.stack([logp, logp - 0.2]),
        budgets=(1, 2),
        grouping_seeds=(42, 43),
    )
    assert ensemble["42"]["2"]["accuracy"] == 1.0
    assert by_fold["1"]["43"]["1"]["accuracy"] == 1.0
    assert across_folds["2"]["accuracy_by_fold"] == [1.0, 1.0]
