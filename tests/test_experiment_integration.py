from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reader_provenance.experiments.agent500 import evaluate
from reader_provenance.experiments.stress import evaluate_external
from reader_provenance.features.io import FeatureBatch, save_features


def _synthetic_batch(*, shift: float = 0.0) -> FeatureBatch:
    labels = ["source-a", "source-b", "source-c"]
    centers = np.asarray(
        [
            [[4.0, 0.0], [1.0, 0.0]],
            [[0.0, 4.0], [0.0, 1.0]],
            [[-4.0, -4.0], [-1.0, -1.0]],
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(123)
    features = []
    row_labels = []
    sample_ids = []
    for prompt_index in range(12):
        for class_index, label in enumerate(labels):
            noise = rng.normal(0.0, 0.03, size=(2, 2)).astype(np.float32)
            features.append(centers[class_index] + noise + shift)
            row_labels.append(label)
            sample_ids.append(f"prompt-{prompt_index:02d}")
    return FeatureBatch(
        features=np.stack(features),
        labels=row_labels,
        sample_ids=sample_ids,
        metadata={"fixture": "three-source prompt-grouped panel"},
    )


def test_agent500_and_no_retraining_round_trip(tmp_path: Path) -> None:
    enrollment_path = tmp_path / "enrollment.npz"
    save_features(enrollment_path, _synthetic_batch())
    report_path = tmp_path / "agent500.json"
    artifacts_dir = tmp_path / "artifacts"

    report = evaluate(
        enrollment_path,
        report_path,
        artifacts_dir,
        component="dc-ac",
        budgets=(1, 2),
        grouping_seeds=(42, 43),
        n_splits=3,
        split_seed=42,
        device="cpu",
    )

    assert report["complete"] is True
    assert report["n_sources"] == 3
    assert report["n_prompts"] == 12
    assert report["metrics_across_grouping_seeds"]["1"]["accuracy_mean"] > 0.9
    assert len(list((artifacts_dir / "probes").glob("fold-*.npz"))) == 3
    with np.load(artifacts_dir / "oof_log_posteriors.npz", allow_pickle=True) as oof:
        assert oof["log_posteriors"].shape == (36, 3)
        assert set(oof["fold_assignments"].tolist()) == {0, 1, 2}
    assert json.loads(report_path.read_text(encoding="utf-8"))["complete"] is True

    external_path = tmp_path / "external.npz"
    save_features(external_path, _synthetic_batch(shift=0.01))
    external_report = evaluate_external(
        external_path,
        artifacts_dir / "probes",
        tmp_path / "stress.json",
        protocol="prompt_matched_out_of_fold",
        budgets=(1, 2),
        grouping_seeds=(42,),
    )
    assert external_report["retrained"] is False
    assert external_report["metrics_across_grouping_seeds"]["1"][
        "accuracy_mean"
    ] > 0.9
    assert external_report["paper_reporting"] == "metrics_across_grouping_seeds"
    assert external_report["metrics_across_folds"]["1"]["accuracy_mean"] > 0.9

    ensemble_report = evaluate_external(
        external_path,
        artifacts_dir / "probes",
        tmp_path / "math-stress.json",
        protocol="fold_model_ensemble",
        budgets=(1, 2),
        grouping_seeds=(42,),
    )
    assert ensemble_report["paper_reporting"] == "metrics_across_folds"
    assert ensemble_report["metrics_across_folds"]["2"]["accuracy_mean"] > 0.9
    assert ensemble_report["fold_logposterior_ensemble"][
        "metrics_across_grouping_seeds"
    ]["2"]["accuracy_mean"] > 0.9
