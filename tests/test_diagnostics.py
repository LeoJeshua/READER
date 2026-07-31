from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reader_provenance.experiments.layers import (
    FEATURE_PROTOCOL,
    evaluate_layer_scan,
)
from reader_provenance.experiments.variance import evaluate_variance
from reader_provenance.features.io import FeatureBatch, save_features


def _balanced_panel() -> tuple[list[str], list[str]]:
    labels = []
    sample_ids = []
    for prompt in range(6):
        for label in ("a", "b"):
            labels.append(label)
            sample_ids.append(f"prompt-{prompt}")
    return labels, sample_ids


def test_layer_scan_selects_the_informative_layer(tmp_path: Path) -> None:
    labels, sample_ids = _balanced_panel()
    feature_dir = tmp_path / "layers"
    feature_dir.mkdir()
    shape = (3, len(labels), 2)
    uninformative = np.zeros(shape, dtype=np.float16)
    informative = np.zeros(shape, dtype=np.float16)
    for index, label in enumerate(labels):
        sign = 3.0 if label == "a" else -3.0
        informative[0, index] = (sign, -sign)
        informative[1, index] = (sign, 0.0)
        informative[2, index] = (0.0, sign)
    np.save(feature_dir / "layer-000.npy", uninformative)
    np.save(feature_dir / "layer-001.npy", informative)
    metadata = {
        "protocol": FEATURE_PROTOCOL,
        "complete": True,
        "layer_indices": [0, 1],
        "labels": labels,
        "sample_ids": sample_ids,
    }
    (feature_dir / "metadata.json").write_text(json.dumps(metadata))

    report = evaluate_layer_scan(
        feature_dir,
        tmp_path / "scan.json",
        n_splits=3,
        device="cpu",
    )
    assert report["complete"] is True
    assert report["best_layers"]["last"] == 1
    assert report["best_layers"]["dc-ac"] == 1
    assert report["methods"]["dc"]["best"]["accuracy"] == 1.0


def test_variance_analysis_separates_source_and_prompt_effects(
    tmp_path: Path,
) -> None:
    labels, sample_ids = _balanced_panel()
    features = []
    for label, prompt_id in zip(labels, sample_ids, strict=True):
        source = 4.0 if label == "a" else -4.0
        prompt = float(prompt_id.rsplit("-", 1)[1]) - 2.5
        features.append([[source, 0.1 * prompt], [0.0, prompt]])
    path = tmp_path / "features.npz"
    save_features(
        path,
        FeatureBatch(
            features=np.asarray(features, dtype=np.float32),
            labels=labels,
            sample_ids=sample_ids,
            metadata={"fixture": True},
        ),
    )
    report = evaluate_variance(
        path,
        tmp_path / "variance.json",
        n_splits=3,
    )
    assert report["complete"] is True
    assert (
        report["summary"]["dc"]["source_prompt_ratio_pooled"]
        > report["summary"]["ac"]["source_prompt_ratio_pooled"]
    )
