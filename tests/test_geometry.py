from __future__ import annotations

import numpy as np
import pytest

from reader_provenance.experiments.geometry import (
    aggregate_signatures,
    cosine_distances,
    cross_domain_retrieval,
    family_purity,
    major_family_groups,
    random_family_expectation,
)
from reader_provenance.features.io import FeatureBatch


def _panel() -> tuple[FeatureBatch, list[str], list[str]]:
    labels = [f"source-{index}" for index in range(6)]
    families = ["A"] * 3 + ["B"] * 3
    rows, row_labels, prompts = [], [], []
    rng = np.random.default_rng(7)
    for prompt in range(8):
        for index, label in enumerate(labels):
            family_sign = 1.0 if index < 3 else -1.0
            source_offset = (index % 3 - 1) * 0.08
            dc = np.asarray([family_sign, source_offset])
            ac = np.asarray([source_offset, family_sign])
            rows.append(np.stack((dc, ac)) + rng.normal(0, 0.005, (2, 2)))
            row_labels.append(label)
            prompts.append(f"prompt-{prompt}")
    return (
        FeatureBatch(np.asarray(rows), row_labels, prompts, {}),
        labels,
        families,
    )


def test_source_geometry_recovers_family_neighbors() -> None:
    batch, labels, families = _panel()
    signatures = aggregate_signatures(batch, labels)
    assert signatures.shape == (6, 4)
    assert np.linalg.norm(signatures[:, :2], axis=1) == pytest.approx(1.0)
    assert np.linalg.norm(signatures[:, 2:], axis=1) == pytest.approx(1.0)

    groups, eligible, _counts = major_family_groups(families, 3)
    purity, per_source = family_purity(
        cosine_distances(signatures), groups, eligible, 2
    )
    assert purity == pytest.approx(1.0)
    assert np.all(per_source == 1.0)
    assert random_family_expectation(groups, eligible) == pytest.approx(0.4)


def test_cross_domain_retrieval_is_exact_for_matching_signatures() -> None:
    batch, labels, _families = _panel()
    signatures = aggregate_signatures(batch, labels)
    report = cross_domain_retrieval(signatures, signatures)
    assert report["top1"] == 1.0
    assert report["top5"] == 1.0
    assert report["mrr"] == 1.0
