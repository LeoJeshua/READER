from __future__ import annotations

import numpy as np

from reader_provenance.baselines.pairwise import (
    exact_match_rate,
    first_four_characters,
    first_nonspace_token,
    fixed_gaussian_projection,
)
from reader_provenance.experiments.bench_a import evaluate_pair_features


def test_output_only_pair_scores() -> None:
    left = [" answer one", "\nsecond value", ""]
    right = [" answers two", "\nsecond value", ""]
    assert exact_match_rate(left, right, transform=first_nonspace_token) == 2 / 3
    assert exact_match_rate(left, right, transform=first_four_characters) == 1.0


def test_fixed_projection_is_deterministic() -> None:
    arrays = {
        "a": np.arange(6, dtype=np.float32).reshape(3, 2),
        "b": np.arange(6, 12, dtype=np.float32).reshape(3, 2),
    }
    first = fixed_gaussian_projection(
        arrays,
        n_components=4,
        seed=42,
        device="cpu",
        feature_chunk_size=2,
    )
    second = fixed_gaussian_projection(
        arrays,
        n_components=4,
        seed=42,
        device="cpu",
        feature_chunk_size=2,
    )
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_fixed_pair_split_readout() -> None:
    pairs = [
        {"pair_id": f"pair-{index}", "label": int(index >= 3)}
        for index in range(6)
    ]
    splits = {
        "splits": [
            {
                "split_id": 0,
                "train_pair_ids": ["pair-0", "pair-1", "pair-3", "pair-4"],
                "test_pair_ids": ["pair-2", "pair-5"],
            }
        ]
    }
    features = {
        "method": np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])[:, None]
    }
    report = evaluate_pair_features(pairs, splits, features)
    assert report["method"]["summary"]["accuracy_mean"] == 1.0
    assert report["method"]["summary"]["auc_mean"] == 1.0
