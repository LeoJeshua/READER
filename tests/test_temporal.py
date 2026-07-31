from pathlib import Path

import numpy as np

from reader_provenance.experiments.temporal import evaluate, fixed_representations
from reader_provenance.features.io import FeatureBatch, save_features
from reader_provenance.reporting.statistics import _dynamic_statistics


def test_fixed_temporal_representations_are_dimension_matched() -> None:
    dct = np.zeros((5, 8, 3), dtype=np.float32)
    pooling = np.zeros((5, 2, 3), dtype=np.float32)
    values = fixed_representations(dct, pooling)
    assert values["mean_pool"].shape == (5, 3)
    assert values["mean_max_pool"].shape == (5, 6)
    assert values["dct_q2"].shape == (5, 6)
    assert values["dct_q8"].shape == (5, 24)


def test_temporal_evaluation_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    labels = []
    sample_ids = []
    dct_rows = []
    pool_rows = []
    for prompt in range(9):
        for class_index, label in enumerate(("a", "b", "c")):
            center = np.zeros((8, 2), dtype=np.float32)
            center[0, class_index % 2] = 4.0 * (1 if class_index < 2 else -1)
            dct_rows.append(center + rng.normal(0, 0.02, center.shape))
            pool_rows.append(rng.normal(0, 0.1, (2, 2)))
            labels.append(label)
            sample_ids.append(f"p{prompt}")
    dct_path = tmp_path / "dct.npz"
    pool_path = tmp_path / "pool.npz"
    save_features(
        dct_path,
        FeatureBatch(np.asarray(dct_rows), labels, sample_ids, {"kind": "dct"}),
    )
    save_features(
        pool_path,
        FeatureBatch(np.asarray(pool_rows), labels, sample_ids, {"kind": "pool"}),
    )
    report = evaluate(
        dct_path=dct_path,
        pooling_path=pool_path,
        output=tmp_path / "report.json",
        predictions=tmp_path / "predictions.npz",
        device="cpu",
        budgets=(1,),
        grouping_seeds=(42,),
        n_splits=3,
        split_seed=42,
    )
    assert report["complete"] is True
    assert set(report["configs"]) == {
        "mean_pool",
        "final_token",
        "max_pool",
        "mean_final_pool",
        "mean_max_pool",
        "final_max_pool",
        "dct_q2",
        "dct_q4",
        "dct_q8",
        "learned_temporal_h1",
        "learned_temporal_h2",
    }
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as archive:
        assert archive["row_labels"].shape == (27,)
        assert archive["row_fold_ids"].shape == (27,)
        assert archive["dct_q2_k1_fold_ids"].shape == (27,)
    statistics = _dynamic_statistics(
        [tmp_path / "predictions.npz"],
        bootstrap_samples=20,
        permutations=20,
        seed=42,
    )
    dct = statistics["proxies"]["predictions"]["1"]["methods"]["dct_q2"]
    assert dct["n_predictions"] == 27
    assert dct["n_fold_source_clusters"] == 9
