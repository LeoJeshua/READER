from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from reader_provenance.reporting.confusion import normalized_confusion
from reader_provenance.reporting.tables import (
    bench_a_pair_table,
    temporal_table,
    write_endpoint_table,
)

ROOT = Path(__file__).resolve().parents[1]


def test_released_endpoint_table_matches_paper_statistics(tmp_path: Path) -> None:
    output = tmp_path / "endpoints.csv"
    write_endpoint_table(ROOT / "results", ROOT / "configs/proxies.yaml", output)
    rows = {
        row["method"]: row
        for row in csv.DictReader(output.open(encoding="utf-8"))
    }

    expected = {
        "FT DeBERTa": (0.32986, 0.295313, 0.568, 0.470954),
        "FT DNA / Qwen-Emb.": (0.55446, 0.530229, 0.85, 0.805667),
        "READER / Llama-3.1-8B-Instruct": (
            0.4562,
            0.455029,
            0.948,
            0.931333,
        ),
        "READER / Gemma-4-12B-it": (0.4601, 0.45673, 0.958, 0.944667),
        "READER / Ministral-3-8B-Instruct-2512": (
            0.48296,
            0.481192,
            0.952,
            0.938667,
        ),
        "READER / Qwen3.5-9B": (0.50386, 0.502896, 0.962, 0.952333),
        "READER / OLMo-3-7B-Instruct": (
            0.46308,
            0.461573,
            0.942,
            0.929667,
        ),
        "READER / Qwen3-8B": (0.44738, 0.446752, 0.948, 0.933),
        "READER / Qwen3-32B": (0.50866, 0.507615, 0.964, 0.954667),
        "READER / Qwen3.5-27B": (0.52254, 0.521732, 0.962, 0.952),
    }
    for method, values in expected.items():
        actual = rows[method]
        assert tuple(
            float(actual[key])
            for key in (
                "accuracy_k1",
                "macro_f1_k1",
                "accuracy_k100",
                "macro_f1_k100",
            )
        ) == pytest.approx(values, abs=5e-7)


def test_math100_release_covers_all_full_proxy_readers() -> None:
    paths = sorted((ROOT / "results/reader/stress/math100").glob("*.json"))
    assert len(paths) == 8


def test_confusion_reconstruction_preserves_fold_local_groups(
    tmp_path: Path,
) -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    folds = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    logp = np.full((8, 2), -5.0, dtype=np.float32)
    logp[np.arange(8), labels] = -0.01
    archive = tmp_path / "oof.npz"
    np.savez_compressed(
        archive,
        log_posteriors=logp,
        labels=labels,
        classes=np.asarray(["a", "b"], dtype=object),
        fold_assignments=folds,
    )
    matrix, classes = normalized_confusion(
        archive, budget=2, grouping_seed=42
    )
    assert classes == ["a", "b"]
    assert matrix == pytest.approx(np.eye(2))


def test_released_bench_a_and_temporal_tables_match_paper(tmp_path: Path) -> None:
    bench_path = bench_a_pair_table(
        ROOT / "results",
        ROOT / "configs/proxies.yaml",
        tmp_path / "bench.csv",
    )
    bench = {
        row["method"]: row
        for row in csv.DictReader(bench_path.open(encoding="utf-8"))
    }
    assert float(bench["READER / Llama-3.1-8B-Instruct"]["accuracy_mean"]) == (
        pytest.approx(0.78125006)
    )
    assert float(bench["READER / Qwen3.5-9B"]["auc_mean"]) == pytest.approx(
        0.80937499
    )

    temporal_path = temporal_table(
        ROOT / "results",
        ROOT / "configs/proxies.yaml",
        tmp_path / "temporal.csv",
    )
    temporal = {
        row["representation"]: row
        for row in csv.DictReader(temporal_path.open(encoding="utf-8"))
    }
    assert float(temporal["DC-AC (DCT q=2)"]["accuracy_k1"]) == pytest.approx(
        0.475775
    )
    assert float(temporal["DCT q=8"]["accuracy_k100"]) == pytest.approx(
        0.9185
    )
