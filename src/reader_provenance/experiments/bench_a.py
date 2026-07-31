"""Bench-A-derived static relationship evaluation."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from reader_provenance.baselines.pairwise import (
    exact_match_rate,
    first_four_characters,
    first_nonspace_token,
    vector_distances,
)
from reader_provenance.data.records import iter_records
from reader_provenance.data.release import DatasetRelease
from reader_provenance.features.io import load_features


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    numerator = np.einsum("ij,ij->i", left, right)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def reader_pair_features(
    feature_path: Path,
    pairs: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    batch = load_features(feature_path)
    grouped: dict[str, list[tuple[str, np.ndarray]]] = {}
    for sample_id, label, feature in zip(
        batch.sample_ids,
        batch.labels,
        batch.features,
        strict=True,
    ):
        grouped.setdefault(label, []).append((sample_id, feature))
    vectors = {}
    expected_ids: list[str] | None = None
    for label, rows in grouped.items():
        rows.sort(key=lambda value: value[0])
        ids = [sample_id for sample_id, _ in rows]
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"unaligned Bench-A prompt panel for {label}")
        vectors[label] = np.stack([feature for _, feature in rows])
    needed = {str(pair[key]) for pair in pairs for key in ("model_a", "model_b")}
    if set(vectors) != needed:
        raise ValueError("Bench-A feature and pair model rosters differ")
    dc = []
    ac = []
    for pair in pairs:
        left = vectors[str(pair["model_a"])]
        right = vectors[str(pair["model_b"])]
        dc.append(float(_cosine_rows(left[:, 0], right[:, 0]).mean()))
        ac.append(float(_cosine_rows(left[:, 1], right[:, 1]).mean()))
    dc_values = np.asarray(dc, dtype=np.float32)
    ac_values = np.asarray(ac, dtype=np.float32)
    return {
        "dc": dc_values[:, None],
        "ac": ac_values[:, None],
        "dc-ac": np.column_stack((dc_values, ac_values)),
    }


def response_baseline_features(
    release: DatasetRelease,
    pairs: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    responses = {}
    for path in release.bench_a_response_paths():
        records = list(iter_records(path, allow_empty=True))
        records.sort(key=lambda value: value.sample_id)
        if not records:
            raise ValueError(f"empty Bench-A response file: {path}")
        responses[records[0].label] = [record.response for record in records]
    mpt = []
    phylolm = []
    for pair in pairs:
        left = responses[str(pair["model_a"])]
        right = responses[str(pair["model_b"])]
        mpt.append(
            exact_match_rate(left, right, transform=first_nonspace_token)
        )
        phylolm.append(
            exact_match_rate(left, right, transform=first_four_characters)
        )
    return {
        "mpt": np.asarray(mpt, dtype=np.float32)[:, None],
        "phylolm": np.asarray(phylolm, dtype=np.float32)[:, None],
    }


def _summary(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    output = {}
    for metric in ("accuracy", "precision", "recall", "f1", "auc"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_std"] = float(values.std(ddof=0))
    return output


def evaluate_pair_features(
    pairs: Sequence[dict[str, Any]],
    split_payload: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    *,
    svm_c: float = 1.0,
) -> dict[str, Any]:
    pair_lookup = {str(row["pair_id"]): index for index, row in enumerate(pairs)}
    if len(pair_lookup) != len(pairs):
        raise ValueError("duplicate pair identifiers")
    labels = np.asarray([int(row["label"]) for row in pairs], dtype=np.int64)
    reports = {}
    for method, raw_values in features.items():
        values = np.asarray(raw_values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if len(values) != len(pairs):
            raise ValueError(f"{method}: pair feature count differs")
        split_rows = []
        for split in split_payload["splits"]:
            train = np.asarray(
                [pair_lookup[value] for value in split["train_pair_ids"]],
                dtype=np.int64,
            )
            test = np.asarray(
                [pair_lookup[value] for value in split["test_pair_ids"]],
                dtype=np.int64,
            )
            scaler = StandardScaler().fit(values[train])
            classifier = SVC(kernel="linear", C=svm_c).fit(
                scaler.transform(values[train]),
                labels[train],
            )
            transformed = scaler.transform(values[test])
            prediction = classifier.predict(transformed)
            decision = classifier.decision_function(transformed)
            truth = labels[test]
            split_rows.append(
                {
                    "split_id": int(split["split_id"]),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "accuracy": float(accuracy_score(truth, prediction)),
                    "precision": float(
                        precision_score(truth, prediction, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(truth, prediction, zero_division=0)
                    ),
                    "f1": float(f1_score(truth, prediction, zero_division=0)),
                    "auc": float(roc_auc_score(truth, decision)),
                }
            )
        reports[method] = {
            "feature_dimension": int(values.shape[1]),
            "summary": _summary(split_rows),
            "splits": split_rows,
        }
    return reports


def evaluate(
    *,
    data_root: Path,
    split_protocol: str,
    output: Path,
    reader_features: Path | None = None,
    dna_vectors: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    release = DatasetRelease(data_root)
    if split_protocol == "pair_disjoint":
        pairs_path = release.bench_a_pairs_path()
    else:
        pairs_path = release.bench_a_disjoint_pairs_path()
    pairs = _read_jsonl(pairs_path)
    splits_path = release.bench_a_split_path(split_protocol)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    pair_features = response_baseline_features(release, pairs)
    if reader_features is not None:
        pair_features.update(
            {
                f"reader_{name}": value
                for name, value in reader_pair_features(
                    reader_features,
                    pairs,
                ).items()
            }
        )
    for tag, path in (dna_vectors or {}).items():
        with np.load(path, allow_pickle=True) as archive:
            names = [str(value) for value in archive["model_names"].tolist()]
            vectors = {
                name: vector
                for name, vector in zip(names, archive["vectors"], strict=True)
            }
        pair_features[f"dna_{tag}"] = vector_distances(pairs, vectors)[:, None]
    report = {
        "schema_version": 1,
        "protocol": "reader_bench_a_relationship_v1",
        "split_protocol": split_protocol,
        "pairs": len(pairs),
        "splits": len(splits["splits"]),
        "methods": evaluate_pair_features(pairs, splits, pair_features),
        "complete": True,
    }
    _atomic_json(output, report)
    return report


def _name_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from error
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--split-protocol",
        choices=("pair_disjoint", "model_disjoint", "family_disjoint"),
        required=True,
    )
    parser.add_argument("--reader-features", type=Path)
    parser.add_argument("--dna-vectors", type=_name_path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(
        data_root=args.data_root,
        split_protocol=args.split_protocol,
        output=args.output,
        reader_features=args.reader_features,
        dna_vectors=dict(args.dna_vectors),
    )


if __name__ == "__main__":
    main()
