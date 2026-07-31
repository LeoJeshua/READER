"""Resumable all-layer extraction and prompt-grouped representation scans."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from reader_provenance.data.records import ResponseRecord, load_records
from reader_provenance.evaluation.metrics import classification_metrics
from reader_provenance.evaluation.protocol import prompt_grouped_folds
from reader_provenance.models.proxy import ProxyConfig, ProxyReader
from reader_provenance.training.probe import fit_source_probe

FEATURE_PROTOCOL = "reader_layerwise_last_dc_ac_v1"
SCAN_PROTOCOL = "reader_layer_scan_prompt_grouped_v1"
METHODS = ("last", "dc", "ac", "dc-ac")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _panel_digest(records: list[ResponseRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.label.encode())
        digest.update(b"\0")
        digest.update(record.sample_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _layer_path(output_dir: Path, layer: int, *, partial: bool = False) -> Path:
    suffix = ".partial.npy" if partial else ".npy"
    return output_dir / f"layer-{layer:03d}{suffix}"


def extract_layerwise(
    records: list[ResponseRecord],
    reader: ProxyReader,
    output_dir: Path,
    *,
    checkpoint_rows: int = 1000,
) -> dict[str, Any]:
    """Persist all layers as float16 memmaps and resume at row boundaries."""
    if checkpoint_rows < 1:
        raise ValueError("checkpoint_rows must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.json"
    digest = _panel_digest(records)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("protocol") != FEATURE_PROTOCOL
            or metadata.get("panel_sha256") != digest
            or not metadata.get("complete")
        ):
            raise ValueError("completed layer cache does not match this panel")
        return metadata

    start = 0
    progress: dict[str, Any] | None = None
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("panel_sha256") != digest:
            raise ValueError("partial layer cache belongs to a different panel")
        start = int(progress["next_row"])
    if not 0 <= start <= len(records):
        raise ValueError("invalid layer-cache resume offset")
    last_checkpoint = start

    lengths: list[int] = []
    if progress is not None:
        lengths = [int(value) for value in progress.get("response_lengths", [])]
    for chunk in reader.iter_layerwise_features(records[start:]):
        global_start = start + chunk.offset
        global_stop = global_start + chunk.features.shape[2]
        if progress is None:
            n_layers, n_channels, _batch, hidden = chunk.features.shape
            shape = (n_channels, len(records), hidden)
            for layer in range(n_layers):
                store = open_memmap(
                    _layer_path(output_dir, layer, partial=True),
                    mode="w+",
                    dtype=np.float16,
                    shape=shape,
                )
                del store
            progress = {
                "schema_version": 1,
                "protocol": FEATURE_PROTOCOL,
                "panel_sha256": digest,
                "next_row": 0,
                "n_layers": n_layers,
                "layer_shape": list(shape),
                "response_lengths": [],
            }
        expected = (
            int(progress["n_layers"]),
            int(progress["layer_shape"][0]),
            chunk.features.shape[2],
            int(progress["layer_shape"][2]),
        )
        if chunk.features.shape != expected:
            raise ValueError(
                f"proxy layer shape changed: {chunk.features.shape} != {expected}"
            )
        for layer in range(int(progress["n_layers"])):
            store = np.load(
                _layer_path(output_dir, layer, partial=True), mmap_mode="r+"
            )
            store[:, global_start:global_stop] = chunk.features[layer]
            store.flush()
            del store
        lengths.extend(int(value) for value in chunk.response_lengths)
        if (
            global_stop - last_checkpoint >= checkpoint_rows
            or global_stop == len(records)
        ):
            progress["next_row"] = global_stop
            progress["response_lengths"] = lengths
            _atomic_json(progress_path, progress)
            last_checkpoint = global_stop

    if progress is None or int(progress["next_row"]) != len(records):
        raise RuntimeError("layerwise extraction ended before all rows were committed")
    for layer in range(int(progress["n_layers"])):
        os.replace(
            _layer_path(output_dir, layer, partial=True),
            _layer_path(output_dir, layer),
        )
    values = np.asarray(lengths, dtype=np.int64)
    metadata = {
        "schema_version": 1,
        "protocol": FEATURE_PROTOCOL,
        "panel_sha256": digest,
        "complete": True,
        "n_samples": len(records),
        "n_sources": len({record.label for record in records}),
        "n_prompts": len({record.sample_id for record in records}),
        "n_layers": int(progress["n_layers"]),
        "layer_indices": list(range(int(progress["n_layers"]))),
        "layer_shape": progress["layer_shape"],
        "channels": ["last_response_token", "dc", "first_ac"],
        "dtype": "float16",
        "labels": [record.label for record in records],
        "sample_ids": [record.sample_id for record in records],
        "proxy": reader.config.model_name_or_path,
        "view": reader.config.view,
        "max_length": reader.config.max_length,
        "response_token_lengths": {
            "min": int(values.min()),
            "median": float(np.median(values)),
            "max": int(values.max()),
            "mean": float(values.mean()),
        },
    }
    _atomic_json(metadata_path, metadata)
    progress_path.unlink(missing_ok=True)
    return metadata


def _representation(store: np.ndarray, method: str) -> np.ndarray:
    if method == "last":
        return np.asarray(store[0], dtype=np.float32)
    if method == "dc":
        return np.asarray(store[1], dtype=np.float32)
    if method == "ac":
        return np.asarray(store[2], dtype=np.float32)
    if method == "dc-ac":
        values = np.asarray(store[1:3], dtype=np.float32)
        return values.transpose(1, 0, 2).reshape(values.shape[1], -1)
    raise ValueError(f"unknown layer representation: {method}")


def evaluate_layer_scan(
    feature_dir: Path,
    output: Path,
    *,
    n_splits: int = 5,
    split_seed: int = 42,
    device: str = "cuda",
    retain_best_only: bool = False,
) -> dict[str, Any]:
    metadata = json.loads(
        (feature_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("protocol") != FEATURE_PROTOCOL or not metadata.get("complete"):
        raise ValueError("invalid or incomplete layerwise feature cache")
    labels = [str(value) for value in metadata["labels"]]
    sample_ids = [str(value) for value in metadata["sample_ids"]]
    classes = sorted(set(labels))
    class_to_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    folds = list(prompt_grouped_folds(sample_ids, n_splits, split_seed))
    layer_indices = [int(value) for value in metadata["layer_indices"]]
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": SCAN_PROTOCOL,
        "feature_protocol": FEATURE_PROTOCOL,
        "n_samples": len(labels),
        "n_sources": len(classes),
        "n_prompts": len(set(sample_ids)),
        "split_seed": split_seed,
        "n_splits": len(folds),
        "probe": {
            "implementation": "full-batch PyTorch multinomial linear probe",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "steps": 40,
            "schedule_horizon": 100,
            "c_value": 1.0,
            "standardization": "fold-local",
        },
        "methods": {method: {"layers": []} for method in METHODS},
        "complete": False,
    }
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if (
            previous.get("protocol") != SCAN_PROTOCOL
            or previous.get("split_seed") != split_seed
            or previous.get("n_samples") != len(labels)
        ):
            raise ValueError("existing layer-scan report has different settings")
        report = previous

    for layer in layer_indices:
        store = np.load(_layer_path(feature_dir, layer), mmap_mode="r")
        for method in METHODS:
            rows = report["methods"][method]["layers"]
            if any(int(row["layer"]) == layer for row in rows):
                continue
            features = _representation(store, method)
            fold_rows = []
            for fold_index, (train_indices, test_indices) in enumerate(folds):
                probe, diagnostics = fit_source_probe(
                    features[train_indices],
                    y[train_indices],
                    classes,
                    device=device,
                )
                scores = probe.log_probabilities(features[test_indices])
                metrics = classification_metrics(y[test_indices], scores)
                fold_rows.append(
                    {"fold": fold_index, "fit": diagnostics, **metrics}
                )
            row = {
                "layer": layer,
                "dimension": int(features.shape[1]),
                "accuracy": float(
                    np.mean([value["accuracy"] for value in fold_rows])
                ),
                "macro_f1": float(
                    np.mean([value["macro_f1"] for value in fold_rows])
                ),
                "folds": fold_rows,
            }
            rows.append(row)
            rows.sort(key=lambda value: int(value["layer"]))
            report["methods"][method]["best"] = max(
                rows,
                key=lambda value: (
                    float(value["accuracy"]),
                    float(value["macro_f1"]),
                    -int(value["layer"]),
                ),
            )
            _atomic_json(output, report)
    best_layers = {
        method: int(report["methods"][method]["best"]["layer"])
        for method in METHODS
    }
    report["best_layers"] = best_layers
    report["complete"] = True
    report["retained_layers"] = layer_indices
    if retain_best_only:
        retained = sorted(set(best_layers.values()))
        for layer in layer_indices:
            if layer not in retained:
                _layer_path(feature_dir, layer).unlink(missing_ok=True)
        report["retained_layers"] = retained
        metadata["retained_layers"] = retained
        metadata["cleanup_policy"] = "union of four validation-selected layers"
        _atomic_json(feature_dir / "metadata.json", metadata)
    _atomic_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--records", type=Path, action="append", required=True)
    extract.add_argument("--proxy", required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument(
        "--view",
        choices=("prompt_response", "response_only"),
        default="prompt_response",
    )
    extract.add_argument("--max-length", type=int, default=1024)
    extract.add_argument("--batch-size", type=int, default=4)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--dtype", default="bfloat16")
    extract.add_argument("--attention", default="auto")
    extract.add_argument("--checkpoint-rows", type=int, default=1000)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--feature-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--n-splits", type=int, default=5)
    evaluate.add_argument("--split-seed", type=int, default=42)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--retain-best-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        records = load_records(args.records)
        reader = ProxyReader(
            ProxyConfig(
                model_name_or_path=args.proxy,
                layer=0,
                view=args.view,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                dtype=args.dtype,
                attention=args.attention,
            )
        )
        try:
            extract_layerwise(
                records,
                reader,
                args.output_dir,
                checkpoint_rows=args.checkpoint_rows,
            )
        finally:
            reader.close()
    else:
        evaluate_layer_scan(
            args.feature_dir,
            args.output,
            n_splits=args.n_splits,
            split_seed=args.split_seed,
            device=args.device,
            retain_best_only=args.retain_best_only,
        )


if __name__ == "__main__":
    main()
