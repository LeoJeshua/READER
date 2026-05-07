from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from provenance_tracker.datasets.schemas import FeatureBatch


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_feature_batch(path: str | Path, batch: FeatureBatch) -> Path:
    output_path = ensure_parent_dir(path)
    np.savez_compressed(output_path, **batch.to_npz_dict())
    return output_path


def load_feature_batch(path: str | Path) -> FeatureBatch:
    with np.load(Path(path), allow_pickle=True) as data:
        layer_index = tuple(int(x) for x in data["layer_index"].tolist())
        return FeatureBatch(
            sample_ids=[str(x) for x in data["sample_ids"].tolist()],
            features=np.asarray(data["features"], dtype=np.float32),
            feature_kind=str(data["feature_kind"]),
            run_id=str(data["run_id"]),
            model_name=str(data["model_name"]),
            labels=[str(x) for x in data["labels"].tolist()],
            layer_index=layer_index,
        )
