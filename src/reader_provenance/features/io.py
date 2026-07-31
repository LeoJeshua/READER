from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class FeatureBatch:
    features: np.ndarray
    labels: list[str]
    sample_ids: list[str]
    metadata: dict[str, Any]

    def validate(self) -> None:
        valid_dct = self.features.ndim == 3 and self.features.shape[1] >= 1
        valid_flat = self.features.ndim == 2
        if not valid_dct and not valid_flat:
            raise ValueError("features must have shape (N, D) or (N, 2, D)")
        if len(self.labels) != len(self.features):
            raise ValueError("feature and label counts differ")
        if len(self.sample_ids) != len(self.features):
            raise ValueError("feature and sample-id counts differ")


def save_features(path: str | Path, batch: FeatureBatch) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    batch.validate()
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=np.asarray(batch.features, dtype=np.float32),
            labels=np.asarray(batch.labels, dtype=object),
            sample_ids=np.asarray(batch.sample_ids, dtype=object),
            metadata=np.asarray(
                json.dumps(batch.metadata, sort_keys=True), dtype=object
            ),
        )
    os.replace(temporary, destination)


def load_features(path: str | Path) -> FeatureBatch:
    with np.load(Path(path), allow_pickle=True) as archive:
        batch = FeatureBatch(
            features=np.asarray(archive["features"], dtype=np.float32),
            labels=[str(value) for value in archive["labels"].tolist()],
            sample_ids=[str(value) for value in archive["sample_ids"].tolist()],
            metadata=json.loads(str(archive["metadata"].item())),
        )
    batch.validate()
    return batch
