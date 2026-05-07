from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(slots=True)
class ProbeSample:
    sample_id: str
    prompt: str
    probe_category: str = "rand"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResponseRecord:
    sample_id: str
    prompt: str
    response: str
    label: str
    model_name: str
    run_id: str
    timestamp: str
    probe_category: str = "rand"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeatureBatch:
    """Per-sample feature matrix, optionally stacked across layers.

    - `features` shape: (N, D) for single-layer / flat; (N, L, D) for multi-layer
    - `layer_index`: tuple of layer ids matching axis=1 when 3D, else ()
    """

    sample_ids: list[str]
    features: np.ndarray
    feature_kind: str
    run_id: str
    model_name: str
    labels: list[str]
    layer_index: tuple[int, ...] = ()

    def to_npz_dict(self) -> dict[str, Any]:
        return {
            "sample_ids": np.asarray(self.sample_ids, dtype=object),
            "features": self.features.astype(np.float32),
            "feature_kind": np.asarray(self.feature_kind, dtype=object),
            "run_id": np.asarray(self.run_id, dtype=object),
            "model_name": np.asarray(self.model_name, dtype=object),
            "labels": np.asarray(self.labels, dtype=object),
            "layer_index": np.asarray(self.layer_index, dtype=np.int32),
        }
