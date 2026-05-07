"""Thin wrapper around ``LogisticRegression`` used as a reusable provenance head.

The heavy lifting of evaluation lives in ``provenance_tracker.evaluation.metrics``;
this class exists mostly for checkpointing / reuse at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass(slots=True)
class _Fitted:
    scaler: StandardScaler
    model: LogisticRegression
    encoder: LabelEncoder


class ProvenanceClassifier:
    def __init__(self, random_state: int = 42, max_iter: int = 3000):
        self.random_state = random_state
        self.max_iter = max_iter
        self._state: _Fitted | None = None

    def fit(self, features: np.ndarray, labels: list[str]) -> None:
        scaler = StandardScaler().fit(features)
        encoder = LabelEncoder().fit(labels)
        model = LogisticRegression(
            max_iter=self.max_iter, random_state=self.random_state, solver="lbfgs"
        )
        model.fit(scaler.transform(features), encoder.transform(labels))
        self._state = _Fitted(scaler, model, encoder)

    def predict(self, features: np.ndarray) -> list[str]:
        if self._state is None:
            raise RuntimeError("classifier has not been fit")
        x = self._state.scaler.transform(features)
        y = self._state.model.predict(x)
        return self._state.encoder.inverse_transform(y).tolist()
