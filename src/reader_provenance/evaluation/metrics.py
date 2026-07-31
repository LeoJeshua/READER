from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def classification_metrics(
    labels: np.ndarray,
    log_probabilities: np.ndarray,
) -> dict[str, Any]:
    probabilities = np.exp(log_probabilities)
    predictions = probabilities.argmax(axis=1)
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "nll": float(
            log_loss(labels, probabilities, labels=list(range(probabilities.shape[1])))
        ),
    }
