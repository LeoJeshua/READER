from __future__ import annotations

import numpy as np


def log_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def accumulate_log_posteriors(
    log_posteriors: np.ndarray,
    *,
    log_prior: np.ndarray | None = None,
) -> np.ndarray:
    """Combine conditionally independent source evidence.

    For observations ``1..K``, the unnormalized class score is
    ``sum_k log p(c | x_k) - (K - 1) log p(c)``. The correction is constant
    for a balanced ecosystem but is retained for general source priors.
    """
    values = np.asarray(log_posteriors, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("log_posteriors must end in a class dimension")
    n_observations = values.shape[-2]
    n_classes = values.shape[-1]
    if n_observations < 1:
        raise ValueError("at least one observation is required")
    if log_prior is None:
        prior = np.full(n_classes, -np.log(n_classes), dtype=np.float64)
    else:
        prior = np.asarray(log_prior, dtype=np.float64)
        if prior.shape != (n_classes,):
            raise ValueError("log_prior must have one value per class")
    scores = values.sum(axis=-2) - (n_observations - 1) * prior
    return log_softmax(scores, axis=-1)
