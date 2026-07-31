from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import numpy as np


def prompt_grouped_folds(
    sample_ids: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Assign every response to the fold of its eliciting prompt."""
    unique_ids = np.asarray(sorted(set(sample_ids)), dtype=object)
    if len(unique_ids) < 2:
        raise ValueError("prompt-grouped evaluation requires two prompts")
    n_splits = min(max(2, n_splits), len(unique_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    ids = np.asarray(sample_ids, dtype=object)
    indices = np.arange(len(ids), dtype=np.int64)
    for held_out in np.array_split(unique_ids, n_splits):
        test_mask = np.isin(ids, held_out)
        train = indices[~test_mask]
        test = indices[test_mask]
        if not len(train) or not len(test):
            raise ValueError("empty prompt-grouped train/test fold")
        yield train, test


def source_groups(
    indices: np.ndarray,
    labels: list[str],
    budget: int,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    """Shuffle held-out responses within source and chunk them into K queries."""
    if budget < 1:
        raise ValueError("budget must be positive")
    by_source: dict[str, list[int]] = defaultdict(list)
    for index in np.asarray(indices, dtype=np.int64):
        by_source[labels[int(index)]].append(int(index))
    rng = np.random.default_rng(seed)
    groups: list[tuple[str, np.ndarray]] = []
    for label, source_indices in by_source.items():
        shuffled = np.asarray(source_indices, dtype=np.int64)
        rng.shuffle(shuffled)
        count = len(shuffled) // budget
        for chunk in shuffled[: count * budget].reshape(count, budget):
            groups.append((label, chunk))
    return groups
