import numpy as np

from reader_provenance.evaluation.protocol import (
    prompt_grouped_folds,
    source_groups,
)


def test_prompt_ids_never_cross_a_fold_boundary() -> None:
    ids = [f"p{prompt}" for prompt in range(10) for _ in range(4)]
    folds = list(prompt_grouped_folds(ids, n_splits=5, seed=42))
    for train, test in folds:
        assert {ids[i] for i in train}.isdisjoint({ids[i] for i in test})
    test_rows = np.concatenate([test for _, test in folds])
    assert sorted(test_rows.tolist()) == list(range(len(ids)))


def test_source_groups_are_source_pure_and_disjoint() -> None:
    labels = ["a"] * 6 + ["b"] * 6
    groups = source_groups(np.arange(12), labels, budget=3, seed=7)
    used = []
    assert len(groups) == 4
    for label, indices in groups:
        assert {labels[index] for index in indices} == {label}
        used.extend(indices.tolist())
    assert len(used) == len(set(used)) == 12
