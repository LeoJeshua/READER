"""Few-shot provenance classifier.

Prompt-level split: pick ``k_shots_per_prompt`` distinct prompts (each prompt
has been answered by all 9 targets, so the train set is class-balanced by
construction with exactly ``k_shots_per_prompt`` samples per class). All
remaining prompts go to the test set. Repeat under ``n_seeds`` random seeds
and aggregate.

Selling point: with hidden-state features, even 5–10 prompts of supervision
beat the mpnet black-box baseline trained on the full dataset.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class FewShotResult:
    method: str
    k_shots_per_prompt: int
    n_seeds: int
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    n_train_samples: int
    n_test_samples: int
    n_classes: int
    per_seed: list[dict] = field(default_factory=list)


def _fit_classifier(X_tr: np.ndarray, y_tr: np.ndarray, k_shots: int,
                    inner_cv: bool):
    """LR head; uses inner CV over a wide C grid to handle few-shot regimes.

    Few-shot LR is heavily regularization-sensitive: with 45 samples and
    PCA-128 features, optimal C is typically 0.01–0.1. Sweep widely.
    """
    if inner_cv and k_shots >= 5:
        n_inner = min(3, max(2, k_shots // 2))
        return LogisticRegressionCV(
            Cs=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            cv=n_inner, max_iter=4000, solver="lbfgs",
            scoring="accuracy",
        ).fit(X_tr, y_tr)
    return LogisticRegression(C=0.1, max_iter=4000, solver="lbfgs").fit(X_tr, y_tr)


def run_few_shot(features: np.ndarray, labels: list[str], sample_ids: list[str],
                 *, k_shots_per_prompt: int, n_seeds: int = 20,
                 inner_cv: bool = True, seed_base: int = 42,
                 method_name: str = "few_shot",
                 pca_dim: int | None = 128) -> FewShotResult:
    """Repeated-subsample few-shot evaluation.

    For each of ``n_seeds`` seeds, pick ``k_shots_per_prompt`` prompts at
    random; train on all (prompt, target) records with those prompt ids,
    test on the rest. Returns mean ± std accuracy / macro-F1.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D (N,D); got {features.shape}")
    if len(labels) != features.shape[0] or len(sample_ids) != features.shape[0]:
        raise ValueError("labels / sample_ids length must match features.shape[0]")

    classes = sorted(set(labels))
    cls_to_i = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_i[l] for l in labels], dtype=np.int64)

    prompt_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, sid in enumerate(sample_ids):
        prompt_to_idx[sid].append(i)
    unique_prompts = np.array(sorted(prompt_to_idx.keys()))
    if k_shots_per_prompt > len(unique_prompts):
        raise ValueError(
            f"k_shots_per_prompt={k_shots_per_prompt} exceeds available prompts "
            f"({len(unique_prompts)})"
        )

    accs, f1s, per_seed = [], [], []
    n_train = n_test = 0
    for seed_offset in range(n_seeds):
        seed = seed_base + seed_offset
        rng = np.random.default_rng(seed)
        train_prompts = rng.choice(unique_prompts, size=k_shots_per_prompt, replace=False)
        train_set = set(train_prompts.tolist())

        train_idx: list[int] = []
        for p in train_prompts:
            train_idx.extend(prompt_to_idx[p])
        train_idx_arr = np.asarray(train_idx, dtype=np.int64)
        test_idx_arr = np.array(
            [i for i, sid in enumerate(sample_ids) if sid not in train_set],
            dtype=np.int64,
        )

        X_tr_raw, y_tr = features[train_idx_arr], y[train_idx_arr]
        X_te_raw, y_te = features[test_idx_arr],  y[test_idx_arr]

        sc = StandardScaler().fit(X_tr_raw)
        X_tr = sc.transform(X_tr_raw)
        X_te = sc.transform(X_te_raw)

        if pca_dim is not None:
            n_comp = min(pca_dim, max(1, min(X_tr.shape) - 1))
            pca = PCA(n_components=n_comp, random_state=0).fit(X_tr)
            X_tr = pca.transform(X_tr)
            X_te = pca.transform(X_te)

        clf = _fit_classifier(X_tr, y_tr, k_shots_per_prompt, inner_cv)
        pred = clf.predict(X_te)
        acc = float(accuracy_score(y_te, pred))
        f1 = float(f1_score(y_te, pred, average="macro"))
        accs.append(acc)
        f1s.append(f1)
        n_train, n_test = len(train_idx_arr), len(test_idx_arr)
        per_seed.append({
            "seed": int(seed), "accuracy": acc, "macro_f1": f1,
            "n_train": int(n_train), "n_test": int(n_test),
        })

    return FewShotResult(
        method=method_name,
        k_shots_per_prompt=k_shots_per_prompt,
        n_seeds=n_seeds,
        accuracy_mean=float(np.mean(accs)),
        accuracy_std=float(np.std(accs)),
        macro_f1_mean=float(np.mean(f1s)),
        macro_f1_std=float(np.std(f1s)),
        n_train_samples=int(n_train),
        n_test_samples=int(n_test),
        n_classes=int(len(classes)),
        per_seed=per_seed,
    )
