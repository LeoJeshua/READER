"""Pairwise black-box provenance baselines for Bench-A style experiments.

The helpers in this module intentionally operate on model outputs or external
response embeddings only. They do not access target-model logits, weights, or
hidden states.
"""
from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC


@dataclass(frozen=True, slots=True)
class PairExample:
    """One related/unrelated model pair."""

    model_a: str
    model_b: str
    label: int
    group: str


@dataclass(frozen=True, slots=True)
class PairwiseMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


@dataclass(frozen=True, slots=True)
class RepeatedPairwiseReport:
    method: str
    n_repeats: int
    n_pairs_mean: float
    accuracy_mean: float
    accuracy_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    auc_mean: float
    auc_std: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "method": self.method,
            "n_repeats": self.n_repeats,
            "n_pairs_mean": self.n_pairs_mean,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "precision_mean": self.precision_mean,
            "precision_std": self.precision_std,
            "recall_mean": self.recall_mean,
            "recall_std": self.recall_std,
            "f1_mean": self.f1_mean,
            "f1_std": self.f1_std,
            "auc_mean": self.auc_mean,
            "auc_std": self.auc_std,
        }


def _require_equal_length(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError(f"response lengths differ: {len(left)} != {len(right)}")


def normalize_response(text: object) -> str:
    if text is None:
        return ""
    return str(text)


def first_nonspace_token(text: object) -> str:
    """Tokenizer-free approximation of the first generated token."""

    match = re.search(r"\S+", normalize_response(text))
    return match.group(0) if match else ""


def first_chars(text: object, n_chars: int = 4) -> str:
    """PhyloLM-style allele: strip leading spaces/newlines, then take chars."""

    return normalize_response(text).lstrip(" ").lstrip("\n")[:n_chars]


def exact_match_rate(
    responses_a: Sequence[object],
    responses_b: Sequence[object],
    *,
    transform: Callable[[object], object],
) -> float:
    _require_equal_length(responses_a, responses_b)
    if not responses_a:
        raise ValueError("responses must not be empty")
    matches = sum(
        1
        for left, right in zip(responses_a, responses_b)
        if transform(left) == transform(right)
    )
    return float(matches / len(responses_a))


def mpt_next_token_agreement(
    responses_a: Sequence[object],
    responses_b: Sequence[object],
    *,
    first_token_fn: Callable[[object], object] = first_nonspace_token,
) -> float:
    """MPT baseline as first generated token agreement over the input set."""

    return exact_match_rate(responses_a, responses_b, transform=first_token_fn)


def phylolm_first4char_agreement(
    responses_a: Sequence[object],
    responses_b: Sequence[object],
    *,
    n_chars: int = 4,
) -> float:
    """PhyloLM N=1 baseline as first-n-character allele agreement."""

    return exact_match_rate(
        responses_a,
        responses_b,
        transform=lambda text: first_chars(text, n_chars=n_chars),
    )


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vector_b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def pair_scores_from_responses(
    pairs: Sequence[PairExample],
    responses_by_model: Mapping[str, Sequence[object]],
    *,
    method: str,
) -> np.ndarray:
    """Score pairs using output exact-match baselines."""

    scores: list[float] = []
    for pair in pairs:
        left = responses_by_model[pair.model_a]
        right = responses_by_model[pair.model_b]
        if method == "mpt_next_token":
            scores.append(mpt_next_token_agreement(left, right))
        elif method == "phylolm_first4char":
            scores.append(phylolm_first4char_agreement(left, right))
        else:
            raise ValueError(f"unknown response baseline method: {method}")
    return np.asarray(scores, dtype=np.float32)


def pair_scores_from_vectors(
    pairs: Sequence[PairExample],
    vectors_by_model: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Score pairs by cosine similarity between model-level vectors."""

    return np.asarray(
        [
            cosine_similarity(vectors_by_model[pair.model_a], vectors_by_model[pair.model_b])
            for pair in pairs
        ],
        dtype=np.float32,
    )


class LlmDnaConcatReducer:
    """Black-box LLM-DNA variant: concatenate response embeddings, then reduce.

    For each model, input embeddings must have shape ``(n_prompts, embed_dim)``.
    The model-level feature is first flattened to ``n_prompts * embed_dim``.
    Dimensionality reduction is then fit across the model-level matrix. This
    avoids the bug in the reference implementation where per-model reduction is
    applied before concatenation/truncation.
    """

    def __init__(
        self,
        *,
        dna_dim: int = 128,
        random_state: int = 42,
        standardize: bool = True,
        backend: str = "auto",
        device: str = "cuda",
        torch_batch_size: int = 16,
    ) -> None:
        self.dna_dim = dna_dim
        self.random_state = random_state
        self.standardize = standardize
        self.backend = backend
        self.device = device
        self.torch_batch_size = torch_batch_size
        self.scaler_: StandardScaler | None = None
        self.reducer_: GaussianRandomProjection | None = None
        self.torch_projection_: object | None = None
        self.model_names_: list[str] = []

    def _make_reducer(self, n_components: int):
        return GaussianRandomProjection(
            n_components=n_components,
            random_state=self.random_state,
        )

    def _use_torch_projection(self) -> bool:
        if self.backend == "numpy":
            return False
        if self.backend not in {"auto", "torch"}:
            raise ValueError(f"unknown backend={self.backend!r}")
        try:
            import torch
        except ImportError:
            if self.backend == "torch":
                raise
            return False
        if self.backend == "auto":
            return bool(torch.cuda.is_available() or self.device == "cpu")
        return True

    def _fit_torch_projection(self, n_features: int, n_components: int) -> None:
        import torch

        generator_device = "cuda" if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.random_state)
        projection = torch.randn(
            (n_features, n_components),
            generator=generator,
            dtype=torch.float32,
            device=generator_device,
        )
        projection /= float(np.sqrt(n_components))
        target_device = self.device if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.torch_projection_ = projection.to(target_device)

    def _torch_transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.torch_projection_ is None:
            raise RuntimeError("torch projection has not been fit")
        import torch

        projection = self.torch_projection_
        device = projection.device
        outs: list[np.ndarray] = []
        batch_size = max(int(self.torch_batch_size), 1)
        with torch.no_grad():
            for start in range(0, matrix.shape[0], batch_size):
                batch = torch.as_tensor(
                    matrix[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                outs.append((batch @ projection).cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    @staticmethod
    def _flatten_embeddings(embeddings_by_model: Mapping[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
        if not embeddings_by_model:
            raise ValueError("embeddings_by_model must not be empty")
        names = sorted(embeddings_by_model)
        flattened: list[np.ndarray] = []
        expected_shape: tuple[int, int] | None = None
        for name in names:
            arr = np.asarray(embeddings_by_model[name], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"{name}: expected 2D embeddings, got {arr.shape}")
            if expected_shape is None:
                expected_shape = arr.shape
            elif arr.shape != expected_shape:
                raise ValueError(
                    f"{name}: embedding shape {arr.shape} differs from {expected_shape}"
                )
            flattened.append(arr.reshape(-1))
        return names, np.stack(flattened, axis=0).astype(np.float32)

    def fit(self, embeddings_by_model: Mapping[str, np.ndarray]) -> "LlmDnaConcatReducer":
        names, matrix = self._flatten_embeddings(embeddings_by_model)
        if self.standardize:
            self.scaler_ = StandardScaler().fit(matrix)
            matrix = self.scaler_.transform(matrix)
        max_components = min(self.dna_dim, matrix.shape[0] - 1, matrix.shape[1])
        if max_components < 1:
            raise ValueError(
                "need at least two models and one feature to fit LLM-DNA reducer"
            )
        if self._use_torch_projection():
            self._fit_torch_projection(matrix.shape[1], max_components)
            self.reducer_ = None
        else:
            self.reducer_ = self._make_reducer(max_components).fit(matrix)
            self.torch_projection_ = None
        self.model_names_ = names
        return self

    def transform(self, embeddings_by_model: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.reducer_ is None and self.torch_projection_ is None:
            raise RuntimeError("reducer has not been fit")
        names, matrix = self._flatten_embeddings(embeddings_by_model)
        if self.scaler_ is not None:
            matrix = self.scaler_.transform(matrix)
        if self.torch_projection_ is not None:
            reduced = self._torch_transform(matrix)
        else:
            assert self.reducer_ is not None
            reduced = self.reducer_.transform(matrix).astype(np.float32)
        if reduced.shape[1] < self.dna_dim:
            pad = np.zeros((reduced.shape[0], self.dna_dim - reduced.shape[1]), dtype=np.float32)
            reduced = np.concatenate([reduced, pad], axis=1)
        elif reduced.shape[1] > self.dna_dim:
            reduced = reduced[:, : self.dna_dim]
        return {name: reduced[idx] for idx, name in enumerate(names)}

    def fit_transform(self, embeddings_by_model: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.fit(embeddings_by_model).transform(embeddings_by_model)


def build_balanced_relatedness_pairs(
    parent_by_model: Mapping[str, str | None],
    *,
    valid_models: set[str] | None = None,
    random_state: int = 42,
    negative_ratio: float = 1.0,
) -> list[PairExample]:
    """Build 1:1 related/unrelated pairs from Bench-A parent labels.

    Positive pairs are ``(parent, derived_model)``. Negative pairs are sampled
    from models whose resolved parent families differ. ``group`` is the derived
    model for positives and a stable pair id for negatives; callers can replace
    this with stricter groups if needed.
    """

    rng = random.Random(random_state)
    if valid_models is None:
        valid_models = set(parent_by_model)

    family_members: dict[str, list[str]] = {}
    positives: list[PairExample] = []
    for model, parent in parent_by_model.items():
        if model not in valid_models or parent is None or parent not in valid_models:
            continue
        if model == parent:
            continue
        family_members.setdefault(parent, []).append(model)
        positives.append(PairExample(parent, model, 1, group=model))

    for model, parent in parent_by_model.items():
        if model in valid_models and parent is not None and parent in valid_models:
            family_members.setdefault(parent, [])
            if parent in valid_models and parent not in family_members[parent]:
                family_members[parent].append(parent)

    all_negative_candidates: list[tuple[str, str]] = []
    families = sorted(family_members)
    for fam_a, fam_b in itertools.combinations(families, 2):
        for model_a in family_members[fam_a]:
            for model_b in family_members[fam_b]:
                if model_a != model_b:
                    all_negative_candidates.append((model_a, model_b))

    rng.shuffle(all_negative_candidates)
    n_negative = min(len(all_negative_candidates), int(round(len(positives) * negative_ratio)))
    negatives = [
        PairExample(a, b, 0, group=f"neg::{min(a, b)}::{max(a, b)}")
        for a, b in all_negative_candidates[:n_negative]
    ]
    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def evaluate_pairwise_svm_repeated(
    *,
    method: str,
    pairs: Sequence[PairExample],
    scores: Sequence[float],
    n_repeats: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
    svm_c: float = 1.0,
) -> RepeatedPairwiseReport:
    """Train/test repeated grouped SVM evaluation on scalar similarities."""

    if len(pairs) != len(scores):
        raise ValueError(f"pairs/scores length mismatch: {len(pairs)} != {len(scores)}")
    y = np.asarray([pair.label for pair in pairs], dtype=np.int32)
    x = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
    groups = np.asarray([pair.group for pair in pairs], dtype=object)
    if len(set(y.tolist())) != 2:
        raise ValueError("pair labels must contain both positive and negative examples")

    splitter = GroupShuffleSplit(
        n_splits=n_repeats,
        test_size=test_size,
        random_state=random_state,
    )
    metrics: list[PairwiseMetrics] = []
    n_pairs: list[int] = []

    for train_idx, test_idx in splitter.split(x, y, groups):
        if len(set(y[train_idx].tolist())) != 2 or len(set(y[test_idx].tolist())) != 2:
            continue
        scaler = StandardScaler().fit(x[train_idx])
        clf = SVC(kernel="linear", C=svm_c)
        clf.fit(scaler.transform(x[train_idx]), y[train_idx])
        x_test = scaler.transform(x[test_idx])
        pred = clf.predict(x_test)
        decision = clf.decision_function(x_test)
        metrics.append(
            PairwiseMetrics(
                accuracy=float(accuracy_score(y[test_idx], pred)),
                precision=float(precision_score(y[test_idx], pred, zero_division=0)),
                recall=float(recall_score(y[test_idx], pred, zero_division=0)),
                f1=float(f1_score(y[test_idx], pred, zero_division=0)),
                auc=float(roc_auc_score(y[test_idx], decision)),
            )
        )
        n_pairs.append(int(len(test_idx)))

    if not metrics:
        raise ValueError("no valid repeated split contained both classes in train/test")

    def mean_std(attr: str) -> tuple[float, float]:
        values = np.asarray([getattr(item, attr) for item in metrics], dtype=np.float32)
        return float(values.mean()), float(values.std(ddof=0))

    acc_mean, acc_std = mean_std("accuracy")
    prec_mean, prec_std = mean_std("precision")
    rec_mean, rec_std = mean_std("recall")
    f1_mean, f1_std = mean_std("f1")
    auc_mean, auc_std = mean_std("auc")
    return RepeatedPairwiseReport(
        method=method,
        n_repeats=len(metrics),
        n_pairs_mean=float(np.mean(n_pairs)),
        accuracy_mean=acc_mean,
        accuracy_std=acc_std,
        precision_mean=prec_mean,
        precision_std=prec_std,
        recall_mean=rec_mean,
        recall_std=rec_std,
        f1_mean=f1_mean,
        f1_std=f1_std,
        auc_mean=auc_mean,
        auc_std=auc_std,
    )
