from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import torch


def first_nonspace_token(text: object) -> str:
    match = re.search(r"\S+", "" if text is None else str(text))
    return match.group(0) if match else ""


def first_four_characters(text: object) -> str:
    return ("" if text is None else str(text)).lstrip(" ").lstrip("\n")[:4]


def exact_match_rate(
    left: Sequence[object],
    right: Sequence[object],
    *,
    transform,
) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("response panels must have the same nonzero length")
    return float(
        np.mean(
            [
                transform(a) == transform(b)
                for a, b in zip(left, right, strict=True)
            ]
        )
    )


def fixed_gaussian_projection(
    arrays_by_model: Mapping[str, np.ndarray],
    *,
    n_components: int = 128,
    seed: int = 42,
    device: str = "cpu",
    feature_chunk_size: int = 16384,
) -> dict[str, np.ndarray]:
    """Flatten and project model panels using the paper's fixed GRP."""
    if not arrays_by_model or n_components < 1 or feature_chunk_size < 1:
        raise ValueError("invalid model arrays or projection configuration")
    names = sorted(arrays_by_model)
    arrays = [np.asarray(arrays_by_model[name], dtype=np.float32) for name in names]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all model arrays must have the same shape")
    flattened = [array.reshape(-1) for array in arrays]
    n_features = flattened[0].size
    target = torch.device(device)
    generator = torch.Generator(device=target)
    generator.manual_seed(seed)
    projected = torch.zeros(
        (len(names), n_components),
        dtype=torch.float32,
        device=target,
    )
    scale = float(np.sqrt(n_components))
    with torch.no_grad():
        for start in range(0, n_features, feature_chunk_size):
            end = min(start + feature_chunk_size, n_features)
            projection = torch.randn(
                (end - start, n_components),
                generator=generator,
                dtype=torch.float32,
                device=target,
            )
            projection /= scale
            block = np.stack([row[start:end] for row in flattened])
            projected.addmm_(
                torch.as_tensor(block, dtype=torch.float32, device=target),
                projection,
            )
    values = projected.cpu().numpy().astype(np.float32)
    return {name: values[index] for index, name in enumerate(names)}


def vector_distances(
    pairs: Sequence[dict],
    vectors: Mapping[str, np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [
            np.linalg.norm(
                np.asarray(vectors[str(pair["model_a"])], dtype=np.float32)
                - np.asarray(vectors[str(pair["model_b"])], dtype=np.float32)
            )
            for pair in pairs
        ],
        dtype=np.float32,
    )
