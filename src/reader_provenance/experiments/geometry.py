"""Original-space and t-SNE analyses of source-level READER signatures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances

from reader_provenance.features.io import FeatureBatch, load_features

NEIGHBOR_BUDGETS = (1, 3, 5, 10)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_components(signatures: np.ndarray) -> np.ndarray:
    """L2-normalize DC and first AC independently, then concatenate."""
    values = np.asarray(signatures, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("signatures must have shape (sources, 2, hidden_size)")
    norms = np.linalg.norm(values, axis=2, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).reshape(len(values), -1)


def aggregate_signatures(
    batch: FeatureBatch,
    labels: list[str],
    *,
    prompt_ids: set[str] | None = None,
) -> np.ndarray:
    """Average a balanced response panel into one DC-AC signature per source."""
    if batch.features.ndim != 3 or batch.features.shape[1] != 2:
        raise ValueError("geometry requires a two-mode DC-AC feature archive")
    row_labels = np.asarray(batch.labels, dtype=str)
    row_prompts = np.asarray(batch.sample_ids, dtype=str)
    if set(row_labels) != set(labels):
        raise ValueError("feature and roster source labels differ")
    signatures = []
    expected_count: int | None = None
    for label in labels:
        mask = row_labels == label
        if prompt_ids is not None:
            mask &= np.isin(row_prompts, list(prompt_ids))
        count = int(mask.sum())
        if not count:
            raise ValueError(f"{label}: no rows in the requested prompt panel")
        if expected_count is None:
            expected_count = count
        elif count != expected_count:
            raise ValueError("geometry requires the same prompt count per source")
        signatures.append(batch.features[mask].mean(axis=0, dtype=np.float64))
    return normalize_components(np.stack(signatures))


def cosine_distances(signatures: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(signatures, metric="cosine")
    np.fill_diagonal(distances, np.inf)
    return distances


def major_family_groups(
    families: list[str], min_family_size: int
) -> tuple[np.ndarray, np.ndarray, Counter[str]]:
    counts = Counter(families)
    groups = np.asarray(
        [
            family if counts[family] >= min_family_size else "Others"
            for family in families
        ],
        dtype=str,
    )
    eligible = np.flatnonzero(groups != "Others")
    if not len(eligible):
        raise ValueError("no family reaches min_family_size")
    return groups, eligible, Counter(groups.tolist())


def neighbor_indices(distances: np.ndarray, k: int) -> np.ndarray:
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    if not 0 < k < len(distances):
        raise ValueError("neighbor count must be between 1 and n_sources - 1")
    return np.argsort(distances, axis=1)[:, :k]


def family_purity(
    distances: np.ndarray,
    groups: np.ndarray,
    eligible: np.ndarray,
    k: int,
) -> tuple[float, np.ndarray]:
    neighbors = neighbor_indices(distances, k)
    per_source = np.asarray(
        [np.mean(groups[neighbors[index]] == groups[index]) for index in eligible],
        dtype=np.float64,
    )
    return float(per_source.mean()), per_source


def random_family_expectation(groups: np.ndarray, eligible: np.ndarray) -> float:
    counts = Counter(groups.tolist())
    return float(
        np.mean(
            [(counts[str(groups[index])] - 1) / (len(groups) - 1) for index in eligible]
        )
    )


def _bootstrap_ci(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    if samples <= 0:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    return tuple(
        float(value) for value in np.quantile(draws.mean(axis=1), (0.025, 0.975))
    )


def permutation_purity(
    distances: np.ndarray,
    groups: np.ndarray,
    eligible: np.ndarray,
    k: int,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed, _ = family_purity(distances, groups, eligible, k)
    neighbors = neighbor_indices(distances, k)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = rng.permutation(groups)
        shuffled_eligible = np.flatnonzero(shuffled != "Others")
        null[index] = np.mean(
            [
                np.mean(shuffled[neighbors[row]] == shuffled[row])
                for row in shuffled_eligible
            ]
        )
    p_value = (np.count_nonzero(null >= observed) + 1) / (permutations + 1)
    return float(null.mean()), float(p_value)


def upper_triangle(values: np.ndarray) -> np.ndarray:
    return np.asarray(values[np.triu_indices(len(values), k=1)], dtype=np.float64)


def distance_spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(spearmanr(upper_triangle(left), upper_triangle(right)).statistic)


def mean_neighbor_jaccard(left: np.ndarray, right: np.ndarray, k: int) -> float:
    scores = []
    for first, second in zip(
        neighbor_indices(left, k), neighbor_indices(right, k), strict=True
    ):
        first_set, second_set = set(first.tolist()), set(second.tolist())
        scores.append(len(first_set & second_set) / len(first_set | second_set))
    return float(np.mean(scores))


def directed_rank(distances: np.ndarray, source: int, target: int) -> int:
    return int(np.flatnonzero(np.argsort(distances[source]) == target)[0] + 1)


def _tsne(values: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    return TSNE(
        n_components=2,
        metric="cosine",
        init="random",
        learning_rate="auto",
        max_iter=2000,
        method="exact",
        random_state=seed,
        perplexity=perplexity,
    ).fit_transform(np.asarray(values, dtype=np.float32))


def _prompt_partitions(sample_ids: list[str], count: int, seed: int) -> list[set[str]]:
    unique = np.unique(np.asarray(sample_ids, dtype=str))
    if len(unique) % count:
        raise ValueError("the number of prompts must be divisible by prompt_splits")
    shuffled = np.random.default_rng(seed).permutation(unique)
    return [set(values.tolist()) for values in np.split(shuffled, count)]


def _parse_named_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected TAG=PATH") from error
    return name, Path(path)


def _parse_pair(value: str) -> tuple[str, str]:
    try:
        left, right = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected LABEL_A=LABEL_B") from error
    return left, right


def _proxy_roles(config: Path) -> dict[str, str]:
    rows = yaml.safe_load(config.read_text(encoding="utf-8"))["models"]
    return {
        str(row["tag"]): "main" if "main" in row.get("roles", []) else "full"
        for row in rows
    }


def _render_geometry(
    path: Path,
    embedding: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        group: plt.get_cmap("tab10")(index % 10)
        for index, group in enumerate(sorted(set(groups)))
    }
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    for group in sorted(set(groups)):
        mask = groups == group
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=24,
            color=colors[group],
            edgecolor="white",
            linewidth=0.4,
            label=f"{group} ({int(mask.sum())})",
        )
    for index, label in enumerate(labels):
        axis.annotate(
            label,
            embedding[index],
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=4.5,
        )
    axis.set_xlabel("t-SNE component 1")
    axis.set_ylabel("t-SNE component 2")
    axis.grid(True, color="#dddddd", linewidth=0.5, alpha=0.6)
    axis.legend(frameon=False, fontsize=6, ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def analyze_agent500(
    *,
    features: dict[str, Path],
    roster_path: Path,
    proxy_config: Path,
    output_dir: Path,
    display_proxy: str,
    candidate_pairs: list[tuple[str, str]],
    min_family_size: int,
    prompt_splits: int,
    permutations: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if permutations < 1 or bootstrap_samples < 1:
        raise ValueError("permutations and bootstrap_samples must be positive")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    labels = [str(row["label"]) for row in roster]
    families = [str(row["plot_family"]) for row in roster]
    groups, eligible, family_counts = major_family_groups(families, min_family_size)
    chance = random_family_expectation(groups, eligible)
    roles = _proxy_roles(proxy_config)
    if display_proxy not in features:
        raise ValueError("display proxy is absent from --feature inputs")

    batches = {tag: load_features(path) for tag, path in features.items()}
    reference = next(iter(batches.values()))
    partitions = _prompt_partitions(reference.sample_ids, prompt_splits, seed)
    vectors: dict[str, np.ndarray] = {}
    distances: dict[str, np.ndarray] = {}
    purity_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for proxy_index, (tag, batch) in enumerate(batches.items()):
        if set(batch.sample_ids) != set(reference.sample_ids):
            raise ValueError(f"{tag}: prompt panel differs from the reference")
        vector = aggregate_signatures(batch, labels)
        distance = cosine_distances(vector)
        vectors[tag], distances[tag] = vector, distance
        for k in NEIGHBOR_BUDGETS:
            if k >= len(labels):
                continue
            purity, per_source = family_purity(distance, groups, eligible, k)
            low, high = _bootstrap_ci(
                per_source,
                bootstrap_samples,
                np.random.default_rng(seed + proxy_index * 1009 + k),
            )
            null, p_value = permutation_purity(
                distance,
                groups,
                eligible,
                k,
                permutations=permutations,
                rng=np.random.default_rng(seed + proxy_index * 7919 + k),
            )
            purity_rows.append(
                {
                    "proxy": tag,
                    "role": roles.get(tag, "unknown"),
                    "k": k,
                    "purity": purity,
                    "ci_low": low,
                    "ci_high": high,
                    "analytic_chance": chance,
                    "permutation_null_mean": null,
                    "enrichment": purity / null,
                    "p_value": p_value,
                }
            )
        for split_index, prompt_ids in enumerate(partitions):
            split_distance = cosine_distances(
                aggregate_signatures(batch, labels, prompt_ids=prompt_ids)
            )
            purity, _ = family_purity(
                split_distance, groups, eligible, min(5, len(labels) - 1)
            )
            split_rows.append(
                {
                    "proxy": tag,
                    "role": roles.get(tag, "unknown"),
                    "split": split_index,
                    "distance_spearman_with_full": distance_spearman(
                        distance, split_distance
                    ),
                    "family_purity_at_5": purity,
                    "family_purity_enrichment_at_5": purity / chance,
                }
            )

    agreement_rows = []
    for left, right in combinations(features, 2):
        agreement_rows.append(
            {
                "proxy_left": left,
                "proxy_right": right,
                "pair_role": "main"
                if roles.get(left) == roles.get(right) == "main"
                else "full",
                "distance_spearman": distance_spearman(
                    distances[left], distances[right]
                ),
                "knn_jaccard_at_10": mean_neighbor_jaccard(
                    distances[left], distances[right], min(10, len(labels) - 1)
                ),
            }
        )

    label_index = {label: index for index, label in enumerate(labels)}
    pair_rows = []
    for left, right in candidate_pairs:
        if left not in label_index or right not in label_index:
            raise ValueError(f"candidate pair is outside the roster: {left}, {right}")
        for tag in features:
            left_rank = directed_rank(
                distances[tag], label_index[left], label_index[right]
            )
            right_rank = directed_rank(
                distances[tag], label_index[right], label_index[left]
            )
            pair_rows.append(
                {
                    "left_label": left,
                    "right_label": right,
                    "proxy": tag,
                    "role": roles.get(tag, "unknown"),
                    "cosine_distance": float(
                        distances[tag][label_index[left], label_index[right]]
                    ),
                    "left_to_right_rank": left_rank,
                    "right_to_left_rank": right_rank,
                    "mutual_rank": max(left_rank, right_rank),
                }
            )

    display_values = vectors[display_proxy]
    embedding = _tsne(display_values, seed, min(15.0, len(labels) - 1.0))
    embedded_distances = pairwise_distances(embedding)
    np.fill_diagonal(embedded_distances, np.inf)
    embedding_rows = [
        {
            "label": row["label"],
            "repo_id": row["repo_id"],
            "plot_group": str(group),
            "tsne_x": float(point[0]),
            "tsne_y": float(point[1]),
            "proxy": display_proxy,
            "seed": seed,
        }
        for row, group, point in zip(roster, groups, embedding, strict=True)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "neighbor_purity.csv", purity_rows)
    _write_csv(output_dir / "prompt_split_stability.csv", split_rows)
    if agreement_rows:
        _write_csv(output_dir / "proxy_agreement.csv", agreement_rows)
    if pair_rows:
        _write_csv(output_dir / "candidate_pair_ranks.csv", pair_rows)
    _write_csv(output_dir / "canonical_embedding.csv", embedding_rows)
    _render_geometry(
        output_dir / "model_signature_geometry.pdf", embedding, groups, labels
    )
    summary = {
        "schema_version": 1,
        "protocol": "reader_source_signature_geometry_v1",
        "sources": len(labels),
        "prompts_per_source": len(set(reference.sample_ids)),
        "proxies": list(features),
        "display_proxy": display_proxy,
        "family_counts": dict(family_counts),
        "analytic_random_expectation": chance,
        "original_space": {
            "component_normalization": (
                "independent L2 normalization of DC and first AC"
            ),
            "distance": "cosine",
        },
        "tsne": {
            "seed": seed,
            "perplexity": min(15.0, len(labels) - 1.0),
            "trustworthiness_at_10": float(
                trustworthiness(
                    display_values,
                    embedding,
                    n_neighbors=min(10, len(labels) // 2 - 1),
                    metric="cosine",
                )
            ),
            "neighbor_jaccard_at_10": mean_neighbor_jaccard(
                distances[display_proxy], embedded_distances, min(10, len(labels) - 1)
            ),
        },
        "complete": True,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def cross_domain_retrieval(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    similarities = source @ target.T
    diagonal = np.diag(similarities)
    ranks = 1 + np.sum(similarities > diagonal[:, None], axis=1)
    other = similarities[~np.eye(len(similarities), dtype=bool)]
    off_diagonal = similarities[
        np.where(~np.eye(len(similarities), dtype=bool))
    ].reshape(len(similarities), -1)
    wins = np.mean(diagonal[:, None] > off_diagonal)
    return {
        "median_same_model_cosine": float(np.median(diagonal)),
        "median_other_model_cosine": float(np.median(other)),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "wins_other": float(wins),
    }


def analyze_cross_domain(
    *,
    agent_features: Path,
    math_features: Path,
    roster_path: Path,
    output: Path,
) -> dict[str, Any]:
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    labels = [str(row["label"]) for row in roster]
    agent = aggregate_signatures(load_features(agent_features), labels)
    math = aggregate_signatures(load_features(math_features), labels)
    raw = cross_domain_retrieval(agent, math)
    centered_agent = agent - agent.mean(axis=0, keepdims=True)
    centered_math = math - math.mean(axis=0, keepdims=True)
    centered_agent /= np.maximum(
        np.linalg.norm(centered_agent, axis=1, keepdims=True), 1e-12
    )
    centered_math /= np.maximum(
        np.linalg.norm(centered_math, axis=1, keepdims=True), 1e-12
    )
    report = {
        "schema_version": 1,
        "protocol": "reader_cross_domain_source_signature_v1",
        "sources": len(labels),
        "raw": raw,
        "domain_centered": cross_domain_retrieval(centered_agent, centered_math),
        "complete": True,
    }
    _atomic_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    agent = commands.add_parser("agent500")
    agent.add_argument(
        "--feature", type=_parse_named_path, action="append", required=True
    )
    agent.add_argument("--roster", type=Path, default=Path("data/rosters/100-way.json"))
    agent.add_argument(
        "--proxy-config", type=Path, default=Path("configs/proxies.yaml")
    )
    agent.add_argument("--output-dir", type=Path, required=True)
    agent.add_argument("--display-proxy", default="qwen35_9b")
    agent.add_argument(
        "--candidate-pair", type=_parse_pair, action="append", default=[]
    )
    agent.add_argument("--min-family-size", type=int, default=5)
    agent.add_argument("--prompt-splits", type=int, default=5)
    agent.add_argument("--permutations", type=int, default=10000)
    agent.add_argument("--bootstrap-samples", type=int, default=2000)
    agent.add_argument("--seed", type=int, default=42)
    cross = commands.add_parser("cross-domain")
    cross.add_argument("--agent-features", type=Path, required=True)
    cross.add_argument("--math-features", type=Path, required=True)
    cross.add_argument("--roster", type=Path, default=Path("data/rosters/100-way.json"))
    cross.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "agent500":
        analyze_agent500(
            features=dict(args.feature),
            roster_path=args.roster,
            proxy_config=args.proxy_config,
            output_dir=args.output_dir,
            display_proxy=args.display_proxy,
            candidate_pairs=args.candidate_pair,
            min_family_size=args.min_family_size,
            prompt_splits=args.prompt_splits,
            permutations=args.permutations,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    else:
        analyze_cross_domain(
            agent_features=args.agent_features,
            math_features=args.math_features,
            roster_path=args.roster,
            output=args.output,
        )


if __name__ == "__main__":
    main()
