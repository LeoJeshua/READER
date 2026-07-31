"""Confidence intervals and paired tests for supplementary experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

SCHEMA_VERSION = 1
PROTOCOL = "cluster_bootstrap_paired_signflip_v1"
DYNAMIC_COMPARISONS = (
    ("dct_q2", "mean_pool"),
    ("dct_q2", "final_token"),
    ("dct_q2", "max_pool"),
    ("dct_q2", "mean_final_pool"),
    ("dct_q2", "mean_max_pool"),
    ("dct_q2", "final_max_pool"),
    ("dct_q2", "learned_temporal_h2"),
    ("dct_q2", "dct_q4"),
    ("dct_q2", "dct_q8"),
    ("mean_pool", "learned_temporal_h1"),
)
BENCH_COMPARISONS = (
    ("dct_dc_ac_fullrank", "dct_dc_fullrank"),
    ("dct_dc_ac_fullrank", "dct_ac_fullrank"),
)
PRIMARY_BENCH_BASELINES = ("phylolm",)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(temporary, path)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_samples: int,
) -> list[float]:
    means = np.empty(n_samples, dtype=np.float64)
    chunk_size = 1000
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        indices = rng.integers(
            0,
            len(values),
            size=(stop - start, len(values)),
        )
        means[start:stop] = values[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _signflip_pvalue(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    observed = abs(float(differences.mean()))
    exceed = 0
    chunk_size = 1000
    for start in range(0, n_permutations, chunk_size):
        count = min(chunk_size, n_permutations - start)
        signs = rng.integers(
            0,
            2,
            size=(count, len(differences)),
            dtype=np.int8,
        )
        signs = signs.astype(np.float32) * 2.0 - 1.0
        permuted = np.abs(
            (signs * differences[None, :]).mean(axis=1)
        )
        exceed += int(np.count_nonzero(permuted >= observed - 1e-12))
    return float((exceed + 1) / (n_permutations + 1))


def _holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(np.asarray(pvalues))
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * pvalues[int(index)])
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted.tolist()


def _cluster_correctness(
    labels: np.ndarray,
    predictions: np.ndarray,
    split_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.column_stack((split_ids, labels))
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    correct = (labels == predictions).astype(np.float64)
    sums = np.bincount(inverse, weights=correct)
    counts = np.bincount(inverse)
    return unique_keys, sums / counts


def _dynamic_statistics(
    prediction_paths: list[Path],
    *,
    bootstrap_samples: int,
    permutations: int,
    seed: int,
) -> dict:
    proxy_reports: dict[str, dict] = {}
    raw_tests: list[dict] = []
    aggregate_differences: dict[tuple[int, str, str], list[np.ndarray]] = {}
    for proxy_index, path in enumerate(prediction_paths):
        proxy = path.stem.replace("_predictions", "")
        with np.load(path, allow_pickle=False) as data:
            proxy_report: dict[str, dict] = {}
            for k in (1, 100):
                methods = sorted(
                    {
                        key[: -len(f"_k{k}_labels")]
                        for key in data.files
                        if key.endswith(f"_k{k}_labels")
                    }
                )
                method_report: dict[str, dict] = {}
                clusters_by_method: dict[
                    str, tuple[np.ndarray, np.ndarray]
                ] = {}
                for method in methods:
                    prefix = f"{method}_k{k}"
                    labels = np.asarray(
                        data[f"{prefix}_labels"],
                        dtype=np.int64,
                    )
                    predictions = np.asarray(
                        data[f"{prefix}_predictions"],
                        dtype=np.int64,
                    )
                    split_ids = np.asarray(
                        data[f"{prefix}_fold_ids"],
                        dtype=np.int64,
                    )
                    keys, cluster_accuracy = _cluster_correctness(
                        labels,
                        predictions,
                        split_ids,
                    )
                    clusters_by_method[method] = (
                        keys,
                        cluster_accuracy,
                    )
                    rng = np.random.default_rng(
                        seed + proxy_index * 1009 + k * 17
                    )
                    method_report[method] = {
                        "n_predictions": int(len(labels)),
                        "n_fold_source_clusters": int(len(keys)),
                        "accuracy": float(
                            np.mean(labels == predictions)
                        ),
                        "accuracy_cluster_bootstrap_95ci": (
                            _bootstrap_mean_ci(
                                cluster_accuracy,
                                rng=rng,
                                n_samples=bootstrap_samples,
                            )
                        ),
                        "macro_f1": float(
                            f1_score(
                                labels,
                                predictions,
                                average="macro",
                            )
                        ),
                    }
                comparisons: list[dict] = []
                for comparison_index, (method_a, method_b) in enumerate(
                    DYNAMIC_COMPARISONS
                ):
                    if (
                        method_a not in clusters_by_method
                        or method_b not in clusters_by_method
                    ):
                        continue
                    keys_a, accuracy_a = clusters_by_method[method_a]
                    keys_b, accuracy_b = clusters_by_method[method_b]
                    if not np.array_equal(keys_a, keys_b):
                        raise ValueError(
                            f"{proxy} K={k}: cluster ordering differs for "
                            f"{method_a} and {method_b}"
                        )
                    differences = accuracy_a - accuracy_b
                    rng = np.random.default_rng(
                        seed
                        + proxy_index * 1009
                        + k * 17
                        + comparison_index * 7919
                    )
                    row = {
                        "method_a": method_a,
                        "method_b": method_b,
                        "accuracy_delta_a_minus_b": float(
                            differences.mean()
                        ),
                        "delta_cluster_bootstrap_95ci": _bootstrap_mean_ci(
                            differences,
                            rng=rng,
                            n_samples=bootstrap_samples,
                        ),
                        "paired_cluster_signflip_p_raw": _signflip_pvalue(
                            differences,
                            rng=rng,
                            n_permutations=permutations,
                        ),
                        "n_fold_source_clusters": int(len(differences)),
                    }
                    comparisons.append(row)
                    raw_tests.append(row)
                    aggregate_differences.setdefault(
                        (k, method_a, method_b),
                        [],
                    ).append(differences)
                proxy_report[str(k)] = {
                    "methods": method_report,
                    "comparisons": comparisons,
                }
            proxy_reports[proxy] = proxy_report

    adjusted = _holm_adjust(
        [
            float(row["paired_cluster_signflip_p_raw"])
            for row in raw_tests
        ]
    )
    for row, p_adjusted in zip(raw_tests, adjusted, strict=True):
        row["paired_cluster_signflip_p_holm"] = p_adjusted
        low, high = row["delta_cluster_bootstrap_95ci"]
        row["significant_0_05"] = bool(
            p_adjusted < 0.05 and (low > 0 or high < 0)
        )

    aggregate_reports: dict[str, dict] = {}
    for aggregate_index, (
        (k, method_a, method_b),
        values,
    ) in enumerate(sorted(aggregate_differences.items())):
        if len(values) != len(prediction_paths):
            continue
        differences = np.concatenate(values)
        rng = np.random.default_rng(
            seed + 1_000_003 + aggregate_index * 7919
        )
        key = f"k{k}:{method_a}-vs-{method_b}"
        aggregate_reports[key] = {
            "K": k,
            "method_a": method_a,
            "method_b": method_b,
            "n_proxies": len(values),
            "n_fold_source_clusters": int(len(differences)),
            "accuracy_delta_a_minus_b": float(differences.mean()),
            "delta_cluster_bootstrap_95ci": _bootstrap_mean_ci(
                differences,
                rng=rng,
                n_samples=bootstrap_samples,
            ),
            "paired_cluster_signflip_p_raw": _signflip_pvalue(
                differences,
                rng=rng,
                n_permutations=permutations,
            ),
        }
    aggregate_adjusted = _holm_adjust(
        [
            float(row["paired_cluster_signflip_p_raw"])
            for row in aggregate_reports.values()
        ]
    )
    for row, p_adjusted in zip(
        aggregate_reports.values(),
        aggregate_adjusted,
        strict=True,
    ):
        row["paired_cluster_signflip_p_holm"] = p_adjusted
        low, high = row["delta_cluster_bootstrap_95ci"]
        row["significant_0_05"] = bool(
            p_adjusted < 0.05 and (low > 0 or high < 0)
        )
    return {
        "proxies": proxy_reports,
        "four_proxy_aggregate": aggregate_reports,
    }


def _bench_statistics(
    report_paths: list[Path],
    *,
    bootstrap_samples: int,
    permutations: int,
    seed: int,
) -> dict:
    proxy_reports: dict[str, dict] = {}
    tests: list[dict] = []
    for proxy_index, path in enumerate(report_paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        proxy = str(report["proxy_tag"])
        configs = report["configs"]
        comparisons: list[dict] = []
        for comparison_index, (method_a, method_b) in enumerate(
            BENCH_COMPARISONS
        ):
            rows_a = configs[method_a]["splits"]
            rows_b = configs[method_b]["splits"]
            ids_a = [row["split_id"] for row in rows_a]
            ids_b = [row["split_id"] for row in rows_b]
            if ids_a != ids_b:
                raise ValueError(
                    f"{proxy}: split ordering differs for "
                    f"{method_a} and {method_b}"
                )
            comparison: dict[str, object] = {
                "method_a": method_a,
                "method_b": method_b,
                "n_splits": len(rows_a),
                "metrics": {},
            }
            for metric_index, metric in enumerate(("accuracy", "f1", "auc")):
                values_a = np.asarray(
                    [row[metric] for row in rows_a],
                    dtype=np.float64,
                )
                values_b = np.asarray(
                    [row[metric] for row in rows_b],
                    dtype=np.float64,
                )
                differences = values_a - values_b
                rng = np.random.default_rng(
                    seed
                    + proxy_index * 1009
                    + comparison_index * 7919
                    + metric_index * 17
                )
                comparison["metrics"][metric] = {
                    "delta_a_minus_b": float(differences.mean()),
                    "delta_split_bootstrap_95ci": _bootstrap_mean_ci(
                        differences,
                        rng=rng,
                        n_samples=bootstrap_samples,
                    ),
                    "paired_split_signflip_p_raw": _signflip_pvalue(
                        differences,
                        rng=rng,
                        n_permutations=permutations,
                    ),
                }
                tests.append(comparison["metrics"][metric])
            comparisons.append(comparison)
        report_key = f"{proxy}:{report['split_type']}"
        proxy_reports[report_key] = {
            "proxy_tag": proxy,
            "split_type": report["split_type"],
            "comparisons": comparisons,
        }
    adjusted = _holm_adjust(
        [float(row["paired_split_signflip_p_raw"]) for row in tests]
    )
    for row, p_adjusted in zip(tests, adjusted, strict=True):
        row["paired_split_signflip_p_holm"] = p_adjusted
        low, high = row["delta_split_bootstrap_95ci"]
        row["significant_0_05"] = bool(
            p_adjusted < 0.05 and (low > 0 or high < 0)
        )
    return {"proxies": proxy_reports}


def _bench_baseline_statistics(
    reader_report_paths: list[Path],
    baseline_report_paths: list[Path],
    *,
    bootstrap_samples: int,
    permutations: int,
    seed: int,
) -> dict:
    baselines_by_split: dict[str, dict] = {}
    for path in baseline_report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        split_type = str(report["split_type"])
        if split_type in baselines_by_split:
            raise ValueError(
                f"duplicate Bench-A baseline report for {split_type}"
            )
        baselines_by_split[split_type] = report

    reports: dict[str, dict] = {}
    tests: list[dict] = []
    for reader_index, path in enumerate(reader_report_paths):
        reader = json.loads(path.read_text(encoding="utf-8"))
        split_type = str(reader["split_type"])
        if split_type not in baselines_by_split:
            continue
        proxy = str(reader["proxy_tag"])
        reader_rows = reader["configs"]["dct_dc_ac_fullrank"]["splits"]
        reader_ids = [row["split_id"] for row in reader_rows]
        comparisons: list[dict] = []
        baseline_configs = baselines_by_split[split_type]["configs"]
        for baseline_index, baseline in enumerate(
            PRIMARY_BENCH_BASELINES
        ):
            if baseline not in baseline_configs:
                raise ValueError(
                    f"{split_type}: missing primary baseline {baseline}"
                )
            payload = baseline_configs[baseline]
            baseline_rows = payload["splits"]
            baseline_ids = [row["split_id"] for row in baseline_rows]
            if reader_ids != baseline_ids:
                raise ValueError(
                    f"{proxy}:{split_type}: split ordering differs for "
                    f"READER and {baseline}"
                )
            comparison: dict[str, object] = {
                "method_a": "dct_dc_ac_fullrank",
                "method_b": baseline,
                "n_splits": len(reader_rows),
                "metrics": {},
            }
            for metric_index, metric in enumerate(
                ("accuracy", "f1", "auc")
            ):
                reader_values = np.asarray(
                    [row[metric] for row in reader_rows],
                    dtype=np.float64,
                )
                baseline_values = np.asarray(
                    [row[metric] for row in baseline_rows],
                    dtype=np.float64,
                )
                differences = reader_values - baseline_values
                rng = np.random.default_rng(
                    seed
                    + reader_index * 1009
                    + baseline_index * 7919
                    + metric_index * 17
                )
                metric_report = {
                    "delta_a_minus_b": float(differences.mean()),
                    "delta_split_bootstrap_95ci": _bootstrap_mean_ci(
                        differences,
                        rng=rng,
                        n_samples=bootstrap_samples,
                    ),
                    "paired_split_signflip_p_raw": _signflip_pvalue(
                        differences,
                        rng=rng,
                        n_permutations=permutations,
                    ),
                }
                comparison["metrics"][metric] = metric_report
                tests.append(metric_report)
            comparisons.append(comparison)
        report_key = f"{proxy}:{split_type}"
        reports[report_key] = {
            "proxy_tag": proxy,
            "split_type": split_type,
            "comparisons": comparisons,
        }

    adjusted = _holm_adjust(
        [float(row["paired_split_signflip_p_raw"]) for row in tests]
    )
    for row, p_adjusted in zip(tests, adjusted, strict=True):
        row["paired_split_signflip_p_holm"] = p_adjusted
        low, high = row["delta_split_bootstrap_95ci"]
        row["significant_0_05"] = bool(
            p_adjusted < 0.05 and (low > 0 or high < 0)
        )
    return {"reader_vs_cached_baselines": reports}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dynamic-predictions",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--bench-report",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--bench-baseline-report",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--permutations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (
        not args.dynamic_predictions
        and not args.bench_report
        and not args.bench_baseline_report
    ):
        parser.error("provide at least one dynamic or Bench-A input")
    if args.bench_baseline_report and not args.bench_report:
        parser.error(
            "--bench-baseline-report requires --bench-report inputs"
        )
    if args.output_json.exists():
        print(f"[statistics] skip existing {args.output_json}", flush=True)
        return
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "bootstrap_samples": args.bootstrap_samples,
        "signflip_permutations": args.permutations,
        "random_seed": args.seed,
        "multiple_testing": (
            "Holm family-wise correction across all reported comparisons "
            "within each experiment family"
        ),
    }
    if args.dynamic_predictions:
        payload["dynamic"] = _dynamic_statistics(
            args.dynamic_predictions,
            bootstrap_samples=args.bootstrap_samples,
            permutations=args.permutations,
            seed=args.seed,
        )
    if args.bench_report:
        payload["bench_a"] = _bench_statistics(
            args.bench_report,
            bootstrap_samples=args.bootstrap_samples,
            permutations=args.permutations,
            seed=args.seed,
        )
        if args.bench_baseline_report:
            payload["bench_a"].update(
                _bench_baseline_statistics(
                    args.bench_report,
                    args.bench_baseline_report,
                    bootstrap_samples=args.bootstrap_samples,
                    permutations=args.permutations,
                    seed=args.seed,
                )
            )
    payload["complete"] = True
    _atomic_json(args.output_json, payload)
    print(f"[statistics] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
