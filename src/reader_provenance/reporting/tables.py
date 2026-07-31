"""Generate compact machine-readable tables for the paper experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _models(config: Path, role: str) -> list[dict[str, Any]]:
    rows = yaml.safe_load(config.read_text(encoding="utf-8"))["models"]
    return [row for row in rows if role in row.get("roles", [])]


def _fold_row(
    report: dict[str, Any], section: str, budget: int
) -> dict[str, float]:
    source = report[section][str(budget)]
    return {
        "accuracy": float(source["accuracy_mean"]),
        "accuracy_std": float(source["accuracy_std_across_folds"]),
        "macro_f1": float(source["macro_f1_mean"]),
        "macro_f1_std": float(source["macro_f1_std_across_folds"]),
    }


def _endpoint_record(
    method: str,
    k1: dict[str, float],
    k100: dict[str, float],
) -> dict[str, Any]:
    output: dict[str, Any] = {"method": method}
    for budget, row in ((1, k1), (100, k100)):
        for metric in ("accuracy", "macro_f1"):
            output[f"{metric}_k{budget}"] = row[metric]
            output[f"{metric}_k{budget}_fold_std"] = row[f"{metric}_std"]
    return output


def _random_endpoint(results: Path, budget: int) -> dict[str, float]:
    source = _json(results / "statistics/fold_variation/random.json")[
        "agent500"
    ][str(budget)]
    return {
        "accuracy": float(source["accuracy_mean"]),
        "accuracy_std": float(source["accuracy_std"]),
        "macro_f1": float(source["macro_f1_mean"]),
        "macro_f1_std": float(source["macro_f1_std"]),
    }


def _deberta_endpoint(results: Path, budget: int) -> dict[str, float]:
    rows = [
        _json(results / "baselines/deberta" / f"fold-{fold}.json")["metrics"][
            f"K{budget}"
        ]
        for fold in range(5)
    ]
    output = {}
    for metric in ("accuracy", "macro_f1"):
        values = np.asarray([row[metric] for row in rows])
        output[metric] = float(values.mean())
        output[f"{metric}_std"] = float(values.std(ddof=0))
    return output


def _llmmap_endpoint(results: Path, budget: int) -> dict[str, float]:
    source = _json(results / "baselines/llmmap/summary.json")["summary"][
        "clean"
    ][str(budget)]
    output = {}
    for metric in ("accuracy", "macro_f1"):
        output[metric] = float(source[metric]["mean"])
        output[f"{metric}_std"] = float(source[metric]["std_across_folds"])
    return output


def _dna_endpoint(results: Path, encoder: str, budget: int) -> dict[str, float]:
    report = _json(
        results / "baselines/dna_dynamic" / f"{encoder}_crossK{budget}.json"
    )
    return {
        "accuracy": float(report["classification_accuracy_fold_mean"]),
        "accuracy_std": float(report["classification_accuracy_fold_std"]),
        "macro_f1": float(report["classification_macro_f1_fold_mean"]),
        "macro_f1_std": float(report["classification_macro_f1_fold_std"]),
    }


def write_endpoint_table(results: Path, config: Path, output: Path) -> Path:
    rows = [
        _endpoint_record(
            "Random", _random_endpoint(results, 1), _random_endpoint(results, 100)
        ),
        _endpoint_record(
            "FT DeBERTa",
            _deberta_endpoint(results, 1),
            _deberta_endpoint(results, 100),
        ),
        _endpoint_record(
            "LLMmap",
            _llmmap_endpoint(results, 1),
            _llmmap_endpoint(results, 100),
        ),
    ]
    for encoder, label in (
        ("mpnet", "DNA / MPNet"),
        ("bge", "DNA / BGE"),
        ("qwen3emb", "DNA / Qwen-Emb."),
    ):
        rows.append(
            _endpoint_record(
                label,
                _dna_endpoint(results, encoder, 1),
                _dna_endpoint(results, encoder, 100),
            )
        )
    for encoder, label in (
        ("mpnet", "FT DNA / MPNet"),
        ("bge", "FT DNA / BGE"),
        ("qwen3emb", "FT DNA / Qwen-Emb."),
    ):
        report = _json(
            results / "statistics/fold_variation" / f"finetuned_{encoder}.json"
        )
        rows.append(
            _endpoint_record(
                label,
                _fold_row(report, "agent500", 1),
                _fold_row(report, "agent500", 100),
            )
        )
    for model in _models(config, "full"):
        report = _json(
            results
            / "statistics/fold_variation"
            / f"reader_{model['tag']}.json"
        )
        rows.append(
            _endpoint_record(
                f"READER / {model['name']}",
                _fold_row(report, "agent500", 1),
                _fold_row(report, "agent500", 100),
            )
        )
    return _write(output, rows)


def _metric_columns(summary: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(summary[key])
        for key in (
            "accuracy_mean",
            "accuracy_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
            "auc_mean",
            "auc_std",
        )
    }


def bench_a_pair_table(results: Path, config: Path, output: Path) -> Path:
    rows: list[dict[str, Any]] = []
    text = _json(results / "bench_a/baselines/summary_all__text_baselines.json")
    for payload, label in zip(text, ("MPT", "PhyloLM"), strict=True):
        rows.append({"method": label, **_metric_columns(payload)})
    for encoder, label in (
        ("mpnet", "DNA / MPNet"),
        ("bge", "DNA / BGE"),
        ("qwen3emb", "DNA / Qwen-Emb."),
    ):
        payloads = _json(
            results / f"bench_a/baselines/summary_all__llm_dna_paper_{encoder}.json"
        )
        payload = next(row for row in payloads if row["svm_kernel"] == "linear")
        rows.append({"method": label, **_metric_columns(payload)})
    components = _json(results / "bench_a/components.json")["profiles"][
        "full500_best_layer"
    ]["fullrank_dc_ac_score_fusion"]["by_proxy"]
    for model in _models(config, "main"):
        summary = components[model["tag"]]
        rows.append({"method": f"READER / {model['name']}", **_metric_columns(summary)})
    return _write(output, rows)


def bench_a_disjoint_table(results: Path, config: Path, output: Path) -> Path:
    rows = []
    for model in _models(config, "main"):
        row: dict[str, Any] = {"method": f"READER / {model['name']}"}
        for directory, prefix in (("model", "model"), ("family", "family")):
            report = _json(
                results / f"bench_a/disjoint/{directory}/{model['tag']}.json"
            )
            summary = report["configs"]["dct_dc_ac_fullrank"]["summary"]
            for metric in ("accuracy", "f1", "auc"):
                row[f"{prefix}_{metric}"] = float(summary[f"{metric}_mean"])
            low, high = summary["accuracy_bootstrap_95ci"]
            row[f"{prefix}_accuracy_ci_low"] = float(low)
            row[f"{prefix}_accuracy_ci_high"] = float(high)
        rows.append(row)
    return _write(output, rows)


def bench_a_component_table(results: Path, output: Path) -> Path:
    report = _json(results / "bench_a/components.json")["profiles"][
        "full500_best_layer"
    ]
    names = (
        ("fullrank_dc", "DC cosine"),
        ("fullrank_ac", "First-AC cosine"),
        ("fullrank_dc_ac_concat", "DC-AC score readout"),
        ("grp128_ac", "GRP-128 first-AC cosine"),
    )
    rows = []
    for key, label in names:
        summary = report[key]["macro_average"]
        rows.append(
            {
                "pair_feature": label,
                "accuracy": float(summary["accuracy"]),
                "f1": float(summary["f1"]),
                "auc": float(summary["auc"]),
            }
        )
    return _write(output, rows)


def temporal_table(results: Path, config: Path, output: Path) -> Path:
    reports = [
        _json(results / f"ablations/temporal/reports/{model['tag']}.json")
        for model in _models(config, "main")
    ]
    labels = (
        ("mean_pool", "All-token mean (DC)", "d", False),
        ("final_token", "Final token", "d", False),
        ("max_pool", "Coordinate-wise maximum", "d", False),
        ("mean_final_pool", "Mean + final", "2d", False),
        ("mean_max_pool", "Mean + maximum", "2d", False),
        ("final_max_pool", "Final + maximum", "2d", False),
        ("learned_temporal_h1", "Learned one-head mixture", "d", True),
        ("learned_temporal_h2", "Learned two-head mixture", "2d", True),
        ("dct_q2", "DC-AC (DCT q=2)", "2d", False),
        ("dct_q4", "DCT q=4", "4d", False),
        ("dct_q8", "DCT q=8", "8d", False),
    )
    rows = []
    for key, label, dimension, learned in labels:
        row: dict[str, Any] = {
            "representation": label,
            "dimension": dimension,
            "learned_temporal_weights": learned,
        }
        for budget in (1, 100):
            for metric in ("accuracy", "macro_f1"):
                values = [
                    report["configs"][key]["metrics_across_grouping_seeds"][
                        str(budget)
                    ][f"{metric}_mean"]
                    for report in reports
                ]
                row[f"{metric}_k{budget}"] = float(np.mean(values))
        rows.append(row)
    return _write(output, rows)


def efficiency_table(results: Path, output: Path) -> Path:
    profiles = results / "efficiency/profiles"
    inference_paths = sorted(profiles.glob("*-inference*.json"))
    rows = []
    for path in inference_paths:
        if "_bs" in path.stem:
            continue
        inference = _json(path)
        prefix = path.name.split("-inference", 1)[0]
        suffix = path.name.split("-inference", 1)[1].removesuffix(".json")
        train_name = f"{prefix}-train{suffix}.json"
        train_path = profiles / train_name
        train = _json(train_path) if train_path.exists() else {}
        rows.append(
            {
                "method": str(inference["method"]),
                "tag": str(inference["tag"]),
                "hardware": str(inference["hardware"]),
                "trainable_parameters": int(inference["trainable_parameters"]),
                "fit_seconds_per_fold": train.get("median_seconds", ""),
                "inference_parameters": int(inference["inference_parameters"]),
                "seconds_per_response": float(
                    inference["timing"]["median_ms_per_response"]
                )
                / 1000.0,
                "batch_size": int(inference["batch_size"]),
                "timing_scope": str(inference["timing_scope"]),
            }
        )
    return _write(output, rows)


def render_all(results: Path, config: Path, output_dir: Path) -> list[Path]:
    return [
        bench_a_pair_table(results, config, output_dir / "bench_a_pair.csv"),
        bench_a_disjoint_table(results, config, output_dir / "bench_a_disjoint.csv"),
        bench_a_component_table(results, output_dir / "bench_a_components.csv"),
        temporal_table(results, config, output_dir / "temporal_controls.csv"),
        efficiency_table(results, output_dir / "efficiency.csv"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--proxy-config", type=Path, default=Path("configs/proxies.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    args = parser.parse_args()
    render_all(args.results, args.proxy_config, args.output_dir)


if __name__ == "__main__":
    main()
