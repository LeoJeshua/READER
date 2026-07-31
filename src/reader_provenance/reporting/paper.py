from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr, spearmanr

from reader_provenance.reporting.tables import write_endpoint_table

BUDGETS = (1, 5, 10, 20, 50, 100)
COMPONENTS = {
    "DC": "dct_dc_fullrank",
    "First AC": "dct_ac_fullrank",
    "DC-AC": "dct_dc_ac_fullrank",
}
METRIC_KEYS = {"accuracy": "accuracy_mean", "macro_f1": "macro_f1_mean"}
BASELINE_COLORS = {
    "LLMmap": "#4d4d4d",
    "DNA / MPNet": "#999999",
    "DNA / BGE": "#e69f00",
    "DNA / Qwen-Emb.": "#56b4e9",
    "FT DeBERTa": "#000000",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _models(config: Path, role: str) -> list[dict[str, Any]]:
    rows = _load_yaml(config)["models"]
    selected = [row for row in rows if role in row.get("roles", [])]
    expected = 4 if role == "main" else 8
    if len(selected) != expected:
        raise ValueError(f"expected {expected} {role} proxies, found {len(selected)}")
    return selected


def _axes(count: int) -> tuple[plt.Figure, np.ndarray]:
    if count == 4:
        figure, axes = plt.subplots(1, 4, figsize=(10.0, 2.45), sharey=True)
    elif count == 8:
        figure, axes = plt.subplots(2, 4, figsize=(10.0, 4.8), sharey=True)
    else:
        raise ValueError("paper layouts support four or eight proxies")
    return figure, np.asarray(axes).reshape(-1)


def _style_axis(axis: plt.Axes, title: str, ylabel: str | None = None) -> None:
    axis.set_title(title, fontsize=8.5)
    axis.grid(True, color="#dddddd", linewidth=0.55, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7.2)
    axis.set_xticks(BUDGETS)
    axis.set_xscale("log")
    axis.set_xlabel("Query budget K", fontsize=7.8)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=7.8)


def _reader_curve(
    results: Path,
    tag: str,
    metric: str,
    *,
    view: str = "ur",
    component: str = "dct_dc_ac_fullrank",
) -> np.ndarray:
    report = _load_json(results / "reader/agent500" / f"{tag}_{view}.json")
    summary = report["configs"][component]["cv_metrics_across_grouping_seeds"]
    key = METRIC_KEYS[metric]
    return np.asarray([summary[str(k)][key] for k in BUDGETS])


def _baseline_curves(results: Path, metric: str) -> dict[str, np.ndarray]:
    key = METRIC_KEYS[metric]
    deberta = _load_json(results / "baselines/deberta/aggregate.json")
    deberta_summary = deberta["agent500"]["metrics_across_grouping_seeds"]
    llmmap = _load_json(results / "baselines/llmmap/summary.json")["summary"][
        "clean"
    ]
    output = {
        "FT DeBERTa": np.asarray([deberta_summary[str(k)][key] for k in BUDGETS]),
        "LLMmap": np.asarray([llmmap[str(k)][metric]["mean"] for k in BUDGETS]),
    }
    dna_fields = {
        "accuracy": "classification_accuracy",
        "macro_f1": "classification_macro_f1",
    }
    for encoder, label in (
        ("mpnet", "DNA / MPNet"),
        ("bge", "DNA / BGE"),
        ("qwen3emb", "DNA / Qwen-Emb."),
    ):
        output[label] = np.asarray(
            [
                _load_json(
                    results / "baselines/dna_dynamic" / f"{encoder}_crossK{k}.json"
                )[dna_fields[metric]]
                for k in BUDGETS
            ]
        )
    return output


def render_agent500(
    results: Path,
    config: Path,
    output: Path,
    *,
    role: str,
    metric: str,
) -> Path:
    models = _models(config, role)
    baselines = _baseline_curves(results, metric)
    figure, axes = _axes(len(models))
    for index, (axis, model) in enumerate(zip(axes, models, strict=True)):
        for name, values in baselines.items():
            axis.plot(
                BUDGETS,
                values,
                color=BASELINE_COLORS[name],
                linewidth=0.8,
                linestyle="--",
                alpha=0.85,
                label=name,
            )
        axis.plot(
            BUDGETS,
            _reader_curve(results, model["tag"], metric),
            color=model["color"],
            marker="o",
            markersize=3.0,
            linewidth=1.7,
            label="READER",
        )
        _style_axis(
            axis,
            model["name"],
            "Accuracy" if metric == "accuracy" and index % 4 == 0 else None,
        )
        axis.set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        fontsize=7.0,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91), pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return output


def render_components(
    results: Path,
    config: Path,
    output: Path,
    *,
    role: str,
) -> Path:
    models = _models(config, role)
    colors = {"DC": "#0072b2", "First AC": "#d55e00", "DC-AC": "#009e73"}
    figure, axes = _axes(len(models))
    for index, (axis, model) in enumerate(zip(axes, models, strict=True)):
        for label, component in COMPONENTS.items():
            axis.plot(
                BUDGETS,
                _reader_curve(
                    results,
                    model["tag"],
                    "accuracy",
                    component=component,
                ),
                color=colors[label],
                marker="o",
                markersize=2.8,
                linewidth=1.25,
                label=label,
            )
        _style_axis(axis, model["name"], "Accuracy" if index % 4 == 0 else None)
        axis.set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=7.2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return output


def render_inputs(
    results: Path,
    config: Path,
    output: Path,
    *,
    role: str,
) -> Path:
    models = _models(config, role)
    figure, axes = _axes(len(models))
    for index, (axis, model) in enumerate(zip(axes, models, strict=True)):
        for view, label, style in (
            ("r", "Response only", "--"),
            ("ur", "Prompt + response", "-"),
        ):
            axis.plot(
                BUDGETS,
                _reader_curve(results, model["tag"], "accuracy", view=view),
                color=model["color"],
                linestyle=style,
                marker="o",
                markersize=2.8,
                linewidth=1.35,
                label=label,
            )
        _style_axis(axis, model["name"], "Accuracy" if index % 4 == 0 else None)
        axis.set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=7.2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return output


def _layer_rows(report: dict[str, Any], method: str) -> list[dict[str, Any]]:
    old_names = {
        "last": "last_token_linear_probe",
        "dc": "dct_dc_linear_probe",
        "ac": "dct_ac_linear_probe",
        "dc-ac": "dct_q2_linear_probe",
    }
    payload = report["methods"].get(method) or report["methods"][old_names[method]]
    return payload.get("layers") or payload["layer_results"]


def render_layers(
    results: Path,
    config: Path,
    output: Path,
    *,
    role: str,
) -> Path:
    models = _models(config, role)
    styles = {
        "last": ("Last token", "#777777", ":"),
        "dc": ("DC", "#0072b2", "-"),
        "ac": ("First AC", "#d55e00", "-"),
        "dc-ac": ("DC-AC", "#009e73", "--"),
    }
    figure, axes = _axes(len(models))
    for index, (axis, model) in enumerate(zip(axes, models, strict=True)):
        report = _load_json(results / "reader/layer_scan" / f"{model['tag']}.json")
        for method, (label, color, linestyle) in styles.items():
            rows = _layer_rows(report, method)
            x = [row.get("layer", row.get("layer_index")) for row in rows]
            y = [row["accuracy"] for row in rows]
            axis.plot(
                x,
                y,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.1,
            )
            best = max(rows, key=lambda row: (row["accuracy"], -x[rows.index(row)]))
            best_x = best.get("layer", best.get("layer_index"))
            axis.scatter(best_x, best["accuracy"], s=15, color=color, zorder=3)
        axis.set_title(model["name"], fontsize=8.5)
        axis.set_xlabel("Layer", fontsize=7.8)
        if index % 4 == 0:
            axis.set_ylabel("Accuracy", fontsize=7.8)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.grid(True, color="#dddddd", linewidth=0.55, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=7.2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return output


def _variance_value(report: dict[str, Any], component: str) -> tuple[float, float]:
    old_names = {
        "dc": "dct_dc_fullrank",
        "ac": "dct_ac_fullrank",
        "dc-ac": "dct_dc_ac_fullrank",
    }
    row = report["summary"].get(component) or report["summary"][old_names[component]]
    return (
        float(row["source_prompt_ratio_fold_mean"]),
        float(row["source_prompt_ratio_fold_std"]),
    )


def render_variance(
    results: Path,
    config: Path,
    output: Path,
    *,
    role: str,
) -> Path:
    models = _models(config, role)
    figure, axes = _axes(len(models))
    width = 0.36
    labels = ("DC", "First AC", "DC-AC")
    components = ("dc", "ac", "dc-ac")
    for index, (axis, model) in enumerate(zip(axes, models, strict=True)):
        for view_index, (view, color) in enumerate(
            (("r", "#d55e00"), ("ur", "#0072b2"))
        ):
            report = _load_json(
                results / "reader/variance" / f"{model['tag']}_{view}.json"
            )
            values, errors = zip(
                *[_variance_value(report, component) for component in components],
                strict=True,
            )
            x = np.arange(3) + (view_index - 0.5) * width
            axis.bar(
                x,
                values,
                width,
                yerr=errors,
                color=color,
                alpha=0.82,
                label="Response only" if view == "r" else "Prompt + response",
                capsize=2,
            )
        axis.set_title(model["name"], fontsize=8.5)
        axis.set_xticks(np.arange(3), labels, fontsize=7.0)
        if index % 4 == 0:
            axis.set_ylabel("Source / prompt variance", fontsize=7.8)
        axis.grid(True, axis="y", color="#dddddd", linewidth=0.55, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="y", labelsize=7.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=7.2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return output


def _external_curve(
    results: Path,
    tag: str,
    panel: str,
    metric: str,
) -> np.ndarray:
    directory = "controlled_length" if panel.startswith("length") else "math100"
    report = _load_json(results / "reader/stress" / directory / f"{tag}.json")
    external = report["configs"]["dct_dc_ac_fullrank"]["external_evaluations"][
        panel
    ]["metrics_across_grouping_seeds"]
    return np.asarray([external[str(k)][METRIC_KEYS[metric]] for k in BUDGETS])


def render_math100(
    results: Path,
    config: Path,
    output: Path,
    *,
    role: str,
    metric: str,
) -> Path:
    models = [
        model
        for model in _models(config, role)
        if (results / "reader/stress/math100" / f"{model['tag']}.json").exists()
    ]
    figure, axes = _axes(4 if role == "main" else 8)
    for index, axis in enumerate(axes):
        if index >= len(models):
            axis.axis("off")
            continue
        model = models[index]
        axis.plot(
            BUDGETS,
            _external_curve(results, model["tag"], "math100", metric),
            color=model["color"],
            marker="o",
            markersize=3,
            linewidth=1.5,
        )
        ylabel = "Accuracy" if metric == "accuracy" and index % 4 == 0 else None
        _style_axis(axis, model["name"], ylabel)
        axis.set_ylim(0, 0.42)
    figure.tight_layout(pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return output


def render_lengths(
    results: Path,
    config: Path,
    output: Path,
    *,
    role: str,
) -> Path:
    models = _models(config, role)
    figure, axes = _axes(len(models))
    conditions = (("length32", "32"), ("length64", "64"), ("length128", "128"))
    for index, (axis, model) in enumerate(zip(axes, models, strict=True)):
        values = [
            _external_curve(results, model["tag"], name, "accuracy")[-1]
            for name, _label in conditions
        ]
        axis.bar(
            np.arange(3),
            values,
            color=model["color"],
            alpha=(0.65),
        )
        axis.set_title(model["name"], fontsize=8.5)
        axis.set_xticks(np.arange(3), [label for _name, label in conditions])
        axis.set_xlabel("Response tokens", fontsize=7.8)
        if index % 4 == 0:
            axis.set_ylabel("Accuracy at K=100", fontsize=7.8)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_ylim(0, 1.02)
        axis.grid(True, axis="y", color="#dddddd", linewidth=0.55, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7.2)
    figure.tight_layout(pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return output


def render_capability(
    results: Path,
    config: Path,
    capabilities: Path,
    output_dir: Path,
) -> list[Path]:
    models = _models(config, "full")
    scores = _load_yaml(capabilities)
    outputs = []
    for benchmark in ("mmlu_pro", "gpqa_diamond"):
        x = np.asarray([scores[benchmark]["scores"][row["tag"]] for row in models])
        for budget in (1, 100):
            y = np.asarray(
                [
                    _reader_curve(results, row["tag"], "accuracy")[
                        BUDGETS.index(budget)
                    ]
                    for row in models
                ]
            )
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            figure, axis = plt.subplots(figsize=(3.35, 2.72))
            for row, x_value, y_value in zip(models, x, y, strict=True):
                axis.scatter(
                    x_value,
                    y_value,
                    s=40,
                    color=row["color"],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
                axis.annotate(
                    row["name"].replace("-Instruct", ""),
                    (x_value, y_value),
                    xytext=(3, 4),
                    textcoords="offset points",
                    fontsize=5.6,
                )
            slope, intercept = np.polyfit(x, y, 1)
            x_fit = np.linspace(x.min(), x.max(), 100)
            axis.plot(x_fit, slope * x_fit + intercept, "--", color="#555555")
            axis.text(
                0.03,
                0.97,
                f"Pearson r = {pearson.statistic:.2f}\n"
                f"Spearman rho = {spearman.statistic:.2f}",
                transform=axis.transAxes,
                va="top",
                fontsize=7.0,
            )
            axis.set_xlabel(benchmark.replace("_", " ").upper(), fontsize=8)
            axis.set_ylabel(f"Agent500 Acc.@{budget}", fontsize=8)
            axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.grid(True, color="#dddddd", linewidth=0.55, alpha=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=7.2)
            figure.tight_layout(pad=0.4)
            output = output_dir / f"{benchmark}_vs_agent500_k{budget}.pdf"
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
            plt.close(figure)
            outputs.append(output)
    return outputs


def render_all(
    results: Path,
    config: Path,
    capabilities: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from reader_provenance.reporting.tables import render_all as render_tables

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    outputs: list[Path] = []
    for role in ("main", "full"):
        outputs.extend(
            [
                render_agent500(
                    results,
                    config,
                    output_dir / f"agent500_accuracy_{role}.pdf",
                    role=role,
                    metric="accuracy",
                ),
                render_agent500(
                    results,
                    config,
                    output_dir / f"agent500_macro_f1_{role}.pdf",
                    role=role,
                    metric="macro_f1",
                ),
                render_components(
                    results,
                    config,
                    output_dir / f"components_{role}.pdf",
                    role=role,
                ),
                render_inputs(
                    results,
                    config,
                    output_dir / f"inputs_{role}.pdf",
                    role=role,
                ),
                render_layers(
                    results,
                    config,
                    output_dir / f"layers_{role}.pdf",
                    role=role,
                ),
                render_variance(
                    results,
                    config,
                    output_dir / f"variance_{role}.pdf",
                    role=role,
                ),
                render_lengths(
                    results,
                    config,
                    output_dir / f"lengths_{role}.pdf",
                    role=role,
                ),
                render_math100(
                    results,
                    config,
                    output_dir / f"math100_accuracy_{role}.pdf",
                    role=role,
                    metric="accuracy",
                ),
                render_math100(
                    results,
                    config,
                    output_dir / f"math100_macro_f1_{role}.pdf",
                    role=role,
                    metric="macro_f1",
                ),
            ]
        )
    outputs.extend(render_capability(results, config, capabilities, output_dir))
    outputs.append(write_endpoint_table(results, config, output_dir / "endpoints.csv"))
    outputs.extend(render_tables(results, config, output_dir / "tables"))
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in outputs
        ],
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
