"""Render figures for the pair-conditional + aggregation ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = [
    ("Acc", "classification_accuracy"),
    ("Macro-F1", "classification_macro_f1"),
    ("Pair-AUC", "mean_pairwise_auc"),
    ("mAP@10", "retrieval_map_at_10"),
    ("ARI", "clustering_ari"),
    ("NMI", "clustering_nmi"),
]

METHOD_ORDER = [
    "mpnet_resp",
    "mpnet_resp_aggK4",
    "mpnet_pair",
    "mpnet_pair_aggK4",
    "proxy_resp_best",
    "proxy_resp_best_aggK4",
    "proxy_pair_best",
    "proxy_pair_best_aggK4",
]

COLOR = {
    "mpnet_resp":                "#2c7bb6",
    "mpnet_resp_aggK4":          "#08519c",
    "mpnet_pair":                "#74add1",
    "mpnet_pair_aggK4":          "#4575b4",
    "proxy_resp_best":           "#d7191c",
    "proxy_resp_best_aggK4":     "#a50f15",
    "proxy_pair_best":           "#fdae61",
    "proxy_pair_best_aggK4":     "#e6550d",
}


def _load_reports(report_dir: Path) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for p in sorted(report_dir.glob("*.json")):
        d = json.loads(p.read_text())
        reports[d.get("method", p.stem)] = d
    return reports


def plot_bars_all_metrics(reports: dict[str, dict], out_path: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in reports]
    n_metrics = len(METRIC_KEYS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (label, key) in zip(axes.flat, METRIC_KEYS):
        vals = [reports[m].get(key, 0.0) for m in methods]
        colors = [COLOR.get(m, "#888") for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_title(label, fontsize=12)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", fontsize=7)
    fig.suptitle("Pair-conditional × Multi-sample aggregation ablation (K=4)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aggregation_effect(reports: dict[str, dict], out_path: Path) -> None:
    """Grouped bars: no-agg vs agg-K=4, side by side per method family."""
    families = [
        ("mpnet_resp",       "mpnet_resp_aggK4",       "mpnet · response"),
        ("mpnet_pair",       "mpnet_pair_aggK4",       "mpnet · pair"),
        ("proxy_resp_best",  "proxy_resp_best_aggK4",  "proxy · response"),
        ("proxy_pair_best",  "proxy_pair_best_aggK4",  "proxy · pair"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [lab for _, _, lab in families]
    x = np.arange(len(families))
    w = 0.35

    acc_single = [reports[s]["classification_accuracy"] for s, _, _ in families]
    acc_agg    = [reports[a]["classification_accuracy"] for _, a, _ in families]

    b1 = ax.bar(x - w/2, acc_single, w, color="#bdbdbd", label="single-sample (N=720)")
    b2 = ax.bar(x + w/2, acc_agg,    w, color="#08519c", label="aggregate K=4 (N=180)")

    for bars, vals in ((b1, acc_single), (b2, acc_agg)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9)
    for xi, (s, a, _) in zip(x, families):
        delta = reports[a]["classification_accuracy"] - reports[s]["classification_accuracy"]
        ax.annotate(f"Δ={delta:+.3f}",
                    xy=(xi, max(reports[s]["classification_accuracy"],
                                reports[a]["classification_accuracy"]) + 0.06),
                    ha="center", fontsize=9, color="#d7191c", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Effect of multi-sample aggregation (K=4) on classification accuracy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pair_vs_response(reports: dict[str, dict], out_path: Path) -> None:
    """How does adding the prompt (pair mode) affect each encoder?"""
    groups = [
        ("mpnet · single", "mpnet_resp",            "mpnet_pair"),
        ("mpnet · aggK4",  "mpnet_resp_aggK4",      "mpnet_pair_aggK4"),
        ("proxy · single", "proxy_resp_best",       "proxy_pair_best"),
        ("proxy · aggK4",  "proxy_resp_best_aggK4", "proxy_pair_best_aggK4"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [g[0] for g in groups]
    x = np.arange(len(groups))
    w = 0.35

    resp = [reports[r]["classification_accuracy"] for _, r, _ in groups]
    pair = [reports[p]["classification_accuracy"] for _, _, p in groups]

    b1 = ax.bar(x - w/2, resp, w, color="#2c7bb6", label="response-only")
    b2 = ax.bar(x + w/2, pair, w, color="#fdae61", label="pair (prompt + response)")

    for bars, vals in ((b1, resp), (b2, pair)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9)

    for xi, (_, r, p) in zip(x, groups):
        delta = reports[p]["classification_accuracy"] - reports[r]["classification_accuracy"]
        color = "#2ca02c" if delta > 0 else "#d62728"
        ax.annotate(f"Δ={delta:+.3f}",
                    xy=(xi, max(reports[r]["classification_accuracy"],
                                reports[p]["classification_accuracy"]) + 0.06),
                    ha="center", fontsize=9, color=color, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Effect of adding the prompt into the input (pair-conditional)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_landscape(reports: dict[str, dict], out_path: Path) -> None:
    """2D scatter: x=Acc, y=Pair-AUC, size=ARI+0.3, colored by encoder family."""
    methods = [m for m in METHOD_ORDER if m in reports]
    acc  = [reports[m]["classification_accuracy"] for m in methods]
    auc  = [reports[m]["mean_pairwise_auc"]       for m in methods]
    ari  = [reports[m]["clustering_ari"]          for m in methods]
    colors = [COLOR.get(m, "#888") for m in methods]
    sizes  = [180 + 900 * a for a in ari]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(acc, auc, s=sizes, c=colors, alpha=0.75, edgecolor="black", linewidth=0.6)
    for m, a, u in zip(methods, acc, auc):
        ax.annotate(m, xy=(a, u), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Classification accuracy")
    ax.set_ylabel("Mean pairwise AUC")
    ax.set_title("Ablation landscape (marker size ∝ clustering ARI)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.55, 1.02)
    ax.set_ylim(0.92, 1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ablation figures")
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    reports = _load_reports(Path(args.reports_dir))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_bars_all_metrics(reports,      out / "ablation_all_metrics.svg")
    plot_aggregation_effect(reports,    out / "ablation_aggregation_effect.svg")
    plot_pair_vs_response(reports,      out / "ablation_pair_vs_response.svg")
    plot_2d_landscape(reports,          out / "ablation_landscape.svg")

    for name in ["ablation_all_metrics.svg", "ablation_aggregation_effect.svg",
                 "ablation_pair_vs_response.svg", "ablation_landscape.svg"]:
        print(f"[render] {name}")


if __name__ == "__main__":
    main()
