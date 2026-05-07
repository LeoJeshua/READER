"""Render bar charts comparing mean-LR baseline vs attention pooling variants.

Inputs: one or more JSON files produced by run_attention_pool.py.
Outputs: a single SVG (grouped bars per proxy×view, methods as colors).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_ORDER = ["mean_pool_lr", "single_query_attn", "multi_head_attn", "transformer_cls"]
METHOD_LABEL = {
    "mean_pool_lr": "Mean+LR (baseline)",
    "single_query_attn": "Single-Q attn",
    "multi_head_attn": "Multi-head attn",
    "transformer_cls": "Transformer+CLS",
}
METHOD_COLOR = {
    "mean_pool_lr": "#bdbdbd",
    "single_query_attn": "#5b9bd5",
    "multi_head_attn": "#ed7d31",
    "transformer_cls": "#70ad47",
}


def _load(paths: list[Path]) -> list[dict]:
    runs = []
    for p in paths:
        d = json.loads(p.read_text())
        for r in d["runs"]:
            r["_source"] = str(p)
            runs.append(r)
    return runs


def _draw(ax, runs, metric_key, title, ylabel, ymax=None):
    n_groups = len(runs)
    width = 0.2
    xs = np.arange(n_groups)
    for i, m in enumerate(METHOD_ORDER):
        ys = [r["methods"][m][metric_key] for r in runs]
        bars = ax.bar(xs + (i - 1.5) * width, ys, width=width,
                      color=METHOD_COLOR[m], label=METHOD_LABEL[m])
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.005 if ymax else 0),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['proxy']}\n{_view_short(r['feature_kind'])}" for r in runs],
                       fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _view_short(kind: str) -> str:
    if "intra" in kind:
        return "intra-M=16"
    if "all_positions" in kind:
        return "all-positions"
    return kind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", action="append", required=True, type=Path)
    ap.add_argument("--output-svg", required=True, type=Path)
    ap.add_argument("--output-md", type=Path)
    args = ap.parse_args()

    runs = _load(args.input_json)
    if not runs:
        raise SystemExit("no runs found")

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.6 * len(runs)), 4.5))
    _draw(axes[0], runs, "accuracy_mean",
          "Classification accuracy (5-fold CV)", "accuracy", ymax=1.0)
    _draw(axes[1], runs, "mean_pairwise_auc",
          "Mean pairwise AUC", "AUC", ymax=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_svg, bbox_inches="tight")
    print(f"[fig] -> {args.output_svg}")

    if args.output_md:
        lines = ["# Attention pooling vs mean-pool LR (single layer, intra & all-positions)\n",
                 "| Proxy | View | Method | Acc | Macro-F1 | Pair-AUC | mAP@10 |",
                 "|---|---|---|---|---|---|---|"]
        for r in runs:
            view = _view_short(r["feature_kind"])
            for m in METHOD_ORDER:
                d = r["methods"][m]
                lines.append(
                    f"| {r['proxy']} | {view} | {METHOD_LABEL[m]} | "
                    f"{d['accuracy_mean']:.4f}±{d['accuracy_std']:.4f} | "
                    f"{d['macro_f1_mean']:.4f} | "
                    f"{d['mean_pairwise_auc']:.4f} | "
                    f"{d['retrieval_map_at_10']:.4f} |"
                )
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text("\n".join(lines) + "\n")
        print(f"[md]  -> {args.output_md}")


if __name__ == "__main__":
    main()
