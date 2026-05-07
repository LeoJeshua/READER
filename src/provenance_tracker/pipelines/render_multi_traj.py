"""Render comparison figures + markdown table from multi_traj JSON reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "meanpool_lr": "#999999",
    "logposterior": "#5e3c99",
    "gaussian_disc": "#2c7bb6",
}
METHOD_LABEL = {
    "meanpool_lr":  "mean-pool + LR (baseline)",
    "logposterior": "log-posterior accumulation",
    "gaussian_disc": "Gaussian disc. (K·Mahalanobis)",
}
METRIC_SPECS = [
    ("accuracy", "Accuracy", (0.0, 1.0), "higher ↑"),
    ("macro_f1", "Macro-F1", (0.0, 1.0), "higher ↑"),
    ("nll", "NLL", None, "lower ↓"),
    ("ece", "ECE", None, "lower ↓"),
]


def _load_aggregators(path: Path) -> dict:
    blob = json.loads(path.read_text())
    out: dict[tuple[str, int], dict] = {}
    for row in blob["aggregators"]:
        out[(row["method"], row["K"])] = row
    return {"tag": blob["tag"], "rows": out, "sprt": blob.get("sprt")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports", nargs="+", required=True,
                   help="multi_traj JSON report paths")
    p.add_argument("--out-figdir", required=True)
    p.add_argument("--out-table", required=True)
    args = p.parse_args()

    fig_dir = Path(args.out_figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = {Path(p).stem: _load_aggregators(Path(p)) for p in args.reports}
    tags = list(data.keys())

    # --------- Figure 1: Aggregator metrics vs K per proxy ----------
    ks = sorted({k for d in data.values() for (_, k) in d["rows"]})
    methods = ["meanpool_lr", "logposterior", "gaussian_disc"]
    fig, axes = plt.subplots(len(tags), len(METRIC_SPECS),
                             figsize=(4.3 * len(METRIC_SPECS), 3.8 * len(tags)),
                             squeeze=False, sharex=True)
    for ri, tag in enumerate(tags):
        rows = data[tag]["rows"]
        for ci, (key, label, ylim, note) in enumerate(METRIC_SPECS):
            ax = axes[ri, ci]
            for m in methods:
                ys = [rows[(m, k)][key] if (m, k) in rows else np.nan for k in ks]
                ax.plot(ks, ys, marker="o", color=METHOD_COLORS[m],
                        label=METHOD_LABEL[m], linewidth=2)
                for x, y in zip(ks, ys):
                    if not np.isnan(y):
                        offset = 0.02 * (ylim[1] - ylim[0]) if ylim else 0.05
                        ax.text(x, y + offset, f"{y:.2f}",
                                ha="center", fontsize=7,
                                color=METHOD_COLORS[m], alpha=0.8)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            if ri == len(tags) - 1:
                ax.set_xlabel("Aggregation K (log scale)")
            if ci == 0:
                ax.set_ylabel(tag)
            ax.set_title(f"{label}   [{note}]", fontsize=11)
            ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Aggregator comparison: mean-pool vs log-posterior vs Gaussian-discriminant",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "multi_traj_aggregators.svg", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --------- Figure 2: SPRT curves ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"8b_L23": "#fdae61", "35_9b_L23": "#2c7bb6"}
    for tag in tags:
        s = data[tag]["sprt"]
        if s is None:
            continue
        alphas = sorted(float(a) for a in s["per_alpha"].keys())
        e_tau = [s["per_alpha"][str(a)]["E_tau"] if str(a) in s["per_alpha"]
                 else s["per_alpha"][a]["E_tau"] for a in alphas]
        dec_frac = [s["per_alpha"][str(a) if str(a) in s["per_alpha"] else a]["decided_frac"]
                    for a in alphas]
        acc_dec = [s["per_alpha"][str(a) if str(a) in s["per_alpha"] else a]["acc_when_decided"]
                   for a in alphas]
        acc_all = [s["per_alpha"][str(a) if str(a) in s["per_alpha"] else a]["overall_acc"]
                   for a in alphas]
        col = colors.get(tag, "#666666")
        axes[0].plot(alphas, e_tau, marker="o", color=col, linewidth=2, label=tag)
        axes[1].plot(alphas, dec_frac, marker="o", color=col, linewidth=2, label=tag)
        axes[2].plot(alphas, acc_dec, marker="o", color=col, linewidth=2,
                     label=f"{tag} (when decided)")
        axes[2].plot(alphas, acc_all, marker="x", color=col, linewidth=1.4,
                     linestyle=":", label=f"{tag} (overall)")
        for ax_i, ys in enumerate([e_tau, dec_frac, acc_dec]):
            for x, y in zip(alphas, ys):
                axes[ax_i].text(x, y + 0.3 if ax_i == 0 else y + 0.02,
                                f"{y:.2f}", ha="center", fontsize=8, color=col)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("target α (log)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("E[τ]  (expected stopping time)")
    axes[0].set_title("Expected trajectories to decision")
    axes[1].set_ylabel("fraction decided before K_max=32")
    axes[1].set_title("Decision coverage")
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylabel("accuracy")
    axes[2].set_title("Decision accuracy")
    axes[2].set_ylim(0, 1.05)
    fig.suptitle("Multi-hypothesis SPRT — stopping time vs error target",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "multi_traj_sprt.svg", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --------- Markdown table ----------
    lines = ["# Multi-trajectory aggregation — method comparison",
             "",
             "## Per-K metrics (proxy hidden states, layer=23)",
             ""]
    for tag in tags:
        rows = data[tag]["rows"]
        lines += [f"### {tag}",
                  "",
                  "| Method | K | N | Acc | Macro-F1 | NLL | ECE |",
                  "|---|---|---|---|---|---|---|"]
        for m in methods:
            for k in ks:
                if (m, k) not in rows:
                    continue
                r = rows[(m, k)]
                lines.append(
                    f"| {METHOD_LABEL[m]} | {k} | {r['n_fingerprints']} | "
                    f"{r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
                    f"{r['nll']:.3f} | {r['ece']:.3f} |"
                )
        lines.append("")

    lines += ["## MSPRT — expected stopping time", "",
              "| Proxy | α | E[τ] | median τ | p10-p90 τ | frac decided | acc (decided) | acc (overall) |",
              "|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        s = data[tag]["sprt"]
        if s is None: continue
        for a_key, row in s["per_alpha"].items():
            a = float(a_key)
            lines.append(
                f"| {tag} | {a:g} | {row['E_tau']:.2f} | {row['median_tau']:.1f} | "
                f"{row['p10_tau']:.1f}–{row['p90_tau']:.1f} | "
                f"{row['decided_frac']:.2f} | {row['acc_when_decided']:.3f} | "
                f"{row['overall_acc']:.3f} |"
            )
    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_table).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_table}")
    print(f"wrote {fig_dir}/multi_traj_aggregators.svg, multi_traj_sprt.svg")


if __name__ == "__main__":
    main()
