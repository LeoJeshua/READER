"""Render figures + markdown table from multi_traj_calib JSON reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOURCE_COLORS = {
    "vanilla_lr": "#999999",
    "calibrated_lr": "#2c7bb6",
    "gaussian_disc": "#d7191c",
}
SOURCE_LABEL = {
    "vanilla_lr": "vanilla LR",
    "calibrated_lr": "temperature-scaled LR",
    "gaussian_disc": "Gaussian disc. (LDA + LW)",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _fmt_alpha_key(d: dict, a: float):
    if str(a) in d:
        return str(a)
    return a if a in d else f"{a}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--out-figdir", required=True)
    p.add_argument("--out-table", required=True)
    args = p.parse_args()

    fig_dir = Path(args.out_figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = {Path(p).stem.replace("_calib", ""): _load(Path(p)) for p in args.reports}
    tags = list(data.keys())
    sources = list(SOURCE_LABEL.keys())
    alphas = sorted(float(a) for a in data[tags[0]]["alphas"])

    # ---------- Figure 1: per-sample calibration (acc / NLL / ECE) ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metric_keys = [("accuracy", "Per-sample accuracy", (0.0, 0.5), "higher ↑"),
                   ("nll", "Per-sample NLL", None, "lower ↓"),
                   ("ece", "Per-sample ECE", (0.0, 0.8), "lower ↓")]
    bar_w = 0.25
    xs = np.arange(len(tags))
    for ci, (k, label, ylim, note) in enumerate(metric_keys):
        ax = axes[ci]
        for si, src in enumerate(sources):
            vals = [data[t]["sources"][src]["calibration"][k] for t in tags]
            offset = (si - 1) * bar_w
            bars = ax.bar(xs + offset, vals, bar_w,
                          color=SOURCE_COLORS[src], label=SOURCE_LABEL[src])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                        color=SOURCE_COLORS[src])
        ax.set_xticks(xs)
        ax.set_xticklabels(tags)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(f"{label}   [{note}]", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Per-sample log-posterior calibration: vanilla LR vs "
                 "temperature-scaled LR vs Gaussian discriminant",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "multi_traj_calibration.svg", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: SPRT under each source ----------
    fig, axes = plt.subplots(len(tags), 3, figsize=(15, 4.4 * len(tags)),
                             squeeze=False)
    for ri, tag in enumerate(tags):
        for src in sources:
            sprt = data[tag]["sources"][src]["sprt"]
            e_tau = []
            dec = []
            acc = []
            for a in alphas:
                key = _fmt_alpha_key(sprt["per_alpha"], a)
                row = sprt["per_alpha"][key]
                e_tau.append(row["E_tau"])
                dec.append(row["decided_frac"])
                acc.append(row["overall_acc"])
            col = SOURCE_COLORS[src]
            axes[ri, 0].plot(alphas, e_tau, marker="o", color=col,
                             linewidth=2, label=SOURCE_LABEL[src])
            axes[ri, 1].plot(alphas, dec, marker="o", color=col,
                             linewidth=2, label=SOURCE_LABEL[src])
            axes[ri, 2].plot(alphas, acc, marker="o", color=col,
                             linewidth=2, label=SOURCE_LABEL[src])
            for x, y in zip(alphas, e_tau):
                axes[ri, 0].text(x, y + 0.4, f"{y:.2f}", fontsize=7,
                                 ha="center", color=col)
            for x, y in zip(alphas, acc):
                axes[ri, 2].text(x, y + 0.02, f"{y:.2f}", fontsize=7,
                                 ha="center", color=col)
        # 1−α reference on accuracy panel
        target = [1 - a for a in alphas]
        axes[ri, 2].plot(alphas, target, color="black", linestyle="--",
                         linewidth=1.2, label="target 1−α")

        for ci, ax in enumerate(axes[ri]):
            ax.set_xscale("log")
            ax.set_xlabel("target α (log)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[ri, 0].set_ylabel(f"{tag}\nE[τ]")
        axes[ri, 1].set_ylabel("fraction decided")
        axes[ri, 1].set_ylim(0, 1.05)
        axes[ri, 2].set_ylabel("decision accuracy")
        axes[ri, 2].set_ylim(0, 1.05)
        if ri == 0:
            axes[ri, 0].set_title("Expected stopping time")
            axes[ri, 1].set_title("Decision coverage")
            axes[ri, 2].set_title("Decision accuracy vs target 1−α")
    fig.suptitle("Calibrated MSPRT — vanilla LR vs temperature-scaled LR vs "
                 "Gaussian discriminant", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "multi_traj_sprt_calibrated.svg", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---------- Markdown table ----------
    lines = ["# Calibrated MSPRT — per-sample calibration + stopping-time",
             "",
             "## Per-sample calibration (K=1 OOF log-posteriors)",
             "",
             "| Proxy | Source | Acc | NLL | ECE | T (mean) |",
             "|---|---|---|---|---|---|"]
    for tag in tags:
        for src in sources:
            cal = data[tag]["sources"][src]["calibration"]
            T = data[tag]["sources"][src].get("T_per_fold")
            T_str = f"{np.mean([float(t) for t in T]):.2f}" if T else "—"
            lines.append(
                f"| {tag} | {SOURCE_LABEL[src]} | "
                f"{cal['accuracy']:.3f} | {cal['nll']:.3f} | "
                f"{cal['ece']:.3f} | {T_str} |"
            )
    lines += ["",
              "## MSPRT — expected stopping time and decision accuracy",
              "",
              "| Proxy | Source | α | E[τ] | median τ | p10–p90 τ | "
              "frac decided | acc (decided) | acc (overall) | target 1−α |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        for src in sources:
            sprt = data[tag]["sources"][src]["sprt"]
            for a in alphas:
                key = _fmt_alpha_key(sprt["per_alpha"], a)
                row = sprt["per_alpha"][key]
                lines.append(
                    f"| {tag} | {SOURCE_LABEL[src]} | {a:g} | "
                    f"{row['E_tau']:.2f} | {row['median_tau']:.1f} | "
                    f"{row['p10_tau']:.1f}–{row['p90_tau']:.1f} | "
                    f"{row['decided_frac']:.2f} | "
                    f"{row['acc_when_decided']:.3f} | "
                    f"{row['overall_acc']:.3f} | {1-a:g} |"
                )
        lines.append("")

    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_table).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_table}")
    print(f"wrote {fig_dir}/multi_traj_calibration.svg, "
          f"multi_traj_sprt_calibrated.svg")


if __name__ == "__main__":
    main()
