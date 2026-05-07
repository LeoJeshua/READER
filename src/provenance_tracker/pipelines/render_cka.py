"""Render figures for the CKA-fingerprint sweep.

Three panels:
  1. Phase A: layer × K accuracy (M=1, no whitening) — picks the best layer.
  2. Phase B: M × K heatmap at the best layer — shows how aggregation helps.
  3. Whitening ablation: top configs with W=0 vs W=32 — does PC removal help?

Inputs: phaseA_<proxy>.json and one or more phaseB_<proxy>_L<L>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> list[dict]:
    return json.load(path.open())


def _filter(rows: list[dict], **eq) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def plot_layer_curves(phaseA: list[dict], out: Path) -> tuple[int, float]:
    """Per-K accuracy curve over layers at M=1, W=0. Returns (best_layer, acc)."""
    rows = _filter(phaseA, intra_m=1, whiten_top_k=0)
    layers = sorted({r["layer"] for r in rows})
    ks = sorted({r["cross_k"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    cmap = plt.cm.viridis
    for i, k in enumerate(ks):
        ys = []
        for L in layers:
            hits = _filter(rows, layer=L, cross_k=k)
            ys.append(hits[0]["classification_accuracy"] if hits else np.nan)
        ax.plot(layers, ys, marker="o", ms=3, lw=1.4,
                color=cmap(i / max(1, len(ks) - 1)), label=f"K={k}")
    # Mark best
    best = max(rows, key=lambda r: r["classification_accuracy"])
    ax.axvline(best["layer"], ls="--", color="crimson", alpha=0.5,
               label=f"best L={best['layer']} (K={best['cross_k']})")
    ax.set_xlabel("proxy layer")
    ax.set_ylabel("classification accuracy")
    ax.set_title("Phase A — CKA accuracy vs layer (M=1, W=0)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return best["layer"], best["classification_accuracy"]


def plot_mk_heatmap(phaseB: list[dict], layer: int, out: Path,
                    metric: str = "classification_accuracy",
                    whiten: int = 0) -> None:
    rows = _filter(phaseB, whiten_top_k=whiten)
    if not rows:
        return
    ms = sorted({r["intra_m"] for r in rows})
    ks = sorted({r["cross_k"] for r in rows})
    Z = np.full((len(ms), len(ks)), np.nan)
    for i, m in enumerate(ms):
        for j, k in enumerate(ks):
            hits = _filter(rows, intra_m=m, cross_k=k)
            if hits:
                Z[i, j] = hits[0][metric]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    im = ax.imshow(Z, aspect="auto", origin="lower",
                   vmin=0, vmax=1, cmap="viridis")
    for i in range(len(ms)):
        for j in range(len(ks)):
            if not np.isnan(Z[i, j]):
                color = "white" if Z[i, j] < 0.55 else "black"
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ms)
    ax.set_xlabel("cross-K (responses per fingerprint)")
    ax.set_ylabel("intra-M (tokens per response)")
    ax.set_title(f"Phase B — accuracy at L={layer} (W={whiten})")
    fig.colorbar(im, ax=ax, fraction=0.04, label="accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_whiten_ablation(phaseB: list[dict], layer: int, out: Path) -> None:
    """Compare W=0 vs W=32 across (M, K) cells. Lines = M, x = K."""
    ms = sorted({r["intra_m"] for r in phaseB})
    ks = sorted({r["cross_k"] for r in phaseB})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    cmap = plt.cm.plasma
    for ax, w in zip(axes, [0, 32]):
        rows = _filter(phaseB, whiten_top_k=w)
        for i, m in enumerate(ms):
            ys = []
            for k in ks:
                hits = _filter(rows, intra_m=m, cross_k=k)
                ys.append(hits[0]["classification_accuracy"] if hits else np.nan)
            ax.plot(ks, ys, marker="o", ms=4, lw=1.5,
                    color=cmap(i / max(1, len(ms) - 1)), label=f"M={m}")
        ax.set_xscale("log")
        ax.set_xlabel("cross-K")
        ax.set_title(f"L={layer}, W={w}")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("CKA whitening ablation: PC-removal (W=32) vs none (W=0)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_layer_compare(phaseB_by_layer: dict[int, list[dict]], out: Path,
                       metric: str = "classification_accuracy") -> None:
    """Side-by-side: best layer candidate(s) at M=16 across K."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cmap = plt.cm.tab10
    for i, (L, rows) in enumerate(sorted(phaseB_by_layer.items())):
        sub = _filter(rows, intra_m=16, whiten_top_k=0)
        sub.sort(key=lambda r: r["cross_k"])
        ks = [r["cross_k"] for r in sub]
        ys = [r[metric] for r in sub]
        ax.plot(ks, ys, marker="o", ms=5, lw=1.6,
                color=cmap(i), label=f"L={L}")
    ax.set_xscale("log")
    ax.set_xlabel("cross-K")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase B — best-layer comparison (M=16, W=0)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phaseA-json", type=Path, required=True)
    p.add_argument("--phaseB-json", type=Path, action="append", required=True,
                   help="phaseB_<proxy>_L<layer>.json — pass once per layer")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--primary-layer", type=int, default=None,
                   help="layer to use for the M×K heatmap; defaults to the layer with highest M=16,K=max accuracy")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phaseA = _load(args.phaseA_json)

    phaseB_by_layer: dict[int, list[dict]] = {}
    for path in args.phaseB_json:
        rows = _load(path)
        if not rows:
            continue
        L = rows[0]["layer"]
        phaseB_by_layer[L] = rows

    # 1. layer curves (phase A)
    bestA_layer, bestA_acc = plot_layer_curves(
        phaseA, args.output_dir / "cka_phaseA_layer_curves.svg"
    )

    # Pick primary layer for the heatmap: highest M=16,K=max acc across loaded layers
    if args.primary_layer is not None and args.primary_layer in phaseB_by_layer:
        primary = args.primary_layer
    else:
        def _peak(rows: list[dict]) -> float:
            sub = _filter(rows, intra_m=16, whiten_top_k=0)
            return max((r["classification_accuracy"] for r in sub), default=0.0)
        primary = max(phaseB_by_layer, key=lambda L: _peak(phaseB_by_layer[L]))

    # 2. M×K heatmap (W=0) at primary layer
    plot_mk_heatmap(
        phaseB_by_layer[primary], primary,
        args.output_dir / f"cka_phaseB_L{primary}_mk_heatmap.svg",
        whiten=0,
    )

    # 3. Whitening ablation at primary layer
    plot_whiten_ablation(
        phaseB_by_layer[primary], primary,
        args.output_dir / f"cka_phaseB_L{primary}_whiten_ablation.svg",
    )

    # 4. Layer comparison if multiple phaseB layers were given
    if len(phaseB_by_layer) > 1:
        plot_layer_compare(
            phaseB_by_layer,
            args.output_dir / "cka_phaseB_layer_compare.svg",
        )

    # Summary text
    summary = {
        "phaseA_best_M1": {"layer": bestA_layer, "accuracy": bestA_acc},
        "phaseB_layers": {},
    }
    for L, rows in phaseB_by_layer.items():
        best = max(rows, key=lambda r: r["classification_accuracy"])
        summary["phaseB_layers"][L] = {
            "best_intra_m": best["intra_m"],
            "best_cross_k": best["cross_k"],
            "best_whiten_top_k": best["whiten_top_k"],
            "best_accuracy": best["classification_accuracy"],
            "best_macro_f1": best["classification_macro_f1"],
            "best_pairwise_auc": best["mean_pairwise_auc"],
        }
    summary["primary_layer"] = primary
    (args.output_dir / "cka_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
