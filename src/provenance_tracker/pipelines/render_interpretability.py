"""Render figures for activation patching / attribution patching / SAE outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_activation_patching(report: dict, out_path: Path) -> None:
    xs = report["layer_indices"]
    imp = report["importance"]
    clean_tgt = report["clean_target_prob"]
    clean_src = report["clean_source_prob"]
    best = report["best_layer"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, imp, marker="o", color="#2c7bb6", label="patched P(source label)")
    ax.axhline(clean_tgt, color="red", linestyle="--", alpha=0.7,
               label=f"clean target P={clean_tgt:.3f}")
    ax.axhline(clean_src, color="green", linestyle=":", alpha=0.7,
               label=f"clean source P={clean_src:.3f}")
    ax.axvline(best, color="gray", linestyle="-.", alpha=0.5,
               label=f"probe readout L={best}")
    ax.set_xlabel("Patch layer (block index, 0 = first transformer block)")
    ax.set_ylabel(f"Recovered P(source label)  |  {report['num_pairs']} pairs")
    ax.set_title("Activation patching — causal localization of authorship signal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attribution_heatmap(report: dict, heatmap: np.ndarray, out_path: Path) -> None:
    per_layer = np.asarray(report["per_layer_total"])
    best = report["best_layer"]

    fig, (ax_h, ax_b) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    vmax = np.nanmax(np.abs(heatmap))
    im = ax_h.imshow(
        heatmap, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower"
    )
    ax_h.set_xlabel(f"Position index (last {heatmap.shape[1]} tokens)")
    ax_h.set_ylabel("Layer index (0 = embedding)")
    ax_h.set_title(
        f"Attribution heatmap  (a · ∂logP/∂a) — probe at layer {best}, {report['num_samples']} samples"
    )
    fig.colorbar(im, ax=ax_h, shrink=0.8)

    ax_b.barh(np.arange(len(per_layer)), per_layer, color="#7570b3")
    ax_b.set_ylim(-0.5, len(per_layer) - 0.5)
    ax_b.axhline(best, color="red", linestyle="--", alpha=0.5)
    ax_b.set_xlabel("Σ per-layer attribution")
    ax_b.set_ylabel("Layer")
    ax_b.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sae(
    report: dict,
    class_means: np.ndarray,
    out_path: Path,
    top_k: int = 48,
) -> None:
    classes = report["classes"]
    top_idx = np.asarray(report["top_feature_idx"])[:top_k]
    sub = class_means[:, top_idx]  # (C, top_k)
    # Z-score per feature so heatmap highlights differential activation.
    mean = sub.mean(axis=0, keepdims=True)
    std = sub.std(axis=0, keepdims=True) + 1e-6
    sub_z = (sub - mean) / std

    fig, ax = plt.subplots(figsize=(max(10, 0.25 * top_k), 0.6 * len(classes) + 2))
    vmax = np.nanmax(np.abs(sub_z))
    im = ax.imshow(sub_z, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels([str(i) for i in top_idx], rotation=90, fontsize=6)
    ax.set_xlabel(f"SAE feature index (top {top_k} by cross-class variance)")
    ax.set_title(
        f"SAE feature × class z-score  |  d_hidden={report['d_hidden']}, "
        f"sparsity={report['sparsity']:.3f}, recon={report['recon_loss']:.4f}"
    )
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render interpretability figures")
    parser.add_argument("--activation-json", default=None)
    parser.add_argument("--attribution-json", default=None)
    parser.add_argument("--attribution-heatmap", default=None,
                        help=".npy from run_attribution_patching")
    parser.add_argument("--sae-json", default=None)
    parser.add_argument("--sae-means", default=None,
                        help=".npy feature-class-mean matrix from train_sae")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-sae-features", type=int, default=48)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.activation_json:
        plot_activation_patching(
            _load_json(args.activation_json),
            out / "activation_patching.svg",
        )
        print(f"[render] activation_patching.svg")

    if args.attribution_json and args.attribution_heatmap:
        heatmap = np.load(args.attribution_heatmap)
        plot_attribution_heatmap(
            _load_json(args.attribution_json),
            heatmap,
            out / "attribution_patching.svg",
        )
        print(f"[render] attribution_patching.svg")

    if args.sae_json and args.sae_means:
        means = np.load(args.sae_means)
        plot_sae(
            _load_json(args.sae_json),
            means,
            out / "sae_features.svg",
            top_k=args.top_sae_features,
        )
        print(f"[render] sae_features.svg")


if __name__ == "__main__":
    main()
