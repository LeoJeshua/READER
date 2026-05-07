"""Render figures + comparison tables from evaluation + circuit JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from provenance_tracker.utils.io import load_feature_batch


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


METRIC_COLS = (
    ("classification_accuracy", "Acc"),
    ("classification_macro_f1", "Macro-F1"),
    ("mean_pairwise_auc", "Pair-AUC"),
    ("retrieval_map_at_5", "mAP@5"),
    ("retrieval_map_at_10", "mAP@10"),
    ("clustering_ari", "ARI"),
    ("clustering_nmi", "NMI"),
)


def render_comparison_table(reports: list[dict], csv_path: Path, md_path: Path) -> None:
    cols = ["method", "n_samples", "n_classes", *(c for c, _ in METRIC_COLS)]
    csv_lines = [",".join(cols)]
    md_lines = ["| " + " | ".join(["Method", "N", "K", *(label for _, label in METRIC_COLS)]) + " |"]
    md_lines.append("|" + "|".join(["---"] * (3 + len(METRIC_COLS))) + "|")
    for r in reports:
        csv_lines.append(
            ",".join(
                [
                    str(r["method"]),
                    str(r["n_samples"]),
                    str(r["n_classes"]),
                    *(f"{r[c]:.4f}" for c, _ in METRIC_COLS),
                ]
            )
        )
        md_lines.append(
            "| " + " | ".join(
                [
                    str(r["method"]),
                    str(r["n_samples"]),
                    str(r["n_classes"]),
                    *(f"{r[c]:.3f}" for c, _ in METRIC_COLS),
                ]
            ) + " |"
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def plot_layer_curve(circuit_json: dict, out_path: Path) -> None:
    probes = circuit_json["layer_probes"]
    xs = [p["layer_index"] for p in probes]
    ys_acc = [p["accuracy"] for p in probes]
    ys_f1 = [p["macro_f1"] for p in probes]
    best = circuit_json["best_layer"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys_acc, marker="o", label="Accuracy")
    ax.plot(xs, ys_f1, marker="s", label="Macro-F1", alpha=0.7)
    ax.axvline(best, color="red", linestyle="--", alpha=0.6, label=f"best L={best}")
    ax.set_xlabel("Proxy layer index (0 = embedding)")
    ax.set_ylabel("5-fold CV score")
    ax.set_title("Proxy-LLM layerwise provenance probe")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_dim_heatmap(circuit_json: dict, out_path: Path) -> None:
    matrix = np.asarray(circuit_json["activation_matrix"])
    classes = circuit_json["class_order"]
    dims = circuit_json["top_dims"]

    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(dims)), 0.6 * len(classes) + 2))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm",
                   vmin=-max(abs(matrix.min()), abs(matrix.max())),
                   vmax=max(abs(matrix.min()), abs(matrix.max())))
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims], rotation=90, fontsize=6)
    ax.set_xlabel("Top discriminative hidden-unit index")
    ax.set_title(f"Class-conditioned activations (z-score) @ layer {circuit_json['best_layer']}")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison_bars(reports: list[dict], out_path: Path) -> None:
    methods = [r["method"] for r in reports]
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(methods)), 5))
    width = 0.12
    x = np.arange(len(methods))
    for i, (key, label) in enumerate(METRIC_COLS):
        vals = [r[key] for r in reports]
        ax.bar(x + (i - len(METRIC_COLS) / 2) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Provenance identification — method comparison")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(
    entries: list[tuple[str, np.ndarray, list[str]]],
    out_path: Path,
    random_state: int = 42,
) -> None:
    n_methods = len(entries)
    cols = min(3, n_methods)
    rows = int(np.ceil(n_methods / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    for ax, (name, features, labels) in zip(axes.flat, entries):
        x = StandardScaler().fit_transform(features)
        perp = min(30, max(5, len(labels) // 5))
        emb = TSNE(
            n_components=2, random_state=random_state, perplexity=perp, init="pca", learning_rate="auto"
        ).fit_transform(x)
        classes = sorted(set(labels))
        cmap = plt.get_cmap("tab20" if len(classes) > 10 else "tab10")
        for i, cls in enumerate(classes):
            mask = [lbl == cls for lbl in labels]
            ax.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.75,
                       color=cmap(i % cmap.N), label=cls)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(entries):]:
        ax.axis("off")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center",
               ncol=min(6, len(legend_labels)), bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_auc_matrix(
    name: str, features: np.ndarray, labels: list[str], out_path: Path
) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    classes = sorted(set(labels))
    y = np.asarray(labels)
    matrix = np.full((len(classes), len(classes)), np.nan)
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            if i == j:
                matrix[i, j] = 1.0
                continue
            if j < i:
                continue
            mask = (y == a) | (y == b)
            x_pair = features[mask]
            y_pair = (y[mask] == a).astype(int)
            try:
                skf = StratifiedKFold(
                    n_splits=min(5, y_pair.sum(), len(y_pair) - y_pair.sum()),
                    shuffle=True,
                    random_state=0,
                )
                aucs: list[float] = []
                for tr, te in skf.split(x_pair, y_pair):
                    scaler = StandardScaler().fit(x_pair[tr])
                    clf = LogisticRegression(max_iter=3000, solver="lbfgs")
                    clf.fit(scaler.transform(x_pair[tr]), y_pair[tr])
                    prob = clf.predict_proba(scaler.transform(x_pair[te]))[:, 1]
                    if len(set(y_pair[te])) < 2:
                        continue
                    aucs.append(roc_auc_score(y_pair[te], prob))
                val = float(np.mean(aucs)) if aucs else 0.5
            except ValueError:
                val = 0.5
            matrix[i, j] = val
            matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(1.0 * len(classes) + 2, 1.0 * len(classes) + 1))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] < 0.75 else "black", fontsize=7)
    ax.set_title(f"Pairwise AUC — {name}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_view_map(raw: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in raw or []:
        if ":" not in entry:
            raise ValueError(f"--feature-view expected 'method_name:path:view' got {entry}")
        name, rest = entry.split(":", 1)
        result[name] = rest
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Render figures + comparison table")
    parser.add_argument("--evaluation-json", action="append", required=True,
                        help="Path to a report produced by pipelines.evaluate")
    parser.add_argument("--circuit-json", required=True)
    parser.add_argument("--feature-view", action="append", default=[],
                        help="method_name:path.npz:view (for t-SNE/pairwise figures)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    reports = [_load_json(p) for p in args.evaluation_json]
    render_comparison_table(
        reports,
        tables_dir / "method_comparison.csv",
        tables_dir / "method_comparison.md",
    )

    circuit = _load_json(args.circuit_json)
    plot_layer_curve(circuit, figures_dir / "layer_accuracy_curve.svg")
    plot_top_dim_heatmap(circuit, figures_dir / "top_dim_heatmap.svg")
    plot_comparison_bars(reports, figures_dir / "method_comparison_bars.svg")

    view_map = _parse_view_map(args.feature_view)
    if view_map:
        from provenance_tracker.pipelines.evaluate import _resolve_view  # noqa: E402

        tsne_entries: list[tuple[str, np.ndarray, list[str]]] = []
        for name, spec in view_map.items():
            path, view = spec.rsplit(":", 1)
            batch = load_feature_batch(path)
            x = _resolve_view(batch.features, batch.labels, view)
            tsne_entries.append((name, x, batch.labels))
            plot_pairwise_auc_matrix(
                name, x, batch.labels, figures_dir / f"pairwise_auc_{name}.svg"
            )
        plot_tsne(tsne_entries, figures_dir / "tsne_methods.svg")

    print(f"[report] wrote figures to {figures_dir} and tables to {tables_dir}")


if __name__ == "__main__":
    main()
