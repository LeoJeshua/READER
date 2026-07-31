"""Reconstruct Agent500 confusion matrices from out-of-fold evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix

from reader_provenance.evaluation.evidence import accumulate_log_posteriors
from reader_provenance.evaluation.protocol import source_groups


def grouped_predictions(
    archive: Path,
    *,
    budget: int,
    grouping_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(archive, allow_pickle=True) as data:
        logp = np.asarray(data["log_posteriors"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        classes = [str(value) for value in data["classes"].tolist()]
        folds = np.asarray(data["fold_assignments"], dtype=np.int64)
    if logp.shape != (len(labels), len(classes)):
        raise ValueError("OOF log-posterior shape differs from labels/classes")
    label_names = [classes[index] for index in labels]
    targets, predictions = [], []
    for fold in sorted(set(folds.tolist())):
        indices = np.flatnonzero(folds == fold)
        groups = source_groups(
            indices,
            label_names,
            budget,
            grouping_seed + 100 + budget * 7919 + fold * 31,
        )
        for label, rows in groups:
            targets.append(classes.index(label))
            predictions.append(int(accumulate_log_posteriors(logp[rows]).argmax()))
    return np.asarray(targets), np.asarray(predictions), classes


def normalized_confusion(
    archive: Path,
    *,
    budget: int,
    grouping_seed: int,
) -> tuple[np.ndarray, list[str]]:
    truth, predictions, classes = grouped_predictions(
        archive, budget=budget, grouping_seed=grouping_seed
    )
    matrix = confusion_matrix(
        truth,
        predictions,
        labels=np.arange(len(classes)),
        normalize="true",
    )
    return np.asarray(matrix, dtype=np.float64), classes


def _models(config: Path, role: str) -> list[dict[str, Any]]:
    rows = yaml.safe_load(config.read_text(encoding="utf-8"))["models"]
    return [row for row in rows if role in row.get("roles", [])]


def render(
    archives: dict[str, Path],
    config: Path,
    output: Path,
    *,
    role: str,
    layout: str,
    budget: int,
    grouping_seed: int,
) -> Path:
    models = _models(config, role)
    if set(archives) != {str(row["tag"]) for row in models}:
        raise ValueError("OOF archives must exactly cover the selected proxy role")
    if layout == "row":
        rows, columns = (1, len(models)) if len(models) == 4 else (2, 4)
    elif layout == "grid":
        rows, columns = (2, 2) if len(models) == 4 else (2, 4)
    else:
        raise ValueError("layout must be row or grid")
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.15 * columns, 2.0 * rows),
        squeeze=False,
    )
    image = None
    for axis, model in zip(axes.reshape(-1), models, strict=True):
        matrix, classes = normalized_confusion(
            archives[str(model["tag"])],
            budget=budget,
            grouping_seed=grouping_seed,
        )
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues", interpolation="none")
        axis.set_title(str(model["name"]), fontsize=8)
        axis.set_xlabel("Predicted source", fontsize=7)
        axis.set_ylabel("True source", fontsize=7)
        if len(classes) <= 20:
            axis.set_xticks(np.arange(len(classes)), classes, rotation=90, fontsize=4)
            axis.set_yticks(np.arange(len(classes)), classes, fontsize=4)
        else:
            axis.set_xticks([])
            axis.set_yticks([])
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015)
    figure.suptitle(f"Agent500 normalized confusion at K={budget}", fontsize=9)
    figure.subplots_adjust(
        left=0.06, right=0.93, bottom=0.08, top=0.89, wspace=0.25, hspace=0.33
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return output


def _named_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected TAG=PATH") from error
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=_named_path, action="append", required=True)
    parser.add_argument(
        "--proxy-config", type=Path, default=Path("configs/proxies.yaml")
    )
    parser.add_argument("--role", choices=("main", "full"), default="main")
    parser.add_argument("--layout", choices=("row", "grid"), default="grid")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--grouping-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    render(
        dict(args.oof),
        args.proxy_config,
        args.output,
        role=args.role,
        layout=args.layout,
        budget=args.budget,
        grouping_seed=args.grouping_seed,
    )


if __name__ == "__main__":
    main()
