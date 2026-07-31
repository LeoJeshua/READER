"""Strictly matched temporal-representation controls for Agent500."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reader_provenance.data.records import load_records
from reader_provenance.evaluation.grouped import grouped_oof_metrics_detailed
from reader_provenance.evaluation.protocol import prompt_grouped_folds, source_groups
from reader_provenance.features.io import load_features, save_features
from reader_provenance.models.proxy import ProxyConfig, ProxyReader
from reader_provenance.training.probe import fit_source_probe

FIXED_CONFIGS = (
    "mean_pool",
    "final_token",
    "max_pool",
    "mean_final_pool",
    "mean_max_pool",
    "final_max_pool",
    "dct_q2",
    "dct_q4",
    "dct_q8",
)
LEARNED_CONFIGS = ("learned_temporal_h1", "learned_temporal_h2")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def fixed_representations(
    dct: np.ndarray,
    pooling: np.ndarray,
) -> dict[str, np.ndarray]:
    if dct.ndim != 3 or dct.shape[1] < 8:
        raise ValueError("matched DCT archive must have shape (N, >=8, D)")
    if pooling.shape != (len(dct), 2, dct.shape[2]):
        raise ValueError("matched pooling archive must have shape (N, 2, D)")
    mean = dct[:, 0]
    maximum = pooling[:, 0]
    final = pooling[:, 1]
    return {
        "mean_pool": mean,
        "final_token": final,
        "max_pool": maximum,
        "mean_final_pool": np.concatenate((mean, final), axis=1),
        "mean_max_pool": np.concatenate((mean, maximum), axis=1),
        "final_max_pool": np.concatenate((final, maximum), axis=1),
        "dct_q2": dct[:, :2].reshape(len(dct), -1),
        "dct_q4": dct[:, :4].reshape(len(dct), -1),
        "dct_q8": dct[:, :8].reshape(len(dct), -1),
    }


class TemporalMixProbe(torch.nn.Module):
    """Fold-local global mode mixture followed by a source head."""

    def __init__(
        self,
        n_modes: int,
        n_heads: int,
        hidden_size: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.mix = torch.nn.Parameter(torch.zeros(n_heads, n_modes))
        with torch.no_grad():
            for head in range(n_heads):
                self.mix[head, min(head, n_modes - 1)] = 1.0
        self.classifier = torch.nn.Linear(n_heads * hidden_size, n_classes)
        torch.nn.init.zeros_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)

    def normalized_mix(self) -> torch.Tensor:
        return self.mix / self.mix.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = torch.einsum("bqd,hq->bhd", features, self.normalized_mix())
        return self.classifier(pooled.flatten(1))


def fit_temporal_mix(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_heads: int,
    n_classes: int,
    device: str,
    learning_rate: float = 1e-3,
    max_steps: int = 40,
    schedule_horizon: int = 100,
    c_value: float = 1.0,
    seed: int = 42,
) -> tuple[TemporalMixProbe, np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=torch_device)
    y = torch.as_tensor(labels, dtype=torch.long, device=torch_device)
    mean = x.mean(dim=0, keepdim=True)
    scale = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    standardized = (x - mean) / scale
    model = TemporalMixProbe(
        n_modes=x.shape[1],
        n_heads=n_heads,
        hidden_size=x.shape[2],
        n_classes=n_classes,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=schedule_horizon,
        eta_min=learning_rate * 0.01,
    )
    l2_scale = 1.0 / (2.0 * c_value * len(features))
    objective_value = float("nan")
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        objective = torch.nn.functional.cross_entropy(model(standardized), y)
        objective = objective + l2_scale * model.classifier.weight.square().sum()
        objective.backward()
        optimizer.step()
        scheduler.step()
        objective_value = float(objective.detach())
    diagnostics = {
        "steps": max_steps,
        "final_objective": objective_value,
        "normalized_mode_weights": model.normalized_mix()
        .detach()
        .cpu()
        .numpy()
        .tolist(),
    }
    return (
        model.eval(),
        mean.cpu().numpy(),
        scale.cpu().numpy(),
        diagnostics,
    )


@torch.inference_mode()
def temporal_log_probabilities(
    features: np.ndarray,
    model: TemporalMixProbe,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    device: str,
    chunk_size: int = 4096,
) -> np.ndarray:
    output = []
    torch_device = torch.device(device)
    mean_tensor = torch.as_tensor(mean, device=torch_device)
    scale_tensor = torch.as_tensor(scale, device=torch_device)
    for start in range(0, len(features), chunk_size):
        values = torch.as_tensor(
            np.asarray(features[start : start + chunk_size], dtype=np.float32),
            device=torch_device,
        )
        logits = model((values - mean_tensor) / scale_tensor)
        output.append(torch.log_softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def evaluate(
    *,
    dct_path: Path,
    pooling_path: Path,
    output: Path,
    predictions: Path,
    device: str,
    budgets: tuple[int, ...],
    grouping_seeds: tuple[int, ...],
    n_splits: int,
    split_seed: int,
) -> dict[str, Any]:
    dct_batch = load_features(dct_path)
    pool_batch = load_features(pooling_path)
    if dct_batch.labels != pool_batch.labels:
        raise ValueError("DCT and pooling labels differ")
    if dct_batch.sample_ids != pool_batch.sample_ids:
        raise ValueError("DCT and pooling prompt IDs differ")
    dct = np.asarray(dct_batch.features, dtype=np.float32)
    pooling = np.asarray(pool_batch.features, dtype=np.float32)
    fixed = fixed_representations(dct, pooling)
    labels = dct_batch.labels
    classes = sorted(set(labels))
    class_to_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    folds = list(prompt_grouped_folds(dct_batch.sample_ids, n_splits, split_seed))
    configs = FIXED_CONFIGS + LEARNED_CONFIGS
    oof = {
        config: np.full((len(labels), len(classes)), np.nan, dtype=np.float32)
        for config in configs
    }
    fold_assignments = np.full(len(labels), -1, dtype=np.int16)
    fold_reports = []
    for fold_index, (train, test) in enumerate(folds):
        fold_assignments[test] = fold_index
        config_reports = {}
        for config, representation in fixed.items():
            probe, diagnostics = fit_source_probe(
                representation[train], y[train], classes, device=device
            )
            oof[config][test] = probe.log_probabilities(
                representation[test]
            ).astype(np.float32)
            config_reports[config] = {
                "feature_dimension": int(representation.shape[1]),
                "fit": diagnostics,
            }
        for n_heads in (1, 2):
            config = f"learned_temporal_h{n_heads}"
            model, mean, scale, diagnostics = fit_temporal_mix(
                dct[train, :8],
                y[train],
                n_heads=n_heads,
                n_classes=len(classes),
                device=device,
                seed=split_seed + fold_index * 101 + n_heads,
            )
            oof[config][test] = temporal_log_probabilities(
                dct[test, :8], model, mean, scale, device=device
            )
            config_reports[config] = {
                "feature_dimension": int(n_heads * dct.shape[2]),
                "fit": diagnostics,
            }
            del model
        fold_reports.append(
            {
                "fold": fold_index,
                "train_responses": int(len(train)),
                "test_responses": int(len(test)),
                "configs": config_reports,
            }
        )
    if np.any(fold_assignments < 0) or any(
        np.isnan(values).any() for values in oof.values()
    ):
        raise AssertionError("incomplete temporal-control OOF predictions")

    config_reports = {}
    prediction_payload = {
        "classes": np.asarray(classes, dtype=str),
        "row_labels": y,
        "row_fold_ids": fold_assignments,
    }
    canonical_seed = grouping_seeds[0]
    for config, values in oof.items():
        by_seed, summary, by_fold, fold_summary = grouped_oof_metrics_detailed(
            labels=labels,
            classes=classes,
            fold_assignments=fold_assignments,
            log_posteriors=values,
            budgets=budgets,
            grouping_seeds=grouping_seeds,
        )
        config_reports[config] = {
            "metrics_by_grouping_seed": by_seed,
            "metrics_across_grouping_seeds": summary,
            "metrics_by_fold": by_fold,
            "metrics_across_folds": fold_summary,
        }
        for budget in budgets:
            grouped_labels = []
            grouped_predictions = []
            grouped_folds = []
            for fold_index in range(n_splits):
                test = np.flatnonzero(fold_assignments == fold_index)
                groups = source_groups(
                    test,
                    labels,
                    budget,
                    canonical_seed + 100 + budget * 7919 + fold_index * 31,
                )
                for label, indices in groups:
                    score = values[indices].sum(axis=0)
                    grouped_labels.append(class_to_index[label])
                    grouped_predictions.append(int(score.argmax()))
                    grouped_folds.append(fold_index)
            prefix = f"{config}_k{budget}"
            prediction_payload[f"{prefix}_labels"] = np.asarray(grouped_labels)
            prediction_payload[f"{prefix}_predictions"] = np.asarray(
                grouped_predictions
            )
            prediction_payload[f"{prefix}_fold_ids"] = np.asarray(grouped_folds)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions, **prediction_payload)
    report = {
        "schema_version": 1,
        "protocol": "reader_matched_temporal_controls_v1",
        "dct_source": str(dct_path),
        "pooling_source": str(pooling_path),
        "predictions": str(predictions),
        "token_span": "identical response-token states for every representation",
        "split_seed": split_seed,
        "grouping_seeds": list(grouping_seeds),
        "budgets": list(budgets),
        "configs": config_reports,
        "folds": fold_reports,
        "complete": True,
    }
    _atomic_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--records", type=Path, action="append", required=True)
    extract.add_argument("--proxy", required=True)
    extract.add_argument("--layer", type=int, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--view", default="prompt_response")
    extract.add_argument("--max-length", type=int, default=1024)
    extract.add_argument("--batch-size", type=int, default=4)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--dtype", default="bfloat16")
    extract.add_argument("--attention", default="auto")
    extract.add_argument("--early-exit", action="store_true")

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--dct-features", type=Path, required=True)
    evaluation.add_argument("--pooling-features", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument(
        "--budgets", type=int, nargs="+", default=(1, 5, 10, 20, 50, 100)
    )
    evaluation.add_argument(
        "--grouping-seeds", type=int, nargs="+", default=(42, 43, 44)
    )
    evaluation.add_argument("--n-splits", type=int, default=5)
    evaluation.add_argument("--split-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        records = load_records(args.records)
        reader = ProxyReader(
            ProxyConfig(
                model_name_or_path=args.proxy,
                layer=args.layer,
                view=args.view,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                dtype=args.dtype,
                attention=args.attention,
                early_exit=args.early_exit,
            )
        )
        try:
            dct_batch, pool_batch = reader.extract_temporal_controls(records)
        finally:
            reader.close()
        save_features(args.output_dir / "dct_q8.npz", dct_batch)
        save_features(args.output_dir / "pooling.npz", pool_batch)
    else:
        evaluate(
            dct_path=args.dct_features,
            pooling_path=args.pooling_features,
            output=args.output,
            predictions=args.predictions,
            device=args.device,
            budgets=tuple(sorted(set(args.budgets))),
            grouping_seeds=tuple(dict.fromkeys(args.grouping_seeds)),
            n_splits=args.n_splits,
            split_seed=args.split_seed,
        )


if __name__ == "__main__":
    main()
