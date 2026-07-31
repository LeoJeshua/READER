"""Dynamic LLMmap adaptation used in the Agent500 experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reader_provenance.baselines.llmmap import (
    E5Embedder,
    EmbeddingPanel,
    LLMmapClosedClassifier,
    PromptResponseEmbeddingDataset,
    seed_everything,
)
from reader_provenance.data.records import iter_records
from reader_provenance.data.release import DatasetRelease
from reader_provenance.evaluation.grouped import (
    grouped_fold_model_metrics,
    grouped_oof_metrics_detailed,
    grouped_panel_metrics,
)
from reader_provenance.evaluation.protocol import prompt_grouped_folds

DEFAULT_BUDGETS = (1, 5, 10, 20, 50, 100)
DEFAULT_GROUPING_SEEDS = (42, 43, 44)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_aligned_panel(
    release: DatasetRelease,
    variant: str,
    benchmark: str,
) -> tuple[list[str], list[str], list[str], list[list[str]]]:
    classes = release.labels(variant)
    paths = release.response_paths(variant, benchmark)
    reference_ids: list[str] | None = None
    reference_prompts: list[str] | None = None
    responses = []
    for label, path in zip(classes, paths, strict=True):
        rows = list(iter_records(path))
        if {row.label for row in rows} != {label}:
            raise ValueError(f"unexpected labels in {path}")
        sample_ids = [row.sample_id for row in rows]
        prompts = [row.prompt for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"duplicate prompt IDs in {path}")
        if reference_ids is None:
            reference_ids = sample_ids
            reference_prompts = prompts
        elif sample_ids != reference_ids or prompts != reference_prompts:
            raise ValueError(f"response panel is not aligned at {path}")
        responses.append([row.response for row in rows])
    if reference_ids is None or reference_prompts is None:
        raise ValueError("empty response panel")
    return classes, reference_ids, reference_prompts, responses


def cache_embeddings(
    *,
    data_root: Path,
    variant: str,
    benchmark: str,
    output_dir: Path,
    model_name_or_path: str,
    device: str,
    batch_size: int,
    max_length: int,
    max_response_chars: int | None,
    max_response_tokens: int | None,
    local_files_only: bool,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        EmbeddingPanel(output_dir)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    release = DatasetRelease(data_root)
    classes, prompt_ids, prompts, responses = _load_aligned_panel(
        release, variant, benchmark
    )
    embedder = E5Embedder(
        model_name_or_path,
        device=device,
        max_length=max_length,
        local_files_only=local_files_only,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "schema_version": 1,
        "protocol": "llmmap_e5_panel_v1",
        "benchmark": benchmark,
        "variant": variant,
        "classes": classes,
        "prompt_ids": prompt_ids,
        "embedding_dim": embedder.embedding_dim,
        "max_length": max_length,
        "max_response_chars": max_response_chars,
        "max_response_tokens": max_response_tokens,
        "embedding_model": model_name_or_path,
    }
    progress_path = output_dir / "progress.json"
    progress = {**expected, "query_complete": False, "completed_classes": 0}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if progress.get(key) != value:
                raise ValueError(f"cache configuration changed for {key}")

    query_path = output_dir / "query_embeddings.npy"
    if query_path.is_file():
        query_array = np.lib.format.open_memmap(query_path, mode="r+")
    else:
        query_array = np.lib.format.open_memmap(
            query_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(prompt_ids), embedder.embedding_dim),
        )
    if query_array.shape != (len(prompt_ids), embedder.embedding_dim):
        raise ValueError("existing query cache has the wrong shape")
    if not progress["query_complete"]:
        query_array[:] = embedder.encode(prompts, batch_size=batch_size).astype(
            np.float16
        )
        query_array.flush()
        progress["query_complete"] = True
        _atomic_json(progress_path, progress)

    response_path = output_dir / "response_embeddings.npy"
    expected_shape = (len(classes), len(prompt_ids), embedder.embedding_dim)
    if response_path.is_file():
        response_array = np.lib.format.open_memmap(response_path, mode="r+")
    else:
        response_array = np.lib.format.open_memmap(
            response_path,
            mode="w+",
            dtype=np.float16,
            shape=expected_shape,
        )
    if response_array.shape != expected_shape:
        raise ValueError("existing response cache has the wrong shape")
    for class_index in range(int(progress["completed_classes"]), len(classes)):
        texts = responses[class_index]
        if max_response_tokens is not None:
            texts = embedder.truncate_by_tokens(texts, max_response_tokens)
        elif max_response_chars is not None:
            texts = [text[:max_response_chars] for text in texts]
        response_array[class_index] = embedder.encode(
            texts, batch_size=batch_size
        ).astype(np.float16)
        response_array.flush()
        progress["completed_classes"] = class_index + 1
        _atomic_json(progress_path, progress)
        print(
            f"embedded {class_index + 1}/{len(classes)} {classes[class_index]}",
            flush=True,
        )
    manifest = {**expected, "complete": True}
    _atomic_json(manifest_path, manifest)
    progress_path.unlink(missing_ok=True)
    return manifest


def _loader(
    panel: EmbeddingPanel,
    positions: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        PromptResponseEmbeddingDataset(panel, positions),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _new_model(panel: EmbeddingPanel, device: torch.device) -> LLMmapClosedClassifier:
    return LLMmapClosedClassifier(
        embedding_dim=panel.metadata.embedding_dim,
        num_classes=len(panel.metadata.classes),
        feature_size=384,
        num_blocks=3,
        num_heads=4,
    ).to(device)


def _train_epoch(model, optimizer, loader, device: torch.device) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for traces, labels in loader:
        traces = traces.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(traces)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        size = labels.numel()
        total_loss += float(loss.detach()) * size
        total_correct += int((logits.argmax(1) == labels).sum())
        total += size
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "examples": total,
    }


@torch.inference_mode()
def _predict(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    values = []
    for traces, _labels in loader:
        logits = model(traces.to(device, non_blocking=True))
        values.append(torch.log_softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def _flat_indices(
    prompt_positions: np.ndarray,
    n_classes: int,
    n_prompts: int,
) -> np.ndarray:
    return np.concatenate(
        [class_index * n_prompts + prompt_positions for class_index in range(n_classes)]
    )


def train_and_evaluate(
    *,
    cache_dir: Path,
    output_dir: Path,
    device: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    num_workers: int,
    n_splits: int,
    split_seed: int,
    train_seed: int,
    budgets: tuple[int, ...],
    grouping_seeds: tuple[int, ...],
) -> dict[str, Any]:
    panel = EmbeddingPanel(cache_dir)
    n_classes = len(panel.metadata.classes)
    n_prompts = len(panel.metadata.prompt_ids)
    labels = [label for label in panel.metadata.classes for _ in range(n_prompts)]
    sample_ids = panel.metadata.prompt_ids * n_classes
    y = np.repeat(np.arange(n_classes, dtype=np.int64), n_prompts)
    oof_logp = np.full((len(labels), n_classes), np.nan, dtype=np.float32)
    fold_assignments = np.full(len(labels), -1, dtype=np.int16)
    fold_reports = []
    torch_device = torch.device(device)
    folds = list(prompt_grouped_folds(panel.metadata.prompt_ids, n_splits, split_seed))
    for fold_index, (train_positions, test_positions) in enumerate(folds):
        fold_seed = train_seed + fold_index * 100003
        seed_everything(fold_seed)
        model = _new_model(panel, torch_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history = []
        for epoch in range(1, epochs + 1):
            train_loader = _loader(
                panel,
                train_positions,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                seed=fold_seed + epoch,
            )
            metrics = _train_epoch(model, optimizer, train_loader, torch_device)
            history.append({"epoch": epoch, **metrics})
        test_loader = _loader(
            panel,
            test_positions,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=fold_seed,
        )
        flat_test = _flat_indices(test_positions, n_classes, n_prompts)
        oof_logp[flat_test] = _predict(model, test_loader, torch_device)
        fold_assignments[flat_test] = fold_index
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "protocol": "llmmap_q1_dynamic_v1",
                "fold": fold_index,
                "classes": panel.metadata.classes,
                "embedding_dim": panel.metadata.embedding_dim,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_dir / f"fold-{fold_index}.pt",
        )
        fold_reports.append(
            {
                "fold": fold_index,
                "train_prompts": int(len(train_positions)),
                "test_prompts": int(len(test_positions)),
                "history": history,
            }
        )
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if np.isnan(oof_logp).any() or np.any(fold_assignments < 0):
        raise AssertionError("incomplete LLMmap out-of-fold predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "oof_log_posteriors.npz",
        log_posteriors=oof_logp,
        labels=y,
        classes=np.asarray(panel.metadata.classes, dtype=object),
        sample_ids=np.asarray(sample_ids, dtype=object),
        fold_assignments=fold_assignments,
    )
    by_seed, summary, by_fold, fold_summary = grouped_oof_metrics_detailed(
        labels=labels,
        classes=panel.metadata.classes,
        fold_assignments=fold_assignments,
        log_posteriors=oof_logp,
        budgets=budgets,
        grouping_seeds=grouping_seeds,
    )
    report = {
        "schema_version": 1,
        "protocol": "llmmap_q1_dynamic_v1",
        "adaptation": "Q=1 response-level head followed by evidence accumulation",
        "cache": str(cache_dir),
        "variant": panel.metadata.variant,
        "n_sources": n_classes,
        "n_prompts": n_prompts,
        "configuration": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "feature_size": 384,
            "transformer_blocks": 3,
            "attention_heads": 4,
            "split_seed": split_seed,
            "train_seed": train_seed,
        },
        "folds": fold_reports,
        "metrics_by_grouping_seed": by_seed,
        "metrics_across_grouping_seeds": summary,
        "metrics_by_fold": by_fold,
        "metrics_across_folds": fold_summary,
        "complete": True,
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def _load_fold_models(
    panel: EmbeddingPanel,
    checkpoints_dir: Path,
    device: torch.device,
) -> list[LLMmapClosedClassifier]:
    models = []
    for path in sorted(checkpoints_dir.glob("fold-*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["classes"] != panel.metadata.classes:
            raise ValueError(f"class roster mismatch in {path}")
        model = _new_model(panel, device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models.append(model)
    if not models:
        raise FileNotFoundError(f"no LLMmap checkpoints under {checkpoints_dir}")
    return models


def evaluate_external(
    *,
    cache_dir: Path,
    checkpoints_dir: Path,
    output: Path,
    protocol: str,
    device: str,
    batch_size: int,
    num_workers: int,
    split_seed: int,
    budgets: tuple[int, ...],
    grouping_seeds: tuple[int, ...],
) -> dict[str, Any]:
    panel = EmbeddingPanel(cache_dir)
    torch_device = torch.device(device)
    models = _load_fold_models(panel, checkpoints_dir, torch_device)
    n_classes = len(panel.metadata.classes)
    n_prompts = len(panel.metadata.prompt_ids)
    positions = np.arange(n_prompts, dtype=np.int64)
    loader = _loader(
        panel,
        positions,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=split_seed,
    )
    protocol_details: dict[str, Any]
    if protocol == "fold_model_ensemble":
        fold_logp = np.stack(
            [_predict(model, loader, torch_device) for model in models]
        )
        by_seed, summary, by_fold, fold_summary = grouped_fold_model_metrics(
            labels=[
                label
                for label in panel.metadata.classes
                for _ in range(n_prompts)
            ],
            classes=panel.metadata.classes,
            fold_log_posteriors=fold_logp,
            budgets=budgets,
            grouping_seeds=grouping_seeds,
        )
        protocol_details = {
            "paper_reporting": "metrics_across_folds",
            "fold_model_aggregation": "mean class log-posterior",
            "metrics_by_fold": by_fold,
            "metrics_across_folds": fold_summary,
            "fold_logposterior_ensemble": {
                "metrics_by_grouping_seed": by_seed,
                "metrics_across_grouping_seeds": summary,
            },
        }
    elif protocol == "prompt_matched_out_of_fold":
        if len(models) != 5:
            raise ValueError("prompt-matched evaluation requires five fold models")
        row_logp = np.full((n_classes * n_prompts, n_classes), np.nan)
        fold_assignments = np.full(n_classes * n_prompts, -1, dtype=np.int16)
        folds = list(prompt_grouped_folds(panel.metadata.prompt_ids, 5, split_seed))
        for fold_index, (model, (_train, test_positions)) in enumerate(
            zip(models, folds, strict=True)
        ):
            test_loader = _loader(
                panel,
                test_positions,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                seed=split_seed,
            )
            flat_test = _flat_indices(test_positions, n_classes, n_prompts)
            row_logp[flat_test] = _predict(model, test_loader, torch_device)
            fold_assignments[flat_test] = fold_index
        if np.isnan(row_logp).any() or np.any(fold_assignments < 0):
            raise AssertionError("incomplete prompt-matched predictions")
        labels = [
            label for label in panel.metadata.classes for _ in range(n_prompts)
        ]
        by_seed, summary = grouped_panel_metrics(
            labels=labels,
            classes=panel.metadata.classes,
            log_posteriors=row_logp,
            budgets=budgets,
            grouping_seeds=grouping_seeds,
        )
        strict_by_seed, strict_summary, by_fold, fold_summary = (
            grouped_oof_metrics_detailed(
                labels=labels,
                classes=panel.metadata.classes,
                fold_assignments=fold_assignments,
                log_posteriors=row_logp,
                budgets=budgets,
                grouping_seeds=grouping_seeds,
            )
        )
        protocol_details = {
            "paper_reporting": "metrics_across_grouping_seeds",
            "paper_grouping_scope": (
                "source-wise groups after row-level prompt-matched OOF scoring"
            ),
            "strict_fold_local_metrics_by_grouping_seed": strict_by_seed,
            "strict_fold_local_metrics_across_grouping_seeds": strict_summary,
            "metrics_by_fold": by_fold,
            "metrics_across_folds": fold_summary,
        }
    else:
        raise ValueError(f"unknown external protocol: {protocol}")
    report = {
        "schema_version": 1,
        "protocol": "llmmap_no_retraining_external_v1",
        "cache": str(cache_dir),
        "fold_models": str(checkpoints_dir),
        "evaluation_protocol": protocol,
        "n_sources": n_classes,
        "n_prompts": n_prompts,
        "metrics_by_grouping_seed": by_seed,
        "metrics_across_grouping_seeds": summary,
        **protocol_details,
        "retrained": False,
        "complete": True,
    }
    _atomic_json(output, report)
    return report


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--data-root", type=Path, default=Path("data"))
    cache.add_argument("--variant", default="100-way")
    cache.add_argument("--benchmark", choices=("agent500", "math100"), required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument(
        "--model", default="intfloat/multilingual-e5-large-instruct"
    )
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--batch-size", type=int, default=256)
    cache.add_argument("--max-length", type=int, default=512)
    cache.add_argument("--max-response-chars", type=int, default=650)
    cache.add_argument("--max-response-tokens", type=int)
    cache.add_argument("--local-files-only", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    _add_shared(evaluate)
    evaluate.add_argument("--epochs", type=int, default=6)
    evaluate.add_argument("--learning-rate", type=float, default=1e-4)
    evaluate.add_argument("--n-splits", type=int, default=5)
    evaluate.add_argument("--split-seed", type=int, default=42)
    evaluate.add_argument("--train-seed", type=int, default=4242)
    evaluate.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    evaluate.add_argument(
        "--grouping-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_GROUPING_SEEDS,
    )

    stress = commands.add_parser("stress")
    stress.add_argument("--cache-dir", type=Path, required=True)
    stress.add_argument("--checkpoints-dir", type=Path, required=True)
    stress.add_argument("--output", type=Path, required=True)
    stress.add_argument(
        "--protocol",
        choices=("fold_model_ensemble", "prompt_matched_out_of_fold"),
        required=True,
    )
    _add_shared(stress)
    stress.add_argument("--split-seed", type=int, default=42)
    stress.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    stress.add_argument(
        "--grouping-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_GROUPING_SEEDS,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "cache":
        max_chars = (
            None
            if args.max_response_tokens is not None
            else args.max_response_chars
        )
        cache_embeddings(
            data_root=args.data_root,
            variant=args.variant,
            benchmark=args.benchmark,
            output_dir=args.output_dir,
            model_name_or_path=args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_response_chars=max_chars,
            max_response_tokens=args.max_response_tokens,
            local_files_only=args.local_files_only,
        )
    elif args.command == "evaluate":
        train_and_evaluate(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_splits=args.n_splits,
            split_seed=args.split_seed,
            train_seed=args.train_seed,
            budgets=tuple(sorted(set(args.budgets))),
            grouping_seeds=tuple(dict.fromkeys(args.grouping_seeds)),
        )
    else:
        evaluate_external(
            cache_dir=args.cache_dir,
            checkpoints_dir=args.checkpoints_dir,
            output=args.output,
            protocol=args.protocol,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_seed=args.split_seed,
            budgets=tuple(sorted(set(args.budgets))),
            grouping_seeds=tuple(dict.fromkeys(args.grouping_seeds)),
        )


if __name__ == "__main__":
    main()
