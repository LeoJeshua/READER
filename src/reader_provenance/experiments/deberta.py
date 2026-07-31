"""Fine-tune the paper's response-only DeBERTa baseline."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from reader_provenance.data.records import load_records
from reader_provenance.data.release import DatasetRelease
from reader_provenance.evaluation.grouped import grouped_oof_metrics_detailed
from reader_provenance.evaluation.protocol import prompt_grouped_folds


class _EncodedResponses(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = {key: value[index] for key, value in self.encodings.items()}
        row["labels"] = int(self.labels[index])
        return row


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _predict(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            values.append(torch.log_softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def train_and_evaluate(
    *,
    data_root: Path,
    variant: str,
    model_name_or_path: str,
    output_dir: Path,
    device: str,
    train_batch_size: int,
    eval_batch_size: int,
    max_length: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    n_splits: int,
    split_seed: int,
    budgets: tuple[int, ...],
    grouping_seeds: tuple[int, ...],
) -> dict[str, Any]:
    release = DatasetRelease(data_root)
    records = load_records(release.response_paths(variant, "agent500"))
    texts = [row.response for row in records]
    labels = [row.label for row in records]
    sample_ids = [row.sample_id for row in records]
    classes = sorted(set(labels))
    class_to_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
        use_fast=True,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    torch_device = torch.device(device)
    oof_logp = np.full((len(records), len(classes)), np.nan, dtype=np.float32)
    fold_assignments = np.full(len(records), -1, dtype=np.int16)
    fold_reports = []
    folds = list(prompt_grouped_folds(sample_ids, n_splits, split_seed))
    for fold_index, (train, test) in enumerate(folds):
        fold_seed = split_seed + fold_index
        _seed(fold_seed)
        train_data = _EncodedResponses(
            [texts[index] for index in train],
            y[train],
            tokenizer,
            max_length,
        )
        test_data = _EncodedResponses(
            [texts[index] for index in test],
            y[test],
            tokenizer,
            max_length,
        )
        generator = torch.Generator().manual_seed(fold_seed)
        train_loader = DataLoader(
            train_data,
            batch_size=train_batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(classes),
            id2label={index: label for index, label in enumerate(classes)},
            label2id=class_to_index,
            trust_remote_code=False,
        ).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        total_steps = epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=round(warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
        model.train()
        for _epoch in range(epochs):
            for batch in train_loader:
                inputs = {key: value.to(torch_device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = model(**inputs).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
        oof_logp[test] = _predict(model, test_loader, torch_device)
        fold_assignments[test] = fold_index
        checkpoint = output_dir / "checkpoints" / f"fold-{fold_index}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        fold_reports.append(
            {
                "fold": fold_index,
                "train_responses": int(len(train)),
                "test_responses": int(len(test)),
                "optimizer_steps": total_steps,
            }
        )
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if np.isnan(oof_logp).any() or np.any(fold_assignments < 0):
        raise AssertionError("incomplete DeBERTa out-of-fold predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "oof_log_posteriors.npz",
        log_posteriors=oof_logp,
        labels=y,
        classes=np.asarray(classes, dtype=object),
        sample_ids=np.asarray(sample_ids, dtype=object),
        fold_assignments=fold_assignments,
    )
    by_seed, summary, by_fold, fold_summary = grouped_oof_metrics_detailed(
        labels=labels,
        classes=classes,
        fold_assignments=fold_assignments,
        log_posteriors=oof_logp,
        budgets=budgets,
        grouping_seeds=grouping_seeds,
    )
    report = {
        "schema_version": 1,
        "protocol": "deberta_prompt_grouped_dynamic_v1",
        "model": model_name_or_path,
        "variant": variant,
        "n_responses": len(records),
        "n_sources": len(classes),
        "n_prompts": len(set(sample_ids)),
        "folds": fold_reports,
        "configuration": {
            "response_only": True,
            "max_length": max_length,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "precision": "fp32",
        },
        "metrics_by_grouping_seed": by_seed,
        "metrics_across_grouping_seeds": summary,
        "metrics_by_fold": by_fold,
        "metrics_across_folds": fold_summary,
        "complete": True,
    }
    temporary = output_dir / f".report.json.tmp-{os.getpid()}"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--variant", default="100-way")
    parser.add_argument("--model", default="microsoft/deberta-v3-large")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    train_and_evaluate(
        data_root=args.data_root,
        variant=args.variant,
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        device=args.device,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        epochs=2,
        learning_rate=6e-6,
        weight_decay=0.01,
        warmup_ratio=0.06,
        n_splits=5,
        split_seed=42,
        budgets=(1, 5, 10, 20, 50, 100),
        grouping_seeds=(42, 43, 44),
    )


if __name__ == "__main__":
    main()
