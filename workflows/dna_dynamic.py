#!/usr/bin/env python3
"""Run a frozen sentence-encoder DNA baseline on Agent500."""

from __future__ import annotations

import argparse
from pathlib import Path

from reader_provenance.baselines.sentence_embeddings import (
    SentenceEmbeddingReader,
    SentenceEncoderConfig,
)
from reader_provenance.data.records import load_records
from reader_provenance.data.release import DatasetRelease
from reader_provenance.experiments.agent500 import evaluate
from reader_provenance.features.io import save_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--stage", choices=("extract", "evaluate", "all"), default="all"
    )
    parser.add_argument(
        "--variant",
        choices=("50-way", "100-way", "165-way"),
        default="100-way",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dna"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_root = args.output_root / args.variant / args.encoder_tag
    feature_path = run_root / "features.npz"
    if args.stage in {"extract", "all"}:
        release = DatasetRelease(args.data_root)
        records = load_records(release.response_paths(args.variant, "agent500"))
        reader = SentenceEmbeddingReader(
            SentenceEncoderConfig(
                model_name_or_path=args.model,
                batch_size=args.batch_size,
                device=args.device,
                max_length=args.max_length,
                include_prompt=False,
                normalize=False,
            )
        )
        try:
            save_features(feature_path, reader.extract(records))
        finally:
            reader.close()
    if args.stage in {"evaluate", "all"}:
        evaluate(
            feature_path,
            run_root / "report.json",
            run_root / "artifacts",
            component="raw",
            budgets=(1, 5, 10, 20, 50, 100),
            grouping_seeds=(42, 43, 44),
            n_splits=5,
            split_seed=42,
            device=args.device,
        )


if __name__ == "__main__":
    main()
