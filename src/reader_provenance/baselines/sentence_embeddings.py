from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from reader_provenance.data.records import ResponseRecord, load_records
from reader_provenance.features.io import FeatureBatch, save_features


@dataclass(frozen=True, slots=True)
class SentenceEncoderConfig:
    model_name_or_path: str
    batch_size: int = 32
    device: str = "cuda"
    max_length: int | None = None
    include_prompt: bool = False
    normalize: bool = False


class SentenceEmbeddingReader:
    def __init__(self, config: SentenceEncoderConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "install reader-provenance[baselines] for sentence encoders"
            ) from error
        self.config = config
        self.encoder = SentenceTransformer(
            config.model_name_or_path,
            device=config.device,
        )
        if config.max_length is not None:
            self.encoder.max_seq_length = config.max_length

    def close(self) -> None:
        del self.encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def extract(self, records: list[ResponseRecord]) -> FeatureBatch:
        if not records:
            raise ValueError("records must be non-empty")
        if self.config.include_prompt:
            texts = [
                f"Prompt: {row.prompt}\nResponse: {row.response}"
                for row in records
            ]
        else:
            texts = [row.response for row in records]
        features = self.encoder.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        ).astype(np.float32)
        return FeatureBatch(
            features=features,
            labels=[row.label for row in records],
            sample_ids=[row.sample_id for row in records],
            metadata={
                "schema_version": 1,
                "protocol": "response_sentence_embedding_v1",
                "config": asdict(self.config),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    records = load_records(args.records, allow_empty=args.allow_empty)
    reader = SentenceEmbeddingReader(
        SentenceEncoderConfig(
            model_name_or_path=args.model,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
            include_prompt=args.include_prompt,
            normalize=args.normalize,
        )
    )
    try:
        save_features(args.output, reader.extract(records))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
