"""LLM-DNA black-box baseline: sentence-transformer encoding of the response.

This mirrors ``_extract_signature_from_text_responses`` in the reference
LLM-DNA implementation, which encodes only the *target text output* with an
external SentenceTransformer (default ``all-mpnet-base-v2``) — no access to
target model internals. Per-sample vectors are returned (no PCA yet) so the
same downstream evaluation can be applied as to our proxy features.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from provenance_tracker.datasets.schemas import FeatureBatch, ResponseRecord


@dataclass(slots=True)
class MpnetConfig:
    encoder_name_or_path: str
    run_id: str
    batch_size: int = 32
    device: str = "cuda"
    include_prompt: bool = False


class MpnetBaselineExtractor:
    def __init__(self, config: MpnetConfig):
        self.config = config
        self.encoder = SentenceTransformer(
            config.encoder_name_or_path,
            device=config.device,
        )

    def close(self) -> None:
        del self.encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _text(self, r: ResponseRecord) -> str:
        if self.config.include_prompt:
            return f"Prompt: {r.prompt}\nResponse: {r.response}"
        return r.response

    def extract(self, records: Sequence[ResponseRecord], label: str) -> FeatureBatch:
        if not records:
            raise ValueError("records must not be empty")
        texts = [self._text(r) for r in records]
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        return FeatureBatch(
            sample_ids=[r.sample_id for r in records],
            features=embeddings,
            feature_kind="mpnet_response",
            run_id=self.config.run_id,
            model_name=self.config.encoder_name_or_path,
            labels=[label] * len(records),
            layer_index=(),
        )
