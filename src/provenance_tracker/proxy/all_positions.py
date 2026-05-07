"""Extract hidden states at EVERY response-span token for a single layer.

Downstream evaluators subsample with any stride Δ without re-running the proxy.
Stored shape: features ``(N, T_max, D)`` zero-padded on the right,
plus a sidecar ``valid_counts[N]`` (int32) recording each sample's true length.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from provenance_tracker.datasets.schemas import FeatureBatch, ResponseRecord


@dataclass(slots=True)
class AllPositionsConfig:
    proxy_model_name_or_path: str
    run_id: str
    layer_index: int = 23
    max_length: int = 1024
    batch_size: int = 4
    device: str = "cuda"
    dtype: str = "bfloat16"
    template: str = "{response}"
    prefix_template: str = ""  # chars before response in `template`


class AllPositionsExtractor:
    def __init__(self, config: AllPositionsConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.proxy_model_name_or_path, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        dtype = getattr(torch, config.dtype, torch.bfloat16)
        kwargs = dict(
            dtype=dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            output_hidden_states=True,
        )
        if config.device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(
                config.proxy_model_name_or_path, device_map="auto", **kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.proxy_model_name_or_path, **kwargs
            ).to(config.device)
        self.model.eval()
        self._input_device = next(self.model.parameters()).device

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _spans(self, records: Sequence[ResponseRecord]) -> list[tuple[int, int]]:
        """Return (resp_start, resp_end) indices into the unpadded per-sample token ids."""
        spans: list[tuple[int, int]] = []
        for r in records:
            prefix = self.config.prefix_template.format(prompt=r.prompt)
            full = self.config.template.format(prompt=r.prompt, response=r.response)
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
            resp_start = len(prefix_ids)
            resp_end = min(len(full_ids), self.config.max_length)
            if resp_end <= resp_start:
                resp_start = max(0, resp_end - 1)
            spans.append((resp_start, resp_end))
        return spans

    def extract(
        self, records: Sequence[ResponseRecord], label: str
    ) -> tuple[FeatureBatch, np.ndarray]:
        if not records:
            raise ValueError("records must not be empty")

        spans = self._spans(records)
        t_max = max(end - start for start, end in spans)
        texts = [
            self.config.template.format(prompt=r.prompt, response=r.response)
            for r in records
        ]
        sample_ids = [r.sample_id for r in records]
        labels = [label] * len(records)

        bs = max(1, self.config.batch_size)
        L = self.config.layer_index
        N = len(records)
        # We'll probe D on first forward pass.
        features: np.ndarray | None = None
        valid_counts = np.zeros(N, dtype=np.int32)

        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            chunk_spans = spans[i : i + bs]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
                add_special_tokens=False,
            ).to(self._input_device)
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True, use_cache=False)
            layer_h = out.hidden_states[L]  # (B, T, D)
            T = layer_h.shape[1]
            D = layer_h.shape[-1]
            if features is None:
                features = np.zeros((N, t_max, D), dtype=np.float32)
            for b, (start, end) in enumerate(chunk_spans):
                span_len = end - start
                if span_len <= 0:
                    continue
                end_clipped = min(end, T)
                start_clipped = max(0, min(start, end_clipped - 1))
                real_len = end_clipped - start_clipped
                if real_len <= 0:
                    continue
                vec = layer_h[b, start_clipped:end_clipped, :].to(
                    dtype=torch.float32
                ).cpu().numpy()
                features[i + b, :real_len, :] = vec
                valid_counts[i + b] = real_len

        assert features is not None
        batch = FeatureBatch(
            sample_ids=sample_ids,
            features=features,
            feature_kind=f"proxy_all_positions_L{L}",
            run_id=self.config.run_id,
            model_name=self.config.proxy_model_name_or_path,
            labels=labels,
            layer_index=(L,),
        )
        return batch, valid_counts


def save_all_positions(path: str | Path, batch: FeatureBatch, counts: np.ndarray) -> Path:
    """Save features + counts into a single .npz (adds ``valid_counts`` key)."""
    from provenance_tracker.utils.io import ensure_parent_dir  # local import
    p = ensure_parent_dir(path)
    payload = batch.to_npz_dict()
    payload["valid_counts"] = counts.astype(np.int32)
    np.savez_compressed(p, **payload)
    return p


def load_all_positions(path: str | Path) -> tuple[FeatureBatch, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as data:
        layer_index = tuple(int(x) for x in data["layer_index"].tolist())
        batch = FeatureBatch(
            sample_ids=[str(x) for x in data["sample_ids"].tolist()],
            features=np.asarray(data["features"], dtype=np.float32),
            feature_kind=str(data["feature_kind"]),
            run_id=str(data["run_id"]),
            model_name=str(data["model_name"]),
            labels=[str(x) for x in data["labels"].tolist()],
            layer_index=layer_index,
        )
        counts = np.asarray(data["valid_counts"], dtype=np.int32)
    return batch, counts
