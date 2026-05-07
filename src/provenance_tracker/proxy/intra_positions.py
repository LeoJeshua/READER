"""Intra-trajectory multi-position hidden-state extractor.

For each (prompt, response) pair, we grab the hidden state at ``M_max`` fixed
positions *inside the response span* of a chosen layer. Downstream evaluators
can then mean-pool subsets of those positions (M=1..M_max) as a cheap,
generation-free alternative to cross-sample aggregation.

Output feature shape: ``(N, M_max, D)``.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from provenance_tracker.datasets.schemas import FeatureBatch, ResponseRecord


@dataclass(slots=True)
class IntraExtractorConfig:
    proxy_model_name_or_path: str
    run_id: str
    layer_index: int = 23
    positions_per_sample: int = 16
    max_length: int = 1024
    batch_size: int = 4
    device: str = "cuda"
    dtype: str = "bfloat16"
    template: str = "Prompt:\n{prompt}\n\nResponse:\n{response}"
    # prefix template must match everything before {response} in `template`
    prefix_template: str = "Prompt:\n{prompt}\n\nResponse:\n"
    attn_implementation: str = "auto"


def _uniform_positions(start: int, end: int, m: int) -> np.ndarray:
    """Return m integer positions uniform over ``[start, end-1]`` (inclusive endpoints).

    If the span is shorter than m, positions are repeated at the available tokens.
    end > start is required (caller guarantees).
    """
    if end <= start:
        raise ValueError(f"empty response span: [{start}, {end})")
    span_len = end - start
    if m == 1:
        return np.asarray([end - 1], dtype=np.int64)
    if span_len >= m:
        pos = np.linspace(start, end - 1, m)
        return np.round(pos).astype(np.int64)
    # fallback: take what we have, pad with the last available position
    pos = np.arange(start, end, dtype=np.int64)
    pad = np.full(m - span_len, end - 1, dtype=np.int64)
    return np.concatenate([pos, pad])


class IntraPositionExtractor:
    def __init__(self, config: IntraExtractorConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.proxy_model_name_or_path, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Right-pad so actual content lives at [0, true_len) and we can
        # compute absolute response-start offsets from prompt-prefix token length.
        self.tokenizer.padding_side = "right"

        dtype = getattr(torch, config.dtype, torch.bfloat16)
        kwargs = dict(
            dtype=dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            output_hidden_states=True,
        )
        attn = config.attn_implementation
        if attn == "auto":
            try:
                import flash_attn  # noqa: F401
                attn = "flash_attention_2"
            except Exception:
                attn = "sdpa"
        kwargs["attn_implementation"] = attn
        try:
            if config.device == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.proxy_model_name_or_path, device_map="auto", **kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.proxy_model_name_or_path, **kwargs
                ).to(config.device)
        except (ValueError, ImportError, RuntimeError) as e:
            print(f"[intra] attn={attn} failed ({e!s:.120}); retrying with sdpa")
            kwargs["attn_implementation"] = "sdpa"
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

    def _token_positions(self, records: Sequence[ResponseRecord]) -> list[np.ndarray]:
        """For each record, compute M_max token positions inside the response span.

        Positions are indices into the right-padded input_ids tensor, which
        matches indices into hidden_states.
        """
        out: list[np.ndarray] = []
        max_len = self.config.max_length
        m = self.config.positions_per_sample
        for r in records:
            prefix = self.config.prefix_template.format(prompt=r.prompt)
            full = self.config.template.format(prompt=r.prompt, response=r.response)
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
            # If tokenizer adds a BOS in `full` but not `prefix`, align by re-adding.
            resp_start = len(prefix_ids)
            resp_end = min(len(full_ids), max_len)
            if resp_end <= resp_start:
                # degenerate: empty response (or full string truncated to ≤
                # prefix length). `extract()` substitutes empty texts with
                # " ", which tokenizes to a single token at position 0, so
                # pin the M positions to [0, 1) to stay consistent.
                resp_start = 0
                resp_end = 1
            out.append(_uniform_positions(resp_start, resp_end, m))
        return out

    def extract(self, records: Sequence[ResponseRecord], label: str) -> FeatureBatch:
        if not records:
            raise ValueError("records must not be empty")

        positions = self._token_positions(records)
        texts = [
            self.config.template.format(prompt=r.prompt, response=r.response)
            for r in records
        ]
        # Same empty-response guard as hidden_states.HiddenStateExtractor:
        # all-empty batches crash the proxy forward with seq_len=0.
        texts = [t if (t and t.strip()) else " " for t in texts]
        sample_ids = [r.sample_id for r in records]
        labels = [label] * len(records)

        bs = max(1, self.config.batch_size)
        L_idx = self.config.layer_index
        m = self.config.positions_per_sample
        all_features: list[np.ndarray] = []

        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            chunk_positions = positions[i : i + bs]
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
            # Pick the requested layer: hidden_states is tuple of (L+1) tensors
            # each (B, T, D). Negative indexing allowed.
            layer_h = out.hidden_states[L_idx]  # (B, T, D)
            T = layer_h.shape[1]
            batch_feats = np.empty((len(chunk), m, layer_h.shape[-1]), dtype=np.float32)
            for b, pos in enumerate(chunk_positions):
                # Clip positions to valid range of this right-padded batch row
                # (truncation may have shortened the true content span).
                clipped = np.clip(pos, 0, T - 1)
                vec = layer_h[b, clipped, :].to(dtype=torch.float32).cpu().numpy()
                batch_feats[b] = vec
            all_features.append(batch_feats)

        features = np.concatenate(all_features, axis=0)  # (N, M, D)
        # layer_index stays singleton since we stored just one layer
        return FeatureBatch(
            sample_ids=sample_ids,
            features=features,
            feature_kind=f"proxy_intra_positions_L{L_idx}",
            run_id=self.config.run_id,
            model_name=self.config.proxy_model_name_or_path,
            labels=labels,
            layer_index=(L_idx,),
        )
