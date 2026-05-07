"""Multi-layer last-token hidden-state extractor for a decoder-only proxy LLM.

Core hypothesis: a pretrained decoder LM, when reading a black-box
``Prompt / Response`` pair, forms intermediate representations that encode
not only semantics but also stylistic / author-identifying signals. We therefore
keep the last-token hidden state at *every* layer and defer layer selection
to the downstream circuit analysis.
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
class ProxyConfig:
    proxy_model_name_or_path: str
    run_id: str
    max_length: int = 1024
    batch_size: int = 8
    device: str = "cuda"
    dtype: str = "bfloat16"
    template: str = "Prompt:\n{prompt}\n\nResponse:\n{response}"
    attn_implementation: str = "auto"   # "auto" | "flash_attention_2" | "sdpa" | "eager"


class LayerwiseProxyExtractor:
    """Forward a proxy LM over `(prompt, response)` pairs and return per-layer
    last-token hidden states.

    Output features have shape ``(N, L+1, D)`` where ``L`` is the number of
    transformer blocks (``hidden_states[0]`` = embedding layer).
    """

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.proxy_model_name_or_path, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

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
            # Some architectures don't support FA2 — fall back transparently.
            print(f"[proxy] attn={attn} failed ({e!s:.120}); retrying with sdpa")
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

    def extract(self, records: Sequence[ResponseRecord], label: str) -> FeatureBatch:
        if not records:
            raise ValueError("records must not be empty")

        # Some target records (notably small base LLMs like Llama-3.2-1B / 3B base)
        # legitimately produce empty responses; if a whole batch comes out empty
        # the tokenizer yields a zero-length tensor and the proxy forward errors
        # with "cannot reshape tensor of 0 elements" (seen with qwen3_32b bs=2).
        # Substitute a single-space placeholder so tokenization always has ≥1 token.
        texts = [
            self.config.template.format(prompt=r.prompt, response=r.response) or " "
            for r in records
        ]
        texts = [t if t.strip() else " " for t in texts]
        sample_ids = [r.sample_id for r in records]
        labels = [label] * len(records)

        bs = max(1, self.config.batch_size)
        all_features: list[np.ndarray] = []
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
            ).to(self._input_device)
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True, use_cache=False)
            # tuple of (L+1) tensors each (B, T, D) — embeddings + L blocks
            hidden_states = out.hidden_states
            # With left padding, the last token is always at position -1.
            stacked = torch.stack([h[:, -1, :] for h in hidden_states], dim=1)
            stacked = stacked.to(dtype=torch.float32).cpu().numpy()
            all_features.append(stacked)

        features = np.concatenate(all_features, axis=0)  # (N, L+1, D)
        layer_index = tuple(range(features.shape[1]))
        return FeatureBatch(
            sample_ids=sample_ids,
            features=features,
            feature_kind="proxy_layerwise_last_token",
            run_id=self.config.run_id,
            model_name=self.config.proxy_model_name_or_path,
            labels=labels,
            layer_index=layer_index,
        )
