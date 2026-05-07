"""Last-token activation patching over a proxy LM.

Given an already-trained linear probe at ``best_layer`` that maps the proxy's
last-token hidden state to a target-label distribution, we measure which
intermediate layer is causally responsible for the authorship decision:

1. Clean run on source sample A (label L_A): cache last-token residual at
   every layer.
2. Corrupted run on target sample B (label L_B ≠ L_A): replace the last-token
   residual at layer ℓ with the cached one from A, then read the final
   ``best_layer`` last-token hidden state.
3. The probe is applied to the patched best-layer state; the recovered
   probability of label L_A tells us how much authorship signal flowed
   through layer ℓ.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from provenance_tracker.datasets.schemas import ResponseRecord


@dataclass(slots=True)
class PatchingConfig:
    proxy_model_name_or_path: str
    best_layer: int
    probe_weights: np.ndarray   # (C, D) standardized-space weights
    probe_bias: np.ndarray      # (C,)
    probe_mean: np.ndarray      # (D,) feature standardizer mean
    probe_std: np.ndarray       # (D,) feature standardizer std
    class_names: list[str]
    max_length: int = 1024
    device: str = "cuda"
    dtype: str = "bfloat16"
    template: str = "Prompt:\n{prompt}\n\nResponse:\n{response}"


@dataclass(slots=True)
class PatchingResult:
    layer_indices: list[int]            # which layers were patched (skipping embed)
    importance: np.ndarray              # (L,) mean recovered P(L_A) across pairs
    clean_target_prob: float            # baseline P(L_A) with no patch (averaged)
    clean_source_prob: float            # upper bound: P(L_A) on source itself
    class_names: list[str]
    num_pairs: int


class ActivationPatcher:
    def __init__(self, config: PatchingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.proxy_model_name_or_path, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype = getattr(torch, config.dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.proxy_model_name_or_path,
            dtype=dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            output_hidden_states=True,
        ).to(config.device)
        self.model.eval()

        # Probe tensors on device for fast logit computation.
        dev = config.device
        self._W = torch.as_tensor(config.probe_weights, dtype=torch.float32, device=dev)
        self._b = torch.as_tensor(config.probe_bias, dtype=torch.float32, device=dev)
        self._mean = torch.as_tensor(config.probe_mean, dtype=torch.float32, device=dev)
        self._std = torch.as_tensor(config.probe_std, dtype=torch.float32, device=dev)

        self._layers = self.model.model.layers  # list of transformer blocks
        self._n_blocks = len(self._layers)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _probe_probs(self, hidden_best: torch.Tensor) -> torch.Tensor:
        """hidden_best: (B, D) → (B, C) softmax over class probs."""
        x = hidden_best.to(torch.float32)
        x = (x - self._mean) / self._std
        logits = x @ self._W.T + self._b
        return torch.softmax(logits, dim=-1)

    def _tokenize(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
        )
        return {k: v.to(self.config.device) for k, v in inputs.items()}

    @torch.no_grad()
    def _clean_last_token_residuals(self, text: str) -> list[torch.Tensor]:
        """Return list of (D,) tensors at last-token position for every layer."""
        inputs = self._tokenize(text)
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        return [h[0, -1, :].detach() for h in out.hidden_states]

    @torch.no_grad()
    def _patched_probe_probs(
        self,
        text_target: str,
        patch_block: int,
        patch_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Forward target with residual at block ``patch_block`` replaced
        at the last-token position with ``patch_vec`` (shape ``(D,)``).

        Reads last-token hidden at ``best_layer`` and runs the probe.
        """
        inputs = self._tokenize(text_target)

        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            patched = h.clone()
            patched[:, -1, :] = patch_vec.to(patched.dtype)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        handle = self._layers[patch_block].register_forward_hook(hook)
        try:
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
            hidden_best = out.hidden_states[self.config.best_layer][0, -1, :].unsqueeze(0)
        finally:
            handle.remove()
        return self._probe_probs(hidden_best)

    def _format(self, record: ResponseRecord) -> str:
        return self.config.template.format(prompt=record.prompt, response=record.response)

    def run(
        self,
        records: Sequence[ResponseRecord],
        *,
        num_pairs: int = 50,
        layer_stride: int = 1,
        seed: int = 42,
    ) -> PatchingResult:
        rng = random.Random(seed)
        class_to_idx = {c: i for i, c in enumerate(self.config.class_names)}
        records = list(records)
        n = len(records)
        pairs: list[tuple[int, int]] = []
        attempts = 0
        while len(pairs) < num_pairs and attempts < num_pairs * 10:
            s, t = rng.randrange(n), rng.randrange(n)
            if records[s].label != records[t].label:
                pairs.append((s, t))
            attempts += 1

        # Cache per unique source sample — all L+1 layer last-token residuals.
        source_cache: dict[int, list[torch.Tensor]] = {}
        for s_idx in sorted({s for s, _ in pairs}):
            source_cache[s_idx] = self._clean_last_token_residuals(
                self._format(records[s_idx])
            )

        layer_indices = list(range(0, self._n_blocks, layer_stride))
        importance = np.zeros(len(layer_indices), dtype=np.float32)
        clean_target_probs: list[float] = []
        clean_source_probs: list[float] = []

        for p_idx, (s, t) in enumerate(pairs):
            label_A = records[s].label
            ci_A = class_to_idx[label_A]
            text_t = self._format(records[t])

            # Clean baseline on target: no patching.
            with torch.no_grad():
                inputs = self._tokenize(text_t)
                out = self.model(**inputs, output_hidden_states=True, use_cache=False)
                h_best_clean = out.hidden_states[self.config.best_layer][0, -1, :].unsqueeze(0)
                p_clean_t = self._probe_probs(h_best_clean)[0, ci_A].item()
            clean_target_probs.append(p_clean_t)
            # Upper bound: source probed directly.
            src_last = source_cache[s][self.config.best_layer].unsqueeze(0)
            p_clean_s = self._probe_probs(src_last)[0, ci_A].item()
            clean_source_probs.append(p_clean_s)

            for li, block_idx in enumerate(layer_indices):
                patch_vec = source_cache[s][block_idx + 1]  # hidden after block_idx
                probs = self._patched_probe_probs(text_t, block_idx, patch_vec)
                importance[li] += probs[0, ci_A].item()
            if (p_idx + 1) % 10 == 0:
                print(f"[patch] processed {p_idx + 1}/{len(pairs)} pairs")

        importance /= max(len(pairs), 1)
        return PatchingResult(
            layer_indices=layer_indices,
            importance=importance,
            clean_target_prob=float(np.mean(clean_target_probs)),
            clean_source_prob=float(np.mean(clean_source_probs)),
            class_names=self.config.class_names,
            num_pairs=len(pairs),
        )
