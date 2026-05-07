"""Gradient-based attribution patching.

Approximates activation patching with a single forward + backward pass per
sample: ``attribution[ℓ, t] = a_ℓ[t] · ∂L/∂a_ℓ[t]``, where ``a_ℓ`` is the
residual-stream activation at layer ℓ and ``L`` is the probe's log-prob of
the ground-truth class. Averaging over samples yields a layer-×-position
importance heatmap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from provenance_tracker.datasets.schemas import ResponseRecord


@dataclass(slots=True)
class AttributionConfig:
    proxy_model_name_or_path: str
    best_layer: int
    probe_weights: np.ndarray
    probe_bias: np.ndarray
    probe_mean: np.ndarray
    probe_std: np.ndarray
    class_names: list[str]
    max_length: int = 1024
    device: str = "cuda"
    dtype: str = "bfloat16"
    template: str = "Prompt:\n{prompt}\n\nResponse:\n{response}"
    last_k_positions: int = 32   # aggregate the last K token positions in the heatmap


@dataclass(slots=True)
class AttributionResult:
    layer_indices: list[int]
    heatmap: np.ndarray           # (L, last_k_positions) signed mean attribution
    per_layer_total: np.ndarray   # (L,) sum over positions, averaged over samples
    class_names: list[str]
    num_samples: int


class AttributionPatcher:
    def __init__(self, config: AttributionConfig):
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
        # Freeze params — grads flow only through ``inputs_embeds`` (set below).
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._embed = self.model.get_input_embeddings()

        dev = config.device
        self._W = torch.as_tensor(config.probe_weights, dtype=torch.float32, device=dev)
        self._b = torch.as_tensor(config.probe_bias, dtype=torch.float32, device=dev)
        self._mean = torch.as_tensor(config.probe_mean, dtype=torch.float32, device=dev)
        self._std = torch.as_tensor(config.probe_std, dtype=torch.float32, device=dev)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _format(self, record: ResponseRecord) -> str:
        return self.config.template.format(prompt=record.prompt, response=record.response)

    def _probe_logprob(self, hidden_best: torch.Tensor, class_idx: int) -> torch.Tensor:
        x = hidden_best.to(torch.float32)
        x = (x - self._mean) / self._std
        logits = x @ self._W.T + self._b
        return torch.log_softmax(logits, dim=-1)[0, class_idx]

    def run(self, records: Sequence[ResponseRecord]) -> AttributionResult:
        cfg = self.config
        class_to_idx = {c: i for i, c in enumerate(cfg.class_names)}
        best_layer = cfg.best_layer
        K = cfg.last_k_positions

        # Determine layer count by a single dummy forward.
        with torch.no_grad():
            dummy = self.tokenizer("x", return_tensors="pt").to(cfg.device)
            out = self.model(**dummy, output_hidden_states=True, use_cache=False)
            n_layers = len(out.hidden_states)  # L+1 including embedding
        layer_indices = list(range(n_layers))

        heatmap = np.zeros((n_layers, K), dtype=np.float32)
        per_layer_total = np.zeros(n_layers, dtype=np.float32)

        for i, rec in enumerate(records):
            text = self._format(rec)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_length,
                padding=False,
            ).to(cfg.device)

            # Route through inputs_embeds so grads can flow (model params are frozen).
            embeds = self._embed(inputs["input_ids"]).detach().clone().requires_grad_(True)
            out = self.model(
                inputs_embeds=embeds,
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
                use_cache=False,
            )
            hs_list = list(out.hidden_states)  # tuple of tensors (1, T, D)
            for h in hs_list:
                h.retain_grad()
            h_best = hs_list[best_layer][0, -1, :].unsqueeze(0)
            ci = class_to_idx[rec.label]
            score = self._probe_logprob(h_best, ci)
            self.model.zero_grad(set_to_none=True)
            score.backward()

            for li, h in enumerate(hs_list):
                if h.grad is None:
                    continue
                # (1, T, D) → per-position attribution = sum_d a_d * grad_d
                attrib = (h.detach().to(torch.float32) * h.grad.to(torch.float32)).sum(dim=-1)[0]
                T = attrib.shape[0]
                if T >= K:
                    tail = attrib[-K:]
                else:
                    tail = torch.cat([torch.zeros(K - T, device=attrib.device), attrib])
                heatmap[li] += tail.cpu().numpy()
                per_layer_total[li] += attrib.sum().item()

            if (i + 1) % 50 == 0:
                print(f"[attr] processed {i + 1}/{len(records)} samples")

        heatmap /= max(len(records), 1)
        per_layer_total /= max(len(records), 1)
        return AttributionResult(
            layer_indices=layer_indices,
            heatmap=heatmap,
            per_layer_total=per_layer_total,
            class_names=cfg.class_names,
            num_samples=len(records),
        )
