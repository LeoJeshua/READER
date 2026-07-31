"""Hugging Face proxy extraction for all-token DC--AC fingerprints."""
from __future__ import annotations

import gc
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from reader_provenance.data.records import ResponseRecord
from reader_provenance.features.dct import encode_padded_trajectories
from reader_provenance.features.io import FeatureBatch

UNIFIED_MODEL_TYPES = {"gemma4_unified", "mistral3"}


@dataclass(frozen=True, slots=True)
class ProxyConfig:
    model_name_or_path: str
    layer: int
    view: str = "prompt_response"
    max_length: int = 1024
    batch_size: int = 4
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention: str = "auto"
    early_exit: bool = False
    max_response_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.view not in {"prompt_response", "response_only"}:
            raise ValueError("view must be prompt_response or response_only")
        if self.layer < 0 or self.max_length < 1 or self.batch_size < 1:
            raise ValueError("invalid proxy layer, max length, or batch size")
        if self.max_response_tokens is not None and self.max_response_tokens < 1:
            raise ValueError("max_response_tokens must be positive")

    @property
    def template(self) -> str:
        if self.view == "response_only":
            return "{response}"
        return "Prompt:\n{prompt}\n\nResponse:\n{response}"

    @property
    def prefix_template(self) -> str:
        if self.view == "response_only":
            return ""
        return "Prompt:\n{prompt}\n\nResponse:\n"


@dataclass(slots=True)
class LayerwiseFeatureChunk:
    """Three temporal readouts for every hidden-state layer in one batch."""

    offset: int
    features: np.ndarray
    response_lengths: np.ndarray


class _SelectedLayerReached(RuntimeError):
    def __init__(self, hidden_state: torch.Tensor):
        super().__init__("selected proxy layer reached")
        self.hidden_state = hidden_state


def _hidden_states(output: Any) -> tuple[torch.Tensor, ...]:
    direct = getattr(output, "hidden_states", None)
    if direct is not None:
        return tuple(direct)
    for attribute in ("language_model_output", "text_model_output", "model_output"):
        nested = getattr(output, attribute, None)
        values = None if nested is None else getattr(nested, "hidden_states", None)
        if values is not None:
            return tuple(values)
    raise RuntimeError(f"{type(output).__name__} did not return text hidden states")


def _hidden_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                return value
    value = getattr(output, "last_hidden_state", None)
    if isinstance(value, torch.Tensor):
        return value
    raise RuntimeError(f"cannot identify hidden state in {type(output).__name__}")


def _decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if not isinstance(module, torch.nn.ModuleList) or not name.endswith("layers"):
            continue
        if "vision" in lowered or "visual" in lowered or lowered.startswith("mtp."):
            continue
        candidates.append(module)
    if not candidates:
        raise RuntimeError("could not locate text decoder layers")
    return max(candidates, key=len)


def _load_tokenizer(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if str(config.model_type) == "mistral3":
        kwargs["fix_mistral_regex"] = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    except (ImportError, KeyError, ValueError):
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                f"processor for {model_path} has no tokenizer"
            ) from None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer, str(config.model_type), config


def _attention_implementation(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _model_class(model_type: str):
    if model_type == "mistral3":
        cls = getattr(transformers, "Mistral3ForConditionalGeneration", None)
    elif model_type == "gemma4_unified":
        cls = getattr(transformers, "AutoModelForMultimodalLM", None)
    else:
        cls = AutoModelForCausalLM
    if cls is None:
        raise RuntimeError(
            f"transformers={transformers.__version__} lacks support for {model_type}"
        )
    return cls


def _load_model(config: ProxyConfig, model_type: str, model_config: Any):
    dtype = getattr(torch, config.dtype)
    attention = _attention_implementation(config.attention)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
        "attn_implementation": attention,
    }
    if model_type == "mistral3":
        quantization = dict(getattr(model_config, "quantization_config", {}) or {})
        if quantization.get("quant_method") == "fp8":
            cls = getattr(transformers, "FineGrainedFP8Config", None)
            if cls is None:
                raise RuntimeError("transformers lacks FineGrainedFP8Config")
            quantization["dequantize"] = True
            kwargs["quantization_config"] = cls(**quantization)
    model_class = _model_class(model_type)

    def load(active: dict[str, Any]):
        if config.device == "auto":
            return model_class.from_pretrained(
                config.model_name_or_path, device_map="auto", **active
            )
        return model_class.from_pretrained(config.model_name_or_path, **active).to(
            config.device
        )

    try:
        model = load(kwargs)
    except (ImportError, RuntimeError, ValueError):
        if kwargs["attn_implementation"] == "sdpa":
            raise
        kwargs["attn_implementation"] = "sdpa"
        model = load(kwargs)
    model.eval()
    return model


class ProxyReader:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.tokenizer, self.model_type, model_config = _load_tokenizer(
            config.model_name_or_path
        )
        self.model = _load_model(config, self.model_type, model_config)
        self.input_device = next(self.model.parameters()).device
        self._layers: torch.nn.ModuleList | None = None

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _forward_kwargs(self, *, output_hidden_states: bool) -> dict[str, Any]:
        values: dict[str, Any] = {
            "output_hidden_states": output_hidden_states,
            "use_cache": False,
        }
        if self.model_type in UNIFIED_MODEL_TYPES:
            values["logits_to_keep"] = 1
        return values

    def _selected_hidden(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.config.early_exit:
            output = self.model(
                **inputs,
                **self._forward_kwargs(output_hidden_states=True),
            )
            return _hidden_states(output)[self.config.layer]
        if self.config.layer < 1:
            raise ValueError("early exit requires a hidden-state layer >= 1")
        if self._layers is None:
            self._layers = _decoder_layers(self.model)
        if self.config.layer > len(self._layers):
            raise ValueError("selected layer exceeds the text decoder depth")

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            raise _SelectedLayerReached(_hidden_from_layer_output(output))

        handle = self._layers[self.config.layer - 1].register_forward_hook(capture)
        try:
            self.model(**inputs, **self._forward_kwargs(output_hidden_states=False))
        except _SelectedLayerReached as reached:
            return reached.hidden_state
        finally:
            handle.remove()
        raise RuntimeError("proxy forward completed before early-exit capture")

    def _response_starts(self, records: Sequence[ResponseRecord]) -> np.ndarray:
        if not self.config.prefix_template:
            return np.zeros(len(records), dtype=np.int64)
        prefixes = [
            self.config.prefix_template.format(prompt=record.prompt)
            for record in records
        ]
        encoded = self.tokenizer(
            prefixes,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        return np.asarray([len(tokens) for tokens in encoded], dtype=np.int64)

    def extract(self, records: Sequence[ResponseRecord]) -> FeatureBatch:
        if not records:
            raise ValueError("records must be non-empty")
        labels = [record.label for record in records]
        sample_ids = [record.sample_id for record in records]
        starts = self._response_starts(records)
        texts = [
            self.config.template.format(
                prompt=record.prompt,
                response=record.response,
            )
            or " "
            for record in records
        ]
        output = []
        effective_lengths = []
        at_cap = 0
        for offset in range(0, len(records), self.config.batch_size):
            chunk = texts[offset : offset + self.config.batch_size]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
                add_special_tokens=False,
            ).to(self.input_device)
            ends = inputs["attention_mask"].sum(dim=1).to(torch.long)
            chunk_starts = torch.as_tensor(
                starts[offset : offset + len(chunk)],
                dtype=torch.long,
                device=self.input_device,
            )
            chunk_starts = torch.minimum(chunk_starts, ends - 1)
            if self.config.max_response_tokens is not None:
                ends = torch.minimum(
                    ends,
                    chunk_starts + self.config.max_response_tokens,
                )
            at_cap += int((ends >= self.config.max_length).sum().item())
            with torch.no_grad():
                hidden = self._selected_hidden(inputs)
                fingerprints = encode_padded_trajectories(
                    hidden, chunk_starts, ends, n_modes=2
                )
            output.append(fingerprints.float().cpu().numpy())
            effective_lengths.extend((ends - chunk_starts).cpu().tolist())
        features = np.concatenate(output, axis=0).astype(np.float32, copy=False)
        return FeatureBatch(
            features=features,
            labels=labels,
            sample_ids=sample_ids,
            metadata={
                "schema_version": 1,
                "protocol": "reader_all_response_tokens_dc_ac_v1",
                "proxy": self.config.model_name_or_path,
                "layer": self.config.layer,
                "view": self.config.view,
                "normalization": "orthonormal DCT-II divided by sqrt(N)",
                "dct_modes": [0, 1],
                "max_length": self.config.max_length,
                "early_exit": self.config.early_exit,
                "response_token_limit": self.config.max_response_tokens,
                "at_max_length": at_cap,
                "response_token_lengths": {
                    "min": int(np.min(effective_lengths)),
                    "median": float(np.median(effective_lengths)),
                    "max": int(np.max(effective_lengths)),
                    "mean": float(np.mean(effective_lengths)),
                },
                "config": asdict(self.config),
            },
        )

    def iter_layerwise_features(
        self,
        records: Sequence[ResponseRecord],
    ) -> Iterator[LayerwiseFeatureChunk]:
        """Yield ``(layer, [last, DC, AC], sample, hidden)`` chunks."""
        if not records:
            raise ValueError("records must be non-empty")
        starts = self._response_starts(records)
        texts = [
            self.config.template.format(
                prompt=record.prompt,
                response=record.response,
            )
            or " "
            for record in records
        ]
        for offset in range(0, len(records), self.config.batch_size):
            chunk = texts[offset : offset + self.config.batch_size]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
                add_special_tokens=False,
            ).to(self.input_device)
            ends = inputs["attention_mask"].sum(dim=1).to(torch.long)
            chunk_starts = torch.as_tensor(
                starts[offset : offset + len(chunk)],
                dtype=torch.long,
                device=self.input_device,
            )
            chunk_starts = torch.minimum(chunk_starts, ends - 1)
            if self.config.max_response_tokens is not None:
                ends = torch.minimum(
                    ends,
                    chunk_starts + self.config.max_response_tokens,
                )
            with torch.no_grad():
                output = self.model(
                    **inputs,
                    **self._forward_kwargs(output_hidden_states=True),
                )
            hidden_states = _hidden_states(output)
            rows = torch.arange(len(chunk), device=self.input_device)
            last_positions = ends - 1
            layer_features = np.empty(
                (
                    len(hidden_states),
                    3,
                    len(chunk),
                    hidden_states[0].shape[-1],
                ),
                dtype=np.float16,
            )
            for layer_index, hidden in enumerate(hidden_states):
                coefficients = encode_padded_trajectories(
                    hidden,
                    chunk_starts,
                    ends,
                    n_modes=2,
                )
                layer_features[layer_index, 0] = (
                    hidden[rows, last_positions].to(torch.float16).cpu().numpy()
                )
                layer_features[layer_index, 1:3] = (
                    coefficients.to(torch.float16)
                    .cpu()
                    .numpy()
                    .transpose(1, 0, 2)
                )
            yield LayerwiseFeatureChunk(
                offset=offset,
                features=layer_features,
                response_lengths=(ends - chunk_starts).cpu().numpy(),
            )

    def extract_temporal_controls(
        self,
        records: Sequence[ResponseRecord],
        *,
        n_modes: int = 8,
    ) -> tuple[FeatureBatch, FeatureBatch]:
        """Extract DCT modes, coordinate maxima, and final states in one pass."""
        if not records or n_modes < 2:
            raise ValueError("temporal controls require records and at least two modes")
        labels = [record.label for record in records]
        sample_ids = [record.sample_id for record in records]
        starts = self._response_starts(records)
        texts = [
            self.config.template.format(
                prompt=record.prompt,
                response=record.response,
            )
            or " "
            for record in records
        ]
        dct_output = []
        pool_output = []
        effective_lengths = []
        at_cap = 0
        for offset in range(0, len(records), self.config.batch_size):
            chunk = texts[offset : offset + self.config.batch_size]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
                add_special_tokens=False,
            ).to(self.input_device)
            raw_ends = inputs["attention_mask"].sum(dim=1).to(torch.long)
            chunk_starts = torch.as_tensor(
                starts[offset : offset + len(chunk)],
                dtype=torch.long,
                device=self.input_device,
            )
            chunk_starts = torch.minimum(chunk_starts, raw_ends - 1)
            ends = raw_ends
            if self.config.max_response_tokens is not None:
                ends = torch.minimum(
                    ends,
                    chunk_starts + self.config.max_response_tokens,
                )
            at_cap += int((raw_ends >= self.config.max_length).sum().item())
            with torch.no_grad():
                hidden = self._selected_hidden(inputs)
                dct = encode_padded_trajectories(
                    hidden, chunk_starts, ends, n_modes=n_modes
                )
                pools = []
                for row, (start, end) in enumerate(
                    zip(chunk_starts.tolist(), ends.tolist(), strict=True)
                ):
                    span = hidden[row, start:end]
                    pools.append(torch.stack((span.amax(dim=0), span[-1])))
            dct_output.append(dct.float().cpu().numpy())
            pool_output.append(torch.stack(pools).float().cpu().numpy())
            effective_lengths.extend((ends - chunk_starts).cpu().tolist())
        shared = {
            "schema_version": 1,
            "proxy": self.config.model_name_or_path,
            "layer": self.config.layer,
            "view": self.config.view,
            "max_length": self.config.max_length,
            "response_token_limit": self.config.max_response_tokens,
            "at_max_length": at_cap,
            "response_token_lengths": {
                "min": int(np.min(effective_lengths)),
                "median": float(np.median(effective_lengths)),
                "max": int(np.max(effective_lengths)),
                "mean": float(np.mean(effective_lengths)),
            },
            "config": asdict(self.config),
        }
        dct_batch = FeatureBatch(
            features=np.concatenate(dct_output).astype(np.float32, copy=False),
            labels=labels,
            sample_ids=sample_ids,
            metadata={
                **shared,
                "protocol": "reader_matched_temporal_dct_v1",
                "dct_modes": list(range(n_modes)),
                "normalization": "orthonormal DCT-II divided by sqrt(N)",
            },
        )
        pool_batch = FeatureBatch(
            features=np.concatenate(pool_output).astype(np.float32, copy=False),
            labels=labels,
            sample_ids=sample_ids,
            metadata={
                **shared,
                "protocol": "reader_matched_temporal_pooling_v1",
                "components": ["coordinate_maximum", "final_response_token"],
            },
        )
        return dct_batch, pool_batch
