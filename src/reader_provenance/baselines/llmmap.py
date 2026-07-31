"""LLMmap's frozen text encoder and closed-set attribution network."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class EmbeddingPanelMetadata:
    benchmark: str
    variant: str
    classes: list[str]
    prompt_ids: list[str]
    embedding_dim: int
    max_length: int
    max_response_chars: int | None
    max_response_tokens: int | None
    embedding_model: str

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingPanelMetadata:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not payload.get("complete", False):
            raise ValueError(f"incomplete LLMmap embedding panel: {path}")
        return cls(
            benchmark=str(payload["benchmark"]),
            variant=str(payload["variant"]),
            classes=[str(value) for value in payload["classes"]],
            prompt_ids=[str(value) for value in payload["prompt_ids"]],
            embedding_dim=int(payload["embedding_dim"]),
            max_length=int(payload["max_length"]),
            max_response_chars=(
                None
                if payload.get("max_response_chars") is None
                else int(payload["max_response_chars"])
            ),
            max_response_tokens=(
                None
                if payload.get("max_response_tokens") is None
                else int(payload["max_response_tokens"])
            ),
            embedding_model=str(payload["embedding_model"]),
        )


class EmbeddingPanel:
    """Memory-mapped query and response embeddings in source-by-prompt order."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.metadata = EmbeddingPanelMetadata.load(self.root / "manifest.json")
        self.query_embeddings = np.load(
            self.root / "query_embeddings.npy", mmap_mode="r"
        )
        self.response_embeddings = np.load(
            self.root / "response_embeddings.npy", mmap_mode="r"
        )
        expected_query = (
            len(self.metadata.prompt_ids),
            self.metadata.embedding_dim,
        )
        expected_response = (
            len(self.metadata.classes),
            len(self.metadata.prompt_ids),
            self.metadata.embedding_dim,
        )
        if self.query_embeddings.shape != expected_query:
            raise ValueError(
                f"query embedding shape {self.query_embeddings.shape} "
                f"does not match {expected_query}"
            )
        if self.response_embeddings.shape != expected_response:
            raise ValueError(
                f"response embedding shape {self.response_embeddings.shape} "
                f"does not match {expected_response}"
            )


class E5Embedder:
    """Mean-pooled E5 encoder used by the published LLMmap implementation."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str,
        max_length: int,
        local_files_only: bool = False,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        load_kwargs = {
            "trust_remote_code": False,
            "local_files_only": local_files_only,
        }
        try:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                dtype=dtype,
                **load_kwargs,
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                **load_kwargs,
            )
        self.model.to(self.device).eval()

    @property
    def embedding_dim(self) -> int:
        value = getattr(self.model.config, "hidden_size", None)
        if value is None:
            raise ValueError("embedding model config has no hidden_size")
        return int(value)

    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                tokens = self.tokenizer(
                    texts[start : start + batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens.attention_mask.unsqueeze(-1).expand_as(hidden).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
                outputs.append(pooled.float().cpu().numpy())
        if not outputs:
            raise ValueError("cannot encode an empty text collection")
        return np.concatenate(outputs)

    def truncate_by_tokens(self, texts: list[str], max_tokens: int) -> list[str]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=max_tokens,
        )["input_ids"]
        return self.tokenizer.batch_decode(
            encoded,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class PromptResponseEmbeddingDataset(Dataset):
    """One query-response E5 trace for every source and selected prompt."""

    def __init__(self, panel: EmbeddingPanel, prompt_positions: np.ndarray) -> None:
        positions = np.asarray(prompt_positions, dtype=np.int64)
        if positions.ndim != 1 or not len(positions):
            raise ValueError("prompt_positions must be a non-empty vector")
        if positions.min() < 0 or positions.max() >= len(panel.metadata.prompt_ids):
            raise ValueError("prompt position outside the embedding panel")
        self.panel = panel
        self.prompt_positions = positions

    def __len__(self) -> int:
        return len(self.panel.metadata.classes) * len(self.prompt_positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        class_index, local_position = divmod(index, len(self.prompt_positions))
        prompt_position = int(self.prompt_positions[local_position])
        query = np.asarray(
            self.panel.query_embeddings[prompt_position], dtype=np.float32
        )
        response = np.asarray(
            self.panel.response_embeddings[class_index, prompt_position],
            dtype=np.float32,
        )
        trace = np.concatenate((query, response))[None, :]
        return torch.from_numpy(trace), class_index


class _ClassToken(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, feature_size))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.token.expand(values.size(0), -1, -1)


class _TransformerBlock(nn.Module):
    """Block layout from LLMmap v0.2."""

    def __init__(self, feature_size: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm1d(feature_size)
        self.attention = nn.MultiheadAttention(
            feature_size, num_heads, batch_first=True
        )
        self.norm2 = nn.BatchNorm1d(feature_size)
        self.mlp = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.GELU(),
        )

    @staticmethod
    def _normalize(layer: nn.BatchNorm1d, values: torch.Tensor) -> torch.Tensor:
        return layer(values.transpose(1, 2)).transpose(1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize(self.norm1, inputs)
        attended, _ = self.attention(normalized, normalized, normalized)
        values = inputs + attended
        return values + self.mlp(self._normalize(self.norm2, values))


class LLMmapClosedClassifier(nn.Module):
    """Official LLMmap encoder with a closed source-classification head."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_classes: int,
        feature_size: int = 384,
        num_blocks: int = 3,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.feature_size = int(feature_size)
        self.projection = nn.Linear(2 * embedding_dim, feature_size)
        self.activation = nn.GELU()
        self.class_token = _ClassToken(feature_size)
        self.blocks = nn.ModuleList(
            _TransformerBlock(feature_size, num_heads)
            for _ in range(num_blocks)
        )
        self.head = nn.Linear(feature_size, num_classes)

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        values = self.activation(self.projection(traces))
        values = torch.cat((self.class_token(values), values), dim=1)
        for block in self.blocks:
            values = block(values)
        return self.head(values[:, 0])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
