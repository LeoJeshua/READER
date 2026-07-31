import json

import numpy as np
import torch

from reader_provenance.baselines.llmmap import (
    EmbeddingPanel,
    LLMmapClosedClassifier,
    PromptResponseEmbeddingDataset,
)


def _panel(tmp_path) -> EmbeddingPanel:
    root = tmp_path / "panel"
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "protocol": "llmmap_e5_panel_v1",
        "benchmark": "agent500",
        "variant": "2-way",
        "classes": ["source-a", "source-b"],
        "prompt_ids": ["p0", "p1", "p2"],
        "embedding_dim": 4,
        "max_length": 512,
        "max_response_chars": 650,
        "max_response_tokens": None,
        "embedding_model": "synthetic-e5",
        "complete": True,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.save(root / "query_embeddings.npy", np.arange(12).reshape(3, 4))
    np.save(root / "response_embeddings.npy", np.arange(24).reshape(2, 3, 4))
    return EmbeddingPanel(root)


def test_llmmap_panel_dataset_preserves_source_prompt_order(tmp_path) -> None:
    panel = _panel(tmp_path)
    dataset = PromptResponseEmbeddingDataset(panel, np.asarray([2, 0]))
    trace, label = dataset[2]
    assert label == 1
    assert trace.shape == (1, 8)
    np.testing.assert_array_equal(trace.numpy()[0, :4], [8, 9, 10, 11])
    np.testing.assert_array_equal(trace.numpy()[0, 4:], [20, 21, 22, 23])


def test_llmmap_closed_classifier_shape() -> None:
    model = LLMmapClosedClassifier(
        embedding_dim=8,
        num_classes=5,
        feature_size=16,
        num_blocks=3,
        num_heads=4,
    ).eval()
    with torch.inference_mode():
        logits = model(torch.randn(3, 1, 16))
    assert logits.shape == (3, 5)
