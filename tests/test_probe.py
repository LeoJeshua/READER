import numpy as np

from reader_provenance.training.probe import LinearSourceProbe, fit_source_probe


def test_adam_source_probe_fits_separable_data(tmp_path) -> None:
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(3), 30)
    centers = np.eye(3, dtype=np.float32) * 5
    features = centers[labels] + rng.normal(0, 0.1, size=(90, 3))
    probe, diagnostics = fit_source_probe(
        features,
        labels,
        ["a", "b", "c"],
        device="cpu",
        max_steps=80,
    )
    assert (probe.logits(features).argmax(axis=1) == labels).mean() > 0.99
    assert diagnostics["steps"] == 80
    path = tmp_path / "probe.npz"
    probe.save(path, held_out_prompts=["p1", "p2"])
    restored, held_out = LinearSourceProbe.load(path)
    assert held_out == ["p1", "p2"]
    assert np.allclose(restored.logits(features), probe.logits(features))
