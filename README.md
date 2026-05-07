# LLM-Provenance-Tracker

Black-box LLM provenance identification via proxy-model hidden states.

A frozen open-weight decoder (the *proxy*) reads `(prompt, response)` pairs
sampled from a target API and produces a low-dimensional fingerprint that
identifies the target across many candidates without ever touching its weights.

This release packages the **library** and **probe / target / MMLU-Pro data**
behind the paper. Per-experiment driver scripts, intermediate JSON reports,
and figure-rendering code live in the authors' internal tree and are not
shipped here.

## What's in this repository

```
src/provenance_tracker/      Python library (importable as `provenance_tracker`)
data/agent/                   500-prompt agentic probe + 50-target registry
data/mmlu_pro/                local copy of TIGER-Lab/MMLU-Pro (CC-BY-4.0)
```

### `src/provenance_tracker/`

| Sub-package | Purpose |
| --- | --- |
| `config.py` | Local HF-cache root, target / proxy registry, experiment constants |
| `datasets/` | `ProbeSample` / `ResponseRecord` / `FeatureBatch` schemas + JSONL loaders |
| `collectors/` | Run a target model locally and dump `(prompt, response)` pairs |
| `proxy/` | Multi-layer last-token activation extractor (the proxy reader) |
| `baselines/` | LLM-DNA-style sentence-encoder baseline (`all-mpnet-base-v2`) |
| `analysis/` | Layer-wise linear probe, top-neuron selection, CKA, SAE, activation/attribution patching |
| `classifiers/` | Scaler + logistic-regression provenance classifier |
| `evaluation/` | Multi-trajectory aggregators (mean-pool, log-posterior, gaussian-disc), few-shot, calibration metrics, retrieval / clustering metrics |
| `pipelines/` | End-to-end CLI entry points (record collection, feature extraction, intra-/cross-K evaluation, etc.) |
| `utils/` | JSONL + `.npz` I/O helpers |

The main fingerprint used in the paper is **intra-sample mean-pool +
cross-sample log-posterior** (`evaluation/logposterior_metrics.py` +
`evaluation/multi_traj.py`).

### `data/agent/`

- `agent_probe.json` — 500 agentic / coding / reasoning prompts (one JSON list).
- `agent_probe_classfied.json` — same prompts annotated by topic.
- `n500_targets.json` — registry of 50 target models (label → repo id, family).

### `data/mmlu_pro/`

`test.json` and `validation.json` are a local copy of the official
[`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
release, redistributed under CC-BY-4.0 for offline reproducibility. Cite the
original dataset when used downstream.

## Installation

```bash
pip install -e .
export PYTHONPATH=./src
```

Required runtime: `torch>=2.4`, `transformers>=4.40`, `numpy`, `scikit-learn`.
The library expects target / proxy weights to be resolvable from a local
HuggingFace hub directory; see `provenance_tracker.config` for how the cache
root is pinned and overridden.

## Quick example

```python
from pathlib import Path
from provenance_tracker.datasets.loaders import load_probe
from provenance_tracker.proxy.hidden_states import extract_layerwise

probes = load_probe(Path("data/agent/agent_probe.json"))[:8]
# `extract_layerwise` runs Qwen3-8B over (prompt, response) pairs and
# returns last-token hidden states for every layer. See the docstring for
# the full signature.
```

For full usage of each pipeline (record collection, intra-M / cross-K
evaluation, log-posterior metrics, multi-trajectory aggregation, etc.), the
canonical entry points live under `src/provenance_tracker/pipelines/` and
each module has a `python -m provenance_tracker.pipelines.<name> --help`.

## Citation
TBD

## License

Code under `src/`: MIT (see `LICENSE`).
Data under `data/mmlu_pro/`: CC-BY-4.0, retained from the upstream
`TIGER-Lab/MMLU-Pro` release.
Data under `data/agent/`: released under MIT alongside the code.
