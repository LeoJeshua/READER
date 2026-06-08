# READER: Robust Evidence-based Authorship Decoding via Extracted Representations

> **Anonymous repository — NeurIPS 2026 submission.**
> Provided for reviewer reproducibility. Author, institution, and
> acknowledgement metadata have been removed in accordance with the
> double-blind review policy.

## Overview

This repository ships the **library** and **probe / target / MMLU-Pro
data** needed to reproduce the main results in the paper.

## Method at a glance

**Provenance setting.** READER targets dynamic black-box auditing: the
auditor sees query-varying prompts and generated responses, but has no
access to target-model internals.

![Provenance settings from white-box to dynamic black-box auditing](https://anonymous.4open.science/api/repo/READER/file/assets/setup.svg?v=f49ee13b)

**READER pipeline.** A frozen proxy LLM reads each black-box response,
temporal filtering aggregates selected hidden states within a response,
and Bayesian evidence accumulation combines per-response posteriors
across prompts for final source-model attribution.

![Overview of the READER pipeline](https://anonymous.4open.science/api/repo/READER/file/assets/pipeline.svg?v=a0216eeb)

## Repository layout

```log
src/provenance_tracker/   Python library (importable as `provenance_tracker`)
data/agent/               500-prompt agentic probe + 50-target registry
data/mmlu_pro/            Local copy of TIGER-Lab/MMLU-Pro (CC-BY-4.0)
```

### `src/provenance_tracker/`

| Sub-package | Purpose |
| --- | --- |
| `config.py` | Local HF-cache root, target / proxy registry, experiment constants |
| `datasets/` | `ProbeSample` / `ResponseRecord` / `FeatureBatch` schemas and JSONL loaders |
| `collectors/` | Run a target model locally and dump `(prompt, response)` pairs |
| `proxy/` | Multi-layer last-token activation extractor (the proxy reader) |
| `baselines/` | Sentence-encoder baseline (`all-mpnet-base-v2`) |
| `analysis/` | Layer-wise linear probes, top-neuron selection, CKA, SAE, activation / attribution patching |
| `classifiers/` | Scaler + logistic-regression provenance classifier |
| `evaluation/` | Multi-trajectory aggregators (mean-pool, log-posterior, gaussian-disc), few-shot, calibration, retrieval and clustering metrics |
| `pipelines/` | End-to-end CLI entry points (record collection, feature extraction, intra-/cross-K evaluation) |
| `utils/` | JSONL and `.npz` I/O helpers |

The fingerprint reported in the paper is **intra-sample mean-pool +
cross-sample log-posterior**, implemented in
`evaluation/logposterior_metrics.py` and `evaluation/multi_traj.py`.

### `data/agent/`

- `agent_probe.json` — 500 agentic, coding, and reasoning prompts.
- `agent_probe_classfied.json` — the same prompts annotated by topic.
- `n500_targets.json` — registry of 50 candidate target models (label, repo id, family).

### `data/mmlu_pro/`

`test.json` and `validation.json` are an unmodified copy of
[`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro),
redistributed under CC-BY-4.0 for offline reproducibility. Please cite
the original dataset when used downstream.

## Installation

```bash
pip install -e .
export PYTHONPATH=./src
```

Runtime requirements: `torch>=2.4`, `transformers>=4.40`, `numpy`,
`scikit-learn`. Target and proxy weights are resolved from a local
HuggingFace hub directory; the cache root is pinned in
`provenance_tracker.config` and can be overridden via environment
variables.

## Quickstart

```python
from pathlib import Path
from provenance_tracker.datasets.loaders import load_probe
from provenance_tracker.proxy.hidden_states import extract_layerwise

probes = load_probe(Path("data/agent/agent_probe.json"))[:8]
# extract_layerwise runs the proxy decoder over (prompt, response) pairs
# and returns last-token hidden states for every layer.
```

Each end-to-end pipeline — record collection, feature extraction,
intra-M / cross-K evaluation, log-posterior metrics, multi-trajectory
aggregation — is exposed as a CLI module under
`src/provenance_tracker/pipelines/`. Run

```bash
python -m provenance_tracker.pipelines.<name> --help
```

for the full argument list of any pipeline.

## Citation

TBD

## License

Code under `src/`: MIT (see `LICENSE`).
Data under `data/mmlu_pro/`: CC-BY-4.0, retained from the upstream
`TIGER-Lab/MMLU-Pro` release.
Data under `data/agent/`: released under MIT alongside the code.
