# Reproducing the experiments

## 1. Data

Validate the shared response pool:

```bash
reader-data --data-root data validate --full
```

Materialize a single stream only when an external tool requires one:

```bash
reader-data --data-root data materialize \
  --variant 100-way --benchmark agent500 \
  --output outputs/agent500-100-way.jsonl.gz
```

The repository workflows read the per-source files directly and preserve the
manifest order.

## 2. Canonical readers

Run each tag in `configs/proxies.yaml` with `workflows/agent500.py`. The `main`
role contains four readers and the `full` role contains eight. Extraction is
the GPU-intensive stage. Evaluation fits only the linear probes and can use
`--device cpu`.

```bash
python workflows/agent500.py --proxy-tag qwen35_9b --stage extract --early-exit
python workflows/agent500.py --proxy-tag qwen35_9b --stage evaluate --device cpu
```

The generated `artifacts/oof_log_posteriors.npz` is the input to
`reader-confusion`. Fold probe archives are reused without retraining by
`workflows/stress_tests.py`.

## 3. Dynamic baselines

### DeBERTa-v3-large

```bash
reader-deberta \
  --data-root data --variant 100-way \
  --model microsoft/deberta-v3-large \
  --output-dir outputs/deberta --device cuda
```

### DNA response encoders

First extract flat response embeddings with `reader-embed`, then evaluate them
with `reader-agent500 --component raw`. The paper uses MPNet, BGE-large, and
Qwen3-Embedding-8B. Install the `baselines` extra first.

```bash
reader-data --data-root data materialize \
  --variant 100-way --benchmark agent500 \
  --output outputs/agent500-100-way.jsonl.gz

reader-embed \
  --records outputs/agent500-100-way.jsonl.gz \
  --model sentence-transformers/all-mpnet-base-v2 \
  --output outputs/mpnet-features.npz

reader-agent500 \
  --features outputs/mpnet-features.npz \
  --component raw \
  --output outputs/mpnet-report.json \
  --artifacts-dir outputs/mpnet-artifacts
```

The materialized stream preserves the same roster and prompt order used by the
paper-level workflows.

### LLMmap

```bash
reader-llmmap cache \
  --data-root data --variant 100-way --benchmark agent500 \
  --output-dir outputs/llmmap/cache

reader-llmmap evaluate \
  --cache-dir outputs/llmmap/cache \
  --output-dir outputs/llmmap/evaluation
```

## 4. Static Bench-A task

```bash
python workflows/bench_a.py --proxy-tag qwen35_9b --stage all --early-exit
```

For original static DNA vectors, embed all 67 model panels, then run:

```bash
reader-dna-static \
  --features outputs/bench-a-qwen-embeddings.npz \
  --components 128 --seed 42 \
  --output outputs/bench-a-qwen-dna-vectors.npz
```

Pass one or more `--dna-vectors NAME=PATH` arguments to `workflows/bench_a.py`
during evaluation.

## 5. Ablations and analyses

```bash
python workflows/input_ablation.py --proxy-tag qwen35_9b --stage all
python workflows/layer_scan.py --proxy-tag qwen35_9b --stage all
reader-temporal extract --help
reader-temporal evaluate --help
reader-statistics --help
reader-geometry --help
reader-confusion --help
```

The all-layer extractor uses float16 memory-mapped arrays and atomic progress
files. `--retain-best-only` removes nonselected layer arrays after a completed
scan while retaining the union of layers selected by last-token, DC, AC, and
DC-AC validation.

## 6. Reports

The bundled `results/` directory is sufficient to regenerate compact plots and
CSV tables without model weights:

```bash
mkdir -p .cache/matplotlib .cache/fontconfig
MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache \
  reader-report \
  --results results \
  --proxy-config configs/proxies.yaml \
  --capabilities configs/capabilities.yaml \
  --output-dir outputs/paper
```

Finally run the isolated audit:

```bash
PYTHONPATH=src python tools/validate_release.py --full-data
```
