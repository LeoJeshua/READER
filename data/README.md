# READER data release

This directory contains the prompt panels and observed model responses needed
to reproduce the released experiments. The 50-way, 100-way, and 165-way
variants share one canonical 165-source response pool. Smaller variants select
ordered roster prefixes and do not duplicate response files.

## Layout

```text
prompts/                 Agent500 and Math100 prompt panels
responses/agent500/      500 records per source
responses/math100/       100 records per source
rosters/                 Exact ordered 50/100/165-source definitions
bench_a/                 Static relationship prompts, responses, pairs, splits
manifests/release.json   Record counts and compressed/raw SHA-256 checksums
```

Every dynamic response record is JSONL with these fields:

```json
{
  "sample_id": "rand-000000",
  "prompt": "...",
  "response": "...",
  "label": "source_label",
  "model_name": "provider/model-identifier",
  "run_id": "collection-run",
  "timestamp": "ISO-8601 timestamp",
  "probe_category": "prompt category"
}
```

All JSONL streams are gzip-compressed. File order follows the prompt panel,
and source order follows the selected roster.

## Counts

| Component | Sources/models | Prompts per source | Records |
|---|---:|---:|---:|
| Agent500 shared pool | 165 | 500 | 82,500 |
| Math100 shared pool | 165 | 100 | 16,500 |
| Bench-A responses | 67 | 600 | 40,200 |
| **Total** |  |  | **139,200** |

The nested dynamic views contain 25,000/5,000 records at 50-way,
50,000/10,000 at 100-way, and 82,500/16,500 at 165-way for
Agent500/Math100, respectively.

Bench-A preserves the source data exactly. This includes 600 empty responses
for `RLM-spell-checker` and one empty `phi-half` response. Static workflows
explicitly opt into these retained empty records.

## Validation and materialization

Validate every compressed checksum, decompressed checksum, JSON row, roster,
and count:

```bash
reader-data --data-root data validate --full
```

Materialize a single interleaved stream only when an external baseline needs
one file:

```bash
reader-data --data-root data materialize \
  --variant 100-way --benchmark agent500 \
  --output outputs/agent500-100-way.jsonl.gz
```

Prompt and response data are provided for research reproduction and remain
subject to applicable source-model and API-provider terms.
