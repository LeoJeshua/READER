"""Extract per-sample (N, M_max, D) intra-trajectory features from a proxy LLM."""
from __future__ import annotations

import argparse

from provenance_tracker.datasets.schemas import ResponseRecord
from provenance_tracker.proxy.intra_positions import (
    IntraExtractorConfig,
    IntraPositionExtractor,
)
from provenance_tracker.utils.io import read_jsonl, save_feature_batch


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract intra-trajectory hidden states at M fixed positions"
    )
    p.add_argument("--records-path", required=True, action="append")
    p.add_argument("--proxy-model-name", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-path", required=True,
                   help=".npz holding (N, M_max, D) features + metadata")
    p.add_argument("--layer-index", type=int, default=23,
                   help="Which proxy layer to extract (best layer was 23 on agent probes)")
    p.add_argument("--positions-per-sample", type=int, default=16,
                   help="M_max: uniform positions inside the response span")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--template", default="Prompt:\n{prompt}\n\nResponse:\n{response}")
    p.add_argument("--prefix-template", default="Prompt:\n{prompt}\n\nResponse:\n",
                   help="Must match everything before {response} in --template")
    p.add_argument("--attn-impl", default="auto",
                   choices=["auto", "flash_attention_2", "sdpa", "eager"])
    args = p.parse_args()

    records: list[ResponseRecord] = []
    labels: list[str] = []
    for path in args.records_path:
        for row in read_jsonl(path):
            rec = ResponseRecord(**row)
            records.append(rec)
            labels.append(rec.label)

    extractor = IntraPositionExtractor(
        IntraExtractorConfig(
            proxy_model_name_or_path=args.proxy_model_name,
            run_id=args.run_id,
            layer_index=args.layer_index,
            positions_per_sample=args.positions_per_sample,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            template=args.template,
            prefix_template=args.prefix_template,
            attn_implementation=args.attn_impl,
        )
    )
    try:
        batch = extractor.extract(records, label="__multi__")
    finally:
        extractor.close()

    batch.labels = labels
    save_feature_batch(args.output_path, batch)
    print(
        f"[intra] features={batch.features.shape} layer={args.layer_index} "
        f"M_max={args.positions_per_sample} -> {args.output_path}"
    )


if __name__ == "__main__":
    main()
