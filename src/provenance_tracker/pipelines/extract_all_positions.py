"""Extract every-response-token hidden states (one layer) for stride sweeps."""
from __future__ import annotations

import argparse

from provenance_tracker.datasets.schemas import ResponseRecord
from provenance_tracker.proxy.all_positions import (
    AllPositionsConfig,
    AllPositionsExtractor,
    save_all_positions,
)
from provenance_tracker.utils.io import read_jsonl


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract hidden states at every response-span token (padded)."
    )
    p.add_argument("--records-path", required=True, action="append")
    p.add_argument("--proxy-model-name", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-path", required=True,
                   help=".npz holding (N, T_max, D) features + valid_counts[N]")
    p.add_argument("--layer-index", type=int, default=23)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--template", default="{response}")
    p.add_argument("--prefix-template", default="")
    args = p.parse_args()

    records: list[ResponseRecord] = []
    labels: list[str] = []
    for path in args.records_path:
        for row in read_jsonl(path):
            rec = ResponseRecord(**row)
            records.append(rec)
            labels.append(rec.label)

    extractor = AllPositionsExtractor(
        AllPositionsConfig(
            proxy_model_name_or_path=args.proxy_model_name,
            run_id=args.run_id,
            layer_index=args.layer_index,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            template=args.template,
            prefix_template=args.prefix_template,
        )
    )
    try:
        batch, counts = extractor.extract(records, label="__multi__")
    finally:
        extractor.close()
    batch.labels = labels
    save_all_positions(args.output_path, batch, counts)
    print(
        f"[all_positions] features={batch.features.shape} "
        f"counts: min={counts.min()} mean={counts.mean():.1f} max={counts.max()} "
        f"-> {args.output_path}"
    )


if __name__ == "__main__":
    main()
