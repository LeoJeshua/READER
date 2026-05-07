from __future__ import annotations

import argparse

from provenance_tracker.baselines.mpnet_encoder import MpnetBaselineExtractor, MpnetConfig
from provenance_tracker.datasets.schemas import ResponseRecord
from provenance_tracker.utils.io import read_jsonl, save_feature_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-DNA black-box baseline: encode responses with all-mpnet-base-v2"
    )
    parser.add_argument("--records-path", required=True, action="append")
    parser.add_argument("--encoder-name", required=True,
                        help="Path or repo id of a SentenceTransformer model")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--include-prompt", action="store_true")
    args = parser.parse_args()

    all_records: list[ResponseRecord] = []
    labels: list[str] = []
    for path in args.records_path:
        for row in read_jsonl(path):
            rec = ResponseRecord(**row)
            all_records.append(rec)
            labels.append(rec.label)

    extractor = MpnetBaselineExtractor(
        MpnetConfig(
            encoder_name_or_path=args.encoder_name,
            run_id=args.run_id,
            device=args.device,
            batch_size=args.batch_size,
            include_prompt=args.include_prompt,
        )
    )
    try:
        batch = extractor.extract(all_records, label="__multi__")
    finally:
        extractor.close()

    batch.labels = labels
    save_feature_batch(args.output_path, batch)
    print(f"[mpnet] features={batch.features.shape} -> {args.output_path}")


if __name__ == "__main__":
    main()
