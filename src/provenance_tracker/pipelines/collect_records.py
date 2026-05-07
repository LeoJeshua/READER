from __future__ import annotations

import argparse
from pathlib import Path

from provenance_tracker.collectors.local_generation import (
    GenerationConfig,
    LocalResponseCollector,
    save_records,
)
from provenance_tracker.datasets.loaders import load_rand_probes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect responses from one local target model into a JSONL of records"
    )
    parser.add_argument("--model-name", required=True,
                        help="HF repo id or absolute snapshot path (loaded from local cache only)")
    parser.add_argument("--label", required=True, help="Short alphabetic label for this model")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--dataset-path",
        default="/data/projects/LLM_Identifier/data/rand/rand_dataset.json",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first OFFSET prompts (sample_ids stay absolute)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    probes = load_rand_probes(
        dataset_path=args.dataset_path, limit=args.max_samples, offset=args.offset
    )
    collector = LocalResponseCollector(
        GenerationConfig(
            model_name_or_path=args.model_name,
            label=args.label,
            run_id=args.run_id,
            device=args.device,
            dtype=args.dtype,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            use_chat_template=not args.no_chat_template,
            system_prompt=args.system_prompt,
            seed=args.seed,
        )
    )
    try:
        records = collector.collect(probes)
    finally:
        collector.close()
    save_records(records, str(Path(args.output_path)))
    print(f"[collect] wrote {len(records)} records -> {args.output_path}")


if __name__ == "__main__":
    main()
