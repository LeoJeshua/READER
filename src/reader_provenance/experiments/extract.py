from __future__ import annotations

import argparse
from pathlib import Path

from reader_provenance.data.records import load_records
from reader_provenance.features.io import save_features
from reader_provenance.models.proxy import ProxyConfig, ProxyReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--view",
        choices=("prompt_response", "response_only"),
        default="prompt_response",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="auto")
    parser.add_argument("--early-exit", action="store_true")
    parser.add_argument("--max-response-tokens", type=int)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    records = load_records(args.records, allow_empty=args.allow_empty)
    reader = ProxyReader(
        ProxyConfig(
            model_name_or_path=args.proxy,
            layer=args.layer,
            view=args.view,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            attention=args.attention,
            early_exit=args.early_exit,
            max_response_tokens=args.max_response_tokens,
        )
    )
    try:
        features = reader.extract(records)
    finally:
        reader.close()
    save_features(args.output, features)


if __name__ == "__main__":
    main()
