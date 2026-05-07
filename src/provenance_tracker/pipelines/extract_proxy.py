from __future__ import annotations

import argparse

from provenance_tracker.datasets.schemas import ResponseRecord
from provenance_tracker.proxy.hidden_states import LayerwiseProxyExtractor, ProxyConfig
from provenance_tracker.utils.io import read_jsonl, save_feature_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract multi-layer last-token hidden states from a proxy LLM"
    )
    parser.add_argument("--records-path", required=True, action="append",
                        help="Can be given multiple times to merge records from several targets")
    parser.add_argument("--proxy-model-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-path", required=True,
                        help=".npz file holding (N, L+1, D) features + metadata")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--template", default="Prompt:\n{prompt}\n\nResponse:\n{response}",
                        help="Python format string with {prompt} and/or {response} placeholders")
    parser.add_argument("--attn-impl", default="auto",
                        choices=["auto", "flash_attention_2", "sdpa", "eager"],
                        help="Attention backend for HF model load (auto picks FA2 if available)")
    args = parser.parse_args()

    all_records: list[ResponseRecord] = []
    labels: list[str] = []
    for path in args.records_path:
        rows = read_jsonl(path)
        for row in rows:
            rec = ResponseRecord(**row)
            all_records.append(rec)
            labels.append(rec.label)

    extractor = LayerwiseProxyExtractor(
        ProxyConfig(
            proxy_model_name_or_path=args.proxy_model_name,
            run_id=args.run_id,
            device=args.device,
            dtype=args.dtype,
            max_length=args.max_length,
            batch_size=args.batch_size,
            template=args.template,
            attn_implementation=args.attn_impl,
        )
    )
    try:
        batch = extractor.extract(all_records, label="__multi__")
    finally:
        extractor.close()

    # Overwrite aggregate label with per-record labels already collected.
    batch.labels = labels
    save_feature_batch(args.output_path, batch)
    print(
        f"[proxy] features={batch.features.shape} layers={len(batch.layer_index)} "
        f"-> {args.output_path}"
    )


if __name__ == "__main__":
    main()
