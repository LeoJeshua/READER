#!/usr/bin/env python3
"""Extract and evaluate READER on the Bench-A-derived relationship task."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import load_proxy

from reader_provenance.data.records import load_records
from reader_provenance.data.release import DatasetRelease
from reader_provenance.experiments.bench_a import evaluate
from reader_provenance.features.io import save_features
from reader_provenance.models.proxy import ProxyConfig, ProxyReader


def _named_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from error
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-tag", required=True)
    parser.add_argument(
        "--stage", choices=("extract", "evaluate", "all"), default="all"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--proxy-config", type=Path, default=Path("configs/proxies.yaml")
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/bench-a"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-exit", action="store_true")
    parser.add_argument("--dna-vectors", type=_named_path, action="append", default=[])
    args = parser.parse_args()

    release = DatasetRelease(args.data_root)
    proxy = load_proxy(args.proxy_config, args.proxy_tag)
    run_root = args.output_root / args.proxy_tag
    feature_path = run_root / "features.npz"
    if args.stage in {"extract", "all"}:
        records = load_records(release.bench_a_response_paths(), allow_empty=True)
        reader = ProxyReader(
            ProxyConfig(
                model_name_or_path=proxy["model"],
                layer=int(proxy.get("bench_a_layer", proxy["layer"])),
                view=proxy["view"],
                max_length=int(proxy["max_length"]),
                batch_size=int(proxy["batch_size"]),
                device=args.device,
                dtype=proxy["dtype"],
                attention=proxy["attention"],
                early_exit=args.early_exit,
            )
        )
        try:
            save_features(feature_path, reader.extract(records))
        finally:
            reader.close()
    if args.stage in {"evaluate", "all"}:
        for protocol in ("pair_disjoint", "model_disjoint", "family_disjoint"):
            evaluate(
                data_root=args.data_root,
                split_protocol=protocol,
                output=run_root / f"{protocol}.json",
                reader_features=feature_path,
                dna_vectors=dict(args.dna_vectors),
            )


if __name__ == "__main__":
    main()
