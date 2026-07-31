#!/usr/bin/env python3
"""Extract and scan every proxy layer on the complete Agent500 panel."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import load_proxy

from reader_provenance.data.records import load_records
from reader_provenance.data.release import DatasetRelease
from reader_provenance.experiments.layers import (
    evaluate_layer_scan,
    extract_layerwise,
)
from reader_provenance.models.proxy import ProxyConfig, ProxyReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-tag", required=True)
    parser.add_argument(
        "--stage", choices=("extract", "evaluate", "all"), default="all"
    )
    parser.add_argument("--variant", default="100-way")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--proxy-config", type=Path, default=Path("configs/proxies.yaml")
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/layer-scan"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retain-best-only", action="store_true")
    args = parser.parse_args()

    proxy = load_proxy(args.proxy_config, args.proxy_tag)
    run_root = args.output_root / args.variant / args.proxy_tag
    if args.stage in {"extract", "all"}:
        release = DatasetRelease(args.data_root)
        records = load_records(release.response_paths(args.variant, "agent500"))
        reader = ProxyReader(
            ProxyConfig(
                model_name_or_path=proxy["model"],
                layer=0,
                view=proxy["view"],
                max_length=int(proxy["max_length"]),
                batch_size=int(proxy["batch_size"]),
                device=args.device,
                dtype=proxy["dtype"],
                attention=proxy["attention"],
            )
        )
        try:
            extract_layerwise(records, reader, run_root / "features")
        finally:
            reader.close()
    if args.stage in {"evaluate", "all"}:
        evaluate_layer_scan(
            run_root / "features",
            run_root / "report.json",
            device=args.device,
            retain_best_only=args.retain_best_only,
        )


if __name__ == "__main__":
    main()
