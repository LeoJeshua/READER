#!/usr/bin/env python3
"""Run the canonical extraction and Agent500 evaluation for one proxy."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import load_proxy

from reader_provenance.data.records import load_records
from reader_provenance.data.release import DatasetRelease
from reader_provenance.experiments.agent500 import evaluate
from reader_provenance.features.io import save_features
from reader_provenance.models.proxy import ProxyConfig, ProxyReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-tag", required=True)
    parser.add_argument(
        "--stage",
        choices=("extract", "evaluate", "all"),
        default="all",
    )
    parser.add_argument(
        "--variant",
        choices=("50-way", "100-way", "165-way"),
        default="100-way",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--proxy-config",
        type=Path,
        default=Path("configs/proxies.yaml"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/agent500"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-exit", action="store_true")
    args = parser.parse_args()

    proxy = load_proxy(args.proxy_config, args.proxy_tag)
    run_root = args.output_root / args.variant / args.proxy_tag
    feature_path = run_root / "features.npz"
    if args.stage in {"extract", "all"}:
        release = DatasetRelease(args.data_root)
        records = load_records(release.response_paths(args.variant, "agent500"))
        reader = ProxyReader(
            ProxyConfig(
                model_name_or_path=proxy["model"],
                layer=int(proxy["layer"]),
                view=proxy["view"],
                max_length=int(proxy["max_length"]),
                batch_size=int(proxy["batch_size"]),
                device=args.device,
                dtype=proxy["dtype"],
                attention=proxy["attention"],
                early_exit=args.early_exit,
                max_response_tokens=None,
            )
        )
        try:
            save_features(feature_path, reader.extract(records))
        finally:
            reader.close()
    if args.stage in {"evaluate", "all"}:
        evaluate(
            feature_path,
            run_root / "report.json",
            run_root / "artifacts",
            component="dc-ac",
            budgets=(1, 5, 10, 20, 50, 100),
            grouping_seeds=(42, 43, 44),
            n_splits=5,
            split_seed=42,
            device=args.device,
        )


if __name__ == "__main__":
    main()
