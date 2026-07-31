#!/usr/bin/env python3
"""Reproduce the no-retraining Agent500 length and Math100 tests."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import load_proxy

from reader_provenance.data.records import load_records
from reader_provenance.data.release import DatasetRelease
from reader_provenance.experiments.stress import evaluate_external
from reader_provenance.features.io import save_features
from reader_provenance.models.proxy import ProxyConfig, ProxyReader


def _extract(
    records,
    proxy: dict,
    output: Path,
    *,
    device: str,
    response_tokens: int | None,
    early_exit: bool,
) -> None:
    reader = ProxyReader(
        ProxyConfig(
            model_name_or_path=proxy["model"],
            layer=int(proxy["layer"]),
            view=proxy["view"],
            max_length=int(proxy["max_length"]),
            batch_size=int(proxy["batch_size"]),
            device=device,
            dtype=proxy["dtype"],
            attention=proxy["attention"],
            early_exit=early_exit,
            max_response_tokens=response_tokens,
        )
    )
    try:
        save_features(output, reader.extract(records))
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-tag", required=True)
    parser.add_argument(
        "--condition",
        choices=("length32", "length64", "length128", "math100", "all"),
        default="all",
    )
    parser.add_argument(
        "--stage", choices=("extract", "evaluate", "all"), default="all"
    )
    parser.add_argument("--variant", default="100-way")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--proxy-config", type=Path, default=Path("configs/proxies.yaml")
    )
    parser.add_argument(
        "--enrollment-root",
        type=Path,
        default=Path("outputs/agent500"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stress"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-exit", action="store_true")
    args = parser.parse_args()

    release = DatasetRelease(args.data_root)
    proxy = load_proxy(args.proxy_config, args.proxy_tag)
    probes_dir = (
        args.enrollment_root
        / args.variant
        / args.proxy_tag
        / "artifacts"
        / "probes"
    )
    selected = (
        ("length32", "agent500", 32, "prompt_matched_out_of_fold"),
        ("length64", "agent500", 64, "prompt_matched_out_of_fold"),
        ("length128", "agent500", 128, "prompt_matched_out_of_fold"),
        ("math100", "math100", None, "fold_model_ensemble"),
    )
    for name, benchmark, token_limit, protocol in selected:
        if args.condition not in {name, "all"}:
            continue
        run_root = args.output_root / args.variant / args.proxy_tag / name
        feature_path = run_root / "features.npz"
        if args.stage in {"extract", "all"}:
            records = load_records(release.response_paths(args.variant, benchmark))
            _extract(
                records,
                proxy,
                feature_path,
                device=args.device,
                response_tokens=token_limit,
                early_exit=args.early_exit,
            )
        if args.stage in {"evaluate", "all"}:
            evaluate_external(
                feature_path,
                probes_dir,
                run_root / "report.json",
                protocol=protocol,
                budgets=(1, 5, 10, 20, 50, 100),
                grouping_seeds=(42, 43, 44),
            )


if __name__ == "__main__":
    main()
