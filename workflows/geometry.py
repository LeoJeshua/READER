#!/usr/bin/env python3
"""Rebuild the Agent500 source-signature geometry from response-only caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from reader_provenance.experiments.geometry import analyze_agent500

PAPER_CANDIDATE_PAIRS = (
    ("ds_distill_qwen_32b", "qwq_32b"),
    ("minimax_m3", "claude_sonnet46"),
    ("claude_sonnet4", "deepseek_v4_pro"),
    ("gemma4_31b", "gemini31_flash_lite"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="100-way")
    parser.add_argument("--role", choices=("main", "full"), default="full")
    parser.add_argument(
        "--proxy-config", type=Path, default=Path("configs/proxies.yaml")
    )
    parser.add_argument(
        "--feature-root", type=Path, default=Path("outputs/input-ablation")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/geometry"))
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    config = yaml.safe_load(args.proxy_config.read_text(encoding="utf-8"))
    models = [row for row in config["models"] if args.role in row.get("roles", [])]
    features = {
        str(row["tag"]): args.feature_root
        / args.variant
        / str(row["tag"])
        / "response_only"
        / "features.npz"
        for row in models
    }
    analyze_agent500(
        features=features,
        roster_path=Path("data/rosters") / f"{args.variant}.json",
        proxy_config=args.proxy_config,
        output_dir=args.output_dir / args.variant / args.role,
        display_proxy="qwen35_9b",
        candidate_pairs=list(PAPER_CANDIDATE_PAIRS),
        min_family_size=5,
        prompt_splits=5,
        permutations=args.permutations,
        bootstrap_samples=args.bootstrap_samples,
        seed=42,
    )


if __name__ == "__main__":
    main()
