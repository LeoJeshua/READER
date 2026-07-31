from __future__ import annotations

import argparse
from pathlib import Path

from reader_provenance.reporting.paper import render_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--proxy-config", type=Path, default=Path("configs/proxies.yaml")
    )
    parser.add_argument(
        "--capabilities", type=Path, default=Path("configs/capabilities.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    render_all(
        args.results,
        args.proxy_config,
        args.capabilities,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
