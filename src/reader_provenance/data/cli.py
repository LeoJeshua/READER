from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from reader_provenance.data.release import DatasetRelease


def _materialize(
    release: DatasetRelease,
    *,
    variant: str,
    benchmark: str,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".gz":
        with output.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            ) as handle:
                for line in release.iter_jsonl_bytes(variant, benchmark):
                    handle.write(line)
        return
    with output.open("wb") as handle:
        for line in release.iter_jsonl_bytes(variant, benchmark):
            handle.write(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--full", action="store_true")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--variant", required=True)
    materialize.add_argument(
        "--benchmark", choices=("agent500", "math100"), required=True
    )
    materialize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = DatasetRelease(args.data_root)
    if args.command == "validate":
        print(json.dumps(release.validate(full=args.full), indent=2))
    else:
        _materialize(
            release,
            variant=args.variant,
            benchmark=args.benchmark,
            output=args.output,
        )


if __name__ == "__main__":
    main()
