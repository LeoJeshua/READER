#!/usr/bin/env python3
"""Validate the public release as an isolated, checksum-stable artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from pathlib import Path

from reader_provenance.data.release import DatasetRelease
from reader_provenance.reporting.tables import write_endpoint_table

FORBIDDEN_TEXT = re.compile(
    r"/data/"
    r"projects/|/mnt/"
    r"shared-storage|"
    r"(?:sk-[A-Za-z0-9]{16,})|(?:hf_[A-Za-z0-9]{16,})|"
    r"(?:api[_-]?key\s*[:=]\s*[\"']?"
    r"(?!EMPTY\b|NONE\b|PLACEHOLDER\b)[A-Za-z0-9_-]{20,})|"
    r"(?:https?://(?:35|100)\.)",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
LOCAL_ONLY_PREFIXES = (
    ("private",),
    ("cache",),
    ("outputs",),
    ("artifacts",),
    (".venv",),
    ("mmlu_pro_api", "results"),
    ("visualization", "node_modules"),
    ("visualization", "dist"),
    ("visualization", ".vite"),
    ("visualization", "playwright-report"),
    ("visualization", "test-results"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_results(root: Path) -> dict[str, int]:
    directory = root / "results"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["files"]
    for row in rows:
        path = directory / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"result size mismatch: {path}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"result checksum mismatch: {path}")
    if len(rows) != manifest["file_count"]:
        raise ValueError("result manifest file_count differs from file list")
    return {"files": len(rows), "bytes": sum(row["bytes"] for row in rows)}


def validate_nested_rosters(root: Path) -> dict[str, int]:
    rosters = {
        size: json.loads(
            (root / f"data/rosters/{size}-way.json").read_text(encoding="utf-8")
        )
        for size in (50, 100, 165)
    }
    if rosters[100][:50] != rosters[50] or rosters[165][:100] != rosters[100]:
        raise ValueError("50/100/165-way rosters are not exact ordered prefixes")
    return {f"{size}-way": len(rows) for size, rows in rosters.items()}


def validate_release_boundary(root: Path) -> dict[str, int]:
    scanned = 0
    symlinks = []
    violations = []
    ignored_parts = {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(
            relative.parts[: len(prefix)] == prefix
            for prefix in LOCAL_ONLY_PREFIXES
        ):
            continue
        if any(part in ignored_parts for part in relative.parts):
            continue
        if path.is_symlink():
            symlinks.append(relative.as_posix())
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        match = FORBIDDEN_TEXT.search(text)
        if match:
            violations.append(f"{relative}:{match.group(0)}")
    if symlinks:
        raise ValueError(f"public release contains symlinks: {symlinks[:5]}")
    if violations:
        raise ValueError(f"public release boundary violations: {violations[:5]}")
    return {"text_files_scanned": scanned, "symlinks": 0}


def validate_paper_endpoints(root: Path) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "endpoints.csv"
        write_endpoint_table(root / "results", root / "configs/proxies.yaml", output)
        rows = {
            row["method"]: row for row in csv.DictReader(output.open(encoding="utf-8"))
        }
    checks = {
        "qwen35_9b_acc_k1": (
            "READER / Qwen3.5-9B",
            "accuracy_k1",
            0.50386,
        ),
        "qwen35_9b_acc_k100": (
            "READER / Qwen3.5-9B",
            "accuracy_k100",
            0.962,
        ),
        "ft_dna_qwen_f1_k1": (
            "FT DNA / Qwen-Emb.",
            "macro_f1_k1",
            0.530229,
        ),
    }
    values = {}
    for name, (method, field, expected) in checks.items():
        actual = float(rows[method][field])
        if abs(actual - expected) > 5e-7:
            raise ValueError(f"paper endpoint changed: {name}={actual}")
        values[name] = actual
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--full-data", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    report = {
        "schema_version": 1,
        "rosters": validate_nested_rosters(root),
        "data": DatasetRelease(root / "data").validate(full=args.full_data),
        "results": validate_results(root),
        "boundary": validate_release_boundary(root),
        "paper_endpoints": validate_paper_endpoints(root),
        "complete": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
