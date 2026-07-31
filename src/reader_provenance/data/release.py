from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class DatasetRelease:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest_path = self.root / "manifests" / "release.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported dataset release schema")

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(self.manifest["variants"])

    def labels(self, variant: str) -> list[str]:
        try:
            return list(self.manifest["variants"][variant]["labels"])
        except KeyError as error:
            raise ValueError(
                f"unknown variant {variant!r}; choose from {self.variants}"
            ) from error

    def response_paths(self, variant: str, benchmark: str) -> list[Path]:
        if benchmark not in {"agent500", "math100"}:
            raise ValueError("benchmark must be agent500 or math100")
        return [
            self.root / self.manifest["responses"][label][benchmark]["path"]
            for label in self.labels(variant)
        ]

    def iter_jsonl_bytes(self, variant: str, benchmark: str) -> Iterator[bytes]:
        for path in self.response_paths(variant, benchmark):
            with gzip.open(path, "rb") as handle:
                yield from handle

    def bench_a_models(self) -> list[dict[str, Any]]:
        path = self.root / self.manifest["bench_a"]["model_roster"]["path"]
        return json.loads(path.read_text(encoding="utf-8"))

    def bench_a_response_paths(self) -> list[Path]:
        responses = self.manifest["bench_a"]["responses"]
        return [
            self.root / responses[row["model"]]["path"]
            for row in self.bench_a_models()
        ]

    def bench_a_pairs_path(self) -> Path:
        return self.root / self.manifest["bench_a"]["main_pairs"]["path"]

    def bench_a_disjoint_pairs_path(self) -> Path:
        return self.root / self.manifest["bench_a"]["disjoint_pairs"]["path"]

    def bench_a_split_path(self, protocol: str) -> Path:
        try:
            relative = self.manifest["bench_a"]["splits"][protocol]["path"]
        except KeyError as error:
            choices = tuple(self.manifest["bench_a"]["splits"])
            raise ValueError(
                f"unknown Bench-A split {protocol!r}; choose from {choices}"
            ) from error
        return self.root / relative

    @staticmethod
    def _validate_gzip(
        path: Path,
        expected: dict[str, Any],
        *,
        full: bool,
    ) -> int:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected["gzip_sha256"]:
            raise ValueError(f"compressed checksum mismatch: {path}")
        if not full:
            return int(expected["records"])
        raw_digest = hashlib.sha256()
        count = 0
        with gzip.open(path, "rb") as handle:
            for line in handle:
                json.loads(line)
                raw_digest.update(line)
                count += 1
        if count != expected["records"]:
            raise ValueError(f"record count mismatch: {path}")
        if raw_digest.hexdigest() != expected["jsonl_sha256"]:
            raise ValueError(f"JSONL checksum mismatch: {path}")
        return count

    def _validate_plain_file(self, entry: dict[str, Any], digest_key: str) -> None:
        path = self.root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry[digest_key]:
            raise ValueError(f"checksum mismatch: {path}")

    def validate(self, *, full: bool = False) -> dict[str, Any]:
        files = 0
        dynamic_records = 0
        for _label, benchmarks in self.manifest["responses"].items():
            for _benchmark, expected in benchmarks.items():
                path = self.root / expected["path"]
                dynamic_records += self._validate_gzip(
                    path,
                    expected,
                    full=full,
                )
                files += 1

        bench_a = self.manifest["bench_a"]
        bench_a_records = 0
        for expected in bench_a["responses"].values():
            bench_a_records += self._validate_gzip(
                self.root / expected["path"],
                expected,
                full=full,
            )
            files += 1
        for key in ("prompt_panel", "main_pairs", "disjoint_pairs"):
            expected = bench_a[key]
            self._validate_gzip(self.root / expected["path"], expected, full=full)
            files += 1
        self._validate_plain_file(bench_a["model_roster"], "sha256")
        files += 1
        for entry in bench_a["splits"].values():
            self._validate_plain_file(entry, "sha256")
            files += 1
        return {
            "targets": len(self.manifest["responses"]),
            "bench_a_models": bench_a["models"],
            "files": files,
            "dynamic_records": dynamic_records,
            "bench_a_records": bench_a_records,
            "records": dynamic_records + bench_a_records,
            "full_validation": full,
        }
