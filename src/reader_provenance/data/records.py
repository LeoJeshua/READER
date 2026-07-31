from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    sample_id: str
    prompt: str
    response: str
    label: str
    model_name: str
    run_id: str = ""
    timestamp: str = ""
    probe_category: str = "rand"

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        allow_empty: bool = False,
    ) -> ResponseRecord:
        fields = {
            "sample_id": str(value["sample_id"]),
            "prompt": str(value["prompt"]),
            "response": str(value["response"]),
            "label": str(value["label"]),
            "model_name": str(value.get("model_name", value["label"])),
            "run_id": str(value.get("run_id", "")),
            "timestamp": str(value.get("timestamp", "")),
            "probe_category": str(value.get("probe_category", "rand")),
        }
        if not allow_empty and not fields["response"].strip():
            raise ValueError("response text must be non-empty")
        return cls(**fields)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_records(
    path: str | Path,
    *,
    allow_empty: bool = False,
) -> Iterator[ResponseRecord]:
    source = Path(path)
    with _open_text(source) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON"
                ) from error
            try:
                yield ResponseRecord.from_dict(value, allow_empty=allow_empty)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid response record"
                ) from error


def load_records(
    paths: list[str | Path],
    *,
    allow_empty: bool = False,
) -> list[ResponseRecord]:
    records = [
        record
        for path in paths
        for record in iter_records(path, allow_empty=allow_empty)
    ]
    if not records:
        raise ValueError("no response records were loaded")
    return records
