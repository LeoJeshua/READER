from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_proxy(config_path: Path, tag: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved = dict(config.get("defaults", {}))
    for row in config["models"]:
        if row["tag"] == tag:
            resolved.update(row)
            return resolved
    raise ValueError(f"unknown proxy tag: {tag}")
