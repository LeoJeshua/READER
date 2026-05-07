from __future__ import annotations

import json
from pathlib import Path

from .schemas import ProbeSample


DEFAULT_DATASET_PATH = Path("/data/projects/LLM_Identifier/data/rand/rand_dataset.json")


def load_rand_probes(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    limit: int | None = None,
    offset: int = 0,
) -> list[ProbeSample]:
    """Load probe prompts from JSON.

    ``sample_id`` always reflects the **absolute** index in the dataset file
    (``rand-{idx:06d}``) regardless of ``offset`` so that records produced
    across multiple resumed runs can be concatenated without collisions.
    """
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    end = (offset + limit) if limit is not None else len(items)
    end = min(end, len(items))

    probes: list[ProbeSample] = []
    for idx in range(offset, end):
        probes.append(
            ProbeSample(
                sample_id=f"rand-{idx:06d}",
                prompt=str(items[idx]).strip(),
                probe_category="rand",
            )
        )
    return probes
