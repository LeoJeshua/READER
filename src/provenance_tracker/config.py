"""Central registry of model paths and experiment constants.

All paths resolve to locally cached Hugging Face snapshots under
/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/.
No model is ever downloaded from the internet by this package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


HF_HUB_ROOT = Path("/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub")

# Force the shared cache — overriding any inherited HF_HUB_CACHE that points
# elsewhere (the sandbox image sets HF_HUB_CACHE=/data/.hub/ by default).
os.environ["HF_HOME"] = str(HF_HUB_ROOT.parent)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_ROOT)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def hub_dir(repo_id: str) -> Path:
    return HF_HUB_ROOT / f"models--{repo_id.replace('/', '--')}"


def snapshot_dir(repo_id: str) -> Path:
    """Return the most recently modified snapshot directory for a repo."""
    base = hub_dir(repo_id) / "snapshots"
    if not base.exists():
        raise FileNotFoundError(f"No snapshots for {repo_id} under {base}")
    snaps = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in snaps:
        if (snap / "config.json").exists():
            return snap
    return snaps[0]


@dataclass(frozen=True)
class TargetModelSpec:
    label: str
    repo_id: str
    family: str
    chat: bool = True


TARGET_MODELS: tuple[TargetModelSpec, ...] = (
    TargetModelSpec("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct", "qwen"),
    TargetModelSpec("qwen3_4b", "Qwen/Qwen3-4B-Instruct-2507", "qwen"),
    TargetModelSpec("qwen25_3b", "Qwen/Qwen2.5-3B-Instruct", "qwen"),
    TargetModelSpec("llama31_8b", "meta-llama/Llama-3.1-8B-Instruct", "llama"),
    TargetModelSpec("llama3_8b", "meta-llama/Meta-Llama-3-8B-Instruct", "llama"),
    TargetModelSpec("llama32_3b", "meta-llama/Llama-3.2-3B-Instruct", "llama"),
    TargetModelSpec("mistral_nemo", "mistralai/Mistral-Nemo-Instruct-2407", "mistral"),
    TargetModelSpec("phi3_mini", "microsoft/Phi-3-mini-4k-instruct", "phi"),
    TargetModelSpec("mistral7b_v03", "mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
)


PROXY_MODEL_REPO = "Qwen/Qwen3-8B"
MPNET_REPO = "sentence-transformers/all-mpnet-base-v2"

DEFAULT_DATASET_PATH = Path("/data/projects/LLM_Identifier/data/rand/rand_dataset.json")
DEFAULT_OUTPUTS_ROOT = Path("/data/projects/LLM_Identifier/outputs/exp")

PROBE_TEMPLATE = "Prompt:\n{prompt}\n\nResponse:\n{response}"


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path = DEFAULT_OUTPUTS_ROOT
    records_dir: Path = field(init=False)
    proxy_dir: Path = field(init=False)
    mpnet_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records_dir", self.root / "records")
        object.__setattr__(self, "proxy_dir", self.root / "proxy_features")
        object.__setattr__(self, "mpnet_dir", self.root / "mpnet_features")
        object.__setattr__(self, "reports_dir", self.root / "reports")
        object.__setattr__(self, "figures_dir", self.root / "figures")
        object.__setattr__(self, "tables_dir", self.root / "tables")

    def mkdirs(self) -> None:
        for p in (
            self.records_dir,
            self.proxy_dir,
            self.mpnet_dir,
            self.reports_dir,
            self.figures_dir,
            self.tables_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
