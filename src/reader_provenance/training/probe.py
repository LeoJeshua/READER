from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(slots=True)
class LinearSourceProbe:
    weight: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    classes: list[str]

    def logits(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        standardized = (values - self.mean) / self.scale
        return standardized @ self.weight.T + self.bias

    def log_probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = self.logits(features).astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    def save(self, path: str | Path, *, held_out_prompts: list[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            weight=self.weight,
            bias=self.bias,
            mean=self.mean,
            scale=self.scale,
            classes=np.asarray(self.classes, dtype=object),
            held_out_prompts=np.asarray(held_out_prompts, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple[LinearSourceProbe, list[str]]:
        with np.load(Path(path), allow_pickle=True) as archive:
            probe = cls(
                weight=np.asarray(archive["weight"], dtype=np.float32),
                bias=np.asarray(archive["bias"], dtype=np.float32),
                mean=np.asarray(archive["mean"], dtype=np.float32),
                scale=np.asarray(archive["scale"], dtype=np.float32),
                classes=[str(value) for value in archive["classes"].tolist()],
            )
            held_out = [
                str(value) for value in archive["held_out_prompts"].tolist()
            ]
        return probe, held_out


def fit_source_probe(
    features: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    *,
    device: str = "cuda",
    learning_rate: float = 1e-3,
    max_steps: int = 40,
    schedule_horizon: int = 100,
    c_value: float = 1.0,
) -> tuple[LinearSourceProbe, dict[str, float | int]]:
    """Fit the canonical full-batch Adam multinomial linear probe."""
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("features and labels must be aligned 2-D/1-D arrays")
    if max_steps < 1 or schedule_horizon < 1 or c_value <= 0:
        raise ValueError("invalid optimizer configuration")
    torch_device = torch.device(device)
    x = torch.as_tensor(values, device=torch_device)
    y = torch.as_tensor(targets, dtype=torch.long, device=torch_device)
    mean = x.mean(dim=0, keepdim=True)
    scale = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / scale
    model = torch.nn.Linear(x.shape[1], len(classes), device=torch_device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=schedule_horizon,
        eta_min=learning_rate * 0.01,
    )
    l2_scale = 1.0 / (2.0 * c_value * len(values))
    objective_value = float("nan")
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        objective = torch.nn.functional.cross_entropy(model(x), y)
        objective = objective + l2_scale * model.weight.square().sum()
        objective.backward()
        optimizer.step()
        scheduler.step()
        objective_value = float(objective.detach().item())
    probe = LinearSourceProbe(
        weight=model.weight.detach().cpu().numpy().astype(np.float32),
        bias=model.bias.detach().cpu().numpy().astype(np.float32),
        mean=mean.detach().cpu().numpy().reshape(-1).astype(np.float32),
        scale=scale.detach().cpu().numpy().reshape(-1).astype(np.float32),
        classes=list(classes),
    )
    return probe, {"steps": max_steps, "final_objective": objective_value}
