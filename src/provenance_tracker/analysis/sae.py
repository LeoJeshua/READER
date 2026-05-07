"""Sparse autoencoder over best-layer proxy hidden states.

Trains an overcomplete autoencoder ``x → ReLU(Wx + b) → W'z → x̂`` with an L1
penalty on the code ``z``. After training we examine which SAE features
activate most differentially across target models:

* ``feature_class_mean`` — mean code activation per class per feature
* ``feature_importance`` — variance of class means (per feature), ranked
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class SAEConfig:
    input_dim: int
    expansion: int = 4        # overcomplete factor: D' = expansion * D
    l1_coef: float = 1e-3
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    device: str = "cuda"
    seed: int = 42


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_in, bias=True)
        # Tie decoder-bias-free, init small.
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z


@dataclass(slots=True)
class SAEResult:
    classes: list[str]
    feature_class_mean: np.ndarray   # (C, d_hidden)
    feature_importance: np.ndarray   # (d_hidden,) — variance across classes
    top_feature_idx: np.ndarray      # sorted descending
    recon_loss: float
    sparsity: float                  # average fraction of active features per sample


def train_sae(
    x: np.ndarray,
    labels: list[str],
    cfg: SAEConfig,
) -> tuple[SparseAutoencoder, SAEResult]:
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    d_hidden = cfg.input_dim * cfg.expansion
    model = SparseAutoencoder(cfg.input_dim, d_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    # Centering makes L1 behave; track stats for later use.
    mean = x_t.mean(dim=0, keepdim=True)
    std = x_t.std(dim=0, keepdim=True) + 1e-6
    x_norm = (x_t - mean) / std
    n = x_norm.shape[0]

    last_recon = 0.0
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        ep_recon = 0.0
        ep_l1 = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            batch = x_norm.index_select(0, idx)
            x_hat, z = model(batch)
            recon = ((x_hat - batch) ** 2).mean()
            l1 = z.abs().mean()
            loss = recon + cfg.l1_coef * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_recon += recon.item() * batch.shape[0]
            ep_l1 += l1.item() * batch.shape[0]
        last_recon = ep_recon / n
        if (epoch + 1) % 25 == 0:
            print(f"[sae] epoch {epoch + 1} recon={ep_recon / n:.4f} l1={ep_l1 / n:.4f}")

    model.eval()
    with torch.no_grad():
        z_all = model.encode(x_norm)  # (N, d_hidden)
        sparsity = float((z_all > 0).float().mean().item())

    classes = sorted(set(labels))
    y = np.asarray(labels)
    z_np = z_all.cpu().numpy()
    feature_class_mean = np.stack(
        [z_np[y == c].mean(axis=0) for c in classes], axis=0
    )  # (C, d_hidden)
    feature_importance = feature_class_mean.var(axis=0)
    order = np.argsort(feature_importance)[::-1]
    result = SAEResult(
        classes=classes,
        feature_class_mean=feature_class_mean.astype(np.float32),
        feature_importance=feature_importance.astype(np.float32),
        top_feature_idx=order,
        recon_loss=float(last_recon),
        sparsity=sparsity,
    )
    return model, result
