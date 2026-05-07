"""Sequence-pooling classifiers over (N, M, D) hidden-state features.

Three trainable poolers that replace the naive mean-pool used in the rest of
the pipeline, plus a mean-pool LR baseline for parity. All four are evaluated
with the same StratifiedKFold protocol and report classification accuracy,
macro-F1, and (via the pooled embedding) pair-AUC + retrieval mAP@10.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from provenance_tracker.evaluation.metrics import pairwise_auc, retrieval_map


# ------------------------------ Models ----------------------------------- #


class SingleQueryAttn(nn.Module):
    """h: (B, M, D), q: (D,) -> softmax(h·q) weighted sum."""

    def __init__(self, dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.q = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.q, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, n_classes)

    def pool(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = h @ self.q  # (B, M)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=1)
        return (w.unsqueeze(-1) * h).sum(dim=1)  # (B, D)

    def forward(self, h, mask=None):
        z = self.pool(h, mask)
        return self.head(self.dropout(z)), z


class MultiHeadAttn(nn.Module):
    """H independent learnable queries; concat-project back to D."""

    def __init__(self, dim: int, n_classes: int, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.qs = nn.Parameter(torch.empty(n_heads, dim))
        nn.init.normal_(self.qs, std=0.02)
        self.proj = nn.Linear(n_heads * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, n_classes)

    def pool(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # scores: (B, H, M)
        scores = torch.einsum("bmd,hd->bhm", h, self.qs)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        w = F.softmax(scores, dim=-1)
        pooled = torch.einsum("bhm,bmd->bhd", w, h)  # (B, H, D)
        return self.proj(pooled.flatten(1))

    def forward(self, h, mask=None):
        z = self.pool(h, mask)
        return self.head(self.dropout(z)), z


class TransformerEncoderCls(nn.Module):
    """1-layer transformer encoder + learnable CLS token."""

    def __init__(
        self,
        dim: int,
        n_classes: int,
        n_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.3,
        n_layers: int = 1,
    ):
        super().__init__()
        self.cls = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, n_classes)

    def pool(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = h.size(0)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (B, 1+M, D)
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            key_padding_mask = ~mask  # True at padded positions
        else:
            key_padding_mask = None
        out = self.enc(h, src_key_padding_mask=key_padding_mask)
        return out[:, 0, :]

    def forward(self, h, mask=None):
        z = self.pool(h, mask)
        return self.head(self.dropout(z)), z


# ------------------------------ Trainer ---------------------------------- #


@dataclass
class FoldResult:
    accuracies: list[float] = field(default_factory=list)
    macro_f1s: list[float] = field(default_factory=list)
    pooled_emb: np.ndarray | None = None


def _per_channel_zscore(x_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std on (mean-pooled) training set, broadcast to (N,M,D)."""
    flat = x_tr.mean(axis=1)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0) + 1e-6
    return (x_tr - mu) / sigma, (x_te - mu) / sigma


def _train_one_fold(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    mask_tr: np.ndarray | None,
    mask_te: np.ndarray | None,
    *,
    device: str,
    epochs: int,
    lr: float,
    wd: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tensors = [torch.from_numpy(x_tr).float()]
    if mask_tr is not None:
        tensors.append(torch.from_numpy(mask_tr).bool())
    tensors.append(torch.from_numpy(y_tr).long())
    ds = TensorDataset(*tensors)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = [t.to(device) for t in batch]
            if mask_tr is not None:
                xb, mb, yb = batch
            else:
                xb, yb = batch
                mb = None
            opt.zero_grad()
            logits, _ = model(xb, mb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x_te).float().to(device)
        mt = torch.from_numpy(mask_te).bool().to(device) if mask_te is not None else None
        logits, z = model(xt, mt)
        preds = logits.argmax(-1).cpu().numpy()
        emb = z.cpu().numpy()
    return preds, emb


def kfold_evaluate(
    model_factory,
    features: np.ndarray,  # (N, M, D)
    labels: list[str],
    *,
    valid_counts: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
    device: str = "cuda",
    epochs: int = 60,
    lr: float = 1e-3,
    wd: float = 1e-3,
    batch_size: int = 64,
) -> FoldResult:
    classes = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_int = np.asarray([label_to_idx[l] for l in labels], dtype=np.int64)

    if valid_counts is not None:
        M = features.shape[1]
        mask_full = (np.arange(M)[None, :] < valid_counts[:, None]).astype(bool)
    else:
        mask_full = None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res = FoldResult(pooled_emb=np.zeros((len(y_int), features.shape[-1]), dtype=np.float32))

    for tr_idx, te_idx in skf.split(features, y_int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_tr, x_te = _per_channel_zscore(features[tr_idx], features[te_idx])
        m_tr = mask_full[tr_idx] if mask_full is not None else None
        m_te = mask_full[te_idx] if mask_full is not None else None
        model = model_factory()
        preds, emb = _train_one_fold(
            model, x_tr, y_int[tr_idx], x_te, m_tr, m_te,
            device=device, epochs=epochs, lr=lr, wd=wd, batch_size=batch_size,
        )
        res.accuracies.append(float(accuracy_score(y_int[te_idx], preds)))
        res.macro_f1s.append(float(f1_score(y_int[te_idx], preds, average="macro")))
        res.pooled_emb[te_idx] = emb
        del model
        torch.cuda.empty_cache()
    return res


def mean_pool_lr_baseline(
    features: np.ndarray,
    labels: list[str],
    *,
    valid_counts: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> FoldResult:
    """Mean-pool over M (mask-aware) then standard LR — same as evaluate.py."""
    if valid_counts is not None:
        M = features.shape[1]
        mask = (np.arange(M)[None, :] < valid_counts[:, None]).astype(np.float32)[..., None]
        pooled = (features * mask).sum(axis=1) / np.maximum(valid_counts[:, None], 1)
    else:
        pooled = features.mean(axis=1)
    classes = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_int = np.asarray([label_to_idx[l] for l in labels])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res = FoldResult(pooled_emb=pooled.astype(np.float32))
    for tr_idx, te_idx in skf.split(pooled, y_int):
        scaler = StandardScaler()
        xtr = scaler.fit_transform(pooled[tr_idx])
        xte = scaler.transform(pooled[te_idx])
        clf = LogisticRegression(max_iter=3000, solver="lbfgs")
        clf.fit(xtr, y_int[tr_idx])
        preds = clf.predict(xte)
        res.accuracies.append(float(accuracy_score(y_int[te_idx], preds)))
        res.macro_f1s.append(float(f1_score(y_int[te_idx], preds, average="macro")))
    return res


def evaluate_pooled_embedding(
    pooled_emb: np.ndarray, labels: list[str]
) -> dict[str, float]:
    """Run pair-AUC + retrieval mAP@10 on the pooled (N, D) embedding."""
    auc = pairwise_auc(pooled_emb, labels, n_splits=5, random_state=42)
    maps = retrieval_map(pooled_emb, labels, k_values=(10,))
    return {"mean_pairwise_auc": float(auc), "retrieval_map_at_10": float(maps[10])}
