"""Primal-form CKA fingerprint utilities.

The metric used here is the **feature-space covariance cosine** proposed in
``docs/CKA.md``:

    sim(X, Y) = <C_X, C_Y>_F / (||C_X||_F * ||C_Y||_F)

where ``C_X = X_c^T X_c`` is the (centered) D x D auto second-moment of the
per-token activations.  This quantity does **not** require ``N_X == N_Y`` (no
row alignment), unlike Kornblith et al.'s linear CKA which equals
``||X_c^T Y_c||_F^2 / (||C_X||_F ||C_Y||_F)`` and needs paired samples.

We deliberately deviate from standard linear CKA so the metric works in our
black-box authorship setting where the per-(class, probe) sample is unique:
train and test fingerprints are built from *disjoint* records, so a paired
form is undefined.

For a sanity check, ``linear_cka_aligned`` returns the standard linear CKA
when ``N_X == N_Y``, used only in unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def compute_gram(
    X: torch.Tensor,
    *,
    center: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the D x D second-moment matrix C = X_c^T X_c.

    Args:
        X: (N, D) feature matrix.
        center: if True, subtract column means before forming the moment.
        dtype: precision used for the matmul (fp32 by default; pass
            ``torch.float64`` for the unit tests that demand strict numerical
            equivalence to the dual form).

    The result is intentionally **not** divided by N - only the cosine of two
    such matrices is reported downstream and the per-N scaling cancels.
    """
    if X.ndim != 2:
        raise ValueError(f"compute_gram expects (N, D); got {tuple(X.shape)}")
    Xf = X.to(dtype)
    if center:
        Xf = Xf - Xf.mean(dim=0, keepdim=True)
    return Xf.T @ Xf


def cov_cosine(C_a: torch.Tensor, C_b: torch.Tensor) -> torch.Tensor:
    """<C_a, C_b>_F / (||C_a||_F * ||C_b||_F).

    Returns a 0-dim tensor in [-1, 1] (in [0, 1] for PSD inputs).
    """
    num = (C_a * C_b).sum()
    den = torch.linalg.norm(C_a) * torch.linalg.norm(C_b)
    return num / den.clamp_min(1e-30)


def cov_cosine_batch(C_q: torch.Tensor, C_class: torch.Tensor) -> torch.Tensor:
    """Cosine of one query fingerprint against a stack of class fingerprints.

    Args:
        C_q:     (D, D) query fingerprint.
        C_class: (C, D, D) class fingerprints.

    Returns: (C,) similarities.
    """
    if C_class.ndim != 3:
        raise ValueError(f"C_class expected (C,D,D); got {tuple(C_class.shape)}")
    Cflat = C_class.reshape(C_class.shape[0], -1)
    qflat = C_q.reshape(-1)
    num = Cflat @ qflat
    den = torch.linalg.norm(Cflat, dim=1) * torch.linalg.norm(qflat)
    return num / den.clamp_min(1e-30)


def linear_cka_aligned(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Standard Kornblith linear CKA. Requires aligned rows (N_X == N_Y)."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"linear_cka_aligned requires aligned rows; got {X.shape}, {Y.shape}"
        )
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    cross = Xc.T @ Yc
    num = (cross * cross).sum()
    den = torch.linalg.norm(Xc.T @ Xc) * torch.linalg.norm(Yc.T @ Yc)
    return num / den.clamp_min(1e-30)


@dataclass(slots=True, frozen=True)
class WhitenSpec:
    top_k: int
    V: torch.Tensor  # (D, top_k); columns are the top principal directions to project OUT


def fit_top_k_pcs(
    X_pool: torch.Tensor,
    *,
    top_k: int,
    center: bool = True,
    niter: int = 6,
) -> WhitenSpec:
    """Fit a global SVD on a pooled feature matrix and keep the top-K right
    singular vectors. These columns span the subspace that ``whiten`` removes.

    Args:
        X_pool: (N_pool, D) features stacked across all classes / records used
            for the projector. Should cover topics broadly so the top
            components encode "topic / surface semantics".
        top_k: number of leading PCs to project out.
        center: subtract column mean before SVD.
        niter: subspace iteration count for ``torch.svd_lowrank``.
    """
    if X_pool.ndim != 2:
        raise ValueError(f"fit_top_k_pcs expects (N, D); got {tuple(X_pool.shape)}")
    Xf = X_pool.float()
    if center:
        Xf = Xf - Xf.mean(dim=0, keepdim=True)
    if top_k <= 0:
        D = Xf.shape[1]
        return WhitenSpec(top_k=0, V=torch.zeros(D, 0, device=Xf.device))
    # svd_lowrank gives U,S,V where V is (D, q). q must satisfy q <= min(N, D).
    q = min(top_k + 4, min(Xf.shape) - 1)
    _, _, V = torch.svd_lowrank(Xf, q=q, niter=niter)
    return WhitenSpec(top_k=top_k, V=V[:, :top_k].contiguous())


def whiten(X: torch.Tensor, spec: WhitenSpec) -> torch.Tensor:
    """Project features onto the orthogonal complement of ``spec.V``."""
    if spec.top_k == 0:
        return X
    if X.shape[-1] != spec.V.shape[0]:
        raise ValueError(
            f"whiten dim mismatch: X.D={X.shape[-1]}, V.D={spec.V.shape[0]}"
        )
    Vd = spec.V.to(device=X.device, dtype=X.dtype)
    proj = (X @ Vd) @ Vd.T
    return X - proj