"""Length-normalized low-order DCT encoding for response trajectories."""
from __future__ import annotations

import math

import torch


def length_normalized_dct_basis(
    starts: torch.Tensor,
    ends: torch.Tensor,
    sequence_length: int,
    n_modes: int = 2,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return per-response DCT-II weights with shape ``(B, Q, T)``.

    The conventional orthonormal coefficients are divided by ``sqrt(N)``.
    Mode zero is therefore exactly the arithmetic token mean, while every mode
    remains invariant to trajectory length under a constant-amplitude signal.
    Positions outside ``[starts[b], ends[b])`` receive zero weight.
    """
    if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
        raise ValueError("starts and ends must be equal-length vectors")
    if sequence_length < 1 or n_modes < 1:
        raise ValueError("sequence_length and n_modes must be positive")
    lengths = ends - starts
    if torch.any(starts < 0) or torch.any(ends > sequence_length):
        raise ValueError("response spans fall outside the padded sequence")
    if torch.any(lengths < 1):
        raise ValueError("every response span must contain at least one token")

    positions = torch.arange(sequence_length, device=starts.device)
    relative = positions[None, :] - starts[:, None]
    valid = (relative >= 0) & (relative < lengths[:, None])
    modes = torch.arange(n_modes, device=starts.device)
    phase = (
        torch.pi
        * (relative.to(torch.float32)[:, None, :] + 0.5)
        * modes.to(torch.float32)[None, :, None]
        / lengths.to(torch.float32)[:, None, None]
    )
    basis = torch.cos(phase)
    if n_modes > 1:
        basis[:, 1:] *= math.sqrt(2.0)
    basis /= lengths.to(torch.float32)[:, None, None]
    basis *= valid[:, None, :]
    basis *= modes[None, :, None] < lengths[:, None, None]
    return basis.to(dtype=dtype)


def encode_padded_trajectories(
    hidden_states: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    n_modes: int = 2,
) -> torch.Tensor:
    """Encode padded hidden-state trajectories as ``(B, Q, D)`` fingerprints."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape (batch, tokens, hidden)")
    if hidden_states.shape[0] != starts.shape[0]:
        raise ValueError("batch and response-span counts differ")
    basis = length_normalized_dct_basis(
        starts,
        ends,
        hidden_states.shape[1],
        n_modes,
        dtype=hidden_states.dtype,
    )
    return torch.einsum("bqt,btd->bqd", basis, hidden_states)
