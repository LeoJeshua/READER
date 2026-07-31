import math

import torch

from reader_provenance.features.dct import encode_padded_trajectories


def test_dc_is_exact_response_mean_for_variable_lengths() -> None:
    hidden = torch.tensor(
        [
            [[99.0], [1.0], [3.0], [5.0], [0.0]],
            [[2.0], [4.0], [8.0], [10.0], [12.0]],
        ]
    )
    starts = torch.tensor([1, 0])
    ends = torch.tensor([4, 5])
    encoded = encode_padded_trajectories(hidden, starts, ends, n_modes=2)
    assert torch.allclose(encoded[:, 0, 0], torch.tensor([3.0, 7.2]))


def test_first_ac_is_zero_for_constant_trajectory() -> None:
    hidden = torch.ones(3, 7, 4)
    starts = torch.tensor([0, 1, 3])
    ends = torch.tensor([7, 7, 7])
    encoded = encode_padded_trajectories(hidden, starts, ends, n_modes=2)
    assert torch.allclose(encoded[:, 0], torch.ones(3, 4))
    assert torch.allclose(encoded[:, 1], torch.zeros(3, 4), atol=1e-6)


def test_first_ac_has_length_normalized_scale() -> None:
    short = torch.tensor([0.0, 1.0]).reshape(1, 2, 1)
    long = torch.tensor([0.0, 0.0, 1.0, 1.0]).reshape(1, 4, 1)
    short_ac = encode_padded_trajectories(
        short, torch.tensor([0]), torch.tensor([2]), 2
    )[0, 1, 0]
    long_ac = encode_padded_trajectories(
        long, torch.tensor([0]), torch.tensor([4]), 2
    )[0, 1, 0]
    assert math.isclose(abs(short_ac.item()), abs(long_ac.item()), rel_tol=0.1)
