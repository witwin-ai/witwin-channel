from __future__ import annotations

import math

import pytest
import torch

from witwin.channel.propagation.consumer._native import consumer_los_jones


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_SPEED_OF_LIGHT_M_S = 299_792_458.0


def _endpoints() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    source_positions = torch.tensor(
        [[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32
    )
    sink_positions = torch.tensor(
        [[0.0, 0.0, 2.0]], device="cuda", dtype=torch.float32
    )
    source_basis = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    sink_basis = torch.tensor(
        [[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    return source_positions, sink_positions, source_basis, sink_basis


def _free_space_scalar(
    *, frequency_hz: float, distance_m: float
) -> complex:
    wave_number = torch.tensor(
        2.0 * math.pi * frequency_hz / _SPEED_OF_LIGHT_M_S,
        dtype=torch.float32,
    ).item()
    phase = -math.fmod(wave_number * distance_m, 2.0 * math.pi)
    amplitude = 1.0 / (2.0 * wave_number * distance_m)
    return amplitude * complex(math.cos(phase), math.sin(phase))


def test_consumer_los_jones_matches_analytic_rotated_basis() -> None:
    source_positions, sink_positions, source_basis, sink_basis = _endpoints()
    frequency_hz = 1.0e9

    rows = consumer_los_jones(
        pair_index=torch.tensor([0], device="cuda", dtype=torch.int64),
        source_positions=source_positions,
        sink_positions=sink_positions,
        source_reference_basis=source_basis,
        sink_reference_basis=sink_basis,
        frequency_hz=frequency_hz,
    )

    scalar = _free_space_scalar(
        frequency_hz=frequency_hz, distance_m=2.0
    )
    expected = torch.tensor(
        [[[0.0j, scalar], [-scalar, 0.0j]]],
        device="cuda",
        dtype=torch.complex64,
    )
    torch.testing.assert_close(rows.matrix, expected, rtol=2.0e-5, atol=2.0e-7)
    torch.testing.assert_close(rows.source_basis, source_basis)
    torch.testing.assert_close(rows.sink_basis, sink_basis)
    assert rows.native_launch_count == 1


def test_consumer_los_jones_accepts_zero_rows_without_launch() -> None:
    source_positions, sink_positions, source_basis, sink_basis = _endpoints()

    rows = consumer_los_jones(
        pair_index=torch.empty((0,), device="cuda", dtype=torch.int64),
        source_positions=source_positions,
        sink_positions=sink_positions,
        source_reference_basis=source_basis,
        sink_reference_basis=sink_basis,
        frequency_hz=1.0e9,
    )

    assert rows.matrix.shape == (0, 2, 2)
    assert rows.matrix.dtype == torch.complex64
    assert rows.source_basis.shape == (0, 2, 3)
    assert rows.sink_basis.shape == (0, 2, 3)
    assert rows.native_launch_count == 0


def test_consumer_los_jones_rejects_ad_inputs() -> None:
    source_positions, sink_positions, source_basis, sink_basis = _endpoints()
    source_positions.requires_grad_()

    with pytest.raises(RuntimeError, match="does not support AD"):
        consumer_los_jones(
            pair_index=torch.tensor([0], device="cuda", dtype=torch.int64),
            source_positions=source_positions,
            sink_positions=sink_positions,
            source_reference_basis=source_basis,
            sink_reference_basis=sink_basis,
            frequency_hz=1.0e9,
        )
