# Copyright Xingyu Chen.
# Tests the host-side coincident-endpoint refusal for point receivers and receiver grids.

from __future__ import annotations

import math

import pytest
import torch
from witwin.core import AntennaState, ReceiverGrid, Scene
from witwin.core.identity import new_antenna_id

from witwin.channel.scene.endpoints import (
    _endpoint_views,
    _grid_samples_near,
    coincidence_tolerance_m,
    refuse_coincident_endpoints,
)

FREQUENCY_HZ = 3.5e9
FAR_FIELD_M = 299_792_458.0 / FREQUENCY_HZ / (4.0 * math.pi)


def _tx(position) -> AntennaState:
    return AntennaState(new_antenna_id(), "tx", torch.tensor(position))


def _rx(position) -> AntennaState:
    return AntennaState(new_antenna_id(), "rx", torch.tensor(position))


def _grid(
    origin=(-2.0, -2.0, 1.5), x_axis=(1.0, 0.0, 0.0), y_axis=(0.0, 1.0, 0.0), shape=(8, 8),
    spacing=(0.5, 0.5),
) -> ReceiverGrid:
    return ReceiverGrid(
        new_antenna_id(), torch.tensor(origin), torch.tensor(x_axis), torch.tensor(y_axis),
        shape, spacing,
    )


def _refuse(*endpoints) -> None:
    refuse_coincident_endpoints(_endpoint_views(Scene(endpoints=list(endpoints))), FREQUENCY_HZ)


def test_tolerance_is_the_far_field_radius_or_the_float32_resolution() -> None:
    assert coincidence_tolerance_m(FREQUENCY_HZ, 1.0) == pytest.approx(FAR_FIELD_M)
    assert coincidence_tolerance_m(FREQUENCY_HZ, 1.0e5) == pytest.approx(1.0e5 * 2.0**-20)


def test_grid_cell_on_the_transmitter_is_refused_by_cell_index() -> None:
    tx = _tx((0.0, 0.0, 1.5))
    grid = _grid()
    with pytest.raises(ValueError, match=r"ReceiverGrid \d+ cell \(row 4, column 4\)") as caught:
        _refuse(tx, grid)
    message = str(caught.value)
    assert f"transmitter {int(tx.antenna_id)}" in message
    assert "distance 0 m" in message
    assert "shift the grid origin" in message


def test_near_coincident_grid_cell_is_refused() -> None:
    with pytest.raises(ValueError, match=r"cell \(row 4, column 4\).*distance 0\.0001 m"):
        _refuse(_tx((0.0, 1.0e-4, 1.5)), _grid())


def test_cell_just_outside_the_far_field_radius_is_accepted() -> None:
    _refuse(_tx((0.0, 1.01 * FAR_FIELD_M, 1.5)), _grid())
    with pytest.raises(ValueError, match="coincides with transmitter"):
        _refuse(_tx((0.0, 0.99 * FAR_FIELD_M, 1.5)), _grid())


@pytest.mark.parametrize("position", [(0.0, 0.25, 1.5), (0.25, 0.0, 1.5), (0.25, 0.25, 1.5)])
def test_transmitter_on_a_grid_line_between_centres_is_accepted(position) -> None:
    _refuse(_tx(position), _grid())


def test_transmitter_off_the_grid_plane_is_accepted() -> None:
    _refuse(_tx((0.0, 0.0, 1.5 + 2.0 * FAR_FIELD_M)), _grid())


def test_second_transmitter_and_count_of_cells_are_named() -> None:
    far, near = _tx((10.0, 10.0, 10.0)), _tx((-1.0, 0.5, 1.5))
    fine = _grid(spacing=(0.004, 0.004), origin=(-1.02, 0.48, 1.5), shape=(11, 11))
    with pytest.raises(ValueError, match=rf"transmitter {int(near.antenna_id)} .*more cell"):
        _refuse(far, near, fine)


def test_point_receiver_near_the_transmitter_is_refused() -> None:
    with pytest.raises(ValueError, match=r"receiver \d+ at .* coincides with transmitter"):
        _refuse(_tx((1.0, 2.0, 3.0)), _rx((1.0, 2.0, 3.0 + 1.0e-5)))
    _refuse(_tx((1.0, 2.0, 3.0)), _rx((1.0, 2.0, 3.0 + 2.0 * FAR_FIELD_M)))


def test_cuda_resident_coordinates_are_not_read() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    tx = AntennaState(new_antenna_id(), "tx", torch.tensor((0.0, 0.0, 1.5), device="cuda"))
    _refuse(tx, _grid())


def _brute_force(origin, steps, shape, point, tolerance):
    rows = torch.arange(shape[0], dtype=torch.float64)[:, None, None]
    cols = torch.arange(shape[1], dtype=torch.float64)[None, :, None]
    cells = origin + rows * steps[0] + cols * steps[1]
    distance = torch.linalg.vector_norm(cells - point, dim=-1)
    inside = distance <= tolerance
    if not bool(inside.any()):
        return 0, None
    flat = int(torch.argmin(distance))
    return int(inside.sum()), divmod(flat, shape[1])


@pytest.mark.parametrize("seed", range(40))
def test_analytic_search_matches_materializing_the_grid(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)

    def uniform(*size):
        return torch.rand(size, generator=generator, dtype=torch.float64)

    shape = (int(1 + 12 * uniform(1)), int(1 + 12 * uniform(1)))
    origin = 4.0 * uniform(3) - 2.0
    steps = 0.3 * (uniform(2, 3) - 0.5)
    if seed % 5 == 0:
        steps[1] = 2.0 * steps[0]
    row = int(uniform(1) * shape[0])
    col = int(uniform(1) * shape[1])
    tolerance = 0.05 * float(uniform(1)) + 1.0e-4
    jitter = (uniform(3) - 0.5) * (3.0 * tolerance if seed % 2 else 0.0)
    point = origin + row * steps[0] + col * steps[1] + jitter
    count, nearest = _grid_samples_near(origin, steps, shape, point, tolerance)
    expected_count, expected_nearest = _brute_force(origin, steps, shape, point, tolerance)
    assert count == expected_count
    if expected_nearest is None:
        assert nearest is None
    else:
        assert nearest is not None
        assert (nearest[0], nearest[1]) == expected_nearest
