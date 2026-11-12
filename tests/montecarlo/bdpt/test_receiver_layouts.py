# Copyright Xingyu Chen.
# Tests BDPT on several receiver grids and on grids mixed with point receivers.

import pytest
import torch

import witwin.channel.experimental.bdpt as bdpt
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_receiver_grid,
    make_transmitter,
)
from witwin.channel.deployment import build_info
from witwin.core import MaterialLayer, PhysicalMaterial, Scene

_FREQUENCY = 3.0e9

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")


def _require_native() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")


def _wall():
    return make_mesh_structure(
        vertices=torch.tensor(
            [[-2.0, -4.0, -4.0], [-2.0, 4.0, -4.0], [-2.0, -4.0, 4.0], [-2.0, 4.0, 4.0]]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=PhysicalMaterial(
            layers=(MaterialLayer(thickness_m=0.1, eps_r=4.0, sigma_e=0.05),), name="lossy"
        ),
        name="wall",
        surface_id=1,
    )


def _receivers():
    return [
        make_receiver(position=torch.tensor([5.0, y, z]))
        for y, z in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, -1.0))
    ]


def _grid(origin_y: float, shape=(2, 3)):
    return make_receiver_grid(
        origin=torch.tensor([5.0, origin_y, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=shape,
        spacing=(0.5, 0.5),
    )


@cuda
@pytest.mark.parametrize("components", [{"los"}, {"los", "reflection"}])
def test_mixed_grid_and_point_receivers_match_separate_solves(components):
    _require_native()
    tx = make_transmitter(position=torch.tensor([0.0, 0.0, 0.0]))
    structures = [_wall()]
    grid_a = _grid(-1.0)
    grid_b = _grid(1.0, shape=(1, 2))
    points = _receivers()
    config = bdpt.Config(samples=64, max_depth=1, components=components)

    def run(*receivers):
        return bdpt.solve(
            Scene(structures=structures, endpoints=[tx, *receivers]),
            config,
            reference_frequency_hz=_FREQUENCY,
        )

    mixed = run(grid_a, *points, grid_b)
    assert mixed.component_maps is None
    assert mixed.path_gain.shape == (1, 6 + len(points) + 2)
    torch.testing.assert_close(
        mixed.path_gain[:, 6:10], run(*points).path_gain, rtol=1.0e-6, atol=0.0
    )
    # Each grid contributes its cells in the order the scene compiler expands it.
    for begin, grid, cells in ((0, grid_a, 6), (10, grid_b, 2)):
        alone = run(grid)
        torch.testing.assert_close(
            mixed.path_gain[:, begin:begin + cells].sum(),
            alone.path_gain.sum(),
            rtol=1.0e-6,
            atol=0.0,
        )
    two_grids = run(grid_a, grid_b)
    assert two_grids.path_gain.shape == (1, 8)
    torch.testing.assert_close(
        two_grids.path_gain, mixed.path_gain[:, [0, 1, 2, 3, 4, 5, 10, 11]], rtol=1.0e-6, atol=0.0
    )

