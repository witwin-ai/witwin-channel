# Copyright Xingyu Chen.
# Tests component layout.

import pytest
import torch

from witwin.core import ReceiverGrid, Scene
from tests.support.core_world import make_receiver_grid
from tests.support.scenes import single_wall_reflection_scene
from witwin.channel.montecarlo.bdpt import Config, solve


def _grid_at_x(x: float) -> ReceiverGrid:
    return make_receiver_grid(
        origin=torch.tensor([x, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(2, 3),
        spacing=(1.0, 0.5),
    )


def test_bdpt_component_maps_use_public_grid_layout_and_sum_to_total():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT component maps")

    base = single_wall_reflection_scene()
    scene = Scene(
        structures=[],
        endpoints=[
            *(endpoint for endpoint in base.endpoints if endpoint.role == "tx"),
            _grid_at_x(5.0),
        ],
    )

    result = solve(
        scene,
        Config(samples=128, seed=3, components={"los"}),
        reference_frequency_hz=3.0e9,
    )

    assert result.component_maps is not None
    assert result.component_maps["los"].shape == (1, 3, 2)
    assert result.component_maps["reflection"].shape == (1, 3, 2)
    assert result.component_maps["diffraction"].shape == (1, 3, 2)
    assert result.path_gain.shape == (1, 3, 2)
    torch.testing.assert_close(result.path_gain, result.component_maps["los"])