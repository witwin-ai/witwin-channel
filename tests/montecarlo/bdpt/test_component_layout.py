import pytest
import torch

from witwin.channel_native import ReceiverGrid
from tests.support.scenes import single_wall_reflection_scene
from witwin.channel_native.montecarlo.bdpt import Config, solve


def _grid_at_x(x: float) -> ReceiverGrid:
    return ReceiverGrid(
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
    scene = type(base)(
        structures=[],
        transmitters=base.transmitters,
        receivers=[_grid_at_x(5.0)],
        frequency=base.frequency,
    )

    result = solve(scene, Config(samples=128, seed=3, components={"los"}))

    assert result.component_maps is not None
    assert result.component_maps["los"].shape == (1, 3, 2)
    assert result.component_maps["reflection"].shape == (1, 3, 2)
    assert result.component_maps["diffraction"].shape == (1, 3, 2)
    assert result.path_gain.shape == (1, 3, 2)
    torch.testing.assert_close(result.path_gain, result.component_maps["los"])
