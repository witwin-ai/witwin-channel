import pytest
import torch

from tests.support.reference_channel import los_path_gain_reference
from witwin.channel_native import ReceiverGrid, Scene, Transmitter
from witwin.channel_native.deterministic import Config, solve


def _grid_scene() -> Scene:
    return Scene(
        structures=[],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]), power_w=1.5)],
        receivers=[
            ReceiverGrid(
                origin=torch.tensor([3.0, -1.0, -0.5]),
                x_axis=torch.tensor([0.0, 1.0, 0.0]),
                y_axis=torch.tensor([0.0, 0.0, 1.0]),
                shape=(2, 3),
                spacing=(1.0, 0.5),
            )
        ],
        frequency=3.0e9,
    )


def test_receiver_grid_uses_mc_basic_public_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic solver")

    scene = _grid_scene()
    grid = scene.receivers[0]
    result = solve(scene, Config(max_depth=0, components={"los"}))
    expected_flat = los_path_gain_reference(scene, device=torch.device("cuda"))
    expected = expected_flat.reshape(len(scene.transmitters), *grid.shape).transpose(1, 2).contiguous()

    assert result.path_gain.shape == (1, 3, 2)
    torch.testing.assert_close(result.path_gain, expected, rtol=1.0e-5, atol=1.0e-8)
    for tensor in result.component_power.values():
        assert tensor.shape == result.path_gain.shape
    for tensor in result.component_fields.values():
        assert tensor.shape == result.path_gain.shape


def test_return_field_false_uses_zero_sized_complex_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic solver")

    scene = _grid_scene()
    result = solve(scene, Config(max_depth=0, components={"los"}, return_field=False))

    assert result.field.dtype == torch.complex64
    assert result.field.numel() == 0
    for tensor in result.component_fields.values():
        assert tensor.dtype == torch.complex64
        assert tensor.numel() == 0
