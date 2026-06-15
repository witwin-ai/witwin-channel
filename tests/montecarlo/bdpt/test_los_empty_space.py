import pytest
import torch

from tests.support.scenes import empty_space_los_scene, single_wall_reflection_scene
from witwin.channel_native.montecarlo.bdpt import Config, solve


def test_bdpt_los_empty_space_matches_analytic_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT LoS")

    scene = empty_space_los_scene()

    result = solve(scene, Config(samples=256, seed=7, components={"los"}))

    tx = torch.stack([item.position for item in scene.transmitters]).to(device=result.path_gain.device)
    tx_power = torch.tensor([item.power_w for item in scene.transmitters], device=result.path_gain.device)
    rx = torch.stack([item.position for item in scene.receivers]).to(device=result.path_gain.device)
    distance = torch.linalg.vector_norm(tx[:, None, :] - rx[None, :, :], dim=-1).clamp_min(1.0e-6)
    wavelength = 299_792_458.0 / scene.frequency
    expected = tx_power[:, None] / ((4.0 * torch.pi * distance / wavelength) ** 2)

    torch.testing.assert_close(result.path_gain, expected, rtol=1.0e-6, atol=1.0e-12)
    torch.testing.assert_close(result.component_power["los"], expected.sum(), rtol=1.0e-6, atol=1.0e-12)
    assert result.component_power["reflection"].item() == 0.0
    assert result.component_power["diffraction"].item() == 0.0


def test_bdpt_point_los_is_masked_by_native_visibility_when_blocked():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT LoS visibility")

    result = solve(single_wall_reflection_scene(), Config(samples=64, seed=5, components={"los"}))

    torch.testing.assert_close(result.path_gain, torch.zeros_like(result.path_gain), rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.component_power["los"], torch.zeros_like(result.component_power["los"]))
