import pytest
import torch

from tests.support.reference_channel import los_path_gain_reference
from tests.support.scenes import empty_space_los_scene
from witwin.channel.deterministic import Config, solve


def test_empty_space_los_matches_analytic_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic solver")

    scene = empty_space_los_scene()
    result = solve(scene, Config(max_depth=0, components={"los"}))
    expected = los_path_gain_reference(scene, device=torch.device("cuda"))

    assert result.path_gain.shape == expected.shape
    assert result.path_gain.is_cuda
    assert result.field.dtype == torch.complex64
    torch.testing.assert_close(result.path_gain, expected, rtol=1.0e-5, atol=1.0e-8)
    torch.testing.assert_close(result.field.abs().square(), expected, rtol=1.0e-5, atol=1.0e-8)
    torch.testing.assert_close(result.component_power["los"], expected, rtol=1.0e-5, atol=1.0e-8)
    assert result.paths is None
