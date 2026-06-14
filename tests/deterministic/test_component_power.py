import pytest
import torch

from tests.support.scenes import same_side_wall_reflection_scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.deterministic import Config, solve


def test_single_component_power_matches_total_for_reflection_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    result = solve(same_side_wall_reflection_scene(), Config(components={"reflection"}, coherent=False))

    torch.testing.assert_close(result.component_power["reflection"], result.path_gain, rtol=5.0e-4, atol=1.0e-8)
    assert torch.count_nonzero(result.component_power["los"]) == 0
    assert torch.count_nonzero(result.component_power["diffraction"]) == 0
