import pytest
import torch

from tests.support.reference_channel import los_path_gain_reference
from tests.support.scenes import empty_space_los_scene, single_wall_reflection_scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.montecarlo.basic import Config, solve


def test_basic_solver_matches_empty_space_los_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic parity")

    scene = empty_space_los_scene()
    result = solve(scene, Config(samples=256, seed=11, components={"los"}))
    expected = los_path_gain_reference(scene, device=torch.device("cuda"))

    torch.testing.assert_close(result.path_gain, expected, rtol=1e-5, atol=1e-8)


def test_single_wall_reflection_scene_is_capability_gated():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic parity")

    scene = single_wall_reflection_scene()
    if not build_info()["uses_raydn_native"]:
        with pytest.raises(RuntimeError, match="reflection requires RayDN native capability"):
            solve(scene, Config(samples=256, seed=11, components={"reflection"}))
        return

    result = solve(scene, Config(samples=256, seed=11, components={"reflection"}))
    status = result.metadata["components"]["reflection"]

    assert status == "enabled"
    assert result.component_power["reflection"].is_cuda
