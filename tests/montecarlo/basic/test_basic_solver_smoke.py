import sys

import pytest
import torch

from witwin.channel import ReceiverPoint, Scene, Transmitter
from witwin.channel.core.kernels.extension import build_info
from witwin.channel.montecarlo.basic import Config, Result, solve


def _empty_space_scene() -> Scene:
    return Scene(
        structures=[],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]), power_w=2.0)],
        receivers=[ReceiverPoint(position=torch.tensor([3.0, 4.0, 0.0]))],
        frequency=3.0e9,
    )


def test_basic_solver_los_smoke_returns_cuda_result():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic solver")

    result = solve(_empty_space_scene(), Config(samples=128, seed=7, components={"los"}))

    assert isinstance(result, Result)
    assert result.path_gain.is_cuda
    assert result.path_gain.shape == (1, 1)
    assert torch.isfinite(result.path_gain).all()
    assert result.component_power["los"].is_cuda
    assert result.component_power["reflection"].item() == 0.0
    assert result.component_power["diffraction"].item() == 0.0


def test_basic_point_receiver_los_is_occluded_by_wall():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic solver")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native visibility is not built")

    from tests.support.scenes import single_wall_reflection_scene

    scene = single_wall_reflection_scene()
    result = solve(scene, Config(samples=128, seed=7, components={"los"}))

    assert result.path_gain.shape == (1, 1)
    assert result.path_gain.item() == 0.0
    assert result.component_power["los"].item() == 0.0


def test_basic_solver_does_not_import_python_rayd():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic solver")

    sys.modules.pop("rayd", None)

    solve(_empty_space_scene(), Config(samples=16, components={"los"}))

    assert "rayd" not in sys.modules


def test_basic_solver_reflection_component_requires_native_capability():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic solver")

    if build_info()["uses_rayd_native"]:
        result = solve(_empty_space_scene(), Config(components={"reflection"}))
        assert result.metadata["components"]["reflection"] == "enabled"
    else:
        with pytest.raises(RuntimeError, match="reflection.*RayD"):
            solve(_empty_space_scene(), Config(components={"reflection"}))
