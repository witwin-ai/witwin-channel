import pytest
import torch

from tests.support.scenes import (
    empty_space_los_scene,
    same_side_wall_reflection_scene,
    single_wall_reflection_scene,
    wedge_diffraction_scene,
)
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.path import Config, Result, solve


def test_path_solver_empty_space_los_returns_one_path_per_pair():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    scene = empty_space_los_scene()
    result = solve(scene, Config(components={"los"}))

    assert isinstance(result, Result)
    assert result.valid.is_cuda
    assert result.valid.tolist() == [True, True, True, True]
    assert result.tx_id.tolist() == [0, 1, 0, 1]
    assert result.rx_id.tolist() == [0, 0, 1, 1]
    assert result.depth.tolist() == [0, 0, 0, 0]
    assert result.component_id.tolist() == [0, 0, 0, 0]
    assert torch.all(result.path_length_m > 0)
    assert torch.all(result.delay_s > 0)
    assert torch.all(result.path_gain > 0)


def test_path_solver_reflection_is_capability_gated():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    if not build_info()["uses_raydn_native"]:
        with pytest.raises(RuntimeError, match="reflection paths require RayDN native capability"):
            solve(single_wall_reflection_scene(), Config(components={"reflection"}))
    else:
        result = solve(single_wall_reflection_scene(), Config(components={"reflection"}))
        assert result.metadata["components"]["reflection"] == "enabled"
        assert result.valid.numel() == 0


def test_path_solver_exports_native_reflection_paths_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    result = solve(same_side_wall_reflection_scene(), Config(components={"reflection"}))

    assert result.metadata["components"]["reflection"] == "enabled"
    assert int((result.component_id == 1).sum().item()) >= 1
    assert torch.all(result.depth[result.component_id == 1] == 1)
    assert torch.all(result.path_gain[result.component_id == 1] > 0)


def test_path_solver_exports_native_diffraction_paths_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    result = solve(wedge_diffraction_scene(), Config(components={"diffraction"}))

    assert result.metadata["components"]["diffraction"] == "enabled"
    assert int((result.component_id == 2).sum().item()) >= 1
    assert torch.all(result.edge_id[result.component_id == 2] >= 0)
    assert torch.all(result.path_gain[result.component_id == 2] > 0)


def test_path_solver_calls_kernel_facade(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    calls = []
    original = ops.path_los_export

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(ops, "path_los_export", wrapped)

    solve(empty_space_los_scene(), Config(components={"los"}))

    assert len(calls) == 1
