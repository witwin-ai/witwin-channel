import pytest
import torch

from tests.support.scenes import empty_space_los_scene, single_wall_reflection_scene
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

    result = solve(single_wall_reflection_scene(), Config(components={"reflection"}))

    if build_info()["uses_raydn_native"]:
        assert result.metadata["components"]["reflection"] == "enabled"
    else:
        assert result.metadata["components"]["reflection"] == "capability-disabled"
        assert result.valid.numel() == 0


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
