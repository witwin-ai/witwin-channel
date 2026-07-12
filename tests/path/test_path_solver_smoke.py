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
from witwin.channel_native.path import Config, InteractionType, PathResult, solve


def test_path_solver_empty_space_los_returns_one_path_per_pair():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    scene = empty_space_los_scene()
    result = solve(scene, Config(components={"los"}))

    assert isinstance(result, PathResult)
    assert result.valid.is_cuda
    assert int(result.valid.sum()) == 4
    assert torch.all(result.num_paths == 1)
    assert torch.all(result.path_length_m[result.valid] > 0)
    assert torch.all(result.tau[result.valid] > 0)
    assert torch.all(result.a[result.valid].abs() > 0)


def test_path_solver_reflection_is_capability_gated():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    if not build_info()["uses_raydn_native"]:
        with pytest.raises(RuntimeError, match="reflection paths require RayDN native capability"):
            solve(single_wall_reflection_scene(), Config(components={"reflection"}))
    else:
        result = solve(single_wall_reflection_scene(), Config(components={"reflection"}))
        assert result.metadata["components"]["reflection"] == "enabled"
        assert int(result.valid.sum()) == 0


def test_path_solver_exports_native_reflection_paths_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    result = solve(same_side_wall_reflection_scene(), Config(components={"reflection"}))

    assert result.metadata["components"]["reflection"] == "enabled"
    reflection = result.valid & (result.interaction_type == int(InteractionType.REFLECTION)).any(dim=-1)
    assert int(reflection.sum().item()) >= 1
    assert torch.all(result.a[reflection].abs() > 0)


def test_path_solver_exports_native_diffraction_paths_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    result = solve(wedge_diffraction_scene(), Config(components={"diffraction"}))

    assert result.metadata["components"]["diffraction"] == "enabled"
    diffraction = result.valid & (result.interaction_type == int(InteractionType.DIFFRACTION)).any(dim=-1)
    assert int(diffraction.sum().item()) >= 1
    assert torch.all(result.primitive_id[diffraction, 0] >= 0)
    assert torch.all(result.a[diffraction].abs() > 0)


def test_path_solver_accepts_transmission_and_scattering_as_empty_plumbing():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    scene = empty_space_los_scene()
    result = solve(
        scene, Config(components={"los", "transmission", "scattering"})
    )

    # (a) validates and the LoS paths still export, (b) no transmission (5) or
    # scattering (6) paths are produced in v1.
    assert int(result.valid.sum()) == 4
    assert not (result.interaction_type == int(InteractionType.TRANSMISSION)).any()
    assert not (result.interaction_type == int(InteractionType.SCATTERING)).any()
    # (c) truthful requested-but-empty metadata status.
    assert result.metadata["components"]["transmission"] == "enabled_no_paths"
    assert result.metadata["components"]["scattering"] == "enabled_no_paths"
    assert result.metadata["components"]["los"] == "enabled"


def test_path_config_rejects_unknown_component():
    with pytest.raises(ValueError, match="components"):
        Config(components={"los", "teleportation"})


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
