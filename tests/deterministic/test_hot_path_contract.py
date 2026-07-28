import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from tests.support.scenes import wedge_diffraction_scene
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config, solve
from witwin.channel.kernels import fields as field_kernels
from witwin.channel.kernels import topology as topology_kernels
import witwin.channel.path as path_package


def test_los_hot_path_uses_channel_kernel_facade(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic hot-path contract")

    calls = []
    original = topology_kernels.path_los_export

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(topology_kernels, "path_los_export", wrapped)

    solve(
        empty_space_los_scene(),
        Config(max_depth=0, components={"los"}),
        reference_frequency_hz=3.0e9,
    )

    assert len(calls) == 1


def test_los_hot_path_uses_native_topology_facade(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic hot-path contract")

    calls = []
    original = topology_kernels.deterministic_los_topology_block

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        topology_kernels, "deterministic_los_topology_block", wrapped
    )

    solve(
        empty_space_los_scene(),
        Config(max_depth=0, components={"los"}),
        reference_frequency_hz=3.0e9,
    )

    assert len(calls) == 1


def test_diffraction_hot_path_uses_native_vector_field_facade(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic hot-path contract")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    calls = []
    original = field_kernels.deterministic_diffraction_vector_field

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        field_kernels, "deterministic_diffraction_vector_field", wrapped
    )

    solve(
        wedge_diffraction_scene(),
        Config(max_depth=1, components={"diffraction"}),
        reference_frequency_hz=3.0e9,
    )

    assert len(calls) > 0


def test_deterministic_solver_does_not_call_path_solver_orchestration(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic hot-path contract")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "deterministic solver must not call witwin.channel.path.solve"
        )

    monkeypatch.setattr(path_package, "solve", forbidden)

    solve(
        empty_space_los_scene(),
        Config(max_depth=0, components={"los"}),
        reference_frequency_hz=3.0e9,
    )
