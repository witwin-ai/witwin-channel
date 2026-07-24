import pytest
import torch

from witwin.channel.runtime import symbols as ops
from tests.support.scenes import same_side_wall_reflection_scene
from witwin.channel.core.kernels.extension import build_info
from witwin.channel.deterministic import Config, solve
from witwin.channel.propagation.fields.kernels import (
    deterministic as deterministic_fields,
)
from witwin.channel.propagation.enumerated import reflection as topology
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as solve_paths


def test_single_plane_reflection_matches_path_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    scene = same_side_wall_reflection_scene()
    result = solve(
        scene,
        Config(
            components={"reflection"},
            coherent=False,
            export_paths=True,
            return_field=False,
        ),
        reference_frequency_hz=3.0e9,
    )
    reference = solve_paths(
        scene, PathConfig(components={"reflection"}), reference_frequency_hz=3.0e9
    )

    assert result.paths is not None
    assert result.paths.valid.numel() == reference.valid.numel()
    torch.testing.assert_close(
        result.paths.path_length_m,
        reference.tau[reference.valid] * 299_792_458.0,
        rtol=1.0e-5,
        atol=1.0e-8,
    )
    expected_field = reference.a[..., 0][reference.valid]
    expected_gain = expected_field.abs().square()
    torch.testing.assert_close(
        result.paths.path_gain, expected_gain, rtol=5.0e-4, atol=1.0e-10
    )
    torch.testing.assert_close(
        result.paths.coefficient, expected_field, rtol=5.0e-4, atol=1.0e-7
    )
    torch.testing.assert_close(
        result.path_gain.reshape(-1).sum(),
        expected_gain.sum(),
        rtol=5.0e-4,
        atol=1.0e-10,
    )


def test_reflection_path_field_export_uses_native_complex_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    result = solve(
        same_side_wall_reflection_scene(),
        Config(components={"reflection"}, coherent=True, export_paths=True),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    path_field = torch.complex(result.paths.field_real, result.paths.field_imag)
    expected_phase = torch.remainder(-torch.angle(path_field), 2.0 * torch.pi)
    assert torch.all(result.paths.valid)
    assert (
        torch.count_nonzero(path_field.abs() > 0.0).item() == result.paths.valid.numel()
    )
    torch.testing.assert_close(
        result.paths.path_gain, path_field.abs().square(), rtol=2.0e-4, atol=1.0e-10
    )
    torch.testing.assert_close(
        result.paths.phase_rad, expected_phase, rtol=2.0e-4, atol=1.0e-6
    )
    torch.testing.assert_close(
        result.path_gain, result.field.abs().square(), rtol=2.0e-4, atol=1.0e-10
    )
    torch.testing.assert_close(
        result.component_power["reflection"],
        result.component_fields["reflection"].abs().square(),
        rtol=2.0e-4,
        atol=1.0e-10,
    )


def test_reflection_solver_uses_native_field_kernel_when_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")
    if not hasattr(ops.native_extension(), "deterministic_reflection_field"):
        pytest.skip("native deterministic reflection field kernel is not built")

    calls = 0
    native_field = deterministic_fields.deterministic_reflection_field

    def count_native_call(**kwargs):
        nonlocal calls
        calls += 1
        return native_field(**kwargs)

    monkeypatch.setattr(
        deterministic_fields, "deterministic_reflection_field", count_native_call
    )
    result = solve(
        same_side_wall_reflection_scene(),
        Config(components={"reflection"}, coherent=True, export_paths=True),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    assert calls > 0
    assert (
        torch.count_nonzero(result.paths.path_gain > 0.0).item()
        == result.paths.valid.numel()
    )


def test_single_plane_reflection_does_not_use_python_triangle_fallback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    assert not hasattr(topology, "_inside_triangle")

    result = solve(
        same_side_wall_reflection_scene(),
        Config(components={"reflection"}, coherent=True, export_paths=True),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    assert result.paths.valid.numel() > 0


def test_reflection_solver_requires_native_field_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    def fail_native_kernel(**kwargs):
        raise RuntimeError(
            "_channel.deterministic_reflection_field CUDA kernel is required"
        )

    monkeypatch.setattr(
        deterministic_fields, "deterministic_reflection_field", fail_native_kernel
    )
    with pytest.raises(
        RuntimeError, match="deterministic_reflection_field CUDA kernel is required"
    ):
        solve(
            same_side_wall_reflection_scene(),
            Config(components={"reflection"}, coherent=True, export_paths=True),
            reference_frequency_hz=3.0e9,
        )
