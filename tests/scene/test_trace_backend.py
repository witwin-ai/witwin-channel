# Copyright Xingyu Chen.
# Tests the trace backend request, its report, and OptiX/CUDA result parity through the solvers.

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from tests.support.core_world import make_receiver, make_transmitter
from tests.support.scenes import (
    same_side_wall_reflection_scene,
    transmission_wall_structure,
    wedge_diffraction_scene,
)
from witwin.channel import path, runtime, runtime_diagnostics
from witwin.channel.scene import clear_compile_cache, compile as compile_scene
from witwin.core import MaterialLayer, PhysicalMaterial, Scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

ROOT = Path(__file__).resolve().parents[2]


def _optix_available() -> bool:
    return bool(runtime.runtime_probe(torch.cuda.current_device())["optix_available"])


def _compile(monkeypatch, backend: str, scene: Scene):
    monkeypatch.setenv(runtime.TRACE_BACKEND_ENV, backend)
    clear_compile_cache()
    return compile_scene(scene, reference_frequency_hz=3.0e9)


def test_forced_cuda_backend_is_selected_and_reported(monkeypatch):
    compiled = _compile(monkeypatch, "cuda", same_side_wall_reflection_scene())

    assert compiled.rayd.requested_trace_backend == "cuda"
    assert compiled.rayd.trace_backend == "cuda"
    scenes = runtime_diagnostics()["trace_backend"]["scenes"]
    assert {"device_index": torch.cuda.current_device(), "trace_backend": "cuda"} in scenes


def test_auto_backend_follows_the_optix_probe(monkeypatch):
    compiled = _compile(monkeypatch, "auto", same_side_wall_reflection_scene())

    expected = "optix" if _optix_available() else "cuda"
    assert compiled.rayd.trace_backend == expected
    optix = runtime_diagnostics()["trace_backend"]["optix"]
    assert optix["available"] is (expected == "optix")


def test_the_compile_cache_keys_on_the_requested_backend(monkeypatch):
    scene = same_side_wall_reflection_scene()
    cuda = _compile(monkeypatch, "cuda", scene)
    monkeypatch.setenv(runtime.TRACE_BACKEND_ENV, "auto")
    auto = compile_scene(scene, reference_frequency_hz=3.0e9)

    assert auto is not cuda
    assert auto.rayd is not cuda.rayd
    assert auto.rayd.requested_trace_backend == "auto"


def test_forcing_optix_without_optix_fails_loudly_at_compile():
    code = textwrap.dedent(
        """
        import torch
        from tests.support.scenes import same_side_wall_reflection_scene
        from witwin.channel.scene import compile
        try:
            compile(same_side_wall_reflection_scene(), reference_frequency_hz=3.0e9)
        except RuntimeError as exc:
            print("REFUSED", exc)
        torch.ones(4, device="cuda").sum().item()
        print("CONTEXT_OK")
        """
    )
    env = dict(os.environ, RAYD_DISABLE_OPTIX="1", WITWIN_CHANNEL_TRACE_BACKEND="optix")
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
        timeout=300, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "REFUSED" in completed.stdout and "RAYD_DISABLE_OPTIX" in completed.stdout
    assert "CONTEXT_OK" in completed.stdout


def _path_outputs(monkeypatch, backend: str, scene: Scene, components: set[str]):
    monkeypatch.setenv(runtime.TRACE_BACKEND_ENV, backend)
    clear_compile_cache()
    result = path.solve(
        scene, path.Config(max_depth=3, components=components), reference_frequency_hz=3.5e9
    )
    return result


def _transmission_scene() -> Scene:
    glass = PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=0.05, eps_r=6.0, sigma_e=0.004),), name="glass"
    )
    reflector = PhysicalMaterial(eps_r=5.0, sigma_e=0.05)
    return Scene(
        structures=[
            transmission_wall_structure(3.0, glass, surface_id=1, half_size=3.0),
            transmission_wall_structure(-2.0, reflector, name="back", surface_id=2),
        ],
        endpoints=[
            make_transmitter(torch.tensor([0.0, -0.4, 1.0])),
            make_receiver(torch.tensor([5.0, 0.7, 1.3])),
            make_receiver(torch.tensor([1.0, 1.5, 0.2])),
        ],
    )


# RayD compiles its OptiX programs with fast math and its CUDA kernels without, so transcendental
# functions in the field (phase, Fresnel and UTD terms) differ by more than float32 rounding.
_FIELD_RTOL = 5e-4


def _assert_parity(optix, cuda) -> None:
    """Discrete outputs exact; delays to float32 rounding; fields to the stated tolerance."""

    for name in ("valid", "num_paths", "interaction_type", "primitive_id", "material_id"):
        assert torch.equal(getattr(optix, name), getattr(cuda, name)), name
    scale = float(optix.position.abs().max())
    torch.testing.assert_close(cuda.position, optix.position, rtol=0.0, atol=16 * 2.0**-23 * scale)
    torch.testing.assert_close(cuda.tau, optix.tau, rtol=16 * 2.0**-23, atol=0.0)
    error = (cuda.a - optix.a).abs()[optix.valid]
    relative = error / optix.a.abs()[optix.valid]
    assert float(relative.max()) <= _FIELD_RTOL, relative


def test_reflection_and_transmission_paths_match_across_backends(monkeypatch):
    if not _optix_available():
        pytest.skip("OptiX is needed as the parity reference")
    components = {"los", "reflection", "transmission"}
    optix = _path_outputs(monkeypatch, "optix", _transmission_scene(), components)
    cuda = _path_outputs(monkeypatch, "cuda", _transmission_scene(), components)

    assert int(optix.valid.sum()) > 2
    _assert_parity(optix, cuda)


def test_diffraction_paths_match_for_x_polarized_sources(monkeypatch):
    if not _optix_available():
        pytest.skip("OptiX is needed as the parity reference")

    def scene() -> Scene:
        wedge = wedge_diffraction_scene()
        return Scene(
            structures=list(wedge.structures),
            endpoints=[
                make_transmitter(
                    torch.tensor([0.0, -1.0, 0.5]), polarization=torch.tensor([1.0, 0.0, 0.0])
                ),
                make_receiver(torch.tensor([3.0, 1.0, 0.5])),
            ],
        )

    optix = _path_outputs(monkeypatch, "optix", scene(), {"diffraction"})
    cuda = _path_outputs(monkeypatch, "cuda", scene(), {"diffraction"})

    assert int(optix.valid.sum()) >= 1
    _assert_parity(optix, cuda)


@pytest.mark.xfail(
    strict=True,
    reason="RayD's CUDA diffraction path exporter fixes the source polarization to +x",
)
def test_diffraction_paths_match_for_vertically_polarized_sources(monkeypatch):
    if not _optix_available():
        pytest.skip("OptiX is needed as the parity reference")
    optix = _path_outputs(monkeypatch, "optix", wedge_diffraction_scene(), {"diffraction"})
    cuda = _path_outputs(monkeypatch, "cuda", wedge_diffraction_scene(), {"diffraction"})

    _assert_parity(optix, cuda)
