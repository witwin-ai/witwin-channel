# Copyright Xingyu Chen.
# Tests solver rough reflection ad.

"""Solver-level AD through the native rough-reflection C_r scale (ADR-010).

End-to-end callers for ``field_rough_reflection_scale_backward`` / ``_jvp``:
the frequency derivative of a rough-wall specular solve flows through the
coherent C_r attenuation (dC_r/df), mirroring the frozen rough-reflection-cr
jvp/vjp baseline cells.
"""

from __future__ import annotations

import pytest
import torch

from tests.ad._fd import central_difference_gradient, relative_error
from tests.ad._tolerances import ABS_TOL, FD_REL_STEP_FREQUENCY, REL_TOL_GENERAL
from tests.support.scenes import rough_wall_structure
from witwin.core import Scene
from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as deterministic_solve
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as path_solve

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for solver AD"
)

_SOLVERS = ("deterministic", "path")
_FREQUENCY_HZ = 3.0e9
_COMPONENTS = frozenset({"reflection"})


def _rough_scene() -> Scene:
    wall = rough_wall_structure(
        2.5, rms_height_m=0.015, corr_length_m=0.15, half_size=2.0
    )
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, -1.0, 0.0])),
            make_receiver(position=torch.tensor([0.0, 1.0, 0.0])),
        ],
    )


def _solve(
    scene: Scene,
    solver: str,
    ad_mode: str,
    *,
    reference_frequency_hz: float | torch.Tensor = _FREQUENCY_HZ,
):
    if solver == "path":
        return path_solve(
            scene,
            PathConfig(max_depth=1, components=_COMPONENTS, ad_mode=ad_mode),
            reference_frequency_hz=reference_frequency_hz,
        )
    return deterministic_solve(
        scene,
        DeterministicConfig(
            max_depth=1,
            components=_COMPONENTS,
            export_paths=True,
            ad_mode=ad_mode,
        ),
        reference_frequency_hz=reference_frequency_hz,
    )


def _coefficients(result, solver: str) -> torch.Tensor:
    if solver == "path":
        return result.a
    return result.paths.coefficient


def _loss(result, solver: str) -> torch.Tensor:
    coefficient = _coefficients(result, solver)
    return coefficient.real.sum() + 0.5 * coefficient.imag.sum()


@pytest.mark.parametrize("solver", _SOLVERS)
def test_rough_reflection_frequency_grad_matches_fd(solver):
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    scene = _rough_scene()
    result = _solve(scene, solver, "vjp", reference_frequency_hz=frequency)
    _loss(result, solver).backward()
    assert frequency.grad is not None
    grad = frequency.grad.detach()
    assert float(grad.abs()) > 0.0

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        fd_scene = _rough_scene()
        fd_result = _solve(
            fd_scene,
            solver,
            "none",
            reference_frequency_hz=float(value),
        )
        return _loss(fd_result, solver).detach()

    fd_grad = central_difference_gradient(
        evaluate,
        torch.tensor(_FREQUENCY_HZ, dtype=torch.float64),
        _FREQUENCY_HZ * FD_REL_STEP_FREQUENCY,
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("solver", _SOLVERS)
def test_rough_reflection_frequency_jvp_matches_vjp(solver):
    # Reverse-mode gradient of the loss.
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    scene = _rough_scene()
    result = _solve(scene, solver, "vjp", reference_frequency_hz=frequency)
    _loss(result, solver).backward()
    vjp_grad = float(frequency.grad.detach())

    # Forward-mode directional derivative along d(frequency) = 1.
    primal = torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda")
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(
            primal.clone(), torch.ones_like(primal)
        )
        dual_scene = _rough_scene()
        dual_result = _solve(
            dual_scene,
            solver,
            "jvp",
            reference_frequency_hz=dual,
        )
        coefficient = _coefficients(dual_result, solver)
        tangent = torch.autograd.forward_ad.unpack_dual(coefficient).tangent
    assert tangent is not None
    jvp_value = float(tangent.real.sum() + 0.5 * tangent.imag.sum())
    assert jvp_value == pytest.approx(vjp_grad, rel=1.0e-4, abs=ABS_TOL)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_rough_reflection_ad_mode_none_stays_primal(solver):
    result = _solve(_rough_scene(), solver, "none")
    coefficient = _coefficients(result, solver)
    assert not coefficient.requires_grad
    assert torch.autograd.forward_ad.unpack_dual(coefficient).tangent is None