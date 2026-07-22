"""AD-1 solver-level tests: material/frequency gradients through solve().

Covers the plan 07 section 9.3 cells delivered by AD-1 for D=deterministic and
P=path: eps_r/sigma_e x single-reflection, eps_r/sigma_e/thickness x
transmission-multilayer, and frequency x LoS/single-reflection/transmission.
Also validates the zero-overhead ad_mode="none" contract, forward-mode duals
through a full solve, and the explicit-failure policy for unsupported
interactions.
"""

from __future__ import annotations

import pytest
import torch

from tests.ad._fd import central_difference_gradient, relative_error
from tests.ad._tolerances import (
    ABS_TOL,
    FD_REL_STEP_FREQUENCY,
    FD_STEP_EPS_R,
    FD_STEP_SIGMA_E,
    FD_STEP_THICKNESS,
    REL_TOL_GENERAL,
)
from tests.support.scenes import (
    empty_space_los_scene,
    transmission_wall_structure,
)
from witwin.channel import ReceiverPoint, Scene, Structure, Transmitter
from witwin.channel.core.materials import Dielectric, Layer, PhysicalSurface
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as deterministic_solve
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as path_solve

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for solver AD"
)

_SOLVERS = ("deterministic", "path")
_FREQUENCY_HZ = 3.0e9


def _reflection_scene(frequency: float | torch.Tensor = _FREQUENCY_HZ) -> Scene:
    wall = Structure(
        vertices=torch.tensor(
            [
                [2.5, -3.0, -1.0],
                [2.5, 3.0, -1.0],
                [2.5, -3.0, 2.0],
                [2.5, 3.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=Dielectric(eps_r=4.0, sigma_e=0.02),
        name="ad-open-wall",
        surface_id=1,
    )
    return Scene(
        structures=[wall],
        transmitters=[Transmitter(position=torch.tensor([0.0, -1.0, 0.5]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 1.0, 0.5]))],
        frequency=frequency,
    )


def _transmission_scene(frequency: float | torch.Tensor = _FREQUENCY_HZ) -> Scene:
    material = PhysicalSurface(
        layers=(
            Layer(thickness_m=0.06, eps_r=4.0, sigma_e=0.02),
            Layer(thickness_m=0.09, eps_r=2.5, sigma_e=0.01),
        ),
        name="ad-thin-sheet",
    )
    return Scene(
        structures=[transmission_wall_structure(3.0, material)],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([6.0, 0.4, 0.2]))],
        frequency=frequency,
    )


def _los_scene(frequency: float | torch.Tensor = _FREQUENCY_HZ) -> Scene:
    scene = empty_space_los_scene()
    return Scene(
        structures=[],
        transmitters=list(scene.transmitters),
        receivers=list(scene.receivers),
        frequency=frequency,
    )


_SCENE_BUILDERS = {
    "los": (_los_scene, frozenset({"los"})),
    "reflection": (_reflection_scene, frozenset({"reflection"})),
    "transmission": (_transmission_scene, frozenset({"transmission"})),
}


def _solve(scene: Scene, solver: str, components: frozenset[str], ad_mode: str):
    if solver == "path":
        return path_solve(
            scene,
            PathConfig(max_depth=1, components=components, ad_mode=ad_mode),
        )
    return deterministic_solve(
        scene,
        DeterministicConfig(
            max_depth=1,
            components=components,
            export_paths=True,
            ad_mode=ad_mode,
        ),
    )


def _coefficients(result, solver: str) -> torch.Tensor:
    if solver == "path":
        return result.a
    return result.paths.coefficient


def _loss(result, solver: str) -> torch.Tensor:
    coefficient = _coefficients(result, solver)
    return coefficient.real.sum() + 0.5 * coefficient.imag.sum()


def _fd_gradient_via_store(
    scene: Scene,
    solver: str,
    components: frozenset[str],
    leaf: torch.Tensor,
    step: float,
) -> torch.Tensor:
    base = leaf.detach().clone()

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            leaf.copy_(values.to(device=leaf.device, dtype=leaf.dtype))
        try:
            result = _solve(scene, solver, components, "none")
            return _loss(result, solver).detach()
        finally:
            with torch.no_grad():
                leaf.copy_(base)

    return central_difference_gradient(evaluate, base, step)


_MATERIAL_STEPS = {
    "eps_r": FD_STEP_EPS_R,
    "sigma_e": FD_STEP_SIGMA_E,
    "layer_eps_r": FD_STEP_EPS_R,
    "layer_sigma_e": FD_STEP_SIGMA_E,
    "layer_thickness_m": FD_STEP_THICKNESS,
}


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize("param", ("eps_r", "sigma_e"))
def test_single_reflection_material_grad_matches_fd(solver, param):
    scene = _reflection_scene()
    components = frozenset({"reflection"})
    compiled = scene.compile()
    leaf = getattr(compiled.materials, param)
    leaf.requires_grad_(True)
    try:
        result = _solve(scene, solver, components, "vjp")
        _loss(result, solver).backward()
        assert leaf.grad is not None
        grad = leaf.grad.detach().clone()
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    assert float(grad.abs().max()) > 0.0
    fd_grad = _fd_gradient_via_store(
        scene, solver, components, leaf, _MATERIAL_STEPS[param]
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize(
    "param", ("layer_eps_r", "layer_sigma_e", "layer_thickness_m")
)
def test_transmission_layer_grad_matches_fd(solver, param):
    scene = _transmission_scene()
    components = frozenset({"transmission"})
    compiled = scene.compile()
    leaf = getattr(compiled.materials, param)
    leaf.requires_grad_(True)
    try:
        result = _solve(scene, solver, components, "vjp")
        _loss(result, solver).backward()
        assert leaf.grad is not None
        grad = leaf.grad.detach().clone()
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    assert float(grad.abs().max()) > 0.0
    fd_grad = _fd_gradient_via_store(
        scene, solver, components, leaf, _MATERIAL_STEPS[param]
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize("interaction", ("los", "reflection", "transmission"))
def test_frequency_grad_matches_fd(solver, interaction):
    builder, components = _SCENE_BUILDERS[interaction]
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    scene = builder(frequency)
    result = _solve(scene, solver, components, "vjp")
    _loss(result, solver).backward()
    assert frequency.grad is not None
    grad = frequency.grad.detach()
    assert float(grad.abs()) > 0.0

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        fd_scene = builder(float(value))
        fd_result = _solve(fd_scene, solver, components, "none")
        return _loss(fd_result, solver).detach()

    fd_grad = central_difference_gradient(
        evaluate,
        torch.tensor(_FREQUENCY_HZ, dtype=torch.float64),
        _FREQUENCY_HZ * FD_REL_STEP_FREQUENCY,
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("solver", _SOLVERS)
def test_forward_mode_dual_solve_matches_fd(solver):
    scene = _reflection_scene()
    components = frozenset({"reflection"})
    compiled = scene.compile()
    base = compiled.materials.eps_r
    tangent = torch.ones_like(base)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(base.detach().clone(), tangent)
        object.__setattr__(compiled.materials, "eps_r", dual)
        try:
            result = _solve(scene, solver, components, "jvp")
            coefficient = _coefficients(result, solver)
            tangent_out = torch.autograd.forward_ad.unpack_dual(coefficient).tangent
        finally:
            object.__setattr__(compiled.materials, "eps_r", base)
    assert tangent_out is not None
    directional = tangent_out.real.sum() + 0.5 * tangent_out.imag.sum()

    fd_grad = _fd_gradient_via_store(scene, solver, components, base, FD_STEP_EPS_R)
    fd_directional = (fd_grad.double() * tangent.double().cpu()).sum()
    assert relative_error(directional, fd_directional, abs_floor=ABS_TOL) <= (
        REL_TOL_GENERAL
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_ad_mode_none_keeps_primal_contract(solver):
    scene = _reflection_scene()
    components = frozenset({"reflection"})
    compiled = scene.compile()
    compiled.materials.eps_r.requires_grad_(True)
    try:
        result_none = _solve(scene, solver, components, "none")
        result_vjp = _solve(scene, solver, components, "vjp")
    finally:
        compiled.materials.eps_r.requires_grad_(False)

    coefficient_none = _coefficients(result_none, solver)
    coefficient_vjp = _coefficients(result_vjp, solver)
    # Primal mode never builds a graph, even for requires_grad materials.
    assert not coefficient_none.requires_grad
    assert coefficient_none.grad_fn is None
    assert coefficient_vjp.requires_grad
    # Same forward values; primal mode keeps zero AD accounting while the
    # AD mode reports its real registered companions and retained tape
    # (plan 07 AD-4 metadata contract).
    torch.testing.assert_close(
        coefficient_none, coefficient_vjp.detach(), rtol=0.0, atol=0.0
    )
    kernel_none = result_none.metadata["kernel"]
    kernel_vjp = result_vjp.metadata["kernel"]
    assert kernel_none["forward_launch_count"] == kernel_vjp["forward_launch_count"]
    assert kernel_none["tape_bytes"] == 0
    assert kernel_none["backward_launch_count"] == 0
    assert kernel_none["jvp_launch_count"] == 0
    assert kernel_vjp["tape_bytes"] > 0
    assert kernel_vjp["backward_launch_count"] > 0
    assert kernel_vjp["jvp_launch_count"] == 0
    assert kernel_none["ad_status"] == "none"
    assert kernel_vjp["ad_status"] == "vjp"
    # none-mode is not AD-instrumented: it takes no timing synchronize and
    # reports exactly zero, which is part of the zero-overhead primal contract
    # (the leading sync would otherwise stall the caller's queued work). The AD
    # solve is instrumented and reports a positive wall time.
    assert kernel_none["forward_time_ms"] == 0.0
    assert kernel_vjp["forward_time_ms"] > 0.0


@pytest.mark.parametrize("solver", _SOLVERS)
def test_jvp_metadata_reports_dual_companions_without_tape(solver):
    """Forward mode runs its dual companions in-solve and retains no tape."""

    scene = _reflection_scene()
    result = _solve(scene, solver, frozenset({"reflection"}), "jvp")
    kernel = result.metadata["kernel"]
    assert kernel["ad_status"] == "jvp"
    assert kernel["jvp_launch_count"] > 0
    assert kernel["backward_launch_count"] == 0
    assert kernel["tape_bytes"] == 0
