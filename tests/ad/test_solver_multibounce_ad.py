"""Solver-level multi-reflection (depth >= 2) AD tests for deterministic/path.

Covers the plan 07 section 9.3 multi-reflection column for D and P: eps_r /
sigma_e / frequency / TX/RX position / mesh vertex, each against a central
finite difference of the primal solve. The loss keeps ONLY the depth-2 rows
so a passing test cannot ride on the single-bounce paths of the same solve;
the kernels are general-depth (kernel-level depth-2 coverage lives in
test_field_em_ad.py) and this file closes the solver-level gap.

The two parallel walls are z-asymmetric (top edges at 2.4 m and 2.3 m) and
the endpoints sit off every symmetry plane, for the same reason the AD-2
reflection fixture uses an asymmetric wall: a specular point on a shared
triangulation diagonal or a symmetry axis makes the discovered winner set
flip under the FD probes, which is a path birth/death discontinuity outside
the fixed-winner contract.
"""

from __future__ import annotations

import pytest
import torch

from tests.ad._fd import central_difference_gradient, relative_error
from tests.ad._tolerances import (
    ABS_TOL,
    FD_REL_STEP_FREQUENCY,
    FD_STEP_EPS_R,
    FD_STEP_POSITION_PHASE,
    FD_STEP_SIGMA_E,
    FD_STEP_VERTEX,
    REL_TOL_GENERAL,
)
from witwin.channel_native import ReceiverPoint, Scene, Structure, Transmitter
from witwin.channel_native.core.materials import Dielectric
from witwin.channel_native.deterministic import Config as DeterministicConfig
from witwin.channel_native.deterministic import solve as deterministic_solve
from witwin.channel_native.path import Config as PathConfig
from witwin.channel_native.path import solve as path_solve

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for solver AD"
)

_SOLVERS = ("deterministic", "path")
_FREQUENCY_HZ = 3.0e9
_COMPONENTS = frozenset({"reflection"})

_TX = (0.3, -1.0, 0.6)
_RX = (-0.2, 1.1, 0.4)
_WALL_A_VERTICES = (
    (2.5, -3.0, -1.0),
    (2.5, 3.0, -1.0),
    (2.5, -3.0, 2.4),
    (2.5, 3.0, 2.4),
)
_WALL_B_VERTICES = (
    (-2.2, -3.0, -1.0),
    (-2.2, 3.0, -1.0),
    (-2.2, -3.0, 2.3),
    (-2.2, 3.0, 2.3),
)


def _vec(values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _scene(
    frequency: float | torch.Tensor = _FREQUENCY_HZ,
    tx: torch.Tensor | None = None,
    rx: torch.Tensor | None = None,
    vertices_a: torch.Tensor | None = None,
    vertices_b: torch.Tensor | None = None,
) -> Scene:
    wall_a = Structure(
        vertices=torch.tensor(_WALL_A_VERTICES) if vertices_a is None else vertices_a,
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=Dielectric(eps_r=4.0, sigma_e=0.02),
        name="ad-wall-a",
        surface_id=1,
    )
    wall_b = Structure(
        vertices=torch.tensor(_WALL_B_VERTICES) if vertices_b is None else vertices_b,
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=Dielectric(eps_r=3.0, sigma_e=0.05),
        name="ad-wall-b",
        surface_id=2,
    )
    return Scene(
        structures=[wall_a, wall_b],
        transmitters=[Transmitter(position=_vec(_TX) if tx is None else tx)],
        receivers=[ReceiverPoint(position=_vec(_RX) if rx is None else rx)],
        frequency=frequency,
    )


def _solve(scene: Scene, solver: str, ad_mode: str):
    if solver == "path":
        return path_solve(
            scene, PathConfig(max_depth=2, components=_COMPONENTS, ad_mode=ad_mode)
        )
    return deterministic_solve(
        scene,
        DeterministicConfig(
            max_depth=2,
            components=_COMPONENTS,
            export_paths=True,
            ad_mode=ad_mode,
        ),
    )


def _depth2_loss(result, solver: str) -> torch.Tensor:
    """Complex-coefficient loss over the depth-2 (double-bounce) rows only."""

    if solver == "path":
        depth = (result.interaction_type == 1).sum(dim=-1)
        coefficient = result.a[depth == 2]
    else:
        coefficient = result.paths.coefficient[result.paths.depth == 2]
    if int(coefficient.shape[0]) == 0:
        raise AssertionError("the multibounce fixture produced no depth-2 paths")
    return coefficient.real.sum() + 0.5 * coefficient.imag.sum()


def test_fixture_has_double_bounce_paths():
    result = _solve(_scene(), "deterministic", "none")
    depths = result.paths.depth
    assert int((depths == 2).sum()) >= 2, "expected both A->B and B->A paths"


_MATERIAL_STEPS = {"eps_r": FD_STEP_EPS_R, "sigma_e": FD_STEP_SIGMA_E}


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize("param", ("eps_r", "sigma_e"))
def test_multibounce_material_grad_matches_fd(solver, param):
    scene = _scene()
    compiled = scene.compile()
    leaf = getattr(compiled.materials, param)
    leaf.requires_grad_(True)
    try:
        _depth2_loss(_solve(scene, solver, "vjp"), solver).backward()
        assert leaf.grad is not None
        grad = leaf.grad.detach().clone()
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    assert float(grad.abs().max()) > 0.0

    base = leaf.detach().clone()

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            leaf.copy_(values.to(device=leaf.device, dtype=leaf.dtype))
        try:
            return _depth2_loss(_solve(scene, solver, "none"), solver).detach()
        finally:
            with torch.no_grad():
                leaf.copy_(base)

    fd_grad = central_difference_gradient(evaluate, base, _MATERIAL_STEPS[param])
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("solver", _SOLVERS)
def test_multibounce_frequency_grad_matches_fd(solver):
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    scene = _scene(frequency)
    _depth2_loss(_solve(scene, solver, "vjp"), solver).backward()
    assert frequency.grad is not None
    grad = frequency.grad.detach()
    assert float(grad.abs()) > 0.0

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        fd_scene = _scene(float(value))
        return _depth2_loss(_solve(fd_scene, solver, "none"), solver).detach()

    fd_grad = central_difference_gradient(
        evaluate,
        torch.tensor(_FREQUENCY_HZ, dtype=torch.float64),
        _FREQUENCY_HZ * FD_REL_STEP_FREQUENCY,
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize("endpoint", ("tx", "rx"))
def test_multibounce_endpoint_position_grad_matches_fd(solver, endpoint):
    base = _vec(_TX if endpoint == "tx" else _RX)
    leaf = base.clone().cuda().requires_grad_(True)
    scene = _scene(**{endpoint: leaf})
    _depth2_loss(_solve(scene, solver, "vjp"), solver).backward()
    assert leaf.grad is not None

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        fd_scene = _scene(**{endpoint: values.clone()})
        return _depth2_loss(_solve(fd_scene, solver, "none"), solver).detach()

    expected = central_difference_gradient(evaluate, base, FD_STEP_POSITION_PHASE)
    assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("solver", _SOLVERS)
def test_multibounce_mesh_vertex_grad_matches_fd(solver):
    base = torch.tensor(_WALL_A_VERTICES, dtype=torch.float32)
    leaf = base.clone().cuda().requires_grad_(True)
    scene = _scene(vertices_a=leaf)
    _depth2_loss(_solve(scene, solver, "vjp"), solver).backward()
    assert leaf.grad is not None
    assert float(leaf.grad.abs().max()) > 0.0

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        fd_scene = _scene(vertices_a=values.clone())
        return _depth2_loss(_solve(fd_scene, solver, "none"), solver).detach()

    expected = central_difference_gradient(evaluate, base, FD_STEP_VERTEX)
    assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_multibounce_forward_mode_matches_reverse():
    """JVP-vs-VJP inner-product duality through the depth-2 rows (eps_r seed)."""

    scene = _scene()
    compiled = scene.compile()
    base = compiled.materials.eps_r
    tangent = torch.ones_like(base)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(base.detach().clone(), tangent)
        object.__setattr__(compiled.materials, "eps_r", dual)
        try:
            loss = _depth2_loss(_solve(scene, "deterministic", "jvp"), "deterministic")
            jvp = torch.autograd.forward_ad.unpack_dual(loss).tangent
        finally:
            object.__setattr__(compiled.materials, "eps_r", base)
    assert jvp is not None

    base.requires_grad_(True)
    try:
        _depth2_loss(
            _solve(scene, "deterministic", "vjp"), "deterministic"
        ).backward()
        vjp = (base.grad * tangent).sum()
    finally:
        base.requires_grad_(False)
        base.grad = None
    assert relative_error(jvp, vjp, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
