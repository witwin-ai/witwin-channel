"""AD-4 solver-level tests: UTD diffraction and coupled R-D gradients.

Covers the plan 07 section 9.3 cells delivered by AD-4 for D=deterministic
and P=path: eps_r / sigma_e / frequency / TX/RX position x wedge diffraction
(component 2, re-evaluated from the frozen topology), and the same parameters
x coupled reflection-diffraction (components 3/4, path solver: coupled paths
are a path-solver feature). The wedge re-evaluation forward-parity gate pins
the re-evaluated field against RayD's order-1 export.
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
    REL_TOL_GENERAL,
)
from tests.support.scenes import coupled_wall_wedge_scene, wedge_diffraction_scene
from witwin.channel_native import Scene
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
_WEDGE_MATERIAL = {"eps_r": 4.0, "sigma_e": 0.02}
_WEDGE_TX = (0.0, -1.0, 0.5)
_WEDGE_RX = (3.0, 1.0, 0.5)
# Off the scene's x = 0 symmetry axis: exactly on it, the ground specular
# point and the wedge stationary points ride degenerate loci where tiny
# endpoint perturbations flip discrete winners, so a central difference of
# the primal never converges (the same reason the AD-2 reflection fixture
# uses an asymmetric wall).
_COUPLED_TX = (0.4, -2.2, 1.15)
_COUPLED_RX = (0.55, 2.3, 4.8)
_COUPLED_COMPONENTS = frozenset({"reflection", "diffraction"})


def _wedge_scene(
    frequency: float | torch.Tensor = _FREQUENCY_HZ,
    tx: torch.Tensor | None = None,
    rx: torch.Tensor | None = None,
) -> Scene:
    return wedge_diffraction_scene(
        Dielectric(**_WEDGE_MATERIAL), tx=tx, rx=rx, frequency=frequency
    )


def _coupled_scene(
    frequency: float | torch.Tensor = _FREQUENCY_HZ,
    tx: torch.Tensor | None = None,
    rx: torch.Tensor | None = None,
) -> Scene:
    return coupled_wall_wedge_scene(
        Dielectric(**_WEDGE_MATERIAL),
        tx=torch.tensor(_COUPLED_TX) if tx is None else tx,
        rx=torch.tensor(_COUPLED_RX) if rx is None else rx,
        frequency=frequency,
    )


def _solve_wedge(scene: Scene, solver: str, ad_mode: str):
    components = frozenset({"diffraction"})
    if solver == "path":
        return path_solve(
            scene, PathConfig(max_depth=1, components=components, ad_mode=ad_mode)
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


def _solve_coupled(scene: Scene, ad_mode: str):
    return path_solve(
        scene,
        PathConfig(
            max_depth=2,
            components=_COUPLED_COMPONENTS,
            coupled_paths=True,
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
    solve,
    solver: str,
    leaf: torch.Tensor,
    step: float,
) -> torch.Tensor:
    base = leaf.detach().clone()

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            leaf.copy_(values.to(device=leaf.device, dtype=leaf.dtype))
        try:
            return _loss(solve(scene, solver, "none"), solver).detach()
        finally:
            with torch.no_grad():
                leaf.copy_(base)

    return central_difference_gradient(evaluate, base, step)


def _solve_wedge_adapter(scene: Scene, solver: str, ad_mode: str):
    return _solve_wedge(scene, solver, ad_mode)


def _solve_coupled_adapter(scene: Scene, solver: str, ad_mode: str):
    del solver
    return _solve_coupled(scene, ad_mode)


# ---------------------------------------------------------------------------
# Forward-parity gate: the AD-mode re-evaluated wedge field must reproduce
# RayD's order-1 export (topology.field_xyz path) to float32 tolerance. This
# is the only guard against convention drift (half-space Fresnel selection,
# +z tx polarization, stationary-point handling).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize("material", ("dielectric", "pec"))
def test_wedge_reevaluation_forward_parity(solver, material):
    scene = (
        _wedge_scene() if material == "dielectric" else wedge_diffraction_scene()
    )
    primal = _solve_wedge(scene, solver, "none")
    reevaluated = _solve_wedge(scene, solver, "vjp")
    a_primal = _coefficients(primal, solver)
    a_reevaluated = _coefficients(reevaluated, solver).detach()
    assert a_primal.shape == a_reevaluated.shape
    assert int(a_primal.shape[0]) > 0
    error = relative_error(a_reevaluated, a_primal, abs_floor=ABS_TOL)
    assert error <= 1.0e-3, f"wedge re-evaluation drifted from the export: {error}"


def test_coupled_forward_stays_primal_without_geometry_leaves():
    """Materials-only coupled AD keeps the exact primal coupled coefficients.

    The coupled Function forward is dispatch only (same native kernel, same
    inputs), so the coupled rows must match bit-for-bit. The pure-diffraction
    rows of the same solve are re-evaluated from the frozen topology in AD
    mode and are held to the float32 parity gate instead.
    """

    scene = _coupled_scene()
    primal = _solve_coupled(scene, "none")
    ad = _solve_coupled(scene, "vjp")
    types = primal.interaction_type
    coupled = (types == 1).any(dim=-1) & (types == 2).any(dim=-1)
    assert bool(coupled.any())
    torch.testing.assert_close(
        primal.a[coupled], ad.a.detach()[coupled], rtol=0.0, atol=0.0
    )
    error = relative_error(ad.a.detach(), primal.a, abs_floor=ABS_TOL)
    assert error <= 1.0e-3, f"coupled-scene AD forward drifted: {error}"


# ---------------------------------------------------------------------------
# Wedge diffraction (component 2): materials, frequency, endpoints.
# ---------------------------------------------------------------------------


_MATERIAL_STEPS = {"eps_r": FD_STEP_EPS_R, "sigma_e": FD_STEP_SIGMA_E}


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize("param", ("eps_r", "sigma_e"))
def test_wedge_material_grad_matches_fd(solver, param):
    scene = _wedge_scene()
    compiled = scene.compile()
    leaf = getattr(compiled.materials, param)
    leaf.requires_grad_(True)
    try:
        result = _solve_wedge(scene, solver, "vjp")
        _loss(result, solver).backward()
        assert leaf.grad is not None
        grad = leaf.grad.detach().clone()
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    assert float(grad.abs().max()) > 0.0
    fd_grad = _fd_gradient_via_store(
        scene, _solve_wedge_adapter, solver, leaf, _MATERIAL_STEPS[param]
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("solver", _SOLVERS)
def test_wedge_frequency_grad_matches_fd(solver):
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    scene = _wedge_scene(frequency)
    result = _solve_wedge(scene, solver, "vjp")
    _loss(result, solver).backward()
    assert frequency.grad is not None
    grad = frequency.grad.detach()
    assert float(grad.abs()) > 0.0

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        fd_scene = _wedge_scene(float(value))
        return _loss(_solve_wedge(fd_scene, solver, "none"), solver).detach()

    fd_grad = central_difference_gradient(
        evaluate,
        torch.tensor(_FREQUENCY_HZ, dtype=torch.float64),
        _FREQUENCY_HZ * FD_REL_STEP_FREQUENCY,
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize("endpoint", ("tx", "rx"))
def test_wedge_endpoint_position_grad_matches_fd(solver, endpoint):
    base = torch.tensor(
        _WEDGE_TX if endpoint == "tx" else _WEDGE_RX, dtype=torch.float32
    )
    leaf = base.clone().cuda().requires_grad_(True)
    scene = _wedge_scene(**{endpoint: leaf})
    _loss(_solve_wedge(scene, solver, "vjp"), solver).backward()
    assert leaf.grad is not None

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        fd_scene = _wedge_scene(**{endpoint: values.clone()})
        return _loss(_solve_wedge(fd_scene, solver, "none"), solver).detach()

    expected = central_difference_gradient(evaluate, base, FD_STEP_POSITION_PHASE)
    assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_wedge_forward_mode_matches_reverse():
    """JVP-vs-VJP inner-product duality on the wedge eps_r seed."""

    scene = _wedge_scene()
    compiled = scene.compile()
    base = compiled.materials.eps_r
    tangent = torch.ones_like(base)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(base.detach().clone(), tangent)
        object.__setattr__(compiled.materials, "eps_r", dual)
        try:
            result = _solve_wedge(scene, "deterministic", "jvp")
            coefficient = _coefficients(result, "deterministic")
            tangent_out = torch.autograd.forward_ad.unpack_dual(coefficient).tangent
        finally:
            object.__setattr__(compiled.materials, "eps_r", base)
    assert tangent_out is not None
    jvp = tangent_out.real.sum() + 0.5 * tangent_out.imag.sum()

    base.requires_grad_(True)
    try:
        _loss(_solve_wedge(scene, "deterministic", "vjp"), "deterministic").backward()
        vjp = (base.grad * tangent).sum()
    finally:
        base.requires_grad_(False)
        base.grad = None
    assert relative_error(jvp, vjp, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


# ---------------------------------------------------------------------------
# Coupled reflection-diffraction (components 3/4, path solver).
# ---------------------------------------------------------------------------


def _assert_coupled_rows(result) -> None:
    # A coupled path carries one reflection (type 1) AND one diffraction
    # (type 2) event along the same path axis.
    types = result.interaction_type
    coupled = (types == 1).any(dim=-1) & (types == 2).any(dim=-1)
    assert bool(coupled.any()), "the coupled fixture no longer produces coupled paths"


@pytest.mark.parametrize("param", ("eps_r", "sigma_e"))
def test_coupled_material_grad_matches_fd(param):
    scene = _coupled_scene()
    compiled = scene.compile()
    leaf = getattr(compiled.materials, param)
    leaf.requires_grad_(True)
    try:
        result = _solve_coupled(scene, "vjp")
        _assert_coupled_rows(result)
        _loss(result, "path").backward()
        assert leaf.grad is not None
        grad = leaf.grad.detach().clone()
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    assert float(grad.abs().max()) > 0.0
    fd_grad = _fd_gradient_via_store(
        scene, _solve_coupled_adapter, "path", leaf, _MATERIAL_STEPS[param]
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_coupled_frequency_grad_matches_fd():
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    scene = _coupled_scene(frequency)
    result = _solve_coupled(scene, "vjp")
    _assert_coupled_rows(result)
    _loss(result, "path").backward()
    assert frequency.grad is not None
    grad = frequency.grad.detach()
    assert float(grad.abs()) > 0.0

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        fd_scene = _coupled_scene(float(value))
        return _loss(_solve_coupled(fd_scene, "none"), "path").detach()

    fd_grad = central_difference_gradient(
        evaluate,
        torch.tensor(_FREQUENCY_HZ, dtype=torch.float64),
        _FREQUENCY_HZ * FD_REL_STEP_FREQUENCY,
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("endpoint", ("tx", "rx"))
def test_coupled_endpoint_position_grad_matches_fd(endpoint):
    base = torch.tensor(
        _COUPLED_TX if endpoint == "tx" else _COUPLED_RX, dtype=torch.float32
    )
    leaf = base.clone().cuda().requires_grad_(True)
    scene = _coupled_scene(**{endpoint: leaf})
    result = _solve_coupled(scene, "vjp")
    _assert_coupled_rows(result)
    _loss(result, "path").backward()
    assert leaf.grad is not None

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        fd_scene = _coupled_scene(**{endpoint: values.clone()})
        return _loss(_solve_coupled(fd_scene, "none"), "path").detach()

    expected = central_difference_gradient(evaluate, base, FD_STEP_POSITION_PHASE)
    assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
