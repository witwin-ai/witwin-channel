# Copyright Xingyu Chen.
# AD solver-level tests: UTD diffraction and coupled R-D gradients.

"""AD solver-level tests: UTD diffraction and coupled R-D gradients."""

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
from tests.support.scenes import coupled_wall_wedge_scene, wedge_diffraction_scene
from witwin.core import Mesh, Scene, Structure
from tests.support.core_world import make_receiver, make_transmitter
from witwin.core import PhysicalMaterial
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as deterministic_solve
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as path_solve
from witwin.channel.scene import compile as compile_scene

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for solver AD"
)

_SOLVERS = ("deterministic", "path")
_FREQUENCY_HZ = 3.0e9
_WEDGE_MATERIAL = {"eps_r": 4.0, "sigma_e": 0.02}
_WEDGE_TX = (0.0, -1.0, 0.5)
# RX must NOT sit mirror-symmetric to TX about the thin-screen plane y = 0.
# With RX_y = +1.0 it did (|TX_y| = |RX_y| = 1, equal z), placing the
# observation exactly on the reflection shadow boundary phi + phi' = 2*n*pi of
# the screen edge (edge_id 2, exterior angle 2*pi). That is a branch cut, and
# the wedge Fermat point falls OFF the physical edge (param = -0.5) there, so
# the near-null diffracted field (~100x below the dominant paths) flickers
# between branches under codegen differences (OptiX fast-math raygen vs the CUDA
# AD kernel) and central differences of the primal never converge. Perturbing
# Move RX_y off symmetry to avoid the branch cut while keeping every path well conditioned.
_WEDGE_RX = (3.0, 1.07, 0.5)
# Off the scene's x = 0 symmetry axis: exactly on it, the ground specular
# point and the wedge stationary points ride degenerate loci where tiny
# endpoint perturbations flip discrete winners, so a central difference of
# the primal never converges (the same reason the AD reflection fixture
# uses an asymmetric wall).
_COUPLED_TX = (0.4, -2.2, 1.15)
_COUPLED_RX = (0.55, 2.3, 4.8)
_COUPLED_COMPONENTS = frozenset({"reflection", "diffraction"})


def _wedge_scene(tx: torch.Tensor | None = None, rx: torch.Tensor | None = None) -> Scene:
    # Route through the module TX/RX (perturbed off the y = 0 mirror plane, see
    # _WEDGE_RX) rather than the shared scenes.py defaults, which are still on it.
    return wedge_diffraction_scene(
        PhysicalMaterial(**_WEDGE_MATERIAL),
        tx=torch.tensor(_WEDGE_TX) if tx is None else tx,
        rx=torch.tensor(_WEDGE_RX) if rx is None else rx,
    )


def _coupled_scene(tx: torch.Tensor | None = None, rx: torch.Tensor | None = None) -> Scene:
    return coupled_wall_wedge_scene(
        PhysicalMaterial(**_WEDGE_MATERIAL),
        tx=torch.tensor(_COUPLED_TX) if tx is None else tx,
        rx=torch.tensor(_COUPLED_RX) if rx is None else rx,
    )


def _solve_wedge(
    scene: Scene, solver: str, ad_mode: str, *,
    reference_frequency_hz: float | torch.Tensor = _FREQUENCY_HZ,
):
    components = frozenset({"diffraction"})
    if solver == "path":
        return path_solve(
            scene,
            PathConfig(max_depth=1, components=components, ad_mode=ad_mode),
            reference_frequency_hz=reference_frequency_hz,
        )
    return deterministic_solve(
        scene,
        DeterministicConfig(
            max_depth=1,
            components=components,
            export_paths=True,
            ad_mode=ad_mode,
        ),
        reference_frequency_hz=reference_frequency_hz,
    )


def _solve_coupled(
    scene: Scene, ad_mode: str, *, reference_frequency_hz: float | torch.Tensor = _FREQUENCY_HZ,
):
    return path_solve(
        scene,
        PathConfig(
            max_depth=2,
            components=_COUPLED_COMPONENTS,
            coupled_paths=True,
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


def _fd_gradient_via_store(
    scene: Scene, solve, solver: str, leaf: torch.Tensor, step: float,
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
        _wedge_scene()
        if material == "dielectric"
        else wedge_diffraction_scene(
            tx=torch.tensor(_WEDGE_TX), rx=torch.tensor(_WEDGE_RX)
        )
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
    compiled = compile_scene(scene, reference_frequency_hz=_FREQUENCY_HZ)
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
    scene = _wedge_scene()
    result = _solve_wedge(
        scene,
        solver,
        "vjp",
        reference_frequency_hz=frequency,
    )
    _loss(result, solver).backward()
    assert frequency.grad is not None
    grad = frequency.grad.detach()
    assert float(grad.abs()) > 0.0

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        fd_scene = _wedge_scene()
        return _loss(
            _solve_wedge(
                fd_scene,
                solver,
                "none",
                reference_frequency_hz=float(value),
            ),
            solver,
        ).detach()

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


# Single-structure wedge with a SHARED edge (vertices 0-1): the mesh-vertex
# tests need welded topology so that one live vertex table drives both wedge
# faces. The two-structure fixture above duplicates the edge vertices, and a
# per-copy FD probe would split the wedge (a topology change outside the
# fixed-winner contract). This fixture drives the wedge with _WEDGE_TX/_WEDGE_RX,
# whose RX is deliberately off the y = 0 mirror plane (see _WEDGE_RX): on the
# plane the screen-edge observation lands on the RSB branch cut with an off-edge
# Fermat point, which is what made the vertex parity/FD gates flake.
_WELDED_WEDGE_VERTICES = (
    (2.0, 0.0, -1.0),
    (2.0, 0.0, 2.0),
    (2.0, 2.0, -1.0),
    (4.0, 0.0, -1.0),
)


def _welded_wedge_scene(vertices: torch.Tensor | None = None) -> Scene:
    resolved_vertices = (
        torch.tensor(_WELDED_WEDGE_VERTICES) if vertices is None else vertices
    )
    wedge = Structure(
        geometry=Mesh(
            resolved_vertices,
            torch.tensor(
                [[0, 1, 2], [0, 3, 1]],
                device=resolved_vertices.device,
            ),
            recenter=False,
            fill_mode="surface",
            topology_diagnostics=False,
        ),
        material=PhysicalMaterial(**_WEDGE_MATERIAL),
        name="welded-wedge",
        surface_id=2,
    )
    return Scene(
        structures=[wedge],
        endpoints=[
            make_transmitter(position=torch.tensor(_WEDGE_TX)),
            make_receiver(position=torch.tensor(_WEDGE_RX)),
        ],
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_wedge_mesh_vertex_grad_matches_fd(solver):
    """Mesh vertex x diffraction (the derivative capability matrix).

 The wedge kernel rebuilds the edge tables (edge anchor/direction/bounds,
 face normals, exterior angle) from the winner vertices on the dual row,
 so moving an edge vertex moves the stationary point, the wedge angle and
 the incidence angles. FD parity of the primal is the gate for the
 rebuilt-tables convention (sign/plane assignment against the frozen
 discovery tables).
 """

    base = torch.tensor(_WELDED_WEDGE_VERTICES, dtype=torch.float32)
    leaf = base.clone().cuda().requires_grad_(True)
    scene = _welded_wedge_scene(vertices=leaf)
    result = _solve_wedge(scene, solver, "vjp")
    assert int(_coefficients(result, solver).shape[0]) > 0
    _loss(result, solver).backward()
    assert leaf.grad is not None
    assert float(leaf.grad.abs().max()) > 0.0

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        fd_scene = _welded_wedge_scene(vertices=values.clone())
        return _loss(_solve_wedge(fd_scene, solver, "none"), solver).detach()

    expected = central_difference_gradient(evaluate, base, FD_STEP_VERTEX)
    assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_wedge_vertex_mode_forward_stays_on_parity():
    """The vertex-rebuilt edge tables must reproduce the discovery tables.

 With a live vertex leaf the AD forward evaluates the wedge from tables
 rebuilt inside the kernel; any convention drift (normal winding, plane
 assignment, exterior angle) would move the primal value away from the
 frozen-table evaluation and fail this float32 parity gate.
 """

    scene_primal = _welded_wedge_scene()
    primal = _solve_wedge(scene_primal, "deterministic", "none")
    leaf = (
        torch.tensor(_WELDED_WEDGE_VERTICES, dtype=torch.float32)
        .cuda()
        .requires_grad_(True)
    )
    ad = _solve_wedge(_welded_wedge_scene(vertices=leaf), "deterministic", "vjp")
    a_primal = _coefficients(primal, "deterministic")
    a_ad = _coefficients(ad, "deterministic").detach()
    assert a_primal.shape == a_ad.shape
    error = relative_error(a_ad, a_primal, abs_floor=ABS_TOL)
    assert error <= 1.0e-3, f"vertex-mode wedge tables drifted: {error}"


def test_wedge_forward_mode_matches_reverse():
    """JVP-vs-VJP inner-product duality on the wedge eps_r seed."""

    scene = _wedge_scene()
    compiled = compile_scene(scene, reference_frequency_hz=_FREQUENCY_HZ)
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
    compiled = compile_scene(scene, reference_frequency_hz=_FREQUENCY_HZ)
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
    scene = _coupled_scene()
    result = _solve_coupled(
        scene,
        "vjp",
        reference_frequency_hz=frequency,
    )
    _assert_coupled_rows(result)
    _loss(result, "path").backward()
    assert frequency.grad is not None
    grad = frequency.grad.detach()
    assert float(grad.abs()) > 0.0

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        fd_scene = _coupled_scene()
        return _loss(
            _solve_coupled(
                fd_scene,
                "none",
                reference_frequency_hz=float(value),
            ),
            "path",
        ).detach()

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


# ---------------------------------------------------------------------------
# Coupled double diffraction (component 7, coupled double diffraction). The coupled fixture
# (an off-mirror-symmetry box, _COUPLED_TX/_COUPLED_RX asymmetric, see the
# _COUPLED_TX note and trap 8) produces cid-7 rows: both interactions along a
# path are diffraction (type 2). These tests isolate the DD rows so the
# channel_field_coupled_dd_backward / _jvp companions are exercised directly.
# ---------------------------------------------------------------------------


def _dd_mask(result) -> torch.Tensor:
    # A DD (cid-7) path carries two diffraction (type 2) events and nothing
    # else, so both interaction slots equal 2.
    types = result.interaction_type
    return (types == 2).sum(dim=-1) == 2


def _assert_dd_rows(result) -> None:
    assert bool(_dd_mask(result).any()), (
        "the coupled fixture no longer produces double-diffraction (cid 7) paths"
    )


def _dd_loss(result) -> torch.Tensor:
    # Sum only the double-diffraction coefficients so the loss depends solely on
    # the cid-7 rows (drops the trailing per-antenna singleton of result.a).
    coefficient = result.a[..., 0][_dd_mask(result)]
    return coefficient.real.sum() + 0.5 * coefficient.imag.sum()


# Same box geometry as coupled_wall_wedge_scene, but with a live vertices leaf
# so the coupled mesh-vertex refusal (coupled double diffraction) can be exercised. Kept off any
# mirror symmetry via the asymmetric TX/RX (trap 8).
_COUPLED_VERTICES = (
    (-5.0, -5.0, 0.0),
    (5.0, -5.0, 0.0),
    (5.0, 5.0, 0.0),
    (-5.0, 5.0, 0.0),
    (2.0, -1.0, 2.0),
    (2.0, 1.0, 2.0),
    (4.0, -1.0, 2.0),
    (4.0, 1.0, 2.0),
    (2.0, -1.0, 4.0),
    (2.0, 1.0, 4.0),
)
_COUPLED_FACES = (
    [0, 1, 2],
    [0, 2, 3],
    [4, 6, 7],
    [4, 7, 5],
    [4, 5, 9],
    [4, 9, 8],
)


def _coupled_vertex_scene(vertices: torch.Tensor) -> Scene:
    structure = Structure(
        geometry=Mesh(
            vertices,
            torch.tensor(
                _COUPLED_FACES,
                dtype=torch.int32,
                device=vertices.device,
            ),
            recenter=False,
            fill_mode="surface",
            topology_diagnostics=False,
        ),
        material=PhysicalMaterial(**_WEDGE_MATERIAL),
        surface_id=0,
    )
    return Scene(
        structures=[structure],
        endpoints=[
            make_transmitter(position=torch.tensor(_COUPLED_TX)),
            make_receiver(position=torch.tensor(_COUPLED_RX)),
        ],
    )


@pytest.mark.parametrize("ad_mode", ("vjp", "jvp"))
def test_coupled_dd_runs_under_ad_modes(ad_mode):
    """The cid-7 field op dispatches through its AD Function in both AD modes."""

    scene = _coupled_scene()
    result = _solve_coupled(scene, ad_mode)
    _assert_dd_rows(result)
    assert bool(torch.isfinite(result.a).all())


@pytest.mark.parametrize("endpoint", ("tx", "rx"))
def test_coupled_dd_endpoint_position_grad_matches_fd(endpoint):
    """DD-only endpoint (tx AND rx) grads vs central differences of the primal.

 tx/rx gradients flow through the native per-leg re-anchoring inside the
 coupled_dd field kernel; Q1/Q2 and the edge bounds are frozen (coupled double diffraction).
 """

    base = torch.tensor(
        _COUPLED_TX if endpoint == "tx" else _COUPLED_RX, dtype=torch.float32
    )
    leaf = base.clone().cuda().requires_grad_(True)
    scene = _coupled_scene(**{endpoint: leaf})
    result = _solve_coupled(scene, "vjp")
    _assert_dd_rows(result)
    _dd_loss(result).backward()
    assert leaf.grad is not None
    assert float(leaf.grad.abs().max()) > 0.0

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        fd_scene = _coupled_scene(**{endpoint: values.clone()})
        return _dd_loss(_solve_coupled(fd_scene, "none")).detach()

    expected = central_difference_gradient(evaluate, base, FD_STEP_POSITION_PHASE)
    assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_coupled_dd_forward_mode_matches_reverse():
    """JVP-vs-VJP inner-product duality on the wedge eps_r seed for cid-7 rows.

 Fires channel_field_coupled_dd_jvp (forward dual) and channel_field_coupled_dd_backward
 (reverse) and checks they agree on the DD-only loss.
 """

    scene = _coupled_scene()
    compiled = compile_scene(scene, reference_frequency_hz=_FREQUENCY_HZ)
    base = compiled.materials.eps_r
    tangent = torch.ones_like(base)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(base.detach().clone(), tangent)
        object.__setattr__(compiled.materials, "eps_r", dual)
        try:
            result = _solve_coupled(scene, "jvp")
            _assert_dd_rows(result)
            mask = _dd_mask(result)
            tangent_a = torch.autograd.forward_ad.unpack_dual(result.a[..., 0]).tangent
        finally:
            object.__setattr__(compiled.materials, "eps_r", base)
    assert tangent_a is not None
    tangent_dd = tangent_a[mask]
    jvp = tangent_dd.real.sum() + 0.5 * tangent_dd.imag.sum()

    base.requires_grad_(True)
    try:
        _dd_loss(_solve_coupled(scene, "vjp")).backward()
        assert base.grad is not None
        vjp = (base.grad * tangent).sum()
    finally:
        base.requires_grad_(False)
        base.grad = None
    assert float(jvp.abs()) > 0.0
    assert relative_error(jvp, vjp, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_coupled_dd_mesh_vertex_grad_raises_loudly():
    """A live mesh-vertex leaf through the coupled family fails loudly (coupled double diffraction).

 The coupled stationary re-solve and the coupled/DD field adjoints treat the
 edge tables as frozen winners, so d(coupled)/d(vertices) would be silently
 missing; evaluation.py refuses instead of returning an incomplete gradient.
 """

    vertices = (
        torch.tensor(_COUPLED_VERTICES, dtype=torch.float32).cuda().requires_grad_(True)
    )
    scene = _coupled_vertex_scene(vertices)
    with pytest.raises(NotImplementedError, match="double-diffraction"):
        _solve_coupled(scene, "vjp")