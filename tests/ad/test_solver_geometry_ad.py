"""AD-2/AD-4 solver-level tests: TX/RX position and mesh-vertex gradients.

Covers the plan 07 section 9.3 geometry cells for D=deterministic and P=path:
TX/RX position x (LoS, single reflection, transmission-multilayer), mesh
vertex x (LoS, single reflection, transmission-multilayer) and the coupled
mesh-vertex gap, each against central finite differences of the primal
solve. Geometry FD steps stay well inside the linear regime of the carrier
phase (k*h << 1 at 3 GHz), which is what forces a tighter step here than the
material tests use.
"""

from __future__ import annotations

import pytest
import torch

from tests.ad._fd import central_difference_gradient, relative_error
from tests.ad._tolerances import (
    ABS_TOL,
    FD_STEP_POSITION,
    FD_STEP_POSITION_PHASE,
    FD_STEP_VERTEX,
    REL_TOL_GENERAL,
)
from tests.support.scenes import transmission_wall_structure
from witwin.channel_native import ReceiverPoint, Scene, Structure, Transmitter
from witwin.channel_native.core.materials import Dielectric, Layer, PhysicalSurface
from witwin.channel_native.deterministic import Config as DeterministicConfig
from witwin.channel_native.deterministic import solve as deterministic_solve
from witwin.channel_native.path import Config as PathConfig
from witwin.channel_native.path import solve as path_solve

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for solver AD"
)

_SOLVERS = ("deterministic", "path")
_FREQUENCY_HZ = 3.0e9

_TX = (0.0, -1.0, 0.5)
_RX = (0.0, 1.0, 0.5)
# The wall quad is deliberately asymmetric in z (top edge at 2.4, not 2.0):
# with a symmetric quad the specular point (2.5, 0, 0.5) falls EXACTLY on the
# shared triangulation diagonal, where a +/-h vertex perturbation flips the
# native discovery between one and two duplicate winner paths. That is a path
# birth/death discontinuity (plan 07 section 7, explicitly outside the
# fixed-winner contract), and the central difference then measures the
# |coefficient|/2h jump instead of the derivative. The 0.2 m diagonal
# clearance keeps the winner topology stable across every FD probe so the FD
# is a valid derivative oracle.
_WALL_VERTICES = (
    (2.5, -3.0, -1.0),
    (2.5, 3.0, -1.0),
    (2.5, -3.0, 2.4),
    (2.5, 3.0, 2.4),
)
_TRANSMISSION_TX = (0.0, 0.0, 0.0)
_TRANSMISSION_RX = (6.0, 0.4, 0.2)


def _vec(values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _reflection_scene(
    tx: torch.Tensor | None = None,
    rx: torch.Tensor | None = None,
    vertices: torch.Tensor | None = None,
) -> Scene:
    wall = Structure(
        vertices=torch.tensor(_WALL_VERTICES) if vertices is None else vertices,
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=Dielectric(eps_r=4.0, sigma_e=0.02),
        name="ad-open-wall",
        surface_id=1,
    )
    return Scene(
        structures=[wall],
        transmitters=[Transmitter(position=_vec(_TX) if tx is None else tx)],
        receivers=[ReceiverPoint(position=_vec(_RX) if rx is None else rx)],
        frequency=_FREQUENCY_HZ,
    )


_TRANSMISSION_WALL_VERTICES = (
    (3.0, -4.0, -4.0),
    (3.0, 4.0, -4.0),
    (3.0, -4.0, 4.2),
    (3.0, 4.0, 4.2),
)


def _transmission_scene(
    tx: torch.Tensor | None = None,
    rx: torch.Tensor | None = None,
    vertices: torch.Tensor | None = None,
) -> Scene:
    material = PhysicalSurface(
        layers=(
            Layer(thickness_m=0.06, eps_r=4.0, sigma_e=0.02),
            Layer(thickness_m=0.09, eps_r=2.5, sigma_e=0.01),
        ),
        name="ad-thin-sheet",
    )
    if vertices is None:
        wall = transmission_wall_structure(3.0, material)
    else:
        wall = Structure(
            vertices=vertices,
            faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
            material=material,
            name="wall",
            surface_id=1,
        )
    return Scene(
        structures=[wall],
        transmitters=[
            Transmitter(position=_vec(_TRANSMISSION_TX) if tx is None else tx)
        ],
        receivers=[ReceiverPoint(position=_vec(_TRANSMISSION_RX) if rx is None else rx)],
        frequency=_FREQUENCY_HZ,
    )


def _los_scene(
    tx: torch.Tensor | None = None, rx: torch.Tensor | None = None
) -> Scene:
    return Scene(
        structures=[],
        transmitters=[Transmitter(position=_vec(_TX) if tx is None else tx)],
        receivers=[ReceiverPoint(position=_vec(_RX) if rx is None else rx)],
        frequency=_FREQUENCY_HZ,
    )


_SCENES = {
    "los": (_los_scene, frozenset({"los"})),
    "reflection": (_reflection_scene, frozenset({"reflection"})),
    "transmission": (_transmission_scene, frozenset({"transmission"})),
}


def _solve(scene: Scene, solver: str, components: frozenset[str], ad_mode: str):
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


def _loss(result, solver: str) -> torch.Tensor:
    coefficient = result.a if solver == "path" else result.paths.coefficient
    return coefficient.real.sum() + 0.5 * coefficient.imag.sum()


def _fd_endpoint_gradient(
    builder, solver: str, components: frozenset[str], *, endpoint: str, base
) -> torch.Tensor:
    def evaluate(values: torch.Tensor) -> torch.Tensor:
        scene = builder(**{endpoint: values.clone()})
        return _loss(_solve(scene, solver, components, "none"), solver).detach()

    return central_difference_gradient(evaluate, base, FD_STEP_POSITION_PHASE)


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize("interaction", ("los", "reflection", "transmission"))
@pytest.mark.parametrize("endpoint", ("tx", "rx"))
def test_endpoint_position_grad_matches_fd(solver, interaction, endpoint):
    builder, components = _SCENES[interaction]
    base = _vec(
        {
            ("los", "tx"): _TX,
            ("los", "rx"): _RX,
            ("reflection", "tx"): _TX,
            ("reflection", "rx"): _RX,
            ("transmission", "tx"): _TRANSMISSION_TX,
            ("transmission", "rx"): _TRANSMISSION_RX,
        }[(interaction, endpoint)]
    )

    leaf = base.clone().cuda().requires_grad_(True)
    scene = builder(**{endpoint: leaf})
    loss = _loss(_solve(scene, solver, components, "vjp"), solver)
    loss.backward()
    assert leaf.grad is not None

    expected = _fd_endpoint_gradient(
        builder, solver, components, endpoint=endpoint, base=base
    )
    assert (
        relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_mesh_vertex_grad_matches_fd(solver):
    components = frozenset({"reflection"})
    base = torch.tensor(_WALL_VERTICES, dtype=torch.float32)

    leaf = base.clone().cuda().requires_grad_(True)
    scene = _reflection_scene(vertices=leaf)
    loss = _loss(_solve(scene, solver, components, "vjp"), solver)
    loss.backward()
    assert leaf.grad is not None

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        scene = _reflection_scene(vertices=values.clone())
        return _loss(_solve(scene, solver, components, "none"), solver).detach()

    expected = central_difference_gradient(evaluate, base, FD_STEP_VERTEX)
    assert (
        relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    )


@pytest.mark.parametrize("interaction", ("los", "reflection"))
def test_forward_mode_endpoint_dual_matches_reverse(interaction):
    """JVP-vs-VJP inner-product duality on the TX position seed."""

    builder, components = _SCENES[interaction]
    base = _vec(_TX).cuda()
    generator = torch.Generator(device="cpu").manual_seed(17)
    tangent = torch.randn(3, generator=generator).cuda()

    with torch.autograd.forward_ad.dual_level():
        dual_tx = torch.autograd.forward_ad.make_dual(base, tangent)
        scene = builder(tx=dual_tx)
        loss = _loss(_solve(scene, "deterministic", components, "jvp"), "deterministic")
        jvp = torch.autograd.forward_ad.unpack_dual(loss).tangent
    assert jvp is not None

    leaf = base.clone().requires_grad_(True)
    scene = builder(tx=leaf)
    _loss(_solve(scene, "deterministic", components, "vjp"), "deterministic").backward()
    vjp = (leaf.grad * tangent).sum()
    assert relative_error(jvp, vjp, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_geometry_ad_is_inert_without_geometry_leaves():
    """Materials-only AD must not pay for the AD-2 reconstruction."""

    scene = _reflection_scene()
    primal = _solve(scene, "deterministic", frozenset({"reflection"}), "none")
    ad = _solve(scene, "deterministic", frozenset({"reflection"}), "vjp")
    assert torch.equal(primal.paths.coefficient, ad.paths.coefficient)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_mesh_vertex_transmission_grad_matches_fd(solver):
    """Mesh vertex x transmission (plan 07 section 9.3).

    A straight tx->rx penetration path never moves with the wall vertices
    (the crossing point is not a path parameter) and its path length is
    vertex-independent, but the incidence COSINE is not: the wall normal is
    a function of the vertices, and the layer-stack response is evaluated
    at that angle. Because no carrier phase rides on the vertices here, the
    FD step is NOT bound by k*h << 1; it uses the coarser incoherent step,
    which the tilt response needs to clear the float32 forward noise floor
    (at 1e-3 the two-point difference of the ~1e-5-scale gradient is only
    a few float32 ulps of the loss and reads reassociation noise).
    """

    components = frozenset({"transmission"})
    base = torch.tensor(_TRANSMISSION_WALL_VERTICES, dtype=torch.float32)

    leaf = base.clone().cuda().requires_grad_(True)
    scene = _transmission_scene(vertices=leaf)
    _loss(_solve(scene, solver, components, "vjp"), solver).backward()
    assert leaf.grad is not None
    assert float(leaf.grad.abs().max()) > 0.0

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        fd_scene = _transmission_scene(vertices=values.clone())
        return _loss(_solve(fd_scene, solver, components, "none"), solver).detach()

    expected = central_difference_gradient(evaluate, base, FD_STEP_POSITION)
    assert (
        relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_mesh_vertex_los_grad_is_structurally_zero(solver):
    """Mesh vertex x LoS: the exact zero, pinned.

    A LoS path touches no face: its coefficient depends on the endpoints and
    the frequency only, and the wall's sole influence (blocking the path) is
    a discrete frozen visibility winner. The true fixed-topology derivative
    is therefore identically zero. No graph edge can reach the vertex leaf,
    so the loss itself carries no graph (the endpoints are not live) and
    the leaf's gradient is the structural zero. Anything else (a live loss
    with a nonzero vertex gradient) would be a regression.
    """

    wall_vertices = torch.tensor(_WALL_VERTICES, dtype=torch.float32)
    leaf = wall_vertices.clone().cuda().requires_grad_(True)
    # The wall sits at x = 2.5; the tx->rx segment runs along x = 0 and never
    # touches it, so the LoS row exists and is unobstructed.
    scene = _reflection_scene(vertices=leaf)
    loss = _loss(_solve(scene, solver, frozenset({"los"}), "vjp"), solver)
    if loss.requires_grad:
        loss.backward()
        assert leaf.grad is None or bool((leaf.grad == 0.0).all())
    else:
        # The graph never reached any live leaf: the vertex gradient is the
        # exact structural zero.
        assert leaf.grad is None


def _coupled_scene_with_vertices(leaf: torch.Tensor) -> Scene:
    from witwin.channel_native.core.materials import PerfectConductor

    return Scene(
        structures=[
            Structure(
                vertices=leaf,
                faces=torch.tensor(
                    [[0, 1, 2], [0, 2, 3], [4, 6, 7], [4, 7, 5], [4, 5, 9], [4, 9, 8]],
                    dtype=torch.int32,
                ),
                material=PerfectConductor(),
                surface_id=0,
            )
        ],
        transmitters=[Transmitter(position=torch.tensor([0.4, -2.2, 1.15]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.55, 2.3, 4.8]))],
        frequency=_FREQUENCY_HZ,
    )


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mesh vertex x coupled R-D is not wired (plan 07 section 9.3 known "
        "gap): the coupled stationary re-solve and the coupled field "
        "adjoints take the wall plane and the edge tables as frozen "
        "winners, so the solver fails loudly instead of returning a "
        "silently incomplete vertex gradient"
    ),
)
def test_mesh_vertex_coupled_grad_exists():
    leaf = (
        torch.tensor(_COUPLED_VERTICES, dtype=torch.float32)
        .cuda()
        .requires_grad_(True)
    )
    result = path_solve(
        _coupled_scene_with_vertices(leaf),
        PathConfig(
            max_depth=2,
            components=frozenset({"reflection", "diffraction"}),
            coupled_paths=True,
            ad_mode="vjp",
        ),
    )
    loss = result.a.real.sum() + 0.5 * result.a.imag.sum()
    loss.backward()
    assert leaf.grad is not None
    assert float(leaf.grad.abs().max()) > 0.0


def test_mesh_vertex_coupled_fails_loudly():
    """The coupled + live-vertex combination must refuse, not silently zero."""

    leaf = (
        torch.tensor(_COUPLED_VERTICES, dtype=torch.float32)
        .cuda()
        .requires_grad_(True)
    )
    with pytest.raises(NotImplementedError, match="vertex"):
        path_solve(
            _coupled_scene_with_vertices(leaf),
            PathConfig(
                max_depth=2,
                components=frozenset({"reflection", "diffraction"}),
                coupled_paths=True,
                ad_mode="vjp",
            ),
        )
