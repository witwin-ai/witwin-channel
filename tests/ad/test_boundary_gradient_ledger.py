# Copyright Xingyu Chen.
# Measures how much field change along a trajectory the fixed-winner gradient misses.

"""Trajectory ledger: integrate the geometry gradient and compare it with the primal field change."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as deterministic_solve
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as path_solve
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for solver AD"
)

# The ledger rests on one identity. For a field E(s) along a trajectory,
# E(s1) - E(s0) equals the integral of dE/ds plus the sum of every jump
# E(b+) - E(b-) at a boundary b. Fixed-winner AD returns only the smooth dE/ds,
# so (primal change) - (integrated gradient) is exactly the boundary term it
# drops. A trapezoid rule estimates the integral; its error per interval is
# about (k h)^3 / 12 of |E|, which is 2e-5 at 3 GHz and h = 1 mm, so a residual
# that survives refinement is a jump, not quadrature error.
_FREQUENCY_HZ = 3.0e9
_HALF_LENGTH_M = 0.1
_SAMPLES = 201
_REFINE_SAMPLES = 33
# Every case crosses its boundary at s = 0. The grid is shifted off that point
# so no sample sits exactly on a boundary, where the field takes a third value
# that belongs to neither one-sided limit.
_GRID_OFFSET_M = 3.7e-4
# A coarse interval is a jump candidate when its residual exceeds this share of
# the trajectory's peak |E|; it is confirmed when the residual summed over its
# refined sub-intervals still exceeds half of it.
_JUMP_FLAG = 5.0e-3
# Away from jumps the integrated gradient must close the primal change to this
# share of the peak |E|, a 5x margin over the measured quadrature error.
_INTERIOR_TOL = 1.0e-2
# AD-mode diffraction re-evaluates the field, so its primal may differ from the
# none-mode primal by float32 rounding.
_PARITY_TOL = 1.0e-4

_TX_REFLECT = (0.0, -1.0, 0.5)
_RX_REFLECT = (0.0, 1.0, 0.5)
_TX_SHADOW = (0.0, -2.0, 0.5)
_RX_SHADOW = (0.0, 2.0, 0.5)
_Y = (0.0, 1.0, 0.0)
_X = (1.0, 0.0, 0.0)
# Wall at x = 2.5 split along the v1-v2 diagonal. With the reflection
# transmitter, a receiver at (0, 1 + s, 0.5) reflects at (2.5, s / 2, 0.5).
# Top edge at z = 2.4 keeps the diagonal 0.35 m from that point.
_WALL_DIAGONAL_FAR = ((2.5, -3.0, -1.0), (2.5, 3.0, -1.0), (2.5, -3.0, 2.4), (2.5, 3.0, 2.4))
# Top edge at z = 2.0 puts the diagonal through (2.5, 0, 0.5).
_WALL_DIAGONAL_CROSSED = ((2.5, -3.0, -1.0), (2.5, 3.0, -1.0), (2.5, -3.0, 2.0), (2.5, 3.0, 2.0))
# The wall ends at y = 0, a reflection shadow boundary (RSB) at s = 0 for a
# moving receiver and for a wall moving along +y alike.
_WALL_HALF = ((2.5, -3.0, -1.0), (2.5, 0.0, -1.0), (2.5, -3.0, 2.4), (2.5, 0.0, 2.4))
# Screen in the plane y = 0 ending at x = 0. A receiver at (s, 2, 0.5) sees
# the transmitter for s > 0: an incident shadow boundary (ISB) at s = 0.
_SCREEN = ((-3.0, 0.0, -1.0), (0.0, 0.0, -1.0), (-3.0, 0.0, 2.4), (0.0, 0.0, 2.4))


# With these faces the wall normal is +x, facing away from the reflection
# transmitter, and the screen normal is -y, facing the shadow transmitter.
# The flipped winding reverses both, which changes how the open edge is read.
_FACES = ((0, 1, 2), (1, 3, 2))
_FACES_FLIPPED = ((0, 2, 1), (1, 2, 3))


@dataclass(frozen=True)
class Case:
    """One trajectory: either the receiver or every wall vertex moves by s along `direction`."""

    wall: tuple[tuple[float, float, float], ...]
    tx: tuple[float, float, float]
    rx: tuple[float, float, float]
    direction: tuple[float, float, float]
    components: frozenset[str]
    moves: str
    faces: tuple[tuple[int, int, int], ...] = _FACES


_REFLECTION = frozenset({"reflection"})
_REFLECTION_DIFFRACTION = frozenset({"reflection", "diffraction"})
_LOS_DIFFRACTION = frozenset({"los", "diffraction"})
_CASES = {
    "control-rx": Case(_WALL_DIAGONAL_FAR, _TX_REFLECT, _RX_REFLECT, _Y, _REFLECTION, "rx"),
    "diagonal-rx": Case(_WALL_DIAGONAL_CROSSED, _TX_REFLECT, _RX_REFLECT, _Y, _REFLECTION, "rx"),
    "diagonal-wall": Case(
        _WALL_DIAGONAL_CROSSED, _TX_REFLECT, _RX_REFLECT, _Y, _REFLECTION, "wall"
    ),
    "rsb-rx": Case(_WALL_HALF, _TX_REFLECT, _RX_REFLECT, _Y, _REFLECTION, "rx"),
    "rsb-rx-diffraction": Case(
        _WALL_HALF, _TX_REFLECT, _RX_REFLECT, _Y, _REFLECTION_DIFFRACTION, "rx"
    ),
    "rsb-rx-diffraction-flipped": Case(
        _WALL_HALF, _TX_REFLECT, _RX_REFLECT, _Y, _REFLECTION_DIFFRACTION, "rx", _FACES_FLIPPED
    ),
    "rsb-wall": Case(_WALL_HALF, _TX_REFLECT, _RX_REFLECT, _Y, _REFLECTION, "wall"),
    "rsb-wall-diffraction": Case(
        _WALL_HALF, _TX_REFLECT, _RX_REFLECT, _Y, _REFLECTION_DIFFRACTION, "wall"
    ),
    "isb-rx": Case(_SCREEN, _TX_SHADOW, _RX_SHADOW, _X, frozenset({"los"}), "rx"),
    "isb-rx-diffraction": Case(_SCREEN, _TX_SHADOW, _RX_SHADOW, _X, _LOS_DIFFRACTION, "rx"),
    "isb-rx-diffraction-flipped": Case(
        _SCREEN, _TX_SHADOW, _RX_SHADOW, _X, _LOS_DIFFRACTION, "rx", _FACES_FLIPPED
    ),
}
# The field is continuous along these trajectories, so the fixed-winner
# gradient must account for the whole primal change.
_CONTINUOUS = (
    "control-rx", "diagonal-rx", "diagonal-wall", "isb-rx-diffraction", "isb-rx-diffraction-flipped",
    "rsb-rx-diffraction-flipped", "rsb-wall-diffraction",
)
# Geometric optics alone: a row appears or disappears and nothing else covers
# the jump. Fixed-winner AD drops the jump by design.
_GO_ONLY = ("rsb-rx", "rsb-wall", "isb-rx")
# Geometric optics plus edge diffraction should be continuous across both
# shadow boundaries. With the wall's normal facing away from the transmitter a
# residual of about 9 % of the peak is left within micrometres of the
# reflection boundary, where the reflection's in-face test and the
# diffraction's float band choose their sides independently.
_DIFFRACTION_GAPS = (
    "rsb-rx-diffraction",
)


@dataclass(frozen=True)
class Ledger:
    peak: float
    primal_change: complex
    boundary_term: complex
    interior_residual: float
    jumps_s: tuple[float, ...]
    parity: float
    boundary_parity: float

    def describe(self, name: str, solver: str) -> str:
        return (
            f"{name}/{solver}: peak|E|={self.peak:.3e} "
            f"|dE|/peak={abs(self.primal_change) / self.peak:.3e} "
            f"|missed|/peak={abs(self.boundary_term) / self.peak:.3e} "
            f"interior/peak={self.interior_residual / self.peak:.3e} "
            f"jumps_at_s={[round(s, 5) for s in self.jumps_s]} parity/peak={self.parity:.3e} "
            f"boundary_parity/peak={self.boundary_parity:.3e}"
        )


def _scene(case: Case, vertices: torch.Tensor, receivers: torch.Tensor) -> Scene:
    wall = Structure(
        Mesh(
            vertices, torch.tensor(case.faces, device=vertices.device), recenter=False,
            fill_mode="surface", topology_diagnostics=False,
        ),
        PhysicalMaterial(eps_r=4.0, sigma_e=0.02),
        name="ledger-wall",
        surface_id=1,
    )
    endpoints = [make_transmitter(position=torch.tensor(case.tx))]
    endpoints += [make_receiver(position=receivers[i]) for i in range(receivers.shape[0])]
    return Scene(structures=[wall], endpoints=endpoints)


def _field(case: Case, solver: str, scene: Scene, ad_mode: str) -> torch.Tensor:
    """Coherent complex field at each receiver, shape [N]."""

    if solver == "path":
        result = path_solve(
            scene, PathConfig(max_depth=1, components=case.components, ad_mode=ad_mode),
            reference_frequency_hz=_FREQUENCY_HZ,
        )
        return result.a[:, 0, 0, 0, :, 0].sum(dim=-1)
    result = deterministic_solve(
        scene, DeterministicConfig(max_depth=1, components=case.components, ad_mode=ad_mode),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    return result.field[0]


def _field_and_slope(case: Case, solver: str, leaf: torch.Tensor, scene: Scene):
    """AD-mode field and its derivative along the trajectory, both complex128 on the host."""

    direction = torch.tensor(case.direction, dtype=torch.float32, device="cuda")
    field = _field(case, solver, scene, "vjp")
    if field.requires_grad:
        (grad_re,) = torch.autograd.grad(field.real.sum(), leaf, retain_graph=True)
        (grad_im,) = torch.autograd.grad(field.imag.sum(), leaf)
    else:
        # No row reaches the receiver, so the field is a constant zero with no graph.
        grad_re = torch.zeros_like(leaf)
        grad_im = torch.zeros_like(leaf)
    # A receiver leaf is [N, 3] with one receiver per row; a wall leaf is [4, 3]
    # and every vertex moves along the same direction, so its slope is the sum.
    slope_re = grad_re @ direction
    slope_im = grad_im @ direction
    if case.moves == "wall":
        slope_re = slope_re.sum(dim=0, keepdim=True)
        slope_im = slope_im.sum(dim=0, keepdim=True)
    slope = slope_re.double().cpu() + 1j * slope_im.double().cpu()
    return field.detach().to(torch.complex128).cpu(), slope


def _samples(case: Case, solver: str, s: torch.Tensor):
    """None-mode primal, d(primal)/ds, and the largest AD/none primal gap at each s."""

    direction = torch.tensor(case.direction, dtype=torch.float64)
    base_wall = torch.tensor(case.wall, dtype=torch.float64)
    rx = torch.tensor(case.rx, dtype=torch.float64)
    if case.moves == "rx":
        receivers = (rx + s[:, None] * direction).to(torch.float32).cuda()
        vertices = base_wall.to(torch.float32).cuda()
        primal = _field(case, solver, _scene(case, vertices, receivers), "none")
        primal = primal.detach().to(torch.complex128).cpu()
        leaf = receivers.clone().requires_grad_(True)
        field, slope = _field_and_slope(case, solver, leaf, _scene(case, vertices, leaf))
        return primal, slope, float((field - primal).abs().max())

    # A wall leaf belongs to one scene, so each sample is its own pair of solves.
    receivers = rx[None].to(torch.float32).cuda()
    primals, slopes, parity = [], [], 0.0
    for value in s.tolist():
        vertices = (base_wall + value * direction).to(torch.float32).cuda()
        primal = _field(case, solver, _scene(case, vertices, receivers), "none")
        primal = primal.detach().to(torch.complex128).cpu()
        leaf = vertices.clone().requires_grad_(True)
        field, slope = _field_and_slope(case, solver, leaf, _scene(case, leaf, receivers))
        primals.append(primal)
        slopes.append(slope)
        parity = max(parity, float((field - primal).abs().max()))
    return torch.cat(primals), torch.cat(slopes), parity


def _residuals(case: Case, solver: str, s: torch.Tensor):
    """Per-interval (primal change - trapezoid of dE/ds)."""

    primal, slope, parity = _samples(case, solver, s)
    step = s[1:] - s[:-1]
    residual = (primal[1:] - primal[:-1]) - 0.5 * step * (slope[1:] + slope[:-1])
    return primal, residual, parity


def _ledger(case: Case, solver: str) -> Ledger:
    s = torch.linspace(-_HALF_LENGTH_M, _HALF_LENGTH_M, _SAMPLES, dtype=torch.float64)
    s = s + _GRID_OFFSET_M
    primal, residual, parity = _residuals(case, solver, s)
    peak = float(primal.abs().max())

    boundary = 0.0j
    interior = 0.0
    # Refined samples land within micrometres of a boundary, where the diffraction
    # field is ill-conditioned; their AD/none gap is reported apart from the grid's.
    boundary_parity = 0.0
    jumps: list[float] = []
    for i in range(residual.shape[0]):
        r = complex(residual[i])
        if abs(r) <= _JUMP_FLAG * peak:
            interior += abs(r)
            continue
        # Refinement separates a jump, whose residual does not shrink with the
        # step, from a steep but continuous field, whose residual does.
        fine_s = torch.linspace(float(s[i]), float(s[i + 1]), _REFINE_SAMPLES, dtype=torch.float64)
        _, fine, fine_parity = _residuals(case, solver, fine_s)
        boundary_parity = max(boundary_parity, fine_parity)
        refined = complex(fine.sum())
        if abs(refined) > 0.5 * _JUMP_FLAG * peak:
            boundary += refined
            jumps.append(0.5 * float(s[i] + s[i + 1]))
        else:
            interior += abs(refined)
    return Ledger(
        peak=peak,
        primal_change=complex(primal[-1] - primal[0]),
        boundary_term=boundary,
        interior_residual=interior,
        jumps_s=tuple(jumps),
        parity=parity / peak,
        boundary_parity=boundary_parity / peak,
    )


_LEDGERS: dict[tuple[str, str], Ledger] = {}


def _cached_ledger(name: str, solver: str) -> Ledger:
    key = (name, solver)
    if key not in _LEDGERS:
        _LEDGERS[key] = _ledger(_CASES[name], solver)
        print(_LEDGERS[key].describe(name, solver))
    return _LEDGERS[key]


@pytest.mark.parametrize("solver", ("deterministic", "path"))
@pytest.mark.parametrize("name", tuple(_CASES))
def test_gradient_closes_the_primal_change_away_from_boundaries(name, solver):
    ledger = _cached_ledger(name, solver)
    assert ledger.parity <= _PARITY_TOL, ledger.describe(name, solver)
    assert ledger.interior_residual <= _INTERIOR_TOL * ledger.peak, ledger.describe(name, solver)


@pytest.mark.parametrize("solver", ("deterministic", "path"))
@pytest.mark.parametrize("name", _CONTINUOUS)
def test_a_continuous_trajectory_has_no_boundary_term(name, solver):
    ledger = _cached_ledger(name, solver)
    assert not ledger.jumps_s, ledger.describe(name, solver)
    assert abs(ledger.boundary_term) <= _INTERIOR_TOL * ledger.peak, ledger.describe(name, solver)


@pytest.mark.xfail(
    strict=True,
    reason="fixed-winner AD has no boundary term; a row birth or death is outside its contract",
)
@pytest.mark.parametrize("solver", ("deterministic", "path"))
@pytest.mark.parametrize("name", _GO_ONLY)
def test_a_geometric_optics_jump_is_covered_by_the_gradient(name, solver):
    ledger = _cached_ledger(name, solver)
    assert abs(ledger.boundary_term) <= _INTERIOR_TOL * ledger.peak, ledger.describe(name, solver)


@pytest.mark.xfail(
    strict=True,
    reason="edge diffraction does not cancel the geometric-optics jump at this shadow boundary",
)
@pytest.mark.parametrize("solver", ("deterministic", "path"))
@pytest.mark.parametrize("name", _DIFFRACTION_GAPS)
def test_diffraction_makes_the_shadow_boundary_continuous(name, solver):
    ledger = _cached_ledger(name, solver)
    assert abs(ledger.boundary_term) <= _INTERIOR_TOL * ledger.peak, ledger.describe(name, solver)
