# Copyright Xingyu Chen.
# Self-consistency: reciprocity, subdivision and scheme invariance, and four-solver agreement.

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from witwin.core import PhysicalMaterial, Scene

from benchmarks import synthetic_city as sc
from tests.accuracy import oracles as O
from tests.accuracy.scenes import (
    FREQ,
    antenna,
    coefficient_tolerance,
    grid,
    surface,
)
from tests.accuracy.test_reference import channel_edge_coefficient, _half_plane, HALF_PLANE_S, \
    HALF_PLANE_S_PRIME

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Channel solves need CUDA")

# Summing a few thousand float32 path powers in a different order moves the total by at most
# n * 2^-24 relative (2e-4 for 3,000 rows); 1e-3 leaves a 5x margin.
SUM_TOLERANCE = 1e-3


def _pair_matrix(result) -> torch.Tensor:
    """Path solver ``[rx, tx]`` matrix of coherent sums and of the incoherent amplitude sums."""
    a = result.a[:, 0, :, 0, :, 0]
    valid = result.valid[:, 0, :, 0, :]
    a = torch.where(valid, a, torch.zeros_like(a))
    return torch.stack([a.sum(-1), a.abs().sum(-1).to(a.dtype)])


@pytest.mark.parametrize("components", [
    pytest.param({"los", "reflection"}, id="lr"),
    pytest.param({"los", "reflection", "diffraction"}, id="lrd", marks=pytest.mark.xfail(
        strict=True, reason="A-4 (M0-3): diffraction is not reciprocal under a Tx/Rx swap")),
])
def test_reciprocity_swap_tx_rx(accuracy, components):
    from witwin.channel.path import Config, solve

    city = sc.city(4)
    points = [p.position.tolist() for p in sc.receiver_points(city.extent_m, 6, seed=3)]
    a_side, b_side = points[:3], points[3:]
    forward = Scene(structures=city.structures,
                    endpoints=[*(antenna("tx", p) for p in a_side), *(antenna("rx", p) for p in b_side)])
    backward = Scene(structures=city.structures,
                     endpoints=[*(antenna("tx", p) for p in b_side), *(antenna("rx", p) for p in a_side)])
    config = Config(components=components, max_depth=1)
    m_f = _pair_matrix(solve(forward, config, reference_frequency_hz=FREQ))  # [2, rx=b, tx=a]
    m_b = _pair_matrix(solve(backward, config, reference_frequency_hz=FREQ))  # [2, rx=a, tx=b]
    scale = m_f[1].real
    error = ((m_f[0] - m_b[0].transpose(0, 1)).abs() / scale.clamp_min(1e-30)).max().item()
    accuracy.check(f"{sorted(components)}", error, coefficient_tolerance(4 * sc.PITCH_M * 3))


def _radiomap(subdiv: int, welded: bool, components: set[str], depth: int = 1) -> torch.Tensor:
    from witwin.channel.deterministic import Config, solve

    city = sc.city(4, subdiv=subdiv, welded=welded)
    scene = Scene(structures=city.structures,
                  endpoints=[sc.transmitter(city.extent_m), sc.receiver_grid(city.extent_m, 32)])
    result = solve(scene, Config(components=components, max_depth=depth, coherent=False),
                   reference_frequency_hz=FREQ)
    return result.path_gain.reshape(-1).double()


C6 = pytest.mark.xfail(strict=True, reason="C-6: diffraction depends on mesh subdivision")


@pytest.mark.parametrize("components,subdiv,welded", [
    pytest.param({"reflection"}, 1, True, id="reflection-s1"),
    pytest.param({"reflection"}, 3, True, id="reflection-s3"),
    pytest.param({"diffraction"}, 1, True, id="diffraction-s1", marks=C6),
    pytest.param({"diffraction"}, 3, True, id="diffraction-s3", marks=C6),
    pytest.param({"diffraction"}, 1, False, id="diffraction-s1-unwelded", marks=C6),
])
def test_subdivision_invariance(accuracy, components, subdiv, welded):
    """Subdividing every triangle 4-way keeps the geometry identical, so the radiomap must not move.

    Tolerance: the subdivided vertices are float64 midpoints rounded to float32, so geometry
    differs by at most an ulp (path power changes below 1e-5), plus ``SUM_TOLERANCE`` for the
    different number and order of rows summed per receiver.
    """

    base = _radiomap(0, True, components)
    other = _radiomap(subdiv, welded, components)
    error = ((other - base).abs().sum() / base.sum()).item()
    accuracy.check(f"{sorted(components)} subdiv={subdiv} welded={welded}", error, SUM_TOLERANCE,
                   base_total=base.sum().item(), other_total=other.sum().item())


def _per_depth(max_depth: int) -> dict[int, tuple[int, float]]:
    from witwin.channel.deterministic import Config, solve

    city = sc.city(4)
    scene = Scene(structures=city.structures,
                  endpoints=[sc.transmitter(city.extent_m), sc.receiver_grid(city.extent_m, 64)])
    paths = solve(scene, Config(components={"reflection"}, max_depth=max_depth, export_paths=True),
                  reference_frequency_hz=FREQ).paths
    depth, gain = paths.depth.cpu(), paths.path_gain.double().cpu()
    return {d: (int((depth == d).sum()), float(gain[depth == d].sum())) for d in range(1, max_depth + 1)}


@pytest.fixture(scope="module")
def scheme_depths() -> tuple[dict[int, tuple[int, float]], dict[int, tuple[int, float]]]:
    return _per_depth(3), _per_depth(4)


C1 = pytest.mark.xfail(strict=True, reason="C-1: exhaustive and ray-discovery schemes disagree")


@pytest.mark.parametrize("depth", [1, pytest.param(2, marks=C1), pytest.param(3, marks=C1)])
def test_scheme_equivalence_on_overlapping_depths(accuracy, scheme_depths, depth):
    """``max_depth=3`` (exhaustive sequences) and ``4`` (ray discovery) must agree on depths 1-3."""
    rows3, gain3 = scheme_depths[0][depth]
    rows4, gain4 = scheme_depths[1][depth]
    accuracy.record(f"depth {depth} rows", abs(rows4 - rows3), 0, rows_d3=rows3, rows_d4=rows4)
    accuracy.check(f"depth {depth} power", abs(gain4 - gain3) / gain3, SUM_TOLERANCE,
                   rows_d3=rows3, rows_d4=rows4)
    assert rows3 == rows4


# --------------------------------------------------------------------------- four solvers


GRID_ORIGIN, GRID_SHAPE, GRID_SPACING = (2.0, -3.0, 0.5), (4, 4), 2.0
TX_LOS, TX_REFLECTION = (0.0, 0.0, 0.5), (0.0, 0.0, 6.0)
WALL_X = 12.0


def _cell_centres() -> list[tuple[float, float, float]]:
    ox, oy, oz = GRID_ORIGIN
    # Grid results are flattened y-major (row j along the second axis, column i along the first).
    return [(ox + i * GRID_SPACING, oy + j * GRID_SPACING, oz)
            for j in range(GRID_SHAPE[1]) for i in range(GRID_SHAPE[0])]


def _wall() -> list:
    pec = PhysicalMaterial.perfect_conductor()
    v = [[WALL_X, -40.0, -20.0], [WALL_X, 40.0, -20.0], [WALL_X, -40.0, 40.0], [WALL_X, 40.0, 40.0]]
    return [surface(v, [[0, 2, 1], [1, 2, 3]], pec)]


def _solve_grid(solver: str, structures: list, tx: tuple[float, float, float],
                components: set[str]) -> torch.Tensor:
    import importlib

    module = importlib.import_module({
        "deterministic": "witwin.channel.deterministic", "mc": "witwin.channel.montecarlo.basic",
        "bdpt": "witwin.channel.montecarlo.bdpt"}[solver])
    extra = {"coherent": False} if solver == "deterministic" else {
        "samples": 1 << 23, "workspace_limit_bytes": 2 << 30} if solver == "mc" else {"samples": 4096}
    scene = Scene(structures=structures, endpoints=[antenna("tx", tx), grid(GRID_ORIGIN, GRID_SHAPE, GRID_SPACING)])
    result = module.solve(scene, module.Config(components=components, max_depth=1, **extra),
                          reference_frequency_hz=FREQ)
    return result.path_gain.reshape(-1).double().cpu()


def _solve_points(structures: list, tx: tuple[float, float, float], components: set[str]) -> torch.Tensor:
    from witwin.channel.path import Config, solve

    scene = Scene(structures=structures,
                  endpoints=[antenna("tx", tx), *(antenna("rx", c) for c in _cell_centres())])
    result = solve(scene, Config(components=components, max_depth=1), reference_frequency_hz=FREQ)
    a = torch.where(result.valid, result.a[..., 0], torch.zeros_like(result.a[..., 0]))
    return a.abs().square().reshape(len(_cell_centres()), -1).sum(-1).double().cpu()


def _image_power(tx, point, vector: bool) -> float:
    """PEC wall image-source power; ``vector`` adds the z-dipole polarization projections."""
    z = (0.0, 0.0, 1.0)
    if vector:
        value, _, _ = O.specular_reflection(tx, point, np.array([WALL_X, 0.0, 0.0]), (-1.0, 0.0, 0.0),
                                            FREQ, z, z, r_te=-1.0 + 0j, r_tm=1.0 + 0j)
        return abs(value) ** 2
    image = np.array([2 * WALL_X - tx[0], tx[1], tx[2]])
    return (O.wavelength(FREQ) / (4 * math.pi * np.linalg.norm(np.asarray(point) - image))) ** 2


def _cell_average(fn, centre, n: int = 32) -> float:
    offsets = (np.arange(n) + 0.5) / n - 0.5
    return float(np.mean([fn((centre[0] + u * GRID_SPACING, centre[1] + v * GRID_SPACING, centre[2]))
                          for u in offsets for v in offsets]))


def _mc_hits(tx, centre) -> float:
    """Expected Monte Carlo rays crossing one cell after the wall: ``samples * Omega / (4 pi)``."""
    image = np.array([2 * WALL_X - tx[0], tx[1], tx[2]])
    r_vec = np.asarray(centre) - image
    r = np.linalg.norm(r_vec)
    omega = GRID_SPACING**2 * abs(r_vec[2]) / r / r**2
    return (1 << 23) * omega / (4 * math.pi)


SOLVERS = ("deterministic", "path", "mc", "bdpt")


@pytest.mark.parametrize("solver", SOLVERS)
def test_four_solvers_line_of_sight(accuracy, solver):
    """Grid at the Tx height (polarization projection exactly 1, so scalar and vector models agree)."""
    centres = _cell_centres()
    reference = np.array([abs(O.free_space(TX_LOS, c, FREQ, (0, 0, 1), (0, 0, 1))) ** 2 for c in centres])
    value = (_solve_points([], TX_LOS, {"los"}) if solver == "path"
             else _solve_grid(solver, [], TX_LOS, {"los"})).numpy()
    accuracy.check(solver, float(np.max(np.abs(value / reference - 1.0))), 2.0 * 1e-4)


@pytest.mark.parametrize("solver", SOLVERS)
def test_four_solvers_single_pec_wall_reflection(accuracy, solver):
    """Enumerated solvers are compared with the vector image solution at each cell centre.

    Monte Carlo Basic is scalar (no polarization) and estimates the cell average, so it is compared
    with the scalar image power averaged over each cell; its tolerance is 5 standard deviations of
    a counting estimate, ``5 / sqrt(expected crossings)``, conservative for its lattice directions.
    """

    centres = _cell_centres()
    if solver == "mc":
        reference = np.array([_cell_average(lambda p: _image_power(TX_REFLECTION, p, False), c)
                              for c in centres])
        tolerance = float(np.max([5.0 / math.sqrt(_mc_hits(TX_REFLECTION, c)) for c in centres]))
    else:
        reference = np.array([_image_power(TX_REFLECTION, c, True) for c in centres])
        tolerance = 2.0 * 1e-4
    value = (_solve_points(_wall(), TX_REFLECTION, {"reflection"}) if solver == "path"
             else _solve_grid(solver, _wall(), TX_REFLECTION, {"reflection"})).numpy()
    accuracy.check(solver, float(np.max(np.abs(value / reference - 1.0))), tolerance)


@pytest.mark.xfail(strict=True, reason="A-3 (M0-3): screen diffraction depends on the face winding")
def test_pec_screen_winding_invariance(accuracy):
    """Flipping a PEC screen's triangle winding changes no physics, so the edge field must not move."""
    phi, phi_prime = math.radians(200.0), math.radians(60.0)
    front = channel_edge_coefficient(_half_plane(True), "soft", phi, phi_prime, HALF_PLANE_S_PRIME,
                                     HALF_PLANE_S)
    back = channel_edge_coefficient(_half_plane(False), "soft", phi, phi_prime, HALF_PLANE_S_PRIME,
                                    HALF_PLANE_S)
    accuracy.check("soft phi=200", abs(back - front) / abs(front), 1e-4)
