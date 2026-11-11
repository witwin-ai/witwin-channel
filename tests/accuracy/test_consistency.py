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
    pytest.param({"los", "reflection", "diffraction"}, id="lrd"),
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


def test_coupled_reflection_diffraction_is_reciprocal(accuracy):
    """The coupled R-D / D-R rows (one reflection and one diffraction) are reciprocal under a swap.

    Each direction's coupled rows are summed per pair; an R-D row of one direction is the D-R row of
    the other. Double-diffraction rows are excluded: their discovery prunes the two edges
    asymmetrically, which is a candidate-set question, not the coupled field.
    """
    from witwin.channel.path import Config, solve

    city = sc.city(2)
    points = [p.position.tolist() for p in sc.receiver_points(city.extent_m, 4, seed=5)]
    a_side, b_side = points[:2], points[2:]

    def coupled_pairs(tx_points, rx_points):
        scene = Scene(structures=city.structures,
                      endpoints=[*(antenna("tx", p) for p in tx_points), *(antenna("rx", p) for p in rx_points)])
        config = Config(components={"los", "reflection", "diffraction"}, max_depth=2, coupled_paths=True)
        result = solve(scene, config, reference_frequency_hz=FREQ)
        types = result.types[:, 0, :, 0, :]
        mixed = result.valid[:, 0, :, 0, :] & (types == 1).any(-1) & (types == 2).any(-1)
        a = torch.where(mixed, result.a[:, 0, :, 0, :, 0], torch.zeros_like(result.a[:, 0, :, 0, :, 0]))
        return torch.stack([a.sum(-1), a.abs().sum(-1).to(a.dtype)])

    forward = coupled_pairs(a_side, b_side)
    backward = coupled_pairs(b_side, a_side)
    scale = forward[1].real.clamp_min(1e-30)
    assert float(forward[1].real.min()) > 0.0
    error = ((forward[0] - backward[0].transpose(0, 1)).abs() / scale).max().item()
    accuracy.check("coupled R-D/D-R rows", error, coefficient_tolerance(2 * sc.PITCH_M * 3))


def _radiomap(subdiv: int, welded: bool, components: set[str], depth: int = 1) -> torch.Tensor:
    from witwin.channel.deterministic import Config, solve

    city = sc.city(4, subdiv=subdiv, welded=welded)
    scene = Scene(structures=city.structures,
                  endpoints=[sc.transmitter(city.extent_m), sc.receiver_grid(city.extent_m, 32)])
    result = solve(scene, Config(components=components, max_depth=depth, coherent=False),
                   reference_frequency_hz=FREQ)
    return result.path_gain.reshape(-1).double()


@pytest.mark.parametrize("components,subdiv,welded", [
    pytest.param({"reflection"}, 1, True, id="reflection-s1"),
    pytest.param({"reflection"}, 3, True, id="reflection-s3"),
    pytest.param({"diffraction"}, 1, True, id="diffraction-s1"),
    pytest.param({"diffraction"}, 3, True, id="diffraction-s3"),
    pytest.param({"diffraction"}, 1, False, id="diffraction-s1-unwelded"),
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
                  endpoints=[sc.transmitter(city.extent_m), sc.receiver_grid(city.extent_m, 16)])
    paths = solve(scene, Config(components={"reflection"}, max_depth=max_depth, export_paths=True),
                  reference_frequency_hz=FREQ).paths
    depth, gain = paths.depth.cpu(), paths.path_gain.double().cpu()
    return {d: (int((depth == d).sum()), float(gain[depth == d].sum())) for d in range(1, max_depth + 1)}


@pytest.fixture(scope="module")
def scheme_depths() -> tuple[dict[int, tuple[int, float]], dict[int, tuple[int, float]]]:
    return _per_depth(3), _per_depth(4)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_scheme_equivalence_on_overlapping_depths(accuracy, scheme_depths, depth):
    """``max_depth=3`` and ``4`` must agree on depths 1-3: each depth's source is its own choice."""
    rows3, gain3 = scheme_depths[0][depth]
    rows4, gain4 = scheme_depths[1][depth]
    accuracy.record(f"depth {depth} rows", abs(rows4 - rows3), 0, rows_d3=rows3, rows_d4=rows4)
    accuracy.check(f"depth {depth} power", abs(gain4 - gain3) / gain3, SUM_TOLERANCE,
                   rows_d3=rows3, rows_d4=rows4)
    assert rows3 == rows4


def _reflection_rows(max_depth: int, discovery: str, n_side: int = 2, grid_n: int = 8):
    """Sorted ``(rx, depth, primitive sequence, length, gain)`` reflection rows of a small city."""
    from witwin.channel.deterministic import Config, solve

    city = sc.city(n_side)
    scene = Scene(structures=city.structures,
                  endpoints=[sc.transmitter(city.extent_m), sc.receiver_grid(city.extent_m, grid_n)])
    result = solve(scene, Config(components={"reflection"}, max_depth=max_depth, export_paths=True,
                                 coherent=False, discovery=discovery), reference_frequency_hz=FREQ)
    p = result.paths
    rows = sorted(
        (int(rx), int(d), tuple(seq[:int(d)]), float(length), float(gain))
        for rx, d, seq, length, gain in zip(p.rx_id.tolist(), p.depth.tolist(),
                                            p.primitive_sequence.tolist(), p.path_length_m.tolist(),
                                            p.path_gain.tolist()))
    return rows, result.metadata


@pytest.mark.parametrize("depth,n_side,grid_n", [(1, 4, 16), (2, 4, 16), (3, 4, 16), (4, 3, 12)])
def test_image_tree_matches_brute_force_enumeration(accuracy, monkeypatch, depth, n_side, grid_n):
    """The pruned image tree with receiver pruning finds exactly the brute-force path set."""
    from witwin.channel.interactions import reflection

    tree, meta = _reflection_rows(depth, "auto", n_side, grid_n)
    with monkeypatch.context() as patch:
        patch.setattr(reflection, "_RECEIVER_TEST_MIN_PAIRS", 1 << 62)
        patch.setattr(reflection, "_DIRECT_GROUP_SIZE", 1 << 62)
        brute, brute_meta = _reflection_rows(depth, "exhaustive", n_side, grid_n)
    assert meta["complete"] and brute_meta["complete"]
    assert set(meta["discovery"]["sources"].values()) == {"image_tree"}
    assert set(brute_meta["discovery"]["sources"].values()) == {"exhaustive"}
    deepest = sum(1 for row in brute if row[1] == depth)
    accuracy.record(f"depth {depth} rows", abs(len(tree) - len(brute)), 0, rows=len(brute),
                    deepest=deepest)
    assert deepest > 0
    assert tree == brute


@pytest.mark.parametrize("max_depth", [2, 3])
def test_ray_discovery_is_a_subset_and_converges(accuracy, monkeypatch, max_depth):
    """Ray discovery finds only rows of the complete source, and all of them with enough rays."""
    from witwin.channel.interactions import reflection

    complete, _ = _reflection_rows(max_depth, "auto", 4, 16)
    rays, meta = _reflection_rows(max_depth, "rays", 4, 16)
    assert meta["complete"] is False
    assert set(rays) <= set(complete)
    with monkeypatch.context() as patch:
        patch.setattr(reflection, "_MULTIBOUNCE_DISCOVERY_RAYS", 1 << 22)
        dense, _ = _reflection_rows(max_depth, "rays", 4, 16)
    accuracy.record(f"max_depth {max_depth} rows missed by 4M rays", len(complete) - len(dense), 0,
                    default_rays=len(rays), complete=len(complete))
    assert dense == complete


def test_rows_do_not_depend_on_the_candidate_source(accuracy):
    """``exhaustive`` and ``auto`` give bit-identical rows on the C-1 scene up to depth 3."""
    auto, _ = _reflection_rows(3, "auto", n_side=4, grid_n=16)
    exhaustive, _ = _reflection_rows(3, "exhaustive", n_side=4, grid_n=16)
    accuracy.record("auto vs exhaustive rows", abs(len(auto) - len(exhaustive)), 0, rows=len(auto))
    assert auto == exhaustive


# --------------------------------------------------------------------------- four solvers


GRID_ORIGIN, GRID_SHAPE, GRID_SPACING = (2.0, -3.0, 0.5), (4, 4), 2.0
TX_LOS, TX_REFLECTION = (0.0, 0.0, 0.5), (0.0, 0.0, 6.0)
WALL_X = 12.0


def _cell_centres() -> list[tuple[float, float, float]]:
    ox, oy, oz = GRID_ORIGIN
    # Grid results are [T, rows, cols]: row i along the grid x axis, column j along y.
    return [(ox + i * GRID_SPACING, oy + j * GRID_SPACING, oz)
            for i in range(GRID_SHAPE[0]) for j in range(GRID_SHAPE[1])]


def _wall() -> list:
    pec = PhysicalMaterial.perfect_conductor()
    v = [[WALL_X, -40.0, -20.0], [WALL_X, 40.0, -20.0], [WALL_X, -40.0, 40.0], [WALL_X, 40.0, 40.0]]
    return [surface(v, [[0, 2, 1], [1, 2, 3]], pec)]


def _solve_grid(
    solver: str, structures: list, tx: tuple[float, float, float], components: set[str],
) -> torch.Tensor:
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


def _solve_points(
    structures: list, tx: tuple[float, float, float], components: set[str],
) -> torch.Tensor:
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


_MULTILAYER_WALL = ((0.02, 3.0, 0.001), (0.15, 6.5, 0.05), (0.02, 3.0, 0.001))


def _multilayer_wall() -> list:
    from witwin.core import MaterialLayer

    material = PhysicalMaterial(layers=tuple(MaterialLayer(thickness_m=t, eps_r=e, sigma_e=s)
                                             for t, e, s in _MULTILAYER_WALL))
    v = [[WALL_X, -40.0, -20.0], [WALL_X, 40.0, -20.0], [WALL_X, -40.0, 40.0], [WALL_X, 40.0, 40.0]]
    return [surface(v, [[0, 2, 1], [1, 2, 3]], material)]


def _multilayer_image_power(tx, point) -> float:
    from tests.reference import em_oracle as EM

    z = (0.0, 0.0, 1.0)
    image = np.array([2 * WALL_X - tx[0], tx[1], tx[2]])
    cos_i = abs(point[0] - image[0]) / float(np.linalg.norm(np.asarray(point) - image))
    stack = EM.layer_stack_rt([(t, e, s, 1.0) for t, e, s in _MULTILAYER_WALL], cos_i, FREQ)
    value, _, _ = O.specular_reflection(tx, point, np.array([WALL_X, 0.0, 0.0]), (-1.0, 0.0, 0.0), FREQ, z, z,
                                        r_te=complex(stack.r_te), r_tm=-complex(stack.r_tm))
    return abs(value) ** 2


@pytest.mark.parametrize("solver", ("deterministic", "path", "mc"))
def test_multilayer_wall_reflection_across_solvers(accuracy, solver):
    """A three-layer wall reflects as its whole stack in every solver that supports it (A-1).

    Enumerated solvers are compared at the cell centres to float32 accuracy; Monte Carlo Basic with
    the cell-averaged power and the counting tolerance of the PEC wall test.
    """

    centres = _cell_centres()
    if solver == "mc":
        reference = np.array([_cell_average(lambda p: _multilayer_image_power(TX_REFLECTION, p), c)
                              for c in centres])
        tolerance = float(np.max([5.0 / math.sqrt(_mc_hits(TX_REFLECTION, c)) for c in centres]))
    else:
        reference = np.array([_multilayer_image_power(TX_REFLECTION, c) for c in centres])
        tolerance = 2.0 * coefficient_tolerance(40.0)
    value = (_solve_points(_multilayer_wall(), TX_REFLECTION, {"reflection"}) if solver == "path"
             else _solve_grid(solver, _multilayer_wall(), TX_REFLECTION, {"reflection"})).numpy()
    accuracy.check(solver, float(np.max(np.abs(value / reference - 1.0))), tolerance)


def test_bdpt_refuses_multilayer_reflection():
    from witwin.channel.montecarlo import bdpt

    scene = Scene(structures=_multilayer_wall(),
                  endpoints=[antenna("tx", TX_REFLECTION), grid(GRID_ORIGIN, GRID_SHAPE, GRID_SPACING)])
    with pytest.raises(NotImplementedError, match="multilayer"):
        bdpt.solve(scene, bdpt.Config(components={"reflection"}, max_depth=1, samples=64),
                   reference_frequency_hz=FREQ)


def test_pec_screen_winding_invariance(accuracy):
    """Flipping a PEC screen's triangle winding changes no physics, so the edge field must not move."""
    phi, phi_prime = math.radians(200.0), math.radians(60.0)
    front = channel_edge_coefficient(_half_plane(True), "soft", phi, phi_prime, HALF_PLANE_S_PRIME,
                                     HALF_PLANE_S)
    back = channel_edge_coefficient(_half_plane(False), "soft", phi, phi_prime, HALF_PLANE_S_PRIME,
                                    HALF_PLANE_S)
    accuracy.check("soft phi=200", abs(back - front) / abs(front), 1e-4)
