# Copyright Xingyu Chen.
# Gate: incoherent deterministic radiomaps agree with MC Basic within its confidence interval.

"""Gate: incoherent deterministic radiomaps agree with MC Basic within its confidence interval."""

from __future__ import annotations

import math

import pytest
import torch
from witwin.core import PhysicalMaterial, Scene

from tests.accuracy.scenes import FREQ, antenna, grid, surface

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Channel solves need CUDA")

# Independent Monte Carlo solves per case; the standard error of each cell mean comes from
# their spread. Fixed seeds keep every run of this gate on the same draws.
SEEDS = 8
SAMPLES = 1 << 20
# A cell passes when |MC mean - deterministic cell average| <= Z_LIMIT standard errors. With
# 7 degrees of freedom, 5 standard errors is a two-sided per-cell tail of about 0.16 %.
Z_LIMIT = 5.0
# Deterministic sub-samples per cell side; their mean is the deterministic cell average.
SUBSAMPLES = 8
SHAPE, SPACING = (4, 4), 2.0


def _wall(x: float, material: object) -> list:
    v = [[x, -40.0, -20.0], [x, 40.0, -20.0], [x, -40.0, 40.0], [x, 40.0, 40.0]]
    return [surface(v, [[0, 2, 1], [1, 2, 3]], material)]


def _ground(material: object) -> list:
    v = [[-200.0, -200.0, 0.0], [200.0, -200.0, 0.0], [-200.0, 200.0, 0.0], [200.0, 200.0, 0.0]]
    return [surface(v, [[0, 1, 2], [1, 3, 2]], material)]


def _mc_mean_and_sem(structures, tx, component, origin, shape, spacing):
    from witwin.channel import radiomap

    scene = Scene(structures=structures, endpoints=[antenna("tx", tx), grid(origin, shape, spacing)])
    maps = torch.stack([
        radiomap.solve(
            scene,
            radiomap.Stochastic(components={component}, max_depth=1, samples=SAMPLES, seed=seed),
            reference_frequency_hz=FREQ,
        ).component_maps[component][0].double().cpu()
        for seed in range(SEEDS)
    ])
    return maps.mean(0), maps.std(0) / math.sqrt(SEEDS)


def _deterministic(structures, tx, component, origin, shape, spacing, *, average: bool):
    """Incoherent deterministic power at each cell centre, or averaged over each cell."""

    from witwin.channel import radiomap

    n = SUBSAMPLES if average else 1
    sub = spacing / n
    start = (origin[0] - (spacing - sub) / 2, origin[1] - (spacing - sub) / 2, origin[2])
    scene = Scene(structures=structures,
                  endpoints=[antenna("tx", tx), grid(start, (shape[0] * n, shape[1] * n), sub)])
    result = radiomap.solve(
        scene, radiomap.Coherent(components={component}, max_depth=1, combine="power"),
        reference_frequency_hz=FREQ,
    )
    values = result.path_gain[0].double().cpu()
    first, second = values.shape
    return values.reshape(first // n, n, second // n, n).mean(dim=(1, 3))


# Diffraction is not part of this gate: MC Basic samples Keller cones and Deterministic
# enumerates UTD rows, two different estimators that are not expected to agree. The
# deterministic diffraction target is the full-wave reference.
# name: (structures, tx, component, grid origin, grid spacing)
CONSISTENT = {
    "pec_wall_reflection_shallow": (
        lambda: _wall(12.0, PhysicalMaterial.perfect_conductor()), (0.0, 0.0, 1.5), "reflection",
        (2.0, -3.0, 0.5), SPACING),
    "dielectric_wall_reflection_shallow": (
        lambda: _wall(12.0, PhysicalMaterial(eps_r=5.0, sigma_e=0.01)), (0.0, 0.0, 1.5),
        "reflection", (2.0, -3.0, 0.5), SPACING),
}
# Measured on the 0.6.0 gate (8 seeds x 2^20 samples): MC mean / deterministic cell average.
GAPS = {
    "pec_wall_reflection_steep": (
        lambda: _wall(12.0, PhysicalMaterial.perfect_conductor()), (0.0, 0.0, 6.0), "reflection",
        (2.0, -3.0, 0.5), SPACING,
        "MC Basic reflection is scalar; deterministic projects the vector field onto the z "
        "dipoles. Measured MC/deterministic 1.060-1.119, up to 133 standard errors."),
    "dielectric_ground_reflection": (
        lambda: _ground(PhysicalMaterial(eps_r=5.0, sigma_e=0.01)), (0.0, 0.0, 10.0),
        "reflection", (20.0, -3.0, 1.5), SPACING,
        "Scalar MC reflection near the Brewster angle versus vector deterministic Fresnel "
        "reflection. Measured MC/deterministic 1.186-1.333, up to 184 standard errors."),
}


def _z_scores(name, case):
    structures, tx, component, origin, spacing = case[:5]
    mean, sem = _mc_mean_and_sem(structures(), tx, component, origin, SHAPE, spacing)
    reference = _deterministic(structures(), tx, component, origin, SHAPE, spacing, average=True)
    assert bool((reference > 0).all()) and bool((mean > 0).all()), name
    return ((mean - reference).abs() / sem.clamp_min(1e-300)), mean / reference


@pytest.mark.parametrize("name", sorted(CONSISTENT))
def test_mc_basic_matches_incoherent_deterministic_cell_average(accuracy, name):
    z, ratio = _z_scores(name, CONSISTENT[name])
    accuracy.check(name, float(z.max()), Z_LIMIT, ratio_min=float(ratio.min()),
                   ratio_max=float(ratio.max()))


@pytest.mark.parametrize("name", [
    pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=case[5]))
    for name, case in sorted(GAPS.items())
])
def test_mc_basic_known_gaps_to_incoherent_deterministic(accuracy, name):
    z, ratio = _z_scores(name, GAPS[name])
    accuracy.check(name, float(z.max()), Z_LIMIT, ratio_min=float(ratio.min()),
                   ratio_max=float(ratio.max()))


@pytest.mark.parametrize("component,structures,tx,origin", [
    pytest.param("los", list, (0.0, 0.0, 1.0), (4.0, -3.0, 1.0), id="los"),
    pytest.param("transmission", lambda: _wall(5.0, PhysicalMaterial(eps_r=4.0, sigma_e=0.01)),
                 (0.0, 0.0, 1.0), (8.0, -3.0, 1.0), id="transmission"),
    pytest.param("los", list, (0.0, 0.0, 6.0), (-3.0, -3.0, 1.0), id="los_steep", marks=pytest.mark.xfail(
        strict=True, reason="MC Basic LoS projects the field onto the transmitter polarization only; "
        "deterministic also projects onto the receiver's z dipole (sin^4 against sin^2 of the "
        "zenith angle). Up to 12 dB brighter near the transmitter's foot.")),
])
def test_mc_basic_point_components_equal_deterministic_cell_centres(
    accuracy, component, structures, tx, origin,
):
    """MC Basic evaluates LoS and straight transmission analytically at each cell centre."""

    from witwin.channel import radiomap

    scene = Scene(structures=structures(), endpoints=[antenna("tx", tx), grid(origin, SHAPE, SPACING)])
    mc = radiomap.solve(scene, radiomap.Stochastic(components={component}, max_depth=1, samples=1 << 12),
                     reference_frequency_hz=FREQ).component_maps[component][0].double().cpu()
    reference = _deterministic(structures(), tx, component, origin, SHAPE, SPACING, average=False)
    accuracy.check(component, float(((mc - reference).abs() / reference).max()), 1e-5)


def _cube(center: tuple[float, float, float], half: float, material: object) -> list:
    cx, cy, cz = center
    v = [[cx + x * half, cy + y * half, cz + z * half]
         for z in (-1.0, 1.0) for y in (-1.0, 1.0) for x in (-1.0, 1.0)]
    f = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
         [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]
    return [surface(v, f, material)]


@pytest.mark.xfail(strict=True, reason=(
    "MC Basic deposits a Keller-cone ray on a receiver grid where it crosses the grid plane. A "
    "vertical edge sends its cone away from a plane at the transmitter's height, so the deep "
    "shadow receives no diffraction power, where the UTD rows of deterministic do."))
def test_mc_basic_diffraction_reaches_the_deep_shadow_of_a_vertical_edge():
    """Support, not agreement: wherever deterministic diffraction is non-zero, MC Basic is too."""

    from witwin.channel import radiomap

    structures = _cube((4.0, 0.0, 1.0), 1.0, PhysicalMaterial(eps_r=4.0, sigma_e=0.01))
    scene = Scene(structures=structures,
                  endpoints=[antenna("tx", (0.0, 0.0, 1.5)), grid((7.0, -1.5, 1.5), (4, 4), 1.0)])
    deterministic = radiomap.solve(
        scene, radiomap.Coherent(components={"diffraction"}, max_depth=1, combine="power"),
        reference_frequency_hz=FREQ,
    ).path_gain[0].double().cpu()
    mc = radiomap.solve(
        scene, radiomap.Stochastic(components={"diffraction"}, max_depth=1, samples=1 << 22),
        reference_frequency_hz=FREQ,
    ).path_gain[0].double().cpu()
    # The centre columns need a second diffraction at max_depth=1; the outer ones do not.
    lit = deterministic > 0
    assert int(lit.sum()) >= 8
    missing = lit & ~(mc > 0)
    assert not bool(missing.any()), f"{int(missing.sum())} of {int(lit.sum())} shadow cells hold no power"
