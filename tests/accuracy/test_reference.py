# Copyright Xingyu Chen.
# Compare Channel solves with float64 references: Friis, two-ray, slabs, half-plane, wedge.

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from witwin.core import MaterialLayer, PhysicalMaterial, Scene

from tests.accuracy import oracles as O
from tests.accuracy.scenes import (
    FREQ,
    antenna,
    coefficient_tolerance,
    deterministic_rows,
    ground,
    half_space,
    path_solver_coefficient,
    surface,
    wall_x,
)
from tests.reference import em_oracle as EM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Channel solves need CUDA")

K = O.wavenumber(FREQ)
LAMBDA = O.wavelength(FREQ)
VERTICAL, HORIZONTAL_Y = (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)


def _rel(value: complex, reference: complex) -> float:
    return abs(value - reference) / abs(reference)


# --------------------------------------------------------------------------- free space


FRIIS_CASES = [  # (rx position, polarization) with the Tx at (0, 0, 1.5)
    ((10.0, 0.0, 1.5), VERTICAL), ((100.0, 0.0, 1.5), VERTICAL), ((500.0, 0.0, 1.5), VERTICAL),
    ((10.0, 0.0, 11.5), VERTICAL), ((3.0, 4.0, -2.0), (0.0, 0.6, 0.8)),
    ((0.0, 50.0, 1.5), (1.0, 0.0, 0.0)),
]


@pytest.mark.parametrize("solver", ["deterministic", "path"])
@pytest.mark.parametrize("rx,pol", FRIIS_CASES, ids=[f"{c[0]}-{c[1]}" for c in FRIIS_CASES])
def test_free_space_friis(accuracy, solver, rx, pol):
    tx = (0.0, 0.0, 1.5)
    scene = Scene(structures=[], endpoints=[antenna("tx", tx, pol), antenna("rx", rx, pol)])
    reference = O.free_space(tx, rx, FREQ, pol, pol)
    if solver == "deterministic":
        rows = deterministic_rows(scene, {"los"}, max_depth=0)
        value = sum(r.coefficient for r in rows)
    else:
        value = path_solver_coefficient(scene, {"los"}, max_depth=0).get(0, 0j)
    d = math.dist(tx, rx)
    accuracy.check(f"{solver} rx={rx} pol={pol}", _rel(value, reference), coefficient_tolerance(d))


# --------------------------------------------------------------------------- two-ray ground


TWO_RAY_EPS, TWO_RAY_SIGMA = 15.0, 0.05


def test_two_ray_layer_is_a_half_space():
    """Precondition: the 30 m lossy layer's back-face echo is far below the float32 noise floor."""
    eps_c = O.relative_permittivity(TWO_RAY_EPS, TWO_RAY_SIGMA, FREQ)
    alpha = K * abs(np.sqrt(eps_c - 1.0).imag)  # weakest attenuation: grazing incidence
    assert math.exp(-2.0 * alpha * 30.0) < 1e-12


@pytest.mark.parametrize("solver", ["deterministic", "path"])
@pytest.mark.parametrize("pol", [VERTICAL, HORIZONTAL_Y, (0.0, 0.6, 0.8)], ids=["V", "H", "mixed"])
@pytest.mark.parametrize("distance", [5.0, 20.0, 100.0, 500.0])
def test_two_ray_ground_reflection(accuracy, solver, pol, distance):
    tx, rx = (0.0, 0.0, 10.0), (distance, 0.0, 1.5)
    scene = Scene(structures=[ground(half_space(TWO_RAY_EPS, TWO_RAY_SIGMA))],
                  endpoints=[antenna("tx", tx, pol), antenna("rx", rx, pol)])
    eps_c = O.relative_permittivity(TWO_RAY_EPS, TWO_RAY_SIGMA, FREQ)
    los = O.free_space(tx, rx, FREQ, pol, pol)
    reflected, _, length = O.specular_reflection(tx, rx, np.zeros(3), (0.0, 0.0, 1.0), FREQ, pol, pol,
                                                 eps_c=eps_c)
    if solver == "deterministic":
        by_depth: dict[int, complex] = {}
        for row in deterministic_rows(scene, {"los", "reflection"}):
            by_depth[row.depth] = by_depth.get(row.depth, 0j) + row.coefficient
    else:
        by_depth = path_solver_coefficient(scene, {"los", "reflection"})
    tol = coefficient_tolerance(length)
    accuracy.check(f"{solver} d={distance} pol={pol} reflected", _rel(by_depth.get(1, 0j), reflected), tol)
    # Coherent sum normalized by the incoherent amplitude sum so a deep interference null does
    # not turn a tiny absolute error into a large relative one.
    total = by_depth.get(0, 0j) + by_depth.get(1, 0j)
    accuracy.check(f"{solver} d={distance} pol={pol} total", abs(total - (los + reflected))
                   / (abs(los) + abs(reflected)), tol)


# --------------------------------------------------------------------------- slabs


SINGLE_SLAB = ((0.1, 5.24, 0.123, 1.0),)
MULTI_SLAB = ((0.02, 3.0, 0.001, 1.0), (0.15, 6.5, 0.05, 1.0), (0.02, 3.0, 0.001, 1.0))
# Reflection off a multilayer wall equals the first layer's reflection alone today.
MULTILAYER_REFLECTION_XFAIL = pytest.mark.xfail(
    strict=True, reason="A-1 (M0-3): multilayer reflection evaluates only the first layer")


def _material(layers: tuple[tuple[float, float, float, float], ...]) -> PhysicalMaterial:
    return PhysicalMaterial(layers=tuple(MaterialLayer(thickness_m=t, eps_r=e, sigma_e=s)
                                         for t, e, s, _ in layers))


@pytest.mark.parametrize("layers", [
    pytest.param(SINGLE_SLAB, id="single"),
    pytest.param(MULTI_SLAB, id="multilayer", marks=MULTILAYER_REFLECTION_XFAIL),
])
@pytest.mark.parametrize("pol", [VERTICAL, (0.0, 1.0, 0.0)], ids=["TE", "TM"])
@pytest.mark.parametrize("offset", [1.0, 4.0, 15.0])
def test_slab_reflection_matches_layer_stack_oracle(accuracy, layers, pol, offset):
    """Wall at x = 5 m; Tx and Rx in front of it, separated by ``2 * offset`` along y."""
    tx, rx = (0.0, -offset, 1.0), (0.0, offset, 1.0)
    scene = Scene(structures=[wall_x(5.0, _material(layers))],
                  endpoints=[antenna("tx", tx, pol), antenna("rx", rx, pol)])
    cos_i = 5.0 / math.hypot(5.0, offset)
    stack = EM.layer_stack_rt(list(layers), cos_i, FREQ)
    # em_oracle's TM amplitude is the tangential-E ratio, the negative of the p = s x k convention.
    reference, _, length = O.specular_reflection(tx, rx, np.array([5.0, 0.0, 0.0]), (-1.0, 0.0, 0.0),
                                                 FREQ, pol, pol, r_te=complex(stack.r_te),
                                                 r_tm=-complex(stack.r_tm))
    value = sum(r.coefficient for r in deterministic_rows(scene, {"reflection"}) if r.depth == 1)
    accuracy.check(f"{len(layers)} layers pol={pol} cos_i={cos_i:.3f}", _rel(value, reference),
                   coefficient_tolerance(length))


@pytest.mark.parametrize("layers", [SINGLE_SLAB, MULTI_SLAB], ids=["single", "multilayer"])
@pytest.mark.parametrize("pol", [VERTICAL, (0.0, 1.0, 0.0)], ids=["TE", "TM"])
@pytest.mark.parametrize("lateral", [0.0, 3.0, 8.0])
def test_slab_transmission_magnitude_matches_layer_stack_oracle(accuracy, layers, pol, lateral):
    """Straight ray through a wall at x = 5 m; compares ``|a| / |a_free|`` with ``|t|``.

    Only the magnitude is compared: the phase reference plane of a straight-ray slab model is
    a convention (entry face, centre plane, or exit face), not physics.
    """

    tx, rx = (0.0, 0.0, 1.0), (10.0, lateral, 1.0)
    scene = Scene(structures=[wall_x(5.0, _material(layers))],
                  endpoints=[antenna("tx", tx, pol), antenna("rx", rx, pol)])
    cos_i = 10.0 / math.hypot(10.0, lateral)
    stack = EM.layer_stack_rt(list(layers), cos_i, FREQ)
    t_ref = abs(complex(stack.t_te if pol == VERTICAL else stack.t_tm))
    free = abs(O.free_space(tx, rx, FREQ, pol, pol))
    rows = [r for r in deterministic_rows(scene, {"transmission"}, max_depth=2)]
    value = abs(sum(r.coefficient for r in rows)) / free
    accuracy.check(f"{len(layers)} layers pol={pol} cos_i={cos_i:.3f}", abs(value - t_ref) / t_ref, 1e-4)


# --------------------------------------------------------------------------- half-plane


HALF_PLANE_EDGE = 400.0
HALF_PLANE_S_PRIME, HALF_PLANE_S = 100.0, 10.0
HALF_PLANE_PHI_PRIME = math.radians(60.0)
# Receivers away from the reflection (120 deg) and shadow (240 deg) boundaries and the faces.
HALF_PLANE_ANGLES = (150.0, 200.0, 270.0, 320.0)


def _half_plane(normal_towards_source: bool = True) -> list:
    e = HALF_PLANE_EDGE
    faces = [[0, 2, 1], [0, 3, 2]] if normal_towards_source else [[0, 1, 2], [0, 2, 3]]
    return [surface([[0.0, 0.0, -e], [e, 0.0, -e], [e, 0.0, e], [0.0, 0.0, e]], faces,
                    PhysicalMaterial.perfect_conductor())]


def _edge_polarizations(
    polarization: str, phi: float, phi_prime: float,
) -> tuple[list[float], list[float]]:
    """Tx/Rx polarizations that read E_z (soft) or eta*H_z (hard) of the incident and diffracted waves."""
    if polarization == "soft":
        return [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]
    k_i = -np.array([math.cos(phi_prime), math.sin(phi_prime), 0.0])
    return list(np.cross([0.0, 0.0, 1.0], k_i)), [-math.sin(phi), math.cos(phi), 0.0]


def channel_edge_coefficient(
    structures: list, polarization: str, phi: float, phi_prime: float, s_prime: float, s: float,
) -> complex:
    """Diffraction coefficient of the edge on the z axis, extracted from a spherical-wave solve.

    Divides the diffracted coefficient by the incident field at the edge, the UTD spreading factor
    ``sqrt(s' / (s (s + s')))`` and ``exp(-j k s)``; rows of the screen's far boundary edges are
    excluded by their diffraction point.
    """

    tx = [s_prime * math.cos(phi_prime), s_prime * math.sin(phi_prime), 0.0]
    rx = [s * math.cos(phi), s * math.sin(phi), 0.0]
    p_tx, p_rx = _edge_polarizations(polarization, phi, phi_prime)
    scene = Scene(structures=structures, endpoints=[antenna("tx", tx, p_tx), antenna("rx", rx, p_rx)])
    rows = deterministic_rows(scene, {"diffraction"})
    value = sum(r.coefficient for r in rows if np.linalg.norm(r.point) < 1.0)
    incident = LAMBDA / (4.0 * math.pi * s_prime) * np.exp(-1j * K * s_prime)
    return complex(value / (incident * math.sqrt(s_prime / (s * (s + s_prime))) * np.exp(-1j * K * s)))


def sommerfeld_coefficient(
    polarization: str, phi: float, phi_prime: float, distance_l: float,
) -> complex:
    """Exact plane-wave half-plane diffracted field at ``rho = L``, expressed as a coefficient."""
    total = O.sommerfeld_half_plane(distance_l, phi, phi_prime, K, polarization)
    go = 0j
    sign = -1.0 if polarization == "soft" else 1.0
    for angle, weight in ((phi - phi_prime, 1.0), (phi + phi_prime, sign)):
        if math.cos(angle / 2.0) > 0.0:
            go += weight * np.exp(1j * K * distance_l * math.cos(angle))
    return complex((total - go) * math.sqrt(distance_l) * np.exp(1j * K * distance_l))


@pytest.mark.parametrize("polarization", ["soft", "hard"])
@pytest.mark.parametrize("phi_deg", HALF_PLANE_ANGLES)
def test_pec_half_plane_matches_sommerfeld(accuracy, polarization, phi_deg):
    """Tolerance: UTD's own asymptotic error against the exact solution (computed here in float64)
    plus 1e-3 for float32 and the 800 m finite edge (its end-point contributions are excluded; the
    stationary row keeps the aperture factor 1 - 2/(pi v), v ~ 600 Fresnel units, about 9e-4)."""

    phi = math.radians(phi_deg)
    s_prime, s = HALF_PLANE_S_PRIME, HALF_PLANE_S
    distance_l = s * s_prime / (s + s_prime)
    exact = sommerfeld_coefficient(polarization, phi, HALF_PLANE_PHI_PRIME, distance_l)
    d_soft, d_hard = O.utd_wedge_coefficients(phi, HALF_PLANE_PHI_PRIME, 2.0, K, distance_l)
    asymptotic = _rel(d_soft if polarization == "soft" else d_hard, exact)
    value = channel_edge_coefficient(_half_plane(), polarization, phi, HALF_PLANE_PHI_PRIME, s_prime, s)
    accuracy.check(f"{polarization} phi={phi_deg}", _rel(value, exact), asymptotic + 1e-3,
                   utd_vs_exact=asymptotic)


# --------------------------------------------------------------------------- 90 degree wedge


def _right_angle_wedge() -> list:
    """PEC wedge with faces on phi = 0 (x >= 0, y = 0) and phi = 270 deg (x = 0, y <= 0)."""
    e = HALF_PLANE_EDGE
    pec = PhysicalMaterial.perfect_conductor()
    face_o = surface([[0.0, 0.0, -e], [e, 0.0, -e], [e, 0.0, e], [0.0, 0.0, e]], [[0, 2, 1], [0, 3, 2]], pec)
    face_n = surface([[0.0, 0.0, -e], [0.0, -e, -e], [0.0, -e, e], [0.0, 0.0, e]], [[0, 1, 2], [0, 2, 3]], pec)
    return [face_o, face_n]


@pytest.mark.parametrize("polarization", ["soft", "hard"])
@pytest.mark.parametrize("phi_deg", [100.0, 180.0, 240.0])
def test_right_angle_wedge_matches_textbook_utd(accuracy, polarization, phi_deg):
    """n = 1.5 wedge, incidence phi' = 30 deg: reflection boundary 150 deg, shadow boundary 210 deg.

    Receivers sit 30 deg or more from both boundaries and from the faces (0 and 270 deg).
    The reference is the Kouyoumjian-Pathak coefficient evaluated in float64; the tolerance is the
    float32 one of ``coefficient_tolerance`` for the unfolded path.
    """

    phi, phi_prime = math.radians(phi_deg), math.radians(30.0)
    s_prime, s = HALF_PLANE_S_PRIME, HALF_PLANE_S
    distance_l = s * s_prime / (s + s_prime)
    d_soft, d_hard = O.utd_wedge_coefficients(phi, phi_prime, 1.5, K, distance_l)
    reference = d_soft if polarization == "soft" else d_hard
    value = channel_edge_coefficient(_right_angle_wedge(), polarization, phi, phi_prime, s_prime, s)
    accuracy.check(f"{polarization} phi={phi_deg}", _rel(value, reference),
                   coefficient_tolerance(s + s_prime))
