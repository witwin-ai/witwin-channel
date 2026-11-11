# Copyright Xingyu Chen.
# CPU self-checks of the accuracy oracles against each other and against limiting cases.

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.accuracy import oracles as O
from tests.reference import em_oracle as EM

FREQ = 3.5e9
K = O.wavenumber(FREQ)


@pytest.mark.parametrize("cos_i", [1.0, 0.7, 0.2, 0.02])
def test_half_space_fresnel_matches_em_oracle(cos_i):
    eps_r, sigma = 15.0, 0.05
    r_te, r_tm = O.fresnel_half_space(cos_i, O.relative_permittivity(eps_r, sigma, FREQ))
    em = EM.fresnel_interface(cos_i, EM.vacuum_medium(FREQ), EM.medium_params(eps_r, sigma, 1.0, FREQ))
    assert r_te == pytest.approx(complex(em.r_te), abs=1e-9)
    # em_oracle's TM amplitude is the tangential-E ratio: the negative of the p = s x k convention.
    assert r_tm == pytest.approx(-complex(em.r_tm), abs=1e-9)


@pytest.mark.parametrize("cos_i", [0.9, 0.3, 0.05])
def test_thick_lossy_layer_is_a_half_space(cos_i):
    eps_r, sigma = 15.0, 0.05
    stack = EM.layer_stack_rt([(30.0, eps_r, sigma, 1.0)], cos_i, FREQ)
    r_te, r_tm = O.fresnel_half_space(cos_i, O.relative_permittivity(eps_r, sigma, FREQ))
    assert complex(stack.r_te) == pytest.approx(r_te, abs=1e-9)
    assert -complex(stack.r_tm) == pytest.approx(r_tm, abs=1e-9)


def test_pec_reflection_is_the_electric_image():
    tx, rx, pol = np.array([0.0, 0.0, 10.0]), np.array([20.0, 3.0, 1.5]), np.array([0.3, 0.4, 0.866])
    value, _, length = O.specular_reflection(tx, rx, np.zeros(3), (0, 0, 1), FREQ, pol, pol,
                                             r_te=-1.0 + 0j, r_tm=1.0 + 0j)
    image_pol = np.array([-pol[0], -pol[1], pol[2]])  # an electric dipole's image in a conductor
    image = np.array([0.0, 0.0, -10.0])
    expected = O.free_space(image, rx, FREQ, image_pol, pol)
    assert length == pytest.approx(np.linalg.norm(rx - image))
    assert value == pytest.approx(expected, rel=1e-12)


def _exact_diffracted(polarization, phi, phi_prime, rho):
    total = O.sommerfeld_half_plane(rho, phi, phi_prime, K, polarization)
    sign = -1.0 if polarization == "soft" else 1.0
    go = sum(w * np.exp(1j * K * rho * math.cos(a)) for a, w in ((phi - phi_prime, 1.0),
             (phi + phi_prime, sign)) if math.cos(a / 2.0) > 0.0)
    return total - go


@pytest.mark.parametrize("polarization", ["soft", "hard"])
@pytest.mark.parametrize("phi_deg", [150.0, 200.0, 270.0, 320.0])
def test_utd_half_plane_approaches_sommerfeld(polarization, phi_deg):
    """Away from shadow boundaries UTD differs from the exact solution by O((k rho)^-1)."""
    phi, phi_prime, rho = math.radians(phi_deg), math.radians(60.0), 9.0
    exact = _exact_diffracted(polarization, phi, phi_prime, rho)
    d_soft, d_hard = O.utd_wedge_coefficients(phi, phi_prime, 2.0, K, rho)
    utd = (d_soft if polarization == "soft" else d_hard) * np.exp(-1j * K * rho) / math.sqrt(rho)
    assert abs(utd - exact) / abs(exact) < 10.0 / (K * rho)


def test_sommerfeld_total_field_is_continuous_across_the_shadow_boundary():
    phi_prime, rho = math.radians(60.0), 20.0
    shadow = math.pi + phi_prime
    left = O.sommerfeld_half_plane(rho, shadow - 1e-6, phi_prime, K, "soft")
    right = O.sommerfeld_half_plane(rho, shadow + 1e-6, phi_prime, K, "soft")
    assert abs(left - right) < 1e-4
    # On the boundary the total field is half the incident wave (plus a small reflected term).
    assert abs(O.sommerfeld_half_plane(rho, shadow, phi_prime, K, "soft")) == pytest.approx(0.5, abs=0.02)


def test_utd_wedge_reduces_to_half_plane_for_n_2():
    phi, phi_prime, distance = math.radians(200.0), math.radians(60.0), 9.0
    d_soft, d_hard = O.utd_wedge_coefficients(phi, phi_prime, 2.0, K, distance)
    # Half-plane closed form (Keller limit): D = -exp(-j pi/4)/(2 sqrt(2 pi k)) [sec((phi-phi')/2) -/+ sec((phi+phi')/2)].
    pre = -np.exp(-1j * math.pi / 4) / (2.0 * math.sqrt(2.0 * math.pi * K))
    keller_soft = pre * (1 / math.cos((phi - phi_prime) / 2) - 1 / math.cos((phi + phi_prime) / 2))
    keller_hard = pre * (1 / math.cos((phi - phi_prime) / 2) + 1 / math.cos((phi + phi_prime) / 2))
    # Far from the boundaries the transition function is ~1 so UTD equals the Keller coefficient.
    assert d_soft == pytest.approx(keller_soft, rel=2e-2)
    assert d_hard == pytest.approx(keller_hard, rel=2e-2)
