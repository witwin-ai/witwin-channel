"""Golden tests for the complex128 EM oracle (plan section 11.1 items 1-8).

All expected values are computed at runtime from independent closed-form
expressions (impedance formulas, analytic TIR phases, exact periodicity),
never from the oracle itself.
"""

import numpy as np
import pytest

from witwin.channel.physics import (
    C0,
    EPS0,
    MU0,
    complex_sqrt_passive,
    fresnel_interface,
    layer_stack_rt,
    medium_params,
    refraction_direction,
    vacuum_medium,
)

F0 = 3.5e9
K0 = 2.0 * np.pi * F0 / C0
LAMBDA0 = C0 / F0
COS_ANGLES = np.array([1.0, 0.8, 0.5, 0.1])


def test_complex_sqrt_passive_branch():
    z = np.array([4.0, -4.0, 1.0 - 2.0j, -1.0 - 2.0j, 0.0])
    w = complex_sqrt_passive(z)
    assert np.allclose(w * w, z, atol=1e-14)
    assert np.all(w.real >= 0.0)
    assert np.all(w.imag <= 0.0)
    # Negative real axis picks the decaying (evanescent) root -j*sqrt(|z|).
    assert np.isclose(complex(complex_sqrt_passive(-9.0)), -3.0j, atol=1e-15)


def test_identical_media_no_interface():
    """Item 1: identical media give r = 0 and single-interface t = 1."""
    for eps_r, sigma_e, mu_r in [(1.0, 0.0, 1.0), (4.0, 0.5, 2.0)]:
        medium = medium_params(eps_r, sigma_e, mu_r, F0)
        coeff = fresnel_interface(COS_ANGLES, medium, medium)
        assert np.allclose(coeff.r_te, 0.0, atol=1e-15)
        assert np.allclose(coeff.r_tm, 0.0, atol=1e-15)
        assert np.allclose(coeff.t_te, 1.0, atol=1e-15)
        assert np.allclose(coeff.t_tm, 1.0, atol=1e-15)


def test_lossless_normal_incidence_impedance_formula():
    """Item 2: normal incidence matches eta formulas; R+T = 1 to 1e-12."""
    vac = vacuum_medium(F0)
    for eps_r, mu_r in [(2.25, 1.0), (4.0, 1.0), (10.0, 2.0)]:
        medium = medium_params(eps_r, 0.0, mu_r, F0)
        eta1 = np.sqrt(MU0 / EPS0)
        eta2 = eta1 * np.sqrt(mu_r / eps_r)
        coeff = fresnel_interface(1.0, vac, medium)
        r_expected = (eta2 - eta1) / (eta2 + eta1)
        t_expected = 2.0 * eta2 / (eta2 + eta1)
        for r, t in [(coeff.r_te, coeff.t_te), (coeff.r_tm, coeff.t_tm)]:
            assert np.isclose(complex(r), r_expected, atol=1e-14)
            assert np.isclose(complex(t), t_expected, atol=1e-14)
        assert np.isclose(coeff.r_te, coeff.r_tm, atol=1e-15)
        assert abs(coeff.R_te + coeff.T_te - 1.0) < 1e-12
        assert abs(coeff.R_tm + coeff.T_tm - 1.0) < 1e-12


@pytest.mark.parametrize("n1,n2", [(1.0, 1.5), (1.5, 1.0), (1.0, 3.0)])
def test_brewster_angle_tm_null(n1, n2):
    """Item 3: R_TM < 1e-20 at theta_B = atan(n2/n1), lossless nonmagnetic."""
    medium1 = medium_params(n1 * n1, 0.0, 1.0, F0)
    medium2 = medium_params(n2 * n2, 0.0, 1.0, F0)
    cos_brewster = n1 / np.hypot(n1, n2)
    coeff = fresnel_interface(cos_brewster, medium1, medium2)
    assert float(coeff.R_tm) < 1e-20
    assert float(coeff.R_te) > 1e-3  # TE has no Brewster null


def test_total_internal_reflection():
    """Item 4: T = 0, |r| = 1, nonzero phase with the correct signs."""
    eps1 = 4.0  # n1 = 2, critical angle 30 deg
    medium1 = medium_params(eps1, 0.0, 1.0, F0)
    medium2 = vacuum_medium(F0)
    theta_i = np.deg2rad(60.0)
    cos_i, sin_i = np.cos(theta_i), np.sin(theta_i)
    n1 = np.sqrt(eps1)
    coeff = fresnel_interface(cos_i, medium1, medium2)
    assert abs(float(coeff.T_te)) < 1e-15
    assert abs(float(coeff.T_tm)) < 1e-15
    assert abs(np.abs(complex(coeff.r_te)) - 1.0) < 1e-12
    assert abs(np.abs(complex(coeff.r_tm)) - 1.0) < 1e-12
    # Analytic phases for exp(+j*w*t), passive branch k_z2 = -j*k0*s:
    #   TE: r = (n1*cos_i + j*s)/(n1*cos_i - j*s) -> arg = +2*atan(s/(n1*cos_i))
    #   TM: r = (Y1 - j*c)/(Y1 + j*c) with c/Y1 = n1*cos_i/(eps1*s) -> arg < 0
    s = np.sqrt(n1 * n1 * sin_i * sin_i - 1.0)
    phase_te = 2.0 * np.arctan2(s, n1 * cos_i)
    phase_tm = -2.0 * np.arctan2(n1 * cos_i, eps1 * s)
    assert np.angle(complex(coeff.r_te)) > 0.0
    assert np.angle(complex(coeff.r_tm)) < 0.0
    assert np.isclose(np.angle(complex(coeff.r_te)), phase_te, atol=1e-12)
    assert np.isclose(np.angle(complex(coeff.r_tm)), phase_tm, atol=1e-12)


def test_pec_limit():
    """Item 5: sigma_e -> 1e9 gives R -> 1, T -> 0, r -> -1."""
    vac = vacuum_medium(F0)
    previous = 0.0
    for sigma_e in [1e6, 1e9]:
        conductor = medium_params(1.0, sigma_e, 1.0, F0)
        for cos_i in [1.0, np.cos(np.deg2rad(45.0))]:
            coeff = fresnel_interface(cos_i, vac, conductor)
            for big_r in [coeff.R_te, coeff.R_tm]:
                assert float(big_r) <= 1.0 + 1e-12
        assert float(coeff.R_te) > previous  # R increases toward the PEC limit
        previous = float(coeff.R_te)
    assert float(coeff.R_te) > 1.0 - 1e-4
    assert float(coeff.R_tm) > 1.0 - 1e-4
    assert float(coeff.T_te) < 1e-4
    assert float(coeff.T_tm) < 1e-4
    assert abs(complex(coeff.r_te) + 1.0) < 1e-2
    assert abs(complex(coeff.r_tm) + 1.0) < 1e-2


def test_zero_thickness_layer_equals_no_layer():
    """Item 6a: a zero-thickness layer is an exact identity."""
    cos_i = 0.7
    bare = layer_stack_rt([], cos_i, F0)
    zero = layer_stack_rt([(0.0, 5.0, 0.3, 2.0)], cos_i, F0)
    assert complex(zero.r_te) == complex(bare.r_te)
    assert complex(zero.t_te) == complex(bare.t_te)
    slab = [(0.04, 4.0, 0.1, 1.0)]
    with_zero = layer_stack_rt([(0.0, 9.0, 1.0, 1.0)] + slab, cos_i, F0)
    alone = layer_stack_rt(slab, cos_i, F0)
    assert np.isclose(complex(with_zero.r_tm), complex(alone.r_tm), atol=1e-15)
    assert np.isclose(complex(with_zero.t_tm), complex(alone.t_tm), atol=1e-15)


def test_fabry_perot_thickness_period():
    """Item 6b: lossless slab R/T are periodic in d with period pi/Re(k_z)."""
    eps_r = 4.0
    cos_i = 0.8
    sin2_i = 1.0 - cos_i * cos_i
    k_z_layer = K0 * np.sqrt(eps_r - sin2_i)  # real (lossless)
    period = np.pi / k_z_layer
    thicknesses = np.linspace(0.01, 2.0, 41) * LAMBDA0
    big_r = np.array(
        [float(layer_stack_rt([(d, eps_r, 0.0, 1.0)], cos_i, F0).R_te) for d in thicknesses]
    )
    shifted = np.array(
        [
            float(layer_stack_rt([(d + period, eps_r, 0.0, 1.0)], cos_i, F0).R_te)
            for d in thicknesses
        ]
    )
    assert np.allclose(shifted, big_r, rtol=1e-10, atol=1e-12)
    assert big_r.max() - big_r.min() > 0.05  # the sweep actually oscillates


@pytest.mark.parametrize("cos_i", [1.0, 0.6])
def test_vacuum_layer_is_pure_propagation(cos_i):
    """Item 6c: vacuum layer gives r = 0 and t = exp(-j*k_z*d) exactly."""
    for d in [0.01, 0.37 * LAMBDA0, 1.234]:
        coeff = layer_stack_rt([(d, 1.0, 0.0, 1.0)], cos_i, F0)
        k_z = K0 * cos_i
        assert abs(complex(coeff.r_te)) < 1e-15
        assert abs(complex(coeff.r_tm)) < 1e-15
        expected = np.exp(-1j * k_z * d)  # free-space k_z propagation phase
        assert np.isclose(complex(coeff.t_te), expected, rtol=1e-12, atol=1e-14)
        assert np.isclose(complex(coeff.t_tm), expected, rtol=1e-12, atol=1e-14)


def test_high_loss_thick_layer_stability():
    """Item 7: T -> 0 smoothly, no NaN/Inf up to d = 10 m, sigma_e = 1e4."""
    cos_i = 0.9
    for sigma_e in [1.0, 100.0, 1e4]:
        transmittances = []
        for d in [0.01, 0.1, 1.0, 10.0]:
            coeff = layer_stack_rt([(d, 4.0, sigma_e, 1.0)], cos_i, F0)
            values = [
                complex(coeff.r_te), complex(coeff.r_tm),
                complex(coeff.t_te), complex(coeff.t_tm),
                float(coeff.R_te), float(coeff.R_tm),
                float(coeff.T_te), float(coeff.T_tm),
                float(coeff.A_te), float(coeff.A_tm),
            ]
            assert np.all(np.isfinite(values))
            assert 0.0 <= float(coeff.T_te) <= 1.0
            assert 0.0 <= float(coeff.R_te) <= 1.0
            assert 0.0 <= float(coeff.A_te) <= 1.0
            transmittances.append(float(coeff.T_te))
        assert all(a >= b for a, b in zip(transmittances, transmittances[1:]))
    assert transmittances[-1] == 0.0  # opaque stack underflows cleanly to zero


def _random_layers(rng, lossless):
    count = rng.integers(1, 5)
    layers = []
    for _ in range(count):
        sigma_e = 0.0 if lossless else float(rng.uniform(0.0, 5.0))
        layers.append(
            (
                float(rng.uniform(0.001, 0.3)),
                float(rng.uniform(1.0, 10.0)),
                sigma_e,
                float(rng.uniform(1.0, 3.0)),
            )
        )
    return layers


def test_energy_conservation_random_stacks():
    """Item 8: lossless stacks R+T = 1 to 1e-10; lossy stacks 0 <= A <= 1."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        cos_i = float(rng.uniform(0.05, 1.0))
        lossless = layer_stack_rt(_random_layers(rng, lossless=True), cos_i, F0)
        assert abs(float(lossless.R_te) + float(lossless.T_te) - 1.0) < 1e-10
        assert abs(float(lossless.R_tm) + float(lossless.T_tm) - 1.0) < 1e-10
        lossy = layer_stack_rt(_random_layers(rng, lossless=False), cos_i, F0)
        for big_r, big_t, big_a in [
            (lossy.R_te, lossy.T_te, lossy.A_te),
            (lossy.R_tm, lossy.T_tm, lossy.A_tm),
        ]:
            assert -1e-10 <= float(big_a) <= 1.0 + 1e-10
            assert float(big_r) >= 0.0
            assert float(big_t) >= 0.0


def test_two_half_layers_equal_one_full_layer():
    cos_i = 0.75
    d = 0.06
    full = layer_stack_rt([(d, 4.0, 0.2, 1.5)], cos_i, F0)
    halves = layer_stack_rt(
        [(d / 2, 4.0, 0.2, 1.5), (d / 2, 4.0, 0.2, 1.5)], cos_i, F0
    )
    assert np.isclose(complex(halves.r_te), complex(full.r_te), atol=1e-14)
    assert np.isclose(complex(halves.r_tm), complex(full.r_tm), atol=1e-14)
    assert np.isclose(complex(halves.t_te), complex(full.t_te), atol=1e-14)
    assert np.isclose(complex(halves.t_tm), complex(full.t_tm), atol=1e-14)


ASYMMETRIC_LAYERS = [
    (0.02, 2.0, 0.1, 1.0),
    (0.05, 6.0, 0.5, 1.5),
    (0.015, 3.5, 0.0, 1.0),
]


def test_transmission_amplitude_reciprocity_same_bounding_media():
    """Same complex t from both sides of an asymmetric stack in vacuum."""
    cos_i = 0.85
    forward = layer_stack_rt(ASYMMETRIC_LAYERS, cos_i, F0)
    backward = layer_stack_rt(ASYMMETRIC_LAYERS[::-1], cos_i, F0)
    assert np.isclose(complex(backward.t_te), complex(forward.t_te), rtol=1e-12)
    assert np.isclose(complex(backward.t_tm), complex(forward.t_tm), rtol=1e-12)
    # r generally differs for an asymmetric lossy stack; t does not.
    assert abs(complex(backward.r_te) - complex(forward.r_te)) > 1e-6


def test_transmittance_reciprocity_different_bounding_media():
    """T_forward == T_backward with distinct lossless bounding media."""
    n_back = 1.5
    backing = medium_params(n_back * n_back, 0.0, 1.0, F0)
    outside = vacuum_medium(F0)
    theta_i = np.deg2rad(35.0)
    cos_fwd = np.cos(theta_i)
    # Match the conserved tangential wavenumber for the reverse direction.
    sin_bwd = np.sin(theta_i) / n_back
    cos_bwd = np.sqrt(1.0 - sin_bwd * sin_bwd)
    forward = layer_stack_rt(
        ASYMMETRIC_LAYERS, cos_fwd, F0, outside=outside, backing=backing
    )
    backward = layer_stack_rt(
        ASYMMETRIC_LAYERS[::-1], cos_bwd, F0, outside=backing, backing=outside
    )
    assert np.isclose(float(backward.T_te), float(forward.T_te), rtol=1e-10)
    assert np.isclose(float(backward.T_tm), float(forward.T_tm), rtol=1e-10)


def test_refraction_direction_snell_and_tir():
    theta_i = np.deg2rad(40.0)
    d_i = np.array([np.sin(theta_i), 0.0, -np.cos(theta_i)])
    n = np.array([0.0, 0.0, 1.0])  # points into the incident medium
    d_t = refraction_direction(d_i, n, 1.0 / 1.5)
    sin_t = np.sin(theta_i) / 1.5
    expected = np.array([sin_t, 0.0, -np.sqrt(1.0 - sin_t * sin_t)])
    assert np.allclose(d_t, expected, atol=1e-14)
    assert abs(np.linalg.norm(d_t) - 1.0) < 1e-14
    # Dense-to-rare beyond the critical angle: TIR -> None.
    theta_tir = np.deg2rad(60.0)
    d_i = np.array([np.sin(theta_tir), 0.0, -np.cos(theta_tir)])
    assert refraction_direction(d_i, n, 2.0) is None
