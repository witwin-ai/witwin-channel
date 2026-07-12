"""Golden tests for the Kirchhoff/phase-screen oracle.

Covers Beckmann-series vs direct quadrature agreement, smooth limits,
phase-screen patch-integral identities (constant height, sinusoidal grating
with Bessel harmonic amplitudes), quadrature convergence, and hemisphere
energy sanity.
"""

import numpy as np
import pytest
from scipy.special import jv

from witwin.channel_native.physics import (
    C0,
    coherent_attenuation,
    hemisphere_integral,
    kirchhoff_diffuse_lobe_quadrature,
    kirchhoff_diffuse_lobe_series,
    phase_screen_patch_integral,
)

F0 = 3.0e9
K0 = 2.0 * np.pi * F0 / C0


@pytest.mark.parametrize("g_target", [0.1, 1.0, 5.0, 20.0])
@pytest.mark.parametrize("aniso", [1.0, 3.0])
def test_series_vs_quadrature(g_target, aniso):
    """Series and direct 2D Fourier quadrature agree to 1e-6 relative."""
    lx = 0.06
    ly = lx / aniso
    q_n = 250.0
    sigma_h = np.sqrt(g_target) / q_n
    q_points = [
        (0.0, 0.0),
        (1.5 / lx, 0.0),
        (0.0, 1.5 / ly),
        (0.8 / lx, 0.6 / ly),
    ]
    for qx, qy in q_points:
        series = kirchhoff_diffuse_lobe_series(
            qx, qy, q_n, sigma_h, lx, ly, n_terms=192
        )
        quad_lo = kirchhoff_diffuse_lobe_quadrature(
            qx, qy, q_n, sigma_h, lx, ly, n_points=200
        )
        quad_hi = kirchhoff_diffuse_lobe_quadrature(
            qx, qy, q_n, sigma_h, lx, ly, n_points=280
        )
        scale = max(abs(quad_hi), abs(series))
        assert scale > 0.0
        assert abs(quad_lo - quad_hi) < 1e-7 * scale  # quadrature converged
        assert abs(series - quad_hi) < 1e-6 * scale


def test_smooth_limit_sigma_to_zero():
    """sigma_h -> 0: diffuse lobe -> 0 everywhere; coherent attenuation -> 1."""
    lx, ly, q_n = 0.05, 0.05, 200.0
    qx = np.array([0.0, 10.0, 40.0])
    qy = np.array([0.0, -5.0, 20.0])
    assert np.all(kirchhoff_diffuse_lobe_series(qx, qy, q_n, 0.0, lx, ly) == 0.0)
    sigmas = 1e-3 * 0.5 ** np.arange(11)
    values = np.array(
        [kirchhoff_diffuse_lobe_series(10.0, 5.0, q_n, s, lx, ly) for s in sigmas]
    )
    assert np.all(values[1:] < values[:-1])  # monotone decrease toward 0
    assert values[-1] < 1e-5 * values[0]
    assert coherent_attenuation(0.0, K0) == 1.0
    k_z1 = K0 * 0.8
    assert np.isclose(
        coherent_attenuation(0.002, k_z1), np.exp(-2.0 * (k_z1 * 0.002) ** 2)
    )


def _rectangle_corners(size_x, size_y):
    """Axis-aligned patch in z = 0: p(u, v) = (u*size_x, v*size_y, 0)."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [size_x, 0.0, 0.0],
            [size_x, size_y, 0.0],
            [0.0, size_y, 0.0],
        ]
    )


def _segment_integral(q, length):
    """Analytic 1D flat integral of exp(-j*q*x) over [0, length]."""
    if q == 0.0:
        return complex(length)
    return (np.exp(-1j * q * length) - 1.0) / (-1j * q)


def test_phase_screen_constant_height():
    """Constant h0 gives exp(-j*q_n*h0) times the analytic flat integral."""
    size = 0.5
    corners = _rectangle_corners(size, size)
    theta_i = np.deg2rad(25.0)
    k_i = K0 * np.array([np.sin(theta_i), 0.0, -np.cos(theta_i)])
    k_s = K0 * np.array([0.3, 0.2, np.sqrt(1.0 - 0.09 - 0.04)])
    q = k_s - k_i
    q_n = q[2]  # patch normal is +z
    flat = phase_screen_patch_integral(
        lambda u, v: np.zeros_like(u), corners, k_i, k_s, F0, n_quad=96
    )
    analytic_flat = _segment_integral(q[0], size) * _segment_integral(q[1], size)
    assert np.isclose(flat, analytic_flat, rtol=1e-9)
    h0 = 0.003
    shifted = phase_screen_patch_integral(
        lambda u, v: np.full_like(u, h0), corners, k_i, k_s, F0, n_quad=96
    )
    assert np.isclose(shifted, np.exp(-1j * q_n * h0) * flat, rtol=1e-12)


@pytest.mark.parametrize("order", [0, 1])
def test_phase_screen_sinusoidal_grating_orders(order):
    """Sinusoidal height gives (-1)^m * J_m(q_n*A) at the grating angles.

    With h(x) = A*sin(kappa*x), Jacobi-Anger expands the phase screen into
    harmonics exp(+j*p*kappa*x) with amplitudes J_p(-q_n*A); over an integer
    number of periods only the order with q_x = m*kappa survives, so the
    integral equals area * (-1)^m * J_m(q_n*A).
    """
    period = 0.5
    kappa = 2.0 * np.pi / period
    n_periods = 8
    size_x = n_periods * period
    size_y = 0.5
    corners = _rectangle_corners(size_x, size_y)
    sin_i = -0.1
    k_i = K0 * np.array([sin_i, 0.0, -np.sqrt(1.0 - sin_i * sin_i)])
    # Grating condition: q_x = order*kappa, q_y = 0, |k_s| = k0.
    ks_x = k_i[0] + order * kappa
    sin_s = ks_x / K0
    k_s = K0 * np.array([sin_s, 0.0, np.sqrt(1.0 - sin_s * sin_s)])
    q_n = (k_s - k_i)[2]
    amplitude = 0.0096
    height_fn = lambda u, v: amplitude * np.sin(kappa * u * size_x)
    integral = phase_screen_patch_integral(
        height_fn, corners, k_i, k_s, F0, n_quad=(512, 4)
    )
    expected = size_x * size_y * (-1.0) ** order * jv(order, q_n * amplitude)
    assert abs(expected) > 1e-3  # the asserted harmonic is not trivially zero
    assert np.isclose(integral, expected, rtol=1e-9, atol=1e-12)


def test_phase_screen_quadrature_convergence():
    """Refining the quadrature converges (spectrally) to a fixed value."""
    size = 3.0  # several oscillation periods, so coarse rules are visibly off
    corners = _rectangle_corners(size, size)
    theta_i = np.deg2rad(25.0)
    k_i = K0 * np.array([np.sin(theta_i), 0.0, -np.cos(theta_i)])
    k_s = K0 * np.array([0.3, 0.17, np.sqrt(1.0 - 0.09 - 0.17 ** 2)])
    height_fn = lambda u, v: 0.002 * np.sin(3.0 * u + 1.0) * np.cos(2.0 * v + 0.5)

    def integral(n):
        return phase_screen_patch_integral(height_fn, corners, k_i, k_s, F0, n_quad=n)

    i8, i16, i32, i128 = integral(8), integral(16), integral(32), integral(128)
    err8 = abs(i8 - i128)
    err16 = abs(i16 - i128)
    err32 = abs(i32 - i128)
    assert err16 < err8
    assert err32 < err16
    assert err32 < 1e-8 * abs(i128)


def test_hemisphere_integral_helper():
    assert np.isclose(hemisphere_integral(lambda mu, phi: 1.0 + 0.0 * mu), 2.0 * np.pi,
                      atol=1e-12)
    assert np.isclose(hemisphere_integral(lambda mu, phi: mu), np.pi, atol=1e-12)


def test_diffuse_lobe_hemisphere_energy_sanity():
    """Hemispheric diffuse energy is finite and decreases as g -> 0."""
    lx = ly = 0.05
    theta_i = np.deg2rad(30.0)
    k_i = K0 * np.array([np.sin(theta_i), 0.0, -np.cos(theta_i)])

    def energy(sigma_h):
        def lobe_flux(mu, phi):
            sin_o = np.sqrt(1.0 - mu * mu)
            qx = K0 * sin_o * np.cos(phi) - k_i[0]
            qy = K0 * sin_o * np.sin(phi) - k_i[1]
            qn = K0 * mu - k_i[2]
            lobe = kirchhoff_diffuse_lobe_series(qx, qy, qn, sigma_h, lx, ly)
            return lobe * mu

        return (K0 ** 2 / (4.0 * np.pi ** 2)) * hemisphere_integral(lobe_flux)

    energies = np.array([energy(s) for s in [0.0005, 0.001, 0.002, 0.004]])
    assert np.all(np.isfinite(energies))
    assert np.all(energies >= 0.0)
    assert np.all(energies[1:] > energies[:-1])  # smaller sigma_h => less energy
    # In the small-g regime energy scales ~ sigma_h^2: an 8x smaller sigma_h
    # must lose well over an order of magnitude of diffuse energy.
    assert energies[0] < 0.02 * energies[-1]
