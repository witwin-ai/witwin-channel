# Copyright Xingyu Chen.
# Float64 NumPy reference solutions for free space, ground reflection, half-plane and wedge fields.

"""Float64 NumPy reference solutions for free space, ground reflection, half-plane and wedge fields."""

from __future__ import annotations

import numpy as np
from scipy import special

C0 = 299792458.0
EPS0 = 8.8541878128e-12


def wavelength(frequency_hz: float) -> float:
    return C0 / frequency_hz


def wavenumber(frequency_hz: float) -> float:
    return 2.0 * np.pi * frequency_hz / C0


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def transverse(polarization: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Component of a polarization vector perpendicular to the propagation direction."""
    p = np.asarray(polarization, dtype=np.complex128)
    d = unit(direction)
    return p - np.dot(p, d) * d


def free_space(
    tx: np.ndarray, rx: np.ndarray, frequency_hz: float, p_tx: np.ndarray, p_rx: np.ndarray,
) -> complex:
    """Friis coefficient ``lambda/(4 pi d) exp(-j k d) (p_tx,perp . p_rx)`` for unit excitation."""
    d_vec = np.asarray(rx, np.float64) - np.asarray(tx, np.float64)
    d = float(np.linalg.norm(d_vec))
    amplitude = wavelength(frequency_hz) / (4.0 * np.pi * d) * np.exp(-1j * wavenumber(frequency_hz) * d)
    return complex(amplitude * np.dot(transverse(p_tx, d_vec), np.asarray(p_rx, np.complex128)))


def relative_permittivity(eps_r: float, sigma_e: float, frequency_hz: float) -> complex:
    """``eps_r - j sigma/(omega eps0)`` for the ``exp(+j omega t)`` time convention."""
    return complex(eps_r, -sigma_e / (2.0 * np.pi * frequency_hz * EPS0))


def fresnel_half_space(cos_i: float, eps_c: complex) -> tuple[complex, complex]:
    """Half-space amplitude coefficients ``(r_te, r_tm)``; ``r_tm`` tends to +1 for a perfect conductor.

    ``r_te`` is the ratio of reflected to incident E perpendicular to the plane of incidence; ``r_tm``
    is the ratio of the in-plane E amplitudes in the basis ``p = s x k`` on both sides, up to the
    sign fixed in ``specular_reflection`` by the perfect-conductor image.
    """
    sin2 = 1.0 - cos_i * cos_i
    root = np.sqrt(eps_c - sin2 + 0j)
    if root.imag > 0:  # passive branch: decaying transmitted wave
        root = -root
    r_te = (cos_i - root) / (cos_i + root)
    r_tm = (eps_c * cos_i - root) / (eps_c * cos_i + root)
    return complex(r_te), complex(r_tm)


def specular_reflection(
    tx: np.ndarray, rx: np.ndarray, plane_point: np.ndarray, normal: np.ndarray,
    frequency_hz: float, p_tx: np.ndarray, p_rx: np.ndarray, r_te: complex | None = None,
    r_tm: complex | None = None, eps_c: complex | None = None,
) -> tuple[complex, np.ndarray, float]:
    """Image-method single reflection off an infinite plane.

    Returns ``(coefficient, reflection point, unfolded path length)``. Without explicit ``r_te`` /
    ``r_tm`` the half-space Fresnel coefficients of ``eps_c`` are used. The in-plane sign is fixed
    physically: a perfect conductor must reproduce the electric image ``-E + 2 (E.n) n``.
    """
    tx = np.asarray(tx, np.float64)
    rx = np.asarray(rx, np.float64)
    n = unit(normal)
    image = tx - 2.0 * np.dot(tx - plane_point, n) * n
    length = float(np.linalg.norm(rx - image))
    # Reflection point: intersection of the image->rx segment with the plane.
    direction = rx - image
    s_param = np.dot(plane_point - image, n) / np.dot(direction, n)
    point = image + s_param * direction
    k_i = unit(point - tx)
    k_r = unit(rx - point)
    cos_i = float(abs(np.dot(k_i, n)))
    if r_te is None or r_tm is None:
        assert eps_c is not None
        r_te, r_tm = fresnel_half_space(cos_i, eps_c)
    s_hat = np.cross(k_i, n)
    if np.linalg.norm(s_hat) < 1e-12:  # normal incidence: any perpendicular axis is a valid s
        s_hat = np.cross(k_i, [1.0, 0.0, 0.0] if abs(k_i[0]) < 0.9 else [0.0, 1.0, 0.0])
    s_hat = unit(s_hat)
    p_i = np.cross(s_hat, k_i)
    p_r = np.cross(s_hat, k_r)
    pec_image_sign = float(np.dot(-p_i + 2.0 * np.dot(p_i, n) * n, p_r))
    field = transverse(p_tx, k_i)
    reflected = r_te * np.dot(field, s_hat) * s_hat + pec_image_sign * r_tm * np.dot(field, p_i) * p_r
    amplitude = wavelength(frequency_hz) / (4.0 * np.pi * length) * np.exp(
        -1j * wavenumber(frequency_hz) * length)
    return complex(amplitude * np.dot(reflected, np.asarray(p_rx, np.complex128))), point, length


# --------------------------------------------------------------------------- diffraction


def fresnel_integral_complex(x: np.ndarray) -> np.ndarray:
    """``F(x) = int_x^inf exp(-j t^2) dt`` evaluated through the Fresnel sine/cosine integrals."""
    x = np.asarray(x, dtype=np.float64)
    s, c = special.fresnel(x * np.sqrt(2.0 / np.pi))
    scale = np.sqrt(np.pi / 2.0)
    return scale * ((0.5 - c) - 1j * (0.5 - s))


def sommerfeld_half_plane(
    rho: float, phi: float, phi_i: float, k: float, polarization: str,
) -> complex:
    """Exact total field of a unit plane wave on a PEC half-plane (Sommerfeld, ``exp(+j omega t)``).

    The screen occupies ``phi = 0`` (and ``2 pi``); the incident wave ``exp(j k rho cos(phi - phi_i))``
    arrives from direction ``phi_i``. ``polarization`` ``"soft"`` (E parallel to the edge, Dirichlet)
    or ``"hard"`` (H parallel to the edge, Neumann). Returns ``u_total = u(phi - phi_i) -/+ u(phi + phi_i)``
    with ``u(a) = exp(j k rho cos a) * sqrt(j/pi) * F(-sqrt(2 k rho) cos(a/2))``.
    """

    def u(angle: float) -> complex:
        arg = -np.sqrt(2.0 * k * rho) * np.cos(angle / 2.0)
        return complex(np.exp(1j * k * rho * np.cos(angle)) * np.sqrt(1j / np.pi)
                       * fresnel_integral_complex(arg))

    sign = -1.0 if polarization == "soft" else 1.0
    return u(phi - phi_i) + sign * u(phi + phi_i)


def utd_transition(x: np.ndarray) -> np.ndarray:
    """Kouyoumjian-Pathak transition function ``F(x) = 2 j sqrt(x) exp(j x) int_sqrt(x)^inf exp(-j t^2) dt``."""
    x = np.asarray(x, dtype=np.float64)
    return 2j * np.sqrt(x) * np.exp(1j * x) * fresnel_integral_complex(np.sqrt(x))


def _a_pm(beta: float, n: float, sign: int) -> float:
    big_n = np.round((beta + sign * np.pi) / (2.0 * np.pi * n))
    return float(2.0 * np.cos((2.0 * np.pi * n * big_n - beta) / 2.0) ** 2)


def utd_wedge_coefficients(
    phi: float, phi_p: float, n: float, k: float, distance_l: float, beta0: float = np.pi / 2,
) -> tuple[complex, complex]:
    """Kouyoumjian-Pathak UTD coefficients ``(D_soft, D_hard)`` for a PEC wedge of exterior angle ``n pi``.

    Angles are measured from the illuminated (``o``) face; ``distance_l`` is the UTD distance
    parameter (``s' s / (s' + s)`` for spherical incidence); ``beta0`` the edge-ray angle.
    """
    prefactor = -np.exp(-1j * np.pi / 4.0) / (2.0 * n * np.sqrt(2.0 * np.pi * k) * np.sin(beta0))
    terms = []
    for beta, sign_cot, sign_a in (
        (phi - phi_p, +1, +1), (phi - phi_p, -1, -1), (phi + phi_p, +1, +1), (phi + phi_p, -1, -1),
    ):
        cot = 1.0 / np.tan((np.pi + sign_cot * beta) / (2.0 * n))
        transition = utd_transition(k * distance_l * _a_pm(beta, n, sign_a))
        terms.append(complex(cot * transition))
    d1, d2, d3, d4 = terms
    return complex(prefactor * (d1 + d2 - (d3 + d4))), complex(prefactor * (d1 + d2 + (d3 + d4)))
