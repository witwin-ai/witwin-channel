# Copyright Xingyu Chen.
# CPU complex128 electromagnetic reference oracle (numpy-only, no torch).

"""CPU complex128 electromagnetic reference oracle (numpy-only, no torch).

Binding conventions (docs/dev/plans/05-implementation-contract.md, section 2):

- Time factor ``exp(+j*w*t)``, propagation ``exp(-j*k*r)``.
- Complex relative permittivity ``eps = eps_r' - j*sigma_e/(w*eps0)``.
- Passive sqrt branch: ``Re >= 0, Im <= 0`` so ``exp(-j*k*z)`` decays.
- Admittances ``Y_TE = k_z/(w*mu)``, ``Y_TM = w*eps/k_z`` (absolute eps/mu).
- Interface amplitudes ``r = (Y1-Y2)/(Y1+Y2)``, ``t = 2*Y1/(Y1+Y2)`` defined
  on the shared tangential electric field; TM uses the same formula (no
  hand-flipped signs).
- Powers ``R = |r|^2``, ``T = Re(Y2)/Re(Y1)*|t|^2``, ``A = 1 - R - T``.

Everything here is a slow, explicit float64/complex128 reference used as the
ground truth for torch/CUDA production code. No clamping, no silent degradation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

C0 = 299792458.0  # vacuum speed of light [m/s]
MU0 = 4.0e-7 * np.pi  # vacuum permeability [H/m] (classic value, self-consistent)
EPS0 = 1.0 / (MU0 * C0 * C0)  # vacuum permittivity [F/m], exact w.r.t. C0/MU0
ETA0 = MU0 * C0  # vacuum impedance [ohm]

__all__ = [
    "C0",
    "EPS0",
    "ETA0",
    "MU0",
    "Medium",
    "RTCoefficients",
    "coherent_attenuation",
    "complex_sqrt_passive",
    "fresnel_interface",
    "hemisphere_integral",
    "kirchhoff_diffuse_lobe_quadrature",
    "kirchhoff_diffuse_lobe_series",
    "layer_stack_rt",
    "medium_params",
    "phase_screen_patch_integral",
    "refraction_direction",
    "vacuum_medium",
]


def complex_sqrt_passive(z):
    """Complex sqrt with the passive-medium branch ``Re >= 0, Im <= 0``.

    Takes the principal sqrt (numpy: ``Re >= 0``) and negates results with
    ``Im > 0``. For arguments reachable from passive media
    (``arg(z) in (-pi, 0]``) the principal sqrt already satisfies both branch
    conditions; the negation only fires on the negative real axis
    (evanescent case, e.g. total internal reflection), where it selects the
    decaying root ``-j*sqrt(|z|)``. Vectorized; always returns complex128.
    """
    w = np.sqrt(np.asarray(z, dtype=np.complex128))
    return np.where(w.imag > 0.0, -w, w)


@dataclass(frozen=True)
class Medium:
    """Homogeneous isotropic medium at a single frequency.

    ``eps``/``mu`` are the ABSOLUTE complex permittivity/permeability
    (eps0/mu0 included); ``k`` is the complex wavenumber on the passive
    branch (``Re >= 0, Im <= 0``).
    """

    frequency_hz: float
    eps: complex
    mu: complex
    k: complex

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.frequency_hz


def medium_params(eps_r, sigma_e, mu_r, frequency_hz) -> Medium:
    """Build a :class:`Medium` from real material parameters.

    ``eps = eps0*(eps_r - j*sigma_e/(w*eps0))`` (conductivity folded into the
    imaginary part, ``exp(+j*w*t)`` convention), ``mu = mu0*mu_r`` (complex
    ``mu_r`` accepted), ``k = k0*sqrt(eps_rel*mu_rel)`` on the passive branch.
    """
    omega = 2.0 * np.pi * float(frequency_hz)
    eps_rel = complex(eps_r) - 1j * float(sigma_e) / (omega * EPS0)
    mu_rel = complex(mu_r)
    k0 = omega / C0
    k = k0 * complex(complex_sqrt_passive(eps_rel * mu_rel))
    return Medium(
        frequency_hz=float(frequency_hz),
        eps=eps_rel * EPS0,
        mu=mu_rel * MU0,
        k=k,
    )


def vacuum_medium(frequency_hz) -> Medium:
    """Vacuum :class:`Medium` at ``frequency_hz``."""
    return medium_params(1.0, 0.0, 1.0, frequency_hz)


@dataclass(frozen=True)
class RTCoefficients:
    """Polarized reflection/transmission amplitudes and power coefficients.

    Amplitudes are tangential-E ratios (contract section 2); powers include
    the ``Re(Y2)/Re(Y1)`` flux factor in T. Fields are complex128/float64
    scalars or arrays (broadcast over the incidence angle input).
    """

    r_te: np.ndarray
    r_tm: np.ndarray
    t_te: np.ndarray
    t_tm: np.ndarray
    R_te: np.ndarray
    R_tm: np.ndarray
    T_te: np.ndarray
    T_tm: np.ndarray
    A_te: np.ndarray
    A_tm: np.ndarray


def _admittances(medium: Medium, k_z):
    """Return ``(Y_TE, Y_TM)`` = ``(k_z/(w*mu), w*eps/k_z)`` for one medium."""
    return k_z / (medium.omega * medium.mu), medium.omega * medium.eps / k_z


def _interface_rt(y1, y2):
    """Amplitudes at a single interface: ``r=(Y1-Y2)/(Y1+Y2), t=2Y1/(Y1+Y2)``."""
    denom = y1 + y2
    return (y1 - y2) / denom, 2.0 * y1 / denom


def _power_coefficients(r, t, y1, y2):
    """``R=|r|^2, T=Re(Y2)/Re(Y1)*|t|^2, A=1-R-T`` (never clamped)."""
    big_r = np.abs(r) ** 2
    big_t = (np.real(y2) / np.real(y1)) * np.abs(t) ** 2
    return big_r, big_t, 1.0 - big_r - big_t


def fresnel_interface(cos_theta_i, medium1, medium2) -> RTCoefficients:
    """Fresnel coefficients for a single planar interface, medium1 -> medium2.

    ``cos_theta_i`` is the REAL cosine of the incidence angle measured in
    medium1 (scalar or array, in ``(0, 1]``). The tangential wavenumber
    ``k_par = k1*sin(theta_i)`` is conserved; ``k_z,m`` uses the passive
    sqrt branch. Both media must share the same frequency.
    """
    if medium1.frequency_hz != medium2.frequency_hz:
        raise ValueError("media must be evaluated at the same frequency")
    cos_i = np.asarray(cos_theta_i, dtype=np.float64)
    sin2_i = 1.0 - cos_i * cos_i
    k_par2 = medium1.k * medium1.k * sin2_i
    k_z1 = complex_sqrt_passive(medium1.k * medium1.k - k_par2)
    k_z2 = complex_sqrt_passive(medium2.k * medium2.k - k_par2)
    y1_te, y1_tm = _admittances(medium1, k_z1)
    y2_te, y2_tm = _admittances(medium2, k_z2)
    r_te, t_te = _interface_rt(y1_te, y2_te)
    r_tm, t_tm = _interface_rt(y1_tm, y2_tm)
    big_r_te, big_t_te, big_a_te = _power_coefficients(r_te, t_te, y1_te, y2_te)
    big_r_tm, big_t_tm, big_a_tm = _power_coefficients(r_tm, t_tm, y1_tm, y2_tm)
    return RTCoefficients(
        r_te=r_te, r_tm=r_tm, t_te=t_te, t_tm=t_tm,
        R_te=big_r_te, R_tm=big_r_tm, T_te=big_t_te, T_tm=big_t_tm,
        A_te=big_a_te, A_tm=big_a_tm,
    )


def _stack_rt_one_pol(y_out, y_layers, deltas, y_back):
    """Transfer-matrix stack solve for one polarization (plan section 5.1).

    Each layer matrix ``[[cos d, j sin d / Y], [j Y sin d, cos d]]`` is
    accumulated in incidence order. To stay finite for thick lossy layers
    (``Im(delta) << 0``) every layer matrix is scaled by ``exp(-a)`` with
    ``a = -Im(delta) >= 0``; the accumulated ``log_scale = sum(a)`` is
    reapplied only to ``t`` (as ``exp(-log_scale)``, which underflows cleanly
    to 0 for opaque stacks). ``r`` is scale-invariant.
    """
    m11 = np.ones_like(y_out)
    m12 = np.zeros_like(y_out)
    m21 = np.zeros_like(y_out)
    m22 = np.ones_like(y_out)
    log_scale = np.zeros(np.shape(y_out), dtype=np.float64)
    for y, delta in zip(y_layers, deltas):
        a = -np.imag(delta)  # >= 0 on the passive branch
        e_plus = np.exp(1j * np.real(delta))  # exp(+j*delta) * exp(-a)
        e_minus = np.exp(-2.0 * a - 1j * np.real(delta))  # exp(-j*delta) * exp(-a)
        cos_s = 0.5 * (e_plus + e_minus)
        sin_s = (e_plus - e_minus) / 2j
        l11 = cos_s
        l12 = 1j * sin_s / y
        l21 = 1j * y * sin_s
        l22 = cos_s
        m11, m12, m21, m22 = (
            m11 * l11 + m12 * l21,
            m11 * l12 + m12 * l22,
            m21 * l11 + m22 * l21,
            m21 * l12 + m22 * l22,
        )
        log_scale = log_scale + a
    b = m11 + m12 * y_back
    c = m21 + m22 * y_back
    denom = y_out * b + c
    r = (y_out * b - c) / denom
    t = (2.0 * y_out / denom) * np.exp(-log_scale)
    return r, t


def layer_stack_rt(
    layers: Sequence[tuple],
    cos_theta_i,
    frequency_hz,
    outside: Medium | None = None,
    backing: Medium | None = None,
) -> RTCoefficients:
    """Reflection/transmission of a planar layer stack, both polarizations.

    ``layers`` is a sequence of ``(thickness_m, eps_r, sigma_e, mu_r)`` in
    incidence order (first tuple is hit first). ``outside``/``backing``
    default to vacuum. ``cos_theta_i`` is the real incidence cosine in the
    outside medium. Zero layers reduce to the bare outside->backing Fresnel
    interface; a zero-thickness layer is an exact identity.
    """
    if outside is None:
        outside = vacuum_medium(frequency_hz)
    if backing is None:
        backing = vacuum_medium(frequency_hz)
    for medium in (outside, backing):
        if medium.frequency_hz != float(frequency_hz):
            raise ValueError("outside/backing media frequency mismatch")
    cos_i = np.asarray(cos_theta_i, dtype=np.float64)
    sin2_i = 1.0 - cos_i * cos_i
    k_par2 = outside.k * outside.k * sin2_i

    def kz(medium: Medium):
        return complex_sqrt_passive(medium.k * medium.k - k_par2)

    kz_out, kz_back = kz(outside), kz(backing)
    y_out = _admittances(outside, kz_out)
    y_back = _admittances(backing, kz_back)
    y_te_layers, y_tm_layers, deltas = [], [], []
    for thickness_m, eps_r, sigma_e, mu_r in layers:
        medium = medium_params(eps_r, sigma_e, mu_r, frequency_hz)
        kz_l = kz(medium)
        y_te, y_tm = _admittances(medium, kz_l)
        y_te_layers.append(y_te)
        y_tm_layers.append(y_tm)
        deltas.append(kz_l * float(thickness_m))
    ones = np.ones_like(cos_i, dtype=np.complex128)
    r_te, t_te = _stack_rt_one_pol(y_out[0] * ones, y_te_layers, deltas, y_back[0])
    r_tm, t_tm = _stack_rt_one_pol(y_out[1] * ones, y_tm_layers, deltas, y_back[1])
    big_r_te, big_t_te, big_a_te = _power_coefficients(r_te, t_te, y_out[0], y_back[0])
    big_r_tm, big_t_tm, big_a_tm = _power_coefficients(r_tm, t_tm, y_out[1], y_back[1])
    return RTCoefficients(
        r_te=r_te, r_tm=r_tm, t_te=t_te, t_tm=t_tm,
        R_te=big_r_te, R_tm=big_r_tm, T_te=big_t_te, T_tm=big_t_tm,
        A_te=big_a_te, A_tm=big_a_tm,
    )


def refraction_direction(d_i, n, n1_over_n2):
    """Vector Snell refraction (plan section 4.2); ``None`` on TIR.

    ``d_i`` is the unit incident propagation direction, ``n`` the unit
    normal pointing INTO the incident medium (so ``n . d_i < 0``),
    ``n1_over_n2`` the real phase-index ratio ``Re(k1)/Re(k2)``. Returns the
    unit transmitted direction, or ``None`` when the discriminant is
    negative (total internal reflection).
    """
    d = np.asarray(d_i, dtype=np.float64)
    d = d / np.linalg.norm(d)
    normal = np.asarray(n, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    eta = float(n1_over_n2)
    cos_i = -float(np.dot(normal, d))
    if cos_i <= 0.0:
        raise ValueError("normal must point toward the incident medium (n . d_i < 0)")
    disc = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if disc < 0.0:
        return None
    d_t = eta * d + (eta * cos_i - np.sqrt(disc)) * normal
    return d_t / np.linalg.norm(d_t)


def coherent_attenuation(sigma_h, k_z1):
    """Coherent specular field attenuation ``exp(-2*(k_z1*sigma_h)^2)``.

    ``k_z1 = k0*cos(theta_i)`` is the (real) normal wavenumber in the
    incidence medium, ``sigma_h`` the RMS height [m] of a Gaussian surface.
    ``sigma_h = 0`` gives exactly 1.
    """
    return np.exp(-2.0 * (np.asarray(k_z1) * sigma_h) ** 2)


def kirchhoff_diffuse_lobe_series(
    q_par_x, q_par_y, q_n, sigma_h, lx, ly, n_terms: int = 64
):
    """Beckmann series for the Gaussian-correlation Kirchhoff diffuse lobe.

    Evaluates (contract section 6)
    ``I(q) = pi*lx*ly*exp(-g) * sum_{m>=1} g^m/(m!*m)
    * exp(-(qx^2*lx^2 + qy^2*ly^2)/(4m))`` with ``g = q_n^2*sigma_h^2``,
    i.e. the 2D Fourier transform of ``exp(-g)*(exp(q_n^2*C(rho)) - 1)``
    for ``C(x,y) = sigma_h^2*exp(-(x/lx)^2 - (y/ly)^2)``.

    Each term is computed in log space, so large ``g`` (up to ~50 with
    enough ``n_terms``; the term peak sits near ``m ~ g``) never overflows.
    ``sigma_h = 0`` returns exactly 0. Broadcasts over the q inputs.
    """
    qx, qy, qn = np.broadcast_arrays(
        np.asarray(q_par_x, dtype=np.float64),
        np.asarray(q_par_y, dtype=np.float64),
        np.asarray(q_n, dtype=np.float64),
    )
    g = (qn * float(sigma_h)) ** 2
    rho2 = (qx * lx) ** 2 + (qy * ly) ** 2
    m_flat = np.arange(1, n_terms + 1, dtype=np.float64)
    shape = (n_terms,) + (1,) * g.ndim
    m = m_flat.reshape(shape)
    log_fact = np.cumsum(np.log(m_flat)).reshape(shape)  # log(m!)
    with np.errstate(divide="ignore"):
        log_g = np.log(g)
    # m*log(g) -> -inf when g == 0, so exp(term) == 0 there (no NaN: m > 0).
    log_term = m * log_g - log_fact - np.log(m) - rho2 / (4.0 * m) - g
    series = np.exp(log_term).sum(axis=0)
    result = np.pi * lx * ly * series
    return result if result.ndim else float(result)


def kirchhoff_diffuse_lobe_quadrature(
    q_par_x,
    q_par_y,
    q_n,
    sigma_h,
    lx,
    ly,
    n_points: int = 220,
    half_width: float = 6.0,
):
    """Kirchhoff diffuse lobe via direct 2D Fourier quadrature (cross-check).

    Computes ``I(q) = exp(-g) * Int exp(-j*q_par.rho)*(exp(q_n^2*C(rho))-1)
    d^2 rho`` by Gauss-Legendre quadrature over one quadrant (the integrand
    is even in x and y separately, so ``exp(-j q.rho)`` reduces to
    ``4*cos(qx*x)*cos(qy*y)``). Integration extends to ``half_width`` times
    the correlation length per axis. Must agree with
    :func:`kirchhoff_diffuse_lobe_series` (same quantity, same conventions).
    """
    qx, qy, qn = np.broadcast_arrays(
        np.asarray(q_par_x, dtype=np.float64),
        np.asarray(q_par_y, dtype=np.float64),
        np.asarray(q_n, dtype=np.float64),
    )
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    x = 0.5 * half_width * lx * (nodes + 1.0)
    wx = 0.5 * half_width * lx * weights
    y = 0.5 * half_width * ly * (nodes + 1.0)
    wy = 0.5 * half_width * ly * weights
    corr = np.exp(-((x[:, None] / lx) ** 2) - (y[None, :] / ly) ** 2)
    w2d = wx[:, None] * wy[None, :]
    out = np.empty(qx.shape, dtype=np.float64)
    for idx in np.ndindex(qx.shape):
        g = (qn[idx] * float(sigma_h)) ** 2
        kernel = np.expm1(g * corr)
        integrand = kernel * np.cos(qx[idx] * x)[:, None] * np.cos(qy[idx] * y)[None, :]
        out[idx] = 4.0 * np.exp(-g) * float((w2d * integrand).sum())
    return out if out.ndim else float(out)


def phase_screen_patch_integral(
    height_fn: Callable,
    patch_corners,
    k_i_vec,
    k_s_vec,
    frequency_hz,
    n_quad=64,
):
    """Direct complex Kirchhoff phase integral over a planar patch.

    Evaluates ``Int_A exp(-j*(k_s - k_i).x) * exp(-j*q_n*h(u,v)) dA`` with
    scalar Kirchhoff kernel 1 and ``q_n = (k_s - k_i).n_hat``; heights enter
    ONLY through the phase (positions stay on the mean plane, per plan
    section 6.7).

    ``patch_corners`` is a (4, 3) array ``[p00, p10, p11, p01]`` describing a
    planar parallelogram with ``p(u, v) = p00 + u*(p10-p00) + v*(p01-p00)``,
    ``u, v in [0, 1]``; the patch normal is ``normalize(e_u x e_v)``.
    ``height_fn(u, v)`` must accept broadcast arrays and return metric height
    [m] along the patch normal. ``k_i_vec``/``k_s_vec`` are wave vectors
    [rad/m] whose magnitudes must match ``2*pi*frequency_hz/c0`` (checked).
    ``n_quad`` is a Gauss-Legendre point count per axis (int or (nu, nv)).
    Returns a complex scalar.
    """
    corners = np.asarray(patch_corners, dtype=np.float64)
    if corners.shape != (4, 3):
        raise ValueError("patch_corners must have shape (4, 3): [p00, p10, p11, p01]")
    p00, p10, p11, p01 = corners
    scale = max(np.linalg.norm(corners - p00, axis=1).max(), 1.0)
    if not np.allclose(p00 + p11, p10 + p01, atol=1e-9 * scale):
        raise ValueError("patch_corners must form a planar parallelogram")
    e_u = p10 - p00
    e_v = p01 - p00
    normal = np.cross(e_u, e_v)
    area_jacobian = np.linalg.norm(normal)
    n_hat = normal / area_jacobian
    k_i = np.asarray(k_i_vec, dtype=np.float64)
    k_s = np.asarray(k_s_vec, dtype=np.float64)
    k0 = 2.0 * np.pi * float(frequency_hz) / C0
    for name, vec in (("k_i_vec", k_i), ("k_s_vec", k_s)):
        if abs(np.linalg.norm(vec) - k0) > 1e-6 * k0:
            raise ValueError(f"|{name}| does not match 2*pi*frequency_hz/c0")
    q = k_s - k_i
    q_n = float(np.dot(q, n_hat))
    nu, nv = (n_quad, n_quad) if np.isscalar(n_quad) else n_quad
    u_nodes, u_weights = np.polynomial.legendre.leggauss(int(nu))
    v_nodes, v_weights = np.polynomial.legendre.leggauss(int(nv))
    u = 0.5 * (u_nodes + 1.0)
    v = 0.5 * (v_nodes + 1.0)
    w2d = (0.5 * u_weights)[:, None] * (0.5 * v_weights)[None, :]
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    pos = (
        p00[None, None, :]
        + grid_u[:, :, None] * e_u[None, None, :]
        + grid_v[:, :, None] * e_v[None, None, :]
    )
    h = np.asarray(height_fn(grid_u, grid_v), dtype=np.float64)
    phase = pos @ q + q_n * h
    return complex((w2d * np.exp(-1j * phase)).sum() * area_jacobian)


def hemisphere_integral(f: Callable, n_theta: int = 64, n_phi: int = 128):
    """Integrate ``f(cos_theta, phi)`` over the upper hemisphere solid angle.

    ``Int f dOmega = Int_0^{2pi} dphi Int_0^1 f d(cos_theta)``:
    Gauss-Legendre in ``mu = cos_theta`` over (0, 1], uniform midpoint rule
    in ``phi`` (spectrally accurate for periodic integrands). ``f`` receives
    broadcastable arrays ``(mu[n_theta, 1], phi[1, n_phi])``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    mu = 0.5 * (nodes + 1.0)
    w_mu = 0.5 * weights
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    values = np.broadcast_to(
        np.asarray(f(mu[:, None], phi[None, :]), dtype=np.float64), (n_theta, n_phi)
    )
    return float((values * w_mu[:, None]).sum() * (2.0 * np.pi / n_phi))


# Preserve the historical public import/pickle path for one compatibility
# cycle while this implementation is owned by ``physics.reference``.