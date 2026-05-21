"""Wave physics math: Fresnel reflection, UTD transition, legacy shadow weights.

All functions are pure DrJit math with no scene/solver state.
New reflection shadow/secondary-visibility attenuation should use
``reflection_segment_attenuation`` from the deterministic reflection package.
"""

from __future__ import annotations

import math

import drjit as dr
from witwin.channel import types as wt

from witwin.channel.core.numerics.constants import EPSILON_0, SMALL_EPS, SPEED_OF_LIGHT


def material_angular_frequency(wavelength) -> wt.Float:
    """Convert wavelength to angular frequency ``omega = 2*pi*c/lambda``."""
    return wt.Float(2.0 * math.pi * SPEED_OF_LIGHT / wavelength)


# ---------------------------------------------------------------------------
# Complex sqrt + Fresnel reflection
# ---------------------------------------------------------------------------


def _complex_sqrt(z: wt.Complex2f) -> wt.Complex2f:
    x, y = dr.real(z), dr.imag(z)
    r = dr.abs(z)
    nonzero = r > 0.0
    x_pos = x >= 0.0
    zero, one = wt.Float(0.0), wt.Float(1.0)

    real_mag = dr.sqrt(dr.select(x_pos & nonzero, 0.5 * (r + x), zero))
    imag_mag = dr.sqrt(dr.select((~x_pos) & nonzero, 0.5 * (r - x), zero))

    safe_real_mag = dr.select(real_mag > 0.0, real_mag, one)
    safe_imag_mag = dr.select(imag_mag > 0.0, imag_mag, one)

    real_part = dr.select(x_pos, real_mag, dr.abs(y) * dr.rcp(2.0 * safe_imag_mag))
    imag_part = dr.select(
        x_pos,
        y * dr.rcp(2.0 * safe_real_mag),
        dr.select(y < 0.0, -imag_mag, imag_mag),
    )
    return wt.Complex2f(
        dr.select(nonzero, real_part, zero),
        dr.select(nonzero, imag_part, zero),
    )


def complex_relative_permittivity(eta_r: wt.Float, sigma: wt.Float, omega: wt.Float) -> wt.Complex2f:
    return wt.Complex2f(eta_r, -sigma * dr.rcp(omega * EPSILON_0))


def fresnel_reflection(cos_theta: wt.Float, eta: wt.Complex2f, mu_r: wt.Float | float = 1.0) -> tuple[wt.Complex2f, wt.Complex2f]:
    """Returns ``(r_te, r_tm)`` Fresnel reflection coefficients."""
    sin_theta_sqr = 1.0 - cos_theta * cos_theta
    mu = wt.Complex2f(wt.Float(mu_r), wt.Float(0.0))
    a = _complex_sqrt(mu * eta - sin_theta_sqr)
    r_te = (mu * cos_theta - a) * dr.rcp(mu * cos_theta + a)
    r_tm = (eta * cos_theta - a) * dr.rcp(eta * cos_theta + a)
    return r_te, r_tm


def scalar_fresnel_reflection(
    cos_theta: wt.Float,
    eta_r: wt.Float,
    sigma: wt.Float,
    omega: wt.Float,
    mu_r: wt.Float | float = 1.0,
    gain: wt.Float | float = 1.0,
) -> wt.Complex2f:
    """Scalar Fresnel reflection coefficient (TE/TM averaged with gain)."""
    cos_theta = dr.clip(cos_theta, wt.Float(SMALL_EPS), wt.Float(1.0))
    eta = complex_relative_permittivity(eta_r, sigma, omega)
    r_te, r_tm = fresnel_reflection(cos_theta, eta, mu_r=mu_r)
    coeff = wt.Complex2f(gain, 0.0) * 0.5 * (r_te + r_tm)
    return wt.Complex2f(
        dr.select(dr.isfinite(coeff.real), coeff.real, wt.Float(0.0)),
        dr.select(dr.isfinite(coeff.imag), coeff.imag, wt.Float(0.0)),
    )


# ---------------------------------------------------------------------------
# UTD: cot, Fresnel integral, transition function
# ---------------------------------------------------------------------------


def cot(x: wt.Float, eps: float = SMALL_EPS) -> wt.Float:
    sin_x, cos_x = dr.sincos(x)
    eps_f = wt.Float(eps)
    denom = dr.select(dr.abs(sin_x) < eps_f, dr.sign(sin_x + eps_f) * eps_f, sin_x)
    y = cos_x / denom
    y = dr.select(dr.isnan(y), 0, y)
    y = dr.select(dr.isinf(y), 0, y)
    return y


# Boersma polynomial coefficients (12 terms each) for the Fresnel integral.
# Rows ordered: (real, low-x), (imag, low-x), (real, high-x), (imag, high-x).
_BOERSMA_R_LO = (
    +1.595769140, -0.000001702, -6.808568854, -0.000576361,
    +6.920691902, -0.016898657, -3.050485660, -0.075752419,
    +0.850663781, -0.025639041, -0.150230960, +0.034404779,
)
_BOERSMA_I_LO = (
    -0.000000033, +4.255387524, -0.000092810, -7.780020400,
    -0.009520895, +5.075161298, -0.138341947, -1.363729124,
    -0.403349276, +0.702222016, -0.216195929, +0.019547031,
)
_BOERSMA_R_HI = (
    +0.000000000, -0.024933975, +0.000003936, +0.005770956,
    +0.000689892, -0.009497136, +0.011948809, -0.006748873,
    +0.000246420, +0.002102967, -0.001217930, +0.000233939,
)
_BOERSMA_I_HI = (
    +0.199471140, +0.000000023, -0.009351341, +0.000023006,
    +0.004851466, +0.001903218, -0.017122914, +0.029064067,
    -0.027928955, +0.016497308, -0.005598515, +0.000838386,
)


def fresnel_integral(x: wt.Float) -> wt.Complex2f:
    """Fresnel integral using Boersma coefficients (unrolled for GPU efficiency)."""
    x_pos = x > 0
    x = dr.abs(x)
    cond = x < 4
    arg = dr.select(cond, x * 0.25, 4 * dr.rcp(x))

    powers = [wt.Float(1.0), arg]
    for _ in range(10):
        powers.append(powers[-1] * arg)

    def _select_poly(coef_lo, coef_hi):
        terms = [dr.select(cond, wt.Float(lo), wt.Float(hi)) * power
                 for lo, hi, power in zip(coef_lo, coef_hi, powers)]
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    arg_sqrt = dr.sqrt(arg)
    r_part = _select_poly(_BOERSMA_R_LO, _BOERSMA_R_HI) * arg_sqrt
    i_part = -_select_poly(_BOERSMA_I_LO, _BOERSMA_I_HI) * arg_sqrt

    sin_x, cos_x = dr.sincos(x)
    f_r = cos_x * r_part - sin_x * i_part
    f_i = cos_x * i_part + sin_x * r_part
    c_out = dr.select(cond, f_r, f_r + 0.5)
    s_out = dr.select(cond, f_i, f_i + 0.5)
    c_out = dr.select(x_pos, c_out, -c_out)
    s_out = dr.select(x_pos, s_out, -s_out)
    return wt.Complex2f(c_out, s_out)


def f_utd(x: wt.Float) -> wt.Complex2f:
    """UTD transition function: ``F(x) = sqrt(pi*x/2) * e^(jx) * (1 + j - 2j*F_c(x))``."""
    f = wt.Complex2f(1, 1)
    f -= wt.Complex2f(0, 2) * dr.conj(fresnel_integral(x))
    f *= dr.sqrt(dr.pi * x / 2)
    f *= dr.exp(wt.Complex2f(0, x))
    return f


# ---------------------------------------------------------------------------
# Shadow support weights (UTD transition completion)
# ---------------------------------------------------------------------------


_BOUNDARY_EPS = 1.0e-3


def _smoothstep01(x):
    xc = dr.clip(x, wt.Float(0.0), wt.Float(1.0))
    return xc * xc * (wt.Float(3.0) - wt.Float(2.0) * xc)


def _shadow_opening_angle(wedge_n):
    return dr.maximum(wt.Float(2.0 * dr.pi) - wedge_n * dr.pi, wt.Float(2.0 * _BOUNDARY_EPS))


def _shadow_angle_ratio(wedge_n):
    return dr.minimum(_shadow_opening_angle(wedge_n) / dr.pi, wt.Float(1.0))


def shadow_decay_span_from_wedge_n(wedge_n):
    opening = _shadow_opening_angle(wedge_n)
    ratio = dr.minimum(opening / dr.pi, wt.Float(1.0))
    span = dr.maximum((wt.Float(0.17) + wt.Float(0.12) * ratio) * opening, wt.Float(8.0 * _BOUNDARY_EPS))
    return dr.minimum(span, wt.Float(0.5) * opening)


def _shadow_completion_weight_from_normalized_distance(u, wedge_n):
    curve = (wt.Float(0.88) * (wt.Float(1.0) - _smoothstep01(u))
             + wt.Float(0.12) * (wt.Float(1.0) - dr.clip(u, wt.Float(0.0), wt.Float(1.0))))
    decay_power = wt.Float(2.0) + (wt.Float(1.0) - _shadow_angle_ratio(wedge_n))
    return dr.power(curve, decay_power)


def shadow_completion_weight_from_distance(distance, wedge_n):
    span = shadow_decay_span_from_wedge_n(wedge_n)
    u = dr.clip(distance / span, wt.Float(0.0), wt.Float(1.0))
    return _shadow_completion_weight_from_normalized_distance(u, wedge_n)


def shadow_support_amplitude_threshold(shadow_support_cutoff_db):
    if shadow_support_cutoff_db is None:
        return None
    cutoff_db = max(0.0, float(shadow_support_cutoff_db))
    return dr.power(wt.Float(10.0), wt.Float(-0.05 * cutoff_db))


def shadow_support_angle_from_cutoff_db(wedge_n, shadow_support_cutoff_db):
    span = shadow_decay_span_from_wedge_n(wedge_n)
    threshold = shadow_support_amplitude_threshold(shadow_support_cutoff_db)
    if threshold is None:
        return span
    if float(shadow_support_cutoff_db) <= 0.0:
        return dr.zeros(wt.Float, dr.width(wedge_n))

    low = dr.zeros(wt.Float, dr.width(wedge_n))
    high = dr.ones(wt.Float, dr.width(wedge_n))
    for _ in range(12):
        mid = wt.Float(0.5) * (low + high)
        keep = _shadow_completion_weight_from_normalized_distance(mid, wedge_n) >= threshold
        low = dr.select(keep, mid, low)
        high = dr.select(keep, high, mid)
    return span * low


__all__ = [
    "complex_relative_permittivity",
    "cot",
    "f_utd",
    "fresnel_integral",
    "fresnel_reflection",
    "material_angular_frequency",
    "scalar_fresnel_reflection",
    "shadow_completion_weight_from_distance",
    "shadow_decay_span_from_wedge_n",
    "shadow_support_amplitude_threshold",
    "shadow_support_angle_from_cutoff_db",
]
