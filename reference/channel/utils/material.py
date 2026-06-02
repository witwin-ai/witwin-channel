from __future__ import annotations

"""Material and Fresnel response helpers."""

import drjit as dr
import witwin as wt

from .constants import EPSILON_0, SMALL_EPS


def complex_sqrt(z: wt.Complex2f) -> wt.Complex2f:
    x = dr.real(z)
    y = dr.imag(z)
    r = dr.abs(z)
    nonzero = r > 0.0
    x_nonnegative = x >= 0.0
    zero = wt.Float(0.0)
    one = wt.Float(1.0)

    real_mag = dr.sqrt(dr.select(x_nonnegative & nonzero, 0.5 * (r + x), zero))
    imag_mag = dr.sqrt(dr.select((~x_nonnegative) & nonzero, 0.5 * (r - x), zero))

    safe_real_mag = dr.select(real_mag > 0.0, real_mag, one)
    safe_imag_mag = dr.select(imag_mag > 0.0, imag_mag, one)

    real_part = dr.select(
        x_nonnegative,
        real_mag,
        dr.abs(y) * dr.rcp(2.0 * safe_imag_mag),
    )
    imag_part = dr.select(
        x_nonnegative,
        y * dr.rcp(2.0 * safe_real_mag),
        dr.select(y < 0.0, -imag_mag, imag_mag),
    )
    return wt.Complex2f(
        dr.select(nonzero, real_part, zero),
        dr.select(nonzero, imag_part, zero),
    )


def complex_relative_permittivity(eta_r: wt.Float, sigma: wt.Float, omega: wt.Float) -> wt.Complex2f:
    return wt.Complex2f(eta_r, -sigma * dr.rcp(omega * EPSILON_0))


def fresnel_reflection(cos_theta: wt.Float, eta: wt.Complex2f) -> tuple[wt.Complex2f, wt.Complex2f]:
    """Returns (r_te, r_tm) Fresnel reflection coefficients."""
    sin_theta_sqr = 1.0 - cos_theta * cos_theta
    a = complex_sqrt(eta - sin_theta_sqr)
    r_te = (cos_theta - a) * dr.rcp(cos_theta + a)
    r_tm = (eta * cos_theta - a) * dr.rcp(eta * cos_theta + a)
    return r_te, r_tm


def scalar_fresnel_reflection(
    cos_theta: wt.Float,
    eta_r: wt.Float,
    sigma: wt.Float,
    omega: wt.Float,
    gain: wt.Float | float = 1.0,
) -> wt.Complex2f:
    """
    Legacy/diagnostic scalar reflection coefficient.

    This helper averages the local TE/TM Fresnel coefficients and applies an
    additional user gain factor per reflection event. The material-aware public
    field outputs no longer use this shortcut by default; they are derived from
    Jones/vector transport instead.
    """
    cos_theta = dr.clip(cos_theta, wt.Float(SMALL_EPS), wt.Float(1.0))
    eta = complex_relative_permittivity(eta_r, sigma, omega)
    r_te, r_tm = fresnel_reflection(cos_theta, eta)
    coeff = wt.Complex2f(gain, 0.0) * 0.5 * (r_te + r_tm)
    return wt.Complex2f(
        dr.select(dr.isfinite(coeff.real), coeff.real, wt.Float(0.0)),
        dr.select(dr.isfinite(coeff.imag), coeff.imag, wt.Float(0.0)),
    )


__all__ = [
    "complex_relative_permittivity",
    "complex_sqrt",
    "fresnel_reflection",
    "scalar_fresnel_reflection",
]

