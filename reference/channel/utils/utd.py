from __future__ import annotations

"""Shared UTD helper functions used across deterministic and MC solvers."""

import drjit as dr
import witwin as wt

from .constants import SMALL_EPS


def cot(x: wt.Float, eps: float = SMALL_EPS) -> wt.Float:
    sin_x, cos_x = dr.sincos(x)
    eps_f = wt.Float(eps)
    denom = dr.select(dr.abs(sin_x) < eps_f, dr.sign(sin_x + eps_f) * eps_f, sin_x)
    y = cos_x / denom
    if __debug__ and not bool(dr.flag(dr.JitFlag.Recording)):
        if dr.hint(dr.any(~dr.isfinite(y)), mode="scalar"):
            import warnings
            warnings.warn("cot: non-finite value after eps guard", stacklevel=2)
    y = dr.select(dr.isnan(y), 0, y)
    y = dr.select(dr.isinf(y), 0, y)
    return y


def fresnel_integral(x: wt.Float) -> wt.Complex2f:
    """Fresnel integral using Boersma coefficients (loop unrolled for GPU efficiency)."""
    a0, a1, a2, a3 = +1.595769140, -0.000001702, -6.808568854, -0.000576361
    a4, a5, a6, a7 = +6.920691902, -0.016898657, -3.050485660, -0.075752419
    a8, a9, a10, a11 = +0.850663781, -0.025639041, -0.150230960, +0.034404779

    b0, b1, b2, b3 = -0.000000033, +4.255387524, -0.000092810, -7.780020400
    b4, b5, b6, b7 = -0.009520895, +5.075161298, -0.138341947, -1.363729124
    b8, b9, b10, b11 = -0.403349276, +0.702222016, -0.216195929, +0.019547031

    c0, c1, c2, c3 = +0.000000000, -0.024933975, +0.000003936, +0.005770956
    c4, c5, c6, c7 = +0.000689892, -0.009497136, +0.011948809, -0.006748873
    c8, c9, c10, c11 = +0.000246420, +0.002102967, -0.001217930, +0.000233939

    d0, d1, d2, d3 = +0.199471140, +0.000000023, -0.009351341, +0.000023006
    d4, d5, d6, d7 = +0.004851466, +0.001903218, -0.017122914, +0.029064067
    d8, d9, d10, d11 = -0.027928955, +0.016497308, -0.005598515, +0.000838386

    x_pos = x > 0
    x = dr.abs(x)
    cond = x < 4
    arg = dr.select(cond, x * 0.25, 4 * dr.rcp(x))

    arg2 = arg * arg
    arg3 = arg2 * arg
    arg4 = arg2 * arg2
    arg5 = arg4 * arg
    arg6 = arg4 * arg2
    arg7 = arg6 * arg
    arg8 = arg4 * arg4
    arg9 = arg8 * arg
    arg10 = arg8 * arg2
    arg11 = arg8 * arg3

    r_coef = dr.select(cond, wt.Float(a0), wt.Float(c0)) + \
             dr.select(cond, wt.Float(a1), wt.Float(c1)) * arg + \
             dr.select(cond, wt.Float(a2), wt.Float(c2)) * arg2 + \
             dr.select(cond, wt.Float(a3), wt.Float(c3)) * arg3 + \
             dr.select(cond, wt.Float(a4), wt.Float(c4)) * arg4 + \
             dr.select(cond, wt.Float(a5), wt.Float(c5)) * arg5 + \
             dr.select(cond, wt.Float(a6), wt.Float(c6)) * arg6 + \
             dr.select(cond, wt.Float(a7), wt.Float(c7)) * arg7 + \
             dr.select(cond, wt.Float(a8), wt.Float(c8)) * arg8 + \
             dr.select(cond, wt.Float(a9), wt.Float(c9)) * arg9 + \
             dr.select(cond, wt.Float(a10), wt.Float(c10)) * arg10 + \
             dr.select(cond, wt.Float(a11), wt.Float(c11)) * arg11

    i_coef = dr.select(cond, wt.Float(b0), wt.Float(d0)) + \
             dr.select(cond, wt.Float(b1), wt.Float(d1)) * arg + \
             dr.select(cond, wt.Float(b2), wt.Float(d2)) * arg2 + \
             dr.select(cond, wt.Float(b3), wt.Float(d3)) * arg3 + \
             dr.select(cond, wt.Float(b4), wt.Float(d4)) * arg4 + \
             dr.select(cond, wt.Float(b5), wt.Float(d5)) * arg5 + \
             dr.select(cond, wt.Float(b6), wt.Float(d6)) * arg6 + \
             dr.select(cond, wt.Float(b7), wt.Float(d7)) * arg7 + \
             dr.select(cond, wt.Float(b8), wt.Float(d8)) * arg8 + \
             dr.select(cond, wt.Float(b9), wt.Float(d9)) * arg9 + \
             dr.select(cond, wt.Float(b10), wt.Float(d10)) * arg10 + \
             dr.select(cond, wt.Float(b11), wt.Float(d11)) * arg11

    arg_sqrt = dr.sqrt(arg)
    r_part = r_coef * arg_sqrt
    i_part = -i_coef * arg_sqrt

    sin_x, cos_x = dr.sincos(x)
    f_r = cos_x * r_part - sin_x * i_part
    f_i = cos_x * i_part + sin_x * r_part
    c_out = dr.select(cond, f_r, f_r + 0.5)
    s_out = dr.select(cond, f_i, f_i + 0.5)
    c_out = dr.select(x_pos, c_out, -c_out)
    s_out = dr.select(x_pos, s_out, -s_out)
    return wt.Complex2f(c_out, s_out)


def f_utd(x: wt.Float) -> wt.Complex2f:
    """UTD transition function: F(x) = sqrt(pi*x/2) * e^(jx) * (1 + j - 2j*F_c*(x))."""
    f = wt.Complex2f(1, 1)
    f -= wt.Complex2f(0, 2) * dr.conj(fresnel_integral(x))
    f *= dr.sqrt(dr.pi * x / 2)
    f *= dr.exp(wt.Complex2f(0, x))
    return f


__all__ = ["cot", "f_utd", "fresnel_integral"]
