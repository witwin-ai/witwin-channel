from __future__ import annotations

"""UTD diffraction helpers using the Kouyoumjian-Pathak formulation."""

import enum

import drjit as dr
import witwin as wt

from ...utils.constants import SMALL_EPS, UTD_SLOPE_DERIVATIVE_STEP
from ...utils.utd import cot, fresnel_integral, f_utd


class ABranch(enum.Enum):
    """Branch selector for the a± rounding in UTD beta terms."""
    PLUS = "plus"
    MINUS = "minus"


class DiffAngle(enum.Enum):
    """Angle selector for UTD diffraction coefficient derivatives."""
    PHI = "phi"
    PHI_PRIME = "phi_prime"


def _shadow_boundary_a_threshold(n: wt.Float) -> wt.Float:
    return wt.Float(8.0 * SMALL_EPS * SMALL_EPS) * dr.maximum(
        dr.square(n),
        wt.Float(1.0),
    )


def _cot_f_product(
    cot_arg: wt.Float,
    a: wt.Float,
    kL: wt.Float,
    *,
    n: wt.Float | None = None,
    cot_sign: float | None = None,
    a_1: wt.Float | None = None,
    transition: wt.Complex2f | None = None,
    cot_value: wt.Float | None = None,
) -> wt.Complex2f:
    resolved_cot_value = cot(cot_arg) if cot_value is None else cot_value
    resolved_transition = f_utd(kL * a) if transition is None else transition
    raw_product = resolved_cot_value * resolved_transition
    if n is None or cot_sign is None or a_1 is None:
        return raw_product

    near_shadow_boundary = a <= _shadow_boundary_a_threshold(n)
    fallback_sign = dr.select(
        resolved_cot_value >= wt.Float(0.0),
        wt.Float(1.0),
        wt.Float(-1.0),
    )
    limit_sign = dr.select(
        dr.abs(a_1) > wt.Float(SMALL_EPS),
        wt.Float(float(cot_sign)) * dr.sign(a_1),
        fallback_sign,
    )
    limit_scale = limit_sign * n * dr.sqrt(dr.pi * dr.maximum(kL, wt.Float(0.0)))
    shadow_boundary_limit = wt.Complex2f(limit_scale, limit_scale)
    blend = dr.minimum(
        wt.Float(1.0),
        a * dr.rcp(dr.maximum(_shadow_boundary_a_threshold(n), wt.Float(1.0e-20))),
    )
    blended_product = wt.Complex2f(
        shadow_boundary_limit.real
        + blend * (raw_product.real - shadow_boundary_limit.real),
        shadow_boundary_limit.imag
        + blend * (raw_product.imag - shadow_boundary_limit.imag),
    )
    return wt.Complex2f(
        dr.select(near_shadow_boundary, blended_product.real, raw_product.real),
        dr.select(near_shadow_boundary, blended_product.imag, raw_product.imag),
    )


def _compute_a_pm(beta: wt.Float, n: wt.Float) -> tuple[wt.Float, wt.Float]:
    two_n_pi = 2 * n * dr.pi
    n_plus = dr.round((beta + dr.pi) * dr.rcp(two_n_pi))
    n_minus = dr.round((beta - dr.pi) * dr.rcp(two_n_pi))
    a_plus = 2 * dr.cos((two_n_pi * n_plus - beta) / 2) ** 2
    a_minus = 2 * dr.cos((two_n_pi * n_minus - beta) / 2) ** 2
    return a_plus, a_minus


def diffraction_coefficient(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    L: wt.Float,
    beta0: wt.Float = None,
    R0: wt.Complex2f = None,
    Rn: wt.Complex2f = None,
) -> wt.Complex2f:
    """UTD diffraction coefficient for a wedge."""
    sin_beta0 = wt.Float(1.0) if beta0 is None else dr.sin(beta0)
    factor = -dr.exp(wt.Complex2f(0, -dr.pi / 4))
    factor *= dr.rcp(2 * n * dr.safe_sqrt(dr.two_pi * k) * sin_beta0)
    dif_phi, sum_phi = phi - phi_prime, phi + phi_prime
    kL = k * L
    d1, _, _ = _beta_term_state(dif_phi, n, kL, +1.0, ABranch.PLUS)
    d2, _, _ = _beta_term_state(dif_phi, n, kL, -1.0, ABranch.MINUS)
    d3, _, _ = _beta_term_state(sum_phi, n, kL, +1.0, ABranch.PLUS)
    d4, _, _ = _beta_term_state(sum_phi, n, kL, -1.0, ABranch.MINUS)
    R0 = wt.Complex2f(-1, 0) if R0 is None else R0
    Rn = wt.Complex2f(-1, 0) if Rn is None else Rn
    return factor * (d1 + d2 + Rn * d3 + R0 * d4)


def diffraction_coefficient_2d(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    s: wt.Float,
    s_prime: wt.Float,
    R0: wt.Complex2f = None,
    Rn: wt.Complex2f = None,
) -> wt.Complex2f:
    """2D wrapper: computes L = s*s'/(s+s')."""
    L = s * s_prime * dr.rcp(s + s_prime)
    return diffraction_coefficient(phi, phi_prime, n, k, L, R0=R0, Rn=Rn)


def _poly_value_first_second(coeffs: tuple[float, ...], x: wt.Float) -> tuple[wt.Float, wt.Float, wt.Float]:
    value = wt.Float(coeffs[-1])
    first = dr.zeros(wt.Float, dr.width(x))
    second = dr.zeros(wt.Float, dr.width(x))
    for coeff in reversed(coeffs[:-1]):
        second = second * x + 2.0 * first
        first = first * x + value
        value = value * x + wt.Float(coeff)
    return value, first, second


def _safe_positive(x: wt.Float) -> wt.Float:
    return dr.maximum(x, wt.Float(SMALL_EPS))


def fresnel_integral_with_derivatives(x: wt.Float) -> tuple[wt.Complex2f, wt.Complex2f, wt.Complex2f]:
    """Return the Fresnel approximation and its first/second x-derivatives."""
    a = (+1.595769140, -0.000001702, -6.808568854, -0.000576361, +6.920691902, -0.016898657,
         -3.050485660, -0.075752419, +0.850663781, -0.025639041, -0.150230960, +0.034404779)
    b = (-0.000000033, +4.255387524, -0.000092810, -7.780020400, -0.009520895, +5.075161298,
         -0.138341947, -1.363729124, -0.403349276, +0.702222016, -0.216195929, +0.019547031)
    c = (+0.000000000, -0.024933975, +0.000003936, +0.005770956, +0.000689892, -0.009497136,
         +0.011948809, -0.006748873, +0.000246420, +0.002102967, -0.001217930, +0.000233939)
    d = (+0.199471140, +0.000000023, -0.009351341, +0.000023006, +0.004851466, +0.001903218,
         -0.017122914, +0.029064067, -0.027928955, +0.016497308, -0.005598515, +0.000838386)

    x_pos = x >= 0.0
    x_abs = dr.abs(x)
    safe_x = _safe_positive(x_abs)
    cond = x_abs < 4.0

    arg_small = x_abs * 0.25
    arg_large = 4.0 * dr.rcp(safe_x)
    arg = dr.select(cond, arg_small, arg_large)
    arg_safe = _safe_positive(arg)

    arg1_small = wt.Float(0.25)
    arg2_small = wt.Float(0.0)
    arg1_large = -4.0 * dr.rcp(safe_x * safe_x)
    arg2_large = 8.0 * dr.rcp(safe_x * safe_x * safe_x)
    arg1 = dr.select(cond, arg1_small, arg1_large)
    arg2 = dr.select(cond, arg2_small, arg2_large)

    r_small, r_small_1, r_small_2 = _poly_value_first_second(a, arg_small)
    r_large, r_large_1, r_large_2 = _poly_value_first_second(c, arg_large)
    i_small, i_small_1, i_small_2 = _poly_value_first_second(b, arg_small)
    i_large, i_large_1, i_large_2 = _poly_value_first_second(d, arg_large)

    r_coef = dr.select(cond, r_small, r_large)
    i_coef = dr.select(cond, i_small, i_large)
    r_coef_1 = dr.select(
        cond,
        r_small_1 * arg1_small,
        r_large_1 * arg1_large,
    )
    i_coef_1 = dr.select(
        cond,
        i_small_1 * arg1_small,
        i_large_1 * arg1_large,
    )
    r_coef_2 = dr.select(
        cond,
        r_small_2 * arg1_small * arg1_small,
        r_large_2 * arg1_large * arg1_large + r_large_1 * arg2_large,
    )
    i_coef_2 = dr.select(
        cond,
        i_small_2 * arg1_small * arg1_small,
        i_large_2 * arg1_large * arg1_large + i_large_1 * arg2_large,
    )

    arg_sqrt = dr.sqrt(arg_safe)
    arg_sqrt_1 = 0.5 * arg1 * dr.rcp(arg_sqrt)
    arg_sqrt_2 = 0.5 * arg2 * dr.rcp(arg_sqrt) - 0.25 * arg1 * arg1 * dr.rcp(arg_safe * arg_sqrt)

    r_part = r_coef * arg_sqrt
    r_part_1 = r_coef_1 * arg_sqrt + r_coef * arg_sqrt_1
    r_part_2 = r_coef_2 * arg_sqrt + 2.0 * r_coef_1 * arg_sqrt_1 + r_coef * arg_sqrt_2
    i_part = -i_coef * arg_sqrt
    i_part_1 = -(i_coef_1 * arg_sqrt + i_coef * arg_sqrt_1)
    i_part_2 = -(i_coef_2 * arg_sqrt + 2.0 * i_coef_1 * arg_sqrt_1 + i_coef * arg_sqrt_2)

    sin_x, cos_x = dr.sincos(x_abs)
    f_r = cos_x * r_part - sin_x * i_part
    f_i = cos_x * i_part + sin_x * r_part
    f_r_1 = cos_x * (r_part_1 - i_part) - sin_x * (r_part + i_part_1)
    f_i_1 = cos_x * (i_part_1 + r_part) + sin_x * (r_part_1 - i_part)
    f_r_2 = cos_x * (r_part_2 - r_part - 2.0 * i_part_1) - sin_x * (2.0 * r_part_1 - i_part + i_part_2)
    f_i_2 = cos_x * (i_part_2 + 2.0 * r_part_1 - i_part) + sin_x * (r_part_2 - r_part - 2.0 * i_part_1)

    c_out = dr.select(cond, f_r, f_r + 0.5)
    s_out = dr.select(cond, f_i, f_i + 0.5)
    c_out_1 = f_r_1
    s_out_1 = f_i_1
    c_out_2 = f_r_2
    s_out_2 = f_i_2

    value = wt.Complex2f(
        dr.select(x_pos, c_out, -c_out),
        dr.select(x_pos, s_out, -s_out),
    )
    first = wt.Complex2f(c_out_1, s_out_1)
    second = wt.Complex2f(
        dr.select(x_pos, c_out_2, -c_out_2),
        dr.select(x_pos, s_out_2, -s_out_2),
    )
    return value, first, second

def f_utd_with_derivatives(x: wt.Float) -> tuple[wt.Complex2f, wt.Complex2f, wt.Complex2f]:
    """Return the transition function and its first/second x-derivatives."""
    safe_x = _safe_positive(x)
    fresnel_value, fresnel_first, fresnel_second = fresnel_integral_with_derivatives(x)
    fresnel_conj = dr.conj(fresnel_value)
    fresnel_first_conj = dr.conj(fresnel_first)
    fresnel_second_conj = dr.conj(fresnel_second)

    prefactor = dr.sqrt(dr.pi * safe_x / 2.0)
    prefactor_1 = 0.5 * prefactor * dr.rcp(safe_x)
    prefactor_2 = -0.25 * prefactor * dr.rcp(safe_x * safe_x)
    phase = dr.exp(wt.Complex2f(0, x))
    phase_1 = wt.Complex2f(0.0, 1.0) * phase
    phase_2 = -phase
    bracket = wt.Complex2f(1.0, 1.0) - wt.Complex2f(0.0, 2.0) * fresnel_conj
    bracket_1 = -wt.Complex2f(0.0, 2.0) * fresnel_first_conj
    bracket_2 = -wt.Complex2f(0.0, 2.0) * fresnel_second_conj

    value = prefactor * phase * bracket
    first = (
        prefactor_1 * phase * bracket
        + prefactor * phase_1 * bracket
        + prefactor * phase * bracket_1
    )
    second = (
        prefactor_2 * phase * bracket
        + 2.0 * prefactor_1 * phase_1 * bracket
        + 2.0 * prefactor_1 * phase * bracket_1
        + prefactor * phase_2 * bracket
        + 2.0 * prefactor * phase_1 * bracket_1
        + prefactor * phase * bracket_2
    )
    return value, first, second

def _beta_term_state(beta: wt.Float, n: wt.Float, kL: wt.Float, cot_sign: float, a_branch: ABranch):
    cot_arg, a, a_1, cot_value, cot_1, cot_2, x, x_1, x_2 = _beta_term_metadata(
        beta,
        n,
        kL,
        cot_sign,
        a_branch,
    )
    transition, transition_1, transition_2 = f_utd_with_derivatives(x)
    forward_transition = f_utd(x)
    product_value = _cot_f_product(
        cot_arg,
        a,
        kL,
        n=n,
        cot_sign=cot_sign,
        a_1=a_1,
        transition=forward_transition,
        cot_value=cot_value,
    )
    return _assemble_beta_term_state(
        product_value,
        cot_value,
        cot_1,
        cot_2,
        x_1,
        x_2,
        transition,
        transition_1,
        transition_2,
    )


def _beta_term_metadata(beta: wt.Float, n: wt.Float, kL: wt.Float, cot_sign: float, a_branch: ABranch):
    two_n = 2.0 * n
    two_n_pi = 2.0 * n * dr.pi
    if a_branch is ABranch.PLUS:
        round_index = dr.round((beta + dr.pi) * dr.rcp(two_n_pi))
    elif a_branch is ABranch.MINUS:
        round_index = dr.round((beta - dr.pi) * dr.rcp(two_n_pi))
    else:
        raise ValueError(f"Unsupported a_branch selector: {a_branch}")

    phase_offset = two_n_pi * round_index - beta
    a = 2.0 * dr.square(dr.cos(0.5 * phase_offset))
    a_1 = dr.sin(phase_offset)
    a_2 = 1.0 - a

    cot_arg = (dr.pi + wt.Float(cot_sign) * beta) / two_n
    cot_value = cot(cot_arg)
    cot_1 = -(wt.Float(cot_sign) / two_n) * (1.0 + dr.square(cot_value))
    cot_2 = 0.5 * cot_value * (1.0 + dr.square(cot_value)) * dr.rcp(n * n)

    x = kL * a
    x_1 = kL * a_1
    x_2 = kL * a_2
    return cot_arg, a, a_1, cot_value, cot_1, cot_2, x, x_1, x_2


def _assemble_beta_term_state(
    product_value: wt.Complex2f,
    cot_value: wt.Float,
    cot_1: wt.Float,
    cot_2: wt.Float,
    x_1: wt.Float,
    x_2: wt.Float,
    transition: wt.Complex2f,
    transition_1: wt.Complex2f,
    transition_2: wt.Complex2f,
):
    value = product_value
    first = cot_1 * transition + cot_value * transition_1 * x_1
    second = (
        cot_2 * transition
        + 2.0 * cot_1 * transition_1 * x_1
        + cot_value * (transition_2 * dr.square(x_1) + transition_1 * x_2)
    )
    return value, first, second


def _diffraction_beta_groups(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    s: wt.Float,
    s_prime: wt.Float,
    R0: wt.Complex2f,
    Rn: wt.Complex2f,
):
    L = s * s_prime * dr.rcp(s + s_prime)
    kL = k * L
    dif_phi = phi - phi_prime
    sum_phi = phi + phi_prime
    factor = -dr.exp(wt.Complex2f(0, -dr.pi / 4))
    factor *= dr.rcp(2.0 * n * dr.safe_sqrt(dr.two_pi * k))
    term_states = [
        _beta_term_state(dif_phi, n, kL, +1.0, ABranch.PLUS),
        _beta_term_state(dif_phi, n, kL, -1.0, ABranch.MINUS),
        _beta_term_state(sum_phi, n, kL, +1.0, ABranch.PLUS),
        _beta_term_state(sum_phi, n, kL, -1.0, ABranch.MINUS),
    ]
    dif_plus, dif_plus_1, dif_plus_2 = term_states[0]
    dif_minus, dif_minus_1, dif_minus_2 = term_states[1]
    sum_plus, sum_plus_1, sum_plus_2 = term_states[2]
    sum_minus, sum_minus_1, sum_minus_2 = term_states[3]

    dif_group = dif_plus + dif_minus
    dif_group_1 = dif_plus_1 + dif_minus_1
    dif_group_2 = dif_plus_2 + dif_minus_2
    sum_group = Rn * sum_plus + R0 * sum_minus
    sum_group_1 = Rn * sum_plus_1 + R0 * sum_minus_1
    sum_group_2 = Rn * sum_plus_2 + R0 * sum_minus_2
    return factor, dif_group, dif_group_1, dif_group_2, sum_group, sum_group_1, sum_group_2


def _diffraction_beta_groups_3d(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    s: wt.Float,
    s_prime: wt.Float,
    sin_beta0: wt.Float,
    R0: wt.Complex2f,
    Rn: wt.Complex2f,
):
    safe_sin_beta0 = dr.maximum(sin_beta0, wt.Float(SMALL_EPS))
    L = s * s_prime * dr.rcp(s + s_prime) * dr.square(safe_sin_beta0)
    kL = k * L
    dif_phi = phi - phi_prime
    sum_phi = phi + phi_prime
    factor = -dr.exp(wt.Complex2f(0, -dr.pi / 4))
    factor *= dr.rcp(2.0 * n * dr.safe_sqrt(dr.two_pi * k) * safe_sin_beta0)
    term_states = [
        _beta_term_state(dif_phi, n, kL, +1.0, ABranch.PLUS),
        _beta_term_state(dif_phi, n, kL, -1.0, ABranch.MINUS),
        _beta_term_state(sum_phi, n, kL, +1.0, ABranch.PLUS),
        _beta_term_state(sum_phi, n, kL, -1.0, ABranch.MINUS),
    ]
    dif_plus, dif_plus_1, dif_plus_2 = term_states[0]
    dif_minus, dif_minus_1, dif_minus_2 = term_states[1]
    sum_plus, sum_plus_1, sum_plus_2 = term_states[2]
    sum_minus, sum_minus_1, sum_minus_2 = term_states[3]

    dif_group = dif_plus + dif_minus
    dif_group_1 = dif_plus_1 + dif_minus_1
    dif_group_2 = dif_plus_2 + dif_minus_2
    sum_group = Rn * sum_plus + R0 * sum_minus
    sum_group_1 = Rn * sum_plus_1 + R0 * sum_minus_1
    sum_group_2 = Rn * sum_plus_2 + R0 * sum_minus_2
    return factor, dif_group, dif_group_1, dif_group_2, sum_group, sum_group_1, sum_group_2


def diffraction_coefficient_2d_angle_derivative(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    s: wt.Float,
    s_prime: wt.Float,
    angle: DiffAngle = DiffAngle.PHI_PRIME,
    step: float = UTD_SLOPE_DERIVATIVE_STEP,
    R0: wt.Complex2f = None,
    Rn: wt.Complex2f = None,
) -> wt.Complex2f:
    del step
    R0 = wt.Complex2f(-1.0, 0.0) if R0 is None else R0
    Rn = wt.Complex2f(-1.0, 0.0) if Rn is None else Rn
    factor, dif_group, dif_group_1, _, sum_group, sum_group_1, _ = _diffraction_beta_groups(
        phi, phi_prime, n, k, s, s_prime, R0, Rn
    )
    if angle is DiffAngle.PHI:
        return factor * (dif_group_1 + sum_group_1)
    if angle is DiffAngle.PHI_PRIME:
        return factor * (-dif_group_1 + sum_group_1)
    raise ValueError(f"Unsupported angle selector: {angle}")


def diffraction_coefficient_3d(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    s: wt.Float,
    s_prime: wt.Float,
    sin_beta0: wt.Float,
    R0: wt.Complex2f = None,
    Rn: wt.Complex2f = None,
) -> wt.Complex2f:
    R0 = wt.Complex2f(-1.0, 0.0) if R0 is None else R0
    Rn = wt.Complex2f(-1.0, 0.0) if Rn is None else Rn
    factor, dif_group, _, _, sum_group, _, _ = _diffraction_beta_groups_3d(
        phi, phi_prime, n, k, s, s_prime, sin_beta0, R0, Rn
    )
    return factor * (dif_group + sum_group)


def diffraction_coefficient_3d_angle_derivative(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    s: wt.Float,
    s_prime: wt.Float,
    sin_beta0: wt.Float,
    angle: DiffAngle = DiffAngle.PHI_PRIME,
    step: float = UTD_SLOPE_DERIVATIVE_STEP,
    R0: wt.Complex2f = None,
    Rn: wt.Complex2f = None,
) -> wt.Complex2f:
    del step
    R0 = wt.Complex2f(-1.0, 0.0) if R0 is None else R0
    Rn = wt.Complex2f(-1.0, 0.0) if Rn is None else Rn
    factor, dif_group, dif_group_1, _, sum_group, sum_group_1, _ = _diffraction_beta_groups_3d(
        phi, phi_prime, n, k, s, s_prime, sin_beta0, R0, Rn
    )
    if angle is DiffAngle.PHI:
        return factor * (dif_group_1 + sum_group_1)
    if angle is DiffAngle.PHI_PRIME:
        return factor * (-dif_group_1 + sum_group_1)
    raise ValueError(f"Unsupported angle selector: {angle}")


def slope_diffraction_coefficient_2d(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    s: wt.Float,
    s_prime: wt.Float,
    step: float = UTD_SLOPE_DERIVATIVE_STEP,
    R0: wt.Complex2f = None,
    Rn: wt.Complex2f = None,
) -> wt.Complex2f:
    d_dphi_prime = diffraction_coefficient_2d_angle_derivative(
        phi, phi_prime, n, k, s, s_prime, angle=DiffAngle.PHI_PRIME, step=step, R0=R0, Rn=Rn
    )
    return wt.Complex2f(0.0, -1.0) * d_dphi_prime / k


def slope_diffraction_coefficient_3d(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    s: wt.Float,
    s_prime: wt.Float,
    sin_beta0: wt.Float,
    step: float = UTD_SLOPE_DERIVATIVE_STEP,
    R0: wt.Complex2f = None,
    Rn: wt.Complex2f = None,
) -> wt.Complex2f:
    d_dphi_prime = diffraction_coefficient_3d_angle_derivative(
        phi, phi_prime, n, k, s, s_prime, sin_beta0, angle=DiffAngle.PHI_PRIME, step=step, R0=R0, Rn=Rn
    )
    return wt.Complex2f(0.0, -1.0) * d_dphi_prime / k


def slope_diffraction_coefficient_2d_angle_derivative(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    s: wt.Float,
    s_prime: wt.Float,
    angle: DiffAngle = DiffAngle.PHI,
    step: float = UTD_SLOPE_DERIVATIVE_STEP,
    R0: wt.Complex2f = None,
    Rn: wt.Complex2f = None,
) -> wt.Complex2f:
    del step
    R0 = wt.Complex2f(-1.0, 0.0) if R0 is None else R0
    Rn = wt.Complex2f(-1.0, 0.0) if Rn is None else Rn
    factor, _, _, dif_group_2, _, _, sum_group_2 = _diffraction_beta_groups(
        phi, phi_prime, n, k, s, s_prime, R0, Rn
    )
    if angle is DiffAngle.PHI:
        cross_derivative = factor * (-dif_group_2 + sum_group_2)
    elif angle is DiffAngle.PHI_PRIME:
        cross_derivative = factor * (dif_group_2 + sum_group_2)
    else:
        raise ValueError(f"Unsupported angle selector: {angle}")
    return wt.Complex2f(0.0, -1.0) * cross_derivative / k


def slope_diffraction_coefficient_3d_angle_derivative(
    phi: wt.Float,
    phi_prime: wt.Float,
    n: wt.Float,
    k: wt.Float,
    s: wt.Float,
    s_prime: wt.Float,
    sin_beta0: wt.Float,
    angle: DiffAngle = DiffAngle.PHI,
    step: float = UTD_SLOPE_DERIVATIVE_STEP,
    R0: wt.Complex2f = None,
    Rn: wt.Complex2f = None,
) -> wt.Complex2f:
    del step
    R0 = wt.Complex2f(-1.0, 0.0) if R0 is None else R0
    Rn = wt.Complex2f(-1.0, 0.0) if Rn is None else Rn
    factor, _, _, dif_group_2, _, _, sum_group_2 = _diffraction_beta_groups_3d(
        phi, phi_prime, n, k, s, s_prime, sin_beta0, R0, Rn
    )
    if angle is DiffAngle.PHI:
        cross_derivative = factor * (-dif_group_2 + sum_group_2)
    elif angle is DiffAngle.PHI_PRIME:
        cross_derivative = factor * (dif_group_2 + sum_group_2)
    else:
        raise ValueError(f"Unsupported angle selector: {angle}")
    return wt.Complex2f(0.0, -1.0) * cross_derivative / k


__all__ = [
    "ABranch",
    "DiffAngle",
    "_cot_f_product",
    "cot",
    "diffraction_coefficient",
    "diffraction_coefficient_2d",
    "diffraction_coefficient_3d",
    "diffraction_coefficient_2d_angle_derivative",
    "diffraction_coefficient_3d_angle_derivative",
    "f_utd",
    "f_utd_with_derivatives",
    "fresnel_integral",
    "fresnel_integral_with_derivatives",
    "slope_diffraction_coefficient_2d",
    "slope_diffraction_coefficient_2d_angle_derivative",
    "slope_diffraction_coefficient_3d",
    "slope_diffraction_coefficient_3d_angle_derivative",
]

