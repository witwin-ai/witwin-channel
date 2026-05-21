"""Pure math primitives: UTD coefficients (UTDMath) and geometric helpers (GeometrySupport)."""

import enum
import math

import drjit as dr

from witwin.channel.deterministic import types as wt

from witwin.channel.core.runtime import Material, Tx, Wave, material_angular_frequency, resolve_surface_material
from witwin.channel.core.numerics.constants import EPS, SMALL_EPS
from witwin.channel.core.numerics.arrays import broadcast_point, broadcast_vector
from witwin.channel.core.numerics.arrays import repeat_complex, repeat_float, repeat_int
from witwin.channel.core.physics.wave_math import complex_relative_permittivity, fresnel_reflection, scalar_fresnel_reflection
from witwin.channel.core.physics.polarization import basis_from_first_vector, jones_operator_add, jones_operator_diagonal, jones_operator_identity, jones_operator_in_basis, jones_operator_scale, stable_perpendicular_basis
from witwin.channel.core.geometry.diffraction import normalize_in_wedge_plane, rotate_vector_around_axis
from witwin.channel.core.physics.wave_math import cot, f_utd

class ABranch(enum.Enum):
    """Branch selector for the a+/- rounding in UTD beta terms."""
    PLUS = 'plus'
    MINUS = 'minus'

class DiffAngle(enum.Enum):
    """Angle selector for UTD diffraction coefficient derivatives."""
    PHI = 'phi'
    PHI_PRIME = 'phi_prime'
OPERATOR_KEYS = ('m00', 'm01', 'm10', 'm11')


class UTDMath:

    def shadow_a_threshold(n: wt.Float) -> wt.Float:
        return wt.Float(8.0 * SMALL_EPS * SMALL_EPS) * dr.maximum(dr.square(n), wt.Float(1.0))

    def cot_f_product(cot_arg: wt.Float, a: wt.Float, kL: wt.Float, *, n: wt.Float | None=None, cot_sign: float | None=None, a_1: wt.Float | None=None, transition: wt.Complex2f | None=None, cot_value: wt.Float | None=None) -> wt.Complex2f:
        resolved_cot_value = cot(cot_arg) if cot_value is None else cot_value
        resolved_transition = f_utd(kL * a) if transition is None else transition
        raw_product = resolved_cot_value * resolved_transition
        if n is None or cot_sign is None or a_1 is None:
            return raw_product
        near_shadow_boundary = a <= UTDMath.shadow_a_threshold(n)
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
        blend = dr.minimum(wt.Float(1.0), a * dr.rcp(dr.maximum(UTDMath.shadow_a_threshold(n), wt.Float(1e-20))))
        blended_product = wt.Complex2f(shadow_boundary_limit.real + blend * (raw_product.real - shadow_boundary_limit.real), shadow_boundary_limit.imag + blend * (raw_product.imag - shadow_boundary_limit.imag))
        return wt.Complex2f(dr.select(near_shadow_boundary, blended_product.real, raw_product.real), dr.select(near_shadow_boundary, blended_product.imag, raw_product.imag))

    def a_pm(beta: wt.Float, n: wt.Float) -> tuple[wt.Float, wt.Float]:
        two_n_pi = 2 * n * dr.pi
        n_plus = dr.round((beta + dr.pi) * dr.rcp(two_n_pi))
        n_minus = dr.round((beta - dr.pi) * dr.rcp(two_n_pi))
        a_plus = 2 * dr.cos((two_n_pi * n_plus - beta) / 2) ** 2
        a_minus = 2 * dr.cos((two_n_pi * n_minus - beta) / 2) ** 2
        return (a_plus, a_minus)

    def coeff(phi: wt.Float, phi_prime: wt.Float, n: wt.Float, k: wt.Float, L: wt.Float, beta0: wt.Float=None, R0: wt.Complex2f=None, Rn: wt.Complex2f=None) -> wt.Complex2f:
        """UTD diffraction coefficient for a wedge."""
        sin_beta0 = wt.Float(1.0) if beta0 is None else dr.sin(beta0)
        factor = -dr.exp(wt.Complex2f(0, -dr.pi / 4))
        factor *= dr.rcp(2 * n * dr.safe_sqrt(dr.two_pi * k) * sin_beta0)
        dif_phi = phi - phi_prime
        sum_phi = phi + phi_prime
        kL = k * L
        d1, _, _ = UTDMath.beta_state(dif_phi, n, kL, +1.0, ABranch.PLUS)
        d2, _, _ = UTDMath.beta_state(dif_phi, n, kL, -1.0, ABranch.MINUS)
        d3, _, _ = UTDMath.beta_state(sum_phi, n, kL, +1.0, ABranch.PLUS)
        d4, _, _ = UTDMath.beta_state(sum_phi, n, kL, -1.0, ABranch.MINUS)
        R0 = wt.Complex2f(-1, 0) if R0 is None else R0
        Rn = wt.Complex2f(-1, 0) if Rn is None else Rn
        return factor * (d1 + d2 + Rn * d3 + R0 * d4)

    def poly_value_first_second(coeffs: tuple[float, ...], x: wt.Float) -> tuple[wt.Float, wt.Float, wt.Float]:
        value = wt.Float(coeffs[-1])
        first = dr.zeros(wt.Float, dr.width(x))
        second = dr.zeros(wt.Float, dr.width(x))
        for coeff in reversed(coeffs[:-1]):
            second = second * x + 2.0 * first
            first = first * x + value
            value = value * x + wt.Float(coeff)
        return (value, first, second)

    def safe_positive(x: wt.Float) -> wt.Float:
        return dr.maximum(x, wt.Float(SMALL_EPS))

    def fresnel_with_derivs(x: wt.Float) -> tuple[wt.Complex2f, wt.Complex2f, wt.Complex2f]:
        """Return the Fresnel approximation and its first/second x-derivatives."""
        a = (+1.59576914, -1.702e-06, -6.808568854, -0.000576361, +6.920691902, -0.016898657, -3.05048566, -0.075752419, +0.850663781, -0.025639041, -0.15023096, +0.034404779)
        b = (-3.3e-08, +4.255387524, -9.281e-05, -7.7800204, -0.009520895, +5.075161298, -0.138341947, -1.363729124, -0.403349276, +0.702222016, -0.216195929, +0.019547031)
        c = (+0.0, -0.024933975, +3.936e-06, +0.005770956, +0.000689892, -0.009497136, +0.011948809, -0.006748873, +0.00024642, +0.002102967, -0.00121793, +0.000233939)
        d = (+0.19947114, +2.3e-08, -0.009351341, +2.3006e-05, +0.004851466, +0.001903218, -0.017122914, +0.029064067, -0.027928955, +0.016497308, -0.005598515, +0.000838386)
        x_pos = x >= 0.0
        x_abs = dr.abs(x)
        safe_x = UTDMath.safe_positive(x_abs)
        cond = x_abs < 4.0
        arg_small = x_abs * 0.25
        arg_large = 4.0 * dr.rcp(safe_x)
        arg = dr.select(cond, arg_small, arg_large)
        arg_safe = UTDMath.safe_positive(arg)
        arg1_small = wt.Float(0.25)
        arg2_small = wt.Float(0.0)
        arg1_large = -4.0 * dr.rcp(safe_x * safe_x)
        arg2_large = 8.0 * dr.rcp(safe_x * safe_x * safe_x)
        arg1 = dr.select(cond, arg1_small, arg1_large)
        arg2 = dr.select(cond, arg2_small, arg2_large)
        r_small, r_small_1, r_small_2 = UTDMath.poly_value_first_second(a, arg_small)
        r_large, r_large_1, r_large_2 = UTDMath.poly_value_first_second(c, arg_large)
        i_small, i_small_1, i_small_2 = UTDMath.poly_value_first_second(b, arg_small)
        i_large, i_large_1, i_large_2 = UTDMath.poly_value_first_second(d, arg_large)
        r_coef = dr.select(cond, r_small, r_large)
        i_coef = dr.select(cond, i_small, i_large)
        r_coef_1 = dr.select(cond, r_small_1 * arg1_small, r_large_1 * arg1_large)
        i_coef_1 = dr.select(cond, i_small_1 * arg1_small, i_large_1 * arg1_large)
        r_coef_2 = dr.select(cond, r_small_2 * arg1_small * arg1_small, r_large_2 * arg1_large * arg1_large + r_large_1 * arg2_large)
        i_coef_2 = dr.select(cond, i_small_2 * arg1_small * arg1_small, i_large_2 * arg1_large * arg1_large + i_large_1 * arg2_large)
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
        value = wt.Complex2f(dr.select(x_pos, c_out, -c_out), dr.select(x_pos, s_out, -s_out))
        first = wt.Complex2f(c_out_1, s_out_1)
        second = wt.Complex2f(dr.select(x_pos, c_out_2, -c_out_2), dr.select(x_pos, s_out_2, -s_out_2))
        return (value, first, second)

    def f_utd_with_derivs(x: wt.Float) -> tuple[wt.Complex2f, wt.Complex2f, wt.Complex2f]:
        """Return the transition function and its first/second x-derivatives."""
        safe_x = UTDMath.safe_positive(x)
        fresnel_value, fresnel_first, fresnel_second = UTDMath.fresnel_with_derivs(x)
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
        first = prefactor_1 * phase * bracket + prefactor * phase_1 * bracket + prefactor * phase * bracket_1
        second = prefactor_2 * phase * bracket + 2.0 * prefactor_1 * phase_1 * bracket + 2.0 * prefactor_1 * phase * bracket_1 + prefactor * phase_2 * bracket + 2.0 * prefactor * phase_1 * bracket_1 + prefactor * phase * bracket_2
        return (value, first, second)

    def beta_state(beta: wt.Float, n: wt.Float, kL: wt.Float, cot_sign: float, a_branch: ABranch):
        cot_arg, a, a_1, cot_value, cot_1, cot_2, x, x_1, x_2 = UTDMath.beta_meta(beta, n, kL, cot_sign, a_branch)
        transition, transition_1, transition_2 = UTDMath.f_utd_with_derivs(x)
        forward_transition = f_utd(x)
        product_value = UTDMath.cot_f_product(cot_arg, a, kL, n=n, cot_sign=cot_sign, a_1=a_1, transition=forward_transition, cot_value=cot_value)
        return UTDMath.assemble_beta_state(product_value, cot_value, cot_1, cot_2, x_1, x_2, transition, transition_1, transition_2)

    def beta_meta(beta: wt.Float, n: wt.Float, kL: wt.Float, cot_sign: float, a_branch: ABranch):
        two_n = 2.0 * n
        two_n_pi = 2.0 * n * dr.pi
        if a_branch is ABranch.PLUS:
            round_index = dr.round((beta + dr.pi) * dr.rcp(two_n_pi))
        elif a_branch is ABranch.MINUS:
            round_index = dr.round((beta - dr.pi) * dr.rcp(two_n_pi))
        else:
            raise ValueError(f'Unsupported a_branch selector: {a_branch}')
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
        return (cot_arg, a, a_1, cot_value, cot_1, cot_2, x, x_1, x_2)

    def assemble_beta_state(product_value: wt.Complex2f, cot_value: wt.Float, cot_1: wt.Float, cot_2: wt.Float, x_1: wt.Float, x_2: wt.Float, transition: wt.Complex2f, transition_1: wt.Complex2f, transition_2: wt.Complex2f):
        value = product_value
        first = cot_1 * transition + cot_value * transition_1 * x_1
        second = cot_2 * transition + 2.0 * cot_1 * transition_1 * x_1 + cot_value * (transition_2 * dr.square(x_1) + transition_1 * x_2)
        return (value, first, second)

    def beta_groups(phi: wt.Float, phi_prime: wt.Float, n: wt.Float, k: wt.Float, s: wt.Float, s_prime: wt.Float, R0: wt.Complex2f, Rn: wt.Complex2f):
        L = s * s_prime * dr.rcp(s + s_prime)
        kL = k * L
        dif_phi = phi - phi_prime
        sum_phi = phi + phi_prime
        factor = -dr.exp(wt.Complex2f(0, -dr.pi / 4))
        factor *= dr.rcp(2.0 * n * dr.safe_sqrt(dr.two_pi * k))
        term_states = [UTDMath.beta_state(dif_phi, n, kL, +1.0, ABranch.PLUS), UTDMath.beta_state(dif_phi, n, kL, -1.0, ABranch.MINUS), UTDMath.beta_state(sum_phi, n, kL, +1.0, ABranch.PLUS), UTDMath.beta_state(sum_phi, n, kL, -1.0, ABranch.MINUS)]
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
        return (factor, dif_group, dif_group_1, dif_group_2, sum_group, sum_group_1, sum_group_2)

    def beta_groups3d(phi: wt.Float, phi_prime: wt.Float, n: wt.Float, k: wt.Float, s: wt.Float, s_prime: wt.Float, sin_beta0: wt.Float, R0: wt.Complex2f, Rn: wt.Complex2f):
        safe_sin_beta0 = dr.maximum(sin_beta0, wt.Float(SMALL_EPS))
        L = s * s_prime * dr.rcp(s + s_prime) * dr.square(safe_sin_beta0)
        kL = k * L
        dif_phi = phi - phi_prime
        sum_phi = phi + phi_prime
        factor = -dr.exp(wt.Complex2f(0, -dr.pi / 4))
        factor *= dr.rcp(2.0 * n * dr.safe_sqrt(dr.two_pi * k) * safe_sin_beta0)
        term_states = [UTDMath.beta_state(dif_phi, n, kL, +1.0, ABranch.PLUS), UTDMath.beta_state(dif_phi, n, kL, -1.0, ABranch.MINUS), UTDMath.beta_state(sum_phi, n, kL, +1.0, ABranch.PLUS), UTDMath.beta_state(sum_phi, n, kL, -1.0, ABranch.MINUS)]
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
        return (factor, dif_group, dif_group_1, dif_group_2, sum_group, sum_group_1, sum_group_2)

    def operator_terms2d(phi, phi_prime, wedge_n, k, s, s_prime):
        zero = wt.Complex2f(0.0, 0.0)
        one = wt.Complex2f(1.0, 0.0)
        factor, dif_group, dif_group_1, dif_group_2, _, _, _ = UTDMath.beta_groups(phi, phi_prime, wedge_n, wt.Float(k), s, s_prime, zero, zero)
        _, _, _, _, sum_plus, sum_plus_1, sum_plus_2 = UTDMath.beta_groups(phi, phi_prime, wedge_n, wt.Float(k), s, s_prime, one, zero)
        _, _, _, _, sum_minus, sum_minus_1, sum_minus_2 = UTDMath.beta_groups(phi, phi_prime, wedge_n, wt.Float(k), s, s_prime, zero, one)
        return {'direct': factor * dif_group, 'face0': factor * sum_plus, 'face1': factor * sum_minus, 'direct_dphi': factor * dif_group_1, 'face0_dphi': factor * sum_plus_1, 'face1_dphi': factor * sum_minus_1, 'direct_dphi_prime': factor * -dif_group_1, 'face0_dphi_prime': factor * sum_plus_1, 'face1_dphi_prime': factor * sum_minus_1, 'direct_d2phi_phi_prime': factor * -dif_group_2, 'face0_d2phi_phi_prime': factor * sum_plus_2, 'face1_d2phi_phi_prime': factor * sum_minus_2}

    def operator_terms3d(phi, phi_prime, wedge_n, k, s, s_prime, sin_beta0):
        zero = wt.Complex2f(0.0, 0.0)
        one = wt.Complex2f(1.0, 0.0)
        factor, dif_group, dif_group_1, dif_group_2, _, _, _ = UTDMath.beta_groups3d(phi, phi_prime, wedge_n, wt.Float(k), s, s_prime, sin_beta0, zero, zero)
        _, _, _, _, sum_plus, sum_plus_1, sum_plus_2 = UTDMath.beta_groups3d(phi, phi_prime, wedge_n, wt.Float(k), s, s_prime, sin_beta0, one, zero)
        _, _, _, _, sum_minus, sum_minus_1, sum_minus_2 = UTDMath.beta_groups3d(phi, phi_prime, wedge_n, wt.Float(k), s, s_prime, sin_beta0, zero, one)
        return {'direct': factor * dif_group, 'face0': factor * sum_plus, 'face1': factor * sum_minus, 'direct_dphi': factor * dif_group_1, 'face0_dphi': factor * sum_plus_1, 'face1_dphi': factor * sum_minus_1, 'direct_dphi_prime': factor * -dif_group_1, 'face0_dphi_prime': factor * sum_plus_1, 'face1_dphi_prime': factor * sum_minus_1, 'direct_d2phi_phi_prime': factor * -dif_group_2, 'face0_d2phi_phi_prime': factor * sum_plus_2, 'face1_d2phi_phi_prime': factor * sum_minus_2}

    def assemble_operator(free_term, face0_term, face1_term, face0_operator, face1_operator):
        width = dr.width(free_term.real)
        total = jones_operator_scale(jones_operator_identity(width), free_term)
        total = jones_operator_add(total, jones_operator_scale(face0_operator, face0_term))
        total = jones_operator_add(total, jones_operator_scale(face1_operator, face1_term))
        return total

    def assemble_material_ops(*, phi, phi_prime, wedge_n, k, s, s_prime, face0_operator, face1_operator, sin_beta0=None, include_normal_derivative_ops: bool=True):
        raw_terms = UTDMath.operator_terms3d(phi, phi_prime, wedge_n, k, s, s_prime, sin_beta0) if sin_beta0 is not None else UTDMath.operator_terms2d(phi, phi_prime, wedge_n, k, s, s_prime)
        operator_terms = {'direct': -raw_terms['direct'], 'face0': raw_terms['face0'], 'face1': raw_terms['face1'], 'direct_dphi': -raw_terms['direct_dphi'], 'face0_dphi': raw_terms['face0_dphi'], 'face1_dphi': raw_terms['face1_dphi'], 'direct_dphi_prime': -raw_terms['direct_dphi_prime'], 'face0_dphi_prime': raw_terms['face0_dphi_prime'], 'face1_dphi_prime': raw_terms['face1_dphi_prime'], 'direct_d2phi_phi_prime': -raw_terms['direct_d2phi_phi_prime'], 'face0_d2phi_phi_prime': raw_terms['face0_d2phi_phi_prime'], 'face1_d2phi_phi_prime': raw_terms['face1_d2phi_phi_prime']}
        slope_factor = wt.Complex2f(0.0, -1.0) * dr.rcp(wt.Float(k))
        result = {'field': UTDMath.assemble_operator(operator_terms['direct'], operator_terms['face0'], operator_terms['face1'], face0_operator, face1_operator), 'slope': UTDMath.assemble_operator(slope_factor * operator_terms['direct_dphi_prime'], slope_factor * operator_terms['face0_dphi_prime'], slope_factor * operator_terms['face1_dphi_prime'], face0_operator, face1_operator), 'terms': operator_terms}
        if include_normal_derivative_ops:
            result['field_dphi'] = UTDMath.assemble_operator(operator_terms['direct_dphi'], operator_terms['face0_dphi'], operator_terms['face1_dphi'], face0_operator, face1_operator)
            result['slope_dphi'] = UTDMath.assemble_operator(slope_factor * operator_terms['direct_d2phi_phi_prime'], slope_factor * operator_terms['face0_d2phi_phi_prime'], slope_factor * operator_terms['face1_d2phi_phi_prime'], face0_operator, face1_operator)
        return result


class GeometrySupport:

    def has_face_material_params(edge_state) -> bool:
        required = ('face0_eta_r', 'face0_sigma', 'face0_gain', 'face1_eta_r', 'face1_sigma', 'face1_gain')
        return all((key in edge_state for key in required))

    def face_operator(face_material, *, cos_theta, normal, incoming_hat, outgoing_hat, incoming_edge_basis, outgoing_edge_basis, wave: Wave):
        omega = material_angular_frequency(wave.wavelength)
        eta = complex_relative_permittivity(face_material['eta_r'], face_material['sigma'], omega)
        r_te, r_tm = fresnel_reflection(cos_theta, eta, mu_r=face_material['mu_r'])
        r_te = wt.Complex2f(dr.select(dr.isfinite(r_te.real), r_te.real, wt.Float(0.0)), dr.select(dr.isfinite(r_te.imag), r_te.imag, wt.Float(0.0)))
        r_tm = wt.Complex2f(dr.select(dr.isfinite(r_tm.real), r_tm.real, wt.Float(0.0)), dr.select(dr.isfinite(r_tm.imag), r_tm.imag, wt.Float(0.0)))
        gain = wt.Complex2f(face_material['gain'], wt.Float(0.0))
        diag_operator = jones_operator_diagonal(gain * r_te, gain * r_tm)
        face_s_in = dr.cross(normal, incoming_hat)
        face_s_out_raw = dr.cross(normal, outgoing_hat)
        face_out_fallback = stable_perpendicular_basis(outgoing_hat, preferred=face_s_in)
        face_s_out = dr.select(
            dr.dot(face_s_out_raw, face_out_fallback) < wt.Float(0.0),
            -face_s_out_raw,
            face_s_out_raw,
        )
        face_in_basis = basis_from_first_vector(incoming_hat, face_s_in)
        face_out_basis = basis_from_first_vector(
            outgoing_hat,
            face_s_out,
            fallback=face_out_fallback,
        )
        return jones_operator_in_basis(diag_operator, face_in_basis, face_out_basis, incoming_edge_basis, outgoing_edge_basis)

    def require_bounds(line_min, line_max, *, context: str, min_name: str, max_name: str):
        if line_min is None or line_max is None:
            raise RuntimeError(f'{context} requires finite-wedge {min_name} and {max_name}.')
        return (line_min, line_max)

    def data_line_bounds(edge_data, *, context: str):
        line_min = None if edge_data is None else edge_data.get('line_min')
        line_max = None if edge_data is None else edge_data.get('line_max')
        return Geo.require_bounds(line_min, line_max, context=context, min_name='line_min', max_name='line_max')

    def state_line_bounds(edge_state, *, context: str):
        edge_line_min = None if edge_state is None else edge_state.get('edge_line_min')
        edge_line_max = None if edge_state is None else edge_state.get('edge_line_max')
        return Geo.require_bounds(edge_line_min, edge_line_max, context=context, min_name='edge_line_min', max_name='edge_line_max')

    def first_order_diffraction_parameter(source_pos, target_pos, edge_origin, edge_dir):
        width = dr.width(target_pos.x)
        source_pos_b = broadcast_point(source_pos, width)
        edge_origin_b = broadcast_point(edge_origin, width)
        edge_dir_b = broadcast_vector(edge_dir, width)
        zeta = edge_dir_b / (dr.norm(edge_dir_b) + EPS)

        target_offset = target_pos - edge_origin_b
        source_offset = source_pos_b - edge_origin_b
        target_projection = dr.dot(target_offset, zeta) * zeta
        source_projection = dr.dot(source_offset, zeta) * zeta
        target_radial = target_offset - target_projection
        source_radial = source_offset - source_projection
        target_radial_norm = dr.norm(target_radial)
        source_radial_norm = dr.norm(source_radial)

        v1 = target_radial / dr.maximum(target_radial_norm, wt.Float(SMALL_EPS))
        v2 = source_radial / dr.maximum(source_radial_norm, wt.Float(SMALL_EPS))
        theta = dr.pi - dr.safe_acos(
            dr.clip(dr.dot(v1, v2), wt.Float(-1.0), wt.Float(1.0))
        )

        rotation_axis = dr.cross(source_radial, target_radial)
        rotation_axis_norm = dr.norm(rotation_axis)
        safe_rotation_axis = rotation_axis / dr.maximum(
            rotation_axis_norm,
            wt.Float(SMALL_EPS),
        )
        rotation_axis = dr.select(
            rotation_axis_norm > wt.Float(SMALL_EPS),
            safe_rotation_axis,
            zeta,
        )

        coplanar_target = rotate_vector_around_axis(
            target_offset,
            rotation_axis,
            theta,
        )
        source_to_target = coplanar_target - source_offset
        source_to_target_norm = dr.norm(source_to_target)
        u0 = source_to_target / dr.maximum(source_to_target_norm, wt.Float(SMALL_EPS))
        u1 = dr.cross(source_offset, u0)
        u2 = dr.cross(zeta, u0)
        u2_norm = dr.norm(u2)
        sign = dr.sign(dr.dot(u1, u2))
        return sign * dr.norm(u1) / dr.maximum(u2_norm, wt.Float(SMALL_EPS))

    def finite_edge_diffraction_point(edge_state, target_pos):
        width = dr.width(target_pos.x)
        edge_line_min, edge_line_max = Geo.state_line_bounds(
            edge_state,
            context='finite_edge_diffraction_point',
        )
        edge_pos_b = broadcast_point(edge_state['edge_pos'], width)
        source_pos_b = broadcast_point(edge_state['source_pos'], width)
        edge_dir_b = broadcast_vector(edge_state['edge_dir'], width)
        edge_hat = edge_dir_b / (dr.norm(edge_dir_b) + EPS)
        line_min = repeat_float(edge_line_min, width)
        line_max = repeat_float(edge_line_max, width)
        edge_origin = edge_pos_b + edge_hat * line_min
        edge_length = line_max - line_min
        diffraction_parameter = Geo.first_order_diffraction_parameter(
            source_pos_b,
            target_pos,
            edge_origin,
            edge_hat,
        )
        diffraction_point = edge_origin + edge_hat * diffraction_parameter
        clamped_parameter = dr.clip(
            diffraction_parameter,
            wt.Float(0.0),
            edge_length,
        )
        visibility_point = edge_origin + edge_hat * clamped_parameter
        finite_valid = (
            (edge_length > wt.Float(SMALL_EPS))
            & dr.isfinite(diffraction_parameter)
        )
        finite_inside = (
            finite_valid
            & (diffraction_parameter > wt.Float(0.0))
            & (diffraction_parameter < edge_length)
        )
        return {
            'point': diffraction_point,
            'visibility_point': visibility_point,
            'edge_line_min': -diffraction_parameter,
            'edge_line_max': edge_length - diffraction_parameter,
            'edge_origin': edge_origin,
            'edge_length': edge_length,
            'parameter': diffraction_parameter,
            'valid': finite_valid,
            'inside': finite_inside,
        }

    def source_field(source_pos, source_weight, target_pos, wave: Wave):
        width = dr.width(target_pos.x)
        source_pos_b = broadcast_point(source_pos, width)
        distance = dr.norm(target_pos - source_pos_b) + EPS
        phase = dr.exp(wt.Complex2f(0, -wave.k * distance))
        source_w = repeat_complex(source_weight, width)
        fspl = (wave.wavelength / wt.Float(4.0 * math.pi)) / distance
        return source_w * fspl * phase

    def source_field_normal_derivative(source_pos, source_weight, target_pos, normal_dir, wave: Wave):
        width = dr.width(target_pos.x)
        source_pos_b = broadcast_point(source_pos, width)
        normal_b = broadcast_vector(normal_dir, width)
        offset = target_pos - source_pos_b
        distance = dr.norm(offset) + EPS
        ray_hat = offset / distance
        projection = dr.dot(ray_hat, normal_b)
        field = Geo.source_field(source_pos, source_weight, target_pos, wave)
        scale = wt.Complex2f(-projection / distance, -projection * wave.k)
        return field * scale

    def face_material_inputs(edge_state, width, material: Material, *, scene=None):
        if 'face0_eta_r' in edge_state and 'face1_eta_r' in edge_state:
            face0_mu_r = edge_state.get('face0_mu_r', dr.ones(wt.Float, width))
            face1_mu_r = edge_state.get('face1_mu_r', dr.ones(wt.Float, width))
            return ({'eta_r': repeat_float(edge_state['face0_eta_r'], width), 'mu_r': repeat_float(face0_mu_r, width), 'sigma': repeat_float(edge_state['face0_sigma'], width), 'gain': repeat_float(edge_state['face0_gain'], width), 'use_fresnel': dr.full(wt.Bool, True, width)}, {'eta_r': repeat_float(edge_state['face1_eta_r'], width), 'mu_r': repeat_float(face1_mu_r, width), 'sigma': repeat_float(edge_state['face1_sigma'], width), 'gain': repeat_float(edge_state['face1_gain'], width), 'use_fresnel': dr.full(wt.Bool, True, width)})
        if scene is None:
            raise RuntimeError(
                "Diffraction edge material resolution requires a scene material table. "
                "Attach witwin.core.Material to every scene structure."
            )
        adjacent_face0 = repeat_int(edge_state['adjacent_face0'], width)
        adjacent_face1 = repeat_int(edge_state['adjacent_face1'], width)
        face0 = resolve_surface_material(scene=scene, prim_idx=adjacent_face0, default_gain=material.gain_scalar, valid_mask=adjacent_face0 >= 0)
        face1 = resolve_surface_material(scene=scene, prim_idx=adjacent_face1, default_gain=material.gain_scalar, valid_mask=adjacent_face1 >= 0)
        return (face0, face1)

    def surface_coeff(*, incident_dir, normal, scene, prim_idx, material: Material, wave: Wave, tx: Tx | None = None, valid_mask=None):
        material_inputs = resolve_surface_material(scene=scene, prim_idx=prim_idx, default_gain=material.gain_scalar, valid_mask=valid_mask)
        del tx
        incident_hat = incident_dir / dr.maximum(dr.norm(incident_dir), wt.Float(EPS))
        normal_hat = normal / dr.maximum(dr.norm(normal), wt.Float(EPS))
        cos_theta = dr.clip(dr.abs(dr.dot(incident_hat, normal_hat)), wt.Float(SMALL_EPS), wt.Float(1.0))
        fresnel_coeff = scalar_fresnel_reflection(cos_theta=cos_theta, eta_r=material_inputs['eta_r'], sigma=material_inputs['sigma'], omega=material_angular_frequency(wave.wavelength), mu_r=material_inputs['mu_r'], gain=material_inputs['gain'])
        return (fresnel_coeff, material_inputs)

    def diagonal_face_operator(coeff):
        width = dr.width(coeff.real)
        zero = wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))
        return {'m00': coeff, 'm01': zero, 'm10': zero, 'm11': coeff}

    def surface_operator(*, incident_dir, normal, scene, prim_idx, material: Material, wave: Wave, valid_mask=None):
        material_inputs = resolve_surface_material(scene=scene, prim_idx=prim_idx, default_gain=material.gain_scalar, valid_mask=valid_mask)
        incident_hat = incident_dir / dr.maximum(dr.norm(incident_dir), wt.Float(EPS))
        normal_hat = normal / dr.maximum(dr.norm(normal), wt.Float(EPS))
        cos_theta = dr.clip(dr.abs(dr.dot(incident_hat, normal_hat)), wt.Float(SMALL_EPS), wt.Float(1.0))
        eta = complex_relative_permittivity(material_inputs['eta_r'], material_inputs['sigma'], material_angular_frequency(wave.wavelength))
        r_te, r_tm = fresnel_reflection(cos_theta, eta, mu_r=material_inputs['mu_r'])
        r_te = wt.Complex2f(dr.select(dr.isfinite(r_te.real), r_te.real, wt.Float(0.0)), dr.select(dr.isfinite(r_te.imag), r_te.imag, wt.Float(0.0)))
        r_tm = wt.Complex2f(dr.select(dr.isfinite(r_tm.real), r_tm.real, wt.Float(0.0)), dr.select(dr.isfinite(r_tm.imag), r_tm.imag, wt.Float(0.0)))
        gain = wt.Complex2f(material_inputs['gain'], wt.Float(0.0))
        return jones_operator_diagonal(gain * r_te, gain * r_tm)

    def face_reflection_operators(edge_state, width, material: Material, wave: Wave, scene=None):
        if 'face0_operator_m00' in edge_state and 'face1_operator_m00' in edge_state:
            face0 = {'m00': repeat_complex(edge_state['face0_operator_m00'], width), 'm01': repeat_complex(edge_state['face0_operator_m01'], width), 'm10': repeat_complex(edge_state['face0_operator_m10'], width), 'm11': repeat_complex(edge_state['face0_operator_m11'], width)}
            face1 = {'m00': repeat_complex(edge_state['face1_operator_m00'], width), 'm01': repeat_complex(edge_state['face1_operator_m01'], width), 'm10': repeat_complex(edge_state['face1_operator_m10'], width), 'm11': repeat_complex(edge_state['face1_operator_m11'], width)}
            return (face0, face1)
        if scene is None:
            pec = wt.Complex2f(-1.0, 0.0)
            return (Geo.diagonal_face_operator(repeat_complex(pec, width)), Geo.diagonal_face_operator(repeat_complex(pec, width)))
        source_pos = broadcast_point(edge_state['source_pos'], width)
        edge_pos = broadcast_point(edge_state['edge_pos'], width)
        edge_dir = broadcast_vector(edge_state['edge_dir'], width)
        n0 = broadcast_vector(edge_state['n0'], width)
        nn = broadcast_vector(edge_state['n_face_n'], width)
        incoming = edge_pos - source_pos
        incoming = incoming / (dr.norm(incoming) + EPS)
        incoming_proj = normalize_in_wedge_plane(incoming, edge_dir)
        n0_proj = normalize_in_wedge_plane(n0, edge_dir)
        nn_proj = normalize_in_wedge_plane(nn, edge_dir)
        adjacent_face0 = repeat_int(edge_state['adjacent_face0'], width)
        adjacent_face1 = repeat_int(edge_state['adjacent_face1'], width)
        face0 = Geo.surface_operator(incident_dir=incoming_proj, normal=n0_proj, scene=scene, prim_idx=adjacent_face0, material=material, wave=wave, valid_mask=adjacent_face0 >= 0)
        face1 = Geo.surface_operator(incident_dir=incoming_proj, normal=nn_proj, scene=scene, prim_idx=adjacent_face1, material=material, wave=wave, valid_mask=adjacent_face1 >= 0)
        return (face0, face1)


Geo = GeometrySupport


__all__ = ['ABranch', 'DiffAngle', 'UTDMath', 'GeometrySupport', 'OPERATOR_KEYS']
