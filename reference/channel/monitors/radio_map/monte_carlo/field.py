from __future__ import annotations

import drjit as dr
import witwin as wt

from ....trace.diffraction.geometry import (
    _compute_edge_geometry,
    _cotangent_pole_safe_mask,
    _wedge_exterior_region_mask,
)
from ....trace.diffraction.utd import cot, f_utd
from ....trace.materials import reflection_material_omega
from ....utils.constants import DIFFRACTION_MIN_DISTANCE, EPS
from ....utils.drjit_ops import (
    ArrayInit,
    Broadcast,
    broadcast_point,
    broadcast_vector,
    complex_abs_sqr,
    repeat_complex,
    repeat_float,
)
from ....utils.material import complex_relative_permittivity, fresnel_reflection
from ....utils.polarization import (
    apply_jones_operator,
    basis_from_first_vector,
    implicit_basis_vector,
    jones_operator_add,
    jones_operator_diagonal,
    jones_operator_scale,
)


def _detach_mask_jones_operator(operator, mask):
    return {
        key: dr.select(mask, operator[key], dr.detach(operator[key]))
        for key in ("m00", "m01", "m10", "m11")
    }


def _normalize_real_vector(vec, fallback):
    vec_norm = dr.norm(vec)
    fallback_norm = dr.norm(fallback)
    safe_fallback = dr.select(
        fallback_norm > wt.Float(EPS),
        fallback / fallback_norm,
        wt.Vector3f(1.0, 0.0, 0.0),
    )
    return dr.select(
        vec_norm > wt.Float(EPS),
        vec / vec_norm,
        safe_fallback,
    )


def _jones_operator_matmul(lhs, rhs):
    return {
        "m00": lhs["m00"] * rhs["m00"] + lhs["m01"] * rhs["m10"],
        "m01": lhs["m00"] * rhs["m01"] + lhs["m01"] * rhs["m11"],
        "m10": lhs["m10"] * rhs["m00"] + lhs["m11"] * rhs["m10"],
        "m11": lhs["m10"] * rhs["m01"] + lhs["m11"] * rhs["m11"],
    }


def _jones_rotator_operator(k, s_current, s_target):
    c = dr.dot(s_current, s_target)
    s = dr.dot(k, dr.cross(s_current, s_target))
    zero = dr.zeros(wt.Float, dr.width(c))
    return {
        "m00": wt.Complex2f(c, zero),
        "m01": wt.Complex2f(s, zero),
        "m10": wt.Complex2f(-s, zero),
        "m11": wt.Complex2f(c, zero),
    }


def _sanitize_complex_coeff(coeff):
    return wt.Complex2f(
        dr.select(dr.isfinite(coeff.real), coeff.real, wt.Float(0.0)),
        dr.select(dr.isfinite(coeff.imag), coeff.imag, wt.Float(0.0)),
    )


def _sampled_face_reflection_diagonal_from_params(
    *,
    eta_r,
    sigma,
    gain,
    use_fresnel,
    cos_theta,
    wavelength,
):
    omega = reflection_material_omega(wavelength)
    eta = complex_relative_permittivity(
        eta_r,
        sigma,
        omega,
    )
    r_te, r_tm = fresnel_reflection(cos_theta, eta)
    gain_complex = wt.Complex2f(gain, wt.Float(0.0))
    coeff_te = _sanitize_complex_coeff(gain_complex * r_te)
    coeff_tm = _sanitize_complex_coeff(gain_complex * r_tm)
    zero = ArrayInit.complex_zero(dr.width(use_fresnel))
    coeff_te = dr.select(use_fresnel, coeff_te, zero)
    coeff_tm = dr.select(use_fresnel, coeff_tm, zero)
    return jones_operator_diagonal(coeff_te, coeff_tm)


def _override_mask(mask_override, default_mask, *, width):
    if mask_override is None:
        return default_mask
    return mask_override if dr.width(mask_override) == width else dr.repeat(mask_override, width)


def _override_value(value_override, default_value, *, width):
    if value_override is None:
        return default_value
    return value_override if dr.width(value_override) == width else dr.repeat(value_override, width)


def _sampled_edge_diffraction_power_to_targets_mc(
    *,
    source_pos,
    edge_dir,
    n0,
    nn,
    wedge_n,
    face0_eta_r,
    face0_sigma,
    face0_gain,
    face0_use_fresnel,
    face1_eta_r,
    face1_sigma,
    face1_gain,
    face1_use_fresnel,
    sampled_edge_pos,
    target_pos,
    k,
    wavelength,
    return_valid=False,
    return_support=False,
    support_override=None,
):
    width = dr.width(target_pos.x)
    edge_geometry = _compute_edge_geometry(
        source_pos,
        sampled_edge_pos,
        edge_dir,
        n0,
        target_pos,
    )
    phi = edge_geometry["phi"]
    phi_prime = edge_geometry["phi_prime"]
    s = edge_geometry["s"]
    s_prime = edge_geometry["s_prime"]
    sin_beta0 = edge_geometry["sin_beta_eff"]
    wedge_n_b = repeat_float(wedge_n, width)
    edge_dir_b = broadcast_vector(edge_dir, width)
    n0_b = broadcast_vector(n0, width)
    nn_b = broadcast_vector(nn, width)
    source_pos_b = broadcast_point(source_pos, width)
    edge_hat = _normalize_real_vector(edge_geometry["edge_hat"], wt.Vector3f(0.0, 0.0, 1.0))
    source_exterior = _wedge_exterior_region_mask(
        source_pos_b - sampled_edge_pos,
        edge_dir_b,
        n0_b,
        nn_b,
    )
    field_valid = (
        source_exterior
        & (s_prime > DIFFRACTION_MIN_DISTANCE)
        & (s > DIFFRACTION_MIN_DISTANCE)
    )
    incident_ray_dir = sampled_edge_pos - source_pos_b
    outgoing_ray_dir = target_pos - sampled_edge_pos
    incident_hat = _normalize_real_vector(incident_ray_dir, wt.Vector3f(0.0, 0.0, -1.0))
    outgoing_hat = _normalize_real_vector(outgoing_ray_dir, wt.Vector3f(0.0, 0.0, 1.0))
    incident_basis = basis_from_first_vector(incident_hat, implicit_basis_vector(incident_hat))
    outgoing_basis = basis_from_first_vector(outgoing_hat, implicit_basis_vector(outgoing_hat))
    incident_jones = {
        "u": repeat_complex(wt.Complex2f(1.0, 0.0), width),
        "v": ArrayInit.complex_zero(width),
    }
    pole_safe = field_valid & _cotangent_pole_safe_mask(
        phi,
        phi_prime,
        wedge_n_b,
        wt.Float(1.0e-6),
    )
    if support_override is not None:
        field_valid = _override_mask(support_override.get("field_valid"), field_valid, width=width)
        pole_safe = _override_mask(support_override.get("pole_safe"), pole_safe, width=width)
    phi_eval = dr.select(pole_safe, phi, 0.5 * wedge_n_b * dr.pi)
    phi_prime_eval = dr.select(pole_safe, phi_prime, 0.5 * wedge_n_b * dr.pi)
    exterior_angle = wedge_n_b * dr.pi
    l = s * s_prime * dr.rcp(s + s_prime + wt.Float(EPS)) * dr.square(sin_beta0)
    n = wedge_n_b
    dif_phi = phi_eval - phi_prime_eval
    sum_phi = phi_eval + phi_prime_eval

    def _a_p_m(beta, *, n_p_override_key, n_m_override_key):
        n_p = dr.round((beta + dr.pi) * dr.rcp(2.0 * exterior_angle))
        n_m = dr.round((beta - dr.pi) * dr.rcp(2.0 * exterior_angle))
        if support_override is not None:
            n_p = _override_value(support_override.get(n_p_override_key), n_p, width=width)
            n_m = _override_value(support_override.get(n_m_override_key), n_m, width=width)
        a_p = 2.0 * dr.square(dr.cos(exterior_angle * n_p - beta * 0.5))
        a_m = 2.0 * dr.square(dr.cos(exterior_angle * n_m - beta * 0.5))
        return a_p, a_m, n_p, n_m

    a1, a2, dif_n_p, dif_n_m = _a_p_m(
        dif_phi,
        n_p_override_key="dif_n_p",
        n_m_override_key="dif_n_m",
    )
    a3, a4, sum_n_p, sum_n_m = _a_p_m(
        sum_phi,
        n_p_override_key="sum_n_p",
        n_m_override_key="sum_n_m",
    )
    factor = -dr.exp(wt.Complex2f(0.0, -0.25 * dr.pi))
    factor *= dr.rcp(
        2.0
        * n
        * dr.safe_sqrt(dr.two_pi * wt.Float(k))
        * dr.maximum(sin_beta0, wt.Float(EPS))
    )
    d1 = cot((dr.pi + dif_phi) * dr.rcp(2.0 * n)) * factor * f_utd(wt.Float(k) * l * a1)
    d2 = cot((dr.pi - dif_phi) * dr.rcp(2.0 * n)) * factor * f_utd(wt.Float(k) * l * a2)
    d3 = cot((dr.pi + sum_phi) * dr.rcp(2.0 * n)) * factor * f_utd(wt.Float(k) * l * a3)
    d4 = cot((dr.pi - sum_phi) * dr.rcp(2.0 * n)) * factor * f_utd(wt.Float(k) * l * a4)
    d12 = -(d1 + d2)

    phi_hat_prime = _normalize_real_vector(dr.cross(incident_hat, edge_hat), incident_basis["u"])
    phi_hat = -_normalize_real_vector(dr.cross(outgoing_hat, edge_hat), outgoing_basis["u"])
    e_i_s_0_hat = _normalize_real_vector(dr.cross(incident_hat, n0_b), phi_hat_prime)
    e_i_s_n_hat = _normalize_real_vector(dr.cross(incident_hat, nn_b), phi_hat_prime)
    w_in = _jones_rotator_operator(incident_hat, incident_basis["u"], phi_hat_prime)
    w_out = _jones_rotator_operator(outgoing_hat, phi_hat, outgoing_basis["u"])
    w_0_in = _jones_rotator_operator(incident_hat, phi_hat_prime, e_i_s_0_hat)
    w_0_out = _jones_rotator_operator(outgoing_hat, e_i_s_0_hat, phi_hat)
    w_n_in = _jones_rotator_operator(incident_hat, phi_hat_prime, e_i_s_n_hat)
    w_n_out = _jones_rotator_operator(outgoing_hat, e_i_s_n_hat, phi_hat)

    cos_theta0 = dr.clip(dr.abs(dr.sin(phi_prime_eval)), wt.Float(1.0e-6), wt.Float(1.0))
    cos_theta1 = dr.clip(
        dr.abs(dr.sin(exterior_angle - phi_eval)),
        wt.Float(1.0e-6),
        wt.Float(1.0),
    )
    face0_diag = _sampled_face_reflection_diagonal_from_params(
        eta_r=repeat_float(face0_eta_r, width),
        sigma=repeat_float(face0_sigma, width),
        gain=repeat_float(face0_gain, width),
        use_fresnel=repeat_float(face0_use_fresnel, width),
        cos_theta=cos_theta0,
        wavelength=wavelength,
    )
    face1_diag = _sampled_face_reflection_diagonal_from_params(
        eta_r=repeat_float(face1_eta_r, width),
        sigma=repeat_float(face1_sigma, width),
        gain=repeat_float(face1_gain, width),
        use_fresnel=repeat_float(face1_use_fresnel, width),
        cos_theta=cos_theta1,
        wavelength=wavelength,
    )
    direct_operator = jones_operator_diagonal(d12, d12)
    face0_operator = _jones_operator_matmul(
        w_0_out,
        _jones_operator_matmul(
            jones_operator_scale(face0_diag, d4),
            w_0_in,
        ),
    )
    face1_operator = _jones_operator_matmul(
        w_n_out,
        _jones_operator_matmul(
            jones_operator_scale(face1_diag, d3),
            w_n_in,
        ),
    )
    total_operator = jones_operator_add(
        direct_operator,
        jones_operator_add(face0_operator, face1_operator),
    )
    total_operator = _jones_operator_matmul(
        w_out,
        _jones_operator_matmul(total_operator, w_in),
    )
    total_operator = _detach_mask_jones_operator(total_operator, pole_safe)
    field_jones = apply_jones_operator(incident_jones, total_operator)
    local_scale = dr.rsqrt(s * s_prime * (s + s_prime) + EPS)
    scaled_field_u = field_jones["u"] * wt.Complex2f(local_scale, wt.Float(0.0))
    scaled_field_v = field_jones["v"] * wt.Complex2f(local_scale, wt.Float(0.0))
    field_power = dr.select(
        field_valid,
        complex_abs_sqr(scaled_field_u) + complex_abs_sqr(scaled_field_v),
        wt.Float(0.0),
    )
    support = {
        "field_valid": field_valid,
        "pole_safe": pole_safe,
        "dif_n_p": dif_n_p,
        "dif_n_m": dif_n_m,
        "sum_n_p": sum_n_p,
        "sum_n_m": sum_n_m,
    }
    if return_valid and return_support:
        return field_power, field_valid, support
    if return_valid:
        return field_power, field_valid
    if return_support:
        return field_power, support
    return field_power


__all__ = ["_sampled_edge_diffraction_power_to_targets_mc"]
