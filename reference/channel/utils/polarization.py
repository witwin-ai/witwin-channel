from __future__ import annotations

"""Polarization and Jones/vector-field transport helpers."""

import drjit as dr
import witwin as wt

from .constants import EPS, SMALL_EPS
from .material import complex_relative_permittivity, fresnel_reflection
from .plane_axes import normalize_axis, tangential_axes_for_axis


_AXIS_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def complex_zero(width: int):
    return wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))


def vector_zero(width: int):
    return {
        # These must be distinct Complex2f buffers. Some accumulation paths
        # update the components in-place via scatter-reduce.
        "x": complex_zero(width),
        "y": complex_zero(width),
        "z": complex_zero(width),
    }


def jones_zero(width: int):
    return {
        "u": complex_zero(width),
        "v": complex_zero(width),
    }


def vector_add(lhs, rhs):
    return {
        "x": lhs["x"] + rhs["x"],
        "y": lhs["y"] + rhs["y"],
        "z": lhs["z"] + rhs["z"],
    }


def vector_scale(vec, coeff):
    return {
        "x": vec["x"] * coeff,
        "y": vec["y"] * coeff,
        "z": vec["z"] * coeff,
    }


def jones_add(lhs, rhs):
    return {
        "u": lhs["u"] + rhs["u"],
        "v": lhs["v"] + rhs["v"],
    }


def jones_scale(jones, coeff):
    return {
        "u": jones["u"] * coeff,
        "v": jones["v"] * coeff,
    }


def jones_operator_identity(width: int):
    one = wt.Complex2f(dr.ones(wt.Float, width), dr.zeros(wt.Float, width))
    zero = complex_zero(width)
    return {
        "m00": one,
        "m01": zero,
        "m10": zero,
        "m11": one,
    }


def jones_operator_diagonal(c0, c1=None):
    if c1 is None:
        c1 = c0
    zero = complex_zero(dr.width(c0.real))
    return {
        "m00": c0,
        "m01": zero,
        "m10": zero,
        "m11": c1,
    }


def jones_operator_add(lhs, rhs):
    return {
        "m00": lhs["m00"] + rhs["m00"],
        "m01": lhs["m01"] + rhs["m01"],
        "m10": lhs["m10"] + rhs["m10"],
        "m11": lhs["m11"] + rhs["m11"],
    }


def jones_operator_scale(operator, coeff):
    return {
        "m00": operator["m00"] * coeff,
        "m01": operator["m01"] * coeff,
        "m10": operator["m10"] * coeff,
        "m11": operator["m11"] * coeff,
    }


def vector_select(mask, true_value, false_value):
    return {
        "x": dr.select(mask, true_value["x"], false_value["x"]),
        "y": dr.select(mask, true_value["y"], false_value["y"]),
        "z": dr.select(mask, true_value["z"], false_value["z"]),
    }


def jones_select(mask, true_value, false_value):
    return {
        "u": dr.select(mask, true_value["u"], false_value["u"]),
        "v": dr.select(mask, true_value["v"], false_value["v"]),
    }


def vector_eval(vec):
    dr.eval(
        vec["x"].real, vec["x"].imag,
        vec["y"].real, vec["y"].imag,
        vec["z"].real, vec["z"].imag,
    )
    return vec


def complex_dot_real(vec, basis):
    return vec["x"] * basis.x + vec["y"] * basis.y + vec["z"] * basis.z


def complex_scale_real(basis, coeff):
    return {
        "x": coeff * basis.x,
        "y": coeff * basis.y,
        "z": coeff * basis.z,
    }


def xy_jones(vec):
    return tangential_jones(vec, axis="z")


def tangential_jones(vec, axis="z"):
    axis_name = normalize_axis(axis)
    tangential_axes = tangential_axes_for_axis(axis_name)
    return {
        tangential_axes[0]: vec[tangential_axes[0]],
        tangential_axes[1]: vec[tangential_axes[1]],
    }


def xy_receiver_polarization(polarization):
    tangential = tangential_receiver_polarization(polarization, axis="z")
    return tangential["x"], tangential["y"]


def tangential_receiver_polarization(polarization, axis="z"):
    axis_name = normalize_axis(axis)
    tangential_axes = tangential_axes_for_axis(axis_name)
    axis_0 = tangential_axes[0]
    axis_1 = tangential_axes[1]
    p0 = wt.Float(float(polarization[_AXIS_INDEX[axis_0]]))
    p1 = wt.Float(float(polarization[_AXIS_INDEX[axis_1]]))
    norm = dr.sqrt(p0 * p0 + p1 * p1)
    safe_norm = dr.select(norm > wt.Float(SMALL_EPS), norm, wt.Float(1.0))
    return {
        axis_0: p0 / safe_norm,
        axis_1: p1 / safe_norm,
    }


def scalarize_tangential_jones(jones, polarization, axis="z"):
    axis_name = normalize_axis(axis)
    tangential_axes = tangential_axes_for_axis(axis_name)
    rx_pol = tangential_receiver_polarization(polarization, axis=axis_name)
    axis_0 = tangential_axes[0]
    axis_1 = tangential_axes[1]
    return jones[axis_0] * rx_pol[axis_0] + jones[axis_1] * rx_pol[axis_1]


def scalarize_vector_to_tangential_polarization(field_vec, polarization, axis="z"):
    return scalarize_tangential_jones(
        tangential_jones(field_vec, axis=axis),
        polarization,
        axis=axis,
    )


def _safe_normalize(vec, fallback):
    norm = dr.norm(vec)
    fallback_norm = dr.norm(fallback)
    fallback_unit = fallback / (fallback_norm + EPS)
    return dr.select(norm > wt.Float(SMALL_EPS), vec / (norm + EPS), fallback_unit)


def stable_perpendicular_basis(ray_dir, preferred=None):
    if preferred is None:
        preferred = wt.Vector3f(0.0, 0.0, 1.0)
    proj = preferred - dr.dot(preferred, ray_dir) * ray_dir
    alt_axis = dr.select(dr.abs(ray_dir.z) < 0.9, wt.Vector3f(0.0, 0.0, 1.0), wt.Vector3f(0.0, 1.0, 0.0))
    alt_proj = alt_axis - dr.dot(alt_axis, ray_dir) * ray_dir
    return _safe_normalize(proj, alt_proj)


def path_basis(ray_dir, preferred=None):
    ray_hat = ray_dir / (dr.norm(ray_dir) + EPS)
    u_hat = stable_perpendicular_basis(ray_hat, preferred=preferred)
    v_hat = _safe_normalize(
        dr.cross(ray_hat, u_hat),
        stable_perpendicular_basis(ray_hat, preferred=wt.Vector3f(0.0, 1.0, 0.0)),
    )
    return {
        "u": u_hat,
        "v": v_hat,
        "k": ray_hat,
    }


def basis_from_first_vector(ray_dir, first_vector, fallback=None):
    ray_hat = ray_dir / (dr.norm(ray_dir) + EPS)
    if fallback is None:
        fallback = stable_perpendicular_basis(ray_hat)
    u_hat = _safe_normalize(
        first_vector - dr.dot(first_vector, ray_hat) * ray_hat,
        fallback,
    )
    v_hat = _safe_normalize(
        dr.cross(ray_hat, u_hat),
        stable_perpendicular_basis(ray_hat, preferred=wt.Vector3f(0.0, 1.0, 0.0)),
    )
    return {
        "u": u_hat,
        "v": v_hat,
        "k": ray_hat,
    }


def implicit_basis_vector(ray_dir):
    ray_hat = _safe_normalize(ray_dir, wt.Vector3f(0.0, 0.0, 1.0))
    radial = dr.sqrt(ray_hat.x * ray_hat.x + ray_hat.y * ray_hat.y)
    fallback = wt.Vector3f(
        dr.select(ray_hat.z >= 0.0, wt.Float(1.0), wt.Float(-1.0)),
        wt.Float(0.0),
        wt.Float(0.0),
    )
    theta_hat = wt.Vector3f(
        dr.select(
            radial > wt.Float(EPS),
            ray_hat.z * ray_hat.x / radial,
            fallback.x,
        ),
        dr.select(
            radial > wt.Float(EPS),
            ray_hat.z * ray_hat.y / radial,
            fallback.y,
        ),
        dr.select(radial > wt.Float(EPS), -radial, fallback.z),
    )
    return _safe_normalize(theta_hat, fallback)


def diffraction_edge_basis(ray_dir, edge_dir, *, outgoing=False):
    ray_hat = ray_dir / (dr.norm(ray_dir) + EPS)
    edge_hat = edge_dir / (dr.norm(edge_dir) + EPS)
    phi_hat = dr.cross(ray_hat, edge_hat)
    if outgoing:
        phi_hat = -phi_hat
    fallback = stable_perpendicular_basis(ray_hat, preferred=edge_hat)
    return basis_from_first_vector(ray_hat, phi_hat, fallback=fallback)


def project_real_polarization_to_ray(polarization, ray_dir):
    pol = wt.Vector3f(float(polarization[0]), float(polarization[1]), float(polarization[2]))
    return stable_perpendicular_basis(ray_dir, preferred=pol)


def project_real_polarization_to_path_basis(polarization, ray_dir):
    preferred = wt.Vector3f(float(polarization[0]), float(polarization[1]), float(polarization[2]))
    return path_basis(ray_dir, preferred=preferred)


def vector_from_scalar_and_real_direction(coeff, direction):
    return {
        "x": coeff * direction.x,
        "y": coeff * direction.y,
        "z": coeff * direction.z,
    }


def vector_from_jones(jones, basis):
    return vector_add(
        complex_scale_real(basis["u"], jones["u"]),
        complex_scale_real(basis["v"], jones["v"]),
    )


def jones_from_vector(field_vec, basis):
    return {
        "u": complex_dot_real(field_vec, basis["u"]),
        "v": complex_dot_real(field_vec, basis["v"]),
    }


def rotate_jones(jones, src_basis, dst_basis):
    return jones_from_vector(vector_from_jones(jones, src_basis), dst_basis)


def reflect_direction(incident_dir, normal):
    incident_hat = incident_dir / (dr.norm(incident_dir) + EPS)
    normal_hat = normal / (dr.norm(normal) + EPS)
    return incident_hat - 2.0 * dr.dot(incident_hat, normal_hat) * normal_hat


def scalarize_vector_to_polarization(field_vec, ray_dir, polarization):
    basis = project_real_polarization_to_ray(polarization, ray_dir)
    return complex_dot_real(field_vec, basis)


def scalarize_xy_jones(jones, polarization):
    return scalarize_tangential_jones(jones, polarization, axis="z")


def scalarize_vector_to_xy_polarization(field_vec, polarization):
    return scalarize_vector_to_tangential_polarization(field_vec, polarization, axis="z")


def effective_rx_polarization(rx_polarization, tx_polarization):
    return tx_polarization if rx_polarization is None else rx_polarization


def _sanitize_complex(coeff):
    return wt.Complex2f(
        dr.select(dr.isfinite(coeff.real), coeff.real, wt.Float(0.0)),
        dr.select(dr.isfinite(coeff.imag), coeff.imag, wt.Float(0.0)),
    )


def reflect_field_vector(field_vec, incident_dir, normal, eta_r, sigma, omega, gain=1.0):
    incident_hat = incident_dir / (dr.norm(incident_dir) + EPS)
    normal_hat = normal / (dr.norm(normal) + EPS)
    reflected_dir = reflect_direction(incident_hat, normal_hat)

    s_pref = dr.cross(normal_hat, incident_hat)
    s_hat = _safe_normalize(s_pref, stable_perpendicular_basis(incident_hat))
    p_in_hat = _safe_normalize(dr.cross(s_hat, incident_hat), stable_perpendicular_basis(incident_hat, preferred=normal_hat))
    p_out_hat = _safe_normalize(dr.cross(s_hat, reflected_dir), stable_perpendicular_basis(reflected_dir, preferred=normal_hat))

    cos_theta = dr.clip(dr.abs(dr.dot(incident_hat, normal_hat)), wt.Float(SMALL_EPS), wt.Float(1.0))
    eta = complex_relative_permittivity(wt.Float(eta_r), wt.Float(sigma), wt.Float(omega))
    r_te, r_tm = fresnel_reflection(cos_theta, eta)
    r_te = wt.Complex2f(
        dr.select(dr.isfinite(r_te.real), r_te.real, wt.Float(0.0)),
        dr.select(dr.isfinite(r_te.imag), r_te.imag, wt.Float(0.0)),
    )
    r_tm = wt.Complex2f(
        dr.select(dr.isfinite(r_tm.real), r_tm.real, wt.Float(0.0)),
        dr.select(dr.isfinite(r_tm.imag), r_tm.imag, wt.Float(0.0)),
    )
    gain_c = wt.Complex2f(wt.Float(gain), 0.0)

    e_s = complex_dot_real(field_vec, s_hat)
    e_p = complex_dot_real(field_vec, p_in_hat)
    return vector_add(
        complex_scale_real(s_hat, gain_c * r_te * e_s),
        complex_scale_real(p_out_hat, gain_c * r_tm * e_p),
    )


def reflection_jones_operator(
    incident_dir,
    normal,
    eta_r,
    sigma,
    omega,
    *,
    gain=1.0,
    incoming_basis=None,
    outgoing_basis=None,
):
    incident_hat = incident_dir / (dr.norm(incident_dir) + EPS)
    reflected_dir = reflect_direction(incident_hat, normal)
    if incoming_basis is None:
        incoming_basis = path_basis(incident_hat, preferred=normal)
    if outgoing_basis is None:
        outgoing_basis = path_basis(reflected_dir, preferred=normal)

    u_field = vector_from_scalar_and_real_direction(wt.Complex2f(1.0, 0.0), incoming_basis["u"])
    v_field = vector_from_scalar_and_real_direction(wt.Complex2f(1.0, 0.0), incoming_basis["v"])
    u_reflected = reflect_field_vector(u_field, incident_hat, normal, eta_r, sigma, omega, gain=gain)
    v_reflected = reflect_field_vector(v_field, incident_hat, normal, eta_r, sigma, omega, gain=gain)
    u_out = jones_from_vector(u_reflected, outgoing_basis)
    v_out = jones_from_vector(v_reflected, outgoing_basis)
    return {
        "m00": u_out["u"],
        "m01": v_out["u"],
        "m10": u_out["v"],
        "m11": v_out["v"],
        "basis_in": incoming_basis,
        "basis_out": outgoing_basis,
    }


def apply_jones_operator(jones, operator):
    return {
        "u": operator["m00"] * jones["u"] + operator["m01"] * jones["v"],
        "v": operator["m10"] * jones["u"] + operator["m11"] * jones["v"],
    }


def jones_operator_in_basis(operator, src_in_basis, src_out_basis, dst_in_basis, dst_out_basis):
    unit_u = {
        "u": wt.Complex2f(1.0, 0.0),
        "v": wt.Complex2f(0.0, 0.0),
    }
    unit_v = {
        "u": wt.Complex2f(0.0, 0.0),
        "v": wt.Complex2f(1.0, 0.0),
    }

    def _map(column_jones):
        field = vector_from_jones(column_jones, dst_in_basis)
        src_jones = jones_from_vector(field, src_in_basis)
        src_out_jones = apply_jones_operator(src_jones, operator)
        out_field = vector_from_jones(src_out_jones, src_out_basis)
        return jones_from_vector(out_field, dst_out_basis)

    mapped_u = _map(unit_u)
    mapped_v = _map(unit_v)
    return {
        "m00": mapped_u["u"],
        "m01": mapped_v["u"],
        "m10": mapped_u["v"],
        "m11": mapped_v["v"],
    }


def polarization_consistent_scalar_reflection(
    incident_dir,
    normal,
    eta_r,
    sigma,
    omega,
    *,
    gain=1.0,
    polarization=(1.0, 0.0, 0.0),
):
    incident_hat = incident_dir / (dr.norm(incident_dir) + EPS)
    normal_hat = normal / (dr.norm(normal) + EPS)
    incident_basis = project_real_polarization_to_ray(polarization, incident_hat)
    incident_field = vector_from_scalar_and_real_direction(wt.Complex2f(1.0, 0.0), incident_basis)
    reflected_field = reflect_field_vector(
        incident_field,
        incident_hat,
        normal_hat,
        eta_r=eta_r,
        sigma=sigma,
        omega=omega,
        gain=gain,
    )
    reflected_dir = reflect_direction(incident_hat, normal_hat)
    return _sanitize_complex(
        scalarize_vector_to_polarization(reflected_field, reflected_dir, polarization)
    )


def reflect_field_vector_scalar(field_vec, incident_dir, normal, gain=1.0):
    incident_hat = incident_dir / (dr.norm(incident_dir) + EPS)
    normal_hat = normal / (dr.norm(normal) + EPS)
    reflected_dir = reflect_direction(incident_hat, normal_hat)

    s_pref = dr.cross(normal_hat, incident_hat)
    s_hat = _safe_normalize(s_pref, stable_perpendicular_basis(incident_hat))
    p_in_hat = _safe_normalize(
        dr.cross(s_hat, incident_hat),
        stable_perpendicular_basis(incident_hat, preferred=normal_hat),
    )
    p_out_hat = _safe_normalize(
        dr.cross(s_hat, reflected_dir),
        stable_perpendicular_basis(reflected_dir, preferred=normal_hat),
    )

    coeff = wt.Complex2f(wt.Float(-gain), 0.0)
    e_s = complex_dot_real(field_vec, s_hat)
    e_p = complex_dot_real(field_vec, p_in_hat)
    return vector_add(
        complex_scale_real(s_hat, coeff * e_s),
        complex_scale_real(p_out_hat, coeff * e_p),
    )


def transport_diffraction_vector(
    incident_vector,
    incident_derivative_vector,
    source_pos,
    edge_pos,
    edge_dir,
    target_pos,
    direct_gain,
    derivative_gain,
):
    width = dr.width(target_pos.x)
    edge_pos_b = edge_pos if dr.width(edge_pos.x) == width else wt.Point3f(
        dr.repeat(edge_pos.x, width),
        dr.repeat(edge_pos.y, width),
        dr.repeat(edge_pos.z, width),
    )
    edge_dir_b = edge_dir if dr.width(edge_dir.x) == width else wt.Vector3f(
        dr.repeat(edge_dir.x, width),
        dr.repeat(edge_dir.y, width),
        dr.repeat(edge_dir.z, width),
    )
    source_pos_b = source_pos if dr.width(source_pos.x) == width else wt.Point3f(
        dr.repeat(source_pos.x, width),
        dr.repeat(source_pos.y, width),
        dr.repeat(source_pos.z, width),
    )
    edge_hat = edge_dir_b / (dr.norm(edge_dir_b) + EPS)
    incoming_hat = (edge_pos_b - source_pos_b) / (dr.norm(edge_pos_b - source_pos_b) + EPS)
    outgoing_hat = (target_pos - edge_pos_b) / (dr.norm(target_pos - edge_pos_b) + EPS)

    phi_in_hat = _safe_normalize(dr.cross(incoming_hat, edge_hat), stable_perpendicular_basis(incoming_hat, preferred=edge_hat))
    phi_out_hat = _safe_normalize(-dr.cross(outgoing_hat, edge_hat), stable_perpendicular_basis(outgoing_hat, preferred=edge_hat))

    inc_phi = complex_dot_real(incident_vector, phi_in_hat)
    inc_edge = complex_dot_real(incident_vector, edge_hat)
    out_direct = vector_scale(
        vector_add(
            complex_scale_real(phi_out_hat, inc_phi),
            complex_scale_real(edge_hat, inc_edge),
        ),
        direct_gain,
    )

    der_phi = complex_dot_real(incident_derivative_vector, phi_in_hat)
    der_edge = complex_dot_real(incident_derivative_vector, edge_hat)
    out_derivative = vector_scale(
        vector_add(
            complex_scale_real(phi_out_hat, der_phi),
            complex_scale_real(edge_hat, der_edge),
        ),
        derivative_gain,
    )
    return vector_add(out_direct, out_derivative)

