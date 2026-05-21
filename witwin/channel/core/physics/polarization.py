"""Polarization, Jones / vector-field transport, and reflection helpers.

Pure DrJit math; no scene/solver state.
"""

from __future__ import annotations

import drjit as dr
from witwin.channel import types as wt

from witwin.channel.core.numerics.arrays import complex_zero
from witwin.channel.core.numerics.constants import EPS, SMALL_EPS
from witwin.channel.core.geometry import normalize_axis, tangential_axes_for_axis
from .wave_math import complex_relative_permittivity, fresnel_reflection, material_angular_frequency


_XYZ = ("x", "y", "z")
_UV = ("u", "v")
_MAT = ("m00", "m01", "m10", "m11")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _polarization_vector(polarization):
    if all(hasattr(polarization, axis) for axis in _XYZ):
        return wt.Vector3f(polarization.x, polarization.y, polarization.z)
    return wt.Vector3f(polarization[0], polarization[1], polarization[2])


def _polarization_component(polarization, axis: str):
    if all(hasattr(polarization, a) for a in _XYZ):
        return getattr(polarization, axis)
    return polarization[_AXIS_INDEX[axis]]


# ---------------------------------------------------------------------------
# Vector primitives (dicts keyed by x/y/z with Complex2f components)
# ---------------------------------------------------------------------------


def vector_zero(width: int):
    # Distinct buffers â€?accumulation paths update in-place.
    return {axis: complex_zero(width) for axis in _XYZ}


def vector_add(lhs, rhs):
    return {axis: lhs[axis] + rhs[axis] for axis in _XYZ}


def vector_scale(vec, coeff):
    return {axis: vec[axis] * coeff for axis in _XYZ}


def vector_select(mask, true_value, false_value):
    return {axis: dr.select(mask, true_value[axis], false_value[axis]) for axis in _XYZ}


def vector_eval(vec) -> dict:
    dr.eval(*(component for axis in _XYZ for component in (vec[axis].real, vec[axis].imag)))
    return vec


def vector_from_scalar(coeff, direction):
    return {axis: coeff * getattr(direction, axis) for axis in _XYZ}


def vector_power(vec) -> wt.Float:
    cx, cy, cz = vec["x"], vec["y"], vec["z"]
    return (cx.real * cx.real + cx.imag * cx.imag
            + cy.real * cy.real + cy.imag * cy.imag
            + cz.real * cz.real + cz.imag * cz.imag)


def complex_dot_real(vec, basis) -> wt.Complex2f:
    return vec["x"] * basis.x + vec["y"] * basis.y + vec["z"] * basis.z


def complex_scale_real(basis, coeff):
    return {axis: coeff * getattr(basis, axis) for axis in _XYZ}


def safe_normalize_with_fallback(vec, fallback):
    """Normalize ``vec``; if its norm is below ``SMALL_EPS`` return the (also
    normalized) ``fallback`` vector instead.

    Differs from :func:`witwin.channel.core.numerics.arrays.safe_normalize`, which
    returns zero on degenerate inputs.
    """
    norm = dr.norm(vec)
    fallback_unit = fallback / (dr.norm(fallback) + EPS)
    return dr.select(norm > wt.Float(SMALL_EPS), vec / (norm + EPS), fallback_unit)


def normalize_real_with_fallback(vec, fallback):
    """Normalize ``vec`` with an ``EPS`` threshold; if degenerate, return the
    normalized ``fallback`` (and a hard ``Vector3f(1,0,0)`` if even fallback is
    near-zero). Used by UTD edge-frame construction where cross products may
    collapse for nearly-parallel inputs."""
    vec_norm = dr.norm(vec)
    fallback_norm = dr.norm(fallback)
    safe_fallback = dr.select(
        fallback_norm > wt.Float(EPS),
        fallback / fallback_norm,
        wt.Vector3f(1.0, 0.0, 0.0),
    )
    return dr.select(vec_norm > wt.Float(EPS), vec / vec_norm, safe_fallback)


def vector_from_jones(jones, basis):
    return vector_add(
        complex_scale_real(basis["u"], jones["u"]),
        complex_scale_real(basis["v"], jones["v"]),
    )


# ---------------------------------------------------------------------------
# Jones vector primitives (dicts keyed by u/v with Complex2f components)
# ---------------------------------------------------------------------------


def jones_zero(width: int):
    return {axis: complex_zero(width) for axis in _UV}


def jones_add(lhs, rhs):
    return {axis: lhs[axis] + rhs[axis] for axis in _UV}


def jones_scale(jones, coeff):
    return {axis: jones[axis] * coeff for axis in _UV}


def jones_select(mask, true_value, false_value):
    return {axis: dr.select(mask, true_value[axis], false_value[axis]) for axis in _UV}


def jones_from_vector(field_vec, basis):
    return {axis: complex_dot_real(field_vec, basis[axis]) for axis in _UV}


def jones_tangential(vec, axis="z"):
    a0, a1 = tangential_axes_for_axis(normalize_axis(axis))
    return {a0: vec[a0], a1: vec[a1]}


# ---------------------------------------------------------------------------
# Jones operator primitives (2x2 matrix: m00/m01/m10/m11 with Complex2f)
# ---------------------------------------------------------------------------


def jones_operator_identity(width: int):
    one = wt.Complex2f(dr.ones(wt.Float, width), dr.zeros(wt.Float, width))
    zero = complex_zero(width)
    return {"m00": one, "m01": zero, "m10": zero, "m11": one}


def jones_operator_diagonal(c0, c1=None):
    if c1 is None:
        c1 = c0
    zero = complex_zero(dr.width(c0.real))
    return {"m00": c0, "m01": zero, "m10": zero, "m11": c1}


def jones_operator_add(lhs, rhs):
    return {key: lhs[key] + rhs[key] for key in _MAT}


def jones_operator_scale(operator, coeff):
    return {key: operator[key] * coeff for key in _MAT}


def apply_jones_operator(jones, operator):
    return {
        "u": operator["m00"] * jones["u"] + operator["m01"] * jones["v"],
        "v": operator["m10"] * jones["u"] + operator["m11"] * jones["v"],
    }


def jones_operator_matmul(lhs, rhs):
    """Compose two Jones operators ``lhs @ rhs`` (2x2 complex matrix product)."""
    return {
        "m00": lhs["m00"] * rhs["m00"] + lhs["m01"] * rhs["m10"],
        "m01": lhs["m00"] * rhs["m01"] + lhs["m01"] * rhs["m11"],
        "m10": lhs["m10"] * rhs["m00"] + lhs["m11"] * rhs["m10"],
        "m11": lhs["m10"] * rhs["m01"] + lhs["m11"] * rhs["m11"],
    }


def jones_operator_rotator(k, s_current, s_target):
    """Operator that rotates the polarization basis around propagation ``k``
    from ``s_current`` to ``s_target``."""
    c = dr.dot(s_current, s_target)
    s = dr.dot(k, dr.cross(s_current, s_target))
    zero = dr.zeros(wt.Float, dr.width(c))
    return {
        "m00": wt.Complex2f(c, zero),
        "m01": wt.Complex2f(s, zero),
        "m10": wt.Complex2f(-s, zero),
        "m11": wt.Complex2f(c, zero),
    }


def jones_operator_mask_zero(operator, mask):
    """Zero out operator entries where ``mask`` is False (per-lane select)."""
    zero = complex_zero(dr.width(mask))
    return {key: dr.select(mask, operator[key], zero) for key in _MAT}


def jones_operator_mask_detach(operator, mask):
    """Detach operator entries where ``mask`` is False (preserves forward
    value but blocks gradient flow on inactive lanes)."""
    return {key: dr.select(mask, operator[key], dr.detach(operator[key])) for key in _MAT}


def fresnel_diagonal_operator(*, eta_r, sigma, gain, use_fresnel, cos_theta, wavelength, mu_r=1.0):
    """Diagonal Jones operator built from per-face Fresnel coefficients.

    Returns a TE/TM diagonal operator scaled by ``gain``; lanes with
    ``use_fresnel`` False produce a zero operator (transparent face)."""
    omega = material_angular_frequency(wavelength)
    eta = complex_relative_permittivity(eta_r, sigma, omega)
    r_te, r_tm = fresnel_reflection(cos_theta, eta, mu_r=mu_r)
    gain_complex = wt.Complex2f(gain, wt.Float(0.0))
    coeff_te = sanitize_complex(gain_complex * r_te)
    coeff_tm = sanitize_complex(gain_complex * r_tm)
    zero = complex_zero(dr.width(use_fresnel))
    coeff_te = dr.select(use_fresnel, coeff_te, zero)
    coeff_tm = dr.select(use_fresnel, coeff_tm, zero)
    return jones_operator_diagonal(coeff_te, coeff_tm)


def jones_operator_in_basis(operator, src_in_basis, src_out_basis, dst_in_basis, dst_out_basis):
    unit_u = {"u": wt.Complex2f(1.0, 0.0), "v": wt.Complex2f(0.0, 0.0)}
    unit_v = {"u": wt.Complex2f(0.0, 0.0), "v": wt.Complex2f(1.0, 0.0)}

    def _map(column_jones):
        field = vector_from_jones(column_jones, dst_in_basis)
        src_out = apply_jones_operator(jones_from_vector(field, src_in_basis), operator)
        return jones_from_vector(vector_from_jones(src_out, src_out_basis), dst_out_basis)

    mapped_u = _map(unit_u)
    mapped_v = _map(unit_v)
    return {
        "m00": mapped_u["u"], "m01": mapped_v["u"],
        "m10": mapped_u["v"], "m11": mapped_v["v"],
    }


# ---------------------------------------------------------------------------
# Polarization basis construction
# ---------------------------------------------------------------------------


def stable_perpendicular_basis(ray_dir, preferred=None):
    if preferred is None:
        preferred = wt.Vector3f(0.0, 0.0, 1.0)
    proj = preferred - dr.dot(preferred, ray_dir) * ray_dir
    alt_axis = dr.select(
        dr.abs(ray_dir.z) < 0.9,
        wt.Vector3f(0.0, 0.0, 1.0),
        wt.Vector3f(0.0, 1.0, 0.0),
    )
    alt_proj = alt_axis - dr.dot(alt_axis, ray_dir) * ray_dir
    return safe_normalize_with_fallback(proj, alt_proj)


def project_real_polarization_to_ray(polarization, ray_dir):
    return stable_perpendicular_basis(ray_dir, preferred=_polarization_vector(polarization))


def path_basis(ray_dir, preferred=None):
    ray_hat = ray_dir / (dr.norm(ray_dir) + EPS)
    u_hat = stable_perpendicular_basis(ray_hat, preferred=preferred)
    v_hat = safe_normalize_with_fallback(
        dr.cross(ray_hat, u_hat),
        stable_perpendicular_basis(ray_hat, preferred=wt.Vector3f(0.0, 1.0, 0.0)),
    )
    return {"u": u_hat, "v": v_hat, "k": ray_hat}


def basis_from_first_vector(ray_dir, first_vector, fallback=None):
    ray_hat = ray_dir / (dr.norm(ray_dir) + EPS)
    if fallback is None:
        fallback = stable_perpendicular_basis(ray_hat)
    u_hat = safe_normalize_with_fallback(
        first_vector - dr.dot(first_vector, ray_hat) * ray_hat,
        fallback,
    )
    v_hat = safe_normalize_with_fallback(
        dr.cross(ray_hat, u_hat),
        stable_perpendicular_basis(ray_hat, preferred=wt.Vector3f(0.0, 1.0, 0.0)),
    )
    return {"u": u_hat, "v": v_hat, "k": ray_hat}


def implicit_basis_vector(ray_dir):
    ray_hat = safe_normalize_with_fallback(ray_dir, wt.Vector3f(0.0, 0.0, 1.0))
    radial = dr.sqrt(ray_hat.x * ray_hat.x + ray_hat.y * ray_hat.y)
    fallback = wt.Vector3f(
        dr.select(ray_hat.z >= 0.0, wt.Float(1.0), wt.Float(-1.0)),
        wt.Float(0.0),
        wt.Float(0.0),
    )
    radial_ok = radial > wt.Float(EPS)
    theta_hat = wt.Vector3f(
        dr.select(radial_ok, ray_hat.z * ray_hat.x / radial, fallback.x),
        dr.select(radial_ok, ray_hat.z * ray_hat.y / radial, fallback.y),
        dr.select(radial_ok, -radial, fallback.z),
    )
    return safe_normalize_with_fallback(theta_hat, fallback)


def diffraction_edge_basis(ray_dir, edge_dir, *, outgoing=False):
    ray_hat = ray_dir / (dr.norm(ray_dir) + EPS)
    edge_hat = edge_dir / (dr.norm(edge_dir) + EPS)
    phi_hat = dr.cross(ray_hat, edge_hat)
    if outgoing:
        phi_hat = -phi_hat
    fallback = stable_perpendicular_basis(ray_hat, preferred=edge_hat)
    return basis_from_first_vector(ray_hat, phi_hat, fallback=fallback)


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


def reflect_direction(incident_dir, normal) -> wt.Vector3f:
    incident_hat = incident_dir / (dr.norm(incident_dir) + EPS)
    normal_hat = normal / (dr.norm(normal) + EPS)
    return incident_hat - 2.0 * dr.dot(incident_hat, normal_hat) * normal_hat


def sanitize_complex(coeff) -> wt.Complex2f:
    """Replace non-finite components of a complex value with zero."""
    return wt.Complex2f(
        dr.select(dr.isfinite(coeff.real), coeff.real, wt.Float(0.0)),
        dr.select(dr.isfinite(coeff.imag), coeff.imag, wt.Float(0.0)),
    )


def reflect_field_vector(field_vec, incident_dir, normal, eta_r, sigma, omega, gain=1.0, mu_r=1.0):
    incident_hat = incident_dir / (dr.norm(incident_dir) + EPS)
    normal_hat = normal / (dr.norm(normal) + EPS)
    reflected_dir = reflect_direction(incident_hat, normal_hat)

    s_hat = safe_normalize_with_fallback(
        dr.cross(normal_hat, incident_hat),
        stable_perpendicular_basis(incident_hat),
    )
    p_in_hat = safe_normalize_with_fallback(
        dr.cross(s_hat, incident_hat),
        stable_perpendicular_basis(incident_hat, preferred=normal_hat),
    )
    p_out_hat = safe_normalize_with_fallback(
        dr.cross(s_hat, reflected_dir),
        stable_perpendicular_basis(reflected_dir, preferred=normal_hat),
    )

    cos_theta = dr.clip(dr.abs(dr.dot(incident_hat, normal_hat)), wt.Float(SMALL_EPS), wt.Float(1.0))
    eta = complex_relative_permittivity(wt.Float(eta_r), wt.Float(sigma), wt.Float(omega))
    r_te, r_tm = fresnel_reflection(cos_theta, eta, mu_r=mu_r)
    gain_c = wt.Complex2f(wt.Float(gain), 0.0)

    e_s = complex_dot_real(field_vec, s_hat)
    e_p = complex_dot_real(field_vec, p_in_hat)
    return vector_add(
        complex_scale_real(s_hat, gain_c * sanitize_complex(r_te) * e_s),
        complex_scale_real(p_out_hat, gain_c * sanitize_complex(r_tm) * e_p),
    )


def effective_rx_polarization(rx_polarization, tx_polarization):
    return tx_polarization if rx_polarization is None else rx_polarization


def _component_as_complex2f(value) -> wt.Complex2f:
    if hasattr(value, "real") and hasattr(value, "imag"):
        return wt.Complex2f(wt.Float(value.real), wt.Float(value.imag))
    return wt.Complex2f(wt.Float(value), wt.Float(0.0))


# ---------------------------------------------------------------------------
# Receiver / scalarization
# ---------------------------------------------------------------------------


def receiver_tangential(polarization, axis="z"):
    a0, a1 = tangential_axes_for_axis(normalize_axis(axis))
    p0 = _component_as_complex2f(_polarization_component(polarization, a0))
    p1 = _component_as_complex2f(_polarization_component(polarization, a1))
    norm = dr.sqrt(
        p0.real * p0.real
        + p0.imag * p0.imag
        + p1.real * p1.real
        + p1.imag * p1.imag
    )
    safe_norm = dr.select(norm > wt.Float(SMALL_EPS), norm, wt.Float(1.0))
    return {
        a0: wt.Complex2f(p0.real / safe_norm, p0.imag / safe_norm),
        a1: wt.Complex2f(p1.real / safe_norm, p1.imag / safe_norm),
    }


def scalarize_tangential_jones(jones, polarization, axis="z"):
    axis_name = normalize_axis(axis)
    a0, a1 = tangential_axes_for_axis(axis_name)
    rx_pol = receiver_tangential(polarization, axis=axis_name)
    return jones[a0] * rx_pol[a0] + jones[a1] * rx_pol[a1]


def scalarize_vector_to_tangential_polarization(field_vec, polarization, axis="z"):
    return scalarize_tangential_jones(jones_tangential(field_vec, axis=axis), polarization, axis=axis)


def scalarize_vector_to_polarization(field_vec, ray_dir, polarization):
    return complex_dot_real(field_vec, project_real_polarization_to_ray(polarization, ray_dir))


__all__ = [
    "apply_jones_operator",
    "basis_from_first_vector",
    "complex_dot_real",
    "complex_scale_real",
    "complex_zero",
    "diffraction_edge_basis",
    "effective_rx_polarization",
    "fresnel_diagonal_operator",
    "implicit_basis_vector",
    "jones_add",
    "jones_from_vector",
    "jones_operator_add",
    "jones_operator_diagonal",
    "jones_operator_identity",
    "jones_operator_in_basis",
    "jones_operator_mask_detach",
    "jones_operator_mask_zero",
    "jones_operator_matmul",
    "jones_operator_rotator",
    "jones_operator_scale",
    "jones_scale",
    "jones_select",
    "jones_tangential",
    "jones_zero",
    "normalize_real_with_fallback",
    "path_basis",
    "project_real_polarization_to_ray",
    "receiver_tangential",
    "reflect_direction",
    "reflect_field_vector",
    "safe_normalize_with_fallback",
    "sanitize_complex",
    "scalarize_tangential_jones",
    "scalarize_vector_to_polarization",
    "scalarize_vector_to_tangential_polarization",
    "stable_perpendicular_basis",
    "vector_add",
    "vector_eval",
    "vector_from_jones",
    "vector_from_scalar",
    "vector_power",
    "vector_scale",
    "vector_select",
    "vector_zero",
]
