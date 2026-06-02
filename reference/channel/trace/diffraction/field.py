"""Field evaluation for diffraction edge states.

The accumulation loop (chunked scatter-reduce over state x receiver pairs)
lives in ``witwin.channel.kernels.trace.utd``.  This module provides the per-pair
field evaluation helper ``_edge_state_field_to_targets`` used by that kernel.
"""

import drjit as dr
import witwin as wt

from ...utils.constants import DIFFRACTION_MIN_DISTANCE, EPS, UTD_SLOPE_DERIVATIVE_STEP
from ...utils.material import complex_relative_permittivity, fresnel_reflection
from ...utils.polarization import (
    apply_jones_operator,
    basis_from_first_vector,
    diffraction_edge_basis,
    implicit_basis_vector,
    jones_add,
    jones_scale,
    jones_from_vector,
    jones_operator_add,
    jones_operator_diagonal,
    jones_operator_scale,
    vector_from_jones,
    vector_add,
    vector_scale,
    vector_select,
    vector_zero,
)
from ...utils.drjit_ops import (
    ArrayInit,
    Broadcast,
    broadcast_point,
    broadcast_vector,
    complex_abs_sqr,
    repeat_complex,
    repeat_float,
)
from ..materials import reflection_material_omega
from .finite_wedge import require_edge_state_line_bounds
from .geometry import (
    _cotangent_pole_safe_mask,
    _compute_edge_geometry,
    _edge_face_material_inputs,
    _point_inside_closed_mesh_mask,
    _slope_derivative_safe_mask,
    _wedge_exterior_region_mask,
)
from .material_ops import (
    face_reflection_operator_from_material_inputs,
    state_has_face_material_params,
)
from .operator import assemble_material_diffraction_operators
from .utd import cot, f_utd, fresnel_integral


def _pair_face_material_operators(
    edge_state,
    *,
    width,
    wavelength,
    material_detail,
    wedge_n,
    phi,
    phi_prime,
    incoming_edge_basis,
    outgoing_edge_basis,
    n0,
    nn,
):
    """Compute per-face Jones reflection operators for a diffraction edge.

    Uses precomputed face material inputs when available (requires wavelength).
    Falls back to pre-stored diagonal operators from the state when wavelength
    is not provided (e.g. hand-built test states).
    """
    if not state_has_face_material_params(edge_state) or wavelength is None:
        face0_op = {key: repeat_complex(edge_state[f"face0_operator_{key}"], width) for key in _OPERATOR_KEYS}
        face1_op = {key: repeat_complex(edge_state[f"face1_operator_{key}"], width) for key in _OPERATOR_KEYS}
        return face0_op, face1_op

    incoming_hat = incoming_edge_basis["k"]
    outgoing_hat = outgoing_edge_basis["k"]
    face0_material, face1_material = _edge_face_material_inputs(
        edge_state,
        width,
        material_detail,
    )
    face0_operator = face_reflection_operator_from_material_inputs(
        face0_material,
        cos_theta=dr.clip(dr.abs(dr.sin(phi_prime)), wt.Float(1e-6), wt.Float(1.0)),
        normal=n0,
        incoming_hat=incoming_hat,
        outgoing_hat=outgoing_hat,
        incoming_edge_basis=incoming_edge_basis,
        outgoing_edge_basis=outgoing_edge_basis,
        wavelength=wavelength,
    )
    face1_operator = face_reflection_operator_from_material_inputs(
        face1_material,
        cos_theta=dr.clip(dr.abs(dr.sin(wedge_n * dr.pi - phi)), wt.Float(1e-6), wt.Float(1.0)),
        normal=nn,
        incoming_hat=incoming_hat,
        outgoing_hat=outgoing_hat,
        incoming_edge_basis=incoming_edge_basis,
        outgoing_edge_basis=outgoing_edge_basis,
        wavelength=wavelength,
    )
    return face0_operator, face1_operator


_OPERATOR_KEYS = ("m00", "m01", "m10", "m11")


def _zero_jones_operator(width):
    zero = ArrayInit.complex_zero(width)
    return {key: zero for key in _OPERATOR_KEYS}


def _mask_jones_operator(operator, mask):
    zero = ArrayInit.complex_zero(dr.width(mask))
    return {
        key: dr.select(mask, operator[key], zero)
        for key in _OPERATOR_KEYS
    }


def _detach_mask_jones_operator(operator, mask):
    return {
        key: dr.select(mask, operator[key], dr.detach(operator[key]))
        for key in _OPERATOR_KEYS
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


def _smoothstep01(x):
    clamped = dr.clip(x, wt.Float(0.0), wt.Float(1.0))
    return clamped * clamped * (wt.Float(3.0) - wt.Float(2.0) * clamped)


def _resolve_incident_diffraction_state(edge_state, *, width):
    """Resolve canonicalized jones+basis state into broadened incident field components.

    All state arrays contain jones and basis fields after
    ``_canonicalize_transport_state`` in ``_make_state_arrays``.
    """
    incident_path_basis = {
        "u": broadcast_vector(edge_state["incident_basis_u"], width),
        "v": broadcast_vector(edge_state["incident_basis_v"], width),
        "k": broadcast_vector(edge_state["incident_basis_k"], width),
    }
    incident_jones = {
        "u": repeat_complex(edge_state["incident_jones_u"], width),
        "v": repeat_complex(edge_state["incident_jones_v"], width),
    }
    incident_derivative_jones = {
        "u": repeat_complex(edge_state["incident_derivative_jones_u"], width),
        "v": repeat_complex(edge_state["incident_derivative_jones_v"], width),
    }
    incident_vector = vector_from_jones(incident_jones, incident_path_basis)
    incident_normal_derivative_vector = vector_from_jones(
        incident_derivative_jones,
        incident_path_basis,
    )
    return (
        incident_path_basis,
        incident_jones,
        incident_derivative_jones,
        incident_vector,
        incident_normal_derivative_vector,
    )


def _finite_wedge_truncation_factor(edge_state, edge_geometry, target_pos, k, *, width):
    edge_line_min, edge_line_max = require_edge_state_line_bounds(
        edge_state,
        context="_finite_wedge_truncation_factor",
    )

    edge_pos_b = broadcast_point(edge_state["edge_pos"], width)
    source_pos_b = broadcast_point(edge_state["source_pos"], width)
    edge_hat = edge_geometry["edge_hat"]
    source_axial = dr.dot(source_pos_b - edge_pos_b, edge_hat)
    target_axial = dr.dot(target_pos - edge_pos_b, edge_hat)
    s_prime_proj = edge_geometry["s_prime_proj"]
    s_proj = edge_geometry["s_proj"]
    stationary_u = (
        s_prime_proj * target_axial + s_proj * source_axial
    ) / (s_proj + s_prime_proj + wt.Float(EPS))
    source_offset = stationary_u - source_axial
    target_offset = target_axial - stationary_u
    source_range = dr.sqrt(s_prime_proj * s_prime_proj + source_offset * source_offset + wt.Float(EPS))
    target_range = dr.sqrt(s_proj * s_proj + target_offset * target_offset + wt.Float(EPS))
    curvature = (
        s_prime_proj * s_prime_proj / (source_range * source_range * source_range + wt.Float(EPS))
        + s_proj * s_proj / (target_range * target_range * target_range + wt.Float(EPS))
    )
    scale = dr.sqrt(dr.maximum(wt.Float(k) * curvature, wt.Float(EPS)) / dr.pi)
    line_min = repeat_float(edge_line_min, width)
    line_max = repeat_float(edge_line_max, width)
    delta_f = fresnel_integral(scale * (line_max - stationary_u)) - fresnel_integral(
        scale * (line_min - stationary_u)
    )
    return wt.Complex2f(0.5, 0.5) * dr.conj(delta_f)


def _edge_state_target_support(
    edge_state,
    target_pos,
    *,
    scene=None,
    smooth_exterior_shadow=False,
):
    width = dr.width(target_pos.x)
    edge_geometry = _compute_edge_geometry(
        edge_state["source_pos"],
        edge_state["edge_pos"],
        edge_state["edge_dir"],
        edge_state["n0"],
        target_pos,
    )
    phi = edge_geometry["phi"]
    phi_prime = edge_geometry["phi_prime"]
    s = edge_geometry["s"]
    s_prime = edge_geometry["s_prime"]
    wedge_n = repeat_float(edge_state["wedge_n"], width)
    edge_pos_b = broadcast_point(edge_state["edge_pos"], width)
    edge_dir_b = broadcast_vector(edge_state["edge_dir"], width)
    n0_b = broadcast_vector(edge_state["n0"], width)
    nn_b = broadcast_vector(edge_state["n_face_n"], width)
    source_pos_b = broadcast_point(edge_state["source_pos"], width)
    source_exterior = _wedge_exterior_region_mask(
        source_pos_b - edge_pos_b,
        edge_dir_b,
        n0_b,
        nn_b,
    )
    target_exterior = _wedge_exterior_region_mask(
        target_pos - edge_pos_b,
        edge_dir_b,
        n0_b,
        nn_b,
    )
    base_valid = (
        source_exterior
        & (s_prime > DIFFRACTION_MIN_DISTANCE)
        & (s > DIFFRACTION_MIN_DISTANCE)
    )
    interior_mask = dr.zeros(wt.Bool, width)
    if smooth_exterior_shadow:
        if scene is not None:
            interior_mask = _point_inside_closed_mesh_mask(
                target_pos,
                scene,
                active=base_valid & ~target_exterior,
            )
        geometry_valid = base_valid & (target_exterior | ~interior_mask)
    else:
        geometry_valid = base_valid & target_exterior
    field_valid = geometry_valid

    shadow_completion_mask = dr.zeros(wt.Bool, width)
    shadow_completion_weight = dr.zeros(wt.Float, width)
    shadow_boundary_distance = dr.zeros(wt.Float, width)
    illuminated_boundary_mask = dr.zeros(wt.Bool, width)
    illuminated_boundary_weight = dr.zeros(wt.Float, width)
    shadow_opening_angle = dr.zeros(wt.Float, width)
    shadow_anchor_offset = dr.zeros(wt.Float, width)
    shadow_anchor_outer_offset = dr.zeros(wt.Float, width)
    shadow_decay_span = dr.zeros(wt.Float, width)
    shadow_decay_power = dr.zeros(wt.Float, width)
    if smooth_exterior_shadow:
        shadow_completion_mask = base_valid & ~target_exterior & ~interior_mask
        if dr.any(shadow_completion_mask):
            boundary_eps = wt.Float(1.0e-3)
            shadow_opening_angle = dr.maximum(
                wt.Float(2.0 * dr.pi) - wedge_n * dr.pi,
                wt.Float(2.0) * boundary_eps,
            )
            shadow_angle_ratio = dr.minimum(
                shadow_opening_angle / dr.pi,
                wt.Float(1.0),
            )
            shadow_decay_span = dr.maximum(
                (wt.Float(0.17) + wt.Float(0.12) * shadow_angle_ratio)
                * shadow_opening_angle,
                wt.Float(8.0) * boundary_eps,
            )
            shadow_decay_span = dr.minimum(
                shadow_decay_span,
                wt.Float(0.5) * shadow_opening_angle,
            )
            shadow_decay_power = (
                wt.Float(2.0)
                + wt.Float(1.0) * (wt.Float(1.0) - shadow_angle_ratio)
            )
            shadow_anchor_offset = dr.maximum(
                (wt.Float(1.0e-2) + wt.Float(1.0e-2) * shadow_angle_ratio)
                * shadow_opening_angle,
                wt.Float(4.0) * boundary_eps,
            )
            shadow_anchor_offset = dr.minimum(
                shadow_anchor_offset,
                wt.Float(0.25) * shadow_opening_angle,
            )
            shadow_anchor_outer_offset = dr.maximum(
                dr.minimum(
                    wt.Float(2.0) * shadow_anchor_offset,
                    wt.Float(0.18) * shadow_opening_angle,
                ),
                shadow_anchor_offset + wt.Float(4.0) * boundary_eps,
            )
            shadow_half_angle = wt.Float(0.5) * shadow_opening_angle
            illuminated_wrap_mask = target_exterior & (phi <= shadow_anchor_outer_offset)
            illuminated_face_mask = target_exterior & (
                phi >= dr.maximum(
                    wedge_n * dr.pi - shadow_anchor_outer_offset,
                    boundary_eps,
                )
            )
            illuminated_boundary_mask = illuminated_wrap_mask | illuminated_face_mask
            wrap_boundary = (
                (
                    shadow_completion_mask
                    & (phi >= (wt.Float(2.0 * dr.pi) - shadow_half_angle))
                )
                | illuminated_wrap_mask
            )
            shadow_boundary_distance = dr.select(
                wrap_boundary,
                wt.Float(2.0 * dr.pi) - phi,
                phi - wedge_n * dr.pi,
            )
            illuminated_boundary_distance = dr.select(
                wrap_boundary,
                phi,
                wedge_n * dr.pi - phi,
            )
            shadow_boundary_u = dr.clip(
                shadow_boundary_distance / shadow_decay_span,
                wt.Float(0.0),
                wt.Float(1.0),
            )
            shadow_completion_curve = (
                wt.Float(0.88) * (wt.Float(1.0) - _smoothstep01(shadow_boundary_u))
                + wt.Float(0.12) * (wt.Float(1.0) - shadow_boundary_u)
            )
            shadow_completion_weight = dr.power(
                shadow_completion_curve,
                shadow_decay_power,
            )
            shadow_completion_mask = shadow_completion_mask & (
                shadow_boundary_distance < shadow_decay_span
            )
            illuminated_boundary_weight = wt.Float(1.0) - _smoothstep01(
                illuminated_boundary_distance / shadow_anchor_outer_offset,
            )
        field_valid = field_valid & (target_exterior | shadow_completion_mask)

    return {
        "width": width,
        "edge_geometry": edge_geometry,
        "phi": phi,
        "phi_prime": phi_prime,
        "s": s,
        "s_prime": s_prime,
        "wedge_n": wedge_n,
        "edge_pos_b": edge_pos_b,
        "edge_dir_b": edge_dir_b,
        "n0_b": n0_b,
        "nn_b": nn_b,
        "source_pos_b": source_pos_b,
        "target_exterior": target_exterior,
        "interior_mask": interior_mask,
        "geometry_valid": geometry_valid,
        "field_valid": field_valid,
        "shadow_completion_mask": shadow_completion_mask,
        "shadow_completion_weight": shadow_completion_weight,
        "shadow_boundary_distance": shadow_boundary_distance,
        "illuminated_boundary_mask": illuminated_boundary_mask,
        "illuminated_boundary_weight": illuminated_boundary_weight,
        "shadow_opening_angle": shadow_opening_angle,
        "shadow_anchor_offset": shadow_anchor_offset,
        "shadow_anchor_outer_offset": shadow_anchor_outer_offset,
        "shadow_decay_span": shadow_decay_span,
        "shadow_decay_power": shadow_decay_power,
    }


def _edge_state_field_to_targets(
    edge_state,
    target_pos,
    k,
    return_normal_derivative=False,
    return_vector=False,
    return_valid=False,
    return_support=False,
    wavelength=None,
    material_detail=None,
    support_override=None,
    scene=None,
    smooth_exterior_shadow=False,
):
    width = dr.width(target_pos.x)
    edge_geometry = _compute_edge_geometry(
        edge_state["source_pos"],
        edge_state["edge_pos"],
        edge_state["edge_dir"],
        edge_state["n0"],
        target_pos,
    )
    phi = edge_geometry["phi"]
    phi_prime = edge_geometry["phi_prime"]
    s = edge_geometry["s"]
    s_prime = edge_geometry["s_prime"]
    sin_beta0 = edge_geometry["sin_beta_eff"]
    wedge_n = repeat_float(edge_state["wedge_n"], width)
    edge_pos_b = broadcast_point(edge_state["edge_pos"], width)
    edge_dir_b = broadcast_vector(edge_state["edge_dir"], width)
    n0_b = broadcast_vector(edge_state["n0"], width)
    nn_b = broadcast_vector(edge_state["n_face_n"], width)
    source_pos_b = broadcast_point(edge_state["source_pos"], width)
    source_exterior = _wedge_exterior_region_mask(
        source_pos_b - edge_pos_b,
        edge_dir_b,
        n0_b,
        nn_b,
    )
    target_exterior = _wedge_exterior_region_mask(
        target_pos - edge_pos_b,
        edge_dir_b,
        n0_b,
        nn_b,
    )
    base_valid = (
        source_exterior
        & (s_prime > DIFFRACTION_MIN_DISTANCE)
        & (s > DIFFRACTION_MIN_DISTANCE)
    )
    interior_mask = dr.zeros(wt.Bool, width)
    if smooth_exterior_shadow:
        # Some callers only need the exterior-shadow boundary completion and do
        # not have ambiguous volumetric interior targets. Allow them to skip the
        # closed-mesh interior query while reusing the same boundary contract.
        if scene is not None:
            interior_mask = _point_inside_closed_mesh_mask(
                target_pos,
                scene,
                active=base_valid & ~target_exterior,
            )
        geometry_valid = base_valid & (target_exterior | ~interior_mask)
    else:
        geometry_valid = base_valid & target_exterior
    field_valid = geometry_valid
    # Exact wedge-face boundary rays have a finite forward field but an undefined
    # angular derivative. Keep the forward value and suppress gradients only on
    # those cotangent-pole samples.
    pole_safe = geometry_valid & _cotangent_pole_safe_mask(phi, phi_prime, wedge_n, wt.Float(1e-6))
    safe_phi = dr.select(pole_safe, phi, 0.5 * wedge_n * dr.pi)
    safe_phi_prime = dr.select(pole_safe, phi_prime, 0.5 * wedge_n * dr.pi)
    step = wt.Float(UTD_SLOPE_DERIVATIVE_STEP)
    slope_safe = field_valid & _slope_derivative_safe_mask(safe_phi, safe_phi_prime, wedge_n, step)
    if support_override is not None:
        field_valid = _override_mask(support_override.get("field_valid"), field_valid, width=width)
        pole_safe = _override_mask(support_override.get("pole_safe"), pole_safe, width=width)
        slope_safe = _override_mask(support_override.get("slope_safe"), slope_safe, width=width)
        safe_phi = dr.select(pole_safe, phi, 0.5 * wedge_n * dr.pi)
        safe_phi_prime = dr.select(pole_safe, phi_prime, 0.5 * wedge_n * dr.pi)
    local_scale = dr.sqrt(s_prime / (s * (s + s_prime) + EPS))
    phase = dr.exp(wt.Complex2f(0, -wt.Float(k) * s))
    finite_wedge_factor = _finite_wedge_truncation_factor(
        edge_state,
        edge_geometry,
        target_pos,
        k,
        width=width,
    )
    incoming_edge_basis = diffraction_edge_basis(edge_pos_b - source_pos_b, edge_dir_b, outgoing=False)
    outgoing_edge_basis = diffraction_edge_basis(target_pos - edge_pos_b, edge_dir_b, outgoing=True)
    (
        incident_path_basis,
        _incident_jones,
        _incident_derivative_jones,
        incident_vector,
        incident_normal_derivative_vector,
    ) = _resolve_incident_diffraction_state(edge_state, width=width)
    incident_jones_edge = jones_from_vector(incident_vector, incoming_edge_basis)
    incident_derivative_jones_edge = jones_from_vector(
        incident_normal_derivative_vector,
        incoming_edge_basis,
    )
    incident_derivative_power = (
        complex_abs_sqr(incident_derivative_jones_edge["u"])
        + complex_abs_sqr(incident_derivative_jones_edge["v"])
    )
    has_slope = (
        incident_derivative_power
        > wt.Float(1e-24)
    ) & slope_safe
    if support_override is not None:
        has_slope = _override_mask(support_override.get("has_slope"), has_slope, width=width)

    face0_operator, face1_operator = _pair_face_material_operators(
        edge_state,
        width=width,
        wavelength=wavelength,
        material_detail=material_detail,
        wedge_n=wedge_n,
        phi=phi,
        phi_prime=phi_prime,
        incoming_edge_basis=incoming_edge_basis,
        outgoing_edge_basis=outgoing_edge_basis,
        n0=n0_b,
        nn=nn_b,
    )
    operator_kwargs = {
        "wedge_n": wedge_n,
        "k": k,
        "s": s,
        "s_prime": s_prime,
        "face0_operator": face0_operator,
        "face1_operator": face1_operator,
        "sin_beta0": sin_beta0 if state_has_face_material_params(edge_state) else None,
    }
    need_normal_derivative_ops = bool(return_normal_derivative)
    operators = assemble_material_diffraction_operators(
        phi=phi,
        phi_prime=phi_prime,
        include_normal_derivative_ops=need_normal_derivative_ops,
        **operator_kwargs,
    )
    safe_operators = assemble_material_diffraction_operators(
        phi=safe_phi,
        phi_prime=safe_phi_prime,
        include_normal_derivative_ops=need_normal_derivative_ops,
        **operator_kwargs,
    )

    field_operator = _detach_mask_jones_operator(operators["field"], pole_safe)
    slope_operator = _mask_jones_operator(safe_operators["slope"], has_slope)
    field_jones = jones_add(
        apply_jones_operator(incident_jones_edge, field_operator),
        apply_jones_operator(incident_derivative_jones_edge, slope_operator),
    )
    field_jones = jones_scale(field_jones, finite_wedge_factor)
    scaled_field_jones = jones_scale(field_jones, local_scale * phase)
    vector_field = vector_from_jones(scaled_field_jones, outgoing_edge_basis)

    def _evaluate_boundary_sample(sample_phi, sample_phi_prime, sample_mask):
        sample_pole_safe = sample_mask & _cotangent_pole_safe_mask(
            sample_phi,
            sample_phi_prime,
            wedge_n,
            wt.Float(1e-6),
        )
        sample_safe_phi = dr.select(sample_pole_safe, sample_phi, 0.5 * wedge_n * dr.pi)
        sample_safe_phi_prime = dr.select(
            sample_pole_safe,
            sample_phi_prime,
            0.5 * wedge_n * dr.pi,
        )
        sample_slope_safe = sample_mask & _slope_derivative_safe_mask(
            sample_safe_phi,
            sample_safe_phi_prime,
            wedge_n,
            step,
        )
        sample_has_slope = (incident_derivative_power > wt.Float(1e-24)) & sample_slope_safe
        sample_face0_operator, sample_face1_operator = _pair_face_material_operators(
            edge_state,
            width=width,
            wavelength=wavelength,
            material_detail=material_detail,
            wedge_n=wedge_n,
            phi=sample_phi,
            phi_prime=sample_phi_prime,
            incoming_edge_basis=incoming_edge_basis,
            outgoing_edge_basis=outgoing_edge_basis,
            n0=n0_b,
            nn=nn_b,
        )
        sample_operator_kwargs = {
            "wedge_n": wedge_n,
            "k": k,
            "s": s,
            "s_prime": s_prime,
            "face0_operator": sample_face0_operator,
            "face1_operator": sample_face1_operator,
            "sin_beta0": sin_beta0 if state_has_face_material_params(edge_state) else None,
        }
        sample_operators = assemble_material_diffraction_operators(
            phi=sample_phi,
            phi_prime=sample_phi_prime,
            include_normal_derivative_ops=need_normal_derivative_ops,
            **sample_operator_kwargs,
        )
        sample_safe_operators = assemble_material_diffraction_operators(
            phi=sample_safe_phi,
            phi_prime=sample_safe_phi_prime,
            include_normal_derivative_ops=need_normal_derivative_ops,
            **sample_operator_kwargs,
        )
        sample_field_operator = _detach_mask_jones_operator(
            sample_operators["field"],
            sample_pole_safe,
        )
        sample_slope_operator = _mask_jones_operator(
            sample_safe_operators["slope"],
            sample_has_slope,
        )
        sample_field_jones = jones_add(
            apply_jones_operator(incident_jones_edge, sample_field_operator),
            apply_jones_operator(incident_derivative_jones_edge, sample_slope_operator),
        )
        sample_field_jones = jones_scale(sample_field_jones, finite_wedge_factor)
        sample_scaled_field_jones = jones_scale(sample_field_jones, local_scale * phase)
        sample_vector = vector_from_jones(sample_scaled_field_jones, outgoing_edge_basis)
        sample_result = {
            "field_jones": sample_scaled_field_jones,
            "vector": sample_vector,
        }
        if return_normal_derivative:
            sample_field_dphi_operator = _mask_jones_operator(
                sample_safe_operators["field_dphi"],
                sample_slope_safe,
            )
            sample_slope_dphi_operator = _mask_jones_operator(
                sample_safe_operators["slope_dphi"],
                sample_has_slope,
            )
            sample_normal_jones = jones_add(
                apply_jones_operator(incident_jones_edge, sample_field_dphi_operator),
                apply_jones_operator(incident_derivative_jones_edge, sample_slope_dphi_operator),
            )
            sample_normal_jones = jones_scale(sample_normal_jones, finite_wedge_factor)
            sample_scaled_normal_jones = jones_scale(
                sample_normal_jones,
                local_scale * phase / (s + EPS),
            )
            sample_result["normal_jones"] = sample_scaled_normal_jones
            sample_result["normal_vector"] = vector_from_jones(
                sample_scaled_normal_jones,
                outgoing_edge_basis,
            )
        return sample_result

    shadow_completion_mask = dr.zeros(wt.Bool, width)
    shadow_completion_weight = dr.zeros(wt.Float, width)
    shadow_boundary_distance = dr.zeros(wt.Float, width)
    illuminated_boundary_mask = dr.zeros(wt.Bool, width)
    illuminated_boundary_weight = dr.zeros(wt.Float, width)
    shadow_opening_angle = dr.zeros(wt.Float, width)
    shadow_anchor_offset = dr.zeros(wt.Float, width)
    shadow_anchor_outer_offset = dr.zeros(wt.Float, width)
    shadow_decay_span = dr.zeros(wt.Float, width)
    shadow_decay_power = dr.zeros(wt.Float, width)
    shadow_completion_field = ArrayInit.complex_zero(width)
    shadow_completion_vector = vector_zero(width)
    shadow_completion_normal = ArrayInit.complex_zero(width)
    shadow_completion_normal_vector = vector_zero(width)
    boundary_anchor_normal = ArrayInit.complex_zero(width)
    boundary_anchor_normal_vector = vector_zero(width)
    illuminated_boundary_field = ArrayInit.complex_zero(width)
    illuminated_boundary_vector = vector_zero(width)
    illuminated_boundary_normal = ArrayInit.complex_zero(width)
    illuminated_boundary_normal_vector = vector_zero(width)
    if smooth_exterior_shadow:
        shadow_completion_mask = base_valid & ~target_exterior & ~interior_mask
        boundary_anchor_mask = dr.zeros(wt.Bool, width)
        if dr.any(shadow_completion_mask):
            boundary_eps = wt.Float(1.0e-3)
            # Let the free-space opening angle alpha drive the exterior-only
            # boundary completion instead of leaking the raw UTD field into the
            # whole shadow sector.
            shadow_opening_angle = dr.maximum(
                wt.Float(2.0 * dr.pi) - wedge_n * dr.pi,
                wt.Float(2.0) * boundary_eps,
            )
            shadow_angle_ratio = dr.minimum(
                shadow_opening_angle / dr.pi,
                wt.Float(1.0),
            )
            shadow_decay_span = dr.maximum(
                (wt.Float(0.17) + wt.Float(0.12) * shadow_angle_ratio)
                * shadow_opening_angle,
                wt.Float(8.0) * boundary_eps,
            )
            shadow_decay_span = dr.minimum(
                shadow_decay_span,
                wt.Float(0.5) * shadow_opening_angle,
            )
            shadow_decay_power = (
                wt.Float(2.0)
                + wt.Float(1.0) * (wt.Float(1.0) - shadow_angle_ratio)
            )
            shadow_anchor_offset = dr.maximum(
                (wt.Float(1.0e-2) + wt.Float(1.0e-2) * shadow_angle_ratio)
                * shadow_opening_angle,
                wt.Float(4.0) * boundary_eps,
            )
            shadow_anchor_offset = dr.minimum(
                shadow_anchor_offset,
                wt.Float(0.25) * shadow_opening_angle,
            )
            shadow_anchor_outer_offset = dr.maximum(
                dr.minimum(
                    wt.Float(2.0) * shadow_anchor_offset,
                    wt.Float(0.18) * shadow_opening_angle,
                ),
                shadow_anchor_offset + wt.Float(4.0) * boundary_eps,
            )
            shadow_half_angle = wt.Float(0.5) * shadow_opening_angle
            illuminated_wrap_mask = target_exterior & (phi <= shadow_anchor_outer_offset)
            illuminated_face_mask = target_exterior & (
                phi >= dr.maximum(
                    wedge_n * dr.pi - shadow_anchor_outer_offset,
                    boundary_eps,
                )
            )
            illuminated_boundary_mask = illuminated_wrap_mask | illuminated_face_mask
            boundary_anchor_mask = shadow_completion_mask | illuminated_boundary_mask
            wrap_boundary = (
                (
                    shadow_completion_mask
                    & (phi >= (wt.Float(2.0 * dr.pi) - shadow_half_angle))
                )
                | illuminated_wrap_mask
            )
            shadow_boundary_distance = dr.select(
                wrap_boundary,
                wt.Float(2.0 * dr.pi) - phi,
                phi - wedge_n * dr.pi,
            )
            illuminated_boundary_distance = dr.select(
                wrap_boundary,
                phi,
                wedge_n * dr.pi - phi,
            )
            shadow_boundary_u = dr.clip(
                shadow_boundary_distance / shadow_decay_span,
                wt.Float(0.0),
                wt.Float(1.0),
            )
            shadow_completion_curve = (
                wt.Float(0.88) * (wt.Float(1.0) - _smoothstep01(shadow_boundary_u))
                + wt.Float(0.12) * (wt.Float(1.0) - shadow_boundary_u)
            )
            shadow_completion_weight = dr.power(
                shadow_completion_curve,
                shadow_decay_power,
            )
            shadow_completion_mask = shadow_completion_mask & (
                shadow_boundary_distance < shadow_decay_span
            )
            illuminated_boundary_weight = wt.Float(1.0) - _smoothstep01(
                illuminated_boundary_distance / shadow_anchor_outer_offset,
            )
            boundary_phi_near = dr.select(
                wrap_boundary,
                shadow_anchor_offset,
                dr.maximum(wedge_n * dr.pi - shadow_anchor_offset, boundary_eps),
            )
            boundary_phi_far = dr.select(
                wrap_boundary,
                shadow_anchor_outer_offset,
                dr.maximum(
                    wedge_n * dr.pi - shadow_anchor_outer_offset,
                    boundary_eps,
                ),
            )
            boundary_phi_prime = phi_prime
            # Estimate the face-boundary anchor from illuminated-side samples
            # instead of evaluating the raw UTD coefficient directly on the face.
            near_boundary_sample = _evaluate_boundary_sample(
                boundary_phi_near,
                boundary_phi_prime,
                boundary_anchor_mask,
            )
            far_boundary_sample = _evaluate_boundary_sample(
                boundary_phi_far,
                boundary_phi_prime,
                boundary_anchor_mask,
            )
            boundary_extrapolation_denom = dr.maximum(
                shadow_anchor_outer_offset - shadow_anchor_offset,
                wt.Float(4.0) * boundary_eps,
            )
            boundary_near_weight = (
                shadow_anchor_outer_offset / boundary_extrapolation_denom
            )
            boundary_far_weight = (
                shadow_anchor_offset / boundary_extrapolation_denom
            )
            boundary_face_field_jones = jones_add(
                jones_scale(
                    near_boundary_sample["field_jones"],
                    boundary_near_weight,
                ),
                jones_scale(
                    far_boundary_sample["field_jones"],
                    -boundary_far_weight,
                ),
            )
            boundary_anchor_vector = vector_add(
                vector_scale(
                    near_boundary_sample["vector"],
                    boundary_near_weight,
                ),
                vector_scale(
                    far_boundary_sample["vector"],
                    -boundary_far_weight,
                ),
            )
            shadow_completion_field = (
                boundary_face_field_jones["u"] * shadow_completion_weight
            )
            shadow_completion_vector = vector_scale(
                boundary_anchor_vector,
                shadow_completion_weight,
            )
            illuminated_boundary_field = (
                scaled_field_jones["u"] * (wt.Float(1.0) - illuminated_boundary_weight)
                + boundary_face_field_jones["u"] * illuminated_boundary_weight
            )
            illuminated_boundary_vector = vector_add(
                vector_scale(vector_field, wt.Float(1.0) - illuminated_boundary_weight),
                vector_scale(boundary_anchor_vector, illuminated_boundary_weight),
            )
            if return_normal_derivative:
                boundary_face_normal_jones = jones_add(
                    jones_scale(
                        near_boundary_sample["normal_jones"],
                        boundary_near_weight,
                    ),
                    jones_scale(
                        far_boundary_sample["normal_jones"],
                        -boundary_far_weight,
                    ),
                )
                boundary_anchor_normal = boundary_face_normal_jones["u"]
                shadow_completion_normal = (
                    boundary_anchor_normal * shadow_completion_weight
                )
                boundary_anchor_normal_vector = vector_add(
                    vector_scale(
                        near_boundary_sample["normal_vector"],
                        boundary_near_weight,
                    ),
                    vector_scale(
                        far_boundary_sample["normal_vector"],
                        -boundary_far_weight,
                    ),
                )
                shadow_completion_normal_vector = vector_scale(
                    boundary_anchor_normal_vector,
                    shadow_completion_weight,
                )
    field_valid = field_valid & (target_exterior | shadow_completion_mask)
    pole_safe = pole_safe & field_valid
    slope_safe = slope_safe & field_valid
    has_slope = has_slope & field_valid
    vector_field = vector_select(field_valid, vector_field, vector_zero(width))
    # Keep the scalar return as a single canonical component derived from the
    # outgoing Jones truth rather than a separate physical diffraction model.
    field = dr.select(field_valid, scaled_field_jones["u"], ArrayInit.complex_zero(width))
    if dr.any(shadow_completion_mask):
        field = dr.select(shadow_completion_mask, shadow_completion_field, field)
        vector_field = vector_select(shadow_completion_mask, shadow_completion_vector, vector_field)
    if dr.any(illuminated_boundary_mask):
        field = dr.select(illuminated_boundary_mask, illuminated_boundary_field, field)
        vector_field = vector_select(
            illuminated_boundary_mask,
            illuminated_boundary_vector,
            vector_field,
        )

    support = {
        "field_valid": field_valid,
        "pole_safe": pole_safe,
        "slope_safe": slope_safe,
        "has_slope": has_slope,
        "interior_mask": interior_mask,
        "target_exterior": target_exterior,
        "shadow_completion_mask": shadow_completion_mask,
        "shadow_completion_weight": shadow_completion_weight,
        "shadow_opening_angle": shadow_opening_angle,
        "shadow_anchor_offset": shadow_anchor_offset,
        "shadow_anchor_outer_offset": shadow_anchor_outer_offset,
        "shadow_decay_span": shadow_decay_span,
        "shadow_decay_power": shadow_decay_power,
        "illuminated_boundary_mask": illuminated_boundary_mask,
        "illuminated_boundary_weight": illuminated_boundary_weight,
    }

    if not return_normal_derivative:
        if return_vector:
            if return_valid and return_support:
                return field, vector_field, field_valid, support
            if return_valid:
                return field, vector_field, field_valid
            if return_support:
                return field, vector_field, support
            return field, vector_field
        if return_valid and return_support:
            return field, field_valid, support
        if return_valid:
            return field, field_valid
        if return_support:
            return field, support
        return field

    field_dphi_operator = _mask_jones_operator(safe_operators["field_dphi"], slope_safe)
    slope_dphi_operator = _mask_jones_operator(safe_operators["slope_dphi"], has_slope)
    normal_jones = jones_add(
        apply_jones_operator(incident_jones_edge, field_dphi_operator),
        apply_jones_operator(incident_derivative_jones_edge, slope_dphi_operator),
    )
    normal_jones = jones_scale(normal_jones, finite_wedge_factor)
    scaled_normal_jones = jones_scale(normal_jones, local_scale * phase / (s + EPS))
    normal_derivative_vector = vector_from_jones(
        scaled_normal_jones,
        outgoing_edge_basis,
    )
    normal_derivative_vector = vector_select(field_valid, normal_derivative_vector, vector_zero(width))
    normal_derivative = dr.select(field_valid, scaled_normal_jones["u"], ArrayInit.complex_zero(width))
    if dr.any(shadow_completion_mask):
        normal_derivative = dr.select(
            shadow_completion_mask,
            shadow_completion_normal,
            normal_derivative,
        )
        normal_derivative_vector = vector_select(
            shadow_completion_mask,
            shadow_completion_normal_vector,
            normal_derivative_vector,
        )
    if dr.any(illuminated_boundary_mask):
        illuminated_boundary_normal = (
            scaled_normal_jones["u"] * (wt.Float(1.0) - illuminated_boundary_weight)
            + boundary_anchor_normal * illuminated_boundary_weight
        )
        illuminated_boundary_normal_vector = vector_add(
            vector_scale(
                normal_derivative_vector,
                wt.Float(1.0) - illuminated_boundary_weight,
            ),
            vector_scale(boundary_anchor_normal_vector, illuminated_boundary_weight),
        )
        normal_derivative = dr.select(
            illuminated_boundary_mask,
            illuminated_boundary_normal,
            normal_derivative,
        )
        normal_derivative_vector = vector_select(
            illuminated_boundary_mask,
            illuminated_boundary_normal_vector,
            normal_derivative_vector,
        )

    if return_normal_derivative and return_vector:
        if return_valid and return_support:
            return field, normal_derivative, vector_field, normal_derivative_vector, field_valid, support
        if return_valid:
            return field, normal_derivative, vector_field, normal_derivative_vector, field_valid
        if return_support:
            return field, normal_derivative, vector_field, normal_derivative_vector, support
        return field, normal_derivative, vector_field, normal_derivative_vector
    if return_normal_derivative:
        if return_valid and return_support:
            return field, normal_derivative, field_valid, support
        if return_valid:
            return field, normal_derivative, field_valid
        if return_support:
            return field, normal_derivative, support
        return field, normal_derivative
    if return_valid and return_support:
        return field, vector_field, field_valid, support
    if return_valid:
        return field, vector_field, field_valid
    if return_support:
        return field, vector_field, support
    return field, vector_field


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
        wt.Float(1e-6),
    )
    phi_eval = dr.select(pole_safe, phi, 0.5 * wedge_n_b * dr.pi)
    phi_prime_eval = dr.select(pole_safe, phi_prime, 0.5 * wedge_n_b * dr.pi)
    exterior_angle = wedge_n_b * dr.pi
    l = s * s_prime * dr.rcp(s + s_prime + wt.Float(EPS)) * dr.square(sin_beta0)
    n = wedge_n_b
    dif_phi = phi_eval - phi_prime_eval
    sum_phi = phi_eval + phi_prime_eval

    def _a_p_m(beta):
        n_p = dr.round((beta + dr.pi) * dr.rcp(2.0 * exterior_angle))
        n_m = dr.round((beta - dr.pi) * dr.rcp(2.0 * exterior_angle))
        a_p = 2.0 * dr.square(dr.cos(exterior_angle * n_p - beta * 0.5))
        a_m = 2.0 * dr.square(dr.cos(exterior_angle * n_m - beta * 0.5))
        return a_p, a_m

    a1, a2 = _a_p_m(dif_phi)
    a3, a4 = _a_p_m(sum_phi)
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

    cos_theta0 = dr.clip(dr.abs(dr.sin(phi_prime_eval)), wt.Float(1e-6), wt.Float(1.0))
    cos_theta1 = dr.clip(
        dr.abs(dr.sin(exterior_angle - phi_eval)),
        wt.Float(1e-6),
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
    scaled_field_jones = jones_scale(field_jones, local_scale)
    field_power = dr.select(
        field_valid,
        complex_abs_sqr(scaled_field_jones["u"]) + complex_abs_sqr(scaled_field_jones["v"]),
        wt.Float(0.0),
    )
    if return_valid:
        return field_power, field_valid
    return field_power


__all__ = [
    "_edge_state_target_support",
    "_edge_state_field_to_targets",
    "_sampled_edge_diffraction_power_to_targets_mc",
]

