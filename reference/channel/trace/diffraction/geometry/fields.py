"""Field computation, reflection coefficients, and material helpers for diffraction geometry."""

import math

import drjit as dr
import witwin as wt

from ....utils.constants import EPS, SMALL_EPS
from ....utils.material import complex_relative_permittivity, fresnel_reflection, scalar_fresnel_reflection
from ....utils.polarization import (
    jones_operator_diagonal,
    project_real_polarization_to_ray,
    reflect_field_vector,
    vector_from_scalar_and_real_direction,
    vector_select,
    vector_zero,
)
from ...materials import normalized_override_material, reflection_material_omega, resolve_surface_material
from ....utils.drjit_ops import (
    ArrayInit,
    Broadcast,
    broadcast_point,
    broadcast_vector,
    repeat_complex,
    repeat_float,
    repeat_int,
)
from .angles import _normalize_in_wedge_plane
from ....utils.geometry import reflect_point_across_plane
from .visibility import (
    _segment_visibility_mask,
    _triangle_surface_intersection,
)


def _point_source_field(source_pos, source_weight, target_pos, wavelength, k):
    width = dr.width(target_pos.x)
    source_pos_b = broadcast_point(source_pos, width)
    distance = dr.norm(target_pos - source_pos_b) + EPS
    phase = dr.exp(wt.Complex2f(0, -wt.Float(k) * distance))
    source_w = repeat_complex(source_weight, width)
    fspl = wt.Float(wavelength / (4.0 * math.pi)) / distance
    return source_w * fspl * phase


def _point_source_field_normal_derivative(source_pos, source_weight, target_pos, normal_dir, wavelength, k):
    width = dr.width(target_pos.x)
    source_pos_b = broadcast_point(source_pos, width)
    normal_b = broadcast_vector(normal_dir, width)
    offset = target_pos - source_pos_b
    distance = dr.norm(offset) + EPS
    ray_hat = offset / distance
    projection = dr.dot(ray_hat, normal_b)
    field = _point_source_field(source_pos, source_weight, target_pos, wavelength, k)
    scale = wt.Complex2f(
        -projection / distance,
        -projection * wt.Float(k),
    )
    return field * scale


def _coerce_material_override(material_override, *, reflection_gain: float):
    return normalized_override_material(
        material_override,
        reflection_coef=reflection_gain,
        eta_r=5.0,
        sigma=0.0,
    )


def _reflection_material_params(material_override, reflection_gain: float = 1.0):
    material = _coerce_material_override(material_override, reflection_gain=reflection_gain)
    if material is None:
        return 5.0, 0.0, float(reflection_gain)
    return (
        float(material["relative_permittivity"]),
        float(material["conductivity"]),
        float(material["gain"]),
    )


def _reflection_uses_fresnel(material_override=None) -> bool:
    del material_override
    return True


def _surface_reflection_material_inputs(
    scene,
    prim_idx,
    material_override,
    reflection_coef,
    valid_mask=None,
    use_scene_materials=False,
):
    normalized_material = _coerce_material_override(material_override, reflection_gain=reflection_coef)
    return resolve_surface_material(
        scene=scene,
        prim_idx=prim_idx,
        override_material=normalized_material,
        reflection_coef=float(reflection_coef),
        default_eta_r=5.0,
        default_sigma=0.0,
        valid_mask=valid_mask,
        use_scene_materials=use_scene_materials,
    )


def _edge_face_material_inputs(
    edge_state,
    width,
    material_detail,
    *,
    scene=None,
    reflection_coef=1.0,
    use_scene_materials=False,
):
    if "face0_eta_r" in edge_state and "face1_eta_r" in edge_state:
        return (
            {
                "eta_r": repeat_float(edge_state["face0_eta_r"], width),
                "sigma": repeat_float(edge_state["face0_sigma"], width),
                "gain": repeat_float(edge_state["face0_gain"], width),
                "use_fresnel": dr.full(wt.Bool, True, width),
            },
            {
                "eta_r": repeat_float(edge_state["face1_eta_r"], width),
                "sigma": repeat_float(edge_state["face1_sigma"], width),
                "gain": repeat_float(edge_state["face1_gain"], width),
                "use_fresnel": dr.full(wt.Bool, True, width),
            },
        )

    if material_detail is None and (scene is None or not use_scene_materials):
        default_material = {
            "eta_r": dr.full(wt.Float, 5.0, width),
            "sigma": dr.zeros(wt.Float, width),
            "gain": dr.full(wt.Float, float(reflection_coef), width),
            "use_fresnel": dr.full(wt.Bool, True, width),
        }
        return default_material, dict(default_material)

    adjacent_face0 = repeat_int(edge_state["adjacent_face0"], width)
    adjacent_face1 = repeat_int(edge_state["adjacent_face1"], width)
    face0 = _surface_reflection_material_inputs(
        scene,
        adjacent_face0,
        material_detail,
        reflection_coef,
        valid_mask=adjacent_face0 >= 0,
        use_scene_materials=use_scene_materials,
    )
    face1 = _surface_reflection_material_inputs(
        scene,
        adjacent_face1,
        material_detail,
        reflection_coef,
        valid_mask=adjacent_face1 >= 0,
        use_scene_materials=use_scene_materials,
    )
    return face0, face1


def _surface_reflection_coefficient(
    *,
    incident_dir,
    normal,
    scene,
    prim_idx,
    material_override,
    reflection_coef,
    wavelength,
    tx_polarization=(1.0, 0.0, 0.0),
    valid_mask=None,
    use_scene_materials=False,
):
    material_inputs = _surface_reflection_material_inputs(
        scene,
        prim_idx,
        material_override,
        reflection_coef,
        valid_mask=valid_mask,
        use_scene_materials=use_scene_materials,
    )
    del tx_polarization
    incident_hat = incident_dir / dr.maximum(dr.norm(incident_dir), wt.Float(EPS))
    normal_hat = normal / dr.maximum(dr.norm(normal), wt.Float(EPS))
    cos_theta = dr.clip(dr.abs(dr.dot(incident_hat, normal_hat)), wt.Float(SMALL_EPS), wt.Float(1.0))
    fresnel_coeff = scalar_fresnel_reflection(
        cos_theta=cos_theta,
        eta_r=material_inputs["eta_r"],
        sigma=material_inputs["sigma"],
        omega=reflection_material_omega(wavelength),
        gain=material_inputs["gain"],
    )
    return fresnel_coeff, material_inputs


def _diagonal_face_operator(coeff):
    width = dr.width(coeff.real)
    zero = wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))
    return {
        "m00": coeff,
        "m01": zero,
        "m10": zero,
        "m11": coeff,
    }


def _surface_reflection_operator(
    *,
    incident_dir,
    normal,
    scene,
    prim_idx,
    material_override,
    reflection_coef,
    wavelength,
    tx_polarization=(1.0, 0.0, 0.0),
    valid_mask=None,
    use_scene_materials=False,
):
    material_inputs = _surface_reflection_material_inputs(
        scene,
        prim_idx,
        material_override,
        reflection_coef,
        valid_mask=valid_mask,
        use_scene_materials=use_scene_materials,
    )
    incident_hat = incident_dir / dr.maximum(dr.norm(incident_dir), wt.Float(EPS))
    normal_hat = normal / dr.maximum(dr.norm(normal), wt.Float(EPS))
    cos_theta = dr.clip(dr.abs(dr.dot(incident_hat, normal_hat)), wt.Float(SMALL_EPS), wt.Float(1.0))
    eta = complex_relative_permittivity(
        material_inputs["eta_r"],
        material_inputs["sigma"],
        reflection_material_omega(wavelength),
    )
    r_te, r_tm = fresnel_reflection(cos_theta, eta)
    if __debug__ and not bool(dr.flag(dr.JitFlag.Recording)):
        _te_bad = dr.any(~dr.isfinite(r_te.real) | ~dr.isfinite(r_te.imag))
        _tm_bad = dr.any(~dr.isfinite(r_tm.real) | ~dr.isfinite(r_tm.imag))
        if dr.hint(_te_bad | _tm_bad, mode="scalar"):
            import warnings
            warnings.warn("fresnel_reflection: non-finite coefficient detected", stacklevel=2)
    r_te = wt.Complex2f(
        dr.select(dr.isfinite(r_te.real), r_te.real, wt.Float(0.0)),
        dr.select(dr.isfinite(r_te.imag), r_te.imag, wt.Float(0.0)),
    )
    r_tm = wt.Complex2f(
        dr.select(dr.isfinite(r_tm.real), r_tm.real, wt.Float(0.0)),
        dr.select(dr.isfinite(r_tm.imag), r_tm.imag, wt.Float(0.0)),
    )
    del reflection_coef
    gain = wt.Complex2f(material_inputs["gain"], wt.Float(0.0))
    return jones_operator_diagonal(gain * r_te, gain * r_tm)


def _edge_face_reflection_coefficients(
    edge_state,
    width,
    material_detail,
    wavelength,
    scene=None,
    reflection_coef=1.0,
    use_scene_materials=False,
    tx_polarization=(1.0, 0.0, 0.0),
):
    if "r_face0" in edge_state and "r_face_n" in edge_state:
        return repeat_complex(edge_state["r_face0"], width), repeat_complex(edge_state["r_face_n"], width)

    if material_detail is None and (scene is None or not use_scene_materials):
        pec_coeff = wt.Complex2f(-1.0, 0.0)
        return repeat_complex(pec_coeff, width), repeat_complex(pec_coeff, width)

    source_pos = broadcast_point(edge_state["source_pos"], width)
    edge_pos = broadcast_point(edge_state["edge_pos"], width)
    edge_dir = broadcast_vector(edge_state["edge_dir"], width)
    n0 = broadcast_vector(edge_state["n0"], width)
    nn = broadcast_vector(edge_state["n_face_n"], width)

    incoming = edge_pos - source_pos
    incoming = incoming / (dr.norm(incoming) + EPS)
    incoming_proj = _normalize_in_wedge_plane(incoming, edge_dir)
    n0_proj = _normalize_in_wedge_plane(n0, edge_dir)
    nn_proj = _normalize_in_wedge_plane(nn, edge_dir)

    adjacent_face0 = repeat_int(edge_state["adjacent_face0"], width)
    adjacent_face1 = repeat_int(edge_state["adjacent_face1"], width)
    r0, _ = _surface_reflection_coefficient(
        incident_dir=incoming_proj,
        normal=n0_proj,
        scene=scene,
        prim_idx=adjacent_face0,
        material_override=material_detail,
        reflection_coef=reflection_coef,
        wavelength=wavelength,
        tx_polarization=tx_polarization,
        valid_mask=adjacent_face0 >= 0,
        use_scene_materials=use_scene_materials,
    )
    rn, _ = _surface_reflection_coefficient(
        incident_dir=incoming_proj,
        normal=nn_proj,
        scene=scene,
        prim_idx=adjacent_face1,
        material_override=material_detail,
        reflection_coef=reflection_coef,
        wavelength=wavelength,
        tx_polarization=tx_polarization,
        valid_mask=adjacent_face1 >= 0,
        use_scene_materials=use_scene_materials,
    )
    return r0, rn


def _edge_face_reflection_operators(
    edge_state,
    width,
    material_detail,
    wavelength,
    scene=None,
    reflection_coef=1.0,
    use_scene_materials=False,
    tx_polarization=(1.0, 0.0, 0.0),
):
    if "face0_operator_m00" in edge_state and "face1_operator_m00" in edge_state:
        face0 = {
            "m00": repeat_complex(edge_state["face0_operator_m00"], width),
            "m01": repeat_complex(edge_state["face0_operator_m01"], width),
            "m10": repeat_complex(edge_state["face0_operator_m10"], width),
            "m11": repeat_complex(edge_state["face0_operator_m11"], width),
        }
        face1 = {
            "m00": repeat_complex(edge_state["face1_operator_m00"], width),
            "m01": repeat_complex(edge_state["face1_operator_m01"], width),
            "m10": repeat_complex(edge_state["face1_operator_m10"], width),
            "m11": repeat_complex(edge_state["face1_operator_m11"], width),
        }
        return face0, face1

    if material_detail is None and (scene is None or not use_scene_materials):
        pec = wt.Complex2f(-1.0, 0.0)
        return _diagonal_face_operator(repeat_complex(pec, width)), _diagonal_face_operator(
            repeat_complex(pec, width)
        )

    source_pos = broadcast_point(edge_state["source_pos"], width)
    edge_pos = broadcast_point(edge_state["edge_pos"], width)
    edge_dir = broadcast_vector(edge_state["edge_dir"], width)
    n0 = broadcast_vector(edge_state["n0"], width)
    nn = broadcast_vector(edge_state["n_face_n"], width)

    incoming = edge_pos - source_pos
    incoming = incoming / (dr.norm(incoming) + EPS)
    incoming_proj = _normalize_in_wedge_plane(incoming, edge_dir)
    n0_proj = _normalize_in_wedge_plane(n0, edge_dir)
    nn_proj = _normalize_in_wedge_plane(nn, edge_dir)
    adjacent_face0 = repeat_int(edge_state["adjacent_face0"], width)
    adjacent_face1 = repeat_int(edge_state["adjacent_face1"], width)
    face0 = _surface_reflection_operator(
        incident_dir=incoming_proj,
        normal=n0_proj,
        scene=scene,
        prim_idx=adjacent_face0,
        material_override=material_detail,
        reflection_coef=reflection_coef,
        wavelength=wavelength,
        tx_polarization=tx_polarization,
        valid_mask=adjacent_face0 >= 0,
        use_scene_materials=use_scene_materials,
    )
    face1 = _surface_reflection_operator(
        incident_dir=incoming_proj,
        normal=nn_proj,
        scene=scene,
        prim_idx=adjacent_face1,
        material_override=material_detail,
        reflection_coef=reflection_coef,
        wavelength=wavelength,
        tx_polarization=tx_polarization,
        valid_mask=adjacent_face1 >= 0,
        use_scene_materials=use_scene_materials,
    )
    return face0, face1


def _evaluate_reflection_prefix_chain(
    image_source,
    target_pos,
    chain_prim_indices,
    scene,
    target_adjacent_faces,
    material_override,
    reflection_gain,
    use_scene_materials,
    wavelength,
    k,
    tx_polarization=(1.0, 0.0, 0.0),
):
    width = dr.width(target_pos.x)
    if scene is None or scene.tri_data_gpu is None or len(chain_prim_indices) == 0:
        return (
            dr.full(wt.Bool, False, width),
            wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width)),
            vector_zero(width),
        )

    current_source = broadcast_point(image_source, width)
    current_target = target_pos
    hit_points_rev = []
    normals_rev = []
    prim_idx_rev = []
    valid = dr.full(wt.Bool, True, width)

    for prim_idx in reversed(chain_prim_indices):
        segment_valid, hit_p, geom_n, resolved_prim_idx = _triangle_surface_intersection(
            current_source, current_target, prim_idx, scene
        )
        valid = valid & segment_valid
        hit_points_rev.append(hit_p)
        normals_rev.append(geom_n)
        prim_idx_rev.append(resolved_prim_idx)
        current_target = hit_p
        current_source = reflect_point_across_plane(current_source, hit_p, geom_n)

    tx_pos = current_source
    hit_points = list(reversed(hit_points_rev))
    normals = list(reversed(normals_rev))
    prim_indices = list(reversed(prim_idx_rev))

    prev_point = tx_pos
    for slot, hit_p in enumerate(hit_points):
        ignore_list = [prim_indices[slot]]
        if slot > 0:
            ignore_list.append(prim_indices[slot - 1])
        valid = valid & _segment_visibility_mask(prev_point, hit_p, scene, ignore_prim_idx=tuple(ignore_list))
        prev_point = hit_p

    last_ignore = list(target_adjacent_faces)
    if len(prim_indices) > 0:
        last_ignore.append(prim_indices[-1])
    valid = valid & _segment_visibility_mask(prev_point, target_pos, scene, ignore_prim_idx=tuple(last_ignore))

    gain = float(reflection_gain)
    chain_weight = wt.Complex2f(dr.ones(wt.Float, width), dr.zeros(wt.Float, width))
    first_segment_dir = (hit_points[0] - tx_pos) / (dr.norm(hit_points[0] - tx_pos) + EPS)
    chain_vector = vector_from_scalar_and_real_direction(
        wt.Complex2f(dr.ones(wt.Float, width), dr.zeros(wt.Float, width)),
        project_real_polarization_to_ray(tx_polarization, first_segment_dir),
    )

    prev_point = tx_pos
    omega = reflection_material_omega(wavelength)
    for hit_p, geom_n, prim_idx in zip(hit_points, normals, prim_indices):
        incoming = hit_p - prev_point
        incoming = incoming / (dr.norm(incoming) + EPS)
        scalar_r, material_inputs = _surface_reflection_coefficient(
            incident_dir=incoming,
            normal=geom_n,
            scene=scene,
            prim_idx=prim_idx,
            material_override=material_override,
            reflection_coef=gain,
            wavelength=wavelength,
            tx_polarization=tx_polarization,
            valid_mask=valid,
            use_scene_materials=use_scene_materials,
        )
        chain_weight = chain_weight * scalar_r
        chain_vector = reflect_field_vector(
            chain_vector,
            incoming,
            geom_n,
            eta_r=material_inputs["eta_r"],
            sigma=material_inputs["sigma"],
            omega=omega,
            gain=material_inputs["gain"],
        )
        prev_point = hit_p

    chain_weight = dr.select(valid, chain_weight, ArrayInit.complex_zero(width))
    chain_vector = vector_select(valid, chain_vector, vector_zero(width))
    return valid, chain_weight, chain_vector


__all__ = [
    "_point_source_field",
    "_point_source_field_normal_derivative",
    "_coerce_material_override",
    "_reflection_material_params",
    "_reflection_uses_fresnel",
    "_surface_reflection_material_inputs",
    "_edge_face_material_inputs",
    "_surface_reflection_coefficient",
    "_diagonal_face_operator",
    "_surface_reflection_operator",
    "_edge_face_reflection_coefficients",
    "_edge_face_reflection_operators",
    "_evaluate_reflection_prefix_chain",
]
