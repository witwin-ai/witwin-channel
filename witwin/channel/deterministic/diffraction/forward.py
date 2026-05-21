"""Forward field evaluation and suffix tracing for deterministic diffraction."""

import math

import drjit as dr

from witwin.channel.deterministic import types as wt

from ..config import coerce_diffraction_execution
from ..kernels.suffix_grid import native_impl as suffix_grid_native
from ..kernels.utd import utd_pair_vectors
from witwin.channel.core.runtime import Material, Tx, Wave
from ..reflection.detail import coerce_material_context
from witwin.channel.core.numerics.constants import DIFFRACTION_MIN_DISTANCE, EPS, RAY_ORIGIN_BIAS, SPEED_OF_LIGHT, UTD_SLOPE_DERIVATIVE_STEP
from witwin.channel.core.numerics.arrays import (
    broadcast_point,
    broadcast_vector,
    complex_abs_sqr,
    complex_zero,
)
from witwin.channel.core.numerics.arrays import repeat_complex, repeat_float, repeat_int
from witwin.channel.core.physics.polarization import apply_jones_operator, diffraction_edge_basis, jones_add, jones_from_vector, jones_operator_mask_detach, jones_operator_mask_zero, jones_scale, project_real_polarization_to_ray, reflect_field_vector, vector_add, vector_from_jones, vector_from_scalar, vector_scale, vector_select, vector_zero
from witwin.channel.core.geometry.diffraction import cotangent_pole_safe_mask, slope_derivative_safe_mask, wedge_exterior_mask, wedge_geometry
from witwin.channel.core.geometry.raygen import generate_circle_directions, generate_sphere_directions
from witwin.channel.core.physics.wave_math import cot, fresnel_integral, f_utd
from .state import Geo, SOURCE_TYPE_DIRECT_TX
from .state import State
from .math import ABranch, DiffAngle, UTDMath, OPERATOR_KEYS


class ForwardEval(UTDMath):

    def pair_face_ops(edge_state, *, width, wave: Wave, material: Material | None, wedge_n, phi, phi_prime, incoming_edge_basis, outgoing_edge_basis, n0, nn):
        """Compute per-face Jones reflection operators for a diffraction edge.

    Uses precomputed face material inputs when available (requires Wave).
    Falls back to pre-stored diagonal operators from the state when Wave
    is not provided (e.g. hand-built test states).
    """
        if not Geo.has_face_material_params(edge_state) or wave is None:
            face0_op = {key: repeat_complex(edge_state[f'face0_operator_{key}'], width) for key in OPERATOR_KEYS}
            face1_op = {key: repeat_complex(edge_state[f'face1_operator_{key}'], width) for key in OPERATOR_KEYS}
            return (face0_op, face1_op)
        incoming_hat = incoming_edge_basis['k']
        outgoing_hat = outgoing_edge_basis['k']
        face0_material, face1_material = Geo.face_material_inputs(edge_state, width, material or Material())
        face0_operator = Geo.face_operator(face0_material, cos_theta=dr.clip(dr.abs(dr.sin(phi_prime)), wt.Float(1e-06), wt.Float(1.0)), normal=n0, incoming_hat=incoming_hat, outgoing_hat=outgoing_hat, incoming_edge_basis=incoming_edge_basis, outgoing_edge_basis=outgoing_edge_basis, wave=wave)
        face1_operator = Geo.face_operator(face1_material, cos_theta=dr.clip(dr.abs(dr.sin(wedge_n * dr.pi - phi)), wt.Float(1e-06), wt.Float(1.0)), normal=nn, incoming_hat=incoming_hat, outgoing_hat=outgoing_hat, incoming_edge_basis=incoming_edge_basis, outgoing_edge_basis=outgoing_edge_basis, wave=wave)
        return (face0_operator, face1_operator)

    def smooth01(x):
        clamped = dr.clip(x, wt.Float(0.0), wt.Float(1.0))
        return clamped * clamped * (wt.Float(3.0) - wt.Float(2.0) * clamped)

    def direct_source_mask(edge_state, *, width):
        if 'source_type_code' not in edge_state:
            return dr.zeros(wt.Bool, width)
        return repeat_int(edge_state['source_type_code'], width) == wt.UInt32(SOURCE_TYPE_DIRECT_TX)

    def direct_first_order_mask(edge_state, *, width):
        direct_mask = ForwardEval.direct_source_mask(edge_state, width=width)
        if 'order' not in edge_state:
            return direct_mask
        return direct_mask & (repeat_int(edge_state['order'], width) == wt.UInt32(1))

    def incident_vectors(edge_state, *, width, edge_pos=None, wave: Wave | None = None, tx: Tx | None = None):
        """Broadcast canonicalized jones+basis state into incident field vectors."""
        incident_path_basis = {'u': broadcast_vector(edge_state['incident_basis_u'], width), 'v': broadcast_vector(edge_state['incident_basis_v'], width), 'k': broadcast_vector(edge_state['incident_basis_k'], width)}
        incident_jones = {'u': repeat_complex(edge_state['incident_jones_u'], width), 'v': repeat_complex(edge_state['incident_jones_v'], width)}
        incident_derivative_jones = {'u': repeat_complex(edge_state['incident_derivative_jones_u'], width), 'v': repeat_complex(edge_state['incident_derivative_jones_v'], width)}
        incident_vector = vector_from_jones(incident_jones, incident_path_basis)
        incident_normal_derivative_vector = vector_from_jones(incident_derivative_jones, incident_path_basis)
        if wave is None or tx is None:
            return incident_vector, incident_normal_derivative_vector
        direct_mask = ForwardEval.direct_first_order_mask(edge_state, width=width)
        if not dr.any(direct_mask):
            return incident_vector, incident_normal_derivative_vector
        edge_pos_b = broadcast_point(edge_state['edge_pos'] if edge_pos is None else edge_pos, width)
        source_pos_b = broadcast_point(edge_state['source_pos'], width)
        direct_field = Geo.source_field(source_pos_b, wt.Complex2f(1.0, 0.0), edge_pos_b, wave)
        ray_dir = (edge_pos_b - source_pos_b) / (dr.norm(edge_pos_b - source_pos_b) + wt.Float(EPS))
        pol_dir = project_real_polarization_to_ray(tx.polarization, ray_dir)
        direct_vector = vector_from_scalar(direct_field, pol_dir)
        direct_derivative_vector = vector_zero(width)
        return (
            vector_select(direct_mask, direct_vector, incident_vector),
            vector_select(direct_mask, direct_derivative_vector, incident_normal_derivative_vector),
        )

    def target_payload(field, *, normal_derivative=None, vector_field=None, normal_derivative_vector=None, field_valid=None, return_normal_derivative: bool=False, return_vector: bool=False, return_valid: bool=False):
        items = (field,)
        if return_normal_derivative:
            items += (normal_derivative,)
        if return_vector:
            items += (vector_field, normal_derivative_vector) if return_normal_derivative else (vector_field,)
        if return_valid:
            items += (field_valid,)
        return items if len(items) > 1 else field

    def truncation_factor_with_bounds(edge_state, edge_geometry, target_pos, wave: Wave, *, width, line_min, line_max, stationary_u=None):
        edge_pos_b = broadcast_point(edge_state['edge_pos'], width)
        source_pos_b = broadcast_point(edge_state['source_pos'], width)
        edge_hat = edge_geometry.edge_hat
        source_axial = dr.dot(source_pos_b - edge_pos_b, edge_hat)
        target_axial = dr.dot(target_pos - edge_pos_b, edge_hat)
        s_prime_proj = edge_geometry.s_prime_proj
        s_proj = edge_geometry.s_proj
        if stationary_u is None:
            stationary_u = (s_prime_proj * target_axial + s_proj * source_axial) / (s_proj + s_prime_proj + wt.Float(EPS))
        else:
            stationary_u = repeat_float(stationary_u, width)
        source_offset = stationary_u - source_axial
        target_offset = target_axial - stationary_u
        source_range = dr.sqrt(s_prime_proj * s_prime_proj + source_offset * source_offset + wt.Float(EPS))
        target_range = dr.sqrt(s_proj * s_proj + target_offset * target_offset + wt.Float(EPS))
        curvature = s_prime_proj * s_prime_proj / (source_range * source_range * source_range + wt.Float(EPS)) + s_proj * s_proj / (target_range * target_range * target_range + wt.Float(EPS))
        scale = dr.sqrt(dr.maximum(wave.k * curvature, wt.Float(EPS)) / dr.pi)
        delta_f = fresnel_integral(scale * (line_max - stationary_u)) - fresnel_integral(scale * (line_min - stationary_u))
        return wt.Complex2f(0.5, 0.5) * dr.conj(delta_f)

    def truncation_factor(edge_state, edge_geometry, target_pos, wave: Wave, *, width):
        edge_line_min, edge_line_max = Geo.state_line_bounds(edge_state, context='_finite_wedge_truncation_factor')
        return ForwardEval.truncation_factor_with_bounds(
            edge_state,
            edge_geometry,
            target_pos,
            wave,
            width=width,
            line_min=repeat_float(edge_line_min, width),
            line_max=repeat_float(edge_line_max, width),
        )

    def stationary_completion_factor(edge_state, edge_geometry, target_pos, wave: Wave, *, width, inside):
        edge_line_min, edge_line_max = Geo.state_line_bounds(edge_state, context='_finite_wedge_stationary_completion_factor')
        line_min = repeat_float(edge_line_min, width)
        line_max = repeat_float(edge_line_max, width)
        edge_length = line_max - line_min
        outside_distance = dr.maximum(
            dr.maximum(line_min, -line_max),
            wt.Float(0.0),
        )
        taper_length = dr.minimum(
            wt.Float(0.25) * edge_length,
            dr.maximum(wt.Float(0.5) * wave.wavelength, wt.Float(EPS)),
        )
        endpoint_u = dr.maximum(
            outside_distance / dr.maximum(taper_length, wt.Float(EPS)),
            wt.Float(0.0),
        )
        endpoint_weight = dr.exp(-endpoint_u * endpoint_u)
        raw = ForwardEval.truncation_factor_with_bounds(
            edge_state,
            edge_geometry,
            target_pos,
            wave,
            width=width,
            line_min=line_min,
            line_max=line_max,
            stationary_u=wt.Float(0.0),
        )
        left_boundary = ForwardEval.truncation_factor_with_bounds(
            edge_state,
            edge_geometry,
            target_pos,
            wave,
            width=width,
            line_min=dr.zeros(wt.Float, width),
            line_max=edge_length,
            stationary_u=wt.Float(0.0),
        )
        right_boundary = ForwardEval.truncation_factor_with_bounds(
            edge_state,
            edge_geometry,
            target_pos,
            wave,
            width=width,
            line_min=-edge_length,
            line_max=dr.zeros(wt.Float, width),
            stationary_u=wt.Float(0.0),
        )
        boundary = dr.select(line_min >= wt.Float(0.0), left_boundary, right_boundary)
        boundary_power = complex_abs_sqr(boundary)
        normalized = raw / dr.select(
            boundary_power > wt.Float(EPS),
            boundary,
            wt.Complex2f(1.0, 0.0),
        )
        completed = dr.select(
            boundary_power > wt.Float(EPS),
            normalized,
            raw,
        )
        completed = completed * endpoint_weight
        return dr.select(inside, wt.Complex2f(1.0, 0.0), completed)

    def target_support(
        edge_state,
        target_pos,
        *,
        scene=None,
        smooth_exterior_shadow=False,
        select_diffraction_point: bool = True,
        enable_segment_visibility: bool = True,
    ):
        width = dr.width(target_pos.x)
        eval_edge_state = dict(edge_state)
        if select_diffraction_point:
            selected_point = Geo.finite_edge_diffraction_point(edge_state, target_pos)
            anchor_point = broadcast_point(edge_state['edge_pos'], width)
            anchor_line_min = repeat_float(edge_state['edge_line_min'], width)
            anchor_line_max = repeat_float(edge_state['edge_line_max'], width)
            select_mask = ForwardEval.direct_first_order_mask(edge_state, width=width)
            diffraction_point = {
                'point': dr.select(select_mask, selected_point['point'], anchor_point),
                'edge_line_min': dr.select(select_mask, selected_point['edge_line_min'], anchor_line_min),
                'edge_line_max': dr.select(select_mask, selected_point['edge_line_max'], anchor_line_max),
                'valid': dr.select(select_mask, selected_point['valid'], dr.full(wt.Bool, True, width)),
                'inside': dr.select(select_mask, selected_point['inside'], dr.full(wt.Bool, True, width)),
                'visibility_point': dr.select(select_mask, selected_point['visibility_point'], anchor_point),
            }
            eval_edge_state['edge_pos'] = diffraction_point['point']
            eval_edge_state['edge_line_min'] = diffraction_point['edge_line_min']
            eval_edge_state['edge_line_max'] = diffraction_point['edge_line_max']
        else:
            diffraction_point = {
                'point': broadcast_point(edge_state['edge_pos'], width),
                'edge_line_min': repeat_float(edge_state['edge_line_min'], width),
                'edge_line_max': repeat_float(edge_state['edge_line_max'], width),
                'valid': dr.full(wt.Bool, True, width),
                'inside': dr.full(wt.Bool, True, width),
                'visibility_point': broadcast_point(edge_state['edge_pos'], width),
            }

        edge_geometry = wedge_geometry(eval_edge_state['source_pos'], eval_edge_state['edge_pos'], eval_edge_state['edge_dir'], eval_edge_state['n0'], target_pos)
        phi = edge_geometry.phi
        phi_prime = edge_geometry.phi_prime
        s = edge_geometry.s
        s_prime = edge_geometry.s_prime
        wedge_n = repeat_float(eval_edge_state['wedge_n'], width)
        edge_pos_b = broadcast_point(eval_edge_state['edge_pos'], width)
        edge_dir_b = broadcast_vector(eval_edge_state['edge_dir'], width)
        n0_b = broadcast_vector(eval_edge_state['n0'], width)
        nn_b = broadcast_vector(eval_edge_state['n_face_n'], width)
        source_pos_b = broadcast_point(eval_edge_state['source_pos'], width)
        source_exterior = wedge_exterior_mask(source_pos_b - edge_pos_b, edge_dir_b, n0_b, nn_b)
        target_exterior = wedge_exterior_mask(target_pos - edge_pos_b, edge_dir_b, n0_b, nn_b)
        visibility_valid = dr.full(wt.Bool, True, width)
        if scene is not None and enable_segment_visibility:
            adjacent_face0 = eval_edge_state.get('adjacent_face0')
            adjacent_face1 = eval_edge_state.get('adjacent_face1')
            ignore_prim_idx = (
                None if adjacent_face0 is None or adjacent_face1 is None
                else (adjacent_face0, adjacent_face1)
            )
            ignore_surface_group_idx = (
                None if adjacent_face0 is None or adjacent_face1 is None
                else (scene.triangle_group_id(adjacent_face0), scene.triangle_group_id(adjacent_face1))
            )
            target_visible = scene.segment_visible(
                diffraction_point['visibility_point'],
                target_pos,
                ignore_prim_idx=ignore_prim_idx,
                ignore_surface_group_idx=ignore_surface_group_idx,
            )
            direct_mask = ForwardEval.direct_source_mask(eval_edge_state, width=width)
            source_visible = scene.segment_visible(
                source_pos_b,
                diffraction_point['visibility_point'],
                ignore_prim_idx=ignore_prim_idx,
            )
            visibility_valid = target_visible & (~direct_mask | source_visible)
        base_valid = diffraction_point['valid'] & visibility_valid & source_exterior & (s_prime > DIFFRACTION_MIN_DISTANCE) & (s > DIFFRACTION_MIN_DISTANCE)
        interior_mask = dr.zeros(wt.Bool, width)
        if smooth_exterior_shadow:
            interior_mask = scene.point_inside_closed_mesh(
                target_pos, robust=True, active=base_valid & ~target_exterior,
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
                boundary_eps = wt.Float(0.001)
                shadow_opening_angle = dr.maximum(wt.Float(2.0 * dr.pi) - wedge_n * dr.pi, wt.Float(2.0) * boundary_eps)
                shadow_angle_ratio = dr.minimum(shadow_opening_angle / dr.pi, wt.Float(1.0))
                shadow_decay_span = dr.maximum((wt.Float(0.17) + wt.Float(0.12) * shadow_angle_ratio) * shadow_opening_angle, wt.Float(8.0) * boundary_eps)
                shadow_decay_span = dr.minimum(shadow_decay_span, wt.Float(0.5) * shadow_opening_angle)
                shadow_decay_power = wt.Float(2.0) + wt.Float(1.0) * (wt.Float(1.0) - shadow_angle_ratio)
                shadow_anchor_offset = dr.maximum((wt.Float(0.01) + wt.Float(0.01) * shadow_angle_ratio) * shadow_opening_angle, wt.Float(4.0) * boundary_eps)
                shadow_anchor_offset = dr.minimum(shadow_anchor_offset, wt.Float(0.25) * shadow_opening_angle)
                shadow_anchor_outer_offset = dr.maximum(dr.minimum(wt.Float(2.0) * shadow_anchor_offset, wt.Float(0.18) * shadow_opening_angle), shadow_anchor_offset + wt.Float(4.0) * boundary_eps)
                shadow_half_angle = wt.Float(0.5) * shadow_opening_angle
                illuminated_wrap_mask = target_exterior & (phi <= shadow_anchor_outer_offset)
                illuminated_face_mask = target_exterior & (phi >= dr.maximum(wedge_n * dr.pi - shadow_anchor_outer_offset, boundary_eps))
                illuminated_boundary_mask = illuminated_wrap_mask | illuminated_face_mask
                wrap_boundary = shadow_completion_mask & (phi >= wt.Float(2.0 * dr.pi) - shadow_half_angle) | illuminated_wrap_mask
                shadow_boundary_distance = dr.select(wrap_boundary, wt.Float(2.0 * dr.pi) - phi, phi - wedge_n * dr.pi)
                illuminated_boundary_distance = dr.select(wrap_boundary, phi, wedge_n * dr.pi - phi)
                shadow_boundary_u = dr.clip(shadow_boundary_distance / shadow_decay_span, wt.Float(0.0), wt.Float(1.0))
                shadow_completion_curve = wt.Float(0.88) * (wt.Float(1.0) - ForwardEval.smooth01(shadow_boundary_u)) + wt.Float(0.12) * (wt.Float(1.0) - shadow_boundary_u)
                shadow_completion_weight = dr.power(shadow_completion_curve, shadow_decay_power)
                shadow_completion_mask = shadow_completion_mask & (shadow_boundary_distance < shadow_decay_span)
                illuminated_boundary_weight = wt.Float(1.0) - ForwardEval.smooth01(illuminated_boundary_distance / shadow_anchor_outer_offset)
            field_valid = field_valid & (target_exterior | shadow_completion_mask)
        return {'width': width, 'edge_state': eval_edge_state, 'diffraction_point': diffraction_point, 'edge_geometry': edge_geometry, 'phi': phi, 'phi_prime': phi_prime, 's': s, 's_prime': s_prime, 'wedge_n': wedge_n, 'edge_pos_b': edge_pos_b, 'edge_dir_b': edge_dir_b, 'n0_b': n0_b, 'nn_b': nn_b, 'source_pos_b': source_pos_b, 'source_exterior': source_exterior, 'base_valid': base_valid, 'target_exterior': target_exterior, 'interior_mask': interior_mask, 'geometry_valid': geometry_valid, 'field_valid': field_valid, 'shadow_completion_mask': shadow_completion_mask, 'shadow_completion_weight': shadow_completion_weight, 'shadow_boundary_distance': shadow_boundary_distance, 'illuminated_boundary_mask': illuminated_boundary_mask, 'illuminated_boundary_weight': illuminated_boundary_weight, 'shadow_opening_angle': shadow_opening_angle, 'shadow_anchor_offset': shadow_anchor_offset, 'shadow_anchor_outer_offset': shadow_anchor_outer_offset, 'shadow_decay_span': shadow_decay_span, 'shadow_decay_power': shadow_decay_power}

    def to_targets(
        edge_state,
        target_pos,
        wave: Wave,
        return_normal_derivative=False,
        return_vector=False,
        return_valid=False,
        material: Material | None = None,
        scene=None,
        smooth_exterior_shadow=False,
        tx: Tx | None = None,
        select_diffraction_point: bool = True,
        enable_segment_visibility: bool = True,
    ):
        support_context = ForwardEval.target_support(
            edge_state,
            target_pos,
            scene=scene,
            smooth_exterior_shadow=smooth_exterior_shadow,
            select_diffraction_point=select_diffraction_point,
            enable_segment_visibility=enable_segment_visibility,
        )
        edge_state = support_context['edge_state']
        width = support_context['width']
        edge_geometry = support_context['edge_geometry']
        phi = support_context['phi']
        phi_prime = support_context['phi_prime']
        s = support_context['s']
        s_prime = support_context['s_prime']
        sin_beta0 = edge_geometry.sin_beta_eff
        wedge_n = support_context['wedge_n']
        edge_pos_b = support_context['edge_pos_b']
        edge_dir_b = support_context['edge_dir_b']
        n0_b = support_context['n0_b']
        nn_b = support_context['nn_b']
        source_pos_b = support_context['source_pos_b']
        target_exterior = support_context['target_exterior']
        geometry_valid = support_context['geometry_valid']
        field_valid = geometry_valid
        pole_safe = geometry_valid & cotangent_pole_safe_mask(phi, phi_prime, wedge_n, wt.Float(1e-06))
        safe_phi = dr.select(pole_safe, phi, 0.5 * wedge_n * dr.pi)
        safe_phi_prime = dr.select(pole_safe, phi_prime, 0.5 * wedge_n * dr.pi)
        step = wt.Float(UTD_SLOPE_DERIVATIVE_STEP)
        slope_safe = field_valid & slope_derivative_safe_mask(safe_phi, safe_phi_prime, wedge_n, step)
        local_scale = dr.sqrt(s_prime / (s * (s + s_prime) + EPS))
        phase = dr.exp(wt.Complex2f(0, -wave.k * s))
        finite_wedge_factor = ForwardEval.truncation_factor(edge_state, edge_geometry, target_pos, wave, width=width)
        direct_stationary_mask = (
            ForwardEval.direct_first_order_mask(edge_state, width=width)
            if select_diffraction_point
            else dr.zeros(wt.Bool, width)
        )
        stationary_completion_factor = ForwardEval.stationary_completion_factor(
            edge_state,
            edge_geometry,
            target_pos,
            wave,
            width=width,
            inside=support_context['diffraction_point']['inside'],
        )
        finite_wedge_factor = dr.select(
            direct_stationary_mask,
            stationary_completion_factor,
            finite_wedge_factor,
        )
        incoming_edge_basis = diffraction_edge_basis(edge_pos_b - source_pos_b, edge_dir_b, outgoing=False)
        outgoing_edge_basis = diffraction_edge_basis(target_pos - edge_pos_b, edge_dir_b, outgoing=True)
        incident_vector, incident_normal_derivative_vector = ForwardEval.incident_vectors(edge_state, width=width, edge_pos=edge_pos_b, wave=wave, tx=tx)
        incident_jones_edge = jones_from_vector(incident_vector, incoming_edge_basis)
        incident_derivative_jones_edge = jones_from_vector(incident_normal_derivative_vector, incoming_edge_basis)
        incident_derivative_power = complex_abs_sqr(incident_derivative_jones_edge['u']) + complex_abs_sqr(incident_derivative_jones_edge['v'])
        has_slope = (incident_derivative_power > wt.Float(1e-24)) & slope_safe
        face0_operator, face1_operator = ForwardEval.pair_face_ops(edge_state, width=width, wave=wave, material=material, wedge_n=wedge_n, phi=phi, phi_prime=phi_prime, incoming_edge_basis=incoming_edge_basis, outgoing_edge_basis=outgoing_edge_basis, n0=n0_b, nn=nn_b)
        operator_kwargs = {'wedge_n': wedge_n, 'k': wave.k, 's': s, 's_prime': s_prime, 'face0_operator': face0_operator, 'face1_operator': face1_operator, 'sin_beta0': sin_beta0 if Geo.has_face_material_params(edge_state) else None}
        need_normal_derivative_ops = bool(return_normal_derivative)
        operators = ForwardEval.assemble_material_ops(phi=phi, phi_prime=phi_prime, include_normal_derivative_ops=need_normal_derivative_ops, **operator_kwargs)
        safe_operators = ForwardEval.assemble_material_ops(phi=safe_phi, phi_prime=safe_phi_prime, include_normal_derivative_ops=need_normal_derivative_ops, **operator_kwargs)
        field_operator = jones_operator_mask_detach(operators['field'], pole_safe)
        slope_operator = jones_operator_mask_zero(safe_operators['slope'], has_slope)
        field_jones = jones_add(apply_jones_operator(incident_jones_edge, field_operator), apply_jones_operator(incident_derivative_jones_edge, slope_operator))
        field_jones = jones_scale(field_jones, finite_wedge_factor)
        scaled_field_jones = jones_scale(field_jones, local_scale * phase)
        vector_field = vector_from_jones(scaled_field_jones, outgoing_edge_basis)

        def _evaluate_boundary_sample(sample_phi, sample_phi_prime, sample_mask):
            sample_pole_safe = sample_mask & cotangent_pole_safe_mask(sample_phi, sample_phi_prime, wedge_n, wt.Float(1e-06))
            sample_safe_phi = dr.select(sample_pole_safe, sample_phi, 0.5 * wedge_n * dr.pi)
            sample_safe_phi_prime = dr.select(sample_pole_safe, sample_phi_prime, 0.5 * wedge_n * dr.pi)
            sample_slope_safe = sample_mask & slope_derivative_safe_mask(sample_safe_phi, sample_safe_phi_prime, wedge_n, step)
            sample_has_slope = (incident_derivative_power > wt.Float(1e-24)) & sample_slope_safe
            sample_face0_operator, sample_face1_operator = ForwardEval.pair_face_ops(edge_state, width=width, wave=wave, material=material, wedge_n=wedge_n, phi=sample_phi, phi_prime=sample_phi_prime, incoming_edge_basis=incoming_edge_basis, outgoing_edge_basis=outgoing_edge_basis, n0=n0_b, nn=nn_b)
            sample_operator_kwargs = {'wedge_n': wedge_n, 'k': wave.k, 's': s, 's_prime': s_prime, 'face0_operator': sample_face0_operator, 'face1_operator': sample_face1_operator, 'sin_beta0': sin_beta0 if Geo.has_face_material_params(edge_state) else None}
            sample_operators = ForwardEval.assemble_material_ops(phi=sample_phi, phi_prime=sample_phi_prime, include_normal_derivative_ops=need_normal_derivative_ops, **sample_operator_kwargs)
            sample_safe_operators = ForwardEval.assemble_material_ops(phi=sample_safe_phi, phi_prime=sample_safe_phi_prime, include_normal_derivative_ops=need_normal_derivative_ops, **sample_operator_kwargs)
            sample_field_operator = jones_operator_mask_detach(sample_operators['field'], sample_pole_safe)
            sample_slope_operator = jones_operator_mask_zero(sample_safe_operators['slope'], sample_has_slope)
            sample_field_jones = jones_add(apply_jones_operator(incident_jones_edge, sample_field_operator), apply_jones_operator(incident_derivative_jones_edge, sample_slope_operator))
            sample_field_jones = jones_scale(sample_field_jones, finite_wedge_factor)
            sample_scaled_field_jones = jones_scale(sample_field_jones, local_scale * phase)
            sample_vector = vector_from_jones(sample_scaled_field_jones, outgoing_edge_basis)
            sample_result = {'field_jones': sample_scaled_field_jones, 'vector': sample_vector}
            if return_normal_derivative:
                sample_field_dphi_operator = jones_operator_mask_zero(sample_safe_operators['field_dphi'], sample_slope_safe)
                sample_slope_dphi_operator = jones_operator_mask_zero(sample_safe_operators['slope_dphi'], sample_has_slope)
                sample_normal_jones = jones_add(apply_jones_operator(incident_jones_edge, sample_field_dphi_operator), apply_jones_operator(incident_derivative_jones_edge, sample_slope_dphi_operator))
                sample_normal_jones = jones_scale(sample_normal_jones, finite_wedge_factor)
                sample_scaled_normal_jones = jones_scale(sample_normal_jones, local_scale * phase / (s + EPS))
                sample_result['normal_jones'] = sample_scaled_normal_jones
                sample_result['normal_vector'] = vector_from_jones(sample_scaled_normal_jones, outgoing_edge_basis)
            return sample_result
        shadow_completion_mask = support_context['shadow_completion_mask']
        shadow_completion_weight = support_context['shadow_completion_weight']
        illuminated_boundary_mask = support_context['illuminated_boundary_mask']
        illuminated_boundary_weight = support_context['illuminated_boundary_weight']
        shadow_opening_angle = support_context['shadow_opening_angle']
        shadow_anchor_offset = support_context['shadow_anchor_offset']
        shadow_anchor_outer_offset = support_context['shadow_anchor_outer_offset']
        shadow_decay_span = support_context['shadow_decay_span']
        shadow_decay_power = support_context['shadow_decay_power']
        shadow_completion_field = complex_zero(width)
        shadow_completion_vector = vector_zero(width)
        shadow_completion_normal = complex_zero(width)
        shadow_completion_normal_vector = vector_zero(width)
        boundary_anchor_normal = complex_zero(width)
        boundary_anchor_normal_vector = vector_zero(width)
        illuminated_boundary_field = complex_zero(width)
        illuminated_boundary_vector = vector_zero(width)
        illuminated_boundary_normal = complex_zero(width)
        illuminated_boundary_normal_vector = vector_zero(width)
        if smooth_exterior_shadow:
            boundary_anchor_mask = shadow_completion_mask | illuminated_boundary_mask
            if dr.any(boundary_anchor_mask):
                boundary_eps = wt.Float(0.001)
                shadow_half_angle = wt.Float(0.5) * shadow_opening_angle
                illuminated_wrap_mask = target_exterior & (phi <= shadow_anchor_outer_offset)
                wrap_boundary = shadow_completion_mask & (phi >= wt.Float(2.0 * dr.pi) - shadow_half_angle) | illuminated_wrap_mask
                boundary_phi_near = dr.select(wrap_boundary, shadow_anchor_offset, dr.maximum(wedge_n * dr.pi - shadow_anchor_offset, boundary_eps))
                boundary_phi_far = dr.select(wrap_boundary, shadow_anchor_outer_offset, dr.maximum(wedge_n * dr.pi - shadow_anchor_outer_offset, boundary_eps))
                boundary_phi_prime = phi_prime
                near_boundary_sample = _evaluate_boundary_sample(boundary_phi_near, boundary_phi_prime, boundary_anchor_mask)
                far_boundary_sample = _evaluate_boundary_sample(boundary_phi_far, boundary_phi_prime, boundary_anchor_mask)
                boundary_extrapolation_denom = dr.maximum(shadow_anchor_outer_offset - shadow_anchor_offset, wt.Float(4.0) * boundary_eps)
                boundary_near_weight = shadow_anchor_outer_offset / boundary_extrapolation_denom
                boundary_far_weight = shadow_anchor_offset / boundary_extrapolation_denom
                boundary_face_field_jones = jones_add(jones_scale(near_boundary_sample['field_jones'], boundary_near_weight), jones_scale(far_boundary_sample['field_jones'], -boundary_far_weight))
                boundary_anchor_vector = vector_add(vector_scale(near_boundary_sample['vector'], boundary_near_weight), vector_scale(far_boundary_sample['vector'], -boundary_far_weight))
                shadow_completion_field = boundary_face_field_jones['u'] * shadow_completion_weight
                shadow_completion_vector = vector_scale(boundary_anchor_vector, shadow_completion_weight)
                illuminated_boundary_field = scaled_field_jones['u'] * (wt.Float(1.0) - illuminated_boundary_weight) + boundary_face_field_jones['u'] * illuminated_boundary_weight
                illuminated_boundary_vector = vector_add(vector_scale(vector_field, wt.Float(1.0) - illuminated_boundary_weight), vector_scale(boundary_anchor_vector, illuminated_boundary_weight))
                if return_normal_derivative:
                    boundary_face_normal_jones = jones_add(jones_scale(near_boundary_sample['normal_jones'], boundary_near_weight), jones_scale(far_boundary_sample['normal_jones'], -boundary_far_weight))
                    boundary_anchor_normal = boundary_face_normal_jones['u']
                    shadow_completion_normal = boundary_anchor_normal * shadow_completion_weight
                    boundary_anchor_normal_vector = vector_add(vector_scale(near_boundary_sample['normal_vector'], boundary_near_weight), vector_scale(far_boundary_sample['normal_vector'], -boundary_far_weight))
                    shadow_completion_normal_vector = vector_scale(boundary_anchor_normal_vector, shadow_completion_weight)
        field_valid = field_valid & (target_exterior | shadow_completion_mask)
        slope_safe = slope_safe & field_valid
        has_slope = has_slope & field_valid
        vector_field = vector_select(field_valid, vector_field, vector_zero(width))
        field = dr.select(field_valid, scaled_field_jones['u'], complex_zero(width))
        if dr.any(shadow_completion_mask):
            field = dr.select(shadow_completion_mask, shadow_completion_field, field)
            vector_field = vector_select(shadow_completion_mask, shadow_completion_vector, vector_field)
        if dr.any(illuminated_boundary_mask):
            field = dr.select(illuminated_boundary_mask, illuminated_boundary_field, field)
            vector_field = vector_select(illuminated_boundary_mask, illuminated_boundary_vector, vector_field)
        if not return_normal_derivative:
            return ForwardEval.target_payload(
                field,
                vector_field=vector_field,
                field_valid=field_valid,
                return_vector=return_vector,
                return_valid=return_valid,
            )
        field_dphi_operator = jones_operator_mask_zero(safe_operators['field_dphi'], slope_safe)
        slope_dphi_operator = jones_operator_mask_zero(safe_operators['slope_dphi'], has_slope)
        normal_jones = jones_add(apply_jones_operator(incident_jones_edge, field_dphi_operator), apply_jones_operator(incident_derivative_jones_edge, slope_dphi_operator))
        normal_jones = jones_scale(normal_jones, finite_wedge_factor)
        scaled_normal_jones = jones_scale(normal_jones, local_scale * phase / (s + EPS))
        normal_derivative_vector = vector_from_jones(scaled_normal_jones, outgoing_edge_basis)
        normal_derivative_vector = vector_select(field_valid, normal_derivative_vector, vector_zero(width))
        normal_derivative = dr.select(field_valid, scaled_normal_jones['u'], complex_zero(width))
        if dr.any(shadow_completion_mask):
            normal_derivative = dr.select(shadow_completion_mask, shadow_completion_normal, normal_derivative)
            normal_derivative_vector = vector_select(shadow_completion_mask, shadow_completion_normal_vector, normal_derivative_vector)
        if dr.any(illuminated_boundary_mask):
            illuminated_boundary_normal = scaled_normal_jones['u'] * (wt.Float(1.0) - illuminated_boundary_weight) + boundary_anchor_normal * illuminated_boundary_weight
            illuminated_boundary_normal_vector = vector_add(vector_scale(normal_derivative_vector, wt.Float(1.0) - illuminated_boundary_weight), vector_scale(boundary_anchor_normal_vector, illuminated_boundary_weight))
            normal_derivative = dr.select(illuminated_boundary_mask, illuminated_boundary_normal, normal_derivative)
            normal_derivative_vector = vector_select(illuminated_boundary_mask, illuminated_boundary_normal_vector, normal_derivative_vector)
        return ForwardEval.target_payload(
            field,
            normal_derivative=normal_derivative,
            vector_field=vector_field,
            normal_derivative_vector=normal_derivative_vector,
            field_valid=field_valid,
            return_normal_derivative=True,
            return_vector=return_vector,
            return_valid=return_valid,
        )

    def roulette_random(state_idx, dir_idx, bounce):
        seed = state_idx * wt.UInt32(747796405) + dir_idx * wt.UInt32(2891336453) + wt.UInt32(bounce + 1) * wt.UInt32(277803737)
        seed = (seed ^ seed >> 16) * wt.UInt32(2246822519)
        seed = (seed ^ seed >> 13) * wt.UInt32(3266489917)
        seed = seed ^ seed >> 16
        mantissa = seed & wt.UInt32(16777215)
        return wt.Float(mantissa) / wt.Float(16777216.0)

    def roulette_survival_probability(reflected_field, hit):
        reflected_power = complex_abs_sqr(reflected_field)
        active_power = dr.select(hit, reflected_power, wt.Float(0.0))
        max_power = dr.max(active_power)
        safe_max_power = dr.maximum(max_power, wt.Float(1e-20))
        normalized_power = active_power / safe_max_power
        survival_probability = dr.sqrt(normalized_power)
        survival_probability = dr.maximum(survival_probability, wt.Float(0.1))
        survival_probability = dr.minimum(survival_probability, wt.Float(1.0))
        return dr.select(hit, survival_probability, wt.Float(0.0))

    def suffix_accumulator(execution):
        if execution.suffix_backend != 'native':
            raise RuntimeError("Reflected diffraction suffix accumulation requires the native backend.")
        return suffix_grid_native.accumulate_reflected_segment_fields_batched

    def trace_suffix(state_arrays, suffix, scene, wave: Wave, tx: Tx, execution=None):
        execution = coerce_diffraction_execution(execution)
        n_rx = suffix.grid.n_cells
        n_states = 0 if state_arrays is None else state_arrays['n_states']
        if scene is None or suffix.n_rays <= 0 or suffix.max_bounces <= 0 or (n_states <= 0):
            return (complex_zero(n_rx), vector_zero(n_rx))
        reflection_context = coerce_material_context(suffix.detail, default_gain=suffix.coef)
        reflection_material = Material(
            reflection_coef=reflection_context.reflection_gain,
        )
        accumulate_segment_fields = ForwardEval.suffix_accumulator(execution)
        rays_per_state = max(1, math.ceil(int(suffix.n_rays) / max(1, n_states)))
        if suffix.mode == '2d':
            base_ray_dir = generate_circle_directions(rays_per_state)
        else:
            base_ray_dir = generate_sphere_directions(rays_per_state)
        n_total_rays = n_states * rays_per_state
        ray_idx = dr.arange(wt.UInt32, n_total_rays)
        state_idx = ray_idx // rays_per_state
        dir_idx = ray_idx % rays_per_state
        batch_states = State.gather_path_export_eval(state_arrays, state_idx)
        ray_origin = batch_states['edge_pos']
        ray_dir = wt.Vector3f(dr.gather(wt.Float, base_ray_dir.x, dir_idx), dr.gather(wt.Float, base_ray_dir.y, dir_idx), dr.gather(wt.Float, base_ray_dir.z, dir_idx))
        active = dr.full(wt.Bool, True, n_total_rays)
        total = complex_zero(n_rx)
        total_vector = vector_zero(n_rx)
        tri_data = scene._triangle_runtime()
        current_field = None
        current_vector = None
        omega = wt.Float(2.0 * math.pi * SPEED_OF_LIGHT) / wave.wavelength
        for bounce in range(suffix.max_bounces):
            hit, _, hit_p, hit_n, hit_prim_idx = scene.intersect_rays_with_prim(ray_origin, ray_dir, active)
            if not dr.any(hit):
                break
            if bounce == 0:
                field_at_hit, field_at_hit_vector = utd_pair_vectors(
                    batch_states,
                    hit_p,
                    wave=wave,
                    material=reflection_material,
                    select_diffraction_point=True,
                )
            else:
                seg_len = dr.norm(hit_p - ray_origin) + EPS
                phase = dr.exp(wt.Complex2f(0, -wave.k * seg_len))
                fspl = (wave.wavelength / wt.Float(4.0 * math.pi)) / seg_len
                field_at_hit = current_field * fspl * phase
                field_at_hit_vector = vector_scale(current_vector, fspl * phase)
            field_at_hit = dr.select(hit, field_at_hit, complex_zero(n_total_rays))
            field_at_hit_vector = vector_select(hit, field_at_hit_vector, vector_zero(n_total_rays))
            safe_ray_dir = dr.select(hit, ray_dir, wt.Vector3f(1.0, 0.0, 0.0))
            safe_hit_n = dr.select(hit, hit_n, wt.Vector3f(0.0, 0.0, 1.0))
            reflection_weight, material_inputs = Geo.surface_coeff(incident_dir=ray_dir, normal=hit_n, scene=scene, prim_idx=wt.Int32(hit_prim_idx), material=reflection_material, wave=wave, tx=tx, valid_mask=hit)
            reflected_field = field_at_hit * reflection_weight
            reflected_vector = reflect_field_vector(field_at_hit_vector, safe_ray_dir, safe_hit_n, eta_r=material_inputs['eta_r'], sigma=material_inputs['sigma'], omega=omega, gain=material_inputs['gain'], mu_r=material_inputs['mu_r'])
            reflected_field = dr.select(hit, reflected_field, complex_zero(n_total_rays))
            reflected_vector = vector_select(hit, reflected_vector, vector_zero(n_total_rays))
            dot_dn = dr.dot(ray_dir, hit_n)
            reflected_dir = ray_dir - 2.0 * dot_dn * hit_n
            next_origin = hit_p + reflected_dir * RAY_ORIGIN_BIAS
            next_hit, next_blocker, _, _, _ = scene.intersect_rays_with_prim(next_origin, reflected_dir, hit)
            accumulate_kwargs = dict(grid=suffix.grid, grid_data=suffix.grid_data, seg_origin=hit_p, seg_dir=reflected_dir, blocker_dist=next_blocker, seg_field=reflected_field, seg_vector=reflected_vector, state_idx=state_idx, n_states=n_states, wavelength=wave.wavelength_scalar, k=wave.k_scalar, active=hit, execution=execution)
            segment_field, segment_vector = accumulate_segment_fields(**accumulate_kwargs)
            total = total + segment_field
            total_vector = vector_add(total_vector, segment_vector)
            if execution.suffix_russian_roulette and bounce > 0 and (bounce + 1 < suffix.max_bounces):
                survival_probability = ForwardEval.roulette_survival_probability(reflected_field, next_hit)
                survive = next_hit & (ForwardEval.roulette_random(state_idx, dir_idx, bounce) < survival_probability)
                survival_scale = dr.rcp(dr.maximum(survival_probability, wt.Float(1e-06)))
                reflected_field = dr.select(survive, reflected_field * wt.Complex2f(survival_scale, wt.Float(0.0)), complex_zero(n_total_rays))
                reflected_vector = vector_select(survive, vector_scale(reflected_vector, survival_scale), vector_zero(n_total_rays))
                next_hit = survive
            ray_origin = next_origin
            ray_dir = reflected_dir
            current_field = reflected_field
            current_vector = reflected_vector
            active = next_hit
        return (total, total_vector)


__all__ = ['ABranch', 'DiffAngle', 'ForwardEval', 'cot', 'f_utd', 'fresnel_integral']
