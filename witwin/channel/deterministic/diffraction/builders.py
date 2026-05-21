"""State construction for diffraction solving."""
import math
import drjit as dr
import rayd
from witwin.channel.deterministic import types as wt
from witwin.channel.core.runtime import Material, Tx, Wave
from ..reflection.detail import (
    coerce_material_context,
    coerce_trace_detail,
)
from witwin.channel.core.numerics.arrays import complex_abs_sqr, complex_zero, gather_point3
from witwin.channel.core.physics.polarization import project_real_polarization_to_ray, reflect_field_vector, vector_from_scalar, vector_power, vector_scale, vector_select, vector_zero
from witwin.channel.core.geometry.diffraction import wedge_exterior_mask
from witwin.channel.core.geometry.raygen import generate_circle_directions, generate_sphere_directions
from .forward import ForwardEval
from .state import APPROX_MODE_DIRECT_FIRST_ORDER, APPROX_MODE_RECURSIVE_DIFFRACTION, APPROX_MODE_SAMPLED_INSERTED_REFLECTION, APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN, APPROX_MODE_SAMPLED_REFLECTION_PREFIX, APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN, SOURCE_TYPE_DIRECT_TX, SOURCE_TYPE_REFLECTION_PREFIX, SPEED_OF_LIGHT, Geo
from .state import State
from witwin.channel.deterministic.kernels.cartesian_filter import compact_index_pairs
from witwin.channel.deterministic.kernels.cartesian_filter.native_impl import deduplicate_cartesian_pairs
from witwin.channel.deterministic.kernels.packed_state import concat_state_arrays, gather_field_evaluation_state_fields, gather_inserted_reflection_state_fields, subset_state_arrays
from witwin.channel.deterministic.kernels.pruning_sort import prune_state_arrays_by_budget, prune_state_arrays_by_budget_pair
_HIGHER_ORDER_CANDIDATE_BACKENDS = {'auto', 'rayd_edge_bvh', 'bruteforce'}
_HIGHER_ORDER_EDGE_BVH_PROBE_COUNT = 18
_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_SCALE = 0.6
_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_MIN = 0.5
_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_MAX = 4.0
_HIGHER_ORDER_EDGE_BVH_EXACT_PAIR_LIMIT = 1 << 12
_PRE_EXPANSION_SOURCE_DIVISOR = 4


def _reduced_budget(base_budget: int | None, *, divisor: int) -> int | None:
    if base_budget is None:
        return None
    budget = max(0, int(base_budget))
    if budget == 0:
        return 0
    divisor = max(1, int(divisor))
    return max(1, (budget + divisor - 1) // divisor)


def _bounded_source_budget(total_state_budget_per_order: int | None, *, inserted_state_budget_per_order: int | None = None) -> int | None:
    budgets = []
    reduced_total_budget = _reduced_budget(total_state_budget_per_order, divisor=_PRE_EXPANSION_SOURCE_DIVISOR)
    if reduced_total_budget is not None:
        budgets.append(int(reduced_total_budget))
    if inserted_state_budget_per_order is not None:
        budgets.append(max(0, int(inserted_state_budget_per_order)))
    if not budgets:
        return None
    return min(budgets)


def pre_expansion_pruning_policy(*, solver_mode: str, memory_profile: str, total_state_budget_per_order: int | None, inserted_state_budget_per_order: int | None) -> dict[str, object]:
    """Resolve Phase 5 source-side pruning before expensive expansion stages."""
    bounded_mode = str(solver_mode) == 'fast_approximate' or str(memory_profile) == 'memory_safe'
    if not bounded_mode:
        return {'enabled': False, 'policy': 'disabled', 'higher_order_source_budget': None, 'inserted_source_budget': None, 'source_budget_divisor': int(_PRE_EXPANSION_SOURCE_DIVISOR), 'reason': 'Accuracy mode with the default memory profile does not apply automatic pre-expansion pruning.'}
    higher_order_source_budget = _bounded_source_budget(total_state_budget_per_order)
    inserted_source_budget = _bounded_source_budget(total_state_budget_per_order, inserted_state_budget_per_order=inserted_state_budget_per_order)
    if str(memory_profile) == 'memory_safe':
        policy = 'memory_safe_topk_power'
        reason = 'Memory-safe mode prunes weak source states before higher-order and inserted-reflection expansion to reduce candidate growth.'
    else:
        policy = 'fast_approximate_topk_power'
        reason = 'Fast approximate mode prunes weak source states before higher-order and inserted-reflection expansion to keep Cartesian growth bounded.'
    return {'enabled': bool(higher_order_source_budget is not None or inserted_source_budget is not None), 'policy': policy, 'higher_order_source_budget': higher_order_source_budget, 'inserted_source_budget': inserted_source_budget, 'source_budget_divisor': int(_PRE_EXPANSION_SOURCE_DIVISOR), 'reason': reason}


def prune_for_pre_expansion(state_arrays: dict, max_states: int | None, *, budget_name: str, policy: str) -> tuple[dict, dict]:
    """Apply a Phase 5 source-side pruning budget and annotate the report."""
    pruned, report = prune_state_arrays_by_budget(state_arrays, max_states, budget_name)
    report = dict(report)
    report['stage'] = 'pre_expansion'
    report['policy'] = str(policy)
    return (pruned, report)


def _state_count(state_arrays) -> int:
    return 0 if state_arrays is None else int(state_arrays['n_states'])


def _prepare_report(*, max_order: int, state_layout: str, edge_count: int, candidate_backend: str, inserted_reflection_enabled: bool, pre_expansion_policy=None) -> dict[str, object]:
    return {
        'max_order': int(max_order),
        'state_layout': str(state_layout),
        'edge_count': int(edge_count),
        'candidate_backend': str(candidate_backend),
        'candidate_probe_count': (
            int(_HIGHER_ORDER_EDGE_BVH_PROBE_COUNT)
            if str(candidate_backend) == 'rayd_edge_bvh'
            else 0
        ),
        'inserted_reflection_enabled': bool(inserted_reflection_enabled),
        'pre_expansion_policy': dict(pre_expansion_policy or {}),
        'stages': {},
        'orders': (),
        'pruning_reports': (),
        'final_state_count': 0,
    }


def _tag_pruning_report(report: dict, *, stage: str, order: int, policy: str | None = None) -> dict:
    tagged = dict(report)
    tagged['stage'] = str(stage)
    tagged['order'] = int(order)
    if policy is not None:
        tagged['policy'] = str(policy)
    return tagged


def face_response(*, edge_pos, edge_dir, n0, nn, source_pos, adjacent_face0, adjacent_face1, wave: Wave, scene, material: Material):
    edge_state = {'edge_pos': edge_pos, 'edge_dir': edge_dir, 'n0': n0, 'n_face_n': nn, 'source_pos': source_pos, 'adjacent_face0': adjacent_face0, 'adjacent_face1': adjacent_face1}
    width = dr.width(edge_pos.x)
    face0_material, face1_material = Geo.face_material_inputs(edge_state, width, material, scene=scene)
    face0_operator, face1_operator = Geo.face_reflection_operators(edge_state, width, material, wave, scene=scene)
    return (face0_operator, face1_operator, face0_material, face1_material)

def prepare(tx: Tx, rx_z, scene, wave: Wave, reflection_detail, material: Material, reflection_n_rays, reflection_max_bounces, reflection_material: Material, reflection_mode, max_diffractions, total_state_budget_per_order=None, inserted_state_budget_per_order=None, max_inserted_reflections_per_path=None, retain_lineage_state=True, solver_mode='accuracy', memory_profile='default', state_layout='full', preserve_higher_order_candidate_topology: bool=False, return_report: bool=False):
    max_order = max(1, int(max_diffractions))
    if max_inserted_reflections_per_path is None:
        max_inserted_reflections_per_path = max(0, max_order - 1)
    else:
        max_inserted_reflections_per_path = max(0, int(max_inserted_reflections_per_path))
    resolved_edge_cache = scene.get_edge_data(rx_z, include_projection=False)
    edge_data = resolved_edge_cache.get('edge_data')
    if edge_data is None:
        empty_states = State.empty(history_size=max_order)
        if str(state_layout) == 'path_export_reduced':
            empty_states = State.reduce_for_path_export(empty_states)
        if return_report:
            report = _prepare_report(
                max_order=max_order,
                state_layout=str(state_layout),
                edge_count=0,
                candidate_backend='none',
                inserted_reflection_enabled=False,
            )
            return (resolved_edge_cache, None, empty_states, report)
        return (resolved_edge_cache, None, empty_states)
    global_to_local_idx = State.global_to_local_index(scene, edge_data)
    higher_order_candidate_backend = resolve_candidate_backend(scene, edge_data, global_to_local_idx, candidate_backend='auto') if max_order > 1 else 'not_used'
    inserted_reflection_enabled = scene is not None and reflection_n_rays > 0 and (reflection_max_bounces > 0) and (max_order > 1) and (max_inserted_reflections_per_path > 0)
    pre_expansion_policy = pre_expansion_pruning_policy(solver_mode=str(solver_mode), memory_profile=str(memory_profile), total_state_budget_per_order=total_state_budget_per_order, inserted_state_budget_per_order=inserted_state_budget_per_order)
    builder_report = _prepare_report(
        max_order=max_order,
        state_layout=str(state_layout),
        edge_count=int(edge_data['n_edges']),
        candidate_backend=higher_order_candidate_backend,
        inserted_reflection_enabled=inserted_reflection_enabled,
        pre_expansion_policy=pre_expansion_policy,
    )
    order_reports = []
    pruning_reports = []
    tx_first_order = tx_first(tx, edge_data, wave, history_size=max_order, retain_lineage_state=retain_lineage_state, scene=scene, material=material)
    reflection_first_order = prefix_first(reflection_detail, scene, edge_data, wave, global_to_local_idx, tx=tx, history_size=max_order, retain_lineage_state=retain_lineage_state, material=material)
    first_order_states = concat_state_arrays([tx_first_order, reflection_first_order])
    lineage_store = None
    next_state_id = 0
    if retain_lineage_state:
        first_order_states, lineage_store, next_state_id = State.finalize_lineage(first_order_states, lineage_store=lineage_store, next_state_id=next_state_id)
    builder_report['stages'] = {
        'tx_first': _state_count(tx_first_order),
        'prefix_first': _state_count(reflection_first_order),
        'first_order': _state_count(first_order_states),
    }
    all_state_arrays = [first_order_states]
    prev_states = first_order_states
    reflection_context = coerce_material_context(reflection_detail, default_gain=reflection_material.gain_scalar)
    for order in range(2, max_order + 1):
        source_state_count = _state_count(prev_states)
        higher_source_states = prev_states
        inserted_source_states = prev_states
        shared_pre_expansion = False
        paired_pre_expansion_sort = False
        if pre_expansion_policy['enabled']:
            higher_budget_value = pre_expansion_policy['higher_order_source_budget']
            inserted_budget_value = pre_expansion_policy['inserted_source_budget'] if inserted_reflection_enabled else None
            if inserted_reflection_enabled:
                higher_source_states, higher_budget, inserted_source_states, inserted_budget = prune_state_arrays_by_budget_pair(prev_states, higher_budget_value, inserted_budget_value, higher_budget_name='pre_expansion_higher_order_source_budget', inserted_budget_name='pre_expansion_inserted_source_budget')
                pruning_reports.append(_tag_pruning_report(higher_budget, stage='pre_expansion', order=order, policy=str(pre_expansion_policy['policy'])))
                pruning_reports.append(_tag_pruning_report(inserted_budget, stage='pre_expansion', order=order, policy=str(pre_expansion_policy['policy'])))
                shared_pre_expansion = higher_budget_value == inserted_budget_value
                paired_pre_expansion_sort = bool(inserted_budget.get('paired_pre_expansion_sort', False))
            else:
                higher_source_states, higher_budget = prune_for_pre_expansion(prev_states, higher_budget_value, budget_name='pre_expansion_higher_order_source_budget', policy=str(pre_expansion_policy['policy']))
                pruning_reports.append(_tag_pruning_report(higher_budget, stage='pre_expansion', order=order, policy=str(pre_expansion_policy['policy'])))
        direct_states = higher(higher_source_states, edge_data, wave, scene=scene, material=material, global_to_local_idx=global_to_local_idx, candidate_backend=higher_order_candidate_backend, tx=tx, retain_lineage_state=retain_lineage_state, preserve_candidate_topology=bool(preserve_higher_order_candidate_topology))
        inserted_states = State.empty(history_size=max_order)
        inserted_candidate_state_count = 0
        if inserted_reflection_enabled:
            if pre_expansion_policy['enabled'] and not (shared_pre_expansion or paired_pre_expansion_sort):
                inserted_source_states, inserted_budget = prune_for_pre_expansion(prev_states, pre_expansion_policy['inserted_source_budget'], budget_name='pre_expansion_inserted_source_budget', policy=str(pre_expansion_policy['policy']))
                pruning_reports.append(_tag_pruning_report(inserted_budget, stage='pre_expansion', order=order, policy=str(pre_expansion_policy['policy'])))
            inserted_reflection_material = Material(
                reflection_coef=reflection_context.reflection_gain,
            )
            inserted_states = inserted(inserted_source_states, scene, edge_data, global_to_local_idx, wave, n_rays=reflection_n_rays, reflection_material=inserted_reflection_material, material=material, reflection_mode=reflection_mode, max_inserted_reflections_per_path=max_inserted_reflections_per_path, tx=tx, retain_lineage_state=retain_lineage_state)
            inserted_candidate_state_count = _state_count(inserted_states)
        inserted_states, inserted_pruning_report = prune_state_arrays_by_budget(inserted_states, inserted_state_budget_per_order, 'inserted_state_budget_per_order')
        pruning_reports.append(_tag_pruning_report(inserted_pruning_report, stage='post_inserted_budget', order=order))
        combined_states = concat_state_arrays([direct_states, inserted_states])
        combined_states, total_pruning_report = prune_state_arrays_by_budget(combined_states, total_state_budget_per_order, 'total_state_budget_per_order')
        pruning_reports.append(_tag_pruning_report(total_pruning_report, stage='post_total_budget', order=order))
        if retain_lineage_state:
            combined_states, lineage_store, next_state_id = State.finalize_lineage(combined_states, lineage_store=lineage_store, next_state_id=next_state_id)
        order_reports.append({
            'order': int(order),
            'source_state_count': int(source_state_count),
            'higher_source_state_count': _state_count(higher_source_states),
            'inserted_source_state_count': (
                _state_count(inserted_source_states) if inserted_reflection_enabled else 0
            ),
            'higher_order_candidate_state_count': _state_count(direct_states),
            'inserted_reflection_candidate_state_count': int(inserted_candidate_state_count),
            'post_inserted_budget_state_count': _state_count(inserted_states),
            'post_total_budget_state_count': _state_count(combined_states),
        })
        prev_states = combined_states
        if prev_states['n_states'] == 0:
            break
        all_state_arrays.append(prev_states)
    final_state_arrays = concat_state_arrays(all_state_arrays)
    if str(state_layout) == 'path_export_reduced':
        final_state_arrays = State.reduce_for_path_export(final_state_arrays)
    builder_report['orders'] = tuple(order_reports)
    builder_report['pruning_reports'] = tuple(pruning_reports)
    builder_report['final_state_count'] = _state_count(final_state_arrays)
    if return_report:
        return (resolved_edge_cache, edge_data, final_state_arrays, builder_report)
    return (resolved_edge_cache, edge_data, final_state_arrays)

def tx_first(tx: Tx, edge_data, wave: Wave, history_size=3, retain_lineage_state=True, scene=None, material: Material | None = None):
    if material is None:
        material = Material()
    n_edges = edge_data['n_edges']
    if n_edges == 0:
        return State.empty(history_size=history_size)
    line_min_all, line_max_all = Geo.data_line_bounds(edge_data, context='_build_tx_first_order_state_arrays')
    edge_idx = dr.arange(wt.UInt32, n_edges)
    edge_pos = edge_data['pos']
    adjacent_face0 = edge_data['adjacent_face0']
    adjacent_face1 = edge_data['adjacent_face1']
    source_pos = wt.Point3f(dr.repeat(tx.position.x, n_edges), dr.repeat(tx.position.y, n_edges), dr.repeat(tx.position.z, n_edges))
    visible = scene.segment_visible(source_pos, edge_pos, ignore_prim_idx=(adjacent_face0, adjacent_face1))
    source_exterior = wedge_exterior_mask(source_pos - edge_pos, edge_data['edge_dir'], edge_data['n0'], edge_data['n_face_n'])
    edge_idx, _ = compact_index_pairs(edge_idx, edge_idx, visible & source_exterior)
    n_states = dr.width(edge_idx)
    if n_states == 0:
        return State.empty(history_size=history_size)
    edge_pos = dr.gather(wt.Point3f, edge_data['pos'], edge_idx)
    edge_dir = dr.gather(wt.Vector3f, edge_data['edge_dir'], edge_idx)
    n0 = dr.gather(wt.Vector3f, edge_data['n0'], edge_idx)
    nn = dr.gather(wt.Vector3f, edge_data['n_face_n'], edge_idx)
    wedge_n = dr.gather(wt.Float, edge_data['wedge_n'], edge_idx)
    edge_line_min = dr.gather(wt.Float, line_min_all, edge_idx)
    edge_line_max = dr.gather(wt.Float, line_max_all, edge_idx)
    adjacent_face0 = dr.gather(wt.Int32, edge_data['adjacent_face0'], edge_idx)
    adjacent_face1 = dr.gather(wt.Int32, edge_data['adjacent_face1'], edge_idx)
    source_pos = wt.Point3f(dr.repeat(tx.position.x, n_states), dr.repeat(tx.position.y, n_states), dr.repeat(tx.position.z, n_states))
    path_length_prefix = dr.norm(edge_pos - source_pos)
    incident_field = Geo.source_field(tx.position, wt.Complex2f(1.0, 0.0), edge_pos, wave)
    incident_normal_derivative = complex_zero(n_states)
    ray_dir = (edge_pos - source_pos) / (dr.norm(edge_pos - source_pos) + 1e-12)
    pol_dir = project_real_polarization_to_ray(tx.polarization, ray_dir)
    incident_vector = vector_from_scalar(incident_field, pol_dir)
    incident_normal_derivative_vector = vector_zero(n_states)
    face0_operator, face1_operator, face0_material, face1_material = face_response(edge_pos=edge_pos, edge_dir=edge_dir, n0=n0, nn=nn, source_pos=source_pos, adjacent_face0=adjacent_face0, adjacent_face1=adjacent_face1, wave=wave, scene=scene, material=material)
    is_direct_tx = dr.full(wt.Bool, True, n_states)
    source_type_code = dr.full(wt.UInt32, SOURCE_TYPE_DIRECT_TX, n_states)
    prefix_reflection_depth = dr.zeros(wt.UInt32, n_states)
    intermediate_reflection_depth = dr.zeros(wt.UInt32, n_states)
    suffix_reflection_depth = dr.zeros(wt.UInt32, n_states)
    approximation_mode_code = dr.full(wt.UInt32, APPROX_MODE_DIRECT_FIRST_ORDER, n_states)
    order = dr.full(wt.UInt32, 1, n_states)
    return State.make(edge_idx=edge_idx, edge_pos=edge_pos, edge_dir=edge_dir, n0=n0, nn=nn, wedge_n=wedge_n, adjacent_face0=adjacent_face0, adjacent_face1=adjacent_face1, source_pos=source_pos, path_length_prefix=path_length_prefix, first_interaction_pos=edge_pos, edge_line_min=edge_line_min, edge_line_max=edge_line_max, incident_field=incident_field, incident_normal_derivative=incident_normal_derivative, incident_vector=incident_vector, incident_normal_derivative_vector=incident_normal_derivative_vector, is_direct_tx=is_direct_tx, face0_operator=face0_operator, face1_operator=face1_operator, face0_material=face0_material, face1_material=face1_material, source_type_code=source_type_code, prefix_reflection_depth=prefix_reflection_depth, intermediate_reflection_depth=intermediate_reflection_depth, suffix_reflection_depth=suffix_reflection_depth, approximation_mode_code=approximation_mode_code, order=order, lineage_parent_state_id=dr.full(wt.Int32, -1, n_states), lineage_last_edge_idx=wt.Int32(edge_idx), lineage_last_reflection_depth_delta=dr.zeros(wt.UInt32, n_states), retain_lineage_state=retain_lineage_state)

def prefix_first(reflection_detail, scene, edge_data, wave: Wave, global_to_local_idx, *, tx: Tx, history_size=3, retain_lineage_state=True, material: Material | None = None):
    if material is None:
        material = Material()
    if reflection_detail is None or scene is None:
        return State.empty(history_size=history_size)
    detail = coerce_trace_detail(reflection_detail)
    per_bounce_states = []
    if scene._triangle_runtime() is None:
        return State.empty(history_size=history_size)
    from ..reflection import epc as _epc
    line_min_all, line_max_all = Geo.data_line_bounds(edge_data, context='_build_reflection_first_order_state_arrays')
    for bounce_idx, paths in enumerate(detail.source_paths_per_bounce):
        chain_depth = 0 if paths is None else int(paths.chain_depth)
        n_paths = 0 if paths is None else int(paths.n_paths)
        if n_paths <= 0 or chain_depth <= 0:
            continue
        n_edges = edge_data['n_edges']
        path_chunk_size = Geo.cart_chunk(n_paths, n_edges)
        for path_start in range(0, n_paths, path_chunk_size):
            chunk_n_paths = min(path_chunk_size, n_paths - path_start)
            n_pairs = chunk_n_paths * n_edges
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            path_idx = pair_idx // n_edges + wt.UInt32(path_start)
            local_edge_idx = pair_idx % n_edges
            source_pos = gather_point3(paths.image_source, path_idx)
            edge_pos = dr.gather(wt.Point3f, edge_data['pos'], local_edge_idx)
            edge_dir = dr.gather(wt.Vector3f, edge_data['edge_dir'], local_edge_idx)
            n0 = dr.gather(wt.Vector3f, edge_data['n0'], local_edge_idx)
            nn = dr.gather(wt.Vector3f, edge_data['n_face_n'], local_edge_idx)
            wedge_n = dr.gather(wt.Float, edge_data['wedge_n'], local_edge_idx)
            edge_line_min = dr.gather(wt.Float, line_min_all, local_edge_idx)
            edge_line_max = dr.gather(wt.Float, line_max_all, local_edge_idx)
            adjacent_face0 = dr.gather(wt.Int32, edge_data['adjacent_face0'], local_edge_idx)
            adjacent_face1 = dr.gather(wt.Int32, edge_data['adjacent_face1'], local_edge_idx)
            has_reflected_support, chain_vector, chain_geometry = _epc.chain_to_target(paths=paths, path_idx=path_idx, target_pos=edge_pos, scene=scene, target_adjacent_faces=(adjacent_face0, adjacent_face1), reflection_detail=detail, wave=wave, tx=tx, return_endpoints=True)
            source_exterior = wedge_exterior_mask(source_pos - edge_pos, edge_dir, n0, nn)
            support_mask = has_reflected_support & source_exterior
            keep_idx = dr.compress(support_mask)
            if dr.width(keep_idx) == 0:
                continue
            source_pos_support = gather_point3(source_pos, keep_idx)
            edge_pos_support = dr.gather(wt.Point3f, edge_pos, keep_idx)
            chain_vector_support = {'x': dr.gather(wt.Complex2f, chain_vector['x'], keep_idx), 'y': dr.gather(wt.Complex2f, chain_vector['y'], keep_idx), 'z': dr.gather(wt.Complex2f, chain_vector['z'], keep_idx)}
            unit_incident_field = Geo.source_field(source_pos_support, wt.Complex2f(1.0, 0.0), edge_pos_support, wave)
            field_power = vector_power(chain_vector_support) * complex_abs_sqr(unit_incident_field)
            valid_field_idx = dr.compress(field_power > wt.Float(1e-20))
            if dr.width(valid_field_idx) == 0:
                continue
            final_idx = dr.gather(wt.UInt32, keep_idx, valid_field_idx)
            local_edge_keep = dr.gather(wt.UInt32, local_edge_idx, final_idx)
            edge_pos_keep = dr.gather(wt.Point3f, edge_pos, final_idx)
            edge_dir_keep = dr.gather(wt.Vector3f, edge_dir, final_idx)
            n0_keep = dr.gather(wt.Vector3f, n0, final_idx)
            nn_keep = dr.gather(wt.Vector3f, nn, final_idx)
            wedge_n_keep = dr.gather(wt.Float, wedge_n, final_idx)
            edge_line_min_keep = dr.gather(wt.Float, edge_line_min, final_idx)
            edge_line_max_keep = dr.gather(wt.Float, edge_line_max, final_idx)
            adjacent_face0_keep = dr.gather(wt.Int32, adjacent_face0, final_idx)
            adjacent_face1_keep = dr.gather(wt.Int32, adjacent_face1, final_idx)
            source_pos_keep = gather_point3(source_pos, final_idx)
            first_interaction_pos_keep = dr.gather(wt.Point3f, chain_geometry['first_hit'], final_idx)
            path_length_prefix = dr.norm(edge_pos_keep - source_pos_keep)
            chain_vector_keep = {'x': dr.gather(wt.Complex2f, chain_vector_support['x'], valid_field_idx), 'y': dr.gather(wt.Complex2f, chain_vector_support['y'], valid_field_idx), 'z': dr.gather(wt.Complex2f, chain_vector_support['z'], valid_field_idx)}
            unit_incident_field = dr.gather(wt.Complex2f, unit_incident_field, valid_field_idx)
            incident_field_keep = unit_incident_field
            incident_vector_keep = vector_scale(chain_vector_keep, unit_incident_field)
            incident_normal_derivative = complex_zero(dr.width(valid_field_idx))
            incident_normal_derivative_vector = vector_zero(dr.width(valid_field_idx))
            n_valid_states = dr.width(valid_field_idx)
            face0_operator, face1_operator, face0_material, face1_material = face_response(edge_pos=edge_pos_keep, edge_dir=edge_dir_keep, n0=n0_keep, nn=nn_keep, source_pos=source_pos_keep, adjacent_face0=adjacent_face0_keep, adjacent_face1=adjacent_face1_keep, wave=wave, scene=scene, material=material)
            prefix_reflection_depth = dr.full(wt.UInt32, bounce_idx + 1, n_valid_states)
            intermediate_reflection_depth = dr.zeros(wt.UInt32, n_valid_states)
            suffix_reflection_depth = dr.zeros(wt.UInt32, n_valid_states)
            approximation_mode_code = dr.full(wt.UInt32, APPROX_MODE_SAMPLED_REFLECTION_PREFIX, n_valid_states)
            order = dr.full(wt.UInt32, 1, n_valid_states)
            per_bounce_states.append(State.make(edge_idx=local_edge_keep, edge_pos=edge_pos_keep, edge_dir=edge_dir_keep, n0=n0_keep, nn=nn_keep, wedge_n=wedge_n_keep, adjacent_face0=adjacent_face0_keep, adjacent_face1=adjacent_face1_keep, source_pos=source_pos_keep, path_length_prefix=path_length_prefix, first_interaction_pos=first_interaction_pos_keep, edge_line_min=edge_line_min_keep, edge_line_max=edge_line_max_keep, incident_field=incident_field_keep, incident_normal_derivative=incident_normal_derivative, incident_vector=incident_vector_keep, incident_normal_derivative_vector=incident_normal_derivative_vector, is_direct_tx=dr.full(wt.Bool, False, n_valid_states), face0_operator=face0_operator, face1_operator=face1_operator, face0_material=face0_material, face1_material=face1_material, source_type_code=dr.full(wt.UInt32, SOURCE_TYPE_REFLECTION_PREFIX, n_valid_states), prefix_reflection_depth=prefix_reflection_depth, intermediate_reflection_depth=intermediate_reflection_depth, suffix_reflection_depth=suffix_reflection_depth, approximation_mode_code=approximation_mode_code, order=order, lineage_parent_state_id=dr.full(wt.Int32, -1, n_valid_states), lineage_last_edge_idx=wt.Int32(local_edge_keep), lineage_last_reflection_depth_delta=prefix_reflection_depth, retain_lineage_state=retain_lineage_state))
    result = concat_state_arrays(per_bounce_states)
    return result

def ensure_source_lineage(state_arrays):
    if not State.has_lineage_state(state_arrays):
        return state_arrays
    state_ids = State.ids(state_arrays)
    lineage_store = State.lineage_store(state_arrays)
    if state_ids is not None and lineage_store is not None and (not bool(dr.any(wt.Int32(state_ids) < 0))):
        return state_arrays
    next_state_id = 0 if lineage_store is None else int(lineage_store.get('n_states', 0))
    finalized, _, _ = State.finalize_lineage(state_arrays, lineage_store=lineage_store, next_state_id=next_state_id)
    return finalized

def resolve_candidate_backend(scene, edge_data, global_to_local_idx, candidate_backend='auto'):
    if candidate_backend not in _HIGHER_ORDER_CANDIDATE_BACKENDS:
        raise ValueError(f"Unsupported higher-order candidate backend '{candidate_backend}'. Supported values are 'auto', 'rayd_edge_bvh', and 'bruteforce'.")
    if candidate_backend == 'auto':
        candidate_backend = 'rayd_edge_bvh'
    return candidate_backend

def selected_edge_mask(edge_data, current_mask):
    n_edges_total = dr.width(current_mask)
    if edge_data is None or edge_data.get('global_idx') is None or edge_data['n_edges'] <= 0 or (n_edges_total == 0):
        return dr.full(wt.Bool, False, n_edges_total)
    selected_flags = dr.zeros(wt.UInt32, n_edges_total)
    global_idx = wt.Int32(edge_data['global_idx'])
    valid = global_idx >= 0
    safe_global_idx = wt.UInt32(dr.select(valid, global_idx, wt.Int32(0)))
    dr.scatter(selected_flags, dr.full(wt.UInt32, 1, dr.width(global_idx)), safe_global_idx, valid)
    return current_mask & (selected_flags != wt.UInt32(0))

def dedupe_pairs(prev_idx, edge_idx, n_edges):
    if dr.width(prev_idx) == 0:
        return (dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0))
    return deduplicate_cartesian_pairs(prev_idx, edge_idx, n_edges)

def bruteforce_pairs(prev_state_arrays, prev_start, chunk_n_prev, n_edges):
    if chunk_n_prev <= 0 or n_edges <= 0:
        return (dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0), 0)
    n_pairs = chunk_n_prev * n_edges
    pair_idx = dr.arange(wt.UInt32, n_pairs)
    prev_idx_all = pair_idx // n_edges + wt.UInt32(prev_start)
    edge_idx_all = pair_idx % n_edges
    prev_edge_idx_all = dr.gather(wt.UInt32, prev_state_arrays['edge_idx'], prev_idx_all)
    distinct_edge = edge_idx_all != prev_edge_idx_all
    prev_idx, edge_idx = compact_index_pairs(prev_idx_all, edge_idx_all, distinct_edge)
    return (prev_idx, edge_idx, int(n_pairs))

def bvh_pairs(prev_state_arrays, edge_data, prev_start, chunk_n_prev, scene, global_to_local_idx):
    if chunk_n_prev <= 0:
        return (dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0))
    n_probes = chunk_n_prev * _HIGHER_ORDER_EDGE_BVH_PROBE_COUNT
    probe_idx = dr.arange(wt.UInt32, n_probes)
    state_offset = probe_idx // _HIGHER_ORDER_EDGE_BVH_PROBE_COUNT
    prev_idx_all = state_offset + wt.UInt32(prev_start)
    probe_slot = probe_idx % _HIGHER_ORDER_EDGE_BVH_PROBE_COUNT
    probe_grid_slot = probe_slot % wt.UInt32(_HIGHER_ORDER_EDGE_BVH_PROBE_COUNT // 2)
    probe_u = wt.Float(probe_grid_slot // wt.UInt32(3)) - 1.0
    probe_v = wt.Float(probe_grid_slot % wt.UInt32(3)) - 1.0
    probe_sign = dr.select(probe_slot < wt.UInt32(_HIGHER_ORDER_EDGE_BVH_PROBE_COUNT // 2), wt.Float(1.0), wt.Float(-1.0))
    edge_pos = dr.gather(wt.Point3f, prev_state_arrays['edge_pos'], prev_idx_all)
    source_pos = dr.gather(wt.Point3f, prev_state_arrays['source_pos'], prev_idx_all)
    basis_u = dr.gather(wt.Vector3f, prev_state_arrays['incident_basis_u'], prev_idx_all)
    basis_v = dr.gather(wt.Vector3f, prev_state_arrays['incident_basis_v'], prev_idx_all)
    basis_k = dr.gather(wt.Vector3f, prev_state_arrays['incident_basis_k'], prev_idx_all)
    prev_edge_idx = dr.gather(wt.UInt32, prev_state_arrays['edge_idx'], prev_idx_all)
    source_distance = dr.norm(edge_pos - source_pos)
    probe_radius = dr.clip(source_distance * _HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_SCALE, wt.Float(_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_MIN), wt.Float(_HIGHER_ORDER_EDGE_BVH_PROBE_RADIUS_MAX))
    origin_offset = basis_u * (probe_radius * probe_u) + basis_v * (probe_radius * probe_v)
    ray_origin = edge_pos + origin_offset
    ray_dir = basis_k * probe_sign
    nearest = scene.nearest_edge(rayd.Ray(ray_origin, ray_dir))
    valid = nearest.is_valid()
    global_edge_idx = wt.Int32(nearest.global_edge_id)
    valid = valid & (global_edge_idx >= 0)
    safe_global_idx = wt.UInt32(dr.select(valid, global_edge_idx, wt.Int32(0)))
    local_edge_idx_i32 = dr.gather(wt.Int32, global_to_local_idx, safe_global_idx)
    valid = valid & (local_edge_idx_i32 >= 0)
    valid = valid & (wt.Int32(prev_edge_idx) != local_edge_idx_i32)
    candidate_idx = dr.compress(valid)
    if dr.width(candidate_idx) == 0:
        return (dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0))
    prev_idx = dr.gather(wt.UInt32, prev_idx_all, candidate_idx)
    edge_idx = dr.gather(wt.UInt32, wt.UInt32(dr.select(valid, local_edge_idx_i32, wt.Int32(0))), candidate_idx)
    return dedupe_pairs(prev_idx, edge_idx, edge_data['n_edges'])

def higher(prev_state_arrays, edge_data, wave: Wave, scene=None, material: Material | None = None, global_to_local_idx=None, candidate_backend='auto', tx: Tx | None = None, retain_lineage_state=True, preserve_candidate_topology: bool=False):
    if material is None:
        material = Material()
    prev_state_arrays = ensure_source_lineage(prev_state_arrays)
    n_prev = prev_state_arrays['n_states']
    n_edges = edge_data['n_edges']
    history_size = Geo.history_size(prev_state_arrays)
    if n_prev == 0 or n_edges == 0:
        return State.empty(history_size=history_size)
    line_min_all, line_max_all = Geo.data_line_bounds(edge_data, context='_build_higher_order_state_arrays')
    resolved_candidate_backend = resolve_candidate_backend(scene, edge_data, global_to_local_idx, candidate_backend=candidate_backend)
    chunk_states = []
    rayd_handle = scene._rayd_scene if resolved_candidate_backend == 'rayd_edge_bvh' else None
    prev_chunk_size = Geo.cart_chunk(n_prev, n_edges if resolved_candidate_backend == 'bruteforce' else _HIGHER_ORDER_EDGE_BVH_PROBE_COUNT)
    saved_edge_mask = None
    if rayd_handle is not None:
        saved_edge_mask = rayd_handle.edge_mask()
        rayd_handle.set_edge_mask(selected_edge_mask(edge_data, saved_edge_mask))
        rayd_handle.sync()
    try:
        for prev_start in range(0, n_prev, prev_chunk_size):
            chunk_n_prev = min(prev_chunk_size, n_prev - prev_start)
            if resolved_candidate_backend == 'bruteforce':
                prev_idx, edge_idx, _ = bruteforce_pairs(prev_state_arrays, prev_start, chunk_n_prev, n_edges)
                if dr.width(prev_idx) == 0:
                    continue
            else:
                if chunk_n_prev * n_edges <= _HIGHER_ORDER_EDGE_BVH_EXACT_PAIR_LIMIT:
                    prev_idx, edge_idx, _ = bruteforce_pairs(prev_state_arrays, prev_start, chunk_n_prev, n_edges)
                else:
                    prev_idx, edge_idx = bvh_pairs(prev_state_arrays, edge_data, prev_start, chunk_n_prev, scene, global_to_local_idx)
                if dr.width(prev_idx) == 0:
                    continue
            visibility_mask = None
            if scene is not None:
                if preserve_candidate_topology and candidate_backend == 'rayd_edge_bvh':
                    visible = dr.full(wt.Bool, True, dr.width(prev_idx))
                else:
                    prev_edge_pos = dr.gather(wt.Point3f, prev_state_arrays['edge_pos'], prev_idx)
                    edge_pos = dr.gather(wt.Point3f, edge_data['pos'], edge_idx)
                    prev_adjacent_face0 = dr.gather(wt.Int32, prev_state_arrays['adjacent_face0'], prev_idx)
                    prev_adjacent_face1 = dr.gather(wt.Int32, prev_state_arrays['adjacent_face1'], prev_idx)
                    next_adjacent_face0 = dr.gather(wt.Int32, edge_data['adjacent_face0'], edge_idx)
                    next_adjacent_face1 = dr.gather(wt.Int32, edge_data['adjacent_face1'], edge_idx)
                    visible = scene.segment_visible(prev_edge_pos, edge_pos, ignore_prim_idx=(prev_adjacent_face0, prev_adjacent_face1, next_adjacent_face0, next_adjacent_face1))
                visible_count = int(dr.width(dr.compress(visible)))
                if preserve_candidate_topology:
                    visibility_mask = visible
                else:
                    prev_idx, edge_idx = compact_index_pairs(prev_idx, edge_idx, visible)
                if not preserve_candidate_topology and visible_count == 0:
                    continue
            else:
                visibility_mask = dr.full(wt.Bool, True, dr.width(prev_idx))
            prev_states = gather_field_evaluation_state_fields(prev_state_arrays, prev_idx)
            candidate_edge_pos = dr.gather(wt.Point3f, edge_data['pos'], edge_idx)
            candidate_adjacent_face0 = dr.gather(wt.Int32, edge_data['adjacent_face0'], edge_idx)
            candidate_adjacent_face1 = dr.gather(wt.Int32, edge_data['adjacent_face1'], edge_idx)
            incident_field, incident_normal_derivative, incident_vector, incident_normal_derivative_vector = ForwardEval.to_targets(prev_states, candidate_edge_pos, wave, return_normal_derivative=True, return_vector=True, material=material, scene=scene, smooth_exterior_shadow=bool(preserve_candidate_topology), tx=tx, select_diffraction_point=False, enable_segment_visibility=False)
            field_power = vector_power(incident_vector)
            field_valid = field_power > wt.Float(1e-20)
            active_mask = field_valid if visibility_mask is None else visibility_mask & field_valid
            keep_count = int(dr.width(dr.compress(active_mask)))
            if not preserve_candidate_topology and keep_count == 0:
                continue
            if preserve_candidate_topology:
                keep_prev_idx = prev_idx
                keep_edge_idx = edge_idx
            else:
                visible_local_idx = dr.arange(wt.UInt32, dr.width(prev_idx))
                keep_idx, keep_edge_idx = compact_index_pairs(visible_local_idx, edge_idx, active_mask)
                keep_prev_idx = dr.gather(wt.UInt32, prev_idx, keep_idx)
            kept_prev_states = gather_inserted_reflection_state_fields(prev_state_arrays, keep_prev_idx)
            if preserve_candidate_topology:
                keep_edge_pos = candidate_edge_pos
                keep_edge_dir = dr.gather(wt.Vector3f, edge_data['edge_dir'], keep_edge_idx)
                keep_n0 = dr.gather(wt.Vector3f, edge_data['n0'], keep_edge_idx)
                keep_nn = dr.gather(wt.Vector3f, edge_data['n_face_n'], keep_edge_idx)
                keep_wedge_n = dr.gather(wt.Float, edge_data['wedge_n'], keep_edge_idx)
                keep_edge_line_min = dr.gather(wt.Float, line_min_all, keep_edge_idx)
                keep_edge_line_max = dr.gather(wt.Float, line_max_all, keep_edge_idx)
                keep_adjacent_face0 = candidate_adjacent_face0
                keep_adjacent_face1 = candidate_adjacent_face1
            else:
                keep_edge_pos = dr.gather(wt.Point3f, candidate_edge_pos, keep_idx)
                keep_edge_dir = dr.gather(wt.Vector3f, edge_data['edge_dir'], keep_edge_idx)
                keep_n0 = dr.gather(wt.Vector3f, edge_data['n0'], keep_edge_idx)
                keep_nn = dr.gather(wt.Vector3f, edge_data['n_face_n'], keep_edge_idx)
                keep_wedge_n = dr.gather(wt.Float, edge_data['wedge_n'], keep_edge_idx)
                keep_edge_line_min = dr.gather(wt.Float, line_min_all, keep_edge_idx)
                keep_edge_line_max = dr.gather(wt.Float, line_max_all, keep_edge_idx)
                keep_adjacent_face0 = dr.gather(wt.Int32, candidate_adjacent_face0, keep_idx)
                keep_adjacent_face1 = dr.gather(wt.Int32, candidate_adjacent_face1, keep_idx)
            keep_source_pos = kept_prev_states['edge_pos']
            keep_path_length_prefix = None
            keep_first_interaction_pos = None
            if retain_lineage_state:
                keep_prev_path_length = kept_prev_states['path_length_prefix']
                keep_path_length_prefix = keep_prev_path_length + dr.norm(keep_edge_pos - keep_source_pos)
                keep_first_interaction_pos = kept_prev_states['first_interaction_pos']
            if preserve_candidate_topology:
                zero_field = complex_zero(dr.width(keep_prev_idx))
                zero_vector = vector_zero(dr.width(keep_prev_idx))
                keep_incident_field = wt.Complex2f(dr.select(active_mask, incident_field.real, zero_field.real), dr.select(active_mask, incident_field.imag, zero_field.imag))
                keep_incident_normal_derivative = wt.Complex2f(dr.select(active_mask, incident_normal_derivative.real, zero_field.real), dr.select(active_mask, incident_normal_derivative.imag, zero_field.imag))
                keep_incident_vector = vector_select(active_mask, incident_vector, zero_vector)
                keep_incident_normal_derivative_vector = vector_select(active_mask, incident_normal_derivative_vector, zero_vector)
            else:
                keep_incident_field = dr.gather(wt.Complex2f, incident_field, keep_idx)
                keep_incident_normal_derivative = dr.gather(wt.Complex2f, incident_normal_derivative, keep_idx)
                keep_incident_vector = {'x': dr.gather(wt.Complex2f, incident_vector['x'], keep_idx), 'y': dr.gather(wt.Complex2f, incident_vector['y'], keep_idx), 'z': dr.gather(wt.Complex2f, incident_vector['z'], keep_idx)}
                keep_incident_normal_derivative_vector = {'x': dr.gather(wt.Complex2f, incident_normal_derivative_vector['x'], keep_idx), 'y': dr.gather(wt.Complex2f, incident_normal_derivative_vector['y'], keep_idx), 'z': dr.gather(wt.Complex2f, incident_normal_derivative_vector['z'], keep_idx)}
            face0_operator, face1_operator, face0_material, face1_material = face_response(edge_pos=keep_edge_pos, edge_dir=keep_edge_dir, n0=keep_n0, nn=keep_nn, source_pos=keep_source_pos, adjacent_face0=keep_adjacent_face0, adjacent_face1=keep_adjacent_face1, wave=wave, scene=scene, material=material)
            keep_prev_order = kept_prev_states['order']
            keep_order = keep_prev_order + 1
            keep_parent_state_id = State.ids(kept_prev_states)
            keep_lineage_store = State.lineage_store(kept_prev_states)
            keep_source_type_code = kept_prev_states['source_type_code'] if retain_lineage_state else None
            keep_prefix_reflection_depth = kept_prev_states['prefix_reflection_depth']
            keep_intermediate_reflection_depth = kept_prev_states['intermediate_reflection_depth']
            keep_suffix_reflection_depth = kept_prev_states['suffix_reflection_depth']
            keep_approximation_mode_code = dr.select(keep_intermediate_reflection_depth > wt.UInt32(0), wt.UInt32(APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN), dr.select(keep_prefix_reflection_depth > wt.UInt32(0), wt.UInt32(APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN), wt.UInt32(APPROX_MODE_RECURSIVE_DIFFRACTION)))
            chunk_states.append(State.make(edge_idx=keep_edge_idx, edge_pos=keep_edge_pos, edge_dir=keep_edge_dir, n0=keep_n0, nn=keep_nn, wedge_n=keep_wedge_n, adjacent_face0=keep_adjacent_face0, adjacent_face1=keep_adjacent_face1, source_pos=keep_source_pos, path_length_prefix=keep_path_length_prefix, first_interaction_pos=keep_first_interaction_pos, edge_line_min=keep_edge_line_min, edge_line_max=keep_edge_line_max, incident_field=keep_incident_field, incident_normal_derivative=keep_incident_normal_derivative, incident_vector=keep_incident_vector, incident_normal_derivative_vector=keep_incident_normal_derivative_vector, is_direct_tx=dr.full(wt.Bool, False, dr.width(keep_prev_idx)) if retain_lineage_state else None, face0_operator=face0_operator, face1_operator=face1_operator, face0_material=face0_material, face1_material=face1_material, source_type_code=keep_source_type_code, prefix_reflection_depth=keep_prefix_reflection_depth, intermediate_reflection_depth=keep_intermediate_reflection_depth, suffix_reflection_depth=keep_suffix_reflection_depth, approximation_mode_code=keep_approximation_mode_code, order=keep_order, lineage_parent_state_id=keep_parent_state_id, lineage_last_edge_idx=wt.Int32(keep_edge_idx), lineage_last_reflection_depth_delta=dr.zeros(wt.UInt32, dr.width(keep_prev_idx)), lineage_store=keep_lineage_store, retain_lineage_state=retain_lineage_state))
    finally:
        if rayd_handle is not None and saved_edge_mask is not None:
            rayd_handle.set_edge_mask(saved_edge_mask)
            rayd_handle.sync()
    result = concat_state_arrays(chunk_states)
    return result

def inserted(prev_state_arrays, scene, edge_data, global_to_local_idx, wave: Wave, n_rays, reflection_material: Material, material: Material | None = None, reflection_mode='2d', max_inserted_reflections_per_path=None, tx: Tx | None = None, retain_lineage_state=True):
    if material is None:
        material = Material()
    if tx is None:
        raise ValueError("inserted diffraction reflection tracing requires an explicit transmitter runtime.")
    prev_state_arrays = ensure_source_lineage(prev_state_arrays)
    history_size = Geo.history_size(prev_state_arrays)
    if scene is None or scene._triangle_runtime() is None or edge_data is None or (global_to_local_idx is None) or (prev_state_arrays['n_states'] == 0) or (n_rays <= 0):
        return State.empty(history_size=history_size)
    line_min_all, line_max_all = Geo.data_line_bounds(edge_data, context='_build_inserted_reflection_state_arrays')
    if max_inserted_reflections_per_path is None:
        eligible = dr.full(wt.Bool, True, prev_state_arrays['n_states'])
    else:
        eligible = prev_state_arrays['intermediate_reflection_depth'] < wt.UInt32(max(0, int(max_inserted_reflections_per_path)))
    prev_state_arrays = subset_state_arrays(prev_state_arrays, eligible)
    if prev_state_arrays['n_states'] == 0:
        return State.empty(history_size=history_size)
    n_states = prev_state_arrays['n_states']
    rays_per_state = max(8, int(math.ceil(n_rays / max(1, n_states))))
    if reflection_mode == '2d':
        base_ray_dir = generate_circle_directions(rays_per_state)
    else:
        base_ray_dir = generate_sphere_directions(rays_per_state)
    chunk_states = []
    state_chunk_size = Geo.cart_chunk(n_states, rays_per_state)
    for state_start in range(0, n_states, state_chunk_size):
        chunk_n_states = min(state_chunk_size, n_states - state_start)
        n_total_rays = chunk_n_states * rays_per_state
        ray_idx = dr.arange(wt.UInt32, n_total_rays)
        state_idx = ray_idx // rays_per_state + wt.UInt32(state_start)
        dir_idx = ray_idx % rays_per_state
        ray_origin = dr.gather(wt.Point3f, prev_state_arrays['edge_pos'], state_idx)
        ray_dir = wt.Vector3f(dr.gather(wt.Float, base_ray_dir.x, dir_idx), dr.gather(wt.Float, base_ray_dir.y, dir_idx), dr.gather(wt.Float, base_ray_dir.z, dir_idx))
        active = dr.full(wt.Bool, True, n_total_rays)
        hit, _, hit_p, hit_n, prim_idx = scene.intersect_rays_with_prim(ray_origin, ray_dir, active)
        hit_idx = dr.compress(hit)
        hit_count = int(dr.width(hit_idx))
        if hit_count == 0:
            continue
        hit_state_idx = dr.gather(wt.UInt32, state_idx, hit_idx)
        batch_states = gather_field_evaluation_state_fields(prev_state_arrays, hit_state_idx)
        batch_state_edge_idx = dr.gather(wt.UInt32, prev_state_arrays['edge_idx'], hit_state_idx)
        hit_p = dr.gather(wt.Point3f, hit_p, hit_idx)
        hit_n = dr.gather(wt.Vector3f, hit_n, hit_idx)
        prim_idx_i32 = dr.gather(wt.Int32, wt.Int32(prim_idx), hit_idx)
        ray_dir = dr.gather(wt.Vector3f, ray_dir, hit_idx)
        n_hit = dr.width(hit_idx)
        field_at_hit, vector_at_hit = ForwardEval.to_targets(batch_states, hit_p, wave, return_vector=True, material=material, scene=scene, tx=tx, select_diffraction_point=False, enable_segment_visibility=False)
        reflection_weight, material_inputs = Geo.surface_coeff(incident_dir=ray_dir, normal=hit_n, scene=scene, prim_idx=prim_idx_i32, material=reflection_material, wave=wave, tx=tx, valid_mask=dr.full(wt.Bool, True, n_hit))
        reflected_field = field_at_hit * reflection_weight
        reflected_vector = reflect_field_vector(vector_at_hit, ray_dir, hit_n, eta_r=material_inputs['eta_r'], sigma=material_inputs['sigma'], omega=wt.Float(2.0 * math.pi * SPEED_OF_LIGHT / wave.wavelength), gain=material_inputs['gain'], mu_r=material_inputs['mu_r'])
        surface_edges = scene.get_triangle_surface_edge_candidates(prim_idx_i32)
        candidate_globals = list(surface_edges['slots'])
        candidate_valids = [edge_idx >= 0 for edge_idx in candidate_globals]
        per_candidate_states = []
        for slot, global_edge_idx in enumerate(candidate_globals):
            slot_valid = candidate_valids[slot]
            for prev_slot in range(slot):
                slot_valid = slot_valid & ((global_edge_idx != candidate_globals[prev_slot]) | ~candidate_valids[prev_slot])
            safe_global_idx = wt.UInt32(dr.select(slot_valid, global_edge_idx, wt.Int32(0)))
            local_edge_idx_i32 = dr.gather(wt.Int32, global_to_local_idx, safe_global_idx)
            slot_valid = slot_valid & (local_edge_idx_i32 >= 0)
            slot_valid = slot_valid & (local_edge_idx_i32 != wt.Int32(batch_state_edge_idx))
            if not dr.any(slot_valid):
                continue
            local_edge_idx = wt.UInt32(dr.select(slot_valid, local_edge_idx_i32, wt.Int32(0)))
            edge_pos = dr.gather(wt.Point3f, edge_data['pos'], local_edge_idx)
            n0 = dr.gather(wt.Vector3f, edge_data['n0'], local_edge_idx)
            adjacent_face0 = dr.gather(wt.Int32, edge_data['adjacent_face0'], local_edge_idx)
            adjacent_face1 = dr.gather(wt.Int32, edge_data['adjacent_face1'], local_edge_idx)
            visible = scene.segment_visible(hit_p, edge_pos, ignore_prim_idx=(prim_idx_i32, adjacent_face0, adjacent_face1))
            slot_valid = slot_valid & visible
            visible_count = int(dr.width(dr.compress(slot_valid)))
            if visible_count == 0:
                continue
            incident_field = Geo.source_field(hit_p, reflected_field, edge_pos, wave)
            incident_normal_derivative = Geo.source_field_normal_derivative(hit_p, reflected_field, edge_pos, n0, wave)
            unit_incident_field = Geo.source_field(hit_p, wt.Complex2f(1.0, 0.0), edge_pos, wave)
            unit_incident_normal_derivative = Geo.source_field_normal_derivative(hit_p, wt.Complex2f(1.0, 0.0), edge_pos, n0, wave)
            incident_vector = vector_scale(reflected_vector, unit_incident_field)
            incident_normal_derivative_vector = vector_scale(reflected_vector, unit_incident_normal_derivative)
            field_power = vector_power(incident_vector)
            keep_idx = dr.compress(slot_valid & (field_power > wt.Float(1e-20)))
            if dr.width(keep_idx) == 0:
                continue
            keep_state_idx = dr.gather(wt.UInt32, hit_state_idx, keep_idx)
            kept_batch_states = gather_inserted_reflection_state_fields(prev_state_arrays, keep_state_idx)
            keep_edge_idx = dr.gather(wt.UInt32, local_edge_idx, keep_idx)
            keep_edge_pos = dr.gather(wt.Point3f, edge_pos, keep_idx)
            keep_edge_dir = dr.gather(wt.Vector3f, edge_data['edge_dir'], keep_edge_idx)
            keep_n0 = dr.gather(wt.Vector3f, n0, keep_idx)
            keep_nn = dr.gather(wt.Vector3f, edge_data['n_face_n'], keep_edge_idx)
            keep_wedge_n = dr.gather(wt.Float, edge_data['wedge_n'], keep_edge_idx)
            keep_edge_line_min = dr.gather(wt.Float, line_min_all, keep_edge_idx)
            keep_edge_line_max = dr.gather(wt.Float, line_max_all, keep_edge_idx)
            keep_adjacent_face0 = dr.gather(wt.Int32, adjacent_face0, keep_idx)
            keep_adjacent_face1 = dr.gather(wt.Int32, adjacent_face1, keep_idx)
            keep_source_pos = wt.Point3f(dr.gather(wt.Float, hit_p.x, keep_idx), dr.gather(wt.Float, hit_p.y, keep_idx), dr.gather(wt.Float, hit_p.z, keep_idx))
            keep_prev_edge_pos = kept_batch_states['edge_pos']
            keep_path_length_prefix = None
            keep_first_interaction_pos = None
            if retain_lineage_state:
                keep_prev_path_length = kept_batch_states['path_length_prefix']
                keep_path_length_prefix = keep_prev_path_length + dr.norm(keep_source_pos - keep_prev_edge_pos) + dr.norm(keep_edge_pos - keep_source_pos)
                keep_first_interaction_pos = kept_batch_states['first_interaction_pos']
            keep_incident_field = wt.Complex2f(dr.gather(wt.Float, incident_field.real, keep_idx), dr.gather(wt.Float, incident_field.imag, keep_idx))
            keep_incident_normal_derivative = wt.Complex2f(dr.gather(wt.Float, incident_normal_derivative.real, keep_idx), dr.gather(wt.Float, incident_normal_derivative.imag, keep_idx))
            keep_incident_vector = {'x': dr.gather(wt.Complex2f, incident_vector['x'], keep_idx), 'y': dr.gather(wt.Complex2f, incident_vector['y'], keep_idx), 'z': dr.gather(wt.Complex2f, incident_vector['z'], keep_idx)}
            keep_incident_normal_derivative_vector = {'x': dr.gather(wt.Complex2f, incident_normal_derivative_vector['x'], keep_idx), 'y': dr.gather(wt.Complex2f, incident_normal_derivative_vector['y'], keep_idx), 'z': dr.gather(wt.Complex2f, incident_normal_derivative_vector['z'], keep_idx)}
            face0_operator, face1_operator, face0_material, face1_material = face_response(edge_pos=keep_edge_pos, edge_dir=keep_edge_dir, n0=keep_n0, nn=keep_nn, source_pos=keep_source_pos, adjacent_face0=keep_adjacent_face0, adjacent_face1=keep_adjacent_face1, wave=wave, scene=scene, material=material)
            keep_prev_order = kept_batch_states['order']
            keep_order = keep_prev_order + 1
            keep_parent_state_id = State.ids(kept_batch_states)
            keep_lineage_store = State.lineage_store(kept_batch_states)
            keep_source_type_code = kept_batch_states['source_type_code'] if retain_lineage_state else None
            keep_prefix_reflection_depth = kept_batch_states['prefix_reflection_depth']
            keep_intermediate_reflection_depth = kept_batch_states['intermediate_reflection_depth'] + wt.UInt32(1)
            keep_suffix_reflection_depth = kept_batch_states['suffix_reflection_depth']
            per_candidate_states.append(State.make(edge_idx=keep_edge_idx, edge_pos=keep_edge_pos, edge_dir=keep_edge_dir, n0=keep_n0, nn=keep_nn, wedge_n=keep_wedge_n, adjacent_face0=keep_adjacent_face0, adjacent_face1=keep_adjacent_face1, source_pos=keep_source_pos, path_length_prefix=keep_path_length_prefix, first_interaction_pos=keep_first_interaction_pos, edge_line_min=keep_edge_line_min, edge_line_max=keep_edge_line_max, incident_field=keep_incident_field, incident_normal_derivative=keep_incident_normal_derivative, incident_vector=keep_incident_vector, incident_normal_derivative_vector=keep_incident_normal_derivative_vector, is_direct_tx=dr.full(wt.Bool, False, dr.width(keep_idx)) if retain_lineage_state else None, face0_operator=face0_operator, face1_operator=face1_operator, face0_material=face0_material, face1_material=face1_material, source_type_code=keep_source_type_code, prefix_reflection_depth=keep_prefix_reflection_depth, intermediate_reflection_depth=keep_intermediate_reflection_depth, suffix_reflection_depth=keep_suffix_reflection_depth, approximation_mode_code=dr.full(wt.UInt32, APPROX_MODE_SAMPLED_INSERTED_REFLECTION, dr.width(keep_idx)), order=keep_order, lineage_parent_state_id=keep_parent_state_id, lineage_last_edge_idx=wt.Int32(keep_edge_idx), lineage_last_reflection_depth_delta=dr.full(wt.UInt32, 1, dr.width(keep_idx)), lineage_store=keep_lineage_store, retain_lineage_state=retain_lineage_state))
        if len(per_candidate_states) > 0:
            chunk_states.append(concat_state_arrays(per_candidate_states))
    result = concat_state_arrays(chunk_states)
    return result

__all__ = [
    'face_response', 'prepare', 'tx_first', 'prefix_first',
    'higher', 'inserted',
]
