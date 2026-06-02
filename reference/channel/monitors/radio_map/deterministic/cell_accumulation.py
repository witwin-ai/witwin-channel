from __future__ import annotations

import drjit as dr
import witwin as wt

from ....kernels.trace.packed_state import (
    gather_field_evaluation_state_fields,
    gather_state_arrays,
)
from ....kernels.monitors.common.receiver_tiles.native_impl import receiver_index_for_tile_slot
from ....kernels.monitors.field.radio_map_accumulate.native_impl import (
    radiomap_accumulate_vector_power_pairs,
    radiomap_matched_isb_completion,
    radiomap_shadow_boundary_incident_statistics,
)
from ....kernels.trace.utd.native_impl import (
    _zero_pair_output_buffers,
    _utd_accumulate_tiled_vector_power_into,
    utd_accumulate_scalar_power_pairs,
)
from ....trace.diffraction.constants import _cartesian_chunk_size, _ownership_code_from_depths
from ....trace.diffraction.field import (
    _edge_state_field_to_targets,
    _edge_state_target_support,
    _finite_wedge_truncation_factor,
)
from ....trace.diffraction.finite_wedge import (
    require_edge_data_line_bounds,
    require_edge_state_line_bounds,
)
from ....trace.diffraction.geometry import (
    DIFFRACTION_MIN_DISTANCE,
    _compute_edge_geometry,
    _edge_owner_structure_idx,
    _point_inside_closed_mesh_mask,
    _point_source_field,
    _segment_visibility_mask,
    _wedge_exterior_region_mask,
)
from ....trace.diffraction.state import (
    gather_path_export_eval_state_fields,
    gather_path_export_field_state_fields,
    gather_path_export_support_state_fields,
    is_path_export_reduced_state_arrays,
)
from ....trace.diffraction.utd import _compute_a_pm, f_utd
from ....trace.reflection.epc import (
    build_reflection_epc_descriptor,
    epc_reflection_chain_to_target,
)
from ....trace.materials import coerce_reflection_trace_detail
from ....utils.constants import EPS
from ....utils.drjit_ops import (
    ArrayInit,
    Broadcast,
    Gather,
    complex_abs_sqr,
    eval_complex,
    repeat_float,
)
from ....utils.polarization import (
    complex_dot_real,
    effective_rx_polarization,
    project_real_polarization_to_ray,
    scalarize_vector_to_polarization,
    vector_from_scalar_and_real_direction,
    vector_scale,
)
from ....utils.shadow_support import (
    shadow_completion_weight_from_distance,
    shadow_decay_span_from_wedge_n,
    shadow_support_amplitude_threshold,
)
from ..backend import _point_grad_enabled
from ..diagnostics import MATCHED_ISB_COMPLETION_BOUNDARY_HALF_FIELD
from ..metadata import (
    _empty_diffraction_diagnostic_counts,
    _merge_diffraction_diagnostic_counts,
)
from .scheduler import (
    resolve_radio_map_receiver_tiles,
    select_radio_map_diffraction_receiver_tiles,
    select_radio_map_reflection_family_tiles,
)


# Radio-map diffraction replay carries more per-pair state than the generic UTD
# monitor kernels, but 256x256 receiver maps benefit from larger chunks to keep
# the matched-isotropic coherent path off the Python launch treadmill. Keep the
# smaller budget only for denser 512^2-style workloads.
_RADIO_MAP_DIFFRACTION_PAIR_CHUNK_BUDGET_SMALL_GRID = 1 << 25
_RADIO_MAP_DIFFRACTION_PAIR_CHUNK_BUDGET_LARGE_GRID = 1 << 24
_RADIO_MAP_DIFFRACTION_FAST_GRID_RX_THRESHOLD = 256 * 256
_MATCHED_ISB_INCIDENT_WEIGHT_AGGREGATION = "clamped_sum_incident_weight"


def _radio_map_diffraction_pair_chunk_budget(receiver_count: int) -> int:
    if max(0, int(receiver_count)) <= int(_RADIO_MAP_DIFFRACTION_FAST_GRID_RX_THRESHOLD):
        return int(_RADIO_MAP_DIFFRACTION_PAIR_CHUNK_BUDGET_SMALL_GRID)
    return int(_RADIO_MAP_DIFFRACTION_PAIR_CHUNK_BUDGET_LARGE_GRID)


def _mask_count(mask) -> int:
    width = 0 if mask is None else int(dr.width(mask))
    if width <= 0:
        return 0
    ones = dr.select(mask, wt.UInt32(1), wt.UInt32(0))
    return int(dr.slice(dr.sum(ones)))


def _radio_map_diffraction_chunk_size(
    n_states: int,
    n_rx: int,
    *,
    pair_chunk_budget: int | None = None,
) -> int:
    left = max(0, int(n_states))
    right = max(0, int(n_rx))
    if left == 0 or right == 0:
        return 0
    budget = (
        _radio_map_diffraction_pair_chunk_budget(right)
        if pair_chunk_budget is None
        else max(1, int(pair_chunk_budget))
    )
    return max(
        1,
        min(
            left,
            int(budget) // right,
        ),
    )


def _utd_transition_weight(x):
    transition_mag = dr.sqrt(complex_abs_sqr(f_utd(x)))
    return dr.maximum(
        wt.Float(0.0),
        wt.Float(1.0) - dr.minimum(transition_mag, wt.Float(1.0)),
    )


def _shadow_boundary_edge_line_bounds(scene, edge_runtime, edge_idx):
    explicit_line_min, explicit_line_max = require_edge_data_line_bounds(
        edge_runtime,
        context="_shadow_boundary_edge_line_bounds",
    )
    return (
        dr.gather(wt.Float, explicit_line_min, edge_idx),
        dr.gather(wt.Float, explicit_line_max, edge_idx),
    )


def _shadow_boundary_edge_batch_state(*, scene, edge_runtime, edge_idx, zero, tx_pos):
    edge_pos_x = dr.gather(wt.Float, edge_runtime["pos"].x, edge_idx)
    edge_pos_y = dr.gather(wt.Float, edge_runtime["pos"].y, edge_idx)
    edge_pos_z = dr.gather(wt.Float, edge_runtime["pos"].z, edge_idx)
    edge_dir_x = dr.gather(wt.Float, edge_runtime["edge_dir"].x, edge_idx)
    edge_dir_y = dr.gather(wt.Float, edge_runtime["edge_dir"].y, edge_idx)
    edge_dir_z = dr.gather(wt.Float, edge_runtime["edge_dir"].z, edge_idx)
    n0_x = dr.gather(wt.Float, edge_runtime["n0"].x, edge_idx)
    n0_y = dr.gather(wt.Float, edge_runtime["n0"].y, edge_idx)
    n0_z = dr.gather(wt.Float, edge_runtime["n0"].z, edge_idx)
    nn_x = dr.gather(wt.Float, edge_runtime["n_face_n"].x, edge_idx)
    nn_y = dr.gather(wt.Float, edge_runtime["n_face_n"].y, edge_idx)
    nn_z = dr.gather(wt.Float, edge_runtime["n_face_n"].z, edge_idx)
    wedge_n = dr.gather(wt.Float, edge_runtime["wedge_n"], edge_idx)
    adjacent_face0 = (
        dr.gather(wt.Int32, edge_runtime["adjacent_face0"], edge_idx)
        if edge_runtime.get("adjacent_face0") is not None
        else wt.Int32(-1)
    )
    adjacent_face1 = (
        dr.gather(wt.Int32, edge_runtime["adjacent_face1"], edge_idx)
        if edge_runtime.get("adjacent_face1") is not None
        else wt.Int32(-1)
    )
    edge_line_min, edge_line_max = _shadow_boundary_edge_line_bounds(
        scene,
        edge_runtime,
        edge_idx,
    )
    source_visible = _segment_visibility_mask(
        tx_pos,
        wt.Point3f(edge_pos_x, edge_pos_y, edge_pos_z),
        scene,
        ignore_prim_idx=(adjacent_face0, adjacent_face1),
    )
    batch_state = {
        "edge_dir": wt.Vector3f(
            zero + edge_dir_x,
            zero + edge_dir_y,
            zero + edge_dir_z,
        ),
        "edge_pos": wt.Point3f(
            zero + edge_pos_x,
            zero + edge_pos_y,
            zero + edge_pos_z,
        ),
        "source_pos": tx_pos,
        "n0": wt.Vector3f(
            zero + n0_x,
            zero + n0_y,
            zero + n0_z,
        ),
        "n_face_n": wt.Vector3f(
            zero + nn_x,
            zero + nn_y,
            zero + nn_z,
        ),
        "wedge_n": zero + wedge_n,
        "source_visible": source_visible,
        "edge_line_min": edge_line_min,
        "edge_line_max": edge_line_max,
    }
    return batch_state, wedge_n


def _shadow_boundary_transition_support_mask(batch_states, batch_rx):
    width = dr.width(batch_rx.x)
    support = dr.full(wt.Bool, True, width)
    nn = batch_states.get("n_face_n")
    if nn is not None:
        source_exterior = _wedge_exterior_region_mask(
            batch_states["source_pos"] - batch_states["edge_pos"],
            batch_states["edge_dir"],
            batch_states["n0"],
            nn,
        )
        target_exterior = _wedge_exterior_region_mask(
            batch_rx - batch_states["edge_pos"],
            batch_states["edge_dir"],
            batch_states["n0"],
            nn,
        )
        support = support & source_exterior & target_exterior
    source_visible = batch_states.get("source_visible")
    if source_visible is not None:
        source_visible_b = source_visible if dr.width(source_visible) == width else dr.repeat(source_visible, width)
        support = support & source_visible_b
    return support


def _gather_shadow_transition_state_fields(batch_states, indices):
    edge_line_min, edge_line_max = require_edge_state_line_bounds(
        batch_states,
        context="_gather_shadow_transition_state_fields",
    )
    gathered = {
        "edge_pos": Gather.point3(batch_states["edge_pos"], indices),
        "edge_dir": Gather.vector3(batch_states["edge_dir"], indices),
        "n0": Gather.vector3(batch_states["n0"], indices),
        "n_face_n": Gather.vector3(batch_states["n_face_n"], indices),
        "wedge_n": dr.gather(wt.Float, batch_states["wedge_n"], indices),
        "source_pos": Gather.point3(batch_states["source_pos"], indices),
        "edge_line_min": dr.gather(wt.Float, edge_line_min, indices),
        "edge_line_max": dr.gather(wt.Float, edge_line_max, indices),
    }
    return gathered


def _shadow_boundary_transition_responses(
    batch_states,
    batch_rx,
    *,
    k: float,
    include_reflection: bool = True,
):
    width = dr.width(batch_rx.x)
    edge_dir = batch_states["edge_dir"]
    source_to_edge = batch_states["edge_pos"] - batch_states["source_pos"]
    source_to_edge_proj = source_to_edge - dr.dot(source_to_edge, edge_dir) * edge_dir
    s_prime_proj = dr.norm(source_to_edge_proj) + EPS
    to_hat = dr.normalize(dr.cross(batch_states["n0"], edge_dir))
    ki_proj = source_to_edge_proj / s_prime_proj
    phi_prime = dr.pi - dr.safe_acos(dr.clip(-dr.dot(ki_proj, to_hat), -1.0, 1.0))
    phi_prime = phi_prime * (-dr.sign(-dr.dot(ki_proj, batch_states["n0"])))
    phi_prime = phi_prime + dr.pi

    edge_to_target = batch_rx - batch_states["edge_pos"]
    edge_to_target_proj = edge_to_target - dr.dot(edge_to_target, edge_dir) * edge_dir
    s_proj = dr.norm(edge_to_target_proj) + EPS
    ko_proj = edge_to_target_proj / s_proj
    phi = dr.pi - dr.safe_acos(dr.clip(dr.dot(ko_proj, to_hat), -1.0, 1.0))
    phi = phi * (-dr.sign(dr.dot(ko_proj, batch_states["n0"])))
    phi = phi + dr.pi

    s = dr.norm(edge_to_target) + EPS
    s_prime = dr.norm(source_to_edge) + EPS
    wedge_n = batch_states["wedge_n"]
    kL = wt.Float(k) * s * s_prime * dr.rcp(s + s_prime)
    dif_phi = phi - phi_prime
    inc_a0, inc_a1 = _compute_a_pm(dif_phi, wedge_n)
    incident_transition = f_utd(kL * dr.minimum(inc_a0, inc_a1))
    incident_weight = _utd_transition_weight(kL * dr.minimum(inc_a0, inc_a1))
    if include_reflection:
        sum_phi = phi + phi_prime
        ref_a0, ref_a1 = _compute_a_pm(sum_phi, wedge_n)
        reflection_transition = f_utd(kL * dr.minimum(ref_a0, ref_a1))
        reflection_weight = _utd_transition_weight(kL * dr.minimum(ref_a0, ref_a1))
    else:
        reflection_transition = ArrayInit.complex_zero(width)
        reflection_weight = dr.zeros(wt.Float, width)
    require_edge_state_line_bounds(
        batch_states,
        context="_shadow_boundary_transition_responses",
    )
    finite_wedge_factor = _finite_wedge_truncation_factor(
        batch_states,
        {
            "edge_hat": edge_dir,
            "s_prime_proj": s_prime_proj,
            "s_proj": s_proj,
        },
        batch_rx,
        k,
        width=width,
    )
    finite_wedge_scale = dr.minimum(
        dr.sqrt(complex_abs_sqr(finite_wedge_factor)),
        wt.Float(1.0),
    )
    incident_transition = finite_wedge_factor * incident_transition
    incident_weight = incident_weight * finite_wedge_scale
    if include_reflection:
        reflection_transition = finite_wedge_factor * reflection_transition
        reflection_weight = reflection_weight * finite_wedge_scale
    support_mask = _shadow_boundary_transition_support_mask(batch_states, batch_rx)
    zero = ArrayInit.complex_zero(width)
    incident_transition = dr.select(support_mask, incident_transition, zero)
    incident_weight = dr.select(support_mask, incident_weight, wt.Float(0.0))
    if include_reflection:
        reflection_transition = dr.select(support_mask, reflection_transition, zero)
        reflection_weight = dr.select(support_mask, reflection_weight, wt.Float(0.0))
    return (
        incident_transition,
        reflection_transition,
        incident_weight,
        reflection_weight,
    )


def _shadow_boundary_transition_weights(batch_states, batch_rx, *, k: float):
    _, _, incident_weight, reflection_weight = _shadow_boundary_transition_responses(
        batch_states,
        batch_rx,
        k=k,
    )
    return (
        incident_weight,
        reflection_weight,
    )


def _matched_isb_continuous_incident_weight(sum_incident_weight):
    return dr.minimum(
        dr.maximum(sum_incident_weight, wt.Float(0.0)),
        wt.Float(1.0),
    )


def _reference_accumulate_shadow_boundary_incident_statistics(
    *,
    rx_pos,
    scene,
    tx_pos,
    k: float,
    include_response: bool,
):
    n_rx = int(dr.width(rx_pos.x))
    sum_incident_weight = dr.zeros(wt.Float, n_rx)
    max_incident_weight = dr.zeros(wt.Float, n_rx)
    weighted_incident_response_real = dr.zeros(wt.Float, n_rx) if include_response else None
    weighted_incident_response_imag = dr.zeros(wt.Float, n_rx) if include_response else None
    edge_runtime = getattr(scene, "_diffraction_edge_gpu", None)
    n_edges = 0 if edge_runtime is None else int(edge_runtime.get("n_edges", 0))
    if n_rx <= 0 or n_edges <= 0:
        return {
            "n_edges": int(n_edges),
            "sum_incident_weight": sum_incident_weight,
            "max_incident_weight": max_incident_weight,
            "weighted_incident_response_real": weighted_incident_response_real,
            "weighted_incident_response_imag": weighted_incident_response_imag,
        }

    edge_chunk_size = max(1, _cartesian_chunk_size(n_edges, n_rx))
    for edge_start in range(0, n_edges, edge_chunk_size):
        chunk_n_edges = min(edge_chunk_size, n_edges - edge_start)
        n_pairs = chunk_n_edges * n_rx
        pair_idx = dr.arange(wt.UInt32, n_pairs)
        rx_idx = pair_idx // chunk_n_edges
        edge_idx = pair_idx % chunk_n_edges + wt.UInt32(edge_start)
        batch_rx = _gather_positions(rx_pos, rx_idx)
        batch_state, wedge_n = _shadow_boundary_edge_batch_state(
            scene=scene,
            edge_runtime=edge_runtime,
            edge_idx=edge_idx,
            zero=dr.zeros(wt.Float, n_pairs),
            tx_pos=tx_pos,
        )
        if include_response:
            incident_response, _, incident_weight, _ = _shadow_boundary_transition_responses(
                batch_state,
                batch_rx,
                k=k,
                include_reflection=False,
            )
        else:
            _, _, incident_weight, _ = _shadow_boundary_transition_responses(
                batch_state,
                batch_rx,
                k=k,
                include_reflection=False,
            )
            incident_response = None
        incident_weight = dr.select(
            wedge_n > wt.Float(1.01),
            incident_weight,
            wt.Float(0.0),
        )
        sum_incident_weight = sum_incident_weight + dr.block_reduce(
            dr.ReduceOp.Add,
            incident_weight,
            int(chunk_n_edges),
            mode="symbolic",
        )
        max_incident_weight = dr.maximum(
            max_incident_weight,
            dr.block_reduce(
                dr.ReduceOp.Max,
                incident_weight,
                int(chunk_n_edges),
                mode="symbolic",
            ),
        )
        if include_response:
            weighted_incident_response_real = (
                weighted_incident_response_real
                + dr.block_reduce(
                    dr.ReduceOp.Add,
                    incident_weight * incident_response.real,
                    int(chunk_n_edges),
                    mode="symbolic",
                )
            )
            weighted_incident_response_imag = (
                weighted_incident_response_imag
                + dr.block_reduce(
                    dr.ReduceOp.Add,
                    incident_weight * incident_response.imag,
                    int(chunk_n_edges),
                    mode="symbolic",
                )
            )

    return {
        "n_edges": int(n_edges),
        "sum_incident_weight": sum_incident_weight,
        "max_incident_weight": max_incident_weight,
        "weighted_incident_response_real": weighted_incident_response_real,
        "weighted_incident_response_imag": weighted_incident_response_imag,
    }


def _accumulate_shadow_boundary_incident_statistics(
    *,
    rx_pos,
    scene,
    tx_pos,
    k: float,
    include_response: bool,
):
    n_rx = int(dr.width(rx_pos.x))
    edge_runtime = getattr(scene, "_diffraction_edge_gpu", None)
    n_edges = 0 if edge_runtime is None else int(edge_runtime.get("n_edges", 0))
    zero_float = dr.zeros(wt.Float, n_rx)
    if n_rx <= 0 or n_edges <= 0:
        return {
            "n_edges": int(n_edges),
            "sum_incident_weight": zero_float,
            "max_incident_weight": zero_float,
            "weighted_incident_response_real": zero_float if include_response else None,
            "weighted_incident_response_imag": zero_float if include_response else None,
            "argmax_edge_idx": dr.full(wt.Int32, -1, n_rx),
            "second_max_incident_weight": zero_float,
            "support_edge_count": dr.zeros(wt.Int32, n_rx),
            "argmax_margin": zero_float,
        }

    explicit_line_min, explicit_line_max = require_edge_data_line_bounds(
        edge_runtime,
        context="_accumulate_shadow_boundary_incident_statistics",
    )
    adjacent_face0 = edge_runtime.get("adjacent_face0")
    adjacent_face1 = edge_runtime.get("adjacent_face1")
    adjacent_surface_group0 = edge_runtime.get("adjacent_surface_group0")
    adjacent_surface_group1 = edge_runtime.get("adjacent_surface_group1")
    ignore_prim_idx = None
    ignore_surface_group_idx = None
    if adjacent_surface_group0 is not None or adjacent_surface_group1 is not None:
        ignore_surface_group_idx = tuple(
            value
            for value in (adjacent_surface_group0, adjacent_surface_group1)
            if value is not None
        )
    if adjacent_face0 is not None or adjacent_face1 is not None:
        ignore_prim_idx = tuple(
            value
            for value in (adjacent_face0, adjacent_face1)
            if value is not None
        )
    source_visible = _segment_visibility_mask(
        tx_pos,
        edge_runtime["pos"],
        scene,
        ignore_prim_idx=ignore_prim_idx,
        ignore_surface_group_idx=ignore_surface_group_idx,
    )
    stats = radiomap_shadow_boundary_incident_statistics(
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        edge_pos=edge_runtime["pos"],
        edge_dir=edge_runtime["edge_dir"],
        n0=edge_runtime["n0"],
        n_face_n=edge_runtime["n_face_n"],
        wedge_n=edge_runtime["wedge_n"],
        edge_line_min=explicit_line_min,
        edge_line_max=explicit_line_max,
        source_visible=source_visible,
        k=k,
        include_diagnostics=True,
    )
    if not include_response:
        stats = dict(stats)
        stats["weighted_incident_response_real"] = None
        stats["weighted_incident_response_imag"] = None
    stats["n_edges"] = int(n_edges)
    return stats


def accumulate_projected_isb_shadow_completion(
    *,
    rx_pos,
    scene,
    tx_pos,
    wavelength: float,
    k: float,
    tx_polarization=(1.0, 0.0, 0.0),
    rx_polarization=None,
    los_coherent,
    diffraction_coherent,
    ratio_target: float = 0.55,
    completion_gain: float = 1.0,
):
    n_rx = int(dr.width(rx_pos.x))
    zero_float = dr.zeros(wt.Float, n_rx)
    zero_complex = ArrayInit.complex_zero(n_rx)
    empty_payload = {
        "coherent": zero_complex,
        "power": zero_float,
        "incident_weight": zero_float,
        "deficiency": zero_float,
        "continued_direct_power": zero_float,
        "amplitude_ratio": zero_float,
    }
    if (
        n_rx <= 0
        or scene is None
        or los_coherent is None
        or diffraction_coherent is None
        or float(ratio_target) <= 0.0
        or float(completion_gain) <= 0.0
    ):
        dr.eval(
            zero_complex.real,
            zero_complex.imag,
            zero_float,
        )
        return empty_payload

    edge_runtime = getattr(scene, "_diffraction_edge_gpu", None)
    n_edges = 0 if edge_runtime is None else int(edge_runtime.get("n_edges", 0))
    if n_edges <= 0:
        dr.eval(
            zero_complex.real,
            zero_complex.imag,
            zero_float,
        )
        return empty_payload

    ray_dir = rx_pos - tx_pos
    distance = dr.norm(ray_dir) + EPS
    continued_direct = (
        wt.Float(wavelength) / (4.0 * dr.pi * distance)
    ) * dr.exp(wt.Complex2f(0.0, -wt.Float(k) * distance))
    tx_pol_dir = project_real_polarization_to_ray(tx_polarization, ray_dir)
    continued_direct_vector = vector_from_scalar_and_real_direction(
        continued_direct,
        tx_pol_dir,
    )
    active_rx_polarization = effective_rx_polarization(
        rx_polarization,
        tx_polarization,
    )
    continued_direct = eval_complex(
        scalarize_vector_to_polarization(
            continued_direct_vector,
            ray_dir,
            active_rx_polarization,
        )
    )
    continued_direct_power = complex_abs_sqr(continued_direct)
    diffraction_power = complex_abs_sqr(diffraction_coherent)
    amplitude_ratio = dr.sqrt(
        diffraction_power / dr.maximum(continued_direct_power, wt.Float(1.0e-20))
    )
    ratio_target_value = wt.Float(float(ratio_target))
    deficiency = dr.maximum(
        wt.Float(0.0),
        dr.minimum(
            wt.Float(1.0),
            (ratio_target_value - amplitude_ratio) / ratio_target_value,
        ),
    )
    shadow_mask = complex_abs_sqr(los_coherent) <= wt.Float(1.0e-14)
    interior_mask = _point_inside_closed_mesh_mask(
        rx_pos,
        scene,
        active=shadow_mask,
    )
    stats = _accumulate_shadow_boundary_incident_statistics(
        rx_pos=rx_pos,
        scene=scene,
        tx_pos=tx_pos,
        k=k,
        include_response=False,
    )
    max_incident_weight = stats["max_incident_weight"]

    completion_scale = dr.select(
        shadow_mask & ~interior_mask,
        max_incident_weight * deficiency * wt.Float(float(completion_gain)),
        wt.Float(0.0),
    )
    completion = eval_complex(
        wt.Complex2f(
            continued_direct.real * completion_scale,
            continued_direct.imag * completion_scale,
        )
    )
    completion_power = complex_abs_sqr(completion)
    dr.eval(
        completion.real,
        completion.imag,
        completion_power,
        max_incident_weight,
        deficiency,
        continued_direct_power,
        amplitude_ratio,
    )
    return {
        "coherent": completion,
        "power": completion_power,
        "incident_weight": max_incident_weight,
        "deficiency": deficiency,
        "continued_direct_power": continued_direct_power,
        "amplitude_ratio": amplitude_ratio,
    }


def accumulate_matched_isb_shadow_completion(
    *,
    rx_pos,
    scene,
    tx_pos,
    wavelength: float,
    k: float,
    tx_polarization=(1.0, 0.0, 0.0),
    rx_polarization=None,
    los_vector_coherent,
    raw_transition_vector,
):
    n_rx = int(dr.width(rx_pos.x))
    zero_float = dr.zeros(wt.Float, n_rx)
    zero_complex = ArrayInit.complex_zero(n_rx)
    zero_vector = _complex_vector_zero(n_rx)
    negative_one_float = dr.full(wt.Float, -1.0, n_rx)
    empty_payload = {
        "coherent": zero_complex,
        "vector_coherent": zero_vector,
        "power": zero_float,
        "incident_weight": zero_float,
        "sum_incident_weight": zero_float,
        "max_incident_weight": zero_float,
        "argmax_margin": zero_float,
        "support_edge_count": zero_float,
        "argmax_edge_idx": negative_one_float,
        "incident_weight_aggregation": _MATCHED_ISB_INCIDENT_WEIGHT_AGGREGATION,
        "continued_direct_power": zero_float,
        "hard_visibility": zero_float,
        "transition_magnitude": zero_float,
        "transition_phase": zero_float,
        "diagnostic_counts": _empty_diffraction_diagnostic_counts(),
    }
    if n_rx <= 0 or scene is None:
        dr.eval(
            zero_complex.real,
            zero_complex.imag,
            zero_vector["x"].real,
            zero_vector["x"].imag,
            zero_vector["y"].real,
            zero_vector["y"].imag,
            zero_vector["z"].real,
            zero_vector["z"].imag,
            zero_float,
        )
        return empty_payload
    if los_vector_coherent is None or raw_transition_vector is None:
        raise RuntimeError(
            "matched_isb_completion requires both los_vector_coherent and raw_transition_vector."
        )

    edge_runtime = getattr(scene, "_diffraction_edge_gpu", None)
    n_edges = 0 if edge_runtime is None else int(edge_runtime.get("n_edges", 0))
    if n_edges <= 0:
        dr.eval(
            zero_complex.real,
            zero_complex.imag,
            zero_vector["x"].real,
            zero_vector["x"].imag,
            zero_vector["y"].real,
            zero_vector["y"].imag,
            zero_vector["z"].real,
            zero_vector["z"].imag,
            zero_float,
        )
        return empty_payload
    ray_dir = rx_pos - tx_pos
    distance = dr.norm(ray_dir) + EPS
    continued_direct = (
        wt.Float(wavelength) / (4.0 * dr.pi * distance)
    ) * dr.exp(wt.Complex2f(0.0, -wt.Float(k) * distance))
    tx_pol_dir = project_real_polarization_to_ray(tx_polarization, ray_dir)
    continued_direct_vector = vector_from_scalar_and_real_direction(
        continued_direct,
        tx_pol_dir,
    )
    active_rx_polarization = effective_rx_polarization(
        rx_polarization,
        tx_polarization,
    )
    hard_visibility = dr.select(
        _vector_power(los_vector_coherent) > wt.Float(1.0e-14),
        wt.Float(1.0),
        wt.Float(0.0),
    )
    shadow_mask = hard_visibility <= wt.Float(0.0)
    interior_mask = _point_inside_closed_mesh_mask(
        rx_pos,
        scene,
        active=shadow_mask,
    )
    stats = _accumulate_shadow_boundary_incident_statistics(
        rx_pos=rx_pos,
        scene=scene,
        tx_pos=tx_pos,
        k=k,
        include_response=True,
    )
    scene_sum_incident_weight = stats["sum_incident_weight"]
    scene_max_incident_weight = stats["max_incident_weight"]
    scene_weighted_incident_response_real = stats["weighted_incident_response_real"]
    scene_weighted_incident_response_imag = stats["weighted_incident_response_imag"]

    safe_sum_weight = dr.maximum(scene_sum_incident_weight, wt.Float(1.0e-6))
    scene_incident_response = wt.Complex2f(
        dr.select(
            scene_sum_incident_weight > wt.Float(1.0e-6),
            scene_weighted_incident_response_real / safe_sum_weight,
            wt.Float(1.0),
        ),
        dr.select(
            scene_sum_incident_weight > wt.Float(1.0e-6),
            scene_weighted_incident_response_imag / safe_sum_weight,
            wt.Float(0.0),
        ),
    )
    aggregated_incident_response = scene_incident_response
    aggregate_incident_weight = _matched_isb_continuous_incident_weight(
        scene_sum_incident_weight
    )

    rx_pol_dir = project_real_polarization_to_ray(active_rx_polarization, ray_dir)
    completion_payload = radiomap_matched_isb_completion(
        continued_direct=continued_direct,
        tx_basis=tx_pol_dir,
        rx_basis=rx_pol_dir,
        hard_visibility=hard_visibility,
        interior_mask=interior_mask,
        incident_weight=aggregate_incident_weight,
        incident_response=aggregated_incident_response,
        raw_transition_vector=raw_transition_vector,
    )
    return {
        "coherent": completion_payload["coherent"],
        "vector_coherent": _eval_complex_vector(completion_payload["vector_coherent"]),
        "power": completion_payload["power"],
        "incident_weight": aggregate_incident_weight,
        "sum_incident_weight": scene_sum_incident_weight,
        "max_incident_weight": scene_max_incident_weight,
        "argmax_margin": stats["argmax_margin"],
        "support_edge_count": wt.Float(stats["support_edge_count"]),
        "argmax_edge_idx": wt.Float(stats["argmax_edge_idx"]),
        "incident_weight_aggregation": _MATCHED_ISB_INCIDENT_WEIGHT_AGGREGATION,
        "continued_direct_power": completion_payload["continued_direct_power"],
        "hard_visibility": completion_payload["hard_visibility"],
        "transition_magnitude": completion_payload["transition_magnitude"],
        "transition_phase": completion_payload["transition_phase"],
        "diagnostic_counts": {
            "prepared_state_count": 0,
            "visible_pair_count": 0,
            "support_pair_count": 0,
            "pair_valid_count": 0,
            "shadow_completion_count": 0,
            "interior_count": _mask_count(interior_mask),
            "hard_visibility_zero_count": _mask_count(hard_visibility <= wt.Float(0.0)),
        },
    }


def _scatter_shadow_boundary_cross_term(
    target,
    reference_vector,
    pair_vector,
    rx_idx,
    weight,
    *,
    scale: float = 1.0,
):
    if (
        target is None
        or reference_vector is None
        or pair_vector is None
        or dr.width(rx_idx) == 0
    ):
        return
    if isinstance(reference_vector, dict) and isinstance(pair_vector, dict):
        reference = _gather_complex_vector(reference_vector, rx_idx)
        cross_real = (
            reference["x"].real * pair_vector["x"].real
            + reference["x"].imag * pair_vector["x"].imag
            + reference["y"].real * pair_vector["y"].real
            + reference["y"].imag * pair_vector["y"].imag
            + reference["z"].real * pair_vector["z"].real
            + reference["z"].imag * pair_vector["z"].imag
        )
    else:
        reference = dr.gather(wt.Complex2f, reference_vector, rx_idx)
        cross_real = reference.real * pair_vector.real + reference.imag * pair_vector.imag
    cross_term = 2.0 * weight * cross_real
    dr.scatter_reduce(
        dr.ReduceOp.Add,
        target,
        cross_term * float(scale),
        rx_idx,
    )


def _scatter_vector_coherent(target, field_vector, rx_idx, *, scale: float = 1.0):
    if target is None or field_vector is None or dr.width(rx_idx) == 0:
        return
    for axis in ("x", "y", "z"):
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            target[axis].real,
            field_vector[axis].real * float(scale),
            rx_idx,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            target[axis].imag,
            field_vector[axis].imag * float(scale),
            rx_idx,
        )


def _complex_vector_zero(width: int):
    return {
        axis: ArrayInit.complex_zero(width)
        for axis in ("x", "y", "z")
    }


def _eval_complex_vector(field_vector):
    if field_vector is None:
        return None
    return {
        axis: eval_complex(field_vector[axis])
        for axis in ("x", "y", "z")
    }


def _complex_grad_enabled(value) -> bool:
    if value is None:
        return False
    for component in (value.real, value.imag):
        try:
            if bool(dr.grad_enabled(component)):
                return True
        except Exception:
            continue
    return False


def _complex_vector_grad_enabled(field_vector) -> bool:
    if field_vector is None:
        return False
    return any(_complex_grad_enabled(field_vector[axis]) for axis in ("x", "y", "z"))


def _gather_complex_vector(field_vector, rx_idx):
    if field_vector is None:
        return None
    return {
        axis: dr.gather(wt.Complex2f, field_vector[axis], rx_idx)
        for axis in ("x", "y", "z")
    }


def _scatter_complex_and_power(
    coherent,
    power,
    value,
    rx_idx,
    *,
    coherent_scale: float = 1.0,
    power_scale: float = 1.0,
):
    if dr.width(rx_idx) == 0:
        return
    dr.scatter_reduce(
        dr.ReduceOp.Add,
        coherent.real,
        value.real * float(coherent_scale),
        rx_idx,
    )
    dr.scatter_reduce(
        dr.ReduceOp.Add,
        coherent.imag,
        value.imag * float(coherent_scale),
        rx_idx,
    )
    dr.scatter_reduce(
        dr.ReduceOp.Add,
        power,
        complex_abs_sqr(value) * float(power_scale),
        rx_idx,
    )


def _scatter_dense_complex_and_power(
    coherent_target,
    power_target,
    coherent_value,
    power_value,
    *,
    rx_idx=None,
    coherent_scale: float = 1.0,
    power_scale: float = 1.0,
):
    if coherent_target is None and power_target is None:
        return
    if rx_idx is None:
        if coherent_target is not None and coherent_value is not None:
            coherent_target.real = coherent_target.real + (
                coherent_value.real * float(coherent_scale)
            )
            coherent_target.imag = coherent_target.imag + (
                coherent_value.imag * float(coherent_scale)
            )
        if power_target is not None and power_value is not None:
            power_target += power_value * float(power_scale)
        return
    n_rx = int(dr.width(rx_idx))
    if n_rx <= 0:
        return
    if coherent_target is not None and coherent_value is not None:
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            coherent_target.real,
            coherent_value.real * float(coherent_scale),
            rx_idx,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            coherent_target.imag,
            coherent_value.imag * float(coherent_scale),
            rx_idx,
        )
    if power_target is not None and power_value is not None:
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            power_target,
            power_value * float(power_scale),
            rx_idx,
        )


def _scatter_dense_vector_coherent(target, field_vector, *, rx_idx=None, scale: float = 1.0):
    if target is None or field_vector is None:
        return
    if rx_idx is None:
        for axis in ("x", "y", "z"):
            target[axis].real = target[axis].real + (field_vector[axis].real * float(scale))
            target[axis].imag = target[axis].imag + (field_vector[axis].imag * float(scale))
        return
    n_rx = int(dr.width(rx_idx))
    if n_rx <= 0:
        return
    for axis in ("x", "y", "z"):
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            target[axis].real,
            field_vector[axis].real * float(scale),
            rx_idx,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            target[axis].imag,
            field_vector[axis].imag * float(scale),
            rx_idx,
        )


def _compact_output_receiver_indices(rx_idx, *, n_rx: int):
    pair_count = int(dr.width(rx_idx))
    if pair_count <= 0 or int(n_rx) <= 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)
    active = dr.full(wt.Bool, True, pair_count)
    seen = dr.zeros(wt.UInt32, int(n_rx))
    previous = dr.scatter_inc(seen, rx_idx, active)
    unique_mask = previous == wt.UInt32(0)
    unique_pair_idx = dr.compress(unique_mask)
    if dr.width(unique_pair_idx) == 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)
    unique_rx_idx = dr.gather(type(rx_idx), rx_idx, unique_pair_idx)
    next_slot = dr.zeros(wt.UInt32, 1)
    local_slot = dr.scatter_inc(next_slot, wt.UInt32(0), unique_mask)
    local_slot_lookup = dr.full(wt.UInt32, pair_count, int(n_rx))
    dr.scatter(local_slot_lookup, local_slot, rx_idx, unique_mask)
    local_output_rx_idx = dr.gather(type(local_slot_lookup), local_slot_lookup, rx_idx)
    return unique_rx_idx, local_output_rx_idx


def _densify_complex_and_power(
    coherent_value,
    power_value,
    *,
    rx_idx,
    n_rx: int,
):
    dense_coherent = ArrayInit.complex_zero(n_rx)
    dense_power = dr.zeros(wt.Float, n_rx)
    _scatter_dense_complex_and_power(
        dense_coherent,
        dense_power,
        coherent_value,
        power_value,
        rx_idx=rx_idx,
    )
    return eval_complex(dense_coherent), dense_power


def _gather_positions(rx_pos, rx_idx):
    return wt.Point3f(
        dr.gather(wt.Float, rx_pos.x, rx_idx),
        dr.gather(wt.Float, rx_pos.y, rx_idx),
        dr.gather(wt.Float, rx_pos.z, rx_idx),
    )


def _vector_power(field_vector):
    return (
        complex_abs_sqr(field_vector["x"])
        + complex_abs_sqr(field_vector["y"])
        + complex_abs_sqr(field_vector["z"])
    )


def _native_diffraction_scalar_power_pairs_enabled(receiver_model: str) -> bool:
    # The native projected-polarized pair replay path retains pair-sized device
    # buffers after function return on large radio-map workloads. Use the direct
    # Dr.Jit replay path until the native lifetime issue is fixed.
    return False


def _diffraction_scalar_backend_name(
    receiver_model: str,
    *,
    native_primal_forward: bool = False,
    native_vector_replay: bool = False,
) -> str:
    if native_primal_forward and str(receiver_model) == "matched_isotropic":
        return "native_radiomap_vector_power_forward_fast"
    if native_vector_replay and str(receiver_model) == "matched_isotropic":
        return "native_radiomap_vector_power"
    if str(receiver_model) == "matched_isotropic":
        return "direct_state_vector_power"
    if _native_diffraction_scalar_power_pairs_enabled(receiver_model):
        return "native_utd_pair_vector_replay"
    return "direct_state_scalar_power"


def _reflection_scalar_backend_name(receiver_model: str) -> str:
    if str(receiver_model) == "matched_isotropic":
        return "native_radiomap_vector_power"
    return "direct_replay_scalar_power"


def _receiver_index_array(receiver_tiles, *, tile_idx=None, n_rx: int):
    if receiver_tiles is None or tile_idx is None:
        return dr.arange(wt.UInt32, n_rx)
    local_n_rx = int(
        receiver_tiles.tile_extent_0[int(tile_idx)] * receiver_tiles.tile_extent_1[int(tile_idx)]
    )
    if local_n_rx <= 0:
        return dr.zeros(wt.UInt32, 0)
    return receiver_index_for_tile_slot(
        receiver_tiles,
        wt.UInt32(int(tile_idx)),
        dr.arange(wt.UInt32, local_n_rx),
    )


def _receiver_tile_streaming_stats(
    receiver_tiles,
    *,
    n_states: int,
    pair_chunk_budget: int | None,
) -> tuple[int, int]:
    n_tiles = 0 if receiver_tiles is None else int(getattr(receiver_tiles, "n_tiles", 0))
    if n_tiles <= 0 or n_states <= 0:
        return 0, 0
    peak_pair_count = 0
    launch_count = 0
    for tile_idx in range(n_tiles):
        receiver_idx = _receiver_index_array(receiver_tiles, tile_idx=tile_idx, n_rx=0)
        local_n_rx = int(dr.width(receiver_idx))
        if local_n_rx <= 0:
            continue
        chunk_size = _radio_map_diffraction_chunk_size(
            n_states,
            local_n_rx,
            pair_chunk_budget=pair_chunk_budget,
        )
        if chunk_size <= 0:
            continue
        peak_pair_count = max(peak_pair_count, int(chunk_size * local_n_rx))
        launch_count += int((int(n_states) + chunk_size - 1) // chunk_size)
    return peak_pair_count, launch_count


def _use_receiver_tile_streaming_fast_path(
    receiver_tiles,
    *,
    n_states: int,
    receiver_count: int,
    pair_chunk_budget: int | None,
) -> bool:
    if receiver_tiles is None or int(getattr(receiver_tiles, "n_tiles", 0)) <= 1:
        return False
    if n_states <= 0 or receiver_count <= 0:
        return False
    cartesian_chunk_size = _radio_map_diffraction_chunk_size(
        n_states,
        receiver_count,
        pair_chunk_budget=pair_chunk_budget,
    )
    if cartesian_chunk_size <= 0 or cartesian_chunk_size >= int(n_states):
        return False
    peak_pair_count, _ = _receiver_tile_streaming_stats(
        receiver_tiles,
        n_states=n_states,
        pair_chunk_budget=pair_chunk_budget,
    )
    return peak_pair_count > 0 and peak_pair_count < int(cartesian_chunk_size * receiver_count)


def _matched_isotropic_forward_fast_supported(
    *,
    state_arrays,
    receiver_model: str,
    pair_chunk_budget: int | None,
    receiver_count: int,
    incident_cross_target=None,
    reflection_cross_target=None,
    shadow_support_cutoff_db: float | None = None,
) -> bool:
    if state_arrays is None or int(state_arrays["n_states"]) <= 0:
        return False
    if int(receiver_count) <= 0:
        return False
    if str(receiver_model) != "matched_isotropic":
        return False
    if incident_cross_target is not None or reflection_cross_target is not None:
        return False
    if shadow_support_cutoff_db is not None:
        return False
    if state_arrays.get("edge_line_min") is None or state_arrays.get("edge_line_max") is None:
        return False
    return int(pair_chunk_budget or 0) > 0


def accumulate_diffraction_matched_isotropic_forward_fast(
    *,
    state_arrays,
    rx_pos,
    scene,
    wavelength: float,
    k: float,
    material_detail,
    tx_polarization=(1.0, 0.0, 0.0),
    rx_polarization=None,
    receiver_axis: str = "z",
    receiver_tiles=None,
    vector_target=None,
    coherent_target=None,
    power_target=None,
    coherent_scale: float = 1.0,
    power_scale: float = 1.0,
    vector_weight: float = 1.0,
    pair_chunk_budget: int | None = None,
    shadow_support_cutoff_db: float | None = None,
):
    n_rx = int(dr.width(rx_pos.x))
    n_states = 0 if state_arrays is None else int(state_arrays["n_states"])
    diagnostic_counts = _empty_diffraction_diagnostic_counts()
    coherent = ArrayInit.complex_zero(n_rx)
    power = dr.zeros(wt.Float, n_rx)
    vector_coherent = _complex_vector_zero(n_rx)
    pair_chunk_budget_value = (
        _radio_map_diffraction_pair_chunk_budget(n_rx)
        if pair_chunk_budget is None
        else max(1, int(pair_chunk_budget))
    )
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
    receiver_tiles = (
        receiver_tiles
        if receiver_tiles is not None
        else resolve_radio_map_receiver_tiles(
            receiver_positions=rx_pos,
        )
    )
    if not _matched_isotropic_forward_fast_supported(
        state_arrays=state_arrays,
        receiver_model="matched_isotropic",
        pair_chunk_budget=pair_chunk_budget_value,
        receiver_count=n_rx,
        incident_cross_target=None,
        reflection_cross_target=None,
        shadow_support_cutoff_db=shadow_support_cutoff_db,
    ):
        raise RuntimeError(
            "accumulate_diffraction_matched_isotropic_forward_fast requires bounded matched-isotropic "
            "diffraction states without shadow-surrogate cross terms."
        )
    scheduler_decision = select_radio_map_diffraction_receiver_tiles(
        state_arrays=state_arrays,
        receiver_tiles=receiver_tiles,
        receiver_count=n_rx,
        shadow_support_cutoff_db=shadow_support_cutoff_db,
        pair_chunk_budget=pair_chunk_budget_value,
    )
    selected_receiver_tiles = scheduler_decision.receiver_tiles
    tile_plan = scheduler_decision.tile_plan
    use_tiled = selected_receiver_tiles is not None and tile_plan is not None
    receiver_tile_streaming = False
    planner_stats = {
        "state_scheduler": str(scheduler_decision.state_scheduler),
        "planner_strategy": str(scheduler_decision.planner_strategy),
        "scalar_backend": "native_radiomap_vector_power_forward_fast",
        "planner_backend": scheduler_decision.planner_backend,
        "planner_skip_reason": scheduler_decision.planner_skip_reason,
        "selected_reason": str(scheduler_decision.selected_reason),
        "tile_task_count": int(scheduler_decision.tile_task_count),
        "estimated_pair_count": int(scheduler_decision.estimated_pair_count),
        "full_pair_count": int(scheduler_decision.full_pair_count),
        "estimated_pair_ratio": float(scheduler_decision.estimated_pair_ratio),
        "pair_chunk_budget": int(scheduler_decision.pair_chunk_budget or 0),
        "cartesian_peak_pair_count": int(scheduler_decision.cartesian_peak_pair_count),
        "tiled_peak_pair_count": int(scheduler_decision.tiled_peak_pair_count),
        "peak_pair_count_estimate": int(scheduler_decision.peak_pair_count_estimate),
        "estimated_launch_count": int(scheduler_decision.estimated_launch_count),
        "forward_fast_path": True,
        "utd_primal_backend": None,
    }
    if receiver_tile_streaming:
        streaming_peak_pair_count, streaming_launch_count = _receiver_tile_streaming_stats(
            receiver_tiles,
            n_states=n_states,
            pair_chunk_budget=pair_chunk_budget_value,
        )
        planner_stats["state_scheduler"] = "receiver_tile_streaming"
        planner_stats["planner_strategy"] = "receiver_tile_streaming"
        planner_stats["selected_reason"] = "receiver_tile_streaming_reduces_chunk_fragmentation"
        planner_stats["peak_pair_count_estimate"] = int(streaming_peak_pair_count)
        planner_stats["estimated_launch_count"] = int(streaming_launch_count)
    path_count = 0
    if use_tiled:
        for tile_idx in range(int(tile_plan.n_tiles)):
            tile_keep_idx = dr.compress(tile_plan.tile_task_tile_idx == wt.UInt32(tile_idx))
            if int(dr.width(tile_keep_idx)) <= 0:
                continue
            local_n_states = int(dr.width(tile_keep_idx))
            if local_n_states <= 0:
                continue
            tile_state_idx = dr.gather(
                type(tile_plan.tile_task_state_idx),
                tile_plan.tile_task_state_idx,
                tile_keep_idx,
            )
            if int(dr.width(tile_state_idx)) <= 0:
                continue
            receiver_idx = _receiver_index_array(
                selected_receiver_tiles,
                tile_idx=tile_idx,
                n_rx=n_rx,
            )
            local_n_rx = int(dr.width(receiver_idx))
            if local_n_rx <= 0:
                continue
            chunk_size = _radio_map_diffraction_chunk_size(
                local_n_states,
                local_n_rx,
                pair_chunk_budget=pair_chunk_budget_value,
            )
            for state_start in range(0, local_n_states, chunk_size):
                chunk_n = min(chunk_size, local_n_states - state_start)
                chunk_state_idx = dr.gather(
                    type(tile_state_idx),
                    tile_state_idx,
                    dr.arange(wt.UInt32, chunk_n) + wt.UInt32(state_start),
                )
                _, _, _, _, chunk_path_count = _accumulate_diffraction_tiled_vector_power_native(
                    local_state_idx=chunk_state_idx,
                    receiver_idx=receiver_idx,
                    state_arrays=state_arrays,
                    rx_pos=rx_pos,
                    scene=scene,
                    k=k,
                    wavelength=wavelength,
                    material_detail=material_detail,
                    rx_polarization=active_rx_polarization,
                    vector_coherent=vector_coherent,
                    coherent_target=coherent_target,
                    power_target=power_target,
                    vector_target=vector_target,
                    shadow_support_cutoff_db=shadow_support_cutoff_db,
                    receiver_axis=receiver_axis,
                    coherent_scale=coherent_scale,
                    power_scale=power_scale,
                    vector_weight=vector_weight,
                    coherent_accumulator=coherent,
                    power_accumulator=power,
                    diagnostic_counts=diagnostic_counts,
                    native_utd_vector_power=False,
                )
                path_count += int(chunk_path_count)
    elif receiver_tile_streaming:
        for tile_idx in range(int(receiver_tiles.n_tiles)):
            receiver_idx = _receiver_index_array(
                receiver_tiles,
                tile_idx=tile_idx,
                n_rx=n_rx,
            )
            local_n_rx = int(dr.width(receiver_idx))
            if local_n_rx <= 0:
                continue
            chunk_size = _radio_map_diffraction_chunk_size(
                n_states,
                local_n_rx,
                pair_chunk_budget=pair_chunk_budget_value,
            )
            for state_start in range(0, n_states, chunk_size):
                chunk_n_states = min(chunk_size, n_states - state_start)
                chunk_state_idx = dr.arange(wt.UInt32, chunk_n_states) + wt.UInt32(state_start)
                _, _, _, _, chunk_path_count = _accumulate_diffraction_tiled_vector_power_native(
                    local_state_idx=chunk_state_idx,
                    receiver_idx=receiver_idx,
                    state_arrays=state_arrays,
                    rx_pos=rx_pos,
                    scene=scene,
                    k=k,
                    wavelength=wavelength,
                    material_detail=material_detail,
                    rx_polarization=active_rx_polarization,
                    vector_coherent=vector_coherent,
                    coherent_target=coherent_target,
                    power_target=power_target,
                    vector_target=vector_target,
                    shadow_support_cutoff_db=shadow_support_cutoff_db,
                    receiver_axis=receiver_axis,
                    coherent_scale=coherent_scale,
                    power_scale=power_scale,
                    vector_weight=vector_weight,
                    coherent_accumulator=coherent,
                    power_accumulator=power,
                    diagnostic_counts=diagnostic_counts,
                    native_utd_vector_power=False,
                )
                path_count += int(chunk_path_count)
    else:
        state_chunk_size = _radio_map_diffraction_chunk_size(
            n_states,
            n_rx,
            pair_chunk_budget=pair_chunk_budget_value,
        )
        for state_start in range(0, n_states, state_chunk_size):
            chunk_n_states = min(state_chunk_size, n_states - state_start)
            chunk_state_idx = dr.arange(wt.UInt32, chunk_n_states) + wt.UInt32(state_start)
            _, _, _, _, chunk_path_count = _accumulate_diffraction_tiled_vector_power_native(
                local_state_idx=chunk_state_idx,
                receiver_idx=dr.arange(wt.UInt32, n_rx),
                state_arrays=state_arrays,
                rx_pos=rx_pos,
                scene=scene,
                k=k,
                wavelength=wavelength,
                material_detail=material_detail,
                rx_polarization=active_rx_polarization,
                vector_coherent=vector_coherent,
                coherent_target=coherent_target,
                power_target=power_target,
                vector_target=vector_target,
                shadow_support_cutoff_db=shadow_support_cutoff_db,
                receiver_axis=receiver_axis,
                coherent_scale=coherent_scale,
                power_scale=power_scale,
                vector_weight=vector_weight,
                coherent_accumulator=coherent,
                power_accumulator=power,
                diagnostic_counts=diagnostic_counts,
                native_utd_vector_power=False,
            )
            path_count += int(chunk_path_count)

    dr.eval(
        coherent.real,
        coherent.imag,
        power,
        vector_coherent["x"].real,
        vector_coherent["x"].imag,
        vector_coherent["y"].real,
        vector_coherent["y"].imag,
        vector_coherent["z"].real,
        vector_coherent["z"].imag,
    )
    return {
        "coherent": eval_complex(coherent),
        "power": power,
        "vector_coherent": _eval_complex_vector(vector_coherent),
        "incident_cross": dr.zeros(wt.Float, n_rx),
        "reflection_cross": dr.zeros(wt.Float, n_rx),
        "path_count": int(path_count),
        "diagnostic_counts": diagnostic_counts,
        "planner_stats": planner_stats,
    }


def _accumulate_reflection_chunk_scalar_power(
    *,
    paths,
    chunk_path_idx,
    receiver_idx,
    rx_pos,
    scene,
    wavelength: float,
    k: float,
    reflection_detail,
    tx_polarization,
    rx_polarization,
    receiver_model: str,
    coherent,
    power,
    vector_coherent=None,
    coherent_target=None,
    power_target=None,
    vector_target=None,
    coherent_scale: float = 1.0,
    power_scale: float = 1.0,
    vector_weight: float = 1.0,
    epc_descriptor=None,
):
    chunk_n_paths = int(dr.width(chunk_path_idx))
    local_n_rx = int(dr.width(receiver_idx))
    if chunk_n_paths <= 0 or local_n_rx <= 0:
        return 0

    descriptor_full_paths = (
        epc_descriptor is not None
        and int(getattr(epc_descriptor, "n_paths", 0)) == int(paths.n_paths)
    )
    if epc_descriptor is None:
        epc_descriptor = build_reflection_epc_descriptor(
            paths=paths,
            path_idx=chunk_path_idx,
            scene=scene,
            reflection_detail=reflection_detail,
        )
    n_pairs = chunk_n_paths * local_n_rx
    pair_idx = dr.arange(wt.UInt32, n_pairs)
    local_path_idx = pair_idx // local_n_rx
    local_rx_slot = pair_idx % local_n_rx
    rx_idx = dr.gather(type(receiver_idx), receiver_idx, local_rx_slot)
    target_pos = _gather_positions(rx_pos, rx_idx)
    descriptor_path_idx = (
        dr.gather(type(chunk_path_idx), chunk_path_idx, local_path_idx)
        if descriptor_full_paths
        else local_path_idx
    )
    image_source = Gather.point3(epc_descriptor.image_source, descriptor_path_idx)
    valid, chain_vector, geometry = epc_reflection_chain_to_target(
        paths=paths,
        path_idx=descriptor_path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=(),
        reflection_detail=reflection_detail,
        wavelength=wavelength,
        tx_polarization=tx_polarization,
        return_endpoints=True,
        epc_descriptor=epc_descriptor,
    )
    keep_idx = dr.compress(valid)
    if dr.width(keep_idx) == 0:
        return 0

    rx_idx_keep = dr.gather(type(rx_idx), rx_idx, keep_idx)
    target_pos_keep = Gather.point3(target_pos, keep_idx)
    image_source_keep = Gather.point3(image_source, keep_idx)
    last_hit_keep = Gather.point3(geometry["last_hit"], keep_idx)
    field_vector = {
        axis: dr.gather(wt.Complex2f, chain_vector[axis], keep_idx)
        for axis in ("x", "y", "z")
    }
    unit_field = _point_source_field(
        image_source_keep,
        wt.Complex2f(1.0, 0.0),
        target_pos_keep,
        wavelength,
        k,
    )
    field_vector = vector_scale(field_vector, unit_field)
    arrival_dir = target_pos_keep - last_hit_keep
    if (
        str(receiver_model) == "matched_isotropic"
        and not _point_grad_enabled(arrival_dir)
        and not _complex_vector_grad_enabled(field_vector)
    ):
        total_coherent, matched_power, total_vector, valid_pair_count = (
            radiomap_accumulate_vector_power_pairs(
                rx_idx_keep,
                field_vector,
                arrival_dir,
                n_output_rx=int(dr.width(rx_pos.x)),
                rx_polarization=rx_polarization,
            )
        )
        _scatter_dense_complex_and_power(
            coherent,
            power,
            total_coherent,
            matched_power,
        )
        _scatter_dense_complex_and_power(
            coherent_target,
            power_target,
            total_coherent,
            matched_power,
            coherent_scale=coherent_scale,
            power_scale=power_scale,
        )
        _scatter_dense_vector_coherent(vector_coherent, total_vector)
        _scatter_dense_vector_coherent(
            vector_target,
            total_vector,
            scale=vector_weight,
        )
        return int(valid_pair_count)

    scalar_coeff = eval_complex(
        scalarize_vector_to_polarization(
            field_vector,
            arrival_dir,
            rx_polarization,
        )
    )
    path_power = (
        _vector_power(field_vector)
        if str(receiver_model) == "matched_isotropic"
        else complex_abs_sqr(scalar_coeff)
    )
    dr.scatter_reduce(dr.ReduceOp.Add, coherent.real, scalar_coeff.real, rx_idx_keep)
    dr.scatter_reduce(dr.ReduceOp.Add, coherent.imag, scalar_coeff.imag, rx_idx_keep)
    dr.scatter_reduce(dr.ReduceOp.Add, power, path_power, rx_idx_keep)
    _scatter_vector_coherent(vector_coherent, field_vector, rx_idx_keep)
    if coherent_target is not None or power_target is not None:
        if coherent_target is not None:
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                coherent_target.real,
                scalar_coeff.real * float(coherent_scale),
                rx_idx_keep,
            )
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                coherent_target.imag,
                scalar_coeff.imag * float(coherent_scale),
                rx_idx_keep,
            )
        if power_target is not None:
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                power_target,
                path_power * float(power_scale),
                rx_idx_keep,
            )
    _scatter_vector_coherent(
        vector_target,
        field_vector,
        rx_idx_keep,
        scale=vector_weight,
    )
    return int(dr.width(rx_idx_keep))


def accumulate_reflection_scalar_power(
    *,
    rx_pos,
    scene,
    wavelength: float,
    k: float,
    reflection_detail,
    tx_polarization=(1.0, 0.0, 0.0),
    rx_polarization=None,
    receiver_model: str = "projected_polarized",
    receiver_tiles=None,
    coherent_target=None,
    power_target=None,
    vector_target=None,
    coherent_scale: float = 1.0,
    power_scale: float = 1.0,
    vector_weight: float = 1.0,
    return_vector_coherent: bool = False,
    allow_tiled_scheduler: bool = True,
):
    n_rx = int(dr.width(rx_pos.x))
    diagnostic_counts = _empty_diffraction_diagnostic_counts()
    coherent = ArrayInit.complex_zero(n_rx)
    power = dr.zeros(wt.Float, n_rx)
    diagnostic_counts = _empty_diffraction_diagnostic_counts()
    vector_coherent = (
        _complex_vector_zero(n_rx)
        if return_vector_coherent or vector_target is not None
        else None
    )
    if reflection_detail is None:
        if vector_coherent is None:
            dr.eval(coherent.real, coherent.imag, power)
        else:
            dr.eval(
                coherent.real,
                coherent.imag,
                power,
                vector_coherent["x"].real,
                vector_coherent["x"].imag,
                vector_coherent["y"].real,
                vector_coherent["y"].imag,
                vector_coherent["z"].real,
                vector_coherent["z"].imag,
            )
        return {
            "coherent": eval_complex(coherent),
            "power": power,
            "vector_coherent": _eval_complex_vector(vector_coherent),
            "path_count": 0,
            "planner_stats": {
                "path_scheduler": "empty",
                "planner_backend": None,
                "selected_reason": "no_reflection_paths",
                "tiled_bounce_count": 0,
                "tile_task_count": 0,
                "estimated_pair_count": 0,
                "full_pair_count": 0,
                "estimated_pair_ratio": 0.0,
                "scalar_backend": _reflection_scalar_backend_name(receiver_model),
            },
        }
    detail = coerce_reflection_trace_detail(reflection_detail)
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
    receiver_tiles = (
        receiver_tiles
        if receiver_tiles is not None
        else resolve_radio_map_receiver_tiles(
            receiver_positions=rx_pos,
        )
    )
    path_count = 0
    planner_stats = {
        "path_scheduler": "cartesian_chunked",
        "planner_backend": None,
        "selected_reason": "cartesian_chunked_preferred",
        "tiled_bounce_count": 0,
        "tile_task_count": 0,
        "estimated_pair_count": 0,
        "full_pair_count": 0,
        "estimated_pair_ratio": 1.0,
        "scalar_backend": _reflection_scalar_backend_name(receiver_model),
    }

    for paths in detail.source_paths_per_bounce:
        chain_depth = 0 if paths is None else int(paths.chain_depth)
        n_paths = 0 if paths is None else int(paths.n_paths)
        if chain_depth <= 0 or n_paths <= 0:
            continue
        scheduler_decision = select_radio_map_reflection_family_tiles(
            paths=paths,
            scene=scene,
            receiver_tiles=receiver_tiles,
            receiver_count=n_rx,
        )
        use_tiled = bool(allow_tiled_scheduler) and scheduler_decision.receiver_tiles is not None
        epc_descriptor = None
        planner_stats["full_pair_count"] += int(scheduler_decision.full_pair_count)
        if use_tiled:
            epc_descriptor = build_reflection_epc_descriptor(
                paths=paths,
                path_idx=dr.arange(wt.UInt32, n_paths),
                scene=scene,
                reflection_detail=reflection_detail,
            )
            planner_stats["path_scheduler"] = str(scheduler_decision.path_scheduler)
            planner_stats["planner_backend"] = scheduler_decision.planner_backend
            planner_stats["selected_reason"] = str(scheduler_decision.selected_reason)
            planner_stats["tiled_bounce_count"] += 1
            planner_stats["tile_task_count"] += int(scheduler_decision.tile_task_count)
            planner_stats["estimated_pair_count"] += int(scheduler_decision.estimated_pair_count)
            tile_plan = scheduler_decision.tile_plan
            tiled_receiver_tiles = scheduler_decision.receiver_tiles
            for tile_idx in range(int(tile_plan.n_tiles)):
                tile_keep_idx = dr.compress(tile_plan.tile_task_tile_idx == wt.UInt32(tile_idx))
                local_n_paths = int(dr.width(tile_keep_idx))
                if local_n_paths <= 0:
                    continue
                tile_family_idx = dr.gather(
                    type(tile_plan.tile_task_family_idx),
                    tile_plan.tile_task_family_idx,
                    tile_keep_idx,
                )
                receiver_idx = _receiver_index_array(
                    tiled_receiver_tiles,
                    tile_idx=tile_idx,
                    n_rx=n_rx,
                )
                chunk_size = _cartesian_chunk_size(local_n_paths, int(dr.width(receiver_idx)))
                for path_start in range(0, local_n_paths, chunk_size):
                    chunk_n = min(chunk_size, local_n_paths - path_start)
                    chunk_path_idx = dr.gather(
                        type(tile_family_idx),
                        tile_family_idx,
                        dr.arange(wt.UInt32, chunk_n) + wt.UInt32(path_start),
                    )
                    path_count += _accumulate_reflection_chunk_scalar_power(
                        paths=paths,
                        chunk_path_idx=chunk_path_idx,
                        receiver_idx=receiver_idx,
                        rx_pos=rx_pos,
                        scene=scene,
                        wavelength=wavelength,
                        k=k,
                        reflection_detail=reflection_detail,
                        tx_polarization=tx_polarization,
                        rx_polarization=active_rx_polarization,
                        receiver_model=receiver_model,
                        coherent=coherent,
                        power=power,
                        vector_coherent=vector_coherent,
                        coherent_target=coherent_target,
                        power_target=power_target,
                        vector_target=vector_target,
                        coherent_scale=coherent_scale,
                        power_scale=power_scale,
                        vector_weight=vector_weight,
                        epc_descriptor=epc_descriptor,
                    )
            continue

        planner_stats["estimated_pair_count"] += int(
            scheduler_decision.full_pair_count
            if not bool(allow_tiled_scheduler)
            else scheduler_decision.estimated_pair_count
        )
        planner_stats["selected_reason"] = (
            "no_diff_dense_fast_path_forced"
            if not bool(allow_tiled_scheduler)
            else str(scheduler_decision.selected_reason)
        )
        receiver_idx = _receiver_index_array(receiver_tiles, n_rx=n_rx)
        chunk_size = _cartesian_chunk_size(n_paths, int(dr.width(receiver_idx)))
        for path_start in range(0, n_paths, chunk_size):
            chunk_n = min(chunk_size, n_paths - path_start)
            chunk_path_idx = dr.arange(wt.UInt32, chunk_n) + wt.UInt32(path_start)
            path_count += _accumulate_reflection_chunk_scalar_power(
                paths=paths,
                chunk_path_idx=chunk_path_idx,
                receiver_idx=receiver_idx,
                rx_pos=rx_pos,
                scene=scene,
                wavelength=wavelength,
                k=k,
                reflection_detail=reflection_detail,
                tx_polarization=tx_polarization,
                rx_polarization=active_rx_polarization,
                receiver_model=receiver_model,
                coherent=coherent,
                power=power,
                vector_coherent=vector_coherent,
                coherent_target=coherent_target,
                power_target=power_target,
                vector_target=vector_target,
                coherent_scale=coherent_scale,
                power_scale=power_scale,
                vector_weight=vector_weight,
                epc_descriptor=epc_descriptor,
            )

    if vector_coherent is None:
        dr.eval(coherent.real, coherent.imag, power)
    else:
        dr.eval(
            coherent.real,
            coherent.imag,
            power,
            vector_coherent["x"].real,
            vector_coherent["x"].imag,
            vector_coherent["y"].real,
            vector_coherent["y"].imag,
            vector_coherent["z"].real,
            vector_coherent["z"].imag,
        )
    planner_stats["estimated_pair_ratio"] = (
        0.0
        if int(planner_stats["full_pair_count"]) <= 0
        else float(planner_stats["estimated_pair_count"]) / float(planner_stats["full_pair_count"])
    )
    return {
        "coherent": eval_complex(coherent),
        "power": power,
        "vector_coherent": _eval_complex_vector(vector_coherent),
        "path_count": int(path_count),
        "planner_stats": planner_stats,
    }


def _radio_map_diffraction_support_mask(
    batch_states,
    batch_rx,
    scene,
    *,
    shadow_support_cutoff_db,
    diagnostic_counts=None,
):
    edge_geometry = _compute_edge_geometry(
        batch_states["source_pos"],
        batch_states["edge_pos"],
        batch_states["edge_dir"],
        batch_states["n0"],
        batch_rx,
    )
    target_exterior = _wedge_exterior_region_mask(
        batch_rx - batch_states["edge_pos"],
        batch_states["edge_dir"],
        batch_states["n0"],
        batch_states["n_face_n"],
    )
    source_width = dr.width(batch_rx.x)
    wedge_n = repeat_float(batch_states["wedge_n"], source_width)
    shadow_decay_span = shadow_decay_span_from_wedge_n(wedge_n)
    base_valid = (
        _wedge_exterior_region_mask(
            batch_states["source_pos"] - batch_states["edge_pos"],
            batch_states["edge_dir"],
            batch_states["n0"],
            batch_states["n_face_n"],
        )
        & (edge_geometry["s_prime"] > DIFFRACTION_MIN_DISTANCE)
        & (edge_geometry["s"] > DIFFRACTION_MIN_DISTANCE)
    )
    if scene is None:
        return base_valid & target_exterior
    interior_mask = _point_inside_closed_mesh_mask(
        batch_rx,
        scene,
        active=base_valid & ~target_exterior,
    )
    shadow_completion_mask = base_valid & ~target_exterior & ~interior_mask
    if diagnostic_counts is not None:
        diagnostic_counts["shadow_completion_count"] += _mask_count(shadow_completion_mask)
        diagnostic_counts["interior_count"] += _mask_count(interior_mask)
    wrap_boundary = shadow_completion_mask & (
        edge_geometry["phi"] >= (wt.Float(2.0 * dr.pi) - wt.Float(0.5) * shadow_decay_span)
    )
    shadow_boundary_distance = dr.select(
        wrap_boundary,
        wt.Float(2.0 * dr.pi) - edge_geometry["phi"],
        edge_geometry["phi"] - wedge_n * dr.pi,
    )
    amplitude_threshold = shadow_support_amplitude_threshold(shadow_support_cutoff_db)
    if amplitude_threshold is None:
        shadow_active = shadow_completion_mask & (
            shadow_completion_weight_from_distance(shadow_boundary_distance, wedge_n)
            > wt.Float(0.0)
        )
    else:
        shadow_active = shadow_completion_mask & (
            shadow_completion_weight_from_distance(shadow_boundary_distance, wedge_n)
            >= amplitude_threshold
        )
    return base_valid & (target_exterior | shadow_active)


def _accumulate_diffraction_tiled_vector_power_native(
    *,
    local_state_idx,
    receiver_idx,
    state_arrays,
    rx_pos,
    scene,
    k: float,
    wavelength: float,
    material_detail,
    rx_polarization,
    vector_coherent=None,
    coherent_target=None,
    power_target=None,
    vector_target=None,
    shadow_support_cutoff_db: float | None = None,
    receiver_axis: str = "z",
    coherent_scale: float = 1.0,
    power_scale: float = 1.0,
    vector_weight: float = 1.0,
    coherent_accumulator=None,
    power_accumulator=None,
    diagnostic_counts=None,
    native_utd_vector_power: bool = False,
):
    n_rx = int(dr.width(rx_pos.x))
    local_n_states = int(dr.width(local_state_idx))
    local_n_rx = int(dr.width(receiver_idx))
    if local_n_states <= 0 or local_n_rx <= 0 or n_rx <= 0:
        return None, None, None, None, 0

    n_pairs = int(local_n_states * local_n_rx)
    pair_idx = dr.arange(wt.UInt32, n_pairs)
    local_state_slot = pair_idx // wt.UInt32(local_n_rx)
    local_rx_slot = pair_idx % wt.UInt32(local_n_rx)
    state_idx = dr.gather(type(local_state_idx), local_state_idx, local_state_slot)
    rx_idx = dr.gather(type(receiver_idx), receiver_idx, local_rx_slot)
    batch_rx = _gather_positions(rx_pos, rx_idx)

    use_native_utd_vector_power = (
        bool(native_utd_vector_power)
        and coherent_target is None
        and power_target is None
        and vector_target is None
        and coherent_accumulator is not None
        and power_accumulator is not None
        and vector_coherent is not None
        and float(coherent_scale) == 1.0
        and float(power_scale) == 1.0
        and float(vector_weight) == 1.0
    )
    if not use_native_utd_vector_power:
        if diagnostic_counts is not None and scene is None:
            diagnostic_counts["visible_pair_count"] += int(dr.width(state_idx))

        if scene is not None:
            state_edge_pos = Gather.point3(state_arrays["edge_pos"], state_idx)
            adjacent_face0 = dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx)
            adjacent_face1 = dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx)
            owner_structure_idx = _edge_owner_structure_idx(scene, adjacent_face0, adjacent_face1)
            visible = _segment_visibility_mask(
                state_edge_pos,
                batch_rx,
                scene,
                ignore_prim_idx=(adjacent_face0, adjacent_face1),
                ignore_structure_idx=owner_structure_idx,
            )
            keep_idx = dr.compress(visible)
            if dr.width(keep_idx) == 0:
                return None, None, None, None, 0
            if diagnostic_counts is not None:
                diagnostic_counts["visible_pair_count"] += int(dr.width(keep_idx))
            state_idx = dr.gather(type(state_idx), state_idx, keep_idx)
            rx_idx = dr.gather(type(rx_idx), rx_idx, keep_idx)
            local_rx_slot = dr.gather(type(local_rx_slot), local_rx_slot, keep_idx)
            batch_rx = Gather.point3(batch_rx, keep_idx)

        reduced_path_export = is_path_export_reduced_state_arrays(state_arrays)
        batch_states = (
            gather_path_export_support_state_fields(state_arrays, state_idx)
            if reduced_path_export
            else gather_field_evaluation_state_fields(state_arrays, state_idx)
        )
        support_keep_idx = dr.compress(
            _radio_map_diffraction_support_mask(
                batch_states,
                batch_rx,
                scene,
                shadow_support_cutoff_db=shadow_support_cutoff_db,
                diagnostic_counts=diagnostic_counts,
            )
        )
        if dr.width(support_keep_idx) == 0:
            return None, None, None, None, 0
        if diagnostic_counts is not None:
            diagnostic_counts["support_pair_count"] += int(dr.width(support_keep_idx))
        state_idx = dr.gather(type(state_idx), state_idx, support_keep_idx)
        batch_rx = Gather.point3(batch_rx, support_keep_idx)
        rx_idx = dr.gather(type(rx_idx), rx_idx, support_keep_idx)
        local_rx_slot = dr.gather(type(local_rx_slot), local_rx_slot, support_keep_idx)
        batch_states = (
            gather_path_export_field_state_fields(state_arrays, state_idx)
            if reduced_path_export
            else gather_field_evaluation_state_fields(state_arrays, state_idx)
        )

        _, pair_vector, pair_valid = _edge_state_field_to_targets(
            batch_states,
            batch_rx,
            k=k,
            wavelength=wavelength,
            material_detail=material_detail,
            scene=scene,
            return_vector=True,
            return_valid=True,
            smooth_exterior_shadow=True,
        )
        keep_idx = dr.compress(pair_valid)
        if dr.width(keep_idx) == 0:
            return None, None, None, None, 0
        if diagnostic_counts is not None:
            diagnostic_counts["pair_valid_count"] += int(dr.width(keep_idx))
        batch_rx = Gather.point3(batch_rx, keep_idx)
        batch_states = _gather_shadow_transition_state_fields(batch_states, keep_idx)
        rx_idx = dr.gather(type(rx_idx), rx_idx, keep_idx)
        local_rx_slot = dr.gather(type(local_rx_slot), local_rx_slot, keep_idx)
        batch_edge_pos = batch_states["edge_pos"]
        pair_vector = {
            axis: dr.gather(wt.Complex2f, pair_vector[axis], keep_idx)
            for axis in ("x", "y", "z")
        }
        arrival_dir = batch_rx - batch_edge_pos
        if local_n_rx < n_rx:
            output_receiver_idx = receiver_idx
            output_rx_idx = local_rx_slot
            output_n_rx = local_n_rx
        else:
            output_receiver_idx, output_rx_idx = _compact_output_receiver_indices(
                rx_idx,
                n_rx=n_rx,
            )
            output_n_rx = int(dr.width(output_receiver_idx))
        if output_n_rx <= 0:
            return None, None, None, None, 0
        total_coherent, matched_power, total_vector, valid_pair_count = (
            radiomap_accumulate_vector_power_pairs(
                output_rx_idx,
                pair_vector,
                arrival_dir,
                n_output_rx=output_n_rx,
                rx_polarization=rx_polarization,
            )
        )

        _scatter_dense_complex_and_power(
            coherent_target,
            power_target,
            total_coherent,
            matched_power,
            rx_idx=output_receiver_idx,
            coherent_scale=coherent_scale,
            power_scale=power_scale,
        )
        _scatter_dense_complex_and_power(
            coherent_accumulator,
            power_accumulator,
            total_coherent,
            matched_power,
            rx_idx=output_receiver_idx,
        )
        _scatter_dense_vector_coherent(
            vector_coherent,
            total_vector,
            rx_idx=output_receiver_idx,
        )
        _scatter_dense_vector_coherent(
            vector_target,
            total_vector,
            rx_idx=output_receiver_idx,
            scale=vector_weight,
        )
        if coherent_accumulator is not None:
            return None, None, None, None, int(valid_pair_count)
        dense_coherent, dense_power = _densify_complex_and_power(
            total_coherent,
            matched_power,
            rx_idx=output_receiver_idx,
            n_rx=n_rx,
        )
        zero_cross = dr.zeros(wt.Float, n_rx)
        return dense_coherent, dense_power, zero_cross, zero_cross, int(valid_pair_count)

    if scene is None:
        visible = dr.full(wt.Bool, True, n_pairs)
        if diagnostic_counts is not None:
            diagnostic_counts["visible_pair_count"] += int(dr.width(state_idx))
    else:
        state_edge_pos = Gather.point3(state_arrays["edge_pos"], state_idx)
        adjacent_face0 = dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx)
        adjacent_face1 = dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx)
        owner_structure_idx = _edge_owner_structure_idx(scene, adjacent_face0, adjacent_face1)
        visible = _segment_visibility_mask(
            state_edge_pos,
            batch_rx,
            scene,
            ignore_prim_idx=(adjacent_face0, adjacent_face1),
            ignore_structure_idx=owner_structure_idx,
        )
        visible_count = int(_mask_count(visible))
        if visible_count <= 0:
            return None, None, None, None, 0
        if diagnostic_counts is not None:
            diagnostic_counts["visible_pair_count"] += visible_count

    reduced_path_export = is_path_export_reduced_state_arrays(state_arrays)
    batch_states = (
        gather_path_export_support_state_fields(state_arrays, state_idx)
        if reduced_path_export
        else gather_field_evaluation_state_fields(state_arrays, state_idx)
    )
    support_mask = _radio_map_diffraction_support_mask(
        batch_states,
        batch_rx,
        scene,
        shadow_support_cutoff_db=shadow_support_cutoff_db,
        diagnostic_counts=diagnostic_counts,
    )
    valid_mask = visible & support_mask
    support_count = int(_mask_count(valid_mask))
    if support_count <= 0:
        return None, None, None, None, 0
    if diagnostic_counts is not None:
        diagnostic_counts["support_pair_count"] += support_count
    if use_native_utd_vector_power:
        local_state_arrays = (
            gather_path_export_eval_state_fields(state_arrays, local_state_idx)
            if reduced_path_export
            else gather_state_arrays(state_arrays, local_state_idx)
        )
        if "prefix_reflection_depth" in local_state_arrays:
            local_prefix_depth = local_state_arrays["prefix_reflection_depth"]
            local_intermediate_depth = local_state_arrays["intermediate_reflection_depth"]
            local_suffix_depth = local_state_arrays["suffix_reflection_depth"]
        else:
            local_prefix_depth = dr.gather(
                wt.UInt32,
                state_arrays["prefix_reflection_depth"],
                local_state_idx,
            )
            local_intermediate_depth = dr.gather(
                wt.UInt32,
                state_arrays["intermediate_reflection_depth"],
                local_state_idx,
            )
            local_suffix_depth = dr.gather(
                wt.UInt32,
                state_arrays["suffix_reflection_depth"],
                local_state_idx,
            )
        local_ownership = wt.Int32(
            _ownership_code_from_depths(
                local_prefix_depth,
                local_intermediate_depth,
                local_suffix_depth,
            )
        )
        exact_support = _edge_state_target_support(
            batch_states,
            batch_rx,
            scene=scene,
            smooth_exterior_shadow=True,
        )
        transition_mask = valid_mask & (
            exact_support["shadow_completion_mask"]
            | exact_support["illuminated_boundary_mask"]
        )
        exact_core_mask = valid_mask & exact_support["field_valid"] & ~transition_mask
        native_valid_pair_count = 0
        if int(_mask_count(exact_core_mask)) > 0:
            native_output_buffers = _zero_pair_output_buffers(n_rx)
            native_power = dr.zeros(wt.Float, n_rx)
            dr.eval(native_power)
            native_valid_pair_count = _utd_accumulate_tiled_vector_power_into(
                state_arrays=local_state_arrays,
                state_idx=dr.arange(wt.UInt32, local_n_states),
                rx_pos=rx_pos,
                rx_idx=receiver_idx,
                valid_mask=dr.select(exact_core_mask, wt.Int32(1), wt.Int32(0)),
                out_buffers=native_output_buffers,
                matched_power=native_power,
                k=k,
                wavelength=wavelength,
                material_detail=material_detail,
                rx_polarization=rx_polarization,
                ownership_code=local_ownership,
            )
            native_coherent = wt.Complex2f(
                native_output_buffers["direct_re"] + native_output_buffers["multi_re"],
                native_output_buffers["direct_im"] + native_output_buffers["multi_im"],
            )
            native_vector = {
                "x": wt.Complex2f(
                    native_output_buffers["direct_vec_x_re"] + native_output_buffers["multi_vec_x_re"],
                    native_output_buffers["direct_vec_x_im"] + native_output_buffers["multi_vec_x_im"],
                ),
                "y": wt.Complex2f(
                    native_output_buffers["direct_vec_y_re"] + native_output_buffers["multi_vec_y_re"],
                    native_output_buffers["direct_vec_y_im"] + native_output_buffers["multi_vec_y_im"],
                ),
                "z": wt.Complex2f(
                    native_output_buffers["direct_vec_z_re"] + native_output_buffers["multi_vec_z_re"],
                    native_output_buffers["direct_vec_z_im"] + native_output_buffers["multi_vec_z_im"],
                ),
            }
            _scatter_dense_complex_and_power(
                coherent_accumulator,
                power_accumulator,
                native_coherent,
                native_power,
            )
            _scatter_dense_vector_coherent(
                vector_coherent,
                native_vector,
            )
        if diagnostic_counts is not None:
            diagnostic_counts["pair_valid_count"] += int(native_valid_pair_count)
        valid_mask = transition_mask
        if int(_mask_count(valid_mask)) <= 0:
            return None, None, None, None, int(native_valid_pair_count)

    keep_idx = dr.compress(valid_mask)
    if dr.width(keep_idx) == 0:
        return None, None, None, None, 0 if not use_native_utd_vector_power else int(native_valid_pair_count)
    state_idx = dr.gather(type(state_idx), state_idx, keep_idx)
    batch_rx = Gather.point3(batch_rx, keep_idx)
    rx_idx = dr.gather(type(rx_idx), rx_idx, keep_idx)
    local_rx_slot = dr.gather(type(local_rx_slot), local_rx_slot, keep_idx)
    batch_states = (
        gather_path_export_field_state_fields(state_arrays, state_idx)
        if reduced_path_export
        else gather_field_evaluation_state_fields(state_arrays, state_idx)
    )

    _, pair_vector, pair_valid = _edge_state_field_to_targets(
        batch_states,
        batch_rx,
        k=k,
        wavelength=wavelength,
        material_detail=material_detail,
        scene=scene,
        return_vector=True,
        return_valid=True,
        smooth_exterior_shadow=True,
    )
    keep_idx = dr.compress(pair_valid)
    if dr.width(keep_idx) == 0:
        return None, None, None, None, 0 if not use_native_utd_vector_power else int(native_valid_pair_count)
    if diagnostic_counts is not None:
        diagnostic_counts["pair_valid_count"] += int(dr.width(keep_idx))
    batch_rx = Gather.point3(batch_rx, keep_idx)
    batch_states = _gather_shadow_transition_state_fields(batch_states, keep_idx)
    rx_idx = dr.gather(type(rx_idx), rx_idx, keep_idx)
    local_rx_slot = dr.gather(type(local_rx_slot), local_rx_slot, keep_idx)
    batch_edge_pos = batch_states["edge_pos"]
    pair_vector = {
        axis: dr.gather(wt.Complex2f, pair_vector[axis], keep_idx)
        for axis in ("x", "y", "z")
    }
    arrival_dir = batch_rx - batch_edge_pos
    if local_n_rx < n_rx:
        output_receiver_idx = receiver_idx
        output_rx_idx = local_rx_slot
        output_n_rx = local_n_rx
    else:
        output_receiver_idx, output_rx_idx = _compact_output_receiver_indices(
            rx_idx,
            n_rx=n_rx,
        )
        output_n_rx = int(dr.width(output_receiver_idx))
    if output_n_rx <= 0:
        return None, None, None, None, 0
    total_coherent, matched_power, total_vector, valid_pair_count = (
        radiomap_accumulate_vector_power_pairs(
            output_rx_idx,
            pair_vector,
            arrival_dir,
            n_output_rx=output_n_rx,
            rx_polarization=rx_polarization,
        )
    )

    _scatter_dense_complex_and_power(
        coherent_target,
        power_target,
        total_coherent,
        matched_power,
        rx_idx=output_receiver_idx,
        coherent_scale=coherent_scale,
        power_scale=power_scale,
    )
    _scatter_dense_complex_and_power(
        coherent_accumulator,
        power_accumulator,
        total_coherent,
        matched_power,
        rx_idx=output_receiver_idx,
    )
    _scatter_dense_vector_coherent(
        vector_coherent,
        total_vector,
        rx_idx=output_receiver_idx,
    )
    _scatter_dense_vector_coherent(
        vector_target,
        total_vector,
        rx_idx=output_receiver_idx,
        scale=vector_weight,
    )
    if coherent_accumulator is not None:
        total_valid_pair_count = int(valid_pair_count)
        if use_native_utd_vector_power:
            total_valid_pair_count += int(native_valid_pair_count)
        return None, None, None, None, total_valid_pair_count
    dense_coherent, dense_power = _densify_complex_and_power(
        total_coherent,
        matched_power,
        rx_idx=output_receiver_idx,
        n_rx=n_rx,
    )
    zero_cross = dr.zeros(wt.Float, n_rx)
    total_valid_pair_count = int(valid_pair_count)
    if use_native_utd_vector_power:
        total_valid_pair_count += int(native_valid_pair_count)
    return dense_coherent, dense_power, zero_cross, zero_cross, total_valid_pair_count


def _accumulate_diffraction_pairs_scalar_power(
    *,
    state_idx,
    rx_idx,
    state_arrays,
    rx_pos,
    scene,
    k: float,
    wavelength: float,
    material_detail,
    rx_polarization,
    receiver_model: str,
    vector_coherent=None,
    coherent_target=None,
    power_target=None,
    vector_target=None,
    incident_reference_vector=None,
    reflection_reference_vector=None,
    incident_cross_target=None,
    reflection_cross_target=None,
    shadow_support_cutoff_db: float | None = None,
    receiver_axis: str = "z",
    coherent_scale: float = 1.0,
    power_scale: float = 1.0,
    cross_scale: float = 1.0,
    vector_weight: float = 1.0,
    output_rx_idx=None,
    output_receiver_idx=None,
    output_n_rx: int | None = None,
    coherent_accumulator=None,
    power_accumulator=None,
    incident_cross_accumulator=None,
    reflection_cross_accumulator=None,
    diagnostic_counts=None,
):
    n_rx = int(dr.width(rx_pos.x))
    if dr.width(state_idx) == 0 or dr.width(rx_idx) == 0:
        return None, None, None, None, 0

    batch_rx = _gather_positions(rx_pos, rx_idx)
    if diagnostic_counts is not None and scene is None:
        diagnostic_counts["visible_pair_count"] += int(dr.width(state_idx))
    if scene is not None:
        state_edge_pos = Gather.point3(state_arrays["edge_pos"], state_idx)
        adjacent_face0 = dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx)
        adjacent_face1 = dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx)
        owner_structure_idx = _edge_owner_structure_idx(scene, adjacent_face0, adjacent_face1)
        visible = _segment_visibility_mask(
            state_edge_pos,
            batch_rx,
            scene,
            ignore_prim_idx=(adjacent_face0, adjacent_face1),
            ignore_structure_idx=owner_structure_idx,
        )
        keep_idx = dr.compress(visible)
        if dr.width(keep_idx) == 0:
            return None, None, None, None, 0
        if diagnostic_counts is not None:
            diagnostic_counts["visible_pair_count"] += int(dr.width(keep_idx))
        state_idx = dr.gather(type(state_idx), state_idx, keep_idx)
        rx_idx = dr.gather(type(rx_idx), rx_idx, keep_idx)
        batch_rx = Gather.point3(batch_rx, keep_idx)

    reduced_path_export = is_path_export_reduced_state_arrays(state_arrays)
    batch_states = (
        gather_path_export_support_state_fields(state_arrays, state_idx)
        if reduced_path_export
        else gather_field_evaluation_state_fields(state_arrays, state_idx)
    )
    support_keep_idx = dr.compress(
        _radio_map_diffraction_support_mask(
            batch_states,
            batch_rx,
            scene,
            shadow_support_cutoff_db=shadow_support_cutoff_db,
            diagnostic_counts=diagnostic_counts,
        )
    )
    if dr.width(support_keep_idx) == 0:
        return None, None, None, None, 0
    if diagnostic_counts is not None:
        diagnostic_counts["support_pair_count"] += int(dr.width(support_keep_idx))
    state_idx = dr.gather(type(state_idx), state_idx, support_keep_idx)
    batch_rx = Gather.point3(batch_rx, support_keep_idx)
    rx_idx = dr.gather(type(rx_idx), rx_idx, support_keep_idx)
    batch_states = (
        gather_path_export_field_state_fields(state_arrays, state_idx)
        if reduced_path_export
        else gather_field_evaluation_state_fields(state_arrays, state_idx)
    )
    if (
        _native_diffraction_scalar_power_pairs_enabled(receiver_model)
        and vector_coherent is None
        and vector_target is None
    ):
        if output_rx_idx is None or output_receiver_idx is None or output_n_rx is None:
            output_receiver_idx, output_rx_idx = _compact_output_receiver_indices(
                rx_idx,
                n_rx=n_rx,
            )
            output_n_rx = int(dr.width(output_receiver_idx))
        if output_n_rx <= 0:
            return None, None, None, None, 0
        native_coherent, native_power, valid_pair_count = utd_accumulate_scalar_power_pairs(
            batch_states,
            batch_rx,
            output_rx_idx,
            n_output_rx=output_n_rx,
            k=k,
            wavelength=wavelength,
            material_detail=material_detail,
            rx_polarization=rx_polarization,
        )
        if diagnostic_counts is not None:
            diagnostic_counts["pair_valid_count"] += int(valid_pair_count)
        _scatter_dense_complex_and_power(
            coherent_target,
            power_target,
            native_coherent,
            native_power,
            rx_idx=output_receiver_idx,
            coherent_scale=coherent_scale,
            power_scale=power_scale,
        )
        if coherent_accumulator is not None or power_accumulator is not None:
            _scatter_dense_complex_and_power(
                coherent_accumulator,
                power_accumulator,
                native_coherent,
                native_power,
                rx_idx=output_receiver_idx,
            )
        zero_cross = dr.zeros(wt.Float, n_rx)
        if coherent_accumulator is not None:
            return None, None, None, None, int(valid_pair_count)
        dense_coherent, dense_power = _densify_complex_and_power(
            native_coherent,
            native_power,
            rx_idx=output_receiver_idx,
            n_rx=n_rx,
        )
        return dense_coherent, dense_power, zero_cross, zero_cross, int(valid_pair_count)

    _, pair_vector, pair_valid = _edge_state_field_to_targets(
        batch_states,
        batch_rx,
        k,
        return_vector=True,
        return_valid=True,
        wavelength=wavelength,
        material_detail=material_detail,
        scene=scene,
        smooth_exterior_shadow=True,
    )
    keep_idx = dr.compress(pair_valid)
    if dr.width(keep_idx) == 0:
        return None, None, None, None, 0
    if diagnostic_counts is not None:
        diagnostic_counts["pair_valid_count"] += int(dr.width(keep_idx))
    batch_rx = Gather.point3(batch_rx, keep_idx)
    batch_states = _gather_shadow_transition_state_fields(batch_states, keep_idx)
    rx_idx = dr.gather(type(rx_idx), rx_idx, keep_idx)
    batch_edge_pos = batch_states["edge_pos"]
    pair_vector = {
        axis: dr.gather(wt.Complex2f, pair_vector[axis], keep_idx)
        for axis in ("x", "y", "z")
    }
    arrival_dir = batch_rx - batch_edge_pos
    scalar_coeff = eval_complex(
        scalarize_vector_to_polarization(pair_vector, arrival_dir, rx_polarization)
    )
    pair_power = (
        _vector_power(pair_vector)
        if str(receiver_model) == "matched_isotropic"
        else complex_abs_sqr(scalar_coeff)
    )
    chunk_coherent = coherent_accumulator if coherent_accumulator is not None else ArrayInit.complex_zero(n_rx)
    chunk_power = power_accumulator if power_accumulator is not None else dr.zeros(wt.Float, n_rx)
    chunk_incident_cross = (
        incident_cross_accumulator
        if incident_cross_accumulator is not None
        else dr.zeros(wt.Float, n_rx)
    )
    chunk_reflection_cross = (
        reflection_cross_accumulator
        if reflection_cross_accumulator is not None
        else dr.zeros(wt.Float, n_rx)
    )
    dr.scatter_reduce(dr.ReduceOp.Add, chunk_coherent.real, scalar_coeff.real, rx_idx)
    dr.scatter_reduce(dr.ReduceOp.Add, chunk_coherent.imag, scalar_coeff.imag, rx_idx)
    dr.scatter_reduce(dr.ReduceOp.Add, chunk_power, pair_power, rx_idx)
    _scatter_vector_coherent(vector_coherent, pair_vector, rx_idx)
    need_shadow_boundary_cross_terms = (
        incident_cross_target is not None
        and incident_reference_vector is not None
    ) or (
        reflection_cross_target is not None
        and reflection_reference_vector is not None
    )
    if need_shadow_boundary_cross_terms:
        incident_weight, reflection_weight = _shadow_boundary_transition_weights(
            batch_states,
            batch_rx,
            k=k,
        )
    else:
        zero_weight = dr.zeros(wt.Float, dr.width(rx_idx))
        incident_weight = zero_weight
        reflection_weight = zero_weight
    _scatter_shadow_boundary_cross_term(
        chunk_incident_cross,
        incident_reference_vector,
        pair_vector,
        rx_idx,
        incident_weight,
    )
    _scatter_shadow_boundary_cross_term(
        chunk_reflection_cross,
        reflection_reference_vector,
        pair_vector,
        rx_idx,
        reflection_weight,
    )
    if coherent_target is not None or power_target is not None:
        if coherent_target is not None:
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                coherent_target.real,
                scalar_coeff.real * float(coherent_scale),
                rx_idx,
            )
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                coherent_target.imag,
                scalar_coeff.imag * float(coherent_scale),
                rx_idx,
            )
        if power_target is not None:
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                power_target,
                pair_power * float(power_scale),
                rx_idx,
            )
    _scatter_vector_coherent(
        vector_target,
        pair_vector,
        rx_idx,
        scale=vector_weight,
    )
    _scatter_shadow_boundary_cross_term(
        incident_cross_target,
        incident_reference_vector,
        pair_vector,
        rx_idx,
        incident_weight,
        scale=cross_scale,
    )
    _scatter_shadow_boundary_cross_term(
        reflection_cross_target,
        reflection_reference_vector,
        pair_vector,
        rx_idx,
        reflection_weight,
        scale=cross_scale,
    )
    if coherent_accumulator is not None:
        return None, None, None, None, int(dr.width(rx_idx))
    dr.eval(
        chunk_coherent.real,
        chunk_coherent.imag,
        chunk_power,
        chunk_incident_cross,
        chunk_reflection_cross,
    )
    return (
        eval_complex(chunk_coherent),
        chunk_power,
        chunk_incident_cross,
        chunk_reflection_cross,
        int(dr.width(rx_idx)),
    )


def accumulate_diffraction_scalar_power(
    *,
    state_arrays,
    rx_pos,
    scene,
    wavelength: float,
    k: float,
    material_detail,
    tx_polarization=(1.0, 0.0, 0.0),
    rx_polarization=None,
    receiver_model: str = "projected_polarized",
    receiver_tiles=None,
    vector_target=None,
    coherent_target=None,
    power_target=None,
    incident_reference_vector=None,
    reflection_reference_vector=None,
    incident_cross_target=None,
    reflection_cross_target=None,
    shadow_support_cutoff_db: float | None = None,
    receiver_axis: str = "z",
    coherent_scale: float = 1.0,
    power_scale: float = 1.0,
    cross_scale: float = 1.0,
    vector_weight: float = 1.0,
    return_vector_coherent: bool = False,
    native_primal_forward: bool = False,
    native_vector_replay: bool = False,
):
    n_rx = int(dr.width(rx_pos.x))
    diagnostic_counts = _empty_diffraction_diagnostic_counts()
    coherent = ArrayInit.complex_zero(n_rx)
    power = dr.zeros(wt.Float, n_rx)
    vector_coherent = (
        _complex_vector_zero(n_rx)
        if return_vector_coherent or vector_target is not None
        else None
    )
    incident_cross = dr.zeros(wt.Float, n_rx)
    reflection_cross = dr.zeros(wt.Float, n_rx)
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
    pair_chunk_budget = _radio_map_diffraction_pair_chunk_budget(n_rx)
    receiver_tiles = (
        receiver_tiles
        if receiver_tiles is not None
        else resolve_radio_map_receiver_tiles(
            receiver_positions=rx_pos,
        )
    )
    native_primal_forward_active = _matched_isotropic_forward_fast_supported(
        state_arrays=state_arrays,
        receiver_model=receiver_model,
        pair_chunk_budget=pair_chunk_budget,
        receiver_count=n_rx,
        incident_cross_target=incident_cross_target,
        reflection_cross_target=reflection_cross_target,
        shadow_support_cutoff_db=shadow_support_cutoff_db,
    ) and bool(native_primal_forward)
    native_vector_replay_active = (
        bool(native_vector_replay)
        and not native_primal_forward_active
        and str(receiver_model) == "matched_isotropic"
        and incident_cross_target is None
        and reflection_cross_target is None
        and state_arrays is not None
        and state_arrays.get("edge_line_min") is not None
        and state_arrays.get("edge_line_max") is not None
    )
    scalar_backend_name = _diffraction_scalar_backend_name(
        receiver_model,
        native_primal_forward=native_primal_forward_active,
        native_vector_replay=native_vector_replay_active,
    )
    if state_arrays is None or int(state_arrays["n_states"]) <= 0 or n_rx <= 0:
        if vector_coherent is None:
            dr.eval(coherent.real, coherent.imag, power)
        else:
            dr.eval(
                coherent.real,
                coherent.imag,
                power,
                vector_coherent["x"].real,
                vector_coherent["x"].imag,
                vector_coherent["y"].real,
                vector_coherent["y"].imag,
                vector_coherent["z"].real,
                vector_coherent["z"].imag,
            )
        return {
            "coherent": eval_complex(coherent),
            "power": power,
            "vector_coherent": _eval_complex_vector(vector_coherent),
            "incident_cross": incident_cross,
            "reflection_cross": reflection_cross,
            "path_count": 0,
            "diagnostic_counts": diagnostic_counts,
            "planner_stats": {
                "state_scheduler": "empty",
                "planner_strategy": "empty",
                "scalar_backend": scalar_backend_name,
                "planner_backend": None,
                "planner_skip_reason": "no_diffraction_states",
                "selected_reason": "no_diffraction_states",
                "tile_task_count": 0,
                "estimated_pair_count": 0,
                "full_pair_count": 0,
                "estimated_pair_ratio": 0.0,
                "pair_chunk_budget": int(pair_chunk_budget),
                "cartesian_peak_pair_count": 0,
                "tiled_peak_pair_count": 0,
                "peak_pair_count_estimate": 0,
                "estimated_launch_count": 0,
                "forward_fast_path": bool(native_primal_forward_active),
                "utd_primal_backend": None,
            },
        }

    n_states = int(state_arrays["n_states"])
    if native_primal_forward_active:
        return accumulate_diffraction_matched_isotropic_forward_fast(
            state_arrays=state_arrays,
            rx_pos=rx_pos,
            scene=scene,
            wavelength=wavelength,
            k=k,
            material_detail=material_detail,
            tx_polarization=tx_polarization,
            rx_polarization=rx_polarization,
            receiver_axis=receiver_axis,
            receiver_tiles=receiver_tiles,
            vector_target=vector_target,
            coherent_target=coherent_target,
            power_target=power_target,
            coherent_scale=coherent_scale,
            power_scale=power_scale,
            vector_weight=vector_weight,
            pair_chunk_budget=pair_chunk_budget,
            shadow_support_cutoff_db=shadow_support_cutoff_db,
        )
    scheduler_decision = select_radio_map_diffraction_receiver_tiles(
        state_arrays=state_arrays,
        receiver_tiles=receiver_tiles,
        receiver_count=n_rx,
        shadow_support_cutoff_db=shadow_support_cutoff_db,
        pair_chunk_budget=pair_chunk_budget,
    )
    selected_receiver_tiles = scheduler_decision.receiver_tiles
    tile_plan = scheduler_decision.tile_plan
    use_tiled = selected_receiver_tiles is not None
    planner_stats = {
        "state_scheduler": str(scheduler_decision.state_scheduler),
        "planner_strategy": str(scheduler_decision.planner_strategy),
        "scalar_backend": scalar_backend_name,
        "planner_backend": scheduler_decision.planner_backend,
        "planner_skip_reason": scheduler_decision.planner_skip_reason,
        "selected_reason": str(scheduler_decision.selected_reason),
        "tile_task_count": int(scheduler_decision.tile_task_count),
        "estimated_pair_count": int(scheduler_decision.estimated_pair_count),
        "full_pair_count": int(scheduler_decision.full_pair_count),
        "estimated_pair_ratio": float(scheduler_decision.estimated_pair_ratio),
        "pair_chunk_budget": int(scheduler_decision.pair_chunk_budget or 0),
        "cartesian_peak_pair_count": int(scheduler_decision.cartesian_peak_pair_count),
        "tiled_peak_pair_count": int(scheduler_decision.tiled_peak_pair_count),
        "peak_pair_count_estimate": int(scheduler_decision.peak_pair_count_estimate),
        "estimated_launch_count": int(scheduler_decision.estimated_launch_count),
        "forward_fast_path": False,
        "utd_primal_backend": None,
    }

    path_count = 0
    if use_tiled:
        if tile_plan is None:
            use_tiled = False
        else:
            for tile_idx in range(int(tile_plan.n_tiles)):
                tile_keep_idx = dr.compress(tile_plan.tile_task_tile_idx == wt.UInt32(tile_idx))
                local_n_states = int(dr.width(tile_keep_idx))
                if local_n_states <= 0:
                    continue
                tile_state_idx = dr.gather(
                    type(tile_plan.tile_task_state_idx),
                    tile_plan.tile_task_state_idx,
                    tile_keep_idx,
                )
                receiver_idx = _receiver_index_array(
                    selected_receiver_tiles,
                    tile_idx=tile_idx,
                    n_rx=n_rx,
                )
                local_n_rx = int(dr.width(receiver_idx))
                chunk_size = _radio_map_diffraction_chunk_size(
                    local_n_states,
                    local_n_rx,
                    pair_chunk_budget=pair_chunk_budget,
                )
                for state_start in range(0, local_n_states, chunk_size):
                    chunk_n = min(chunk_size, local_n_states - state_start)
                    chunk_state_idx = dr.gather(
                        type(tile_state_idx),
                        tile_state_idx,
                        dr.arange(wt.UInt32, chunk_n) + wt.UInt32(state_start),
                    )
                    if native_vector_replay_active:
                        (
                            chunk_coherent,
                            chunk_power,
                            chunk_incident_cross,
                            chunk_reflection_cross,
                            chunk_path_count,
                        ) = _accumulate_diffraction_tiled_vector_power_native(
                            local_state_idx=chunk_state_idx,
                            receiver_idx=receiver_idx,
                            state_arrays=state_arrays,
                            rx_pos=rx_pos,
                            scene=scene,
                            k=k,
                            wavelength=wavelength,
                            material_detail=material_detail,
                            rx_polarization=active_rx_polarization,
                            vector_coherent=vector_coherent,
                            coherent_target=coherent_target,
                            power_target=power_target,
                            vector_target=vector_target,
                            shadow_support_cutoff_db=shadow_support_cutoff_db,
                            coherent_scale=coherent_scale,
                            power_scale=power_scale,
                            vector_weight=vector_weight,
                            coherent_accumulator=coherent,
                            power_accumulator=power,
                            diagnostic_counts=diagnostic_counts,
                        )
                    else:
                        n_pairs = chunk_n * local_n_rx
                        pair_idx = dr.arange(wt.UInt32, n_pairs)
                        local_state_slot = pair_idx // local_n_rx
                        local_rx_slot = pair_idx % local_n_rx
                        state_idx = dr.gather(type(chunk_state_idx), chunk_state_idx, local_state_slot)
                        rx_idx = dr.gather(type(receiver_idx), receiver_idx, local_rx_slot)
                        (
                            chunk_coherent,
                            chunk_power,
                            chunk_incident_cross,
                            chunk_reflection_cross,
                            chunk_path_count,
                        ) = _accumulate_diffraction_pairs_scalar_power(
                            state_idx=state_idx,
                            rx_idx=rx_idx,
                            state_arrays=state_arrays,
                            rx_pos=rx_pos,
                            scene=scene,
                            k=k,
                            wavelength=wavelength,
                            material_detail=material_detail,
                            rx_polarization=active_rx_polarization,
                            receiver_model=receiver_model,
                            vector_coherent=vector_coherent,
                            vector_target=vector_target,
                            coherent_target=coherent_target,
                            power_target=power_target,
                            incident_reference_vector=incident_reference_vector,
                            reflection_reference_vector=reflection_reference_vector,
                            incident_cross_target=incident_cross_target,
                            reflection_cross_target=reflection_cross_target,
                            shadow_support_cutoff_db=shadow_support_cutoff_db,
                            coherent_scale=coherent_scale,
                            power_scale=power_scale,
                            cross_scale=cross_scale,
                            vector_weight=vector_weight,
                            output_rx_idx=local_rx_slot,
                            output_receiver_idx=receiver_idx,
                            output_n_rx=local_n_rx,
                            coherent_accumulator=coherent,
                            power_accumulator=power,
                            incident_cross_accumulator=incident_cross,
                            reflection_cross_accumulator=reflection_cross,
                            diagnostic_counts=diagnostic_counts,
                        )
                    path_count += chunk_path_count
    if not use_tiled:
        state_chunk_size = _radio_map_diffraction_chunk_size(
            n_states,
            n_rx,
            pair_chunk_budget=pair_chunk_budget,
        )
        for state_start in range(0, n_states, state_chunk_size):
            chunk_n_states = min(state_chunk_size, n_states - state_start)
            chunk_state_idx = dr.arange(wt.UInt32, chunk_n_states) + wt.UInt32(state_start)
            if native_vector_replay_active:
                (
                    chunk_coherent,
                    chunk_power,
                    chunk_incident_cross,
                    chunk_reflection_cross,
                    chunk_path_count,
                ) = _accumulate_diffraction_tiled_vector_power_native(
                    local_state_idx=chunk_state_idx,
                    receiver_idx=dr.arange(wt.UInt32, n_rx),
                    state_arrays=state_arrays,
                    rx_pos=rx_pos,
                    scene=scene,
                    k=k,
                    wavelength=wavelength,
                    material_detail=material_detail,
                    rx_polarization=active_rx_polarization,
                    vector_coherent=vector_coherent,
                    coherent_target=coherent_target,
                    power_target=power_target,
                    vector_target=vector_target,
                    shadow_support_cutoff_db=shadow_support_cutoff_db,
                    coherent_scale=coherent_scale,
                    power_scale=power_scale,
                    vector_weight=vector_weight,
                    coherent_accumulator=coherent,
                    power_accumulator=power,
                    diagnostic_counts=diagnostic_counts,
                )
            else:
                n_pairs = chunk_n_states * n_rx
                pair_idx = dr.arange(wt.UInt32, n_pairs)
                state_idx = pair_idx // n_rx + wt.UInt32(state_start)
                rx_idx = pair_idx % n_rx
                (
                    chunk_coherent,
                    chunk_power,
                    chunk_incident_cross,
                    chunk_reflection_cross,
                    chunk_path_count,
                ) = _accumulate_diffraction_pairs_scalar_power(
                    state_idx=state_idx,
                    rx_idx=rx_idx,
                    state_arrays=state_arrays,
                    rx_pos=rx_pos,
                    scene=scene,
                    k=k,
                    wavelength=wavelength,
                    material_detail=material_detail,
                    rx_polarization=active_rx_polarization,
                    receiver_model=receiver_model,
                    vector_coherent=vector_coherent,
                    vector_target=vector_target,
                    coherent_target=coherent_target,
                    power_target=power_target,
                    incident_reference_vector=incident_reference_vector,
                    reflection_reference_vector=reflection_reference_vector,
                    incident_cross_target=incident_cross_target,
                    reflection_cross_target=reflection_cross_target,
                    shadow_support_cutoff_db=shadow_support_cutoff_db,
                    coherent_scale=coherent_scale,
                    power_scale=power_scale,
                    cross_scale=cross_scale,
                    vector_weight=vector_weight,
                    coherent_accumulator=coherent,
                    power_accumulator=power,
                    incident_cross_accumulator=incident_cross,
                    reflection_cross_accumulator=reflection_cross,
                    diagnostic_counts=diagnostic_counts,
                )
            path_count += chunk_path_count

    if vector_coherent is None:
        dr.eval(coherent.real, coherent.imag, power, incident_cross, reflection_cross)
    else:
        dr.eval(
            coherent.real,
            coherent.imag,
            power,
            incident_cross,
            reflection_cross,
            vector_coherent["x"].real,
            vector_coherent["x"].imag,
            vector_coherent["y"].real,
            vector_coherent["y"].imag,
            vector_coherent["z"].real,
            vector_coherent["z"].imag,
        )
    return {
        "coherent": eval_complex(coherent),
        "power": power,
        "vector_coherent": _eval_complex_vector(vector_coherent),
        "incident_cross": incident_cross,
        "reflection_cross": reflection_cross,
        "path_count": int(path_count),
        "diagnostic_counts": diagnostic_counts,
        "planner_stats": planner_stats,
    }


__all__ = [
    "accumulate_diffraction_matched_isotropic_forward_fast",
    "accumulate_diffraction_scalar_power",
    "accumulate_matched_isb_shadow_completion",
    "accumulate_projected_isb_shadow_completion",
    "accumulate_reflection_scalar_power",
]
