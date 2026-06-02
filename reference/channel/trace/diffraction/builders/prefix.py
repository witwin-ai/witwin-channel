"""Reflection-prefix first-order diffraction state construction."""

import time

import drjit as dr
import witwin as wt

from ...materials import coerce_reflection_trace_detail
from ....utils.polarization import (
    vector_scale,
    vector_zero,
)
from ....utils.drjit_ops import ArrayInit, Gather, complex_abs_sqr
from ..constants import (
    APPROX_MODE_SAMPLED_REFLECTION_PREFIX,
    SOURCE_TYPE_REFLECTION_PREFIX,
    _cartesian_chunk_size,
)
from ..finite_wedge import require_edge_data_line_bounds
from ..geometry import (
    _point_source_field,
    _wedge_exterior_region_mask,
)
from ...reflection.epc import epc_reflection_chain_to_target
from ..state import (
    _empty_state_arrays,
    _make_state_arrays,
)
from witwin.channel.kernels.trace.packed_state import concat_state_arrays
from . import _state_edge_face_material_response, _vector_field_power

__all__ = [
    "_build_reflection_first_order_state_arrays",
]


def _build_reflection_first_order_state_arrays(
    reflection_detail,
    scene,
    edge_data,
    wavelength,
    k,
    global_to_local_idx,
    history_size=3,
    retain_cold_metadata=True,
    material_detail=None,
    reflection_coef=1.0,
    use_scene_materials=False,
    tx_polarization=(1.0, 0.0, 0.0),
    stats=None,
):
    total_start = time.perf_counter()
    if stats is None:
        stats = {}
    stats.clear()
    stats.update(
        {
            "bounce_count": 0,
            "input_paths": 0,
            "chunk_count": 0,
            "candidate_pairs": 0,
            "max_candidate_pairs_per_chunk": 0,
            "support_kept_count": 0,
            "field_kept_count": 0,
            "output_states": 0,
            "timing": {
                "replay_seconds": 0.0,
                "support_filter_seconds": 0.0,
                "field_filter_seconds": 0.0,
                "state_pack_seconds": 0.0,
                "material_response_seconds": 0.0,
                "state_array_assembly_seconds": 0.0,
                "concat_seconds": 0.0,
                "total_seconds": 0.0,
            },
        }
    )
    if reflection_detail is None or scene is None:
        stats["timing"]["total_seconds"] = time.perf_counter() - total_start
        return _empty_state_arrays(history_size=history_size)
    detail = coerce_reflection_trace_detail(reflection_detail)

    per_bounce_states = []
    if scene.tri_data_gpu is None:
        stats["timing"]["total_seconds"] = time.perf_counter() - total_start
        return _empty_state_arrays(history_size=history_size)
    line_min_all, line_max_all = require_edge_data_line_bounds(
        edge_data,
        context="_build_reflection_first_order_state_arrays",
    )

    for bounce_idx, paths in enumerate(detail.source_paths_per_bounce):
        chain_depth = 0 if paths is None else int(paths.chain_depth)
        n_paths = 0 if paths is None else int(paths.n_paths)
        if n_paths <= 0 or chain_depth <= 0:
            continue

        stats["bounce_count"] += 1
        stats["input_paths"] += n_paths
        n_edges = edge_data["n_edges"]
        path_chunk_size = _cartesian_chunk_size(n_paths, n_edges)
        for path_start in range(0, n_paths, path_chunk_size):
            stats["chunk_count"] += 1
            chunk_n_paths = min(path_chunk_size, n_paths - path_start)
            n_pairs = chunk_n_paths * n_edges
            stats["candidate_pairs"] += int(n_pairs)
            stats["max_candidate_pairs_per_chunk"] = max(
                int(stats["max_candidate_pairs_per_chunk"]),
                int(n_pairs),
            )
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            path_idx = pair_idx // n_edges + wt.UInt32(path_start)
            local_edge_idx = pair_idx % n_edges

            source_pos = Gather.point3(paths.image_source, path_idx)
            edge_pos = dr.gather(wt.Point3f, edge_data["pos"], local_edge_idx)
            edge_dir = dr.gather(wt.Vector3f, edge_data["edge_dir"], local_edge_idx)
            n0 = dr.gather(wt.Vector3f, edge_data["n0"], local_edge_idx)
            nn = dr.gather(wt.Vector3f, edge_data["n_face_n"], local_edge_idx)
            wedge_n = dr.gather(wt.Float, edge_data["wedge_n"], local_edge_idx)
            edge_line_min = dr.gather(wt.Float, line_min_all, local_edge_idx)
            edge_line_max = dr.gather(wt.Float, line_max_all, local_edge_idx)
            adjacent_face0 = dr.gather(wt.Int32, edge_data["adjacent_face0"], local_edge_idx)
            adjacent_face1 = dr.gather(wt.Int32, edge_data["adjacent_face1"], local_edge_idx)

            replay_start = time.perf_counter()
            has_reflected_support, chain_vector, chain_geometry = epc_reflection_chain_to_target(
                paths=paths,
                path_idx=path_idx,
                target_pos=edge_pos,
                scene=scene,
                target_adjacent_faces=(adjacent_face0, adjacent_face1),
                reflection_detail=detail,
                wavelength=wavelength,
                tx_polarization=tx_polarization,
                return_endpoints=True,
            )
            stats["timing"]["replay_seconds"] += time.perf_counter() - replay_start
            support_start = time.perf_counter()
            source_exterior = _wedge_exterior_region_mask(
                source_pos - edge_pos,
                edge_dir,
                n0,
                nn,
            )
            support_mask = has_reflected_support & source_exterior
            keep_idx = dr.compress(support_mask)
            stats["timing"]["support_filter_seconds"] += time.perf_counter() - support_start
            stats["support_kept_count"] += int(dr.width(keep_idx))
            if dr.width(keep_idx) == 0:
                continue

            field_start = time.perf_counter()
            source_pos_support = Gather.point3(source_pos, keep_idx)
            edge_pos_support = dr.gather(wt.Point3f, edge_pos, keep_idx)
            chain_vector_support = {
                "x": dr.gather(wt.Complex2f, chain_vector["x"], keep_idx),
                "y": dr.gather(wt.Complex2f, chain_vector["y"], keep_idx),
                "z": dr.gather(wt.Complex2f, chain_vector["z"], keep_idx),
            }
            unit_incident_field = _point_source_field(
                source_pos_support,
                wt.Complex2f(1.0, 0.0),
                edge_pos_support,
                wavelength,
                k,
            )
            field_power = _vector_field_power(chain_vector_support) * complex_abs_sqr(unit_incident_field)
            valid_field_idx = dr.compress(field_power > wt.Float(1e-20))
            stats["timing"]["field_filter_seconds"] += time.perf_counter() - field_start
            stats["field_kept_count"] += int(dr.width(valid_field_idx))
            if dr.width(valid_field_idx) == 0:
                continue

            pack_start = time.perf_counter()
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
            source_pos_keep = Gather.point3(source_pos, final_idx)
            first_interaction_pos_keep = dr.gather(
                wt.Point3f,
                chain_geometry["first_hit"],
                final_idx,
            )
            path_length_prefix = dr.norm(edge_pos_keep - source_pos_keep)
            chain_vector_keep = {
                "x": dr.gather(wt.Complex2f, chain_vector_support["x"], valid_field_idx),
                "y": dr.gather(wt.Complex2f, chain_vector_support["y"], valid_field_idx),
                "z": dr.gather(wt.Complex2f, chain_vector_support["z"], valid_field_idx),
            }
            unit_incident_field = dr.gather(wt.Complex2f, unit_incident_field, valid_field_idx)
            incident_field_keep = unit_incident_field
            incident_vector_keep = vector_scale(chain_vector_keep, unit_incident_field)
            # Reflection-prefix first-order states are still point-source excitations at the
            # diffraction edge. Carrying a point-source normal derivative here spuriously
            # activates the slope diffraction branch and overwhelms the mixed field.
            incident_normal_derivative = ArrayInit.complex_zero(dr.width(valid_field_idx))
            incident_normal_derivative_vector = vector_zero(dr.width(valid_field_idx))
            n_valid_states = dr.width(valid_field_idx)
            stats["timing"]["state_pack_seconds"] += time.perf_counter() - pack_start
            material_start = time.perf_counter()
            face0_operator, face1_operator, face0_material, face1_material = _state_edge_face_material_response(
                edge_pos=edge_pos_keep,
                edge_dir=edge_dir_keep,
                n0=n0_keep,
                nn=nn_keep,
                source_pos=source_pos_keep,
                adjacent_face0=adjacent_face0_keep,
                adjacent_face1=adjacent_face1_keep,
                wavelength=wavelength,
                scene=scene,
                material_detail=material_detail,
                reflection_coef=reflection_coef,
                use_scene_materials=use_scene_materials,
                tx_polarization=tx_polarization,
            )
            stats["timing"]["material_response_seconds"] += time.perf_counter() - material_start
            prefix_reflection_depth = dr.full(wt.UInt32, bounce_idx + 1, n_valid_states)
            intermediate_reflection_depth = dr.zeros(wt.UInt32, n_valid_states)
            suffix_reflection_depth = dr.zeros(wt.UInt32, n_valid_states)
            approximation_mode_code = dr.full(wt.UInt32, APPROX_MODE_SAMPLED_REFLECTION_PREFIX, n_valid_states)
            order = dr.full(wt.UInt32, 1, n_valid_states)
            assembly_start = time.perf_counter()
            per_bounce_states.append(_make_state_arrays(
                edge_idx=local_edge_keep,
                edge_pos=edge_pos_keep,
                edge_dir=edge_dir_keep,
                n0=n0_keep,
                nn=nn_keep,
                wedge_n=wedge_n_keep,
                adjacent_face0=adjacent_face0_keep,
                adjacent_face1=adjacent_face1_keep,
                source_pos=source_pos_keep,
                path_length_prefix=path_length_prefix,
                first_interaction_pos=first_interaction_pos_keep,
                edge_line_min=edge_line_min_keep,
                edge_line_max=edge_line_max_keep,
                incident_field=incident_field_keep,
                incident_normal_derivative=incident_normal_derivative,
                incident_vector=incident_vector_keep,
                incident_normal_derivative_vector=incident_normal_derivative_vector,
                is_direct_tx=dr.full(wt.Bool, False, n_valid_states),
                face0_operator=face0_operator,
                face1_operator=face1_operator,
                face0_material=face0_material,
                face1_material=face1_material,
                source_type_code=dr.full(wt.UInt32, SOURCE_TYPE_REFLECTION_PREFIX, n_valid_states),
                prefix_reflection_depth=prefix_reflection_depth,
                intermediate_reflection_depth=intermediate_reflection_depth,
                suffix_reflection_depth=suffix_reflection_depth,
                approximation_mode_code=approximation_mode_code,
                order=order,
                lineage_parent_state_id=dr.full(wt.Int32, -1, n_valid_states),
                lineage_last_edge_idx=wt.Int32(local_edge_keep),
                lineage_last_reflection_depth_delta=prefix_reflection_depth,
                retain_cold_metadata=retain_cold_metadata,
            ))
            stats["timing"]["state_array_assembly_seconds"] += time.perf_counter() - assembly_start

    concat_start = time.perf_counter()
    result = concat_state_arrays(per_bounce_states)
    stats["timing"]["concat_seconds"] = time.perf_counter() - concat_start
    stats["output_states"] = int(result["n_states"])
    stats["timing"]["total_seconds"] = time.perf_counter() - total_start
    return result
