"""TX-first-order diffraction state construction."""

import drjit as dr
import witwin as wt

from ....utils.polarization import (
    project_real_polarization_to_ray,
    vector_from_scalar_and_real_direction,
    vector_zero,
)
from ....utils.drjit_ops import ArrayInit
from ..constants import (
    APPROX_MODE_DIRECT_FIRST_ORDER,
    SOURCE_TYPE_DIRECT_TX,
)
from ..finite_wedge import require_edge_data_line_bounds
from ..geometry import (
    _point_source_field,
    _segment_visibility_mask,
    _wedge_exterior_region_mask,
)
from ..state import (
    _empty_state_arrays,
    _make_state_arrays,
)
from . import _state_edge_face_material_response
from witwin.channel.kernels.trace.cartesian_filter import compact_index_pairs

__all__ = [
    "_build_tx_first_order_state_arrays",
]


def _build_tx_first_order_state_arrays(
    tx_pos,
    edge_data,
    wavelength,
    k,
    history_size=3,
    retain_cold_metadata=True,
    scene=None,
    material_detail=None,
    reflection_coef=1.0,
    use_scene_materials=False,
    tx_polarization=(1.0, 0.0, 0.0),
):
    n_edges = edge_data["n_edges"]
    if n_edges == 0:
        return _empty_state_arrays(history_size=history_size)
    line_min_all, line_max_all = require_edge_data_line_bounds(
        edge_data,
        context="_build_tx_first_order_state_arrays",
    )

    edge_idx = dr.arange(wt.UInt32, n_edges)
    edge_pos = edge_data["pos"]
    adjacent_face0 = edge_data["adjacent_face0"]
    adjacent_face1 = edge_data["adjacent_face1"]
    source_pos = wt.Point3f(dr.repeat(tx_pos.x, n_edges), dr.repeat(tx_pos.y, n_edges), dr.repeat(tx_pos.z, n_edges))
    visible = _segment_visibility_mask(source_pos, edge_pos, scene, ignore_prim_idx=(adjacent_face0, adjacent_face1))
    source_exterior = _wedge_exterior_region_mask(
        source_pos - edge_pos,
        edge_data["edge_dir"],
        edge_data["n0"],
        edge_data["n_face_n"],
    )
    edge_idx, _ = compact_index_pairs(edge_idx, edge_idx, visible & source_exterior)
    n_states = dr.width(edge_idx)
    if n_states == 0:
        return _empty_state_arrays(history_size=history_size)

    edge_pos = dr.gather(wt.Point3f, edge_data["pos"], edge_idx)
    edge_dir = dr.gather(wt.Vector3f, edge_data["edge_dir"], edge_idx)
    n0 = dr.gather(wt.Vector3f, edge_data["n0"], edge_idx)
    nn = dr.gather(wt.Vector3f, edge_data["n_face_n"], edge_idx)
    wedge_n = dr.gather(wt.Float, edge_data["wedge_n"], edge_idx)
    edge_line_min = dr.gather(wt.Float, line_min_all, edge_idx)
    edge_line_max = dr.gather(wt.Float, line_max_all, edge_idx)
    adjacent_face0 = dr.gather(wt.Int32, edge_data["adjacent_face0"], edge_idx)
    adjacent_face1 = dr.gather(wt.Int32, edge_data["adjacent_face1"], edge_idx)
    source_pos = wt.Point3f(
        dr.repeat(tx_pos.x, n_states),
        dr.repeat(tx_pos.y, n_states),
        dr.repeat(tx_pos.z, n_states),
    )
    path_length_prefix = dr.norm(edge_pos - source_pos)
    incident_field = _point_source_field(
        tx_pos,
        wt.Complex2f(1.0, 0.0),
        edge_pos,
        wavelength,
        k,
    )
    incident_normal_derivative = ArrayInit.complex_zero(n_states)
    ray_dir = (edge_pos - source_pos) / (dr.norm(edge_pos - source_pos) + 1e-12)
    pol_dir = project_real_polarization_to_ray(tx_polarization, ray_dir)
    incident_vector = vector_from_scalar_and_real_direction(incident_field, pol_dir)
    incident_normal_derivative_vector = vector_zero(n_states)
    face0_operator, face1_operator, face0_material, face1_material = _state_edge_face_material_response(
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=n0,
        nn=nn,
        source_pos=source_pos,
        adjacent_face0=adjacent_face0,
        adjacent_face1=adjacent_face1,
        wavelength=wavelength,
        scene=scene,
        material_detail=material_detail,
        reflection_coef=reflection_coef,
        use_scene_materials=use_scene_materials,
        tx_polarization=tx_polarization,
    )
    is_direct_tx = dr.full(wt.Bool, True, n_states)
    source_type_code = dr.full(wt.UInt32, SOURCE_TYPE_DIRECT_TX, n_states)
    prefix_reflection_depth = dr.zeros(wt.UInt32, n_states)
    intermediate_reflection_depth = dr.zeros(wt.UInt32, n_states)
    suffix_reflection_depth = dr.zeros(wt.UInt32, n_states)
    approximation_mode_code = dr.full(wt.UInt32, APPROX_MODE_DIRECT_FIRST_ORDER, n_states)
    order = dr.full(wt.UInt32, 1, n_states)
    return _make_state_arrays(
        edge_idx=edge_idx,
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=n0,
        nn=nn,
        wedge_n=wedge_n,
        adjacent_face0=adjacent_face0,
        adjacent_face1=adjacent_face1,
        source_pos=source_pos,
        path_length_prefix=path_length_prefix,
        first_interaction_pos=edge_pos,
        edge_line_min=edge_line_min,
        edge_line_max=edge_line_max,
        incident_field=incident_field,
        incident_normal_derivative=incident_normal_derivative,
        incident_vector=incident_vector,
        incident_normal_derivative_vector=incident_normal_derivative_vector,
        is_direct_tx=is_direct_tx,
        face0_operator=face0_operator,
        face1_operator=face1_operator,
        face0_material=face0_material,
        face1_material=face1_material,
        source_type_code=source_type_code,
        prefix_reflection_depth=prefix_reflection_depth,
        intermediate_reflection_depth=intermediate_reflection_depth,
        suffix_reflection_depth=suffix_reflection_depth,
        approximation_mode_code=approximation_mode_code,
        order=order,
        lineage_parent_state_id=dr.full(wt.Int32, -1, n_states),
        lineage_last_edge_idx=wt.Int32(edge_idx),
        lineage_last_reflection_depth_delta=dr.zeros(wt.UInt32, n_states),
        retain_cold_metadata=retain_cold_metadata,
    )
