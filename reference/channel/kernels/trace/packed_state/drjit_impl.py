"""
Pure-DrJit reference implementation of packed state buffer operations.

This path now treats lineage and path/audit metadata as optional cold state.
"""

from __future__ import annotations

import drjit as dr
import witwin as wt

from witwin.channel.trace.diffraction.constants import (
    SK_ADJACENT_FACE0,
    SK_ADJACENT_FACE1,
    SK_APPROXIMATION_MODE_CODE,
    SK_EDGE_DIR,
    SK_EDGE_LINE_MAX,
    SK_EDGE_LINE_MIN,
    SK_EDGE_IDX,
    SK_EDGE_POS,
    SK_FACE0_ETA_R,
    SK_FACE0_GAIN,
    SK_FACE0_OPERATOR_M00,
    SK_FACE0_OPERATOR_M01,
    SK_FACE0_OPERATOR_M10,
    SK_FACE0_OPERATOR_M11,
    SK_FACE0_SIGMA,
    SK_FACE0_USE_FRESNEL,
    SK_FACE1_ETA_R,
    SK_FACE1_GAIN,
    SK_FACE1_OPERATOR_M00,
    SK_FACE1_OPERATOR_M01,
    SK_FACE1_OPERATOR_M10,
    SK_FACE1_OPERATOR_M11,
    SK_FACE1_SIGMA,
    SK_FACE1_USE_FRESNEL,
    SK_FIRST_INTERACTION_POS,
    SK_INCIDENT_BASIS_K,
    SK_INCIDENT_BASIS_U,
    SK_INCIDENT_BASIS_V,
    SK_INCIDENT_DERIVATIVE_JONES_U,
    SK_INCIDENT_DERIVATIVE_JONES_V,
    SK_INCIDENT_FIELD,
    SK_INCIDENT_JONES_U,
    SK_INCIDENT_JONES_V,
    SK_INCIDENT_NORMAL_DERIVATIVE,
    SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X,
    SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y,
    SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z,
    SK_INCIDENT_VECTOR_X,
    SK_INCIDENT_VECTOR_Y,
    SK_INCIDENT_VECTOR_Z,
    SK_INTERMEDIATE_REFLECTION_DEPTH,
    SK_IS_DIRECT_TX,
    SK_N0,
    SK_NN,
    SK_N_STATES,
    SK_ORDER,
    SK_PATH_LENGTH_PREFIX,
    SK_PREFIX_REFLECTION_DEPTH,
    SK_R0,
    SK_RN,
    SK_SOURCE_POS,
    SK_SOURCE_TYPE_CODE,
    SK_SUFFIX_REFLECTION_DEPTH,
    SK_WEDGE_N,
    _state_history_size,
)
from witwin.channel.trace.diffraction.state.arrays import (
    _attach_state_metadata,
    _concat_state_metadata,
    _empty_state_arrays,
    _gather_state_metadata,
    _materialize_state_history,
    _state_has_cold_metadata,
)
from witwin.channel.types import InteractionType
from witwin.channel.utils import scalar
from witwin.channel.utils.drjit_ops import ArrayInit, Concat

_OPTIONAL_COLD_SPECS = (
    (SK_PATH_LENGTH_PREFIX, wt.Float),
    (SK_FIRST_INTERACTION_POS, wt.Point3f),
    (SK_IS_DIRECT_TX, wt.Bool),
    (SK_SOURCE_TYPE_CODE, wt.UInt32),
    (SK_APPROXIMATION_MODE_CODE, wt.UInt32),
    (SK_EDGE_LINE_MIN, wt.Float),
    (SK_EDGE_LINE_MAX, wt.Float),
)

_FIELD_EVAL_BASE_SPECS = (
    (SK_EDGE_POS, wt.Point3f),
    (SK_EDGE_DIR, wt.Vector3f),
    (SK_N0, wt.Vector3f),
    (SK_NN, wt.Vector3f),
    (SK_WEDGE_N, wt.Float),
    (SK_SOURCE_POS, wt.Point3f),
    (SK_EDGE_LINE_MIN, wt.Float),
    (SK_EDGE_LINE_MAX, wt.Float),
    (SK_INCIDENT_JONES_U, wt.Complex2f),
    (SK_INCIDENT_JONES_V, wt.Complex2f),
    (SK_INCIDENT_DERIVATIVE_JONES_U, wt.Complex2f),
    (SK_INCIDENT_DERIVATIVE_JONES_V, wt.Complex2f),
    (SK_INCIDENT_BASIS_U, wt.Vector3f),
    (SK_INCIDENT_BASIS_V, wt.Vector3f),
    (SK_INCIDENT_BASIS_K, wt.Vector3f),
)

_FIELD_EVAL_FACE_PARAM_SPECS = (
    (SK_FACE0_ETA_R, wt.Float),
    (SK_FACE0_SIGMA, wt.Float),
    (SK_FACE0_GAIN, wt.Float),
    (SK_FACE1_ETA_R, wt.Float),
    (SK_FACE1_SIGMA, wt.Float),
    (SK_FACE1_GAIN, wt.Float),
)

_FIELD_EVAL_OPERATOR_SPECS = (
    (SK_FACE0_OPERATOR_M00, wt.Complex2f),
    (SK_FACE0_OPERATOR_M01, wt.Complex2f),
    (SK_FACE0_OPERATOR_M10, wt.Complex2f),
    (SK_FACE0_OPERATOR_M11, wt.Complex2f),
    (SK_FACE1_OPERATOR_M00, wt.Complex2f),
    (SK_FACE1_OPERATOR_M01, wt.Complex2f),
    (SK_FACE1_OPERATOR_M10, wt.Complex2f),
    (SK_FACE1_OPERATOR_M11, wt.Complex2f),
)


def _gather_spec_fields(state_arrays: dict, indices, specs) -> dict:
    gathered = {}
    for key, dtype in specs:
        if key in state_arrays:
            gathered[key] = dr.gather(dtype, state_arrays[key], indices)
    return gathered


def _gather_optional(gathered: dict, state_arrays: dict, indices) -> None:
    for key, dtype in _OPTIONAL_COLD_SPECS:
        if key in state_arrays:
            gathered[key] = dr.gather(dtype, state_arrays[key], indices)


def _concat_optional(state_arrays: dict, non_empty: list[dict]) -> None:
    for key, dtype in _OPTIONAL_COLD_SPECS:
        if not any(key in state for state in non_empty):
            continue
        parts = []
        for source in non_empty:
            width = int(source[SK_N_STATES])
            if key in source:
                parts.append(source[key])
            elif dtype is wt.Float:
                parts.append(dr.zeros(wt.Float, width))
            elif dtype is wt.Point3f:
                zeros = dr.zeros(wt.Float, width)
                parts.append(wt.Point3f(zeros, zeros, zeros))
            elif dtype is wt.Bool:
                parts.append(dr.zeros(wt.Bool, width))
            else:
                parts.append(dr.zeros(dtype, width))
        if dtype is wt.Point3f:
            state_arrays[key] = wt.Point3f(
                Concat.arrays(wt.Float, [part.x for part in parts]),
                Concat.arrays(wt.Float, [part.y for part in parts]),
                Concat.arrays(wt.Float, [part.z for part in parts]),
            )
        else:
            state_arrays[key] = Concat.arrays(dtype, parts)


def gather_inserted_reflection_state_fields(state_arrays: dict, indices) -> dict:
    selected = {
        SK_EDGE_POS: dr.gather(wt.Point3f, state_arrays[SK_EDGE_POS], indices),
        SK_PREFIX_REFLECTION_DEPTH: dr.gather(
            wt.UInt32,
            state_arrays[SK_PREFIX_REFLECTION_DEPTH],
            indices,
        ),
        SK_INTERMEDIATE_REFLECTION_DEPTH: dr.gather(
            wt.UInt32,
            state_arrays[SK_INTERMEDIATE_REFLECTION_DEPTH],
            indices,
        ),
        SK_SUFFIX_REFLECTION_DEPTH: dr.gather(
            wt.UInt32,
            state_arrays[SK_SUFFIX_REFLECTION_DEPTH],
            indices,
        ),
        SK_ORDER: dr.gather(wt.UInt32, state_arrays[SK_ORDER], indices),
        SK_N_STATES: dr.width(indices),
    }
    if SK_PATH_LENGTH_PREFIX in state_arrays:
        selected[SK_PATH_LENGTH_PREFIX] = dr.gather(
            wt.Float,
            state_arrays[SK_PATH_LENGTH_PREFIX],
            indices,
        )
    if SK_FIRST_INTERACTION_POS in state_arrays:
        selected[SK_FIRST_INTERACTION_POS] = dr.gather(
            wt.Point3f,
            state_arrays[SK_FIRST_INTERACTION_POS],
            indices,
        )
    if SK_SOURCE_TYPE_CODE in state_arrays:
        selected[SK_SOURCE_TYPE_CODE] = dr.gather(
            wt.UInt32,
            state_arrays[SK_SOURCE_TYPE_CODE],
            indices,
        )
    return _attach_state_metadata(
        selected,
        _gather_state_metadata(state_arrays, indices),
        _state_history_size(state_arrays),
    )


def gather_field_evaluation_state_fields(state_arrays: dict, indices, *, include_stored_operators: bool = False) -> dict:
    gathered = _gather_spec_fields(
        state_arrays,
        indices,
        _FIELD_EVAL_BASE_SPECS,
    )
    face_param_keys = {key for key, _ in _FIELD_EVAL_FACE_PARAM_SPECS}
    operator_keys = {key for key, _ in _FIELD_EVAL_OPERATOR_SPECS}
    if face_param_keys.issubset(state_arrays):
        gathered.update(
            _gather_spec_fields(
                state_arrays,
                indices,
                _FIELD_EVAL_FACE_PARAM_SPECS,
            )
        )
        if include_stored_operators and operator_keys.issubset(state_arrays):
            gathered.update(
                _gather_spec_fields(
                    state_arrays,
                    indices,
                    _FIELD_EVAL_OPERATOR_SPECS,
                )
            )
    else:
        gathered.update(
            _gather_spec_fields(
                state_arrays,
                indices,
                _FIELD_EVAL_OPERATOR_SPECS,
            )
        )
    gathered[SK_N_STATES] = dr.width(indices)
    return _attach_state_metadata(
        gathered,
        _gather_state_metadata(state_arrays, indices),
        _state_history_size(state_arrays),
    )


def build_diffraction_path_slots(
    *,
    keep_states,
    edge_data,
    edge_object_idx,
    return_geometry: bool,
):
    history_size = _state_history_size(keep_states)
    materialized_edge_slots, materialized_reflection_depth_slots = _materialize_state_history(
        keep_states
    )
    count = int(keep_states["n_states"])
    if count <= 0:
        return (
            (dr.zeros(wt.Int32, 0),),
            None,
            None,
            None,
            1,
        )

    order = wt.Int32(keep_states["order"])
    prefix_depth = (
        wt.Int32(materialized_reflection_depth_slots[0])
        if history_size > 0 and len(materialized_reflection_depth_slots) > 0
        else dr.zeros(wt.Int32, count)
    )
    inserted_depth_slots = [
        wt.Int32(materialized_reflection_depth_slots[slot])
        for slot in range(1, min(history_size, len(materialized_reflection_depth_slots)))
    ]
    path_edge_slots = [
        wt.Int32(materialized_edge_slots[slot])
        for slot in range(min(history_size, len(materialized_edge_slots)))
    ]

    total_depth = prefix_depth + order
    for slot, inserted_depth in enumerate(inserted_depth_slots):
        active = order > wt.Int32(slot + 1)
        total_depth = total_depth + dr.select(active, inserted_depth, wt.Int32(0))
    max_depth = max(1, int(scalar(dr.max(total_depth))))

    type_slots = [dr.zeros(wt.Int32, count) for _ in range(max_depth)]
    vertex_slots = None
    normal_slots = None
    object_slots = None
    if return_geometry:
        vertex_slots = [ArrayInit.zeros_point3(count) for _ in range(max_depth)]
        normal_slots = [ArrayInit.zeros_vector3(count) for _ in range(max_depth)]
        object_slots = [dr.full(wt.Int32, -1, count) for _ in range(max_depth)]

    for depth_idx in range(max_depth):
        prefix_mask = prefix_depth > wt.Int32(depth_idx)
        type_slots[depth_idx] = dr.select(
            prefix_mask,
            wt.Int32(InteractionType.REFLECTION),
            type_slots[depth_idx],
        )

    if return_geometry and history_size > 0:
        prefix_mask = prefix_depth > 0
        vertex_slots[0] = dr.select(
            prefix_mask,
            keep_states["first_interaction_pos"],
            vertex_slots[0],
        )

    inserted_before = dr.zeros(wt.Int32, count)
    n_edges = 0 if edge_data is None else int(edge_data["n_edges"])
    for diff_slot in range(history_size):
        active = order > wt.Int32(diff_slot)
        diff_position = prefix_depth + wt.Int32(diff_slot) + inserted_before
        edge_idx = path_edge_slots[diff_slot]

        for depth_idx in range(max_depth):
            diff_mask = active & (diff_position == wt.Int32(depth_idx))
            type_slots[depth_idx] = dr.select(
                diff_mask,
                wt.Int32(InteractionType.DIFFRACTION),
                type_slots[depth_idx],
            )
            if return_geometry and n_edges > 0:
                valid_edge = diff_mask & (edge_idx >= 0) & (edge_idx < wt.Int32(n_edges))
                safe_edge = wt.UInt32(dr.select(valid_edge, edge_idx, wt.Int32(0)))
                edge_pos = dr.gather(wt.Point3f, edge_data["pos"], safe_edge)
                edge_normal = dr.gather(wt.Vector3f, edge_data["n0"], safe_edge)
                edge_object = dr.gather(wt.Int32, edge_object_idx, safe_edge)
                vertex_slots[depth_idx] = dr.select(valid_edge, edge_pos, vertex_slots[depth_idx])
                normal_slots[depth_idx] = dr.select(
                    valid_edge, edge_normal, normal_slots[depth_idx]
                )
                object_slots[depth_idx] = dr.select(
                    valid_edge, edge_object, object_slots[depth_idx]
                )

        if diff_slot < len(inserted_depth_slots):
            inserted_depth = inserted_depth_slots[diff_slot]
            inserted_start = diff_position + 1
            for offset in range(max_depth):
                inserted_mask = (
                    (order > wt.Int32(diff_slot + 1))
                    & (inserted_depth > wt.Int32(offset))
                    & (inserted_start + wt.Int32(offset) < wt.Int32(max_depth))
                )
                depth_position = inserted_start + wt.Int32(offset)
                for depth_idx in range(max_depth):
                    mask = inserted_mask & (depth_position == wt.Int32(depth_idx))
                    type_slots[depth_idx] = dr.select(
                        mask,
                        wt.Int32(InteractionType.REFLECTION),
                        type_slots[depth_idx],
                    )
            inserted_before = inserted_before + dr.select(
                order > wt.Int32(diff_slot + 1),
                inserted_depth,
                wt.Int32(0),
            )

    return type_slots, vertex_slots, normal_slots, object_slots, max_depth


def gather_state_arrays(state_arrays: dict, indices) -> dict:
    gathered = {
        SK_EDGE_IDX: dr.gather(wt.UInt32, state_arrays[SK_EDGE_IDX], indices),
        SK_EDGE_POS: dr.gather(wt.Point3f, state_arrays[SK_EDGE_POS], indices),
        SK_EDGE_DIR: dr.gather(wt.Vector3f, state_arrays[SK_EDGE_DIR], indices),
        SK_N0: dr.gather(wt.Vector3f, state_arrays[SK_N0], indices),
        SK_NN: dr.gather(wt.Vector3f, state_arrays[SK_NN], indices),
        SK_WEDGE_N: dr.gather(wt.Float, state_arrays[SK_WEDGE_N], indices),
        SK_ADJACENT_FACE0: dr.gather(wt.Int32, state_arrays[SK_ADJACENT_FACE0], indices),
        SK_ADJACENT_FACE1: dr.gather(wt.Int32, state_arrays[SK_ADJACENT_FACE1], indices),
        SK_SOURCE_POS: dr.gather(wt.Point3f, state_arrays[SK_SOURCE_POS], indices),
        SK_INCIDENT_FIELD: dr.gather(wt.Complex2f, state_arrays[SK_INCIDENT_FIELD], indices),
        SK_INCIDENT_NORMAL_DERIVATIVE: dr.gather(
            wt.Complex2f,
            state_arrays[SK_INCIDENT_NORMAL_DERIVATIVE],
            indices,
        ),
        SK_INCIDENT_JONES_U: dr.gather(wt.Complex2f, state_arrays[SK_INCIDENT_JONES_U], indices),
        SK_INCIDENT_JONES_V: dr.gather(wt.Complex2f, state_arrays[SK_INCIDENT_JONES_V], indices),
        SK_INCIDENT_DERIVATIVE_JONES_U: dr.gather(
            wt.Complex2f,
            state_arrays[SK_INCIDENT_DERIVATIVE_JONES_U],
            indices,
        ),
        SK_INCIDENT_DERIVATIVE_JONES_V: dr.gather(
            wt.Complex2f,
            state_arrays[SK_INCIDENT_DERIVATIVE_JONES_V],
            indices,
        ),
        SK_R0: dr.gather(wt.Complex2f, state_arrays[SK_R0], indices),
        SK_RN: dr.gather(wt.Complex2f, state_arrays[SK_RN], indices),
        SK_INCIDENT_BASIS_U: dr.gather(wt.Vector3f, state_arrays[SK_INCIDENT_BASIS_U], indices),
        SK_INCIDENT_BASIS_V: dr.gather(wt.Vector3f, state_arrays[SK_INCIDENT_BASIS_V], indices),
        SK_INCIDENT_BASIS_K: dr.gather(wt.Vector3f, state_arrays[SK_INCIDENT_BASIS_K], indices),
        SK_FACE0_OPERATOR_M00: dr.gather(wt.Complex2f, state_arrays[SK_FACE0_OPERATOR_M00], indices),
        SK_FACE0_OPERATOR_M01: dr.gather(wt.Complex2f, state_arrays[SK_FACE0_OPERATOR_M01], indices),
        SK_FACE0_OPERATOR_M10: dr.gather(wt.Complex2f, state_arrays[SK_FACE0_OPERATOR_M10], indices),
        SK_FACE0_OPERATOR_M11: dr.gather(wt.Complex2f, state_arrays[SK_FACE0_OPERATOR_M11], indices),
        SK_FACE1_OPERATOR_M00: dr.gather(wt.Complex2f, state_arrays[SK_FACE1_OPERATOR_M00], indices),
        SK_FACE1_OPERATOR_M01: dr.gather(wt.Complex2f, state_arrays[SK_FACE1_OPERATOR_M01], indices),
        SK_FACE1_OPERATOR_M10: dr.gather(wt.Complex2f, state_arrays[SK_FACE1_OPERATOR_M10], indices),
        SK_FACE1_OPERATOR_M11: dr.gather(wt.Complex2f, state_arrays[SK_FACE1_OPERATOR_M11], indices),
        SK_FACE0_ETA_R: dr.gather(wt.Float, state_arrays[SK_FACE0_ETA_R], indices),
        SK_FACE0_SIGMA: dr.gather(wt.Float, state_arrays[SK_FACE0_SIGMA], indices),
        SK_FACE0_GAIN: dr.gather(wt.Float, state_arrays[SK_FACE0_GAIN], indices),
        SK_FACE0_USE_FRESNEL: dr.gather(wt.Bool, state_arrays[SK_FACE0_USE_FRESNEL], indices),
        SK_FACE1_ETA_R: dr.gather(wt.Float, state_arrays[SK_FACE1_ETA_R], indices),
        SK_FACE1_SIGMA: dr.gather(wt.Float, state_arrays[SK_FACE1_SIGMA], indices),
        SK_FACE1_GAIN: dr.gather(wt.Float, state_arrays[SK_FACE1_GAIN], indices),
        SK_FACE1_USE_FRESNEL: dr.gather(wt.Bool, state_arrays[SK_FACE1_USE_FRESNEL], indices),
        SK_INCIDENT_VECTOR_X: dr.gather(wt.Complex2f, state_arrays[SK_INCIDENT_VECTOR_X], indices),
        SK_INCIDENT_VECTOR_Y: dr.gather(wt.Complex2f, state_arrays[SK_INCIDENT_VECTOR_Y], indices),
        SK_INCIDENT_VECTOR_Z: dr.gather(wt.Complex2f, state_arrays[SK_INCIDENT_VECTOR_Z], indices),
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X: dr.gather(
            wt.Complex2f,
            state_arrays[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X],
            indices,
        ),
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y: dr.gather(
            wt.Complex2f,
            state_arrays[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y],
            indices,
        ),
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z: dr.gather(
            wt.Complex2f,
            state_arrays[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z],
            indices,
        ),
        SK_PREFIX_REFLECTION_DEPTH: dr.gather(
            wt.UInt32,
            state_arrays[SK_PREFIX_REFLECTION_DEPTH],
            indices,
        ),
        SK_INTERMEDIATE_REFLECTION_DEPTH: dr.gather(
            wt.UInt32,
            state_arrays[SK_INTERMEDIATE_REFLECTION_DEPTH],
            indices,
        ),
        SK_SUFFIX_REFLECTION_DEPTH: dr.gather(
            wt.UInt32,
            state_arrays[SK_SUFFIX_REFLECTION_DEPTH],
            indices,
        ),
        SK_ORDER: dr.gather(wt.UInt32, state_arrays[SK_ORDER], indices),
        SK_N_STATES: dr.width(indices),
    }
    _gather_optional(gathered, state_arrays, indices)
    return _attach_state_metadata(
        gathered,
        _gather_state_metadata(state_arrays, indices),
        _state_history_size(state_arrays),
    )


def concat_state_arrays(state_arrays_list: list[dict]) -> dict:
    non_empty = [
        state_arrays
        for state_arrays in state_arrays_list
        if state_arrays is not None and state_arrays[SK_N_STATES] > 0
    ]
    if len(non_empty) == 0:
        history_size = 0
        retain_cold = False
        for state_arrays in state_arrays_list:
            if state_arrays is not None:
                history_size = max(history_size, _state_history_size(state_arrays))
                retain_cold = retain_cold or _state_has_cold_metadata(state_arrays)
        return _empty_state_arrays(history_size=history_size, retain_cold_metadata=retain_cold)
    if len(non_empty) == 1:
        return non_empty[0]

    history_size = max(_state_history_size(state_arrays) for state_arrays in non_empty)
    state_arrays = {
        SK_EDGE_IDX: Concat.arrays(wt.UInt32, [state[SK_EDGE_IDX] for state in non_empty]),
        SK_EDGE_POS: wt.Point3f(
            Concat.arrays(wt.Float, [state[SK_EDGE_POS].x for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_EDGE_POS].y for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_EDGE_POS].z for state in non_empty]),
        ),
        SK_EDGE_DIR: wt.Vector3f(
            Concat.arrays(wt.Float, [state[SK_EDGE_DIR].x for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_EDGE_DIR].y for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_EDGE_DIR].z for state in non_empty]),
        ),
        SK_N0: wt.Vector3f(
            Concat.arrays(wt.Float, [state[SK_N0].x for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_N0].y for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_N0].z for state in non_empty]),
        ),
        SK_NN: wt.Vector3f(
            Concat.arrays(wt.Float, [state[SK_NN].x for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_NN].y for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_NN].z for state in non_empty]),
        ),
        SK_WEDGE_N: Concat.arrays(wt.Float, [state[SK_WEDGE_N] for state in non_empty]),
        SK_ADJACENT_FACE0: Concat.arrays(wt.Int32, [state[SK_ADJACENT_FACE0] for state in non_empty]),
        SK_ADJACENT_FACE1: Concat.arrays(wt.Int32, [state[SK_ADJACENT_FACE1] for state in non_empty]),
        SK_SOURCE_POS: wt.Point3f(
            Concat.arrays(wt.Float, [state[SK_SOURCE_POS].x for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_SOURCE_POS].y for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_SOURCE_POS].z for state in non_empty]),
        ),
        SK_INCIDENT_FIELD: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_FIELD].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_FIELD].imag for state in non_empty]),
        ),
        SK_INCIDENT_NORMAL_DERIVATIVE: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_NORMAL_DERIVATIVE].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_NORMAL_DERIVATIVE].imag for state in non_empty]),
        ),
        SK_INCIDENT_JONES_U: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_JONES_U].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_JONES_U].imag for state in non_empty]),
        ),
        SK_INCIDENT_JONES_V: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_JONES_V].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_JONES_V].imag for state in non_empty]),
        ),
        SK_INCIDENT_DERIVATIVE_JONES_U: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_DERIVATIVE_JONES_U].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_DERIVATIVE_JONES_U].imag for state in non_empty]),
        ),
        SK_INCIDENT_DERIVATIVE_JONES_V: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_DERIVATIVE_JONES_V].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_DERIVATIVE_JONES_V].imag for state in non_empty]),
        ),
        SK_R0: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_R0].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_R0].imag for state in non_empty]),
        ),
        SK_RN: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_RN].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_RN].imag for state in non_empty]),
        ),
        SK_INCIDENT_BASIS_U: wt.Vector3f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_BASIS_U].x for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_BASIS_U].y for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_BASIS_U].z for state in non_empty]),
        ),
        SK_INCIDENT_BASIS_V: wt.Vector3f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_BASIS_V].x for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_BASIS_V].y for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_BASIS_V].z for state in non_empty]),
        ),
        SK_INCIDENT_BASIS_K: wt.Vector3f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_BASIS_K].x for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_BASIS_K].y for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_BASIS_K].z for state in non_empty]),
        ),
        SK_FACE0_OPERATOR_M00: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_FACE0_OPERATOR_M00].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_FACE0_OPERATOR_M00].imag for state in non_empty]),
        ),
        SK_FACE0_OPERATOR_M01: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_FACE0_OPERATOR_M01].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_FACE0_OPERATOR_M01].imag for state in non_empty]),
        ),
        SK_FACE0_OPERATOR_M10: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_FACE0_OPERATOR_M10].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_FACE0_OPERATOR_M10].imag for state in non_empty]),
        ),
        SK_FACE0_OPERATOR_M11: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_FACE0_OPERATOR_M11].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_FACE0_OPERATOR_M11].imag for state in non_empty]),
        ),
        SK_FACE1_OPERATOR_M00: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_FACE1_OPERATOR_M00].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_FACE1_OPERATOR_M00].imag for state in non_empty]),
        ),
        SK_FACE1_OPERATOR_M01: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_FACE1_OPERATOR_M01].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_FACE1_OPERATOR_M01].imag for state in non_empty]),
        ),
        SK_FACE1_OPERATOR_M10: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_FACE1_OPERATOR_M10].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_FACE1_OPERATOR_M10].imag for state in non_empty]),
        ),
        SK_FACE1_OPERATOR_M11: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_FACE1_OPERATOR_M11].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_FACE1_OPERATOR_M11].imag for state in non_empty]),
        ),
        SK_FACE0_ETA_R: Concat.arrays(wt.Float, [state[SK_FACE0_ETA_R] for state in non_empty]),
        SK_FACE0_SIGMA: Concat.arrays(wt.Float, [state[SK_FACE0_SIGMA] for state in non_empty]),
        SK_FACE0_GAIN: Concat.arrays(wt.Float, [state[SK_FACE0_GAIN] for state in non_empty]),
        SK_FACE0_USE_FRESNEL: Concat.arrays(wt.Bool, [state[SK_FACE0_USE_FRESNEL] for state in non_empty]),
        SK_FACE1_ETA_R: Concat.arrays(wt.Float, [state[SK_FACE1_ETA_R] for state in non_empty]),
        SK_FACE1_SIGMA: Concat.arrays(wt.Float, [state[SK_FACE1_SIGMA] for state in non_empty]),
        SK_FACE1_GAIN: Concat.arrays(wt.Float, [state[SK_FACE1_GAIN] for state in non_empty]),
        SK_FACE1_USE_FRESNEL: Concat.arrays(wt.Bool, [state[SK_FACE1_USE_FRESNEL] for state in non_empty]),
        SK_INCIDENT_VECTOR_X: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_VECTOR_X].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_VECTOR_X].imag for state in non_empty]),
        ),
        SK_INCIDENT_VECTOR_Y: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_VECTOR_Y].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_VECTOR_Y].imag for state in non_empty]),
        ),
        SK_INCIDENT_VECTOR_Z: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_VECTOR_Z].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_VECTOR_Z].imag for state in non_empty]),
        ),
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X].imag for state in non_empty]),
        ),
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y].imag for state in non_empty]),
        ),
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z: wt.Complex2f(
            Concat.arrays(wt.Float, [state[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z].real for state in non_empty]),
            Concat.arrays(wt.Float, [state[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z].imag for state in non_empty]),
        ),
        SK_PREFIX_REFLECTION_DEPTH: Concat.arrays(wt.UInt32, [state[SK_PREFIX_REFLECTION_DEPTH] for state in non_empty]),
        SK_INTERMEDIATE_REFLECTION_DEPTH: Concat.arrays(
            wt.UInt32,
            [state[SK_INTERMEDIATE_REFLECTION_DEPTH] for state in non_empty],
        ),
        SK_SUFFIX_REFLECTION_DEPTH: Concat.arrays(wt.UInt32, [state[SK_SUFFIX_REFLECTION_DEPTH] for state in non_empty]),
        SK_ORDER: Concat.arrays(wt.UInt32, [state[SK_ORDER] for state in non_empty]),
        SK_N_STATES: sum(state[SK_N_STATES] for state in non_empty),
    }
    _concat_optional(state_arrays, non_empty)
    return _attach_state_metadata(
        state_arrays,
        _concat_state_metadata(non_empty),
        history_size,
    )


def subset_state_arrays(state_arrays: dict, mask) -> dict:
    history_size = _state_history_size(state_arrays)
    if state_arrays is None or state_arrays[SK_N_STATES] == 0:
        return _empty_state_arrays(
            history_size=history_size,
            retain_cold_metadata=_state_has_cold_metadata(state_arrays),
        )
    keep_idx = dr.compress(mask)
    if dr.width(keep_idx) == 0:
        return _empty_state_arrays(
            history_size=history_size,
            retain_cold_metadata=_state_has_cold_metadata(state_arrays),
        )
    gathered = gather_state_arrays(state_arrays, keep_idx)
    gathered[SK_N_STATES] = dr.width(keep_idx)
    return gathered
