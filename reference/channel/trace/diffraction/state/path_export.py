"""Path-export specific reduced diffraction state helpers."""

from __future__ import annotations

import drjit as dr
import witwin as wt

from ..constants import (
    STATE_HISTORY_SIZE_METADATA_KEY,
    SK_ADJACENT_FACE0,
    SK_ADJACENT_FACE1,
    SK_EDGE_DIR,
    SK_EDGE_LINE_MAX,
    SK_EDGE_LINE_MIN,
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
    SK_INCIDENT_FIELD,
    SK_INCIDENT_DERIVATIVE_JONES_U,
    SK_INCIDENT_DERIVATIVE_JONES_V,
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
from .arrays import (
    _META_LAST_EDGE_IDX,
    _META_LAST_REFLECTION_DEPTH_DELTA,
    _META_LINEAGE_STORE,
    _META_PARENT_STATE_ID,
    _META_STATE_ID,
    _attach_state_metadata,
    _state_metadata,
)
_PATH_EXPORT_STATE_LAYOUT_KEY = "__path_export_state_layout__"
_PATH_EXPORT_LINEAGE_METADATA_KEY = "__path_export_lineage_metadata__"
PATH_EXPORT_REDUCED_STATE_LAYOUT = "reduced_v2"

_PATH_EXPORT_EVAL_KEYS = (
    SK_EDGE_POS,
    SK_EDGE_DIR,
    SK_N0,
    SK_NN,
    SK_WEDGE_N,
    SK_ADJACENT_FACE0,
    SK_ADJACENT_FACE1,
    SK_EDGE_LINE_MIN,
    SK_EDGE_LINE_MAX,
    SK_SOURCE_POS,
    SK_INCIDENT_FIELD,
    SK_INCIDENT_NORMAL_DERIVATIVE,
    SK_INCIDENT_JONES_U,
    SK_INCIDENT_JONES_V,
    SK_INCIDENT_DERIVATIVE_JONES_U,
    SK_INCIDENT_DERIVATIVE_JONES_V,
    SK_INCIDENT_BASIS_U,
    SK_INCIDENT_BASIS_V,
    SK_INCIDENT_BASIS_K,
    SK_R0,
    SK_RN,
    SK_INCIDENT_VECTOR_X,
    SK_INCIDENT_VECTOR_Y,
    SK_INCIDENT_VECTOR_Z,
    SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X,
    SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y,
    SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z,
    SK_FACE0_OPERATOR_M00,
    SK_FACE0_OPERATOR_M01,
    SK_FACE0_OPERATOR_M10,
    SK_FACE0_OPERATOR_M11,
    SK_FACE1_OPERATOR_M00,
    SK_FACE1_OPERATOR_M01,
    SK_FACE1_OPERATOR_M10,
    SK_FACE1_OPERATOR_M11,
    SK_FACE0_ETA_R,
    SK_FACE0_SIGMA,
    SK_FACE0_GAIN,
    SK_FACE0_USE_FRESNEL,
    SK_FACE1_ETA_R,
    SK_FACE1_SIGMA,
    SK_FACE1_GAIN,
    SK_FACE1_USE_FRESNEL,
    SK_PATH_LENGTH_PREFIX,
)

_PATH_EXPORT_FIELD_EVAL_BASE_KEYS = (
    SK_EDGE_POS,
    SK_EDGE_DIR,
    SK_N0,
    SK_NN,
    SK_WEDGE_N,
    SK_SOURCE_POS,
    SK_EDGE_LINE_MIN,
    SK_EDGE_LINE_MAX,
    SK_INCIDENT_JONES_U,
    SK_INCIDENT_JONES_V,
    SK_INCIDENT_DERIVATIVE_JONES_U,
    SK_INCIDENT_DERIVATIVE_JONES_V,
    SK_INCIDENT_BASIS_U,
    SK_INCIDENT_BASIS_V,
    SK_INCIDENT_BASIS_K,
)

_PATH_EXPORT_FIELD_EVAL_FACE_PARAM_KEYS = (
    SK_FACE0_ETA_R,
    SK_FACE0_SIGMA,
    SK_FACE0_GAIN,
    SK_FACE1_ETA_R,
    SK_FACE1_SIGMA,
    SK_FACE1_GAIN,
)

_PATH_EXPORT_FIELD_EVAL_OPERATOR_KEYS = (
    SK_FACE0_OPERATOR_M00,
    SK_FACE0_OPERATOR_M01,
    SK_FACE0_OPERATOR_M10,
    SK_FACE0_OPERATOR_M11,
    SK_FACE1_OPERATOR_M00,
    SK_FACE1_OPERATOR_M01,
    SK_FACE1_OPERATOR_M10,
    SK_FACE1_OPERATOR_M11,
)

_PATH_EXPORT_SUPPORT_KEYS = (
    SK_EDGE_POS,
    SK_EDGE_DIR,
    SK_N0,
    SK_NN,
    SK_WEDGE_N,
    SK_SOURCE_POS,
)

_PATH_EXPORT_REPLAY_KEYS = (
    SK_EDGE_POS,
    SK_PATH_LENGTH_PREFIX,
    SK_FIRST_INTERACTION_POS,
    SK_SOURCE_TYPE_CODE,
    SK_PREFIX_REFLECTION_DEPTH,
    SK_INTERMEDIATE_REFLECTION_DEPTH,
    SK_SUFFIX_REFLECTION_DEPTH,
    SK_ORDER,
)

_PATH_EXPORT_REDUCED_KEYS = tuple(
    dict.fromkeys(_PATH_EXPORT_EVAL_KEYS + _PATH_EXPORT_REPLAY_KEYS)
)


def path_export_state_layout(state_arrays: dict | None) -> str | None:
    if state_arrays is None:
        return None
    return state_arrays.get(_PATH_EXPORT_STATE_LAYOUT_KEY)


def is_path_export_reduced_state_arrays(state_arrays: dict | None) -> bool:
    return path_export_state_layout(state_arrays) == PATH_EXPORT_REDUCED_STATE_LAYOUT


def _path_export_lineage_metadata(state_arrays: dict | None):
    if state_arrays is None:
        return None
    return state_arrays.get(
        _PATH_EXPORT_LINEAGE_METADATA_KEY,
        _state_metadata(state_arrays),
    )


def _attach_path_export_lineage_metadata(state_arrays: dict, metadata, history_size: int) -> dict:
    state_arrays[STATE_HISTORY_SIZE_METADATA_KEY] = int(history_size)
    if metadata is None:
        state_arrays.pop(_PATH_EXPORT_LINEAGE_METADATA_KEY, None)
    else:
        state_arrays[_PATH_EXPORT_LINEAGE_METADATA_KEY] = metadata
    return state_arrays


def _gather_path_export_lineage_metadata(state_arrays: dict, indices):
    metadata = _path_export_lineage_metadata(state_arrays)
    if metadata is None:
        return None
    gathered = {}
    if _META_PARENT_STATE_ID in metadata:
        gathered[_META_PARENT_STATE_ID] = dr.gather(
            wt.Int32,
            metadata[_META_PARENT_STATE_ID],
            indices,
        )
    if _META_LAST_EDGE_IDX in metadata:
        gathered[_META_LAST_EDGE_IDX] = dr.gather(
            wt.Int32,
            metadata[_META_LAST_EDGE_IDX],
            indices,
        )
    if _META_LAST_REFLECTION_DEPTH_DELTA in metadata:
        gathered[_META_LAST_REFLECTION_DEPTH_DELTA] = dr.gather(
            wt.UInt32,
            metadata[_META_LAST_REFLECTION_DEPTH_DELTA],
            indices,
        )
    if _META_STATE_ID in metadata:
        gathered[_META_STATE_ID] = dr.gather(
            wt.Int32,
            metadata[_META_STATE_ID],
            indices,
        )
    if _META_LINEAGE_STORE in metadata:
        gathered[_META_LINEAGE_STORE] = metadata[_META_LINEAGE_STORE]
    return gathered


def reduce_state_arrays_for_path_export(state_arrays: dict | None) -> dict | None:
    if state_arrays is None or is_path_export_reduced_state_arrays(state_arrays):
        return state_arrays
    history_size = _state_history_size(state_arrays)
    reduced = {
        key: state_arrays[key]
        for key in _PATH_EXPORT_REDUCED_KEYS
        if key in state_arrays
    }
    reduced[SK_N_STATES] = int(state_arrays[SK_N_STATES])
    reduced[_PATH_EXPORT_STATE_LAYOUT_KEY] = PATH_EXPORT_REDUCED_STATE_LAYOUT
    _attach_state_metadata(reduced, None, history_size)
    return _attach_path_export_lineage_metadata(
        reduced,
        _state_metadata(state_arrays),
        history_size,
    )


def _gather_path_export_keys(
    state_arrays: dict,
    indices,
    keys,
    *,
    attach_lineage_metadata: bool,
) -> dict:
    gathered = {}
    for key in keys:
        if key not in state_arrays:
            continue
        value = state_arrays[key]
        if isinstance(value, wt.Point3f):
            gathered[key] = dr.gather(wt.Point3f, value, indices)
        elif isinstance(value, wt.Vector3f):
            gathered[key] = dr.gather(wt.Vector3f, value, indices)
        elif isinstance(value, wt.Complex2f):
            gathered[key] = dr.gather(wt.Complex2f, value, indices)
        elif isinstance(value, wt.Float):
            gathered[key] = dr.gather(wt.Float, value, indices)
        elif isinstance(value, wt.Int32):
            gathered[key] = dr.gather(wt.Int32, value, indices)
        elif isinstance(value, wt.UInt32):
            gathered[key] = dr.gather(wt.UInt32, value, indices)
        else:
            gathered[key] = dr.gather(type(value), value, indices)
    gathered[SK_N_STATES] = int(dr.width(indices))
    gathered[_PATH_EXPORT_STATE_LAYOUT_KEY] = PATH_EXPORT_REDUCED_STATE_LAYOUT
    return _attach_state_metadata(
        gathered,
        (
            _gather_path_export_lineage_metadata(state_arrays, indices)
            if attach_lineage_metadata
            else None
        ),
        _state_history_size(state_arrays),
    )


def gather_path_export_eval_state_fields(state_arrays: dict, indices) -> dict:
    if not is_path_export_reduced_state_arrays(state_arrays):
        from ....kernels.trace.packed_state.drjit_impl import gather_state_arrays

        return gather_state_arrays(state_arrays, indices)
    return _gather_path_export_keys(
        state_arrays,
        indices,
        _PATH_EXPORT_EVAL_KEYS,
        attach_lineage_metadata=False,
    )


def gather_path_export_field_state_fields(state_arrays: dict, indices) -> dict:
    if not is_path_export_reduced_state_arrays(state_arrays):
        from ....kernels.trace.packed_state.drjit_impl import (
            gather_field_evaluation_state_fields,
        )

        return gather_field_evaluation_state_fields(state_arrays, indices)
    keys = list(_PATH_EXPORT_FIELD_EVAL_BASE_KEYS)
    if all(key in state_arrays for key in _PATH_EXPORT_FIELD_EVAL_FACE_PARAM_KEYS):
        keys.extend(_PATH_EXPORT_FIELD_EVAL_FACE_PARAM_KEYS)
    else:
        keys.extend(_PATH_EXPORT_FIELD_EVAL_OPERATOR_KEYS)
    return _gather_path_export_keys(
        state_arrays,
        indices,
        tuple(keys),
        attach_lineage_metadata=False,
    )


def gather_path_export_support_state_fields(state_arrays: dict, indices) -> dict:
    if not is_path_export_reduced_state_arrays(state_arrays):
        return gather_path_export_eval_state_fields(state_arrays, indices)
    return _gather_path_export_keys(
        state_arrays,
        indices,
        _PATH_EXPORT_SUPPORT_KEYS,
        attach_lineage_metadata=False,
    )


def gather_path_export_replay_state_fields(state_arrays: dict, indices) -> dict:
    if not is_path_export_reduced_state_arrays(state_arrays):
        from ....kernels.trace.packed_state.drjit_impl import gather_inserted_reflection_state_fields

        return gather_inserted_reflection_state_fields(state_arrays, indices)
    return _gather_path_export_keys(
        state_arrays,
        indices,
        _PATH_EXPORT_REPLAY_KEYS,
        attach_lineage_metadata=True,
    )


__all__ = [
    "PATH_EXPORT_REDUCED_STATE_LAYOUT",
    "gather_path_export_eval_state_fields",
    "gather_path_export_field_state_fields",
    "gather_path_export_replay_state_fields",
    "gather_path_export_support_state_fields",
    "is_path_export_reduced_state_arrays",
    "path_export_state_layout",
    "reduce_state_arrays_for_path_export",
]
