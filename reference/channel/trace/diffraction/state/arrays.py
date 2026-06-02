"""Core state array CRUD operations for diffraction state."""

import drjit as dr
import torch
import witwin as wt

from ....utils import drjit_to_torch_view
from ....utils.drjit_ops import ArrayInit, Concat
from ....utils.polarization import jones_from_vector, path_basis, vector_from_jones
from ..finite_wedge import require_edge_state_line_bounds
from ..constants import (
    STATE_COLD_KEYS,
    STATE_HISTORY_SIZE_METADATA_KEY,
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
    _validate_state_arrays,
)
from ..geometry import _diagonal_face_operator

__all__ = [
    "_canonicalize_transport_state",
    "_concat_state_arrays",
    "_default_face_material",
    "_empty_state_arrays",
    "_finalize_state_lineage",
    "_gather_state_arrays",
    "_make_state_arrays",
    "_materialize_state_history",
    "_state_has_cold_metadata",
    "_state_ids",
    "_state_lineage_store",
    "_state_lineage_first_edge_idx",
    "_subset_state_arrays",
    "_take_state_arrays",
    "_torch_state_key",
]

_STATE_LAYOUT_VERSION_KEY = "__state_layout_version__"
_STATE_LAYOUT_VERSION = "parent_link_split_v1"
_STATE_METADATA_KEY = "__state_metadata__"
_META_PARENT_STATE_ID = "parent_state_id"
_META_LAST_EDGE_IDX = "last_edge_idx"
_META_LAST_REFLECTION_DEPTH_DELTA = "last_reflection_depth_delta"
_META_STATE_ID = "state_id"
_META_LINEAGE_STORE = "lineage_store"


def _torch_state_key(state_arrays, key, *, detach=True):
    value = drjit_to_torch_view(state_arrays[key], detach=detach)
    if value.dtype == torch.uint32:
        return value.to(dtype=torch.int32)
    return value


def _zero_float(width: int):
    return dr.zeros(wt.Float, width)


def _zero_point(width: int):
    zeros = _zero_float(width)
    return wt.Point3f(zeros, zeros, zeros)


def _zero_vector(width: int):
    zeros = _zero_float(width)
    return wt.Vector3f(zeros, zeros, zeros)


def _zero_complex(width: int):
    zeros = _zero_float(width)
    return wt.Complex2f(zeros, zeros)


def _empty_lineage_store():
    return {
        _META_PARENT_STATE_ID: dr.full(wt.Int32, -1, 0),
        _META_LAST_EDGE_IDX: dr.full(wt.Int32, -1, 0),
        _META_LAST_REFLECTION_DEPTH_DELTA: dr.zeros(wt.UInt32, 0),
        SK_N_STATES: 0,
    }


def _state_metadata(state_arrays):
    return None if state_arrays is None else state_arrays.get(_STATE_METADATA_KEY)


def _state_uses_parent_link_lineage(state_arrays) -> bool:
    return (
        state_arrays is not None
        and state_arrays.get(_STATE_LAYOUT_VERSION_KEY) == _STATE_LAYOUT_VERSION
    )


def _state_has_cold_metadata(state_arrays) -> bool:
    if state_arrays is None:
        return False
    if any(key in state_arrays for key in STATE_COLD_KEYS):
        return True
    return _state_metadata(state_arrays) is not None


def _state_ids(state_arrays):
    metadata = _state_metadata(state_arrays)
    if metadata is None:
        return None
    state_id = metadata.get(_META_STATE_ID)
    if state_id is None:
        return None
    if bool(dr.all(wt.Int32(state_id) < 0)):
        return None
    return state_id


def _state_lineage_store(state_arrays):
    metadata = _state_metadata(state_arrays)
    if metadata is None:
        return None
    return metadata.get(_META_LINEAGE_STORE)


def _attach_state_metadata(state_arrays, metadata, history_size: int):
    state_arrays[STATE_HISTORY_SIZE_METADATA_KEY] = int(history_size)
    state_arrays[_STATE_LAYOUT_VERSION_KEY] = _STATE_LAYOUT_VERSION
    if metadata is None:
        state_arrays.pop(_STATE_METADATA_KEY, None)
    else:
        state_arrays[_STATE_METADATA_KEY] = metadata
    return state_arrays


def _default_face_material(width: int, gain=None):
    if gain is None:
        gain = dr.ones(wt.Float, width)
    return {
        "eta_r": dr.full(wt.Float, 5.0, width),
        "sigma": dr.zeros(wt.Float, width),
        "gain": gain,
        "use_fresnel": dr.full(wt.Bool, True, width),
    }


def _canonicalize_transport_state(
    *,
    edge_pos,
    edge_dir,
    source_pos,
    r0,
    rn,
    incident_vector,
    incident_normal_derivative_vector,
    incident_jones,
    incident_derivative_jones,
    incident_basis,
    face0_operator,
    face1_operator,
):
    if incident_basis is None:
        incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    if incident_jones is None:
        incident_jones = jones_from_vector(incident_vector, incident_basis)
    if incident_derivative_jones is None:
        incident_derivative_jones = jones_from_vector(
            incident_normal_derivative_vector,
            incident_basis,
        )
    incident_vector = vector_from_jones(incident_jones, incident_basis)
    incident_normal_derivative_vector = vector_from_jones(
        incident_derivative_jones,
        incident_basis,
    )
    if face0_operator is None:
        face0_operator = _diagonal_face_operator(r0)
    if face1_operator is None:
        face1_operator = _diagonal_face_operator(rn)
    return {
        "incident_vector": incident_vector,
        "incident_normal_derivative_vector": incident_normal_derivative_vector,
        "incident_jones": incident_jones,
        "incident_derivative_jones": incident_derivative_jones,
        "incident_basis": incident_basis,
        "face0_operator": face0_operator,
        "face1_operator": face1_operator,
    }


def _empty_state_metadata(width: int = 0):
    return {
        _META_PARENT_STATE_ID: dr.full(wt.Int32, -1, width),
        _META_LAST_EDGE_IDX: dr.full(wt.Int32, -1, width),
        _META_LAST_REFLECTION_DEPTH_DELTA: dr.zeros(wt.UInt32, width),
        _META_STATE_ID: dr.full(wt.Int32, -1, width),
    }


def _empty_state_arrays(history_size=0, *, retain_cold_metadata=True):
    width = 0
    state_arrays = {
        SK_EDGE_IDX: dr.zeros(wt.UInt32, width),
        SK_EDGE_POS: _zero_point(width),
        SK_EDGE_DIR: _zero_vector(width),
        SK_N0: _zero_vector(width),
        SK_NN: _zero_vector(width),
        SK_WEDGE_N: _zero_float(width),
        SK_ADJACENT_FACE0: dr.zeros(wt.Int32, width),
        SK_ADJACENT_FACE1: dr.zeros(wt.Int32, width),
        SK_SOURCE_POS: _zero_point(width),
        SK_INCIDENT_FIELD: _zero_complex(width),
        SK_INCIDENT_NORMAL_DERIVATIVE: _zero_complex(width),
        SK_INCIDENT_JONES_U: _zero_complex(width),
        SK_INCIDENT_JONES_V: _zero_complex(width),
        SK_INCIDENT_DERIVATIVE_JONES_U: _zero_complex(width),
        SK_INCIDENT_DERIVATIVE_JONES_V: _zero_complex(width),
        SK_R0: _zero_complex(width),
        SK_RN: _zero_complex(width),
        SK_INCIDENT_BASIS_U: _zero_vector(width),
        SK_INCIDENT_BASIS_V: _zero_vector(width),
        SK_INCIDENT_BASIS_K: _zero_vector(width),
        SK_FACE0_OPERATOR_M00: _zero_complex(width),
        SK_FACE0_OPERATOR_M01: _zero_complex(width),
        SK_FACE0_OPERATOR_M10: _zero_complex(width),
        SK_FACE0_OPERATOR_M11: _zero_complex(width),
        SK_FACE1_OPERATOR_M00: _zero_complex(width),
        SK_FACE1_OPERATOR_M01: _zero_complex(width),
        SK_FACE1_OPERATOR_M10: _zero_complex(width),
        SK_FACE1_OPERATOR_M11: _zero_complex(width),
        SK_FACE0_ETA_R: _zero_float(width),
        SK_FACE0_SIGMA: _zero_float(width),
        SK_FACE0_GAIN: _zero_float(width),
        SK_FACE0_USE_FRESNEL: dr.zeros(wt.Bool, width),
        SK_FACE1_ETA_R: _zero_float(width),
        SK_FACE1_SIGMA: _zero_float(width),
        SK_FACE1_GAIN: _zero_float(width),
        SK_FACE1_USE_FRESNEL: dr.zeros(wt.Bool, width),
        SK_INCIDENT_VECTOR_X: _zero_complex(width),
        SK_INCIDENT_VECTOR_Y: _zero_complex(width),
        SK_INCIDENT_VECTOR_Z: _zero_complex(width),
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X: _zero_complex(width),
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y: _zero_complex(width),
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z: _zero_complex(width),
        SK_PREFIX_REFLECTION_DEPTH: dr.zeros(wt.UInt32, width),
        SK_INTERMEDIATE_REFLECTION_DEPTH: dr.zeros(wt.UInt32, width),
        SK_SUFFIX_REFLECTION_DEPTH: dr.zeros(wt.UInt32, width),
        SK_ORDER: dr.zeros(wt.UInt32, width),
        SK_N_STATES: 0,
    }
    metadata = None
    if retain_cold_metadata:
        state_arrays[SK_PATH_LENGTH_PREFIX] = _zero_float(width)
        state_arrays[SK_FIRST_INTERACTION_POS] = _zero_point(width)
        state_arrays[SK_IS_DIRECT_TX] = dr.zeros(wt.Bool, width)
        state_arrays[SK_SOURCE_TYPE_CODE] = dr.zeros(wt.UInt32, width)
        state_arrays[SK_APPROXIMATION_MODE_CODE] = dr.zeros(wt.UInt32, width)
        metadata = _empty_state_metadata(width)
    _attach_state_metadata(state_arrays, metadata, history_size)
    if __debug__:
        _validate_state_arrays(state_arrays, context="_empty_state_arrays")
    return state_arrays


def _materialize_history_from_lineage_store(state_arrays, metadata, history_size: int):
    path_edge_history = [
        dr.full(wt.Int32, -1, state_arrays[SK_N_STATES]) for _ in range(history_size)
    ]
    path_reflection_depth_history = [
        dr.zeros(wt.UInt32, state_arrays[SK_N_STATES]) for _ in range(history_size)
    ]
    lineage_store = metadata.get(_META_LINEAGE_STORE)
    state_id = metadata.get(_META_STATE_ID)
    if state_id is not None and bool(dr.all(wt.Int32(state_id) < 0)):
        state_id = None
    last_edge = metadata.get(_META_LAST_EDGE_IDX)
    last_reflection = metadata.get(_META_LAST_REFLECTION_DEPTH_DELTA)
    terminal_slot = wt.Int32(state_arrays[SK_ORDER]) - wt.Int32(1)

    if state_id is None:
        if last_edge is None or last_reflection is None:
            return tuple(path_edge_history), tuple(path_reflection_depth_history)
        for slot in range(history_size):
            slot_mask = terminal_slot == wt.Int32(slot)
            path_edge_history[slot] = dr.select(slot_mask, last_edge, path_edge_history[slot])
            path_reflection_depth_history[slot] = dr.select(
                slot_mask,
                last_reflection,
                path_reflection_depth_history[slot],
            )
        if lineage_store is None:
            return tuple(path_edge_history), tuple(path_reflection_depth_history)
        current_state_id = metadata.get(_META_PARENT_STATE_ID)
        if current_state_id is None:
            return tuple(path_edge_history), tuple(path_reflection_depth_history)
        current_state_id = wt.Int32(current_state_id)
        remaining_slot = terminal_slot - wt.Int32(1)
    else:
        if lineage_store is None:
            if last_edge is None or last_reflection is None:
                return tuple(path_edge_history), tuple(path_reflection_depth_history)
            for slot in range(history_size):
                slot_mask = terminal_slot == wt.Int32(slot)
                path_edge_history[slot] = dr.select(slot_mask, last_edge, path_edge_history[slot])
                path_reflection_depth_history[slot] = dr.select(
                    slot_mask,
                    last_reflection,
                    path_reflection_depth_history[slot],
                )
            return tuple(path_edge_history), tuple(path_reflection_depth_history)
        current_state_id = wt.Int32(state_id)
        remaining_slot = terminal_slot

    for _ in range(history_size):
        active = (current_state_id >= 0) & (remaining_slot >= 0)
        if not bool(dr.any(active)):
            break
        safe_state_id = wt.UInt32(dr.select(active, current_state_id, wt.Int32(0)))
        edge_value = dr.gather(
            wt.Int32,
            lineage_store[_META_LAST_EDGE_IDX],
            safe_state_id,
        )
        reflection_value = dr.gather(
            wt.UInt32,
            lineage_store[_META_LAST_REFLECTION_DEPTH_DELTA],
            safe_state_id,
        )
        parent_value = dr.gather(
            wt.Int32,
            lineage_store[_META_PARENT_STATE_ID],
            safe_state_id,
        )
        for slot in range(history_size):
            slot_mask = active & (remaining_slot == wt.Int32(slot))
            path_edge_history[slot] = dr.select(slot_mask, edge_value, path_edge_history[slot])
            path_reflection_depth_history[slot] = dr.select(
                slot_mask,
                reflection_value,
                path_reflection_depth_history[slot],
            )
        current_state_id = dr.select(active, parent_value, wt.Int32(-1))
        remaining_slot = remaining_slot - dr.select(active, wt.Int32(1), wt.Int32(0))
    return tuple(path_edge_history), tuple(path_reflection_depth_history)


def _materialize_state_history(state_arrays):
    history_size = _state_history_size(state_arrays)
    if history_size <= 0 or state_arrays is None or state_arrays[SK_N_STATES] == 0:
        return tuple(), tuple()
    metadata = _state_metadata(state_arrays)
    if metadata is None:
        return tuple(), tuple()
    return _materialize_history_from_lineage_store(state_arrays, metadata, history_size)


def _state_lineage_first_edge_idx(state_arrays):
    history_size = _state_history_size(state_arrays)
    if history_size <= 0:
        return wt.Int32(state_arrays[SK_EDGE_IDX])
    history, _ = _materialize_state_history(state_arrays)
    if len(history) == 0:
        return wt.Int32(state_arrays[SK_EDGE_IDX])
    return history[0]


def _gather_state_metadata(state_arrays, indices):
    metadata = _state_metadata(state_arrays)
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


def _concat_state_metadata(state_arrays_list):
    non_empty = [
        state_arrays
        for state_arrays in state_arrays_list
        if state_arrays is not None and int(state_arrays[SK_N_STATES]) >= 0
    ]
    if len(non_empty) == 0:
        return None
    metadata_list = [_state_metadata(state_arrays) for state_arrays in non_empty]
    if not any(metadata is not None for metadata in metadata_list):
        return None

    def _concat_optional_int(key, fill_value):
        parts = []
        for state_arrays, metadata in zip(non_empty, metadata_list):
            width = int(state_arrays[SK_N_STATES])
            if metadata is not None and key in metadata:
                parts.append(metadata[key])
            else:
                parts.append(dr.full(wt.Int32, fill_value, width))
        return Concat.arrays(wt.Int32, parts)

    def _concat_optional_uint(key):
        parts = []
        for state_arrays, metadata in zip(non_empty, metadata_list):
            width = int(state_arrays[SK_N_STATES])
            if metadata is not None and key in metadata:
                parts.append(metadata[key])
            else:
                parts.append(dr.zeros(wt.UInt32, width))
        return Concat.arrays(wt.UInt32, parts)

    merged = {
        _META_PARENT_STATE_ID: _concat_optional_int(_META_PARENT_STATE_ID, -1),
        _META_LAST_EDGE_IDX: _concat_optional_int(_META_LAST_EDGE_IDX, -1),
        _META_LAST_REFLECTION_DEPTH_DELTA: _concat_optional_uint(
            _META_LAST_REFLECTION_DEPTH_DELTA
        ),
    }

    shared_store = None
    stores = [
        metadata.get(_META_LINEAGE_STORE)
        for metadata in metadata_list
        if metadata is not None and metadata.get(_META_LINEAGE_STORE) is not None
    ]
    if stores and all(store is stores[0] for store in stores):
        shared_store = stores[0]
    elif len(stores) > 0:
        raise RuntimeError(
            "Concatenating finalized diffraction state arrays from different lineage stores is not supported."
        )

    if shared_store is not None:
        merged[_META_LINEAGE_STORE] = shared_store
        if all(metadata is not None and _META_STATE_ID in metadata for metadata in metadata_list):
            merged[_META_STATE_ID] = Concat.arrays(
                wt.Int32,
                [metadata[_META_STATE_ID] for metadata in metadata_list],
            )
    return merged


def _finalize_state_lineage(state_arrays, lineage_store=None, next_state_id=0):
    metadata = _state_metadata(state_arrays)
    if metadata is None or state_arrays[SK_N_STATES] == 0:
        return state_arrays, lineage_store, int(next_state_id)

    n_states = int(state_arrays[SK_N_STATES])
    if lineage_store is None:
        lineage_store = _empty_lineage_store()

    parent_state_id = metadata.get(
        _META_PARENT_STATE_ID,
        dr.full(wt.Int32, -1, n_states),
    )
    last_edge_idx = metadata.get(_META_LAST_EDGE_IDX, wt.Int32(state_arrays[SK_EDGE_IDX]))
    last_reflection_depth_delta = metadata.get(
        _META_LAST_REFLECTION_DEPTH_DELTA,
        dr.zeros(wt.UInt32, n_states),
    )

    state_id = dr.arange(wt.Int32, n_states) + wt.Int32(int(next_state_id))
    lineage_store[_META_PARENT_STATE_ID] = Concat.arrays(
        wt.Int32,
        [lineage_store[_META_PARENT_STATE_ID], parent_state_id],
    )
    lineage_store[_META_LAST_EDGE_IDX] = Concat.arrays(
        wt.Int32,
        [lineage_store[_META_LAST_EDGE_IDX], last_edge_idx],
    )
    lineage_store[_META_LAST_REFLECTION_DEPTH_DELTA] = Concat.arrays(
        wt.UInt32,
        [
            lineage_store[_META_LAST_REFLECTION_DEPTH_DELTA],
            last_reflection_depth_delta,
        ],
    )
    lineage_store[SK_N_STATES] = int(next_state_id) + n_states

    metadata[_META_STATE_ID] = state_id
    metadata[_META_LINEAGE_STORE] = lineage_store
    _attach_state_metadata(state_arrays, metadata, _state_history_size(state_arrays))
    return state_arrays, lineage_store, int(next_state_id) + n_states


def _make_state_arrays(
    edge_idx,
    edge_pos,
    edge_dir,
    n0,
    nn,
    wedge_n,
    adjacent_face0,
    adjacent_face1,
    source_pos,
    incident_field,
    incident_normal_derivative,
    path_length_prefix=None,
    first_interaction_pos=None,
    edge_line_min=None,
    edge_line_max=None,
    r0=None,
    rn=None,
    incident_vector=None,
    incident_normal_derivative_vector=None,
    is_direct_tx=None,
    incident_jones=None,
    incident_derivative_jones=None,
    incident_basis=None,
    face0_operator=None,
    face1_operator=None,
    face0_material=None,
    face1_material=None,
    order=None,
    source_type_code=None,
    prefix_reflection_depth=None,
    intermediate_reflection_depth=None,
    suffix_reflection_depth=None,
    approximation_mode_code=None,
    lineage_parent_state_id=None,
    lineage_last_edge_idx=None,
    lineage_last_reflection_depth_delta=None,
    lineage_store=None,
    retain_cold_metadata=True,
):
    n_states = dr.width(edge_idx)
    if order is None:
        order = dr.zeros(wt.UInt32, n_states)
    history_size = 0 if n_states == 0 else int(dr.max(order)[0])
    if n_states == 0:
        return _empty_state_arrays(
            history_size=history_size,
            retain_cold_metadata=retain_cold_metadata,
        )
    edge_line_min, edge_line_max = require_edge_state_line_bounds(
        {
            "edge_line_min": edge_line_min,
            "edge_line_max": edge_line_max,
        },
        context="_make_state_arrays",
    )

    if prefix_reflection_depth is None:
        prefix_reflection_depth = dr.zeros(wt.UInt32, n_states)
    if intermediate_reflection_depth is None:
        intermediate_reflection_depth = dr.zeros(wt.UInt32, n_states)
    if suffix_reflection_depth is None:
        suffix_reflection_depth = dr.zeros(wt.UInt32, n_states)
    if r0 is None:
        r0 = ArrayInit.complex_zero(n_states)
    if rn is None:
        rn = ArrayInit.complex_zero(n_states)
    canonical = _canonicalize_transport_state(
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        source_pos=source_pos,
        r0=r0,
        rn=rn,
        incident_vector=incident_vector,
        incident_normal_derivative_vector=incident_normal_derivative_vector,
        incident_jones=incident_jones,
        incident_derivative_jones=incident_derivative_jones,
        incident_basis=incident_basis,
        face0_operator=face0_operator,
        face1_operator=face1_operator,
    )
    incident_vector = canonical["incident_vector"]
    incident_normal_derivative_vector = canonical["incident_normal_derivative_vector"]
    incident_jones = canonical["incident_jones"]
    incident_derivative_jones = canonical["incident_derivative_jones"]
    incident_basis = canonical["incident_basis"]
    face0_operator = canonical["face0_operator"]
    face1_operator = canonical["face1_operator"]
    if face0_material is None:
        face0_material = _default_face_material(n_states)
    if face1_material is None:
        face1_material = _default_face_material(n_states)

    dr.eval(
        edge_idx,
        edge_pos,
        edge_dir,
        n0,
        nn,
        wedge_n,
        edge_line_min,
        edge_line_max,
        adjacent_face0,
        adjacent_face1,
        source_pos,
        incident_field,
        incident_normal_derivative,
        r0,
        rn,
        incident_jones["u"],
        incident_jones["v"],
        incident_derivative_jones["u"],
        incident_derivative_jones["v"],
        incident_basis["u"],
        incident_basis["v"],
        incident_basis["k"],
        face0_operator["m00"],
        face0_operator["m01"],
        face0_operator["m10"],
        face0_operator["m11"],
        face1_operator["m00"],
        face1_operator["m01"],
        face1_operator["m10"],
        face1_operator["m11"],
        face0_material["eta_r"],
        face0_material["sigma"],
        face0_material["gain"],
        face0_material["use_fresnel"],
        face1_material["eta_r"],
        face1_material["sigma"],
        face1_material["gain"],
        face1_material["use_fresnel"],
        incident_vector["x"],
        incident_vector["y"],
        incident_vector["z"],
        incident_normal_derivative_vector["x"],
        incident_normal_derivative_vector["y"],
        incident_normal_derivative_vector["z"],
        prefix_reflection_depth,
        intermediate_reflection_depth,
        suffix_reflection_depth,
        order,
    )
    state_arrays = {
        SK_EDGE_IDX: edge_idx,
        SK_EDGE_POS: edge_pos,
        SK_EDGE_DIR: edge_dir,
        SK_N0: n0,
        SK_NN: nn,
        SK_WEDGE_N: wedge_n,
        SK_ADJACENT_FACE0: adjacent_face0,
        SK_ADJACENT_FACE1: adjacent_face1,
        SK_SOURCE_POS: source_pos,
        SK_INCIDENT_FIELD: incident_field,
        SK_INCIDENT_NORMAL_DERIVATIVE: incident_normal_derivative,
        SK_INCIDENT_JONES_U: incident_jones["u"],
        SK_INCIDENT_JONES_V: incident_jones["v"],
        SK_INCIDENT_DERIVATIVE_JONES_U: incident_derivative_jones["u"],
        SK_INCIDENT_DERIVATIVE_JONES_V: incident_derivative_jones["v"],
        SK_R0: r0,
        SK_RN: rn,
        SK_INCIDENT_BASIS_U: incident_basis["u"],
        SK_INCIDENT_BASIS_V: incident_basis["v"],
        SK_INCIDENT_BASIS_K: incident_basis["k"],
        SK_FACE0_OPERATOR_M00: face0_operator["m00"],
        SK_FACE0_OPERATOR_M01: face0_operator["m01"],
        SK_FACE0_OPERATOR_M10: face0_operator["m10"],
        SK_FACE0_OPERATOR_M11: face0_operator["m11"],
        SK_FACE1_OPERATOR_M00: face1_operator["m00"],
        SK_FACE1_OPERATOR_M01: face1_operator["m01"],
        SK_FACE1_OPERATOR_M10: face1_operator["m10"],
        SK_FACE1_OPERATOR_M11: face1_operator["m11"],
        SK_FACE0_ETA_R: face0_material["eta_r"],
        SK_FACE0_SIGMA: face0_material["sigma"],
        SK_FACE0_GAIN: face0_material["gain"],
        SK_FACE0_USE_FRESNEL: face0_material["use_fresnel"],
        SK_FACE1_ETA_R: face1_material["eta_r"],
        SK_FACE1_SIGMA: face1_material["sigma"],
        SK_FACE1_GAIN: face1_material["gain"],
        SK_FACE1_USE_FRESNEL: face1_material["use_fresnel"],
        SK_INCIDENT_VECTOR_X: incident_vector["x"],
        SK_INCIDENT_VECTOR_Y: incident_vector["y"],
        SK_INCIDENT_VECTOR_Z: incident_vector["z"],
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X: incident_normal_derivative_vector["x"],
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y: incident_normal_derivative_vector["y"],
        SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z: incident_normal_derivative_vector["z"],
        SK_PREFIX_REFLECTION_DEPTH: prefix_reflection_depth,
        SK_INTERMEDIATE_REFLECTION_DEPTH: intermediate_reflection_depth,
        SK_SUFFIX_REFLECTION_DEPTH: suffix_reflection_depth,
        SK_ORDER: order,
        SK_N_STATES: n_states,
        SK_EDGE_LINE_MIN: edge_line_min,
        SK_EDGE_LINE_MAX: edge_line_max,
    }
    metadata = None
    if retain_cold_metadata:
        if source_type_code is None:
            source_type_code = dr.zeros(wt.UInt32, n_states)
        if approximation_mode_code is None:
            approximation_mode_code = dr.zeros(wt.UInt32, n_states)
        if is_direct_tx is None:
            is_direct_tx = dr.zeros(wt.Bool, n_states)
        if path_length_prefix is None:
            path_length_prefix = dr.norm(edge_pos - source_pos)
        if first_interaction_pos is None:
            first_interaction_pos = edge_pos
        dr.eval(
            path_length_prefix,
            first_interaction_pos,
            is_direct_tx,
            source_type_code,
            approximation_mode_code,
        )
        state_arrays[SK_PATH_LENGTH_PREFIX] = path_length_prefix
        state_arrays[SK_FIRST_INTERACTION_POS] = first_interaction_pos
        state_arrays[SK_IS_DIRECT_TX] = is_direct_tx
        state_arrays[SK_SOURCE_TYPE_CODE] = source_type_code
        state_arrays[SK_APPROXIMATION_MODE_CODE] = approximation_mode_code
        metadata = _empty_state_metadata(n_states)
        metadata.pop(_META_STATE_ID, None)
        metadata[_META_PARENT_STATE_ID] = (
            metadata[_META_PARENT_STATE_ID]
            if lineage_parent_state_id is None
            else lineage_parent_state_id
        )
        metadata[_META_LAST_EDGE_IDX] = (
            wt.Int32(edge_idx)
            if lineage_last_edge_idx is None
            else lineage_last_edge_idx
        )
        metadata[_META_LAST_REFLECTION_DEPTH_DELTA] = (
            dr.zeros(wt.UInt32, n_states)
            if lineage_last_reflection_depth_delta is None
            else lineage_last_reflection_depth_delta
        )
        if lineage_store is not None:
            metadata[_META_LINEAGE_STORE] = lineage_store
    _attach_state_metadata(state_arrays, metadata, history_size)
    if __debug__:
        _validate_state_arrays(state_arrays, context="_make_state_arrays")
    return state_arrays


def _gather_state_arrays(state_arrays, indices):
    from witwin.channel.kernels.trace.packed_state import gather_state_arrays

    return gather_state_arrays(state_arrays, indices)


def _concat_state_arrays(state_arrays_list):
    from witwin.channel.kernels.trace.packed_state import concat_state_arrays

    return concat_state_arrays(state_arrays_list)


def _subset_state_arrays(state_arrays, mask):
    from witwin.channel.kernels.trace.packed_state import subset_state_arrays

    return subset_state_arrays(state_arrays, mask)


def _take_state_arrays(state_arrays, keep_idx):
    history_size = _state_history_size(state_arrays)
    retain_cold_metadata = _state_has_cold_metadata(state_arrays)
    if state_arrays is None or state_arrays[SK_N_STATES] == 0 or dr.width(keep_idx) == 0:
        return _empty_state_arrays(
            history_size=history_size,
            retain_cold_metadata=retain_cold_metadata,
        )
    gathered = _gather_state_arrays(state_arrays, keep_idx)
    gathered[SK_N_STATES] = dr.width(keep_idx)
    return gathered
