"""Diffraction state schema (SK_* keys, enums, Geo namespace) and State CRUD."""

import drjit as dr

from witwin.channel.deterministic import types as wt

from witwin.channel.core.numerics.constants import DIFFRACTION_MIN_DISTANCE, SPEED_OF_LIGHT
from witwin.channel.core.numerics.arrays import complex_zero, concat_arrays
from witwin.channel.core.physics.polarization import jones_from_vector, path_basis, vector_from_jones
from .math import GeometrySupport
SOURCE_TYPE_DIRECT_TX = 0
SOURCE_TYPE_REFLECTION_PREFIX = 1
APPROX_MODE_DIRECT_FIRST_ORDER = 0
APPROX_MODE_RECURSIVE_DIFFRACTION = 1
APPROX_MODE_SAMPLED_REFLECTION_PREFIX = 2
APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN = 3
APPROX_MODE_SAMPLED_INSERTED_REFLECTION = 4
APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN = 5
OWNERSHIP_DIRECT_DIFFRACTION = 0
OWNERSHIP_MIXED_DIFFRACTION = 1
CARTESIAN_PAIR_CHUNK_BUDGET = 1 << 25
STATE_HISTORY_SIZE_KEY = '__state_history_size__'
SK_EDGE_IDX = 'edge_idx'
SK_EDGE_POS = 'edge_pos'
SK_EDGE_DIR = 'edge_dir'
SK_EDGE_LINE_MIN = 'edge_line_min'
SK_EDGE_LINE_MAX = 'edge_line_max'
SK_N0 = 'n0'
SK_NN = 'n_face_n'
SK_WEDGE_N = 'wedge_n'
SK_ADJACENT_FACE0 = 'adjacent_face0'
SK_ADJACENT_FACE1 = 'adjacent_face1'
SK_SOURCE_POS = 'source_pos'
SK_PATH_LENGTH_PREFIX = 'path_length_prefix'
SK_FIRST_INTERACTION_POS = 'first_interaction_pos'
SK_INCIDENT_FIELD = 'incident_field'
SK_INCIDENT_NORMAL_DERIVATIVE = 'incident_normal_derivative'
SK_INCIDENT_JONES_U = 'incident_jones_u'
SK_INCIDENT_JONES_V = 'incident_jones_v'
SK_INCIDENT_DERIVATIVE_JONES_U = 'incident_derivative_jones_u'
SK_INCIDENT_DERIVATIVE_JONES_V = 'incident_derivative_jones_v'
SK_R0 = 'r_face0'
SK_RN = 'r_face_n'
SK_INCIDENT_BASIS_U = 'incident_basis_u'
SK_INCIDENT_BASIS_V = 'incident_basis_v'
SK_INCIDENT_BASIS_K = 'incident_basis_k'
SK_FACE0_OPERATOR_M00 = 'face0_operator_m00'
SK_FACE0_OPERATOR_M01 = 'face0_operator_m01'
SK_FACE0_OPERATOR_M10 = 'face0_operator_m10'
SK_FACE0_OPERATOR_M11 = 'face0_operator_m11'
SK_FACE1_OPERATOR_M00 = 'face1_operator_m00'
SK_FACE1_OPERATOR_M01 = 'face1_operator_m01'
SK_FACE1_OPERATOR_M10 = 'face1_operator_m10'
SK_FACE1_OPERATOR_M11 = 'face1_operator_m11'
SK_FACE0_ETA_R = 'face0_eta_r'
SK_FACE0_MU_R = 'face0_mu_r'
SK_FACE0_SIGMA = 'face0_sigma'
SK_FACE0_GAIN = 'face0_gain'
SK_FACE0_USE_FRESNEL = 'face0_use_fresnel'
SK_FACE1_ETA_R = 'face1_eta_r'
SK_FACE1_MU_R = 'face1_mu_r'
SK_FACE1_SIGMA = 'face1_sigma'
SK_FACE1_GAIN = 'face1_gain'
SK_FACE1_USE_FRESNEL = 'face1_use_fresnel'
SK_INCIDENT_VECTOR_X = 'incident_vector_x'
SK_INCIDENT_VECTOR_Y = 'incident_vector_y'
SK_INCIDENT_VECTOR_Z = 'incident_vector_z'
SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X = 'incident_normal_derivative_vector_x'
SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y = 'incident_normal_derivative_vector_y'
SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z = 'incident_normal_derivative_vector_z'
SK_IS_DIRECT_TX = 'is_direct_tx'
SK_SOURCE_TYPE_CODE = 'source_type_code'
SK_PREFIX_REFLECTION_DEPTH = 'prefix_reflection_depth'
SK_INTERMEDIATE_REFLECTION_DEPTH = 'intermediate_reflection_depth'
SK_SUFFIX_REFLECTION_DEPTH = 'suffix_reflection_depth'
SK_APPROXIMATION_MODE_CODE = 'approximation_mode_code'
SK_ORDER = 'order'
SK_N_STATES = 'n_states'
STATE_STATIC_KEYS = (SK_EDGE_IDX, SK_EDGE_POS, SK_EDGE_DIR, SK_N0, SK_NN, SK_WEDGE_N, SK_ADJACENT_FACE0, SK_ADJACENT_FACE1, SK_SOURCE_POS, SK_PATH_LENGTH_PREFIX, SK_FIRST_INTERACTION_POS, SK_INCIDENT_FIELD, SK_INCIDENT_NORMAL_DERIVATIVE, SK_INCIDENT_JONES_U, SK_INCIDENT_JONES_V, SK_INCIDENT_DERIVATIVE_JONES_U, SK_INCIDENT_DERIVATIVE_JONES_V, SK_R0, SK_RN, SK_INCIDENT_BASIS_U, SK_INCIDENT_BASIS_V, SK_INCIDENT_BASIS_K, SK_FACE0_OPERATOR_M00, SK_FACE0_OPERATOR_M01, SK_FACE0_OPERATOR_M10, SK_FACE0_OPERATOR_M11, SK_FACE1_OPERATOR_M00, SK_FACE1_OPERATOR_M01, SK_FACE1_OPERATOR_M10, SK_FACE1_OPERATOR_M11, SK_FACE0_ETA_R, SK_FACE0_MU_R, SK_FACE0_SIGMA, SK_FACE0_GAIN, SK_FACE0_USE_FRESNEL, SK_FACE1_ETA_R, SK_FACE1_MU_R, SK_FACE1_SIGMA, SK_FACE1_GAIN, SK_FACE1_USE_FRESNEL, SK_INCIDENT_VECTOR_X, SK_INCIDENT_VECTOR_Y, SK_INCIDENT_VECTOR_Z, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z, SK_IS_DIRECT_TX, SK_SOURCE_TYPE_CODE, SK_PREFIX_REFLECTION_DEPTH, SK_INTERMEDIATE_REFLECTION_DEPTH, SK_SUFFIX_REFLECTION_DEPTH, SK_APPROXIMATION_MODE_CODE, SK_ORDER)
STATE_COLD_KEYS = (SK_PATH_LENGTH_PREFIX, SK_FIRST_INTERACTION_POS, SK_IS_DIRECT_TX, SK_SOURCE_TYPE_CODE, SK_APPROXIMATION_MODE_CODE)
STATE_HOT_KEYS = tuple((key for key in STATE_STATIC_KEYS if key not in STATE_COLD_KEYS))
class Geo(GeometrySupport):

    def ownership_code(prefix_reflection_depth, intermediate_reflection_depth, suffix_reflection_depth):
        has_reflection = (prefix_reflection_depth > wt.UInt32(0)) | (intermediate_reflection_depth > wt.UInt32(0)) | (suffix_reflection_depth > wt.UInt32(0))
        return dr.select(has_reflection, wt.UInt32(OWNERSHIP_MIXED_DIFFRACTION), wt.UInt32(OWNERSHIP_DIRECT_DIFFRACTION))

    def history_size(state_arrays):
        if state_arrays is None:
            return 0
        return int(state_arrays.get(STATE_HISTORY_SIZE_KEY, 0))

    def cart_chunk(left_size, right_size):
        """Return a left-dimension chunk length that caps left*right temporary pairs."""
        left = max(0, int(left_size))
        right = max(0, int(right_size))
        if left == 0 or right == 0:
            return 0
        pair_budget = max(1, int(CARTESIAN_PAIR_CHUNK_BUDGET))
        return max(1, min(left, pair_budget // right if right > 0 else pair_budget))



_STATE_LAYOUT_VERSION_KEY = '__state_layout_version__'
_STATE_LAYOUT_VERSION = 'parent_link_split_v1'
_STATE_LINEAGE_KEY = '__state_lineage__'
_META_PARENT_STATE_ID = 'parent_state_id'
_META_LAST_EDGE_IDX = 'last_edge_idx'
_META_LAST_REFLECTION_DEPTH_DELTA = 'last_reflection_depth_delta'
_META_STATE_ID = 'state_id'
_META_LINEAGE_STORE = 'lineage_store'
_PATH_EXPORT_STATE_LAYOUT_KEY = '__path_export_state_layout__'
_PATH_EXPORT_LINEAGE_KEY = '__path_export_lineage__'
PATH_EXPORT_REDUCED_STATE_LAYOUT = 'reduced_v2'
_PATH_EXPORT_EVAL_KEYS = (SK_EDGE_POS, SK_EDGE_DIR, SK_N0, SK_NN, SK_WEDGE_N, SK_ADJACENT_FACE0, SK_ADJACENT_FACE1, SK_EDGE_LINE_MIN, SK_EDGE_LINE_MAX, SK_SOURCE_POS, SK_INCIDENT_FIELD, SK_INCIDENT_NORMAL_DERIVATIVE, SK_INCIDENT_JONES_U, SK_INCIDENT_JONES_V, SK_INCIDENT_DERIVATIVE_JONES_U, SK_INCIDENT_DERIVATIVE_JONES_V, SK_INCIDENT_BASIS_U, SK_INCIDENT_BASIS_V, SK_INCIDENT_BASIS_K, SK_R0, SK_RN, SK_INCIDENT_VECTOR_X, SK_INCIDENT_VECTOR_Y, SK_INCIDENT_VECTOR_Z, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z, SK_FACE0_OPERATOR_M00, SK_FACE0_OPERATOR_M01, SK_FACE0_OPERATOR_M10, SK_FACE0_OPERATOR_M11, SK_FACE1_OPERATOR_M00, SK_FACE1_OPERATOR_M01, SK_FACE1_OPERATOR_M10, SK_FACE1_OPERATOR_M11, SK_FACE0_ETA_R, SK_FACE0_MU_R, SK_FACE0_SIGMA, SK_FACE0_GAIN, SK_FACE0_USE_FRESNEL, SK_FACE1_ETA_R, SK_FACE1_MU_R, SK_FACE1_SIGMA, SK_FACE1_GAIN, SK_FACE1_USE_FRESNEL, SK_PATH_LENGTH_PREFIX)
_PATH_EXPORT_REPLAY_KEYS = (SK_EDGE_POS, SK_PATH_LENGTH_PREFIX, SK_FIRST_INTERACTION_POS, SK_SOURCE_TYPE_CODE, SK_PREFIX_REFLECTION_DEPTH, SK_INTERMEDIATE_REFLECTION_DEPTH, SK_SUFFIX_REFLECTION_DEPTH, SK_ORDER)
_PATH_EXPORT_REDUCED_KEYS = tuple(dict.fromkeys(_PATH_EXPORT_EVAL_KEYS + _PATH_EXPORT_REPLAY_KEYS))

class State:

    def has_lineage_state(state_arrays) -> bool:
        if state_arrays is None:
            return False
        if any((key in state_arrays for key in STATE_COLD_KEYS)):
            return True
        return state_arrays.get(_STATE_LINEAGE_KEY) is not None

    def ids(state_arrays):
        lineage = None if state_arrays is None else state_arrays.get(_STATE_LINEAGE_KEY)
        if lineage is None:
            return None
        state_id = lineage.get(_META_STATE_ID)
        if state_id is None:
            return None
        if bool(dr.all(wt.Int32(state_id) < 0)):
            return None
        return state_id

    def lineage_store(state_arrays):
        lineage = None if state_arrays is None else state_arrays.get(_STATE_LINEAGE_KEY)
        if lineage is None:
            return None
        return lineage.get(_META_LINEAGE_STORE)

    def attach_lineage(state_arrays, lineage, history_size: int):
        state_arrays[STATE_HISTORY_SIZE_KEY] = int(history_size)
        state_arrays[_STATE_LAYOUT_VERSION_KEY] = _STATE_LAYOUT_VERSION
        if lineage is None:
            state_arrays.pop(_STATE_LINEAGE_KEY, None)
        else:
            state_arrays[_STATE_LINEAGE_KEY] = lineage
        return state_arrays

    def default_face_material(width: int, gain=None):
        if gain is None:
            gain = dr.ones(wt.Float, width)
        return {'eta_r': dr.full(wt.Float, 5.0, width), 'mu_r': dr.ones(wt.Float, width), 'sigma': dr.zeros(wt.Float, width), 'gain': gain, 'use_fresnel': dr.full(wt.Bool, True, width)}

    def canonicalize_transport(*, edge_pos, edge_dir, source_pos, r0, rn, incident_vector, incident_normal_derivative_vector, incident_jones, incident_derivative_jones, incident_basis, face0_operator, face1_operator):
        if incident_basis is None:
            incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
        if incident_jones is None:
            incident_jones = jones_from_vector(incident_vector, incident_basis)
        if incident_derivative_jones is None:
            incident_derivative_jones = jones_from_vector(incident_normal_derivative_vector, incident_basis)
        incident_vector = vector_from_jones(incident_jones, incident_basis)
        incident_normal_derivative_vector = vector_from_jones(incident_derivative_jones, incident_basis)
        if face0_operator is None:
            face0_operator = Geo.diagonal_face_operator(r0)
        if face1_operator is None:
            face1_operator = Geo.diagonal_face_operator(rn)
        return {'incident_vector': incident_vector, 'incident_normal_derivative_vector': incident_normal_derivative_vector, 'incident_jones': incident_jones, 'incident_derivative_jones': incident_derivative_jones, 'incident_basis': incident_basis, 'face0_operator': face0_operator, 'face1_operator': face1_operator}

    def empty_lineage(width: int=0):
        return {_META_PARENT_STATE_ID: dr.full(wt.Int32, -1, width), _META_LAST_EDGE_IDX: dr.full(wt.Int32, -1, width), _META_LAST_REFLECTION_DEPTH_DELTA: dr.zeros(wt.UInt32, width), _META_STATE_ID: dr.full(wt.Int32, -1, width)}

    def empty(history_size=0, *, retain_lineage_state=True):
        width = 0
        zero_float = dr.zeros(wt.Float, width)
        zero_point = wt.Point3f(zero_float, zero_float, zero_float)
        zero_vector = wt.Vector3f(zero_float, zero_float, zero_float)
        zero_complex = wt.Complex2f(zero_float, zero_float)
        state_arrays = {SK_EDGE_IDX: dr.zeros(wt.UInt32, width), SK_EDGE_POS: zero_point, SK_EDGE_DIR: zero_vector, SK_N0: zero_vector, SK_NN: zero_vector, SK_WEDGE_N: zero_float, SK_ADJACENT_FACE0: dr.zeros(wt.Int32, width), SK_ADJACENT_FACE1: dr.zeros(wt.Int32, width), SK_SOURCE_POS: zero_point, SK_INCIDENT_FIELD: zero_complex, SK_INCIDENT_NORMAL_DERIVATIVE: zero_complex, SK_INCIDENT_JONES_U: zero_complex, SK_INCIDENT_JONES_V: zero_complex, SK_INCIDENT_DERIVATIVE_JONES_U: zero_complex, SK_INCIDENT_DERIVATIVE_JONES_V: zero_complex, SK_R0: zero_complex, SK_RN: zero_complex, SK_INCIDENT_BASIS_U: zero_vector, SK_INCIDENT_BASIS_V: zero_vector, SK_INCIDENT_BASIS_K: zero_vector, SK_FACE0_OPERATOR_M00: zero_complex, SK_FACE0_OPERATOR_M01: zero_complex, SK_FACE0_OPERATOR_M10: zero_complex, SK_FACE0_OPERATOR_M11: zero_complex, SK_FACE1_OPERATOR_M00: zero_complex, SK_FACE1_OPERATOR_M01: zero_complex, SK_FACE1_OPERATOR_M10: zero_complex, SK_FACE1_OPERATOR_M11: zero_complex, SK_FACE0_ETA_R: zero_float, SK_FACE0_MU_R: zero_float, SK_FACE0_SIGMA: zero_float, SK_FACE0_GAIN: zero_float, SK_FACE0_USE_FRESNEL: dr.zeros(wt.Bool, width), SK_FACE1_ETA_R: zero_float, SK_FACE1_MU_R: zero_float, SK_FACE1_SIGMA: zero_float, SK_FACE1_GAIN: zero_float, SK_FACE1_USE_FRESNEL: dr.zeros(wt.Bool, width), SK_INCIDENT_VECTOR_X: zero_complex, SK_INCIDENT_VECTOR_Y: zero_complex, SK_INCIDENT_VECTOR_Z: zero_complex, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X: zero_complex, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y: zero_complex, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z: zero_complex, SK_PREFIX_REFLECTION_DEPTH: dr.zeros(wt.UInt32, width), SK_INTERMEDIATE_REFLECTION_DEPTH: dr.zeros(wt.UInt32, width), SK_SUFFIX_REFLECTION_DEPTH: dr.zeros(wt.UInt32, width), SK_ORDER: dr.zeros(wt.UInt32, width), SK_N_STATES: 0}
        lineage = None
        if retain_lineage_state:
            state_arrays[SK_PATH_LENGTH_PREFIX] = zero_float
            state_arrays[SK_FIRST_INTERACTION_POS] = zero_point
            state_arrays[SK_IS_DIRECT_TX] = dr.zeros(wt.Bool, width)
            state_arrays[SK_SOURCE_TYPE_CODE] = dr.zeros(wt.UInt32, width)
            state_arrays[SK_APPROXIMATION_MODE_CODE] = dr.zeros(wt.UInt32, width)
            lineage = State.empty_lineage(width)
        State.attach_lineage(state_arrays, lineage, history_size)
        return state_arrays

    def materialize_history_from_store(state_arrays, lineage, history_size: int):
        path_edge_history = [dr.full(wt.Int32, -1, state_arrays[SK_N_STATES]) for _ in range(history_size)]
        path_reflection_depth_history = [dr.zeros(wt.UInt32, state_arrays[SK_N_STATES]) for _ in range(history_size)]
        lineage_store = lineage.get(_META_LINEAGE_STORE)
        state_id = lineage.get(_META_STATE_ID)
        if state_id is not None and bool(dr.all(wt.Int32(state_id) < 0)):
            state_id = None
        last_edge = lineage.get(_META_LAST_EDGE_IDX)
        last_reflection = lineage.get(_META_LAST_REFLECTION_DEPTH_DELTA)
        terminal_slot = wt.Int32(state_arrays[SK_ORDER]) - wt.Int32(1)
        if state_id is None:
            if last_edge is None or last_reflection is None:
                return (tuple(path_edge_history), tuple(path_reflection_depth_history))
            for slot in range(history_size):
                slot_mask = terminal_slot == wt.Int32(slot)
                path_edge_history[slot] = dr.select(slot_mask, last_edge, path_edge_history[slot])
                path_reflection_depth_history[slot] = dr.select(slot_mask, last_reflection, path_reflection_depth_history[slot])
            if lineage_store is None:
                return (tuple(path_edge_history), tuple(path_reflection_depth_history))
            current_state_id = lineage.get(_META_PARENT_STATE_ID)
            if current_state_id is None:
                return (tuple(path_edge_history), tuple(path_reflection_depth_history))
            current_state_id = wt.Int32(current_state_id)
            remaining_slot = terminal_slot - wt.Int32(1)
        else:
            if lineage_store is None:
                if last_edge is None or last_reflection is None:
                    return (tuple(path_edge_history), tuple(path_reflection_depth_history))
                for slot in range(history_size):
                    slot_mask = terminal_slot == wt.Int32(slot)
                    path_edge_history[slot] = dr.select(slot_mask, last_edge, path_edge_history[slot])
                    path_reflection_depth_history[slot] = dr.select(slot_mask, last_reflection, path_reflection_depth_history[slot])
                return (tuple(path_edge_history), tuple(path_reflection_depth_history))
            current_state_id = wt.Int32(state_id)
            remaining_slot = terminal_slot
        for _ in range(history_size):
            active = (current_state_id >= 0) & (remaining_slot >= 0)
            if not bool(dr.any(active)):
                break
            safe_state_id = wt.UInt32(dr.select(active, current_state_id, wt.Int32(0)))
            edge_value = dr.gather(wt.Int32, lineage_store[_META_LAST_EDGE_IDX], safe_state_id)
            reflection_value = dr.gather(wt.UInt32, lineage_store[_META_LAST_REFLECTION_DEPTH_DELTA], safe_state_id)
            parent_value = dr.gather(wt.Int32, lineage_store[_META_PARENT_STATE_ID], safe_state_id)
            for slot in range(history_size):
                slot_mask = active & (remaining_slot == wt.Int32(slot))
                path_edge_history[slot] = dr.select(slot_mask, edge_value, path_edge_history[slot])
                path_reflection_depth_history[slot] = dr.select(slot_mask, reflection_value, path_reflection_depth_history[slot])
            current_state_id = dr.select(active, parent_value, wt.Int32(-1))
            remaining_slot = remaining_slot - dr.select(active, wt.Int32(1), wt.Int32(0))
        return (tuple(path_edge_history), tuple(path_reflection_depth_history))

    def history(state_arrays):
        history_size = Geo.history_size(state_arrays)
        if history_size <= 0 or state_arrays is None or state_arrays[SK_N_STATES] == 0:
            return (tuple(), tuple())
        lineage = state_arrays.get(_STATE_LINEAGE_KEY)
        if lineage is None:
            return (tuple(), tuple())
        return State.materialize_history_from_store(state_arrays, lineage, history_size)

    def gather_lineage(state_arrays, indices):
        lineage = None if state_arrays is None else state_arrays.get(_STATE_LINEAGE_KEY)
        if lineage is None:
            return None
        gathered = {}
        if _META_PARENT_STATE_ID in lineage:
            gathered[_META_PARENT_STATE_ID] = dr.gather(wt.Int32, lineage[_META_PARENT_STATE_ID], indices)
        if _META_LAST_EDGE_IDX in lineage:
            gathered[_META_LAST_EDGE_IDX] = dr.gather(wt.Int32, lineage[_META_LAST_EDGE_IDX], indices)
        if _META_LAST_REFLECTION_DEPTH_DELTA in lineage:
            gathered[_META_LAST_REFLECTION_DEPTH_DELTA] = dr.gather(wt.UInt32, lineage[_META_LAST_REFLECTION_DEPTH_DELTA], indices)
        if _META_STATE_ID in lineage:
            gathered[_META_STATE_ID] = dr.gather(wt.Int32, lineage[_META_STATE_ID], indices)
        if _META_LINEAGE_STORE in lineage:
            gathered[_META_LINEAGE_STORE] = lineage[_META_LINEAGE_STORE]
        return gathered

    def concat_lineage(state_arrays_list):
        non_empty = [state_arrays for state_arrays in state_arrays_list if state_arrays is not None and int(state_arrays[SK_N_STATES]) >= 0]
        if len(non_empty) == 0:
            return None
        lineage_list = [state_arrays.get(_STATE_LINEAGE_KEY) for state_arrays in non_empty]
        if not any((lineage is not None for lineage in lineage_list)):
            return None

        def _concat_optional_int(key, fill_value):
            parts = []
            for state_arrays, lineage in zip(non_empty, lineage_list):
                width = int(state_arrays[SK_N_STATES])
                if lineage is not None and key in lineage:
                    parts.append(lineage[key])
                else:
                    parts.append(dr.full(wt.Int32, fill_value, width))
            return concat_arrays(wt.Int32, parts)

        def _concat_optional_uint(key):
            parts = []
            for state_arrays, lineage in zip(non_empty, lineage_list):
                width = int(state_arrays[SK_N_STATES])
                if lineage is not None and key in lineage:
                    parts.append(lineage[key])
                else:
                    parts.append(dr.zeros(wt.UInt32, width))
            return concat_arrays(wt.UInt32, parts)
        merged = {_META_PARENT_STATE_ID: _concat_optional_int(_META_PARENT_STATE_ID, -1), _META_LAST_EDGE_IDX: _concat_optional_int(_META_LAST_EDGE_IDX, -1), _META_LAST_REFLECTION_DEPTH_DELTA: _concat_optional_uint(_META_LAST_REFLECTION_DEPTH_DELTA)}
        shared_store = None
        stores = [lineage.get(_META_LINEAGE_STORE) for lineage in lineage_list if lineage is not None and lineage.get(_META_LINEAGE_STORE) is not None]
        if stores and all((store is stores[0] for store in stores)):
            shared_store = stores[0]
        elif len(stores) > 0:
            raise RuntimeError('Concatenating finalized diffraction state arrays from different lineage stores is not supported.')
        if shared_store is not None:
            merged[_META_LINEAGE_STORE] = shared_store
            if all((lineage is not None and _META_STATE_ID in lineage for lineage in lineage_list)):
                merged[_META_STATE_ID] = concat_arrays(wt.Int32, [lineage[_META_STATE_ID] for lineage in lineage_list])
        return merged

    def finalize_lineage(state_arrays, lineage_store=None, next_state_id=0):
        lineage = None if state_arrays is None else state_arrays.get(_STATE_LINEAGE_KEY)
        if lineage is None or state_arrays[SK_N_STATES] == 0:
            return (state_arrays, lineage_store, int(next_state_id))
        n_states = int(state_arrays[SK_N_STATES])
        if lineage_store is None:
            lineage_store = {_META_PARENT_STATE_ID: dr.full(wt.Int32, -1, 0), _META_LAST_EDGE_IDX: dr.full(wt.Int32, -1, 0), _META_LAST_REFLECTION_DEPTH_DELTA: dr.zeros(wt.UInt32, 0), SK_N_STATES: 0}
        parent_state_id = lineage.get(_META_PARENT_STATE_ID, dr.full(wt.Int32, -1, n_states))
        last_edge_idx = lineage.get(_META_LAST_EDGE_IDX, wt.Int32(state_arrays[SK_EDGE_IDX]))
        last_reflection_depth_delta = lineage.get(_META_LAST_REFLECTION_DEPTH_DELTA, dr.zeros(wt.UInt32, n_states))
        state_id = dr.arange(wt.Int32, n_states) + wt.Int32(int(next_state_id))
        lineage_store[_META_PARENT_STATE_ID] = concat_arrays(wt.Int32, [lineage_store[_META_PARENT_STATE_ID], parent_state_id])
        lineage_store[_META_LAST_EDGE_IDX] = concat_arrays(wt.Int32, [lineage_store[_META_LAST_EDGE_IDX], last_edge_idx])
        lineage_store[_META_LAST_REFLECTION_DEPTH_DELTA] = concat_arrays(wt.UInt32, [lineage_store[_META_LAST_REFLECTION_DEPTH_DELTA], last_reflection_depth_delta])
        lineage_store[SK_N_STATES] = int(next_state_id) + n_states
        lineage[_META_STATE_ID] = state_id
        lineage[_META_LINEAGE_STORE] = lineage_store
        State.attach_lineage(state_arrays, lineage, Geo.history_size(state_arrays))
        return (state_arrays, lineage_store, int(next_state_id) + n_states)

    def make(edge_idx, edge_pos, edge_dir, n0, nn, wedge_n, adjacent_face0, adjacent_face1, source_pos, incident_field, incident_normal_derivative, path_length_prefix=None, first_interaction_pos=None, edge_line_min=None, edge_line_max=None, r0=None, rn=None, incident_vector=None, incident_normal_derivative_vector=None, is_direct_tx=None, incident_jones=None, incident_derivative_jones=None, incident_basis=None, face0_operator=None, face1_operator=None, face0_material=None, face1_material=None, order=None, source_type_code=None, prefix_reflection_depth=None, intermediate_reflection_depth=None, suffix_reflection_depth=None, approximation_mode_code=None, lineage_parent_state_id=None, lineage_last_edge_idx=None, lineage_last_reflection_depth_delta=None, lineage_store=None, retain_lineage_state=True):
        n_states = dr.width(edge_idx)
        if order is None:
            order = dr.zeros(wt.UInt32, n_states)
        history_size = 0 if n_states == 0 else int(dr.max(order)[0])
        if n_states == 0:
            return State.empty(history_size=history_size, retain_lineage_state=retain_lineage_state)
        edge_line_min, edge_line_max = Geo.state_line_bounds({'edge_line_min': edge_line_min, 'edge_line_max': edge_line_max}, context='_make_state_arrays')
        if prefix_reflection_depth is None:
            prefix_reflection_depth = dr.zeros(wt.UInt32, n_states)
        if intermediate_reflection_depth is None:
            intermediate_reflection_depth = dr.zeros(wt.UInt32, n_states)
        if suffix_reflection_depth is None:
            suffix_reflection_depth = dr.zeros(wt.UInt32, n_states)
        if r0 is None:
            r0 = complex_zero(n_states)
        if rn is None:
            rn = complex_zero(n_states)
        canonical = State.canonicalize_transport(edge_pos=edge_pos, edge_dir=edge_dir, source_pos=source_pos, r0=r0, rn=rn, incident_vector=incident_vector, incident_normal_derivative_vector=incident_normal_derivative_vector, incident_jones=incident_jones, incident_derivative_jones=incident_derivative_jones, incident_basis=incident_basis, face0_operator=face0_operator, face1_operator=face1_operator)
        incident_vector = canonical['incident_vector']
        incident_normal_derivative_vector = canonical['incident_normal_derivative_vector']
        incident_jones = canonical['incident_jones']
        incident_derivative_jones = canonical['incident_derivative_jones']
        incident_basis = canonical['incident_basis']
        face0_operator = canonical['face0_operator']
        face1_operator = canonical['face1_operator']
        if face0_material is None:
            face0_material = State.default_face_material(n_states)
        if face1_material is None:
            face1_material = State.default_face_material(n_states)
        if 'mu_r' not in face0_material:
            face0_material = dict(face0_material)
            face0_material['mu_r'] = dr.ones(wt.Float, n_states)
        if 'mu_r' not in face1_material:
            face1_material = dict(face1_material)
            face1_material['mu_r'] = dr.ones(wt.Float, n_states)
        dr.eval(edge_idx, edge_pos, edge_dir, n0, nn, wedge_n, edge_line_min, edge_line_max, adjacent_face0, adjacent_face1, source_pos, incident_field, incident_normal_derivative, r0, rn, incident_jones['u'], incident_jones['v'], incident_derivative_jones['u'], incident_derivative_jones['v'], incident_basis['u'], incident_basis['v'], incident_basis['k'], face0_operator['m00'], face0_operator['m01'], face0_operator['m10'], face0_operator['m11'], face1_operator['m00'], face1_operator['m01'], face1_operator['m10'], face1_operator['m11'], face0_material['eta_r'], face0_material['mu_r'], face0_material['sigma'], face0_material['gain'], face0_material['use_fresnel'], face1_material['eta_r'], face1_material['mu_r'], face1_material['sigma'], face1_material['gain'], face1_material['use_fresnel'], incident_vector['x'], incident_vector['y'], incident_vector['z'], incident_normal_derivative_vector['x'], incident_normal_derivative_vector['y'], incident_normal_derivative_vector['z'], prefix_reflection_depth, intermediate_reflection_depth, suffix_reflection_depth, order)
        state_arrays = {SK_EDGE_IDX: edge_idx, SK_EDGE_POS: edge_pos, SK_EDGE_DIR: edge_dir, SK_N0: n0, SK_NN: nn, SK_WEDGE_N: wedge_n, SK_ADJACENT_FACE0: adjacent_face0, SK_ADJACENT_FACE1: adjacent_face1, SK_SOURCE_POS: source_pos, SK_INCIDENT_FIELD: incident_field, SK_INCIDENT_NORMAL_DERIVATIVE: incident_normal_derivative, SK_INCIDENT_JONES_U: incident_jones['u'], SK_INCIDENT_JONES_V: incident_jones['v'], SK_INCIDENT_DERIVATIVE_JONES_U: incident_derivative_jones['u'], SK_INCIDENT_DERIVATIVE_JONES_V: incident_derivative_jones['v'], SK_R0: r0, SK_RN: rn, SK_INCIDENT_BASIS_U: incident_basis['u'], SK_INCIDENT_BASIS_V: incident_basis['v'], SK_INCIDENT_BASIS_K: incident_basis['k'], SK_FACE0_OPERATOR_M00: face0_operator['m00'], SK_FACE0_OPERATOR_M01: face0_operator['m01'], SK_FACE0_OPERATOR_M10: face0_operator['m10'], SK_FACE0_OPERATOR_M11: face0_operator['m11'], SK_FACE1_OPERATOR_M00: face1_operator['m00'], SK_FACE1_OPERATOR_M01: face1_operator['m01'], SK_FACE1_OPERATOR_M10: face1_operator['m10'], SK_FACE1_OPERATOR_M11: face1_operator['m11'], SK_FACE0_ETA_R: face0_material['eta_r'], SK_FACE0_MU_R: face0_material['mu_r'], SK_FACE0_SIGMA: face0_material['sigma'], SK_FACE0_GAIN: face0_material['gain'], SK_FACE0_USE_FRESNEL: face0_material['use_fresnel'], SK_FACE1_ETA_R: face1_material['eta_r'], SK_FACE1_MU_R: face1_material['mu_r'], SK_FACE1_SIGMA: face1_material['sigma'], SK_FACE1_GAIN: face1_material['gain'], SK_FACE1_USE_FRESNEL: face1_material['use_fresnel'], SK_INCIDENT_VECTOR_X: incident_vector['x'], SK_INCIDENT_VECTOR_Y: incident_vector['y'], SK_INCIDENT_VECTOR_Z: incident_vector['z'], SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X: incident_normal_derivative_vector['x'], SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y: incident_normal_derivative_vector['y'], SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z: incident_normal_derivative_vector['z'], SK_PREFIX_REFLECTION_DEPTH: prefix_reflection_depth, SK_INTERMEDIATE_REFLECTION_DEPTH: intermediate_reflection_depth, SK_SUFFIX_REFLECTION_DEPTH: suffix_reflection_depth, SK_ORDER: order, SK_N_STATES: n_states, SK_EDGE_LINE_MIN: edge_line_min, SK_EDGE_LINE_MAX: edge_line_max}
        lineage = None
        if retain_lineage_state:
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
            dr.eval(path_length_prefix, first_interaction_pos, is_direct_tx, source_type_code, approximation_mode_code)
            state_arrays[SK_PATH_LENGTH_PREFIX] = path_length_prefix
            state_arrays[SK_FIRST_INTERACTION_POS] = first_interaction_pos
            state_arrays[SK_IS_DIRECT_TX] = is_direct_tx
            state_arrays[SK_SOURCE_TYPE_CODE] = source_type_code
            state_arrays[SK_APPROXIMATION_MODE_CODE] = approximation_mode_code
            lineage = State.empty_lineage(n_states)
            lineage.pop(_META_STATE_ID, None)
            lineage[_META_PARENT_STATE_ID] = lineage[_META_PARENT_STATE_ID] if lineage_parent_state_id is None else lineage_parent_state_id
            lineage[_META_LAST_EDGE_IDX] = wt.Int32(edge_idx) if lineage_last_edge_idx is None else lineage_last_edge_idx
            lineage[_META_LAST_REFLECTION_DEPTH_DELTA] = dr.zeros(wt.UInt32, n_states) if lineage_last_reflection_depth_delta is None else lineage_last_reflection_depth_delta
            if lineage_store is not None:
                lineage[_META_LINEAGE_STORE] = lineage_store
        State.attach_lineage(state_arrays, lineage, history_size)
        return state_arrays

    def take(state_arrays, keep_idx):
        history_size = Geo.history_size(state_arrays)
        retain_lineage_state = State.has_lineage_state(state_arrays)
        if state_arrays is None or state_arrays[SK_N_STATES] == 0 or dr.width(keep_idx) == 0:
            return State.empty(history_size=history_size, retain_lineage_state=retain_lineage_state)
        from ..kernels.packed_state import gather_state_arrays
        gathered = gather_state_arrays(state_arrays, keep_idx)
        gathered[SK_N_STATES] = dr.width(keep_idx)
        return gathered

    def global_to_local_index(scene, edge_data):
        if scene is None or edge_data is None or edge_data.get('global_idx') is None:
            return None
        global_idx = edge_data['global_idx']
        if edge_data['n_edges'] <= 0 or dr.width(global_idx) == 0:
            return None
        valid = global_idx >= 0
        if not bool(dr.any(valid)):
            return None
        max_global_idx = dr.max(dr.select(valid, global_idx, wt.Int32(0)))
        n_global_edges = int(max_global_idx[0]) + 1
        mapping = dr.full(wt.Int32, -1, n_global_edges)
        local_idx = dr.arange(wt.Int32, edge_data['n_edges'])
        dr.scatter(mapping, local_idx, wt.UInt32(dr.select(valid, global_idx, wt.Int32(0))), valid)
        dr.eval(mapping)
        return mapping

    def gather_path_export_lineage(state_arrays: dict, indices):
        lineage = state_arrays.get(_PATH_EXPORT_LINEAGE_KEY, state_arrays.get(_STATE_LINEAGE_KEY))
        if lineage is None:
            return None
        gathered = {}
        if _META_PARENT_STATE_ID in lineage:
            gathered[_META_PARENT_STATE_ID] = dr.gather(wt.Int32, lineage[_META_PARENT_STATE_ID], indices)
        if _META_LAST_EDGE_IDX in lineage:
            gathered[_META_LAST_EDGE_IDX] = dr.gather(wt.Int32, lineage[_META_LAST_EDGE_IDX], indices)
        if _META_LAST_REFLECTION_DEPTH_DELTA in lineage:
            gathered[_META_LAST_REFLECTION_DEPTH_DELTA] = dr.gather(wt.UInt32, lineage[_META_LAST_REFLECTION_DEPTH_DELTA], indices)
        if _META_STATE_ID in lineage:
            gathered[_META_STATE_ID] = dr.gather(wt.Int32, lineage[_META_STATE_ID], indices)
        if _META_LINEAGE_STORE in lineage:
            gathered[_META_LINEAGE_STORE] = lineage[_META_LINEAGE_STORE]
        return gathered

    def reduce_for_path_export(state_arrays: dict | None) -> dict | None:
        if state_arrays is None or state_arrays.get(_PATH_EXPORT_STATE_LAYOUT_KEY) == PATH_EXPORT_REDUCED_STATE_LAYOUT:
            return state_arrays
        history_size = Geo.history_size(state_arrays)
        reduced = {key: state_arrays[key] for key in _PATH_EXPORT_REDUCED_KEYS if key in state_arrays}
        reduced[SK_N_STATES] = int(state_arrays[SK_N_STATES])
        reduced[_PATH_EXPORT_STATE_LAYOUT_KEY] = PATH_EXPORT_REDUCED_STATE_LAYOUT
        State.attach_lineage(reduced, None, history_size)
        reduced[STATE_HISTORY_SIZE_KEY] = int(history_size)
        lineage = state_arrays.get(_STATE_LINEAGE_KEY)
        if lineage is not None:
            reduced[_PATH_EXPORT_LINEAGE_KEY] = lineage
        return reduced

    def gather_path_export_keys(state_arrays: dict, indices, keys, *, attach_lineage: bool) -> dict:
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
        return State.attach_lineage(gathered, State.gather_path_export_lineage(state_arrays, indices) if attach_lineage else None, Geo.history_size(state_arrays))

    def gather_path_export_eval(state_arrays: dict, indices) -> dict:
        if state_arrays.get(_PATH_EXPORT_STATE_LAYOUT_KEY) != PATH_EXPORT_REDUCED_STATE_LAYOUT:
            from ..kernels.packed_state import gather_state_arrays
            return gather_state_arrays(state_arrays, indices)
        return State.gather_path_export_keys(state_arrays, indices, _PATH_EXPORT_EVAL_KEYS, attach_lineage=False)

    def gather_path_export_replay(state_arrays: dict, indices) -> dict:
        if state_arrays.get(_PATH_EXPORT_STATE_LAYOUT_KEY) != PATH_EXPORT_REDUCED_STATE_LAYOUT:
            from ..kernels.packed_state.drjit_impl import gather_inserted_reflection_state_fields
            return gather_inserted_reflection_state_fields(state_arrays, indices)
        return State.gather_path_export_keys(state_arrays, indices, _PATH_EXPORT_REPLAY_KEYS, attach_lineage=True)


__all__ = ["State", "PATH_EXPORT_REDUCED_STATE_LAYOUT", "Geo"]
