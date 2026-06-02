"""Domain constants, labels, and edge preloading."""

import drjit as dr
import witwin as wt

from ...utils.constants import EPS, SPEED_OF_LIGHT  # noqa: F401 鈥?re-exported


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
STATE_HISTORY_SIZE_METADATA_KEY = "__state_history_size__"


def _edge_attr(edge, name, default=None):
    return getattr(edge, name, default)


def preload_diffraction_edges(diffraction_points, global_indices=None):
    """
    Preload all diffraction edge data to GPU arrays using DrJit Point3f/Vector3f.

    This function preserves gradient connections by concatenating DrJit arrays
    directly instead of extracting scalar values.
    """
    n_edges = len(diffraction_points)
    if n_edges == 0:
        return None

    pos_list = []
    edge_dir_list = []
    n0_list = []
    nn_list = []
    wedge_n_list = []
    line_min_list = []
    line_max_list = []
    adjacent_face0_list = []
    adjacent_face1_list = []
    valid_edges = []
    global_idx_list = []

    if global_indices is None:
        global_indices = [None] * n_edges

    for dif_point, global_idx in zip(diffraction_points, global_indices):
        face_normals = _edge_attr(dif_point, "face_normals_3d", [])
        if len(face_normals) < 2:
            continue

        pos_list.append(_edge_attr(dif_point, "position"))

        edge_vec = _edge_attr(dif_point, "edge_vector")
        edge_len = dr.norm(edge_vec)
        edge_dir_list.append(edge_vec / dr.maximum(edge_len, wt.Float(EPS)))

        n0_list.append(face_normals[0])
        nn_list.append(face_normals[1])

        wedge_n_list.append(_edge_attr(dif_point, "wedge_n"))
        line_min = _edge_attr(dif_point, "line_min")
        line_max = _edge_attr(dif_point, "line_max")
        if line_min is None or line_max is None:
            raise ValueError(
                "Finite diffraction edge bounds require line_min and line_max for every preloaded edge."
            )
        line_min_list.append(line_min)
        line_max_list.append(line_max)
        adjacent_faces = tuple(int(face_idx) for face_idx in _edge_attr(dif_point, "adjacent_faces", ()))
        adjacent_face0_list.append(adjacent_faces[0] if len(adjacent_faces) > 0 else -1)
        adjacent_face1_list.append(adjacent_faces[1] if len(adjacent_faces) > 1 else -1)
        valid_edges.append(dif_point)
        edge_global_idx = int(_edge_attr(dif_point, "global_index", -1))
        if edge_global_idx < 0 and global_idx is not None:
            edge_global_idx = int(global_idx)
        global_idx_list.append(edge_global_idx)

    n_valid = len(valid_edges)
    if n_valid == 0:
        return None

    pos_dr = wt.Point3f(
        dr.concat([p.x for p in pos_list]),
        dr.concat([p.y for p in pos_list]),
        dr.concat([p.z for p in pos_list]),
    )
    edge_dir_dr = wt.Vector3f(
        dr.concat([e.x for e in edge_dir_list]),
        dr.concat([e.y for e in edge_dir_list]),
        dr.concat([e.z for e in edge_dir_list]),
    )
    n0_dr = wt.Vector3f(
        dr.concat([n.x for n in n0_list]),
        dr.concat([n.y for n in n0_list]),
        dr.concat([n.z for n in n0_list]),
    )
    nn_dr = wt.Vector3f(
        dr.concat([n.x for n in nn_list]),
        dr.concat([n.y for n in nn_list]),
        dr.concat([n.z for n in nn_list]),
    )
    wedge_n_dr = dr.concat(wedge_n_list)

    edge_data = {
        "n_edges": n_valid,
        "pos": pos_dr,
        "edge_dir": edge_dir_dr,
        "n0": n0_dr,
        "n_face_n": nn_dr,
        "wedge_n": wedge_n_dr,
        "adjacent_face0": wt.Int32(adjacent_face0_list),
        "adjacent_face1": wt.Int32(adjacent_face1_list),
        "global_idx": wt.Int32(global_idx_list),
        "valid_edges": valid_edges,
    }
    if len(line_min_list) != n_valid or len(line_max_list) != n_valid:
        raise ValueError("Finite diffraction edge bounds require line_min and line_max for every preloaded edge.")
    edge_data["line_min"] = dr.concat(line_min_list)
    edge_data["line_max"] = dr.concat(line_max_list)
    return edge_data


def _source_type_label(code):
    if code == SOURCE_TYPE_DIRECT_TX:
        return "direct_tx"
    if code == SOURCE_TYPE_REFLECTION_PREFIX:
        return "reflection_prefix"
    return f"unknown_source_type_{code}"


def _approximation_mode_label(code):
    if code == APPROX_MODE_DIRECT_FIRST_ORDER:
        return "exact_direct_first_order"
    if code == APPROX_MODE_RECURSIVE_DIFFRACTION:
        return "approx_recursive_diffraction"
    if code == APPROX_MODE_SAMPLED_REFLECTION_PREFIX:
        return "approx_sampled_reflection_prefix"
    if code == APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN:
        return "approx_sampled_reflection_prefix_chain"
    if code == APPROX_MODE_SAMPLED_INSERTED_REFLECTION:
        return "approx_sampled_inserted_reflection"
    if code == APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN:
        return "approx_sampled_inserted_reflection_chain"
    return f"unknown_approximation_mode_{code}"


def _ownership_label(code):
    if code == OWNERSHIP_DIRECT_DIFFRACTION:
        return "direct_diffraction"
    if code == OWNERSHIP_MIXED_DIFFRACTION:
        return "mixed_diffraction"
    return f"unknown_ownership_{code}"


def _ownership_code_from_depths(
    prefix_reflection_depth,
    intermediate_reflection_depth,
    suffix_reflection_depth,
):
    has_reflection = (
        (prefix_reflection_depth > wt.UInt32(0))
        | (intermediate_reflection_depth > wt.UInt32(0))
        | (suffix_reflection_depth > wt.UInt32(0))
    )
    return dr.select(
        has_reflection,
        wt.UInt32(OWNERSHIP_MIXED_DIFFRACTION),
        wt.UInt32(OWNERSHIP_DIRECT_DIFFRACTION),
    )


def _path_sequence_label(
    order,
    prefix_reflection_depth,
    intermediate_reflection_depth=0,
    suffix_reflection_depth=0,
    path_reflection_depth_history=None,
):
    steps = ["S"]

    if path_reflection_depth_history is None:
        if prefix_reflection_depth > 0:
            steps.append("R" if prefix_reflection_depth == 1 else f"R^{prefix_reflection_depth}")
        if intermediate_reflection_depth > 0 and order > 0:
            steps.extend("D" for _ in range(max(0, order - 1)))
            steps.append("R" if intermediate_reflection_depth == 1 else f"R^{intermediate_reflection_depth}")
            steps.append("D")
        else:
            steps.extend("D" for _ in range(max(0, order)))
    else:
        prefix_depth = int(path_reflection_depth_history[0]) if len(path_reflection_depth_history) > 0 else 0
        if prefix_depth > 0:
            steps.append("R" if prefix_depth == 1 else f"R^{prefix_depth}")
        for slot in range(max(0, order)):
            steps.append("D")
            if slot + 1 >= order:
                continue
            reflection_depth = (
                int(path_reflection_depth_history[slot + 1])
                if slot + 1 < len(path_reflection_depth_history)
                else 0
            )
            if reflection_depth > 0:
                steps.append("R" if reflection_depth == 1 else f"R^{reflection_depth}")
    if suffix_reflection_depth > 0:
        steps.append("R" if suffix_reflection_depth == 1 else f"R^{suffix_reflection_depth}")
    return " -> ".join(steps)


def _history_keys(history_size):
    return [f"path_edge_idx_{slot}" for slot in range(history_size)]


def _reflection_history_keys(history_size):
    return [f"path_reflection_depth_{slot}" for slot in range(history_size)]


def _state_history_size(state_arrays):
    if state_arrays is None:
        return 0
    return int(state_arrays.get(STATE_HISTORY_SIZE_METADATA_KEY, 0))


def _distance_to_cot_pole(arg):
    nearest_pole = dr.round(arg / dr.pi) * dr.pi
    return dr.abs(arg - nearest_pole)


def _cartesian_chunk_size(left_size, right_size):
    """Return a left-dimension chunk length that caps left*right temporary pairs."""
    left = max(0, int(left_size))
    right = max(0, int(right_size))
    if left == 0 or right == 0:
        return 0
    pair_budget = max(1, int(CARTESIAN_PAIR_CHUNK_BUDGET))
    return max(1, min(left, pair_budget // right if right > 0 else pair_budget))


# ---------------------------------------------------------------------------
# State array key constants
# ---------------------------------------------------------------------------

SK_EDGE_IDX = "edge_idx"
SK_EDGE_POS = "edge_pos"
SK_EDGE_DIR = "edge_dir"
SK_EDGE_LINE_MIN = "edge_line_min"
SK_EDGE_LINE_MAX = "edge_line_max"
SK_N0 = "n0"
SK_NN = "n_face_n"
SK_WEDGE_N = "wedge_n"
SK_ADJACENT_FACE0 = "adjacent_face0"
SK_ADJACENT_FACE1 = "adjacent_face1"
SK_SOURCE_POS = "source_pos"
SK_PATH_LENGTH_PREFIX = "path_length_prefix"
SK_FIRST_INTERACTION_POS = "first_interaction_pos"
SK_INCIDENT_FIELD = "incident_field"
SK_INCIDENT_NORMAL_DERIVATIVE = "incident_normal_derivative"
SK_INCIDENT_JONES_U = "incident_jones_u"
SK_INCIDENT_JONES_V = "incident_jones_v"
SK_INCIDENT_DERIVATIVE_JONES_U = "incident_derivative_jones_u"
SK_INCIDENT_DERIVATIVE_JONES_V = "incident_derivative_jones_v"
SK_R0 = "r_face0"
SK_RN = "r_face_n"
SK_INCIDENT_BASIS_U = "incident_basis_u"
SK_INCIDENT_BASIS_V = "incident_basis_v"
SK_INCIDENT_BASIS_K = "incident_basis_k"
SK_FACE0_OPERATOR_M00 = "face0_operator_m00"
SK_FACE0_OPERATOR_M01 = "face0_operator_m01"
SK_FACE0_OPERATOR_M10 = "face0_operator_m10"
SK_FACE0_OPERATOR_M11 = "face0_operator_m11"
SK_FACE1_OPERATOR_M00 = "face1_operator_m00"
SK_FACE1_OPERATOR_M01 = "face1_operator_m01"
SK_FACE1_OPERATOR_M10 = "face1_operator_m10"
SK_FACE1_OPERATOR_M11 = "face1_operator_m11"
SK_FACE0_ETA_R = "face0_eta_r"
SK_FACE0_SIGMA = "face0_sigma"
SK_FACE0_GAIN = "face0_gain"
SK_FACE0_USE_FRESNEL = "face0_use_fresnel"
SK_FACE1_ETA_R = "face1_eta_r"
SK_FACE1_SIGMA = "face1_sigma"
SK_FACE1_GAIN = "face1_gain"
SK_FACE1_USE_FRESNEL = "face1_use_fresnel"
SK_INCIDENT_VECTOR_X = "incident_vector_x"
SK_INCIDENT_VECTOR_Y = "incident_vector_y"
SK_INCIDENT_VECTOR_Z = "incident_vector_z"
SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X = "incident_normal_derivative_vector_x"
SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y = "incident_normal_derivative_vector_y"
SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z = "incident_normal_derivative_vector_z"
SK_IS_DIRECT_TX = "is_direct_tx"
SK_SOURCE_TYPE_CODE = "source_type_code"
SK_PREFIX_REFLECTION_DEPTH = "prefix_reflection_depth"
SK_INTERMEDIATE_REFLECTION_DEPTH = "intermediate_reflection_depth"
SK_SUFFIX_REFLECTION_DEPTH = "suffix_reflection_depth"
SK_APPROXIMATION_MODE_CODE = "approximation_mode_code"
SK_ORDER = "order"
SK_N_STATES = "n_states"

STATE_STATIC_KEYS = (
    SK_EDGE_IDX, SK_EDGE_POS, SK_EDGE_DIR, SK_N0, SK_NN, SK_WEDGE_N,
    SK_ADJACENT_FACE0, SK_ADJACENT_FACE1, SK_SOURCE_POS,
    SK_PATH_LENGTH_PREFIX, SK_FIRST_INTERACTION_POS,
    SK_INCIDENT_FIELD, SK_INCIDENT_NORMAL_DERIVATIVE,
    SK_INCIDENT_JONES_U, SK_INCIDENT_JONES_V,
    SK_INCIDENT_DERIVATIVE_JONES_U, SK_INCIDENT_DERIVATIVE_JONES_V,
    SK_R0, SK_RN,
    SK_INCIDENT_BASIS_U, SK_INCIDENT_BASIS_V, SK_INCIDENT_BASIS_K,
    SK_FACE0_OPERATOR_M00, SK_FACE0_OPERATOR_M01,
    SK_FACE0_OPERATOR_M10, SK_FACE0_OPERATOR_M11,
    SK_FACE1_OPERATOR_M00, SK_FACE1_OPERATOR_M01,
    SK_FACE1_OPERATOR_M10, SK_FACE1_OPERATOR_M11,
    SK_FACE0_ETA_R, SK_FACE0_SIGMA, SK_FACE0_GAIN, SK_FACE0_USE_FRESNEL,
    SK_FACE1_ETA_R, SK_FACE1_SIGMA, SK_FACE1_GAIN, SK_FACE1_USE_FRESNEL,
    SK_INCIDENT_VECTOR_X, SK_INCIDENT_VECTOR_Y, SK_INCIDENT_VECTOR_Z,
    SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X,
    SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y,
    SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z,
    SK_IS_DIRECT_TX, SK_SOURCE_TYPE_CODE,
    SK_PREFIX_REFLECTION_DEPTH, SK_INTERMEDIATE_REFLECTION_DEPTH,
    SK_SUFFIX_REFLECTION_DEPTH, SK_APPROXIMATION_MODE_CODE, SK_ORDER,
)

STATE_COLD_KEYS = (
    SK_PATH_LENGTH_PREFIX,
    SK_FIRST_INTERACTION_POS,
    SK_IS_DIRECT_TX,
    SK_SOURCE_TYPE_CODE,
    SK_APPROXIMATION_MODE_CODE,
)

STATE_HOT_KEYS = tuple(key for key in STATE_STATIC_KEYS if key not in STATE_COLD_KEYS)


def _dynamic_state_keys(history_size):
    """Return dynamic history key names for a given history size."""
    keys = []
    for slot in range(history_size):
        keys.append(f"path_edge_idx_{slot}")
        keys.append(f"path_reflection_depth_{slot}")
    return tuple(keys)


def _validate_state_arrays(state_arrays, *, context=""):
    """Debug-mode schema check for state array dicts."""
    missing = [k for k in STATE_HOT_KEYS if k not in state_arrays]
    if missing:
        raise KeyError(f"State arrays missing required hot keys ({context}): {missing}")
    if SK_N_STATES not in state_arrays:
        raise KeyError(f"State arrays missing '{SK_N_STATES}' ({context})")
    if not isinstance(state_arrays[SK_N_STATES], int):
        raise TypeError(
            f"state_arrays['{SK_N_STATES}'] must be int, got {type(state_arrays[SK_N_STATES])} ({context})"
        )


_SK_NAMES = [
    "SK_EDGE_IDX", "SK_EDGE_POS", "SK_EDGE_DIR", "SK_EDGE_LINE_MIN", "SK_EDGE_LINE_MAX",
    "SK_N0", "SK_NN", "SK_WEDGE_N",
    "SK_ADJACENT_FACE0", "SK_ADJACENT_FACE1", "SK_SOURCE_POS",
    "SK_PATH_LENGTH_PREFIX", "SK_FIRST_INTERACTION_POS",
    "SK_INCIDENT_FIELD", "SK_INCIDENT_NORMAL_DERIVATIVE",
    "SK_INCIDENT_JONES_U", "SK_INCIDENT_JONES_V",
    "SK_INCIDENT_DERIVATIVE_JONES_U", "SK_INCIDENT_DERIVATIVE_JONES_V",
    "SK_R0", "SK_RN",
    "SK_INCIDENT_BASIS_U", "SK_INCIDENT_BASIS_V", "SK_INCIDENT_BASIS_K",
    "SK_FACE0_OPERATOR_M00", "SK_FACE0_OPERATOR_M01",
    "SK_FACE0_OPERATOR_M10", "SK_FACE0_OPERATOR_M11",
    "SK_FACE1_OPERATOR_M00", "SK_FACE1_OPERATOR_M01",
    "SK_FACE1_OPERATOR_M10", "SK_FACE1_OPERATOR_M11",
    "SK_FACE0_ETA_R", "SK_FACE0_SIGMA", "SK_FACE0_GAIN", "SK_FACE0_USE_FRESNEL",
    "SK_FACE1_ETA_R", "SK_FACE1_SIGMA", "SK_FACE1_GAIN", "SK_FACE1_USE_FRESNEL",
    "SK_INCIDENT_VECTOR_X", "SK_INCIDENT_VECTOR_Y", "SK_INCIDENT_VECTOR_Z",
    "SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X",
    "SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y",
    "SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z",
    "SK_IS_DIRECT_TX", "SK_SOURCE_TYPE_CODE",
    "SK_PREFIX_REFLECTION_DEPTH", "SK_INTERMEDIATE_REFLECTION_DEPTH",
    "SK_SUFFIX_REFLECTION_DEPTH", "SK_APPROXIMATION_MODE_CODE", "SK_ORDER",
    "SK_N_STATES",
]

__all__ = [
    "APPROX_MODE_DIRECT_FIRST_ORDER",
    "APPROX_MODE_RECURSIVE_DIFFRACTION",
    "APPROX_MODE_SAMPLED_INSERTED_REFLECTION",
    "APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN",
    "APPROX_MODE_SAMPLED_REFLECTION_PREFIX",
    "APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN",
    "CARTESIAN_PAIR_CHUNK_BUDGET",
    "OWNERSHIP_DIRECT_DIFFRACTION",
    "OWNERSHIP_MIXED_DIFFRACTION",
    "SOURCE_TYPE_DIRECT_TX",
    "SOURCE_TYPE_REFLECTION_PREFIX",
    "SPEED_OF_LIGHT",
    "STATE_COLD_KEYS",
    "STATE_HISTORY_SIZE_METADATA_KEY",
    "STATE_HOT_KEYS",
    "STATE_STATIC_KEYS",
    "_approximation_mode_label",
    "_cartesian_chunk_size",
    "_distance_to_cot_pole",
    "_dynamic_state_keys",
    "_edge_attr",
    "_history_keys",
    "_ownership_code_from_depths",
    "_ownership_label",
    "_path_sequence_label",
    "_reflection_history_keys",
    "_source_type_label",
    "_state_history_size",
    "_validate_state_arrays",
    "preload_diffraction_edges",
] + _SK_NAMES
