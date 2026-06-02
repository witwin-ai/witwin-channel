"""Audit trail and state analysis for diffraction state arrays."""

import drjit as dr
import witwin as wt

from ..constants import (
    _approximation_mode_label,
    _history_keys,
    _ownership_code_from_depths,
    _ownership_label,
    _path_sequence_label,
    _reflection_history_keys,
    _source_type_label,
    _state_history_size,
)
from ..geometry import _compute_incident_edge_geometry
from .arrays import _materialize_state_history


__all__ = [
    "_build_global_to_local_index",
    "_build_state_audit",
    "_empty_state_audit",
]


def _build_global_to_local_index(scene, edge_data):
    if scene is None or edge_data is None or edge_data.get("global_idx") is None:
        return None

    global_idx = edge_data["global_idx"]
    if edge_data["n_edges"] <= 0 or dr.width(global_idx) == 0:
        return None

    valid = global_idx >= 0
    if not bool(dr.any(valid)):
        return None

    max_global_idx = dr.max(dr.select(valid, global_idx, wt.Int32(0)))
    n_global_edges = int(max_global_idx[0]) + 1
    mapping = dr.full(wt.Int32, -1, n_global_edges)
    local_idx = dr.arange(wt.Int32, edge_data["n_edges"])
    dr.scatter(mapping, local_idx, wt.UInt32(dr.select(valid, global_idx, wt.Int32(0))), valid)
    dr.eval(mapping)
    return mapping


def _empty_state_audit(history_size):
    audit = {
        "n_states": 0,
        "history_size": int(history_size),
        "path_length_prefix": dr.zeros(wt.Float, 0),
        "first_interaction_pos": wt.Point3f(
            dr.zeros(wt.Float, 0),
            dr.zeros(wt.Float, 0),
            dr.zeros(wt.Float, 0),
        ),
        "source_type_code": dr.zeros(wt.UInt32, 0),
        "ownership_code": dr.zeros(wt.UInt32, 0),
        "prefix_reflection_depth": dr.zeros(wt.UInt32, 0),
        "intermediate_reflection_depth": dr.zeros(wt.UInt32, 0),
        "suffix_reflection_depth": dr.zeros(wt.UInt32, 0),
        "approximation_mode_code": dr.zeros(wt.UInt32, 0),
        "source_type": (),
        "ownership": (),
        "approximation_mode": (),
        "path_sequence": (),
    }
    for key in _history_keys(history_size):
        audit[key] = dr.zeros(wt.Int32, 0)
        audit[key.replace("path_edge_idx_", "path_global_edge_idx_")] = dr.zeros(wt.Int32, 0)
    for key in _reflection_history_keys(history_size):
        audit[key] = dr.zeros(wt.UInt32, 0)
    return audit


def _build_state_audit(state_arrays, edge_data):
    history_size = _state_history_size(state_arrays)
    if state_arrays["n_states"] == 0:
        return _empty_state_audit(history_size)
    path_edge_history, path_reflection_depth_history = _materialize_state_history(state_arrays)

    phi_prime, s_prime = _compute_incident_edge_geometry(
        state_arrays["source_pos"],
        state_arrays["edge_pos"],
        state_arrays["edge_dir"],
        state_arrays["n0"],
    )
    edge_global_idx = dr.zeros(wt.Int32, state_arrays["n_states"])
    if edge_data is not None and edge_data.get("global_idx") is not None:
        edge_global_idx = dr.gather(wt.Int32, edge_data["global_idx"], state_arrays["edge_idx"])

    ownership_code = _ownership_code_from_depths(
        state_arrays["prefix_reflection_depth"],
        state_arrays["intermediate_reflection_depth"],
        state_arrays["suffix_reflection_depth"],
    )

    audit = {
        "n_states": int(state_arrays["n_states"]),
        "history_size": int(history_size),
        "order": state_arrays["order"],
        "edge_idx": wt.Int32(state_arrays["edge_idx"]),
        "edge_global_idx": edge_global_idx,
        "edge_pos": state_arrays["edge_pos"],
        "edge_dir": state_arrays["edge_dir"],
        "source_pos": state_arrays["source_pos"],
        "path_length_prefix": state_arrays["path_length_prefix"],
        "first_interaction_pos": state_arrays["first_interaction_pos"],
        "wedge_n": state_arrays["wedge_n"],
        "adjacent_face0": state_arrays["adjacent_face0"],
        "adjacent_face1": state_arrays["adjacent_face1"],
        "phi_prime": phi_prime,
        "s_prime": s_prime,
        "incident_field": state_arrays["incident_field"],
        "incident_normal_derivative": state_arrays["incident_normal_derivative"],
        "is_direct_tx": state_arrays["is_direct_tx"],
        "source_type_code": state_arrays["source_type_code"],
        "ownership_code": ownership_code,
        "prefix_reflection_depth": state_arrays["prefix_reflection_depth"],
        "intermediate_reflection_depth": state_arrays["intermediate_reflection_depth"],
        "suffix_reflection_depth": state_arrays["suffix_reflection_depth"],
        "approximation_mode_code": state_arrays["approximation_mode_code"],
    }
    for slot in range(history_size):
        key = f"path_edge_idx_{slot}"
        local_idx = (
            path_edge_history[slot]
            if slot < len(path_edge_history)
            else dr.full(wt.Int32, -1, state_arrays["n_states"])
        )
        audit[key] = local_idx
        global_key = f"path_global_edge_idx_{slot}"
        if edge_data is not None and edge_data.get("global_idx") is not None:
            valid = local_idx >= 0
            safe_idx = dr.select(valid, wt.UInt32(local_idx), wt.UInt32(0))
            global_idx = dr.gather(wt.Int32, edge_data["global_idx"], safe_idx)
            audit[global_key] = dr.select(valid, global_idx, wt.Int32(-1))
        else:
            audit[global_key] = wt.Int32(local_idx)
        reflection_key = f"path_reflection_depth_{slot}"
        audit[reflection_key] = (
            path_reflection_depth_history[slot]
            if slot < len(path_reflection_depth_history)
            else dr.zeros(wt.UInt32, state_arrays["n_states"])
        )

    dr.eval(
        audit["order"],
        audit["edge_idx"],
        audit["edge_global_idx"],
        audit["edge_pos"],
        audit["edge_dir"],
        audit["source_pos"],
        audit["path_length_prefix"],
        audit["first_interaction_pos"],
        audit["wedge_n"],
        audit["adjacent_face0"],
        audit["adjacent_face1"],
        audit["phi_prime"],
        audit["s_prime"],
        audit["incident_field"],
        audit["incident_normal_derivative"],
        audit["is_direct_tx"],
        audit["source_type_code"],
        audit["ownership_code"],
        audit["prefix_reflection_depth"],
        audit["intermediate_reflection_depth"],
        audit["suffix_reflection_depth"],
        audit["approximation_mode_code"],
        *[audit[key] for key in _history_keys(history_size)],
        *[audit[f"path_global_edge_idx_{slot}"] for slot in range(history_size)],
        *[audit[key] for key in _reflection_history_keys(history_size)],
    )
    n_states = state_arrays["n_states"]
    audit["source_type"] = tuple(
        _source_type_label(int(audit["source_type_code"][idx]))
        for idx in range(n_states)
    )
    audit["ownership"] = tuple(
        _ownership_label(int(audit["ownership_code"][idx]))
        for idx in range(n_states)
    )
    audit["approximation_mode"] = tuple(
        _approximation_mode_label(int(audit["approximation_mode_code"][idx]))
        for idx in range(n_states)
    )
    audit["path_sequence"] = tuple(
        _path_sequence_label(
            int(audit["order"][idx]),
            int(audit["prefix_reflection_depth"][idx]),
            int(audit["intermediate_reflection_depth"][idx]),
            int(audit["suffix_reflection_depth"][idx]),
            path_reflection_depth_history=[
                int(audit[f"path_reflection_depth_{slot}"][idx])
                for slot in range(history_size)
            ],
        )
        for idx in range(n_states)
    )
    return audit
