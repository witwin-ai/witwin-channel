"""Diffraction state-array storage, pruning, and audit helpers.

This module re-exports all public names from the sub-modules so that
existing ``from .state import X`` imports continue to work unchanged.
"""

from .arrays import (  # noqa: F401
    _canonicalize_transport_state,
    _concat_state_arrays,
    _default_face_material,
    _empty_state_arrays,
    _finalize_state_lineage,
    _gather_state_arrays,
    _make_state_arrays,
    _materialize_state_history,
    _state_has_cold_metadata,
    _state_ids,
    _state_lineage_store,
    _state_lineage_first_edge_idx,
    _subset_state_arrays,
    _take_state_arrays,
    _torch_state_key,
)
from .pruning import (  # noqa: F401
    _prune_state_arrays_by_budget,
    _state_pruning_metric,
)
from .audit import (  # noqa: F401
    _build_global_to_local_index,
    _build_state_audit,
    _empty_state_audit,
)
from .path_export import (  # noqa: F401
    PATH_EXPORT_REDUCED_STATE_LAYOUT,
    gather_path_export_eval_state_fields,
    gather_path_export_field_state_fields,
    gather_path_export_replay_state_fields,
    gather_path_export_support_state_fields,
    is_path_export_reduced_state_arrays,
    path_export_state_layout,
    reduce_state_arrays_for_path_export,
)

__all__ = [
    "_build_global_to_local_index",
    "_build_state_audit",
    "_canonicalize_transport_state",
    "_concat_state_arrays",
    "_default_face_material",
    "_empty_state_arrays",
    "_empty_state_audit",
    "_finalize_state_lineage",
    "_gather_state_arrays",
    "_make_state_arrays",
    "_materialize_state_history",
    "_prune_state_arrays_by_budget",
    "_state_has_cold_metadata",
    "_state_ids",
    "_state_lineage_store",
    "_state_lineage_first_edge_idx",
    "_state_pruning_metric",
    "_subset_state_arrays",
    "_take_state_arrays",
    "_torch_state_key",
    "PATH_EXPORT_REDUCED_STATE_LAYOUT",
    "gather_path_export_eval_state_fields",
    "gather_path_export_field_state_fields",
    "gather_path_export_replay_state_fields",
    "gather_path_export_support_state_fields",
    "is_path_export_reduced_state_arrays",
    "path_export_state_layout",
    "reduce_state_arrays_for_path_export",
]
