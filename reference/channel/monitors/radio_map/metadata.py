from __future__ import annotations

from typing import Mapping

import numpy as np

from .backend import _normalize_radio_map_accumulation_backend
from .diagnostics import (
    MATCHED_ISB_COMPLETION_BOUNDARY_HALF_FIELD,
    PROJECTED_ISB_COMPLETION_GAIN,
    PROJECTED_ISB_COMPLETION_RATIO_TARGET,
    _matched_isb_completion_enabled,
    _projected_isb_completion_enabled,
    _shadow_boundary_mode,
    _utd_cross_term_surrogate_enabled,
)
from .monitor import RadioMapMonitor
from ...scene import Scene
from ...trace.materials import ReflectionTraceDetail
from ..profiler import capture_cuda_memory_report


_DIFFRACTION_DIAGNOSTIC_COUNT_KEYS = (
    "prepared_state_count",
    "visible_pair_count",
    "support_pair_count",
    "pair_valid_count",
    "shadow_completion_count",
    "interior_count",
    "hard_visibility_zero_count",
)


def _empty_diffraction_diagnostic_counts() -> dict[str, int]:
    return {
        key: 0
        for key in _DIFFRACTION_DIAGNOSTIC_COUNT_KEYS
    }


def _merge_diffraction_diagnostic_counts(*counts_payloads) -> dict[str, int]:
    merged = _empty_diffraction_diagnostic_counts()
    for payload in counts_payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in _DIFFRACTION_DIAGNOSTIC_COUNT_KEYS:
            merged[key] += int(payload.get(key, 0))
    return merged


def _resolve_noise_power(scene: Scene, monitor: RadioMapMonitor) -> tuple[float, str]:
    if monitor.noise_power is not None:
        return float(monitor.noise_power), "monitor_override"
    metadata = getattr(scene, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("noise_power", "thermal_noise_power"):
            if key in metadata:
                return float(metadata[key]), f"scene_metadata:{key}"
    for key in ("noise_power", "thermal_noise_power"):
        value = getattr(scene, key, None)
        if value is not None:
            return float(value), f"scene_attribute:{key}"
    return 0.0, "default_zero"


def _count_nonzero_complex(value) -> int:
    array = np.asarray(value, dtype=np.complex64)
    return int(np.count_nonzero(np.abs(array) > 0.0))


def _count_reflection_paths(reflection_detail: ReflectionTraceDetail | None) -> int:
    if reflection_detail is None:
        return 0
    groups = tuple(reflection_detail.source_paths_per_bounce)
    total = 0
    for group in groups:
        if group is None:
            continue
        total += int(group.get("n_paths", 0))
    return int(total)


def _radio_map_diffraction_state_layout(accumulation_backend: str) -> str:
    from ...trace.diffraction.state import PATH_EXPORT_REDUCED_STATE_LAYOUT

    return (
        "full"
        if str(accumulation_backend) == "native_coherent"
        else PATH_EXPORT_REDUCED_STATE_LAYOUT
    )


def _radio_map_diffraction_cache_key(base_key, *, state_layout: str):
    if base_key is None:
        return None
    return tuple(base_key) + ("radio_map_state_layout", str(state_layout))


class RadioMapMetadata(dict):
    def __init__(
        self,
        *,
        monitor: RadioMapMonitor,
        grid,
        scene: Scene,
        solver_controls,
        path_counts,
        sample_metadata,
        aggregate_runtime_reuse,
        aggregate_runtime_backends,
        reflection_detail: ReflectionTraceDetail | None,
        radio_map_accumulation_backend: str,
        resolved_accumulation_backend: str,
        noise_power: float,
        noise_power_source: str,
    ) -> None:
        matched_isb_completion = _matched_isb_completion_enabled(monitor)
        projected_isb_completion = _projected_isb_completion_enabled(monitor)
        utd_cross_term_surrogate = _utd_cross_term_surrogate_enabled(monitor)
        diffraction_diagnostics = _merge_diffraction_diagnostic_counts(
            *(sample.get("diffraction_diagnostics") for sample in sample_metadata)
        )
        matched_isotropic_vector_coherent = (
            str(monitor.combine_mode) == "coherent"
            and str(monitor.receiver_model) == "matched_isotropic"
        )
        if projected_isb_completion:
            coherent_diag_definition = (
                "sample-weighted projected path-coefficient sums with shadow-boundary completion "
                "folded into diffraction and total diagnostics while explicit completion-only "
                "diagnostics remain available"
            )
            coherent_power_definition = (
                "sample-weighted squared magnitude of coherent projected sums with shadow-boundary "
                "completion folded into diffraction and total diagnostics"
            )
        elif matched_isb_completion:
            coherent_diag_definition = (
                "sample-weighted projected component diagnostics with ISB-completed diffraction "
                "and total retained alongside matched-isotropic vector coherent path gain"
            )
            coherent_power_definition = (
                "sample-weighted matched-isotropic vector coherent powers per component with "
                "ISB-completed diffraction and total, while explicit completion-only diagnostics "
                "remain separate"
            )
        elif str(monitor.receiver_model) == "projected_polarized":
            coherent_diag_definition = "sample-weighted projected path-coefficient sums"
            coherent_power_definition = "sample-weighted squared magnitude of coherent projected sums"
        elif matched_isotropic_vector_coherent:
            coherent_diag_definition = (
                "sample-weighted projected component diagnostics retained for parity debugging while "
                "path_gain uses matched-isotropic vector coherent accumulation"
            )
            coherent_power_definition = (
                "sample-weighted matched-isotropic vector coherent powers per component and total"
            )
        else:
            coherent_diag_definition = (
                "sample-weighted projected component diagnostics retained for parity debugging while "
                "path_gain uses matched-isotropic power accumulation"
            )
            coherent_power_definition = (
                "sample-weighted squared magnitude of projected coherent diagnostics retained alongside "
                "matched-isotropic incoherent path-gain accumulation"
            )
        metadata = {
        "receiver_sampling": {
            "monitor_name": monitor.name,
            "monitor_kind": monitor.kind,
            "sampling_mode": str(getattr(monitor, "sampling_mode", "deterministic")),
            "surface_mode": grid.surface_mode,
            "axis": grid.axis,
            "position": grid.position,
            "bounds": grid.bounds,
            "center": grid.center,
            "orientation": grid.orientation,
            "size": grid.size,
            "tangential_axes": grid.tangential_axes,
            "grid_shape": grid.grid_shape,
            "cell_size": grid.cell_size,
            "cell_centered": True,
            "quadrature_mode": monitor.quadrature_mode,
            "samples_per_cell": monitor.samples_per_cell,
            "sample_offsets_local": tuple(sample.offset_local for sample in grid.sample_sets),
        },
        "metric_contract": {
            "path_gain": (
                (
                    "squared_norm_of_matched_isotropic_vector_coherent_sum_plus_isb_visibility_"
                    "completion_weighted_by_fixed_cell_quadrature"
                )
                if matched_isb_completion
                else (
                    "squared_norm_of_matched_isotropic_vector_coherent_sum_weighted_by_fixed_cell_quadrature"
                    if matched_isotropic_vector_coherent
                    else (
                        (
                            "squared_norm_of_receiver_model_matched_coherent_sum_plus_projected_isb_"
                            "shadow_completion_weighted_by_fixed_cell_quadrature"
                        )
                        if projected_isb_completion
                        else "squared_norm_of_receiver_model_matched_coherent_sum_weighted_by_fixed_cell_quadrature"
                    )
                    if monitor.combine_mode == "coherent"
                    else (
                        (
                            "sum_vector_component_powers_plus_utd_shadow_boundary_surrogate_cross_terms_"
                            "weighted_by_fixed_cell_quadrature"
                        )
                        if utd_cross_term_surrogate
                        else "sum_vector_component_powers_weighted_by_fixed_cell_quadrature"
                    )
                    if str(monitor.receiver_model) == "matched_isotropic"
                    else "sum_squared_projected_path_amplitudes_weighted_by_fixed_cell_quadrature"
                )
            ),
            "rss": "tx_power_times_path_gain",
            "sinr": "rss_over_noise_plus_other_tx_rss",
        },
        "metric": monitor.metric,
        "combine_mode": monitor.combine_mode,
        "receiver_model": monitor.receiver_model,
        "shadow_boundary_mode": getattr(monitor, "shadow_boundary_mode", "none"),
        "shadow_support_cutoff_db": getattr(monitor, "shadow_support_cutoff_db", None),
        "tx_power": float(monitor.tx_power),
        "noise_power": float(noise_power),
        "noise_power_source": str(noise_power_source),
        "path_counts": path_counts,
        "diffraction_diagnostics": diffraction_diagnostics,
        "sample_evaluations": tuple(sample_metadata),
        "runtime_reuse": {
            "diffraction_state_prep_cache": {
                "mode": str(aggregate_runtime_reuse["cache_mode"]),
                "hits": int(aggregate_runtime_reuse["state_preparation_hits"]),
                "misses": int(aggregate_runtime_reuse["state_preparation_misses"]),
                "state_layout": str(aggregate_runtime_reuse["state_layout"]),
            },
        },
        "accumulation_backend": {
            "requested": _normalize_radio_map_accumulation_backend(radio_map_accumulation_backend),
            "resolved": resolved_accumulation_backend,
            "cell_accumulation_mode": (
                "direct_in_loop_scatter"
                if resolved_accumulation_backend in {"cell_accumulation", "native_monte_carlo"}
                else "sample_reduction"
            ),
        },
        "solver_mode": solver_controls,
        "execution_intent": dict(solver_controls["execution_intent"]),
        "performance_memory": {
            "torch_cuda": capture_cuda_memory_report(),
        },
        "coherent_diagnostics": {
            "definition": coherent_diag_definition,
            "components": ("los", "reflection", "diffraction", "total"),
        },
        "coherent_power_diagnostics": {
            "definition": coherent_power_definition,
            "components": (
                "los",
                "reflection",
                "diffraction",
                "total",
                "raw_diffraction",
                "matched_isb_completion_only",
                "folded_diffraction",
            ),
        },
        "shadow_boundary_surrogate": {
            "mode": _shadow_boundary_mode(monitor),
            "enabled": bool(_shadow_boundary_mode(monitor) != "none"),
            "support_cutoff_db": getattr(monitor, "shadow_support_cutoff_db", None),
            "cross_term_model": (
                "matched_isotropic_vector_inner_product"
                if utd_cross_term_surrogate
                else None
            ),
            "completion_model": (
                "projected_shadow_side_direct_continuation"
                if projected_isb_completion
                else (
                    "matched_isotropic_isb_scene_edge_complex_transition_residual_completion"
                    if matched_isb_completion
                    else None
                )
            ),
            "visibility_model": (
                "scene_edge_incident_transition_weighted_average_with_incident_gated_direct_mode_residual_matching"
                if matched_isb_completion
                else None
            ),
            "diagnostic_components": (
                "utd_surrogate_incident_cross",
                "utd_surrogate_reflection_cross",
                "utd_surrogate_total",
            )
            if utd_cross_term_surrogate
            else (
                (
                    "projected_isb_completion_weight",
                    "projected_isb_completion_deficiency",
                    "projected_isb_continued_direct_power",
                    "projected_isb_amplitude_ratio",
                    "projected_isb_completion",
                    "projected_isb_surrogate_total",
                )
                if projected_isb_completion
                else (
                    (
                        "matched_isb_completion_weight",
                        "matched_isb_sum_incident_weight",
                        "matched_isb_max_incident_weight",
                        "matched_isb_argmax_margin",
                        "matched_isb_support_edge_count",
                        "matched_isb_argmax_edge_idx",
                        "matched_isb_continued_direct_power",
                        "matched_isb_hard_visibility",
                        "matched_isb_transition_magnitude",
                        "matched_isb_transition_phase",
                        "matched_isb_completion",
                        "matched_isb_surrogate_total",
                    )
                    if matched_isb_completion
                    else ()
                )
            ),
            "parameters": (
                {
                    "ratio_target": float(PROJECTED_ISB_COMPLETION_RATIO_TARGET),
                    "completion_gain": float(PROJECTED_ISB_COMPLETION_GAIN),
                }
                if projected_isb_completion
                else (
                    {
                        "shadow_boundary_half_field_limit": float(
                            MATCHED_ISB_COMPLETION_BOUNDARY_HALF_FIELD
                        ),
                        "incident_weight_aggregation": "clamped_sum_incident_weight",
                        "incident_response_aggregation": (
                            "scene_edge_incident_transition_weighted_average_over_sum_incident_weight"
                        ),
                        "max_incident_weight_usage": "diagnostic_only",
                    }
                    if matched_isb_completion
                    else {}
                )
            ),
        },
        }
        if reflection_detail is not None:
            metadata["reflection_sampling"] = dict(reflection_detail.get("reflection_sampling", {}))
            metadata["reflection_backend"] = dict(reflection_detail.get("dda_stats", {}))
        reflection_runtime_backend = {}
        if metadata.get("reflection_backend"):
            reflection_runtime_backend.update(dict(metadata["reflection_backend"]))
        reflection_runtime_backend.update(dict(aggregate_runtime_backends["reflection"]))
        metadata["runtime_backends"] = {
            "reflection": reflection_runtime_backend,
            "diffraction": dict(aggregate_runtime_backends["diffraction"]),
            "suffix": dict(aggregate_runtime_backends["suffix"]),
            "no_diff_fast_path": bool(aggregate_runtime_backends.get("no_diff_fast_path", False)),
            "no_diff_reflection_scheduler": aggregate_runtime_backends.get(
                "no_diff_reflection_scheduler"
            ),
        }
        super().__init__(metadata)


__all__ = [
    "RadioMapMetadata",
    "_count_nonzero_complex",
    "_count_reflection_paths",
    "_empty_diffraction_diagnostic_counts",
    "_merge_diffraction_diagnostic_counts",
    "_radio_map_diffraction_cache_key",
    "_radio_map_diffraction_state_layout",
    "_resolve_noise_power",
]
