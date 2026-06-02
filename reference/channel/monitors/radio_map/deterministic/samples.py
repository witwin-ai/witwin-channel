from __future__ import annotations

import os
import time
from typing import Callable, Mapping

import drjit as dr
import numpy as np
import witwin as wt
from .coherent import (
    accumulate_radio_map_diffraction_coherent,
    accumulate_radio_map_los_coherent,
    accumulate_radio_map_reflection_coherent,
)
from ..grid import AxisAlignedRadioMapNativeGrid
from .cell_accumulation import (
    accumulate_diffraction_matched_isotropic_forward_fast,
    accumulate_diffraction_scalar_power,
    accumulate_reflection_scalar_power,
)
from ..backend import _point_grad_enabled
from ..monitor import RadioMapMonitor
from .scheduler import resolve_radio_map_receiver_tiles
from ..diagnostics import (
    _add_complex,
    _add_complex_vector,
    _add_float,
    _diffraction_anchor_coordinate,
    _gather_positions,
    _remap_raw_rx_index,
    _scale_complex,
    _scale_complex_vector,
    _scale_float,
    _scatter_float,
    _utd_cross_term_surrogate_enabled,
    _vector_power,
    _vector_power_symbolic,
    _zero_float,
)
from ..metadata import (
    _count_nonzero_complex,
    _count_reflection_paths,
    _empty_diffraction_diagnostic_counts,
    _merge_diffraction_diagnostic_counts,
    _radio_map_diffraction_cache_key,
)
from ...common import group_positions_by_z_coordinate
from ...path.collectors import collect_diffraction_state_paths
from ...orchestration import ResolvedTraceConfig
from ....config import ReflectionSuffixConfig
from ....scene import Scene
from ....trace.diffraction.api import _finalize_solver_metadata
from ....trace.diffraction.builders import (
    _build_solver_metadata,
    _prepare_diffraction_state_arrays,
)
from ....trace.diffraction.state import path_export_state_layout
from ....trace.materials import ReflectionTraceDetail
from ....trace.reflection import discover_reflection_paths
from ....utils.drjit_ops import ArrayInit, EvalSync, complex_abs_sqr
from ....utils.polarization import (
    effective_rx_polarization,
    project_real_polarization_to_ray,
    vector_from_scalar_and_real_direction,
)


def _radiomap_retain_diffraction_cold_metadata() -> bool:
    # Path-export-reduced radio-map states still require replay/history fields
    # during deterministic path collection and higher-order path slot
    # materialization. Keep cold metadata until those requirements are split
    # from lineage/audit-only storage.
    return True


def _prepared_state_diagnostic_counts(state_arrays) -> dict[str, int]:
    counts = _empty_diffraction_diagnostic_counts()
    counts["prepared_state_count"] = (
        0 if state_arrays is None else int(state_arrays["n_states"])
    )
    return counts


def _radiomap_diff2_forward_fast_path_supported(
    *,
    tx_pos,
    monitor: RadioMapMonitor,
    solver_controls,
    shadow_support_cutoff_db=None,
) -> bool:
    effective = solver_controls["effective"]
    if str(os.environ.get("WITWIN_RADIOMAP_DIFF2_FORWARD_FAST_PATH", "1")).strip() in {
        "0",
        "false",
        "False",
    }:
        return False
    return (
        int(effective["max_diffractions"]) == 2
        and str(getattr(monitor, "surface_mode", "axis_aligned")) == "axis_aligned"
        and str(monitor.combine_mode) == "coherent"
        and str(monitor.receiver_model) == "matched_isotropic"
        and str(getattr(monitor, "shadow_boundary_mode", "none"))
        in {"none", "matched_isb_completion"}
        and shadow_support_cutoff_db is None
        and not _utd_cross_term_surrogate_enabled(monitor)
        and not _point_grad_enabled(tx_pos)
    )


def _augment_diffraction_runtime_backend(
    runtime_backend,
    planner_stats,
    *,
    forward_fast_path: bool,
):
    runtime_backend["forward_fast_path"] = bool(forward_fast_path)
    runtime_backend["utd_primal_backend"] = planner_stats.get("utd_primal_backend")
    runtime_backend["planner_strategy"] = str(
        planner_stats.get("planner_strategy", planner_stats.get("state_scheduler", "cartesian_chunked"))
    )
    runtime_backend["planner_skip_reason"] = planner_stats.get("planner_skip_reason")
    runtime_backend["pair_chunk_budget"] = int(planner_stats.get("pair_chunk_budget", 0))
    runtime_backend["peak_pair_count_estimate"] = int(
        planner_stats.get(
            "peak_pair_count_estimate",
            planner_stats.get("cartesian_peak_pair_count", 0),
        )
    )
    runtime_backend["full_pair_count"] = int(planner_stats.get("full_pair_count", 0))
    runtime_backend["cartesian_peak_pair_count"] = int(
        planner_stats.get("cartesian_peak_pair_count", 0)
    )
    runtime_backend["tiled_peak_pair_count"] = int(planner_stats.get("tiled_peak_pair_count", 0))
    runtime_backend["estimated_launch_count"] = int(planner_stats.get("estimated_launch_count", 0))


def _baseline_matched_isotropic_reflection_power(
    *,
    sample_positions,
    scene: Scene,
    config: ResolvedTraceConfig,
    reflection_detail: ReflectionTraceDetail,
):
    reflection_payload = accumulate_reflection_scalar_power(
        rx_pos=sample_positions,
        scene=scene,
        wavelength=config.wavelength,
        k=config.k,
        reflection_detail=reflection_detail,
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        receiver_model="matched_isotropic",
        return_vector_coherent=True,
    )
    runtime_backend = {}
    if reflection_detail is not None:
        runtime_backend["radio_map_power_backend"] = "baseline_vector_power_replay"
        runtime_backend["pair_replay_backend"] = "direct_replay_vector_power"
        runtime_backend["path_scheduler"] = str(reflection_payload["planner_stats"]["path_scheduler"])
        runtime_backend["selected_reason"] = str(reflection_payload["planner_stats"]["selected_reason"])
        runtime_backend["estimated_pair_ratio"] = float(
            reflection_payload["planner_stats"]["estimated_pair_ratio"]
        )
        if reflection_payload["planner_stats"]["planner_backend"] is not None:
            runtime_backend["planner_backend"] = str(
                reflection_payload["planner_stats"]["planner_backend"]
            )
    return reflection_payload["power"], reflection_payload["vector_coherent"], runtime_backend


def _baseline_matched_isotropic_diffraction_power(
    *,
    diffraction_raw_collections,
    scene: Scene,
    config: ResolvedTraceConfig,
    n_rx: int,
    receiver_axis: str,
    sample_grid=None,
    grad_preserving_dense_replay: bool = False,
    shadow_support_cutoff_db=None,
    los_reference_vector=None,
    reflection_reference_vector=None,
):
    diffraction_coherent = ArrayInit.complex_zero(n_rx)
    diffraction_power = _zero_float(n_rx)
    diffraction_vector_coherent = None
    incident_cross = _zero_float(n_rx)
    reflection_cross = _zero_float(n_rx)
    runtime_backend = {}
    diagnostic_counts = _empty_diffraction_diagnostic_counts()
    active_rx_polarization = effective_rx_polarization(
        config.rx_polarization,
        config.tx_polarization,
    )
    for raw in diffraction_raw_collections:
        receiver_index_map = raw.get("radio_map_receiver_index_map")
        local_positions = raw.get("rx_positions")
        state_arrays = raw.get("state_arrays")
        local_counts = _merge_diffraction_diagnostic_counts(raw.get("diagnostic_counts"))
        if receiver_index_map is None or local_positions is None or state_arrays is None:
            diagnostic_counts = _merge_diffraction_diagnostic_counts(
                diagnostic_counts,
                local_counts,
            )
            continue
        if grad_preserving_dense_replay:
            edge_data = raw.get("edge_data")
            local_coherent, local_vector, scheduler_decision, _ = (
                accumulate_radio_map_diffraction_coherent(
                    state_arrays=state_arrays,
                    edge_data=edge_data,
                    sample_grid=sample_grid,
                    rx_pos=local_positions,
                    scene=scene,
                    wavelength=config.wavelength,
                    k=config.k,
                    material_detail=config.diffraction_material,
                    suffix=ReflectionSuffixConfig(),
                    tx_polarization=config.tx_polarization,
                    rx_polarization=active_rx_polarization,
                    execution=config.diffraction_execution,
                    return_timing=False,
                    return_vector=True,
                    receiver_axis=str(receiver_axis),
                )
            )
            local_power = _vector_power_symbolic(local_vector)
            local_incident_cross = _zero_float(int(dr.width(local_positions.x)))
            local_reflection_cross = _zero_float(int(dr.width(local_positions.x)))
            local_runtime_backend = {
                "radio_map_power_backend": "symbolic_vector_power_after_dense_accumulation",
                "pair_replay_backend": "disabled_field_style_dense_accumulation",
                "field_accumulation_backend": "accumulate_radio_map_diffraction_coherent",
                "gradient_preserving": True,
                "support_contract": "zeroed_invalid_pairs_after_dense_vector_accumulation",
                "suffix_included": False,
                "state_scheduler": str(scheduler_decision.state_scheduler),
                "selected_reason": str(scheduler_decision.selected_reason),
                "estimated_pair_ratio": float(scheduler_decision.estimated_pair_ratio),
            }
            local_runtime_backend["forward_fast_path"] = False
            local_runtime_backend["utd_primal_backend"] = None
            local_runtime_backend["planner_strategy"] = str(scheduler_decision.planner_strategy)
            local_runtime_backend["planner_skip_reason"] = scheduler_decision.planner_skip_reason
            local_runtime_backend["pair_chunk_budget"] = int(
                scheduler_decision.pair_chunk_budget or 0
            )
            local_runtime_backend["peak_pair_count_estimate"] = int(
                scheduler_decision.peak_pair_count_estimate
            )
            if scheduler_decision.planner_backend is not None:
                local_runtime_backend["planner_backend"] = str(
                    scheduler_decision.planner_backend
                )
        else:
            local_los_reference_vector = (
                None
                if los_reference_vector is None
                else {
                    axis: dr.gather(wt.Complex2f, los_reference_vector[axis], receiver_index_map)
                    for axis in ("x", "y", "z")
                }
            )
            local_reflection_reference_vector = (
                None
                if reflection_reference_vector is None
                else {
                    axis: dr.gather(
                        wt.Complex2f,
                        reflection_reference_vector[axis],
                        receiver_index_map,
                    )
                    for axis in ("x", "y", "z")
                }
            )
            diffraction_payload = accumulate_diffraction_scalar_power(
                state_arrays=state_arrays,
                rx_pos=local_positions,
                scene=scene,
                wavelength=config.wavelength,
                k=config.k,
                material_detail=config.diffraction_material,
                tx_polarization=config.tx_polarization,
                rx_polarization=config.rx_polarization,
                receiver_model="matched_isotropic",
                receiver_axis=str(receiver_axis),
                return_vector_coherent=True,
                incident_reference_vector=local_los_reference_vector,
                reflection_reference_vector=local_reflection_reference_vector,
                shadow_support_cutoff_db=shadow_support_cutoff_db,
            )
            local_coherent = diffraction_payload["coherent"]
            local_vector = diffraction_payload["vector_coherent"]
            local_power = diffraction_payload["power"]
            local_incident_cross = diffraction_payload["incident_cross"]
            local_reflection_cross = diffraction_payload["reflection_cross"]
            local_runtime_backend = {
                "radio_map_power_backend": "baseline_vector_power_replay",
                "pair_replay_backend": str(
                    diffraction_payload["planner_stats"].get(
                        "scalar_backend",
                        "direct_state_vector_power",
                    )
                ),
                "state_scheduler": str(diffraction_payload["planner_stats"]["state_scheduler"]),
                "selected_reason": str(diffraction_payload["planner_stats"]["selected_reason"]),
                "estimated_pair_ratio": float(
                    diffraction_payload["planner_stats"]["estimated_pair_ratio"]
                ),
                "suffix_included": False,
            }
            _augment_diffraction_runtime_backend(
                local_runtime_backend,
                diffraction_payload["planner_stats"],
                forward_fast_path=bool(
                    diffraction_payload["planner_stats"].get("forward_fast_path", False)
                ),
            )
            local_counts = _merge_diffraction_diagnostic_counts(
                local_counts,
                diffraction_payload.get("diagnostic_counts"),
            )
            if diffraction_payload["planner_stats"]["planner_backend"] is not None:
                local_runtime_backend["planner_backend"] = str(
                    diffraction_payload["planner_stats"]["planner_backend"]
                )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            diffraction_coherent.real,
            local_coherent.real,
            receiver_index_map,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            diffraction_coherent.imag,
            local_coherent.imag,
            receiver_index_map,
        )
        if local_vector is not None:
            if diffraction_vector_coherent is None:
                diffraction_vector_coherent = {axis: ArrayInit.complex_zero(n_rx) for axis in ("x", "y", "z")}
            for axis in ("x", "y", "z"):
                dr.scatter_reduce(
                    dr.ReduceOp.Add,
                    diffraction_vector_coherent[axis].real,
                    local_vector[axis].real,
                    receiver_index_map,
                )
                dr.scatter_reduce(
                    dr.ReduceOp.Add,
                    diffraction_vector_coherent[axis].imag,
                    local_vector[axis].imag,
                    receiver_index_map,
                )
        _scatter_float(diffraction_power, local_power, receiver_index_map)
        _scatter_float(incident_cross, local_incident_cross, receiver_index_map)
        _scatter_float(reflection_cross, local_reflection_cross, receiver_index_map)
        diagnostic_counts = _merge_diffraction_diagnostic_counts(
            diagnostic_counts,
            local_counts,
        )
        runtime_backend = dict(local_runtime_backend)
    runtime_backend.setdefault("forward_fast_path", False)
    runtime_backend.setdefault("utd_primal_backend", None)
    runtime_backend.setdefault("planner_strategy", "cartesian_chunked")
    runtime_backend.setdefault("planner_skip_reason", None)
    runtime_backend.setdefault("pair_chunk_budget", 0)
    runtime_backend.setdefault("peak_pair_count_estimate", 0)
    runtime_backend["diagnostic_counts"] = diagnostic_counts
    return (
        diffraction_coherent,
        diffraction_power,
        incident_cross,
        reflection_cross,
        diffraction_vector_coherent,
        runtime_backend,
    )


def _trace_baseline_matched_isotropic_diffraction_fast(
    *,
    sample_positions,
    sample_grid=None,
    tx_pos,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls,
    monitor: RadioMapMonitor,
    reflection_detail,
    persistent_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None,
    local_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None,
    diffraction_state_cache_key_fn: Callable[[float, object], tuple[object, ...]] | None,
    state_layout: str,
):
    effective = solver_controls["effective"]
    mixed_reflection_detail = reflection_detail if config.enable_rd_diffraction else None
    reflection_n_rays = effective["reflection_n_rays"] if config.enable_rd_diffraction else 0
    reflection_max_bounces = (
        effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0
    )
    selected_diffraction_state_cache = local_diffraction_state_cache
    cache_mode = "disabled"
    if mixed_reflection_detail is None and persistent_diffraction_state_cache is not None:
        selected_diffraction_state_cache = persistent_diffraction_state_cache
        cache_mode = "persistent"
    elif local_diffraction_state_cache is not None:
        cache_mode = "local"

    n_rx = int(dr.width(sample_positions.x))
    diffraction_coherent = ArrayInit.complex_zero(n_rx)
    diffraction_power = _zero_float(n_rx)
    diffraction_vector_coherent = {axis: ArrayInit.complex_zero(n_rx) for axis in ("x", "y", "z")}
    diagnostic_counts = _empty_diffraction_diagnostic_counts()
    runtime_backend = {
        "radio_map_power_backend": "baseline_vector_power_replay",
        "pair_replay_backend": "native_radiomap_vector_power_forward_fast",
        "suffix_included": False,
        "forward_fast_path": True,
    }
    group_metadata = []
    cache_hits = 0
    cache_misses = 0
    path_count = 0

    for receiver_z, group_indices in group_positions_by_z_coordinate(sample_positions):
        group_positions = _gather_positions(sample_positions, group_indices)
        cache_key = (
            None
            if diffraction_state_cache_key_fn is None
            else _radio_map_diffraction_cache_key(
                diffraction_state_cache_key_fn(receiver_z, mixed_reflection_detail),
                state_layout=state_layout,
            )
        )
        prepared_state_group = (
            None
            if selected_diffraction_state_cache is None or cache_key is None
            else selected_diffraction_state_cache.get(cache_key)
        )
        cache_hit = prepared_state_group is not None
        if cache_hit:
            cache_hits += 1
        else:
            prepared_state_group = _prepare_diffraction_state_arrays(
                tx_pos,
                receiver_z,
                scene,
                config.wavelength,
                config.k,
                mixed_reflection_detail,
                config.diffraction_material,
                reflection_n_rays,
                reflection_max_bounces,
                config.reflection_coef,
                monitor.ray_mode,
                effective["max_diffractions"],
                total_state_budget_per_order=effective["diffraction_state_budget"],
                inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
                max_inserted_reflections_per_path=effective["max_inserted_reflections_per_path"],
                retain_cold_metadata=_radiomap_retain_diffraction_cold_metadata(),
                use_scene_materials=config.use_scene_materials_for_diffraction,
                tx_polarization=config.tx_polarization,
                solver_mode=solver_controls["selected"],
                memory_profile=effective["memory_profile"],
                state_layout=state_layout,
            )
            if selected_diffraction_state_cache is not None and cache_key is not None:
                selected_diffraction_state_cache[cache_key] = prepared_state_group
                cache_misses += 1
        edge_cache, edge_data, state_arrays, path_budget_report = prepared_state_group or (
            None,
            None,
            None,
            None,
        )
        receiver_index_map = wt.UInt32(group_indices.astype(np.uint32, copy=False))
        prepared_counts = _prepared_state_diagnostic_counts(state_arrays)
        solver_metadata = _finalize_solver_metadata(
            _build_solver_metadata(
                scene=scene,
                max_diffractions=effective["max_diffractions"],
                reflection_detail=mixed_reflection_detail,
                reflection_n_rays=reflection_n_rays,
                reflection_max_bounces=reflection_max_bounces,
                reflected_suffix_enabled=False,
                inserted_reflection_enabled=(
                    config.enable_rd_diffraction
                    and reflection_n_rays > 0
                    and reflection_max_bounces > 0
                    and effective["max_diffractions"] > 1
                    and (effective["max_inserted_reflections_per_path"] or 0) > 0
                ),
                max_inserted_reflections_per_path=(
                    0
                    if effective["max_inserted_reflections_per_path"] is None
                    else int(effective["max_inserted_reflections_per_path"])
                ),
                total_state_budget_per_order=effective["diffraction_state_budget"],
                inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
                path_budget_report=path_budget_report,
            ),
            scene=scene,
            material_detail=config.diffraction_material,
            use_scene_materials=config.use_scene_materials_for_diffraction,
            execution=config.diffraction_execution,
            rx_polarization=config.rx_polarization,
            active_rx_polarization=effective_rx_polarization(
                config.rx_polarization,
                config.tx_polarization,
            ),
            receiver_axis=monitor.axis if monitor.axis is not None else "z",
        )
        if state_arrays is None or int(state_arrays["n_states"]) <= 0:
            local_counts = prepared_counts
            planner_stats = {
                "state_scheduler": "empty",
                "planner_strategy": "empty",
                "scalar_backend": "native_radiomap_vector_power_forward_fast",
                "planner_backend": None,
                "planner_skip_reason": "no_diffraction_states",
                "selected_reason": "no_diffraction_states",
                "tile_task_count": 0,
                "estimated_pair_count": 0,
                "full_pair_count": 0,
                "estimated_pair_ratio": 0.0,
                "pair_chunk_budget": 0,
                "cartesian_peak_pair_count": 0,
                "tiled_peak_pair_count": 0,
                "peak_pair_count_estimate": 0,
                "estimated_launch_count": 0,
                "forward_fast_path": True,
                "utd_primal_backend": None,
            }
        else:
            group_receiver_tiles = resolve_radio_map_receiver_tiles(
                grid=sample_grid,
                receiver_positions=group_positions,
            )
            diffraction_payload = accumulate_diffraction_matched_isotropic_forward_fast(
                state_arrays=state_arrays,
                rx_pos=group_positions,
                scene=scene,
                wavelength=config.wavelength,
                k=config.k,
                material_detail=config.diffraction_material,
                tx_polarization=config.tx_polarization,
                rx_polarization=config.rx_polarization,
                receiver_axis=monitor.axis if monitor.axis is not None else "z",
                receiver_tiles=group_receiver_tiles,
                shadow_support_cutoff_db=getattr(monitor, "shadow_support_cutoff_db", None),
            )
            local_counts = _merge_diffraction_diagnostic_counts(
                prepared_counts,
                diffraction_payload.get("diagnostic_counts"),
            )
            local_coherent = diffraction_payload["coherent"]
            local_vector = diffraction_payload["vector_coherent"]
            local_power = diffraction_payload["power"]
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                diffraction_coherent.real,
                local_coherent.real,
                receiver_index_map,
            )
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                diffraction_coherent.imag,
                local_coherent.imag,
                receiver_index_map,
            )
            for axis in ("x", "y", "z"):
                dr.scatter_reduce(
                    dr.ReduceOp.Add,
                    diffraction_vector_coherent[axis].real,
                    local_vector[axis].real,
                    receiver_index_map,
                )
                dr.scatter_reduce(
                    dr.ReduceOp.Add,
                    diffraction_vector_coherent[axis].imag,
                    local_vector[axis].imag,
                    receiver_index_map,
                )
            _scatter_float(diffraction_power, local_power, receiver_index_map)
            path_count += int(diffraction_payload["path_count"])
            planner_stats = dict(diffraction_payload["planner_stats"])
        diagnostic_counts = _merge_diffraction_diagnostic_counts(
            diagnostic_counts,
            local_counts,
        )
        _augment_diffraction_runtime_backend(
            runtime_backend,
            planner_stats,
            forward_fast_path=True,
        )
        runtime_backend["state_scheduler"] = str(planner_stats["state_scheduler"])
        runtime_backend["selected_reason"] = str(planner_stats["selected_reason"])
        runtime_backend["estimated_pair_ratio"] = float(planner_stats["estimated_pair_ratio"])
        if planner_stats["planner_backend"] is not None:
            runtime_backend["planner_backend"] = str(planner_stats["planner_backend"])
        group_metadata.append(
            {
                "receiver_z": float(receiver_z),
                "receiver_count": int(group_indices.shape[0]),
                "cache_hit": bool(cache_hit),
                "n_states": 0 if state_arrays is None else int(state_arrays["n_states"]),
                "state_layout": path_export_state_layout(state_arrays) or str(state_layout),
                "state_layout_requested": str(state_layout),
                "diagnostic_counts": dict(local_counts),
                "planner_stats": dict(planner_stats),
                "solver_metadata": solver_metadata,
                "edge_data_present": edge_data is not None or edge_cache is not None,
            }
        )

    runtime_backend["diagnostic_counts"] = diagnostic_counts
    return {
        "diffraction_coherent": diffraction_coherent,
        "diffraction_power": diffraction_power,
        "diffraction_vector_coherent": diffraction_vector_coherent,
        "incident_cross": _zero_float(n_rx),
        "reflection_cross": _zero_float(n_rx),
        "runtime_backend": runtime_backend,
        "diffraction_group_metadata": tuple(group_metadata),
        "runtime_reuse": {
            "cache_mode": cache_mode,
            "state_preparation_hits": int(cache_hits),
            "state_preparation_misses": int(cache_misses),
            "state_layout": (
                str(group_metadata[0]["state_layout"])
                if len(group_metadata) > 0
                else str(state_layout)
            ),
        },
        "path_count": int(path_count),
        "diagnostic_counts": diagnostic_counts,
    }


def _discover_radio_map_reflection_detail(
    *,
    sample_grid,
    tx_pos,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls,
    monitor: RadioMapMonitor,
    reflection_detail,
):
    if reflection_detail is not None:
        return reflection_detail
    effective = solver_controls["effective"]
    if effective["reflection_n_rays"] <= 0 or effective["reflection_max_bounces"] <= 0:
        return reflection_detail
    return discover_reflection_paths(
        tx_pos=tx_pos,
        scene=scene,
        wavelength=config.wavelength,
        k=config.k,
        n_rays=effective["reflection_n_rays"],
        max_reflections=effective["reflection_max_bounces"],
        mode=monitor.ray_mode,
        reflection_coef=config.reflection_coef,
        ray_sampling="full_sphere" if monitor.ray_mode == "3d" else "circle",
        tx_polarization=config.tx_polarization,
        reflection_relative_permittivity=config.reflection_relative_permittivity,
        reflection_conductivity=config.reflection_conductivity,
        reflection_material=config.reflection_material,
        use_scene_materials=config.use_scene_materials_for_reflection,
        sampling_axis=sample_grid.axis,
        sampling_plane_position=sample_grid.position,
        sampling_bounds=sample_grid.bounds,
    )


def _trace_diffraction_raw_collections(
    *,
    sample_positions,
    tx_pos,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls,
    monitor: RadioMapMonitor,
    reflection_detail,
    persistent_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None,
    local_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None,
    diffraction_state_cache_key_fn: Callable[[float, object], tuple[object, ...]] | None,
    state_layout: str,
    preserve_higher_order_candidate_topology: bool = False,
):
    effective = solver_controls["effective"]
    if effective["max_diffractions"] <= 0:
        return [], (), {
            "cache_mode": "disabled",
            "state_preparation_hits": 0,
            "state_preparation_misses": 0,
            "state_layout": str(state_layout),
        }

    mixed_reflection_detail = reflection_detail if config.enable_rd_diffraction else None
    reflection_n_rays = effective["reflection_n_rays"] if config.enable_rd_diffraction else 0
    reflection_max_bounces = (
        effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0
    )
    selected_diffraction_state_cache = local_diffraction_state_cache
    raw_collections = []
    group_metadata = []
    cache_hits = 0
    cache_misses = 0
    cache_mode = "disabled"
    if mixed_reflection_detail is None and persistent_diffraction_state_cache is not None:
        selected_diffraction_state_cache = persistent_diffraction_state_cache
        cache_mode = "persistent"
    elif local_diffraction_state_cache is not None:
        cache_mode = "local"

    for receiver_z, group_indices in group_positions_by_z_coordinate(sample_positions):
        group_positions = _gather_positions(sample_positions, group_indices)
        path_collection_stats = {}
        cache_key = (
            None
            if diffraction_state_cache_key_fn is None
            else _radio_map_diffraction_cache_key(
                diffraction_state_cache_key_fn(receiver_z, mixed_reflection_detail),
                state_layout=state_layout,
            )
        )
        prepared_state_group = (
            None
            if selected_diffraction_state_cache is None or cache_key is None
            else selected_diffraction_state_cache.get(cache_key)
        )
        cache_hit = prepared_state_group is not None
        if cache_hit:
            cache_hits += 1
        else:
            prepared_state_group = _prepare_diffraction_state_arrays(
                tx_pos,
                receiver_z,
                scene,
                config.wavelength,
                config.k,
                mixed_reflection_detail,
                config.diffraction_material,
                reflection_n_rays,
                reflection_max_bounces,
                config.reflection_coef,
                monitor.ray_mode,
                effective["max_diffractions"],
                total_state_budget_per_order=effective["diffraction_state_budget"],
                inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
                max_inserted_reflections_per_path=effective["max_inserted_reflections_per_path"],
                retain_cold_metadata=_radiomap_retain_diffraction_cold_metadata(),
                use_scene_materials=config.use_scene_materials_for_diffraction,
                tx_polarization=config.tx_polarization,
                solver_mode=solver_controls["selected"],
                memory_profile=effective["memory_profile"],
                state_layout=state_layout,
                preserve_higher_order_candidate_topology=bool(
                    preserve_higher_order_candidate_topology
                ),
            )
            if selected_diffraction_state_cache is not None and cache_key is not None:
                selected_diffraction_state_cache[cache_key] = prepared_state_group
                cache_misses += 1
        edge_cache, edge_data, state_arrays, path_budget_report = prepared_state_group
        raw = collect_diffraction_state_paths(
            state_arrays=state_arrays,
            edge_data=edge_data if edge_data is not None else edge_cache.get("edge_data"),
            scene=scene,
            rx_positions=group_positions,
            tx_pos=tx_pos,
            wavelength=config.wavelength,
            k=config.k,
            tx_polarization=config.tx_polarization,
            rx_polarization=config.rx_polarization,
            material_detail=config.diffraction_material,
            return_geometry=False,
            ignore_emitter_structure_visibility=True,
            stats=path_collection_stats,
        )
        raw["radio_map_receiver_index_map"] = wt.UInt32(group_indices.astype(np.uint32, copy=False))
        raw["diagnostic_counts"] = _prepared_state_diagnostic_counts(state_arrays)
        _remap_raw_rx_index(raw, group_indices)
        raw_collections.append(raw)
        solver_metadata = _finalize_solver_metadata(
            _build_solver_metadata(
                scene=scene,
                max_diffractions=effective["max_diffractions"],
                reflection_detail=mixed_reflection_detail,
                reflection_n_rays=reflection_n_rays,
                reflection_max_bounces=reflection_max_bounces,
                reflected_suffix_enabled=False,
                inserted_reflection_enabled=(
                    config.enable_rd_diffraction
                    and reflection_n_rays > 0
                    and reflection_max_bounces > 0
                    and effective["max_diffractions"] > 1
                    and (effective["max_inserted_reflections_per_path"] or 0) > 0
                ),
                max_inserted_reflections_per_path=(
                    0
                    if effective["max_inserted_reflections_per_path"] is None
                    else int(effective["max_inserted_reflections_per_path"])
                ),
                total_state_budget_per_order=effective["diffraction_state_budget"],
                inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
                path_budget_report=path_budget_report,
            ),
            scene=scene,
            material_detail=config.diffraction_material,
            use_scene_materials=config.use_scene_materials_for_diffraction,
            execution=config.diffraction_execution,
            rx_polarization=config.rx_polarization,
            active_rx_polarization=effective_rx_polarization(
                config.rx_polarization,
                config.tx_polarization,
            ),
            receiver_axis=monitor.axis if monitor.axis is not None else "z",
        )
        group_metadata.append(
            {
                "receiver_z": float(receiver_z),
                "receiver_count": int(group_indices.shape[0]),
                "cache_hit": bool(cache_hit),
                "n_states": 0 if state_arrays is None else int(state_arrays["n_states"]),
                "state_layout": path_export_state_layout(state_arrays) or str(state_layout),
                "state_layout_requested": str(state_layout),
                "diagnostic_counts": _prepared_state_diagnostic_counts(state_arrays),
                "path_collection": dict(path_collection_stats),
                "solver_metadata": solver_metadata,
            }
        )
    return raw_collections, tuple(group_metadata), {
        "cache_mode": cache_mode,
        "state_preparation_hits": int(cache_hits),
        "state_preparation_misses": int(cache_misses),
        "state_layout": (
            str(group_metadata[0]["state_layout"])
            if len(group_metadata) > 0
            else str(state_layout)
        ),
    }


def _trace_native_coherent_sample(
    *,
    grid,
    sample_set,
    tx_pos,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls,
    monitor: RadioMapMonitor,
    reflection_detail,
    persistent_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None,
    local_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None,
    diffraction_state_cache_key_fn: Callable[[float, object], tuple[object, ...]] | None,
    state_layout: str,
    return_timing: bool,
):
    effective = solver_controls["effective"]
    sample_grid = AxisAlignedRadioMapNativeGrid.from_grid(grid, sample_index=sample_set.index)
    sample_positions = sample_grid.receivers
    sample_timing = {"state_preparation_seconds": 0.0} if return_timing else None
    matched_isotropic_vector_coherent = (
        str(monitor.combine_mode) == "coherent"
        and str(monitor.receiver_model) == "matched_isotropic"
    )

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    los_coherent = accumulate_radio_map_los_coherent(
        scene=scene,
        rx_pos=sample_positions,
        tx_pos=tx_pos,
        wavelength=config.wavelength,
        k=config.k,
    )
    los_field_vector = None
    if matched_isotropic_vector_coherent:
        ray_dir = sample_positions - tx_pos
        tx_pol_dir = project_real_polarization_to_ray(config.tx_polarization, ray_dir)
        los_field_vector = vector_from_scalar_and_real_direction(los_coherent, tx_pol_dir)
        los_power = _vector_power(los_field_vector)
    else:
        los_power = complex_abs_sqr(los_coherent)
    if return_timing:
        sample_timing["los_seconds"] = time.perf_counter() - t0

    reflection_coherent, reflection_vector_coherent, reflection_detail, reflection_seconds = accumulate_radio_map_reflection_coherent(
        sample_grid=sample_grid,
        tx_pos=tx_pos,
        scene=scene,
        wavelength=config.wavelength,
        k=config.k,
        reflection_n_rays=effective["reflection_n_rays"],
        reflection_max_bounces=effective["reflection_max_bounces"],
        ray_mode=monitor.ray_mode,
        reflection_coef=config.reflection_coef,
        min_ray_contribution_threshold=config.min_ray_contribution_threshold,
        reflection_field_backend=config.reflection_field_backend,
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        reflection_relative_permittivity=config.reflection_relative_permittivity,
        reflection_conductivity=config.reflection_conductivity,
        reflection_material=config.reflection_material,
        use_scene_materials=config.use_scene_materials_for_reflection,
        reflection_detail=reflection_detail,
        return_timing=return_timing,
        return_vector=matched_isotropic_vector_coherent,
    )
    reflection_power = (
        _zero_float(int(sample_grid.n_cells))
        if reflection_vector_coherent is None and matched_isotropic_vector_coherent
        else (
            _vector_power(reflection_vector_coherent)
            if matched_isotropic_vector_coherent
            else complex_abs_sqr(reflection_coherent)
        )
    )
    if return_timing:
        sample_timing["reflection_seconds"] = float(reflection_seconds)

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    mixed_reflection_detail = reflection_detail if config.enable_rd_diffraction else None
    reflection_n_rays = effective["reflection_n_rays"] if config.enable_rd_diffraction else 0
    reflection_max_bounces = (
        effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0
    )
    selected_diffraction_state_cache = local_diffraction_state_cache
    cache_mode = "disabled"
    if mixed_reflection_detail is None and persistent_diffraction_state_cache is not None:
        selected_diffraction_state_cache = persistent_diffraction_state_cache
        cache_mode = "persistent"
    elif local_diffraction_state_cache is not None:
        cache_mode = "local"
    anchor_coordinate = _diffraction_anchor_coordinate(sample_grid.axis, tx_pos, sample_grid.position)
    cache_key = (
        None
        if diffraction_state_cache_key_fn is None
        else _radio_map_diffraction_cache_key(
            diffraction_state_cache_key_fn(anchor_coordinate, mixed_reflection_detail),
            state_layout=state_layout,
        )
    )
    prepared_state_group = (
        None
        if selected_diffraction_state_cache is None or cache_key is None
        else selected_diffraction_state_cache.get(cache_key)
    )
    cache_hit = prepared_state_group is not None
    cache_miss = 0
    if effective["max_diffractions"] > 0:
        if return_timing:
            EvalSync.sync()
            t0 = time.perf_counter()
        if not cache_hit:
            prepared_state_group = _prepare_diffraction_state_arrays(
                tx_pos,
                anchor_coordinate,
                scene,
                config.wavelength,
                config.k,
                mixed_reflection_detail,
                config.diffraction_material,
                reflection_n_rays,
                reflection_max_bounces,
                config.reflection_coef,
                monitor.ray_mode,
                effective["max_diffractions"],
                total_state_budget_per_order=effective["diffraction_state_budget"],
                inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
                max_inserted_reflections_per_path=effective["max_inserted_reflections_per_path"],
                retain_cold_metadata=_radiomap_retain_diffraction_cold_metadata(),
                use_scene_materials=config.use_scene_materials_for_diffraction,
                tx_polarization=config.tx_polarization,
                solver_mode=solver_controls["selected"],
                memory_profile=effective["memory_profile"],
                state_layout=state_layout,
            )
            if selected_diffraction_state_cache is not None and cache_key is not None:
                selected_diffraction_state_cache[cache_key] = prepared_state_group
                cache_miss = 1
        if return_timing:
            sample_timing["state_preparation_seconds"] = time.perf_counter() - t0
    edge_cache, edge_data, state_arrays, path_budget_report = prepared_state_group or (
        None,
        None,
        None,
        None,
    )
    diffraction_diagnostics = _prepared_state_diagnostic_counts(state_arrays)
    diffraction_group_metadata = ()
    diffraction_coherent = ArrayInit.complex_zero(sample_grid.n_cells)
    diffraction_vector_coherent = None
    solver_metadata = _finalize_solver_metadata(
        _build_solver_metadata(
            scene=scene,
            max_diffractions=effective["max_diffractions"],
            reflection_detail=mixed_reflection_detail,
            reflection_n_rays=reflection_n_rays,
            reflection_max_bounces=reflection_max_bounces,
            reflected_suffix_enabled=(reflection_n_rays > 0 and reflection_max_bounces > 0),
            inserted_reflection_enabled=(
                config.enable_rd_diffraction
                and reflection_n_rays > 0
                and reflection_max_bounces > 0
                and effective["max_diffractions"] > 1
                and (effective["max_inserted_reflections_per_path"] or 0) > 0
            ),
            max_inserted_reflections_per_path=(
                0
                if effective["max_inserted_reflections_per_path"] is None
                else int(effective["max_inserted_reflections_per_path"])
            ),
            total_state_budget_per_order=effective["diffraction_state_budget"],
            inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
            path_budget_report=path_budget_report,
        ),
        scene=scene,
        material_detail=config.diffraction_material,
        use_scene_materials=config.use_scene_materials_for_diffraction,
        execution=config.diffraction_execution,
        rx_polarization=config.rx_polarization,
        active_rx_polarization=effective_rx_polarization(
            config.rx_polarization,
            config.tx_polarization,
        ),
        receiver_axis=sample_grid.axis,
    )
    if edge_data is not None and state_arrays is not None and int(state_arrays["n_states"]) > 0:
        suffix = ReflectionSuffixConfig(
            n_rays=reflection_n_rays,
            max_bounces=reflection_max_bounces,
            coef=config.reflection_coef,
            mode=monitor.ray_mode,
            detail=mixed_reflection_detail,
            grid=sample_grid,
            grid_data=sample_grid.get_coordinates(),
        )
        if return_timing:
            diffraction_coherent, diffraction_vector_coherent, scheduler_decision, scalar_timing = accumulate_radio_map_diffraction_coherent(
                state_arrays=state_arrays,
                edge_data=edge_data,
                sample_grid=sample_grid,
                rx_pos=sample_positions,
                scene=scene,
                wavelength=config.wavelength,
                k=config.k,
                material_detail=config.diffraction_material,
                suffix=suffix,
                tx_polarization=config.tx_polarization,
                rx_polarization=effective_rx_polarization(
                    config.rx_polarization,
                    config.tx_polarization,
                ),
                execution=config.diffraction_execution,
                return_timing=True,
                return_vector=matched_isotropic_vector_coherent,
            )
            sample_timing["diffraction_seconds"] = (
                float(scalar_timing["utd_accumulation_seconds"])
                + float(scalar_timing["suffix_seconds"])
                + float(scalar_timing["postprocess_seconds"])
            )
        else:
            diffraction_coherent, diffraction_vector_coherent, scheduler_decision, _ = accumulate_radio_map_diffraction_coherent(
                state_arrays=state_arrays,
                edge_data=edge_data,
                sample_grid=sample_grid,
                rx_pos=sample_positions,
                scene=scene,
                wavelength=config.wavelength,
                k=config.k,
                material_detail=config.diffraction_material,
                suffix=suffix,
                tx_polarization=config.tx_polarization,
                rx_polarization=effective_rx_polarization(
                    config.rx_polarization,
                    config.tx_polarization,
                ),
                execution=config.diffraction_execution,
                return_timing=False,
                return_vector=matched_isotropic_vector_coherent,
            )
        diffraction_group_metadata = (
            {
                "receiver_anchor_coordinate": float(anchor_coordinate),
                "receiver_count": int(sample_grid.n_cells),
                "cache_hit": bool(cache_hit),
                "n_states": int(state_arrays["n_states"]),
                "state_layout": path_export_state_layout(state_arrays) or str(state_layout),
                "state_layout_requested": str(state_layout),
                "diagnostic_counts": dict(diffraction_diagnostics),
                "solver_metadata": solver_metadata,
                "scheduler": {
                    "state_scheduler": str(scheduler_decision.state_scheduler),
                    "planner_backend": scheduler_decision.planner_backend,
                    "selected_reason": str(scheduler_decision.selected_reason),
                    "tile_task_count": int(scheduler_decision.tile_task_count),
                    "estimated_pair_count": int(scheduler_decision.estimated_pair_count),
                    "full_pair_count": int(scheduler_decision.full_pair_count),
                    "estimated_pair_ratio": float(scheduler_decision.estimated_pair_ratio),
                    "cartesian_peak_pair_count": int(
                        scheduler_decision.cartesian_peak_pair_count
                    ),
                    "tiled_peak_pair_count": int(
                        scheduler_decision.tiled_peak_pair_count
                    ),
                    "estimated_launch_count": int(
                        scheduler_decision.estimated_launch_count
                    ),
                },
            },
        )
    elif return_timing:
        sample_timing["diffraction_seconds"] = 0.0
    diffraction_power = (
        _zero_float(int(sample_grid.n_cells))
        if diffraction_vector_coherent is None and matched_isotropic_vector_coherent
        else (
            _vector_power(diffraction_vector_coherent)
            if matched_isotropic_vector_coherent
            else complex_abs_sqr(diffraction_coherent)
        )
    )

    runtime_reuse = {
        "cache_mode": cache_mode,
        "state_preparation_hits": int(1 if cache_hit else 0),
        "state_preparation_misses": int(cache_miss),
        "state_layout": path_export_state_layout(state_arrays) or str(state_layout),
    }
    runtime_backends = {
        "reflection": (
            {}
            if reflection_detail is None
            else dict(reflection_detail.get("dda_stats", {}))
        ),
        "diffraction": dict(solver_metadata.get("diffraction_accumulation_backend", {})),
        "suffix": dict(solver_metadata.get("reflection_suffix_backend", {})),
    }
    runtime_backends["diffraction"]["diagnostic_counts"] = dict(diffraction_diagnostics)
    if len(diffraction_group_metadata) > 0:
        runtime_backends["diffraction"]["state_scheduler"] = str(
            diffraction_group_metadata[0]["scheduler"]["state_scheduler"]
        )
        runtime_backends["diffraction"]["selected_reason"] = str(
            diffraction_group_metadata[0]["scheduler"]["selected_reason"]
        )
        runtime_backends["diffraction"]["estimated_pair_ratio"] = float(
            diffraction_group_metadata[0]["scheduler"]["estimated_pair_ratio"]
        )
        runtime_backends["diffraction"]["cartesian_peak_pair_count"] = int(
            diffraction_group_metadata[0]["scheduler"]["cartesian_peak_pair_count"]
        )
        runtime_backends["diffraction"]["tiled_peak_pair_count"] = int(
            diffraction_group_metadata[0]["scheduler"]["tiled_peak_pair_count"]
        )
        runtime_backends["diffraction"]["estimated_launch_count"] = int(
            diffraction_group_metadata[0]["scheduler"]["estimated_launch_count"]
        )
        if diffraction_group_metadata[0]["scheduler"]["planner_backend"] is not None:
            runtime_backends["diffraction"]["planner_backend"] = str(
                diffraction_group_metadata[0]["scheduler"]["planner_backend"]
            )
    path_counts = {
        "los": _count_nonzero_complex(los_coherent),
        "reflection": _count_reflection_paths(reflection_detail),
        "diffraction": 0 if state_arrays is None else int(state_arrays["n_states"]),
    }
    path_counts["total"] = int(path_counts["los"] + path_counts["reflection"] + path_counts["diffraction"])
    return {
        "los_coherent": los_coherent,
        "los_power": los_power,
        "los_field_vector": los_field_vector,
        "reflection_coherent": reflection_coherent,
        "reflection_power": reflection_power,
        "reflection_vector_coherent": reflection_vector_coherent,
        "diffraction_coherent": diffraction_coherent,
        "diffraction_power": diffraction_power,
        "diffraction_vector_coherent": diffraction_vector_coherent,
        "reflection_detail": reflection_detail,
        "diffraction_group_metadata": diffraction_group_metadata,
        "runtime_reuse": runtime_reuse,
        "path_counts": path_counts,
        "diffraction_diagnostics": diffraction_diagnostics,
        "timing": sample_timing,
        "runtime_backends": runtime_backends,
    }


def _trace_cell_accumulation_sample(
    *,
    grid,
    sample_set,
    tx_pos,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls,
    monitor: RadioMapMonitor,
    reflection_detail,
    persistent_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None,
    local_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None,
    diffraction_state_cache_key_fn: Callable[[float, object], tuple[object, ...]] | None,
    state_layout: str,
    diagnostics_targets=None,
    sample_weight: float = 1.0,
    return_timing: bool = False,
):
    effective = solver_controls["effective"]
    sample_grid = AxisAlignedRadioMapNativeGrid.from_grid(grid, sample_index=sample_set.index)
    sample_positions = sample_grid.receivers
    receiver_tiles = resolve_radio_map_receiver_tiles(
        grid=sample_grid,
        receiver_positions=sample_positions,
    )
    sample_timing = {"state_preparation_seconds": 0.0} if return_timing else None
    matched_isotropic_vector_coherent = (
        str(monitor.combine_mode) == "coherent"
        and str(monitor.receiver_model) == "matched_isotropic"
    )
    use_no_diff_fast_path = (
        matched_isotropic_vector_coherent and int(effective["max_diffractions"]) <= 0
    )
    use_diff2_forward_fast_path = (
        matched_isotropic_vector_coherent
        and _radiomap_diff2_forward_fast_path_supported(
            tx_pos=tx_pos,
            monitor=monitor,
            solver_controls=solver_controls,
            shadow_support_cutoff_db=getattr(monitor, "shadow_support_cutoff_db", None),
        )
    )
    direct_component_updates = (
        diagnostics_targets is not None and not matched_isotropic_vector_coherent
    )

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    los_coherent = accumulate_radio_map_los_coherent(
        scene=scene,
        rx_pos=sample_positions,
        tx_pos=tx_pos,
        wavelength=config.wavelength,
        k=config.k,
    )
    los_field_vector = None
    if str(monitor.receiver_model) == "matched_isotropic":
        ray_dir = sample_positions - tx_pos
        tx_pol_dir = project_real_polarization_to_ray(config.tx_polarization, ray_dir)
        los_field_vector = vector_from_scalar_and_real_direction(los_coherent, tx_pol_dir)
        los_power = _vector_power(los_field_vector)
    else:
        los_power = complex_abs_sqr(los_coherent)
    if direct_component_updates:
        vector_diagnostics = diagnostics_targets.get("vector_coherent")
        diagnostics_targets["coherent"]["los"] = _add_complex(
            diagnostics_targets["coherent"]["los"],
            _scale_complex(los_coherent, sample_weight),
        )
        diagnostics_targets["incoherent"]["los"] = _add_float(
            diagnostics_targets["incoherent"]["los"],
            _scale_float(los_power, sample_weight),
        )
        diagnostics_targets["coherent_power"]["los"] = _add_float(
            diagnostics_targets["coherent_power"]["los"],
            _scale_float(los_power, sample_weight),
        )
        if vector_diagnostics is not None and los_field_vector is not None:
            vector_diagnostics["los"] = _add_complex_vector(
                vector_diagnostics["los"],
                _scale_complex_vector(los_field_vector, sample_weight),
            )
    if return_timing:
        sample_timing["los_seconds"] = time.perf_counter() - t0

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    reflection_detail = _discover_radio_map_reflection_detail(
        sample_grid=sample_grid,
        tx_pos=tx_pos,
        scene=scene,
        config=config,
        solver_controls=solver_controls,
        monitor=monitor,
        reflection_detail=reflection_detail,
    )
    reflection_payload = accumulate_reflection_scalar_power(
        rx_pos=sample_positions,
        scene=scene,
        wavelength=config.wavelength,
        k=config.k,
        reflection_detail=reflection_detail,
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        receiver_model=monitor.receiver_model,
        receiver_tiles=receiver_tiles,
        coherent_target=(
            None
            if not direct_component_updates
            else diagnostics_targets["coherent"]["reflection"]
        ),
        power_target=(
            None
            if not direct_component_updates
            else diagnostics_targets["incoherent"]["reflection"]
        ),
        vector_target=(
            None
            if not direct_component_updates
            else diagnostics_targets.get("vector_coherent", {}).get("reflection")
        ),
        return_vector_coherent=(
            matched_isotropic_vector_coherent or _utd_cross_term_surrogate_enabled(monitor)
        ),
        coherent_scale=sample_weight,
        power_scale=sample_weight,
        allow_tiled_scheduler=not (use_no_diff_fast_path or use_diff2_forward_fast_path),
    )
    reflection_coherent = reflection_payload["coherent"]
    reflection_power = reflection_payload["power"]
    reflection_vector_coherent = reflection_payload.get("vector_coherent")
    if return_timing:
        sample_timing["reflection_seconds"] = time.perf_counter() - t0

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    mixed_reflection_detail = reflection_detail if config.enable_rd_diffraction else None
    reflection_n_rays = effective["reflection_n_rays"] if config.enable_rd_diffraction else 0
    reflection_max_bounces = (
        effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0
    )
    selected_diffraction_state_cache = local_diffraction_state_cache
    cache_mode = "disabled"
    if mixed_reflection_detail is None and persistent_diffraction_state_cache is not None:
        selected_diffraction_state_cache = persistent_diffraction_state_cache
        cache_mode = "persistent"
    elif local_diffraction_state_cache is not None:
        cache_mode = "local"
    anchor_coordinate = _diffraction_anchor_coordinate(sample_grid.axis, tx_pos, sample_grid.position)
    cache_key = (
        None
        if diffraction_state_cache_key_fn is None
        else _radio_map_diffraction_cache_key(
            diffraction_state_cache_key_fn(anchor_coordinate, mixed_reflection_detail),
            state_layout=state_layout,
        )
    )
    prepared_state_group = (
        None
        if selected_diffraction_state_cache is None or cache_key is None
        else selected_diffraction_state_cache.get(cache_key)
    )
    cache_hit = prepared_state_group is not None
    cache_miss = 0
    if effective["max_diffractions"] > 0:
        if return_timing:
            EvalSync.sync()
            t0 = time.perf_counter()
        if not cache_hit:
            prepared_state_group = _prepare_diffraction_state_arrays(
                tx_pos,
                anchor_coordinate,
                scene,
                config.wavelength,
                config.k,
                mixed_reflection_detail,
                config.diffraction_material,
                reflection_n_rays,
                reflection_max_bounces,
                config.reflection_coef,
                monitor.ray_mode,
                effective["max_diffractions"],
                total_state_budget_per_order=effective["diffraction_state_budget"],
                inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
                max_inserted_reflections_per_path=effective["max_inserted_reflections_per_path"],
                retain_cold_metadata=_radiomap_retain_diffraction_cold_metadata(),
                use_scene_materials=config.use_scene_materials_for_diffraction,
                tx_polarization=config.tx_polarization,
                solver_mode=solver_controls["selected"],
                memory_profile=effective["memory_profile"],
                state_layout=state_layout,
            )
            if selected_diffraction_state_cache is not None and cache_key is not None:
                selected_diffraction_state_cache[cache_key] = prepared_state_group
                cache_miss = 1
        if return_timing:
            sample_timing["state_preparation_seconds"] = time.perf_counter() - t0
    edge_cache, edge_data, state_arrays, path_budget_report = prepared_state_group or (
        None,
        None,
        None,
        None,
    )
    prepared_state_diagnostics = _prepared_state_diagnostic_counts(state_arrays)
    solver_metadata = _finalize_solver_metadata(
        _build_solver_metadata(
            scene=scene,
            max_diffractions=effective["max_diffractions"],
            reflection_detail=mixed_reflection_detail,
            reflection_n_rays=reflection_n_rays,
            reflection_max_bounces=reflection_max_bounces,
            reflected_suffix_enabled=False,
            inserted_reflection_enabled=(
                config.enable_rd_diffraction
                and reflection_n_rays > 0
                and reflection_max_bounces > 0
                and effective["max_diffractions"] > 1
                and (effective["max_inserted_reflections_per_path"] or 0) > 0
            ),
            max_inserted_reflections_per_path=(
                0
                if effective["max_inserted_reflections_per_path"] is None
                else int(effective["max_inserted_reflections_per_path"])
            ),
            total_state_budget_per_order=effective["diffraction_state_budget"],
            inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
            path_budget_report=path_budget_report,
        ),
        scene=scene,
        material_detail=config.diffraction_material,
        use_scene_materials=config.use_scene_materials_for_diffraction,
        execution=config.diffraction_execution,
        rx_polarization=config.rx_polarization,
        active_rx_polarization=effective_rx_polarization(
            config.rx_polarization,
            config.tx_polarization,
        ),
        receiver_axis=sample_grid.axis,
    )
    diffraction_payload = accumulate_diffraction_scalar_power(
        state_arrays=state_arrays,
        rx_pos=sample_positions,
        scene=scene,
        wavelength=config.wavelength,
        k=config.k,
        material_detail=config.diffraction_material,
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        receiver_model=monitor.receiver_model,
        receiver_tiles=receiver_tiles,
        return_vector_coherent=matched_isotropic_vector_coherent,
        coherent_target=(
            None
            if not direct_component_updates
            else diagnostics_targets["coherent"]["diffraction"]
        ),
        power_target=(
            None
            if not direct_component_updates
            else diagnostics_targets["incoherent"]["diffraction"]
        ),
        vector_target=(
            None
            if not direct_component_updates
            else diagnostics_targets.get("vector_coherent", {}).get("diffraction")
        ),
        incident_reference_vector=(
            los_field_vector if _utd_cross_term_surrogate_enabled(monitor) else None
        ),
        reflection_reference_vector=(
            reflection_vector_coherent if _utd_cross_term_surrogate_enabled(monitor) else None
        ),
        incident_cross_target=(
            None
            if diagnostics_targets is None or not _utd_cross_term_surrogate_enabled(monitor)
            else diagnostics_targets["incoherent"]["utd_surrogate_incident_cross"]
        ),
        reflection_cross_target=(
            None
            if diagnostics_targets is None or not _utd_cross_term_surrogate_enabled(monitor)
            else diagnostics_targets["incoherent"]["utd_surrogate_reflection_cross"]
        ),
        shadow_support_cutoff_db=getattr(monitor, "shadow_support_cutoff_db", None),
        receiver_axis=sample_grid.axis,
        coherent_scale=sample_weight,
        power_scale=sample_weight,
        cross_scale=sample_weight,
        # Keep the diff=2 cell path on the validated Dr.Jit replay until the
        # shared native forward path reaches forward-parity in direct replay mode.
        native_primal_forward=False,
        # The native matched-isotropic pair accumulator uses atomic reduction
        # and is not repeatable for dense diffraction workloads.
        native_vector_replay=False,
    )
    diffraction_coherent = diffraction_payload["coherent"]
    diffraction_power = diffraction_payload["power"]
    diffraction_vector_coherent = diffraction_payload.get("vector_coherent")
    if direct_component_updates:
        if matched_isotropic_vector_coherent:
            reflection_coherent_power = (
                _zero_float(int(dr.width(los_power)))
                if reflection_vector_coherent is None
                else _vector_power(reflection_vector_coherent)
            )
            diffraction_coherent_power = (
                _zero_float(int(dr.width(los_power)))
                if diffraction_vector_coherent is None
                else _vector_power(diffraction_vector_coherent)
            )
            total_vector = _add_complex_vector(
                _add_complex_vector(los_field_vector, reflection_vector_coherent),
                diffraction_vector_coherent,
            )
            total_coherent_power = (
                _zero_float(int(dr.width(los_power)))
                if total_vector is None
                else _vector_power(total_vector)
            )
            diagnostics_targets["coherent_power"]["reflection"] = _add_float(
                diagnostics_targets["coherent_power"]["reflection"],
                _scale_float(reflection_coherent_power, sample_weight),
            )
            diagnostics_targets["coherent_power"]["diffraction"] = _add_float(
                diagnostics_targets["coherent_power"]["diffraction"],
                _scale_float(diffraction_coherent_power, sample_weight),
            )
            diagnostics_targets["coherent_power"]["total"] = _add_float(
                diagnostics_targets["coherent_power"]["total"],
                _scale_float(total_coherent_power, sample_weight),
            )
        else:
            total_coherent = _add_complex(
                _add_complex(los_coherent, reflection_coherent),
                diffraction_coherent,
            )
            diagnostics_targets["coherent_power"]["reflection"] = _add_float(
                diagnostics_targets["coherent_power"]["reflection"],
                _scale_float(complex_abs_sqr(reflection_coherent), sample_weight),
            )
            diagnostics_targets["coherent_power"]["diffraction"] = _add_float(
                diagnostics_targets["coherent_power"]["diffraction"],
                _scale_float(complex_abs_sqr(diffraction_coherent), sample_weight),
            )
            diagnostics_targets["coherent_power"]["total"] = _add_float(
                diagnostics_targets["coherent_power"]["total"],
                _scale_float(complex_abs_sqr(total_coherent), sample_weight),
            )
    if return_timing:
        sample_timing["diffraction_seconds"] = time.perf_counter() - t0

    diffraction_group_metadata = ()
    if state_arrays is not None:
        diffraction_diagnostics = _merge_diffraction_diagnostic_counts(
            prepared_state_diagnostics,
            diffraction_payload.get("diagnostic_counts"),
        )
        diffraction_group_metadata = (
            {
                "receiver_anchor_coordinate": float(anchor_coordinate),
                "receiver_count": int(sample_grid.n_cells),
                "cache_hit": bool(cache_hit),
                "n_states": int(state_arrays["n_states"]),
                "state_layout": path_export_state_layout(state_arrays) or str(state_layout),
                "state_layout_requested": str(state_layout),
                "diagnostic_counts": dict(diffraction_diagnostics),
                "solver_metadata": solver_metadata,
                "planner_stats": dict(diffraction_payload["planner_stats"]),
            },
        )
    else:
        diffraction_diagnostics = dict(prepared_state_diagnostics)

    runtime_reuse = {
        "cache_mode": cache_mode,
        "state_preparation_hits": int(1 if cache_hit else 0),
        "state_preparation_misses": int(cache_miss),
        "state_layout": path_export_state_layout(state_arrays) or str(state_layout),
    }
    reflection_backend = (
        {}
        if reflection_detail is None
        else dict(reflection_detail.get("dda_stats", {}))
    )
    reflection_scalar_backend = str(
        reflection_payload["planner_stats"].get("scalar_backend", "direct_replay_scalar_power")
    )
    reflection_backend["radio_map_scalar_power_backend"] = (
        "native_dense_vector_power"
        if reflection_scalar_backend == "native_radiomap_vector_power"
        else "direct_in_loop_cell_scatter"
    )
    reflection_backend["pair_replay_backend"] = reflection_scalar_backend
    reflection_backend["path_scheduler"] = str(reflection_payload["planner_stats"]["path_scheduler"])
    reflection_backend["selected_reason"] = str(reflection_payload["planner_stats"]["selected_reason"])
    reflection_backend["estimated_pair_ratio"] = float(
        reflection_payload["planner_stats"]["estimated_pair_ratio"]
    )
    if reflection_payload["planner_stats"]["planner_backend"] is not None:
        reflection_backend["planner_backend"] = str(
            reflection_payload["planner_stats"]["planner_backend"]
        )
    diffraction_backend = dict(solver_metadata.get("diffraction_accumulation_backend", {}))
    diffraction_backend["radio_map_scalar_power_backend"] = (
        "native_radiomap_vector_power_forward_fast"
        if bool(diffraction_payload["planner_stats"].get("forward_fast_path", False))
        else "direct_in_loop_cell_scatter"
    )
    diffraction_backend["pair_replay_backend"] = str(
        diffraction_payload["planner_stats"].get("scalar_backend", "direct_state_scalar_power")
    )
    diffraction_backend["state_scheduler"] = str(diffraction_payload["planner_stats"]["state_scheduler"])
    diffraction_backend["selected_reason"] = str(diffraction_payload["planner_stats"]["selected_reason"])
    diffraction_backend["estimated_pair_ratio"] = float(
        diffraction_payload["planner_stats"]["estimated_pair_ratio"]
    )
    diffraction_backend["suffix_included"] = False
    diffraction_backend["diagnostic_counts"] = dict(diffraction_diagnostics)
    _augment_diffraction_runtime_backend(
        diffraction_backend,
        diffraction_payload["planner_stats"],
        forward_fast_path=bool(diffraction_payload["planner_stats"].get("forward_fast_path", False)),
    )
    if diffraction_payload["planner_stats"]["planner_backend"] is not None:
        diffraction_backend["planner_backend"] = str(
            diffraction_payload["planner_stats"]["planner_backend"]
        )
    runtime_backends = {
        "reflection": reflection_backend,
        "diffraction": diffraction_backend,
        "suffix": {
            "requested_backend": config.diffraction_execution.suffix_backend,
            "resolved_backend": "disabled_for_cell_accumulation",
            "implementation": "disabled",
        },
        "no_diff_fast_path": bool(use_no_diff_fast_path),
        "no_diff_reflection_scheduler": (
            str(reflection_backend.get("path_scheduler", "cartesian_chunked"))
            if use_no_diff_fast_path
            else None
        ),
    }
    path_counts = {
        "los": _count_nonzero_complex(los_coherent),
        "reflection": int(reflection_payload["path_count"]),
        "diffraction": int(diffraction_payload["path_count"]),
    }
    path_counts["total"] = int(path_counts["los"] + path_counts["reflection"] + path_counts["diffraction"])
    return {
        "los_coherent": los_coherent,
        "los_power": los_power,
        "los_field_vector": los_field_vector,
        "reflection_coherent": reflection_coherent,
        "reflection_power": reflection_power,
        "reflection_vector_coherent": reflection_vector_coherent,
        "diffraction_coherent": diffraction_coherent,
        "diffraction_power": diffraction_power,
        "diffraction_vector_coherent": diffraction_vector_coherent,
        "reflection_detail": reflection_detail,
        "diffraction_group_metadata": diffraction_group_metadata,
        "runtime_reuse": runtime_reuse,
        "path_counts": path_counts,
        "diffraction_diagnostics": diffraction_diagnostics,
        "timing": sample_timing,
        "runtime_backends": runtime_backends,
        "component_buffers_updated": direct_component_updates,
        "no_diff_fast_path": bool(use_no_diff_fast_path),
    }


__all__ = [
    "_radiomap_diff2_forward_fast_path_supported",
    "_baseline_matched_isotropic_diffraction_power",
    "_baseline_matched_isotropic_reflection_power",
    "_trace_baseline_matched_isotropic_diffraction_fast",
    "_discover_radio_map_reflection_detail",
    "_trace_diffraction_raw_collections",
    "_trace_native_coherent_sample",
    "_trace_cell_accumulation_sample",
]
