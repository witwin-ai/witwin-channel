from __future__ import annotations

import time
from typing import Callable, Mapping

from ..grid import AxisAlignedRadioMapNativeGrid, RadioMapGrid
from ..monte_carlo.trace import trace as trace_monte_carlo
from .cell_accumulation import (
    accumulate_matched_isb_shadow_completion,
    accumulate_projected_isb_shadow_completion,
)
from ..monitor import RadioMapMonitor
from ..backend import (
    _radio_map_grad_sensitive_workload,
    _resolve_radio_map_accumulation_backend,
)
from ..diagnostics import (
    _accumulate_sample_diagnostics_no_diff_matched_isotropic,
    PROJECTED_ISB_COMPLETION_GAIN,
    PROJECTED_ISB_COMPLETION_RATIO_TARGET,
    _accumulate_complex_by_rx,
    _accumulate_power_by_rx,
    _accumulate_sample_diagnostics,
    _add_complex,
    _add_complex_vector,
    _add_float,
    _baseline_los_power,
    _empty_radio_map_diagnostics,
    _ensure_diffraction_breakdown_diagnostics,
    _ensure_utd_shadow_boundary_diagnostics,
    _finalize_matched_isb_completion_total_no_diff,
    _finalize_no_diff_matched_isotropic_totals,
    _finalize_matched_isb_completion_total,
    _finalize_projected_isb_completion_total,
    _finalize_radio_map_component_totals,
    _finalize_utd_shadow_boundary_surrogate_total,
    _matched_isb_completion_enabled,
    _projected_isb_completion_enabled,
    _raw_path_count,
    _scale_float,
    _single_tx_sinr,
    _utd_cross_term_surrogate_enabled,
    _vector_power_symbolic,
    _zero_float,
)
from ..metadata import (
    RadioMapMetadata,
    _count_reflection_paths,
    _empty_diffraction_diagnostic_counts,
    _merge_diffraction_diagnostic_counts,
    _radio_map_diffraction_state_layout,
    _resolve_noise_power,
)
from ..payload import RadioMapPayload
from .coherent import accumulate_radio_map_reflection_coherent
from .samples import (
    _radiomap_diff2_forward_fast_path_supported,
    _baseline_matched_isotropic_diffraction_power,
    _baseline_matched_isotropic_reflection_power,
    _trace_baseline_matched_isotropic_diffraction_fast,
    _trace_diffraction_raw_collections,
    _trace_native_coherent_sample,
    _trace_cell_accumulation_sample,
)
from ...path.collectors import collect_los_paths, collect_reflection_paths
from ...orchestration import ResolvedTraceConfig
from ....scene import Scene
from ....trace.materials import ReflectionTraceDetail
from ....trace.reflection import discover_reflection_paths
from ....utils.drjit_ops import ArrayInit, EvalSync, complex_abs_sqr


def trace_radio_map_monitor(
    tx_pos,
    monitor: RadioMapMonitor,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls: Mapping[str, object],
    *,
    reflection_detail: ReflectionTraceDetail | Mapping[str, object] | None = None,
    persistent_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None = None,
    local_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None = None,
    diffraction_state_cache_key_fn: Callable[[float, ReflectionTraceDetail | None], tuple[object, ...]] | None = None,
    radio_map_accumulation_backend: str = "auto",
    verbose: bool = False,
    return_timing: bool = False,
    return_reflection_detail: bool = False,
) -> dict[str, object] | tuple[dict[str, object], ReflectionTraceDetail]:
    del verbose
    if str(getattr(monitor, "sampling_mode", "deterministic")) == "monte_carlo":
        return trace_monte_carlo(
            tx_pos,
            monitor,
            scene,
            config,
            solver_controls,
            reflection_detail=reflection_detail,
            persistent_diffraction_state_cache=persistent_diffraction_state_cache,
            local_diffraction_state_cache=local_diffraction_state_cache,
            diffraction_state_cache_key_fn=diffraction_state_cache_key_fn,
            radio_map_accumulation_backend=radio_map_accumulation_backend,
            return_timing=return_timing,
            return_reflection_detail=return_reflection_detail,
        )
    timing = {} if return_timing else None
    grid = RadioMapGrid.from_monitor(monitor, default_cell_size=config.cell_size)
    resolved_accumulation_backend = _resolve_radio_map_accumulation_backend(
        requested_backend=radio_map_accumulation_backend,
        monitor=monitor,
        grid=grid,
        config=config,
        tx_pos=tx_pos,
        scene=scene,
    )
    diffraction_state_layout = _radio_map_diffraction_state_layout(
        resolved_accumulation_backend
    )
    n_rx = int(grid.n_cells)
    matched_isotropic_vector_coherent = (
        str(monitor.combine_mode) == "coherent"
        and str(monitor.receiver_model) == "matched_isotropic"
    )
    no_diff_fast_path_enabled = (
        resolved_accumulation_backend == "cell_accumulation"
        and matched_isotropic_vector_coherent
        and int(solver_controls["effective"]["max_diffractions"]) <= 0
    )
    grad_sensitive_matched_isotropic_reflection = (
        resolved_accumulation_backend == "baseline"
        and matched_isotropic_vector_coherent
        and grid.surface_mode == "axis_aligned"
        and _radio_map_grad_sensitive_workload(
            config,
            tx_pos=tx_pos,
            scene=scene,
        )
    )
    grad_sensitive_matched_isotropic_diffraction = (
        resolved_accumulation_backend == "baseline"
        and matched_isotropic_vector_coherent
        and grid.surface_mode == "axis_aligned"
        and _radio_map_grad_sensitive_workload(
            config,
            tx_pos=tx_pos,
            scene=scene,
        )
        and not _utd_cross_term_surrogate_enabled(monitor)
    )
    weighted_diagnostics = _empty_radio_map_diagnostics(
        n_rx,
        include_vector_coherent=(
            _matched_isb_completion_enabled(monitor) or no_diff_fast_path_enabled
        ),
    )
    if _utd_cross_term_surrogate_enabled(monitor):
        weighted_diagnostics = _ensure_utd_shadow_boundary_diagnostics(
            weighted_diagnostics,
            n_rx=n_rx,
        )

    sample_payload_positions = []
    sample_metadata = []
    selected_local_diffraction_state_cache = (
        {}
        if local_diffraction_state_cache is None
        else local_diffraction_state_cache
    )
    aggregate_runtime_reuse = {
        "cache_mode": "disabled",
        "state_preparation_hits": 0,
        "state_preparation_misses": 0,
        "state_layout": "full",
    }
    aggregate_runtime_backends = {
        "reflection": {},
        "diffraction": {},
        "suffix": {},
        "no_diff_fast_path": bool(no_diff_fast_path_enabled),
        "no_diff_reflection_scheduler": None,
    }

    for sample_set in grid.sample_sets:
        sample_payload_positions.append(sample_set.positions)
        sample_updates_component_buffers = False
        sample_diffraction_diagnostics = _empty_diffraction_diagnostic_counts()
        los_field_vector = None
        reflection_vector_coherent = None
        diffraction_vector_coherent = None
        diffraction_incident_cross = _zero_float(n_rx)
        diffraction_reflection_cross = _zero_float(n_rx)
        sample_runtime_backends = {
            "reflection": {},
            "diffraction": {},
            "suffix": {},
            "no_diff_fast_path": False,
            "no_diff_reflection_scheduler": None,
        }
        sample_no_diff_fast_path = False

        if resolved_accumulation_backend == "native_coherent":
            sample_payload = _trace_native_coherent_sample(
                grid=grid,
                sample_set=sample_set,
                tx_pos=tx_pos,
                scene=scene,
                config=config,
                solver_controls=solver_controls,
                monitor=monitor,
                reflection_detail=reflection_detail,
                persistent_diffraction_state_cache=persistent_diffraction_state_cache,
                local_diffraction_state_cache=selected_local_diffraction_state_cache,
                diffraction_state_cache_key_fn=diffraction_state_cache_key_fn,
                state_layout=diffraction_state_layout,
                return_timing=return_timing,
            )
            reflection_detail = sample_payload["reflection_detail"]
            sample_timing = sample_payload["timing"]
            diffraction_group_metadata = sample_payload["diffraction_group_metadata"]
            runtime_reuse = sample_payload["runtime_reuse"]
            los_coherent = sample_payload["los_coherent"]
            reflection_coherent = sample_payload["reflection_coherent"]
            diffraction_coherent = sample_payload["diffraction_coherent"]
            los_power = sample_payload["los_power"]
            reflection_power = sample_payload["reflection_power"]
            diffraction_power = sample_payload["diffraction_power"]
            los_field_vector = sample_payload.get("los_field_vector")
            reflection_vector_coherent = sample_payload.get("reflection_vector_coherent")
            diffraction_vector_coherent = sample_payload.get("diffraction_vector_coherent")
            sample_runtime_backends = dict(sample_payload["runtime_backends"])
            sample_path_counts = dict(sample_payload["path_counts"])
            sample_diffraction_diagnostics = dict(
                sample_payload.get("diffraction_diagnostics", {})
            )
            sample_no_diff_fast_path = bool(sample_payload.get("no_diff_fast_path", False))
        elif resolved_accumulation_backend == "cell_accumulation":
            sample_payload = _trace_cell_accumulation_sample(
                grid=grid,
                sample_set=sample_set,
                tx_pos=tx_pos,
                scene=scene,
                config=config,
                solver_controls=solver_controls,
                monitor=monitor,
                reflection_detail=reflection_detail,
                persistent_diffraction_state_cache=persistent_diffraction_state_cache,
                local_diffraction_state_cache=selected_local_diffraction_state_cache,
                diffraction_state_cache_key_fn=diffraction_state_cache_key_fn,
                state_layout=diffraction_state_layout,
                diagnostics_targets=weighted_diagnostics,
                sample_weight=sample_set.weight,
                return_timing=return_timing,
            )
            sample_updates_component_buffers = bool(
                sample_payload.get("component_buffers_updated", False)
            )
            reflection_detail = sample_payload["reflection_detail"]
            sample_timing = sample_payload["timing"]
            diffraction_group_metadata = sample_payload["diffraction_group_metadata"]
            runtime_reuse = sample_payload["runtime_reuse"]
            los_coherent = sample_payload["los_coherent"]
            los_power = sample_payload["los_power"]
            reflection_coherent = sample_payload["reflection_coherent"]
            reflection_power = sample_payload["reflection_power"]
            diffraction_coherent = sample_payload["diffraction_coherent"]
            diffraction_power = sample_payload["diffraction_power"]
            los_field_vector = sample_payload.get("los_field_vector")
            reflection_vector_coherent = sample_payload.get("reflection_vector_coherent")
            diffraction_vector_coherent = sample_payload.get("diffraction_vector_coherent")
            sample_runtime_backends = dict(sample_payload["runtime_backends"])
            sample_path_counts = dict(sample_payload["path_counts"])
            sample_diffraction_diagnostics = dict(
                sample_payload.get("diffraction_diagnostics", {})
            )
            sample_no_diff_fast_path = bool(sample_payload.get("no_diff_fast_path", False))
        else:
            sample_timing = {} if return_timing else None
            reflection_backend = {}
            reflection_path_count = 0
            use_grad_preserving_reflection = bool(
                grad_sensitive_matched_isotropic_reflection
            )
            sample_grid = None
            if (
                use_grad_preserving_reflection
                or bool(grad_sensitive_matched_isotropic_diffraction)
            ):
                sample_grid = AxisAlignedRadioMapNativeGrid.from_grid(
                    grid,
                    sample_index=sample_set.index,
                )
            if return_timing:
                EvalSync.sync()
                t0 = time.perf_counter()
            los_raw = collect_los_paths(
                scene=scene,
                rx_positions=sample_set.positions,
                tx_pos=tx_pos,
                wavelength=config.wavelength,
                k=config.k,
                tx_polarization=config.tx_polarization,
                rx_polarization=config.rx_polarization,
            )
            if return_timing:
                sample_timing["los_seconds"] = time.perf_counter() - t0

            reflection_raw = None
            reflection_coherent = ArrayInit.complex_zero(n_rx)
            reflection_power = _zero_float(n_rx)
            reflection_vector_coherent = None
            if return_timing:
                EvalSync.sync()
                t0 = time.perf_counter()
            if use_grad_preserving_reflection:
                reflection_detail = discover_reflection_paths(
                    tx_pos=tx_pos,
                    scene=scene,
                    wavelength=config.wavelength,
                    k=config.k,
                    n_rays=solver_controls["effective"]["reflection_n_rays"],
                    max_reflections=solver_controls["effective"]["reflection_max_bounces"],
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
                (
                    reflection_coherent,
                    reflection_vector_coherent,
                    reflection_detail,
                    _reflection_seconds,
                ) = accumulate_radio_map_reflection_coherent(
                    sample_grid=sample_grid,
                    tx_pos=tx_pos,
                    scene=scene,
                    wavelength=config.wavelength,
                    k=config.k,
                    reflection_n_rays=solver_controls["effective"]["reflection_n_rays"],
                    reflection_max_bounces=solver_controls["effective"]["reflection_max_bounces"],
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
                    return_vector=True,
                )
                reflection_power = (
                    _zero_float(int(sample_grid.n_cells))
                    if reflection_vector_coherent is None
                    else _vector_power_symbolic(reflection_vector_coherent)
                )
                reflection_path_count = _count_reflection_paths(reflection_detail)
                reflection_backend = (
                    {}
                    if reflection_detail is None
                    else dict(reflection_detail.get("dda_stats", {}))
                )
                reflection_backend["radio_map_power_backend"] = "symbolic_vector_power"
                reflection_backend["field_accumulation_backend"] = "epc_reflection_detail_replay"
                reflection_backend["gradient_preserving"] = True
            else:
                reflection_raw, reflection_detail = collect_reflection_paths(
                    scene=scene,
                    rx_positions=sample_set.positions,
                    tx_pos=tx_pos,
                    wavelength=config.wavelength,
                    k=config.k,
                    n_rays=solver_controls["effective"]["reflection_n_rays"],
                    max_reflections=solver_controls["effective"]["reflection_max_bounces"],
                    mode=monitor.ray_mode,
                    tx_polarization=config.tx_polarization,
                    rx_polarization=config.rx_polarization,
                    reflection_coef=config.reflection_coef,
                    min_ray_contribution_threshold=config.min_ray_contribution_threshold,
                    reflection_relative_permittivity=config.reflection_relative_permittivity,
                    reflection_conductivity=config.reflection_conductivity,
                    reflection_material=config.reflection_material,
                    use_scene_materials=config.use_scene_materials_for_reflection,
                    return_geometry=False,
                    reflection_detail=reflection_detail,
                )
                reflection_path_count = int(_raw_path_count(reflection_raw))
            if return_timing:
                reflection_elapsed = time.perf_counter() - t0
                sample_timing["reflection_seconds"] = reflection_elapsed

            shadow_support_cutoff_db = getattr(monitor, "shadow_support_cutoff_db", None)
            use_baseline_diff2_forward_fast_path = (
                not bool(grad_sensitive_matched_isotropic_diffraction)
                and _radiomap_diff2_forward_fast_path_supported(
                    tx_pos=tx_pos,
                    monitor=monitor,
                    solver_controls=solver_controls,
                    shadow_support_cutoff_db=shadow_support_cutoff_db,
                )
            )
            if return_timing:
                EvalSync.sync()
                t0 = time.perf_counter()
            if use_baseline_diff2_forward_fast_path:
                if sample_grid is None:
                    sample_grid = AxisAlignedRadioMapNativeGrid.from_grid(
                        grid,
                        sample_index=sample_set.index,
                    )
                fast_diffraction = _trace_baseline_matched_isotropic_diffraction_fast(
                    sample_positions=sample_set.positions,
                    sample_grid=sample_grid,
                    tx_pos=tx_pos,
                    scene=scene,
                    config=config,
                    solver_controls=solver_controls,
                    monitor=monitor,
                    reflection_detail=reflection_detail,
                    persistent_diffraction_state_cache=persistent_diffraction_state_cache,
                    local_diffraction_state_cache=selected_local_diffraction_state_cache,
                    diffraction_state_cache_key_fn=diffraction_state_cache_key_fn,
                    state_layout=diffraction_state_layout,
                )
                diffraction_raw_collections = ()
                diffraction_group_metadata = fast_diffraction["diffraction_group_metadata"]
                runtime_reuse = fast_diffraction["runtime_reuse"]
            else:
                diffraction_raw_collections, diffraction_group_metadata, runtime_reuse = (
                    _trace_diffraction_raw_collections(
                        sample_positions=sample_set.positions,
                        tx_pos=tx_pos,
                        scene=scene,
                        config=config,
                        solver_controls=solver_controls,
                        monitor=monitor,
                        reflection_detail=reflection_detail,
                        persistent_diffraction_state_cache=persistent_diffraction_state_cache,
                        local_diffraction_state_cache=selected_local_diffraction_state_cache,
                        diffraction_state_cache_key_fn=diffraction_state_cache_key_fn,
                        state_layout=diffraction_state_layout,
                        preserve_higher_order_candidate_topology=bool(
                            grad_sensitive_matched_isotropic_diffraction
                        ),
                    )
                )
            if return_timing:
                sample_timing["diffraction_seconds"] = time.perf_counter() - t0

            los_coherent = _accumulate_complex_by_rx(los_raw, n_rx=n_rx)
            los_power, los_field_vector = _baseline_los_power(
                monitor=monitor,
                sample_positions=sample_set.positions,
                tx_pos=tx_pos,
                config=config,
                los_coherent=los_coherent,
            )
            if not use_grad_preserving_reflection:
                reflection_coherent = _accumulate_complex_by_rx(reflection_raw, n_rx=n_rx)
            diffraction_coherent = ArrayInit.complex_zero(n_rx)
            diffraction_vector_coherent = None
            diffraction_incident_cross = _zero_float(n_rx)
            diffraction_reflection_cross = _zero_float(n_rx)

            matched_isotropic_vector_coherent = (
                str(monitor.combine_mode) == "coherent"
                and str(monitor.receiver_model) == "matched_isotropic"
            )
            if str(monitor.receiver_model) == "matched_isotropic":
                if not use_grad_preserving_reflection:
                    (
                        reflection_power,
                        reflection_vector_coherent,
                        reflection_backend,
                    ) = _baseline_matched_isotropic_reflection_power(
                        sample_positions=sample_set.positions,
                        scene=scene,
                        config=config,
                        reflection_detail=reflection_detail,
                    )
                if use_baseline_diff2_forward_fast_path:
                    diffraction_coherent = fast_diffraction["diffraction_coherent"]
                    diffraction_power = fast_diffraction["diffraction_power"]
                    diffraction_incident_cross = fast_diffraction["incident_cross"]
                    diffraction_reflection_cross = fast_diffraction["reflection_cross"]
                    diffraction_vector_coherent = fast_diffraction["diffraction_vector_coherent"]
                    diffraction_backend = fast_diffraction["runtime_backend"]
                    diffraction_path_count = int(fast_diffraction["path_count"])
                    sample_diffraction_diagnostics = dict(fast_diffraction["diagnostic_counts"])
                else:
                    (
                        diffraction_coherent,
                        diffraction_power,
                        diffraction_incident_cross,
                        diffraction_reflection_cross,
                        diffraction_vector_coherent,
                        diffraction_backend,
                    ) = _baseline_matched_isotropic_diffraction_power(
                        diffraction_raw_collections=diffraction_raw_collections,
                        scene=scene,
                        config=config,
                        n_rx=n_rx,
                        receiver_axis=monitor.axis if monitor.axis is not None else "z",
                        sample_grid=sample_grid,
                        grad_preserving_dense_replay=bool(
                            grad_sensitive_matched_isotropic_diffraction
                        ),
                        shadow_support_cutoff_db=shadow_support_cutoff_db,
                        los_reference_vector=los_field_vector,
                        reflection_reference_vector=reflection_vector_coherent,
                    )
                sample_runtime_backends["reflection"] = reflection_backend
                sample_runtime_backends["diffraction"] = diffraction_backend
                sample_diffraction_diagnostics = dict(
                    diffraction_backend.get("diagnostic_counts", {})
                )
                sample_runtime_backends["suffix"] = {
                    "requested_backend": config.diffraction_execution.suffix_backend,
                    "resolved_backend": "disabled_for_baseline_incoherent",
                    "implementation": "disabled",
                }
            else:
                reflection_power = _accumulate_power_by_rx(reflection_raw, n_rx=n_rx)
                diffraction_power = _zero_float(n_rx)

            if not use_baseline_diff2_forward_fast_path:
                diffraction_path_count = 0
                for raw in diffraction_raw_collections:
                    if str(monitor.receiver_model) != "matched_isotropic":
                        diffraction_coherent = _add_complex(
                            diffraction_coherent,
                            _accumulate_complex_by_rx(raw, n_rx=n_rx),
                        )
                        diffraction_power = _add_float(
                            diffraction_power,
                            _accumulate_power_by_rx(raw, n_rx=n_rx),
                        )
                    diffraction_path_count += _raw_path_count(raw)
            sample_path_counts = {
                "los": int(_raw_path_count(los_raw)),
                "reflection": int(reflection_path_count),
                "diffraction": int(diffraction_path_count),
                "total": int(
                    _raw_path_count(los_raw)
                    + reflection_path_count
                    + diffraction_path_count
                ),
            }

        for backend_key, backend_value in sample_runtime_backends.items():
            if backend_key in {"no_diff_fast_path", "no_diff_reflection_scheduler"}:
                if backend_key == "no_diff_fast_path":
                    aggregate_runtime_backends[backend_key] = bool(backend_value)
                elif backend_value is not None:
                    aggregate_runtime_backends[backend_key] = str(backend_value)
                continue
            if backend_value:
                aggregate_runtime_backends[backend_key] = dict(backend_value)
        if runtime_reuse["cache_mode"] != "disabled":
            aggregate_runtime_reuse["cache_mode"] = runtime_reuse["cache_mode"]
        aggregate_runtime_reuse["state_preparation_hits"] += int(
            runtime_reuse["state_preparation_hits"]
        )
        aggregate_runtime_reuse["state_preparation_misses"] += int(
            runtime_reuse["state_preparation_misses"]
        )
        if runtime_reuse["state_layout"] != "full":
            aggregate_runtime_reuse["state_layout"] = runtime_reuse["state_layout"]

        if sample_no_diff_fast_path:
            weighted_diagnostics = _accumulate_sample_diagnostics_no_diff_matched_isotropic(
                weighted_diagnostics,
                sample_weight=sample_set.weight,
                los_coherent=los_coherent,
                reflection_coherent=reflection_coherent,
                los_power=los_power,
                reflection_power=reflection_power,
                los_field_vector=los_field_vector,
                reflection_vector_coherent=reflection_vector_coherent,
            )
        else:
            weighted_diagnostics = _accumulate_sample_diagnostics(
                weighted_diagnostics,
                monitor=monitor,
                n_rx=n_rx,
                sample_weight=sample_set.weight,
                sample_updates_component_buffers=sample_updates_component_buffers,
                los_coherent=los_coherent,
                reflection_coherent=reflection_coherent,
                diffraction_coherent=diffraction_coherent,
                los_power=los_power,
                reflection_power=reflection_power,
                diffraction_power=diffraction_power,
                los_field_vector=los_field_vector,
                reflection_vector_coherent=reflection_vector_coherent,
                diffraction_vector_coherent=diffraction_vector_coherent,
                diffraction_incident_cross=diffraction_incident_cross,
                diffraction_reflection_cross=diffraction_reflection_cross,
            )

        sample_metadata.append(
            {
                "sample_index": int(sample_set.index),
                "offset_local": tuple(float(value) for value in sample_set.offset_local),
                "weight": float(sample_set.weight),
                "path_counts": dict(sample_path_counts),
                "diffraction_diagnostics": dict(sample_diffraction_diagnostics),
                "runtime_reuse": dict(runtime_reuse),
                "diffraction_groups": diffraction_group_metadata,
                "timing": None if sample_timing is None else dict(sample_timing),
            }
        )

    if no_diff_fast_path_enabled:
        _finalize_no_diff_matched_isotropic_totals(
            weighted_diagnostics,
            compute_total_power=not _matched_isb_completion_enabled(monitor),
        )
    else:
        weighted_diagnostics = _finalize_radio_map_component_totals(weighted_diagnostics)

    utd_surrogate_total = (
        _finalize_utd_shadow_boundary_surrogate_total(weighted_diagnostics)
        if not no_diff_fast_path_enabled and _utd_cross_term_surrogate_enabled(monitor)
        else None
    )
    projected_isb_surrogate_total = None
    completion_diagnostic_counts = _empty_diffraction_diagnostic_counts()
    if not no_diff_fast_path_enabled and _projected_isb_completion_enabled(monitor):
        projected_isb_surrogate_total = _finalize_projected_isb_completion_total(
            weighted_diagnostics,
            accumulate_projected_isb_shadow_completion(
                rx_pos=grid.cell_centers,
                scene=scene,
                tx_pos=tx_pos,
                wavelength=config.wavelength,
                k=config.k,
                tx_polarization=config.tx_polarization,
                rx_polarization=config.rx_polarization,
                los_coherent=weighted_diagnostics["coherent"]["los"],
                diffraction_coherent=weighted_diagnostics["coherent"]["diffraction"],
                ratio_target=PROJECTED_ISB_COMPLETION_RATIO_TARGET,
                completion_gain=PROJECTED_ISB_COMPLETION_GAIN,
            ),
        )
    matched_isb_surrogate_total = None
    if _matched_isb_completion_enabled(monitor):
        completion_payload = accumulate_matched_isb_shadow_completion(
            rx_pos=grid.cell_centers,
            scene=scene,
            tx_pos=tx_pos,
            wavelength=config.wavelength,
            k=config.k,
            tx_polarization=config.tx_polarization,
            rx_polarization=config.rx_polarization,
            los_vector_coherent=weighted_diagnostics["vector_coherent"]["los"],
            raw_transition_vector=_add_complex_vector(
                weighted_diagnostics["vector_coherent"]["los"],
                weighted_diagnostics["vector_coherent"]["diffraction"],
            ),
        )
        matched_isb_surrogate_total = (
            _finalize_matched_isb_completion_total_no_diff(
                weighted_diagnostics,
                completion_payload,
            )
            if no_diff_fast_path_enabled
            else _finalize_matched_isb_completion_total(
                weighted_diagnostics,
                completion_payload,
            )
        )
        completion_diagnostic_counts = dict(
            completion_payload.get("diagnostic_counts", {})
        )
    _ensure_diffraction_breakdown_diagnostics(weighted_diagnostics)

    path_gain = (
        (
            matched_isb_surrogate_total
            if matched_isb_surrogate_total is not None
            else (
                projected_isb_surrogate_total
                if projected_isb_surrogate_total is not None
                else weighted_diagnostics["coherent_power"]["total"]
            )
        )
        if monitor.combine_mode == "coherent"
        else (
            utd_surrogate_total
            if utd_surrogate_total is not None
            else weighted_diagnostics["incoherent"]["total"]
        )
    )
    rss = _scale_float(path_gain, monitor.tx_power)
    noise_power, noise_power_source = _resolve_noise_power(scene, monitor)
    sinr = _single_tx_sinr(rss, noise_power=noise_power)

    if return_timing:
        timing["total_seconds"] = float(
            sum(
                (sample["timing"] or {}).get("state_preparation_seconds", 0.0)
                + (sample["timing"] or {}).get("los_seconds", 0.0)
                + (sample["timing"] or {}).get("reflection_seconds", 0.0)
                + (sample["timing"] or {}).get("diffraction_seconds", 0.0)
                for sample in sample_metadata
            )
        )

    path_counts = {
        "los": int(sum(sample["path_counts"]["los"] for sample in sample_metadata)),
        "reflection": int(sum(sample["path_counts"]["reflection"] for sample in sample_metadata)),
        "diffraction": int(sum(sample["path_counts"]["diffraction"] for sample in sample_metadata)),
    }
    path_counts["total"] = int(path_counts["los"] + path_counts["reflection"] + path_counts["diffraction"])

    metadata = RadioMapMetadata(
        monitor=monitor,
        grid=grid,
        scene=scene,
        solver_controls=solver_controls,
        path_counts=path_counts,
        sample_metadata=sample_metadata,
        aggregate_runtime_reuse=aggregate_runtime_reuse,
        aggregate_runtime_backends=aggregate_runtime_backends,
        reflection_detail=reflection_detail,
        radio_map_accumulation_backend=radio_map_accumulation_backend,
        resolved_accumulation_backend=resolved_accumulation_backend,
        noise_power=noise_power,
        noise_power_source=noise_power_source,
    )
    metadata["diffraction_diagnostics"] = _merge_diffraction_diagnostic_counts(
        metadata.get("diffraction_diagnostics"),
        completion_diagnostic_counts,
    )
    result = RadioMapPayload(
        monitor=monitor,
        grid=grid,
        weighted_diagnostics=weighted_diagnostics,
        metadata=metadata,
        path_gain=path_gain,
        rss=rss,
        sinr=sinr,
        tx_pos=tx_pos,
        noise_power=noise_power,
        sample_payload_positions=sample_payload_positions,
        timing=timing if return_timing else None,
    )
    if return_reflection_detail:
        return result, reflection_detail
    return result


__all__ = [
    "trace_radio_map_monitor",
]
