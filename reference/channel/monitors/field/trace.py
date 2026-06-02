"""FieldMonitor-specific solver orchestration."""

from __future__ import annotations

import time
from typing import Mapping

import drjit as dr
import witwin as wt

from .monitor import FieldMonitor
from ...utils.plane_axes import tangential_axes_for_axis
from ...utils.polarization import (
    effective_rx_polarization,
    project_real_polarization_to_ray,
    scalarize_tangential_jones,
    tangential_jones,
    vector_add,
    vector_eval,
    vector_from_scalar_and_real_direction,
    vector_zero,
)
from ...scene import Scene
from ...utils import scalar
from ...utils.drjit_ops import EvalSync, eval_complex
from ..._native import native_extension_available
from ...trace.diffraction import compute_diffraction_field
from ...trace.los import compute_los_field
from ...trace.materials import ReflectionTraceDetail, coerce_reflection_trace_detail
from ...trace.reflection import compute_reflection_field
from ..orchestration import (
    ResolvedTraceConfig,
)
from ..profiler import (
    build_state_guardrail_profile,
    capture_cuda_memory_report,
)

def _scalarize_monitor_vector_field(polarization_field, *, axis: str, rx_polarization):
    return scalarize_tangential_jones(
        tangential_jones(polarization_field, axis=axis),
        rx_polarization,
        axis=axis,
    )


def _compute_los(
    scene: Scene,
    rx_positions,
    monitor_axis: str,
    tx_pos,
    config: ResolvedTraceConfig,
    active_rx_polarization,
):
    a_los = compute_los_field(scene, rx_positions, tx_pos, config.wavelength, config.k)
    los_ray_dir = rx_positions - tx_pos
    los_ray_dir = los_ray_dir / (dr.norm(los_ray_dir) + 1e-12)
    los_pol_dir = project_real_polarization_to_ray(config.tx_polarization, los_ray_dir)
    polarization_los = vector_eval(vector_from_scalar_and_real_direction(a_los, los_pol_dir))
    a_los = scalarize_tangential_jones(
        tangential_jones(polarization_los, axis=monitor_axis),
        active_rx_polarization,
        axis=monitor_axis,
    )
    return a_los, polarization_los


def _compute_reflection(
    field,
    calculation_height: float,
    tx_pos,
    scene: Scene,
    config: ResolvedTraceConfig,
    effective: Mapping[str, object],
    monitor: FieldMonitor,
    coords: Mapping[str, object],
    *,
    reflection_detail: ReflectionTraceDetail | Mapping[str, object] | None = None,
    include_field_payload: bool = True,
) -> tuple[wt.Complex2f, list[wt.Complex2f], ReflectionTraceDetail]:
    return compute_reflection_field(
        grid=field,
        rx_z=calculation_height,
        tx_pos=tx_pos,
        scene=scene,
        wavelength=config.wavelength,
        k=config.k,
        n_rays=effective["reflection_n_rays"],
        max_reflections=effective["reflection_max_bounces"],
        mode=monitor.ray_mode,
        ray_sampling=monitor.ray_sampling,
        reflection_coef=config.reflection_coef,
        min_ray_contribution_threshold=config.min_ray_contribution_threshold,
        reflection_field_backend=config.reflection_field_backend,
        tx_polarization=config.tx_polarization,
        reflection_relative_permittivity=config.reflection_relative_permittivity,
        reflection_conductivity=config.reflection_conductivity,
        reflection_material=config.reflection_material,
        use_scene_materials=config.use_scene_materials_for_reflection,
        rx_polarization=config.rx_polarization,
        return_per_bounce=False,
        grid_data=coords,
        reflection_detail=reflection_detail,
        include_field_payload=include_field_payload,
        prefer_epc=False,
    )


def _diffraction_edge_anchor_coordinate(field, tx_pos):
    if field.axis == "z":
        return float(field.position)
    return float(scalar(tx_pos.z))


def _compute_diffraction(
    X,
    Y,
    calculation_height,
    tx_pos,
    scene: Scene,
    config: ResolvedTraceConfig,
    effective,
    monitor: FieldMonitor,
    field,
    coords,
    reflection_detail,
    return_diffraction_audit,
):
    return compute_diffraction_field(
        X,
        Y,
        calculation_height,
        tx_pos,
        scene,
        config.wavelength,
        config.k,
        reflection_detail=reflection_detail if config.enable_rd_diffraction else None,
        max_diffractions=effective["max_diffractions"],
        reflection_n_rays=effective["reflection_n_rays"] if config.enable_rd_diffraction else 0,
        reflection_max_bounces=effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0,
        reflection_coef=config.reflection_coef,
        reflection_mode=monitor.ray_mode,
        grid=field,
        grid_data=coords,
        return_components=True,
        return_per_edge=False,
        return_state_audit=return_diffraction_audit,
        diffraction_material=config.diffraction_material,
        use_scene_materials=config.use_scene_materials_for_diffraction,
        total_state_budget_per_order=effective["diffraction_state_budget"],
        inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
        max_inserted_reflections_per_path=effective["max_inserted_reflections_per_path"],
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        rx_pos=field.receivers,
        receiver_axis=field.axis,
        edge_anchor_coordinate=_diffraction_edge_anchor_coordinate(field, tx_pos),
        execution=config.diffraction_execution,
        solver_mode=config.solver_mode,
        memory_profile=effective["memory_profile"],
    )


def _compute_diffraction_summary(
    X,
    Y,
    calculation_height,
    tx_pos,
    scene: Scene,
    config: ResolvedTraceConfig,
    effective,
    monitor: FieldMonitor,
    field,
    coords,
    reflection_detail,
    return_diffraction_audit,
):
    return compute_diffraction_field(
        X,
        Y,
        calculation_height,
        tx_pos,
        scene,
        config.wavelength,
        config.k,
        reflection_detail=reflection_detail if config.enable_rd_diffraction else None,
        max_diffractions=effective["max_diffractions"],
        reflection_n_rays=effective["reflection_n_rays"] if config.enable_rd_diffraction else 0,
        reflection_max_bounces=effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0,
        reflection_coef=config.reflection_coef,
        reflection_mode=monitor.ray_mode,
        grid=field,
        grid_data=coords,
        return_components=False,
        return_per_edge=False,
        return_solver_metadata=True,
        return_state_audit=return_diffraction_audit,
        diffraction_material=config.diffraction_material,
        use_scene_materials=config.use_scene_materials_for_diffraction,
        total_state_budget_per_order=effective["diffraction_state_budget"],
        inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
        max_inserted_reflections_per_path=effective["max_inserted_reflections_per_path"],
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        rx_pos=field.receivers,
        receiver_axis=field.axis,
        edge_anchor_coordinate=_diffraction_edge_anchor_coordinate(field, tx_pos),
        execution=config.diffraction_execution,
        solver_mode=config.solver_mode,
        memory_profile=effective["memory_profile"],
    )


def _prepare_field_monitor_trace_context(
    tx_pos,
    monitor: FieldMonitor,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls,
    *,
    verbose: bool,
):
    calculation_height = float(monitor.position)
    effective_resolution = (
        monitor.resolution if monitor.resolution is not None else config.resolution_wavelength
    )
    field = monitor.to_field(
        config.wavelength,
        default_resolution=config.resolution_wavelength,
    )
    grid_shape = tuple(int(value) for value in field.size)

    if verbose and monitor.grid_size is None:
        print(
            f"Auto grid shape for monitor '{monitor.name}': {grid_shape} "
            f"(cell_size={effective_resolution * config.wavelength:.4f}m = "
            f"lambda/{config.wavelength / (effective_resolution * config.wavelength):.1f})"
        )

    coords = field.get_coordinates()

    if scene.n_diffraction_edges == 0 and verbose:
        selection_mode = getattr(scene, "edge_selection_mode", "vertical_only")
        print(f"Warning: No valid diffraction edges found for edge_selection_mode='{selection_mode}'!")

    return {
        "calculation_height": calculation_height,
        "effective_resolution": effective_resolution,
        "field": field,
        "grid_shape": grid_shape,
        "coords": coords,
        "X": coords["X"],
        "Y": coords["Y"],
        "rx_positions": field.receivers,
        "effective": solver_controls["effective"],
        "active_rx_polarization": effective_rx_polarization(
            config.rx_polarization,
            config.tx_polarization,
        ),
    }


def _jones_metadata(axis: str, *, explicit_receiver_projection: bool) -> dict[str, object]:
    if axis == "z":
        return {
            "result_basis": "global_xy_jones_projection",
            "result_jones_basis": "global_xy",
            "result_jones_axes": ("x", "y"),
            "scalar_projection_rule": (
                "global_xy_receiver_projection_from_jones"
                if explicit_receiver_projection
                else "global_xy_default_receiver_projection_from_jones"
            ),
            "reflection_scalarization": (
                "explicit_receiver_projection_from_jones"
                if explicit_receiver_projection
                else "default_receiver_projection_from_jones"
            ),
            "diffraction_face_scalarization": (
                "explicit_receiver_projection_from_jones_face_operator"
                if explicit_receiver_projection
                else "default_receiver_projection_from_jones_face_operator"
            ),
        }
    return {
        "result_basis": "monitor_tangential_jones_projection",
        "result_jones_basis": "monitor_tangential",
        "result_jones_axes": tangential_axes_for_axis(axis),
        "scalar_projection_rule": (
            "explicit_monitor_tangential_receiver_projection_from_jones"
            if explicit_receiver_projection
            else "default_monitor_tangential_receiver_projection_from_jones"
        ),
        "reflection_scalarization": (
            "explicit_monitor_tangential_receiver_projection_from_jones"
            if explicit_receiver_projection
            else "default_monitor_tangential_receiver_projection_from_jones"
        ),
        "diffraction_face_scalarization": (
            "explicit_monitor_tangential_receiver_projection_from_jones_face_operator"
            if explicit_receiver_projection
            else "default_monitor_tangential_receiver_projection_from_jones_face_operator"
        ),
    }


def _build_performance_guardrails(solver_controls, diffraction_components):
    report = diffraction_components["solver_metadata"]["path_budget_policy"]["report"]
    return build_state_guardrail_profile(
        solver_controls,
        path_budget_report=report,
        fallback_history_size=int(solver_controls["effective"]["max_diffractions"]),
        final_total_states=int(diffraction_components["n_edge_states"]),
    )


def _public_execution_metadata(execution) -> dict[str, object]:
    if not isinstance(execution, Mapping):
        return {}
    public = dict(execution)
    public.pop("suffix_backend", None)
    return public


def _monitor_reflection_sampling_metadata(
    reflection_detail: ReflectionTraceDetail,
    *,
    axis: str,
) -> dict[str, object]:
    sampling = {}
    discovery_sampling = reflection_detail.get("discovery_sampling")
    if isinstance(discovery_sampling, Mapping):
        sampling.update(dict(discovery_sampling))
    else:
        reflection_sampling = reflection_detail.get("reflection_sampling")
        if isinstance(reflection_sampling, Mapping):
            sampling.update(dict(reflection_sampling))

    dda_stats = reflection_detail.get("dda_stats")
    if isinstance(dda_stats, Mapping):
        for key in (
            "requested_backend",
            "resolved_backend",
            "implementation",
            "native_ad_mode",
            "projected_direction_norm_threshold",
            "diagnostic_dt_threshold",
            "per_bounce",
            "policy",
            "reused_discovery",
            "discovery_gradients_preserved",
            "tx_grad_enabled",
            "scene_geometry_grad_enabled",
            "scene_material_grad_enabled",
            "epc_eligible",
        ):
            if key in dda_stats and key not in sampling:
                sampling[key] = dda_stats[key]

    sampling["backend"] = "dda_planar_grid" if axis == "z" else "ray_plane_scatter"
    return sampling


def _build_trace_metadata(
    dif_components,
    reflection_detail: ReflectionTraceDetail,
    solver_controls: Mapping[str, object],
    monitor: FieldMonitor,
    scene: Scene,
    config: ResolvedTraceConfig,
    grid_shape,
    effective_resolution,
    calculation_height,
) -> dict[str, object]:
    metadata = dict(dif_components["solver_metadata"])
    if "execution" in metadata:
        metadata["execution"] = _public_execution_metadata(metadata["execution"])
    reflection_model_source = coerce_reflection_trace_detail(
        reflection_detail
    ).reflection_model_source
    use_receiver_projection = config.rx_polarization is not None
    jones_metadata = _jones_metadata(
        monitor.axis,
        explicit_receiver_projection=use_receiver_projection,
    )
    metadata["polarization_transport"] = {
        "enabled": True,
        "tx_polarization": config.tx_polarization,
        "rx_polarization": effective_rx_polarization(
            config.rx_polarization,
            config.tx_polarization,
        ),
        "rx_polarization_source": (
            "explicit" if use_receiver_projection else "default_from_tx_polarization"
        ),
        "transport_basis": "path_transverse",
        "result_basis": jones_metadata["result_basis"],
        "result_jones_basis": jones_metadata["result_jones_basis"],
        "result_jones_axes": jones_metadata["result_jones_axes"],
        "scalar_projection_rule": jones_metadata["scalar_projection_rule"],
        "los_transport": "project transmit polarization onto the plane transverse to each LoS ray",
        "reflection_transport": (
            "TE/TM Fresnel transport with basis rotation in the reflection plane; "
            "scalar reflection field is derived from the final receiver projection of the world-vector result"
        ),
        "reflection_scalarization": jones_metadata["reflection_scalarization"],
        "diffraction_transport": (
            "edge-local Jones diffraction operator assembled from canonical wedge terms and per-face TE/TM "
            "material responses; scalar diffraction field is derived from the final receiver projection of the "
            "world-vector result"
        ),
        "diffraction_face_scalarization": jones_metadata["diffraction_face_scalarization"],
    }
    metadata["transport_basis"] = "path_transverse"
    metadata["result_jones_basis"] = jones_metadata["result_jones_basis"]
    metadata["result_jones_axes"] = jones_metadata["result_jones_axes"]
    metadata["scalar_projection_rule"] = jones_metadata["scalar_projection_rule"]
    metadata["reflection_material_model"] = "jones_te_tm"
    metadata["diffraction_material_model"] = "jones_face_operator"
    metadata["material_sources"] = {
        "reflection": reflection_model_source,
        "diffraction": metadata.get("diffraction_face_material_source", "default"),
    }
    metadata["reflection_model_source"] = reflection_model_source
    finite_edge_notes = (
        "Diffraction uses finite-wedge edge bounds throughout scene compilation, state construction, "
        "field evaluation, and radio-map accumulation. Gradient-sensitive traces fall back to the "
        "validated Dr.Jit evaluator when native UTD finite-wedge AD support is unavailable."
    )
    metadata["finite_edge_treatment"] = {
        "mode": "finite_wedge",
        "bounds_required": True,
        "notes": finite_edge_notes,
    }
    metadata["receiver_sampling"] = {
        "monitor_name": monitor.name,
        "monitor_kind": monitor.kind,
        "axis": monitor.axis,
        "tangential_axes": monitor.tangential_axes,
        "plane_position": calculation_height,
        "bounds": monitor.bounds,
        "grid_shape": grid_shape,
        "ray_mode": monitor.ray_mode,
        "ray_sampling": monitor.ray_sampling,
        "resolution_wavelength": effective_resolution,
        "resolution_source": "monitor" if monitor.resolution is not None else "tracer_default",
        "sample_positions": "boundary_points",
        "index_partitioning": "span_over_n_bins",
        "backend": "dda_planar_grid" if monitor.axis == "z" else "axis_aligned_plane_monitor",
        "future_3d_plane_switch_preserved": True,
    }
    metadata["reflection_sampling"] = _monitor_reflection_sampling_metadata(
        reflection_detail,
        axis=monitor.axis,
    )
    metadata["reflection_backend"] = dict(reflection_detail.get("dda_stats", {}))
    if monitor.ray_mode == "3d":
        selected_sampling = metadata["reflection_sampling"].get("selected_ray_sampling")
        metadata["reflection_sampling"]["recommended_ray_count_multiplier_vs_2d"] = (
            5.0 if selected_sampling == "hemisphere_facing_monitor" else 10.0
        )
    metadata["solver_mode"] = solver_controls
    metadata["execution_intent"] = dict(solver_controls["execution_intent"])
    metadata["performance_guardrails"] = _build_performance_guardrails(
        solver_controls,
        dif_components,
    )
    metadata["performance_memory"] = {
        "torch_cuda": capture_cuda_memory_report(),
    }
    metadata["runtime_backends"] = {
        "reflection": dict(metadata["reflection_backend"]),
        "diffraction": dict(metadata.get("diffraction_accumulation_backend", {})),
        "suffix": dict(metadata.get("reflection_suffix_backend", {})),
    }
    if native_extension_available() and not bool(metadata.get("diffraction_skipped", False)):
        metadata["execution_validation_tier"] = "native_strict"
        metadata["execution_validation_note"] = (
            "This end-to-end field trace uses bundled native CUDA custom-op execution "
            "for supported kernels and does not silently downgrade the validated native path."
        )
    return metadata


def _zero_diffraction_components(
    field,
    scene: Scene,
    config: ResolvedTraceConfig,
    effective,
    *,
    reason,
    return_diffraction_audit,
):
    n_rx = field.n_cells
    zero_real = dr.zeros(wt.Float, n_rx)
    zero_imag = dr.zeros(wt.Float, n_rx)
    zero_field = wt.Complex2f(zero_real, zero_imag)
    zero_vector = vector_zero(n_rx)
    if native_extension_available():
        execution_validation_tier = "native_default_dispatch"
        execution_validation_note = (
            "Diffraction solver was skipped before state construction. "
            "When the bundled native extension is available, kernel dispatch still defaults "
            "to the native CUDA path for supported workloads."
        )
    else:
        execution_validation_tier = "validated_drjit"
        execution_validation_note = "Diffraction solver was skipped before state construction."
    solver_metadata = {
        "edge_selection_mode": getattr(scene, "edge_selection_mode", "vertical_only"),
        "boundary_edge_policy": getattr(scene, "boundary_edge_policy", "exclude"),
        "finite_edge_mode": "finite_wedge",
        "finite_edge_bounds_required": True,
            "edge_selection_summary": dict(
                getattr(getattr(scene, "_runtime", scene), "edge_selection_summary", {})
            ),
        "vertical_ratio": None if scene is None else float(scene.vertical_ratio),
        "max_diffractions": int(max(0, effective["max_diffractions"])),
        "reflection_prefix_path_count": 0,
        "reflection_suffix_enabled": False,
        "reflection_suffix_budget": {
            "n_rays": int(effective["reflection_n_rays"] if config.enable_rd_diffraction else 0),
            "max_bounces": int(
                effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0
            ),
        },
        "path_budget_policy": {
            "enabled": False,
            "pruning_metric": "incident_power",
            "total_state_budget_per_order": effective["diffraction_state_budget"],
            "inserted_state_budget_per_order": effective["inserted_reflection_state_budget"],
            "report": {
                "history_size": int(max(1, effective["max_diffractions"])),
                "peak_total_states_before_prune": 0,
                "peak_total_states_after_prune": 0,
                "final_total_states": 0,
                "per_order": (),
            },
        },
        "mixed_chain_budget": {
            "max_inserted_reflections_per_path": (
                0
                if effective["max_inserted_reflections_per_path"] is None
                else int(effective["max_inserted_reflections_per_path"])
            ),
        },
        "execution": config.diffraction_execution.to_dict(),
        "execution_validation_tier": execution_validation_tier,
        "execution_validation_note": execution_validation_note,
        "diffraction_skipped": True,
        "diffraction_skip_reason": str(reason),
    }
    components = {
        "a_direct": zero_field,
        "a_multi": zero_field,
        "polarization_direct": zero_vector,
        "polarization_multi": zero_vector,
        "jones_direct": tangential_jones(zero_vector, axis=field.axis),
        "jones_multi": tangential_jones(zero_vector, axis=field.axis),
        "n_point_sources": 0,
        "n_edge_states": 0,
        "solver_metadata": solver_metadata,
        "state_audit": None,
    }
    if return_diffraction_audit:
        components["state_audit"] = {
            "enabled": False,
            "reason": str(reason),
        }
    return zero_real, zero_imag, [], components


def trace_field_monitor(
    tx_pos,
    monitor: FieldMonitor,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls: Mapping[str, object],
    *,
    reflection_detail: ReflectionTraceDetail | Mapping[str, object] | None = None,
    verbose: bool = False,
    return_timing: bool = False,
    return_diffraction_audit: bool = False,
) -> tuple[dict[str, object], ReflectionTraceDetail]:
    """Run LoS, reflection, and diffraction for a FieldMonitor."""

    timing = {} if return_timing else None
    context = _prepare_field_monitor_trace_context(
        tx_pos,
        monitor,
        scene,
        config,
        solver_controls,
        verbose=verbose,
    )
    calculation_height = context["calculation_height"]
    effective_resolution = context["effective_resolution"]
    field = context["field"]
    grid_shape = context["grid_shape"]
    coords = context["coords"]
    X = context["X"]
    Y = context["Y"]
    rx_positions = context["rx_positions"]
    effective = context["effective"]
    active_rx_polarization = context["active_rx_polarization"]

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    a_los, polarization_los = _compute_los(
        scene,
        rx_positions,
        monitor.axis,
        tx_pos,
        config,
        active_rx_polarization,
    )
    if return_timing:
        EvalSync.and_sync(a_los)
        timing["los"] = time.perf_counter() - t0

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    a_ref_total, _, reflection_detail = _compute_reflection(
        field,
        calculation_height,
        tx_pos,
        scene,
        config,
        effective,
        monitor,
        coords,
        reflection_detail=reflection_detail,
    )
    if return_timing:
        EvalSync.and_sync(reflection_detail["polarization_field_total"])
        timing["reflection"] = time.perf_counter() - t0

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    if effective["max_diffractions"] > 0:
        dif_real, dif_imag, _, dif_components = _compute_diffraction(
            X,
            Y,
            calculation_height,
            tx_pos,
            scene,
            config,
            effective,
            monitor,
            field,
            coords,
            reflection_detail,
            return_diffraction_audit,
        )
    else:
        skip_reason = (
            "max_diffractions_disabled"
            if effective["max_diffractions"] <= 0
            else "diffraction_disabled"
        )
        dif_real, dif_imag, _, dif_components = _zero_diffraction_components(
            field,
            scene,
            config,
            effective,
            reason=skip_reason,
            return_diffraction_audit=return_diffraction_audit,
        )
    if return_timing:
        EvalSync.and_sync(dif_real, dif_imag)
        timing["diffraction"] = time.perf_counter() - t0

    polarization_ref = reflection_detail["polarization_field_total"]
    polarization_dif_direct = dif_components["polarization_direct"]
    polarization_dif_mixed = dif_components["polarization_multi"]
    polarization_dif = vector_add(polarization_dif_direct, polarization_dif_mixed)
    polarization_tot = vector_eval(
        vector_add(vector_add(polarization_los, polarization_ref), polarization_dif)
    )
    a_ref_total = _scalarize_monitor_vector_field(
        polarization_ref,
        axis=monitor.axis,
        rx_polarization=active_rx_polarization,
    )
    # Reuse the diffraction solver's scalar outputs directly so material/scene
    # AD follows the same scalar path as the diffraction accumulator.
    a_dif_direct = eval_complex(dif_components["a_direct"])
    a_dif_multi = eval_complex(dif_components["a_multi"])
    a_dif = eval_complex(a_dif_direct + a_dif_multi)
    a_tot = eval_complex(a_los + a_ref_total + a_dif)
    a_dif = eval_complex(a_dif)
    a_tot = eval_complex(a_tot)
    dr.eval(a_los, a_ref_total, a_dif, a_tot)

    metadata = _build_trace_metadata(
        dif_components,
        reflection_detail,
        solver_controls,
        monitor,
        scene,
        config,
        grid_shape,
        effective_resolution,
        calculation_height,
    )
    if return_timing:
        performance_timing = {
            "los_seconds": float(timing.get("los", 0.0)),
            "reflection_total_seconds": float(timing.get("reflection", 0.0)),
            "diffraction_total_seconds": float(timing.get("diffraction", 0.0)),
        }
        diffraction_timing = dif_components["solver_metadata"].get("timing")
        if isinstance(diffraction_timing, Mapping):
            for key, value in diffraction_timing.items():
                performance_timing[f"diffraction_{key}"] = value
        metadata["performance_timing"] = performance_timing
    result = {
        "name": monitor.name,
        "kind": monitor.kind,
        "axis": monitor.axis,
        "plane_position": calculation_height,
        "ray_mode": monitor.ray_mode,
        "bounds": monitor.bounds,
        "grid_shape": grid_shape,
        "coords": {
            "grid_x": X,
            "grid_y": Y,
            "x": coords["x_coords"],
            "y": coords["y_coords"],
            "axis_x": coords["axis_x"],
            "axis_y": coords["axis_y"],
            "tangential_axes": coords["tangential_axes"],
        },
        "field": {
            "los": a_los,
            "reflection": a_ref_total,
            "diffraction_direct": a_dif_direct,
            "diffraction_mixed": a_dif_multi,
            "diffraction": a_dif,
            "total": a_tot,
        },
        "vector": {
            "los": polarization_los,
            "reflection": polarization_ref,
            "diffraction_direct": polarization_dif_direct,
            "diffraction_mixed": polarization_dif_mixed,
            "diffraction": polarization_dif,
            "total": polarization_tot,
        },
        "jones": {
            "los": tangential_jones(polarization_los, axis=monitor.axis),
            "reflection": tangential_jones(polarization_ref, axis=monitor.axis),
            "diffraction_direct": tangential_jones(polarization_dif_direct, axis=monitor.axis),
            "diffraction_mixed": tangential_jones(polarization_dif_mixed, axis=monitor.axis),
            "diffraction": tangential_jones(polarization_dif, axis=monitor.axis),
            "total": tangential_jones(polarization_tot, axis=monitor.axis),
        },
        "metadata": metadata,
        "tx_pos": (scalar(tx_pos.x), scalar(tx_pos.y), scalar(tx_pos.z)),
    }
    if return_diffraction_audit:
        result["diffraction_detail"] = dif_components
    if return_timing:
        result["timing"] = timing
    return result, reflection_detail


def trace_field_monitor_total_only(
    tx_pos,
    monitor: FieldMonitor,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls: Mapping[str, object],
    *,
    reflection_detail: ReflectionTraceDetail | Mapping[str, object] | None = None,
    verbose: bool = False,
    return_timing: bool = False,
    return_diffraction_audit: bool = False,
) -> tuple[dict[str, object], ReflectionTraceDetail]:
    """Run a FieldMonitor trace and keep only the total scalar field payload."""

    timing = {} if return_timing else None
    context = _prepare_field_monitor_trace_context(
        tx_pos,
        monitor,
        scene,
        config,
        solver_controls,
        verbose=verbose,
    )
    calculation_height = context["calculation_height"]
    effective_resolution = context["effective_resolution"]
    field = context["field"]
    grid_shape = context["grid_shape"]
    coords = context["coords"]
    X = context["X"]
    Y = context["Y"]
    rx_positions = context["rx_positions"]
    effective = context["effective"]
    active_rx_polarization = context["active_rx_polarization"]

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    a_los, _ = _compute_los(
        scene,
        rx_positions,
        monitor.axis,
        tx_pos,
        config,
        active_rx_polarization,
    )
    if return_timing:
        EvalSync.and_sync(a_los)
        timing["los"] = time.perf_counter() - t0

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    a_ref_total, _, reflection_detail = _compute_reflection(
        field,
        calculation_height,
        tx_pos,
        scene,
        config,
        effective,
        monitor,
        coords,
        reflection_detail=reflection_detail,
        include_field_payload=False,
    )
    if return_timing:
        EvalSync.and_sync(a_ref_total)
        timing["reflection"] = time.perf_counter() - t0

    if return_timing:
        EvalSync.sync()
        t0 = time.perf_counter()
    if effective["max_diffractions"] > 0:
        dif_real, dif_imag, _, diffraction_summary = _compute_diffraction_summary(
            X,
            Y,
            calculation_height,
            tx_pos,
            scene,
            config,
            effective,
            monitor,
            field,
            coords,
            reflection_detail,
            return_diffraction_audit,
        )
    else:
        skip_reason = (
            "max_diffractions_disabled"
            if effective["max_diffractions"] <= 0
            else "diffraction_disabled"
        )
        dif_real, dif_imag, _, diffraction_summary = _zero_diffraction_components(
            field,
            scene,
            config,
            effective,
            reason=skip_reason,
            return_diffraction_audit=return_diffraction_audit,
        )
    if return_timing:
        EvalSync.and_sync(dif_real, dif_imag)
        timing["diffraction"] = time.perf_counter() - t0

    a_dif = eval_complex(wt.Complex2f(dif_real, dif_imag))
    a_tot = eval_complex(a_los + a_ref_total + a_dif)
    dr.eval(a_los, a_ref_total, a_dif, a_tot)

    metadata = _build_trace_metadata(
        diffraction_summary,
        reflection_detail,
        solver_controls,
        monitor,
        scene,
        config,
        grid_shape,
        effective_resolution,
        calculation_height,
    )
    if return_timing:
        performance_timing = {
            "los_seconds": float(timing.get("los", 0.0)),
            "reflection_total_seconds": float(timing.get("reflection", 0.0)),
            "diffraction_total_seconds": float(timing.get("diffraction", 0.0)),
        }
        diffraction_timing = diffraction_summary["solver_metadata"].get("timing")
        if isinstance(diffraction_timing, Mapping):
            for key, value in diffraction_timing.items():
                performance_timing[f"diffraction_{key}"] = value
        metadata["performance_timing"] = performance_timing

    result = {
        "name": monitor.name,
        "kind": monitor.kind,
        "axis": monitor.axis,
        "plane_position": calculation_height,
        "ray_mode": monitor.ray_mode,
        "bounds": monitor.bounds,
        "grid_shape": grid_shape,
        "field": {
            "total": a_tot,
        },
        "metadata": metadata,
        "tx_pos": (scalar(tx_pos.x), scalar(tx_pos.y), scalar(tx_pos.z)),
        "payload_kind": "field_total_only",
    }
    if return_diffraction_audit:
        result["diffraction_detail"] = diffraction_summary
    if return_timing:
        result["timing"] = timing
    return result, reflection_detail


__all__ = ["trace_field_monitor", "trace_field_monitor_total_only"]
