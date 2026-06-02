"""Public diffraction solver APIs built on the modular diffraction package."""

import time

import drjit as dr
import witwin as wt
from ...monitors.field import Field
from ...utils.plane_axes import normalize_axis, tangential_axes_for_axis
from ...utils.polarization import (
    effective_rx_polarization,
    scalarize_tangential_jones,
    tangential_jones,
    vector_eval,
    vector_zero,
)
from ..materials import material_source_label, reflection_model_label
from .builders import (
    _build_solver_metadata,
    _count_point_sources,
    _prepare_diffraction_state_arrays,
)
from ...config import coerce_diffraction_execution, ReflectionSuffixConfig
from ...utils.drjit_ops import ArrayInit, EvalSync, eval_complex
from ...kernels.monitors.common.receiver_tiles import resolve_receiver_tiles
from ...kernels.trace.utd import utd_accumulate_forward as _accumulate_edge_states_to_receivers
from .state import _build_state_audit, _empty_state_audit, _state_lineage_first_edge_idx
from ..._native import native_extension_available
from witwin.channel.kernels.trace.packed_state import subset_state_arrays
from .suffix import trace_reflected_suffix_from_edge_states


def _resolve_receiver_axis(receiver_axis, grid):
    if grid is not None:
        return normalize_axis(grid.axis)
    return normalize_axis(receiver_axis)


def _resolve_receiver_positions(*, X, Y, rx_z, rx_pos, grid):
    if rx_pos is not None:
        return rx_pos
    if grid is not None and isinstance(grid, Field):
        if rx_z is None:
            return grid.receivers
        return grid.receiver_positions_3d(position=float(rx_z))
    if X is None or Y is None or rx_z is None:
        raise ValueError("Diffraction receiver positions require either rx_pos or X/Y/rx_z inputs.")
    return wt.Point3f(X, Y, wt.Float(rx_z))


def _resolve_receiver_plane_position(*, rx_z, grid):
    if grid is None:
        return None
    if rx_z is not None:
        return float(rx_z)
    return getattr(grid, "position", None)


def _diffraction_jones_metadata(receiver_axis, *, explicit_receiver_projection):
    axis_name = normalize_axis(receiver_axis)
    if axis_name == "z":
        return {
            "result_jones_basis": "global_xy",
            "result_jones_axes": ("x", "y"),
            "scalar_projection_rule": (
                "global_xy_receiver_projection_from_jones"
                if explicit_receiver_projection
                else "global_xy_default_receiver_projection_from_jones"
            ),
            "diffraction_face_scalarization": (
                "explicit_receiver_projection_from_jones_face_operator"
                if explicit_receiver_projection
                else "default_receiver_projection_from_jones_face_operator"
            ),
        }
    return {
        "result_jones_basis": "monitor_tangential",
        "result_jones_axes": tangential_axes_for_axis(axis_name),
        "scalar_projection_rule": (
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


def _finalize_solver_metadata(
    solver_metadata,
    scene,
    material_detail,
    use_scene_materials,
    execution,
    rx_polarization,
    active_rx_polarization,
    receiver_axis,
):
    """Add common metadata fields shared by both diffraction solver entry points."""
    native_available = bool(native_extension_available())
    solver_metadata["diffraction_face_material_source"] = material_source_label(
        scene,
        material_detail,
        use_scene_materials=use_scene_materials,
    )
    solver_metadata["diffraction_face_reflection_model"] = reflection_model_label(
        scene,
        material_detail,
        use_scene_materials=use_scene_materials,
    )
    solver_metadata["execution"] = execution.to_dict()
    solver_metadata["diffraction_accumulation_backend"] = {
        "native_extension_available": native_available,
        "requested_primal": execution.accumulate_primal,
        "requested_jvp": execution.accumulate_jvp,
        "requested_backward": execution.accumulate_backward,
        "resolved_primal": (
            "native_cuda_custom_op"
            if native_available
            else "drjit_reference"
        ),
        "resolved_jvp": (
            "native_inner_custom_op_forward_mode"
            if native_available
            else "drjit_replay"
        ),
        "resolved_backward": (
            "native_inner_custom_op_backward"
            if native_available
            else "drjit_replay"
        ),
        "implementation": (
            "native_cuda_custom_op"
            if native_available
            else "drjit_reference"
        ),
    }
    solver_metadata["reflection_suffix_backend"] = {
        "requested_backend": execution.suffix_backend,
        "resolved_backend": execution.suffix_backend,
        "implementation": (
            "native_cuda_custom_op"
            if execution.suffix_backend == "native"
            else f"drjit_{execution.suffix_dda}"
        ),
        "native_ad_mode": (
            "drjit_custom_op_forward_backward"
            if execution.suffix_backend == "native"
            else None
        ),
    }
    suffix_enabled = bool(solver_metadata.get("reflection_suffix_enabled", False))
    if execution.suffix_backend == "native" and suffix_enabled:
        solver_metadata["execution_validation_tier"] = "native_strict"
        solver_metadata["execution_validation_note"] = (
            "Reflected suffix grid accumulation is pinned to the bundled native CUDA "
            "CustomOp implementation for primal, forward-mode, and reverse-mode AD. "
            "This path does not silently fall back to the Dr.Jit suffix accumulator."
        )
    elif native_extension_available():
        solver_metadata["execution_validation_tier"] = "native_default_dispatch"
        solver_metadata["execution_validation_note"] = (
            "Supported non-suffix kernels continue to use native CUDA dispatch, "
            "while reflected suffix accumulation stays on the requested Dr.Jit reference backend."
        )
    else:
        solver_metadata["execution_validation_tier"] = "validated_drjit"
        solver_metadata["execution_validation_note"] = (
            "Validated Jones diffraction path uses strict Dr.Jit accumulation/replay "
            "with symbolic suffix traversal."
        )
    solver_metadata["rx_polarization"] = active_rx_polarization
    solver_metadata["rx_polarization_source"] = (
        "explicit" if rx_polarization is not None else "default_from_tx_polarization"
    )
    solver_metadata["transport_basis"] = "path_transverse"
    jones_metadata = _diffraction_jones_metadata(
        receiver_axis,
        explicit_receiver_projection=rx_polarization is not None,
    )
    solver_metadata["result_jones_basis"] = jones_metadata["result_jones_basis"]
    solver_metadata["result_jones_axes"] = jones_metadata["result_jones_axes"]
    solver_metadata["scalar_projection_rule"] = jones_metadata["scalar_projection_rule"]
    solver_metadata["diffraction_face_scalarization"] = jones_metadata["diffraction_face_scalarization"]
    return solver_metadata


def _accumulate_state_subset_field(
    state_arrays,
    rx_pos,
    scene,
    wavelength,
    k,
    n_edges,
    material_detail,
    suffix,
    tx_polarization=(1.0, 0.0, 0.0),
    rx_polarization=None,
    receiver_axis="z",
    execution=None,
    return_timing: bool = False,
    receiver_tiles=None,
):
    n_rx = dr.width(rx_pos.x)
    if state_arrays is None or state_arrays["n_states"] == 0:
        zero = ArrayInit.complex_zero(n_rx)
        if return_timing:
            return zero, {
                "utd_accumulation_seconds": 0.0,
                "suffix_seconds": 0.0,
                "postprocess_seconds": 0.0,
            }
        return zero
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)

    timing = None
    if return_timing:
        timing = {
            "utd_accumulation_seconds": 0.0,
            "suffix_seconds": 0.0,
            "postprocess_seconds": 0.0,
        }

    if return_timing:
        EvalSync.sync()
    t0 = time.perf_counter()
    _, _, direct_vector_total, multi_vector_total, _ = _accumulate_edge_states_to_receivers(
        state_arrays,
        rx_pos,
        k,
        n_edges,
        False,
        scene=scene,
        wavelength=wavelength,
        material_detail=material_detail,
        rx_polarization=active_rx_polarization,
        receiver_axis=receiver_axis,
        execution=execution,
        receiver_tiles=receiver_tiles if native_extension_available() else None,
    )
    if return_timing:
        EvalSync.barrier(direct_vector_total, multi_vector_total)
        timing["utd_accumulation_seconds"] = time.perf_counter() - t0
    if suffix.enabled:
        suffix_receiver_tiles = receiver_tiles
        if suffix_receiver_tiles is None:
            suffix_receiver_tiles = resolve_receiver_tiles(
                grid=suffix.grid,
                plane_position=_resolve_receiver_plane_position(rx_z=suffix.rx_z, grid=suffix.grid),
                grid_data=suffix.grid_data,
            )
        if return_timing:
            EvalSync.sync()
        t0 = time.perf_counter()
        _, reflected_suffix_vector = trace_reflected_suffix_from_edge_states(
            state_arrays=state_arrays,
            suffix=suffix,
            scene=scene,
            wavelength=wavelength,
            k=k,
            tx_polarization=tx_polarization,
            execution=execution,
            receiver_tiles=suffix_receiver_tiles,
        )
        multi_vector_total = vector_eval({
            "x": multi_vector_total["x"] + reflected_suffix_vector["x"],
            "y": multi_vector_total["y"] + reflected_suffix_vector["y"],
            "z": multi_vector_total["z"] + reflected_suffix_vector["z"],
        })
        if return_timing:
            EvalSync.barrier(reflected_suffix_vector, multi_vector_total)
            timing["suffix_seconds"] = time.perf_counter() - t0

    if return_timing:
        EvalSync.sync()
    t0 = time.perf_counter()
    direct_total = eval_complex(
        scalarize_tangential_jones(
            tangential_jones(direct_vector_total, axis=receiver_axis),
            active_rx_polarization,
            axis=receiver_axis,
        )
    )
    multi_total = eval_complex(
        scalarize_tangential_jones(
            tangential_jones(multi_vector_total, axis=receiver_axis),
            active_rx_polarization,
            axis=receiver_axis,
        )
    )
    total = eval_complex(direct_total + multi_total)
    if return_timing:
        EvalSync.and_sync(total, direct_total, multi_total)
        timing["postprocess_seconds"] = time.perf_counter() - t0
        return total, timing
    return total


def _diffraction_summary_payload(
    *,
    solver_metadata,
    n_point_sources: int,
    n_edge_states: int,
    state_audit,
):
    return {
        "n_point_sources": int(n_point_sources),
        "n_edge_states": int(n_edge_states),
        "solver_metadata": solver_metadata,
        "state_audit": state_audit,
    }


def compute_diffraction_order_breakdown(
    X,
    Y,
    rx_z,
    tx_pos,
    scene,
    wavelength,
    k,
    reflection_detail=None,
    max_diffractions=2,
    reflection_n_rays=0,
    reflection_max_bounces=0,
    reflection_coef=0.7,
    reflection_mode="2d",
    grid=None,
    grid_data=None,
    split_by_edge=False,
    diffraction_material=None,
    use_scene_materials=False,
    total_state_budget_per_order=None,
    inserted_state_budget_per_order=None,
    max_inserted_reflections_per_path=None,
    tx_polarization=(1.0, 0.0, 0.0),
    rx_polarization=None,
    rx_pos=None,
    receiver_axis="z",
    edge_anchor_coordinate=None,
    execution=None,
    solver_mode="accuracy",
    memory_profile="default",
):
    execution = coerce_diffraction_execution(execution)
    max_order = max(1, int(max_diffractions))
    receiver_axis = _resolve_receiver_axis(receiver_axis, grid)
    rx_pos = _resolve_receiver_positions(X=X, Y=Y, rx_z=rx_z, rx_pos=rx_pos, grid=grid)
    n_rx = dr.width(rx_pos.x)
    anchor_coordinate = float(rx_z if edge_anchor_coordinate is None else edge_anchor_coordinate)

    effective_inserted_reflection_budget = (
        max(0, max_order - 1)
        if max_inserted_reflections_per_path is None
        else max(0, int(max_inserted_reflections_per_path))
    )
    inserted_reflection_enabled = (
        scene is not None
        and reflection_n_rays > 0
        and reflection_max_bounces > 0
        and max_order > 1
        and effective_inserted_reflection_budget > 0
    )
    material_detail = diffraction_material
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
    suffix = ReflectionSuffixConfig(
        n_rays=reflection_n_rays,
        max_bounces=reflection_max_bounces,
        coef=reflection_coef,
        mode=reflection_mode,
        detail=reflection_detail,
        grid=grid,
        grid_data=grid_data,
        rx_z=rx_z,
    )
    edge_cache, edge_data, state_arrays, path_budget_report = _prepare_diffraction_state_arrays(
        tx_pos,
        anchor_coordinate,
        scene,
        wavelength,
        k,
        reflection_detail,
        material_detail,
        reflection_n_rays,
        reflection_max_bounces,
        reflection_coef,
        reflection_mode,
        max_order,
        use_scene_materials=use_scene_materials,
        total_state_budget_per_order=total_state_budget_per_order,
        inserted_state_budget_per_order=inserted_state_budget_per_order,
        max_inserted_reflections_per_path=effective_inserted_reflection_budget,
        retain_cold_metadata=True,
        tx_polarization=tx_polarization,
        solver_mode=solver_mode,
        memory_profile=memory_profile,
    )
    solver_metadata = _finalize_solver_metadata(
        _build_solver_metadata(
            scene=scene,
            max_diffractions=max_order,
            reflection_detail=reflection_detail,
            reflection_n_rays=reflection_n_rays,
            reflection_max_bounces=reflection_max_bounces,
            reflected_suffix_enabled=suffix.enabled,
            inserted_reflection_enabled=inserted_reflection_enabled,
            max_inserted_reflections_per_path=effective_inserted_reflection_budget,
            total_state_budget_per_order=total_state_budget_per_order,
            inserted_state_budget_per_order=inserted_state_budget_per_order,
            path_budget_report=path_budget_report,
        ),
        scene=scene,
        material_detail=material_detail,
        use_scene_materials=use_scene_materials,
        execution=execution,
        rx_polarization=rx_polarization,
        active_rx_polarization=active_rx_polarization,
        receiver_axis=receiver_axis,
    )

    zero_field = ArrayInit.complex_zero(n_rx)
    n_edges = 0 if edge_data is None else edge_data["n_edges"]
    order_fields = []
    order_edge_fields = [] if split_by_edge else None
    order_first_edge_fields = [] if split_by_edge else None

    for order in range(1, max_order + 1):
        order_states = subset_state_arrays(state_arrays, state_arrays["order"] == wt.UInt32(order))
        order_field = _accumulate_state_subset_field(
            order_states, rx_pos, scene, wavelength, k, n_edges,
            material_detail, suffix, tx_polarization, active_rx_polarization, receiver_axis, execution,
        ) if order_states["n_states"] > 0 else zero_field
        order_fields.append(order_field)

        if split_by_edge:
            per_edge_fields = []
            per_first_edge_fields = []
            first_edge_idx = _state_lineage_first_edge_idx(state_arrays)
            for edge_idx_scalar in range(n_edges):
                edge_mask = (state_arrays["order"] == wt.UInt32(order)) & (
                    state_arrays["edge_idx"] == wt.UInt32(edge_idx_scalar)
                )
                edge_states = subset_state_arrays(state_arrays, edge_mask)
                edge_field = _accumulate_state_subset_field(
                    edge_states, rx_pos, scene, wavelength, k, n_edges,
                    material_detail, suffix, tx_polarization, active_rx_polarization, receiver_axis, execution,
                ) if edge_states["n_states"] > 0 else zero_field
                per_edge_fields.append(edge_field)
                if order == 1:
                    first_edge_mask = edge_mask
                else:
                    first_edge_mask = (state_arrays["order"] == wt.UInt32(order)) & (
                        first_edge_idx == wt.Int32(edge_idx_scalar)
                    )
                first_edge_states = subset_state_arrays(state_arrays, first_edge_mask)
                first_edge_field = _accumulate_state_subset_field(
                    first_edge_states, rx_pos, scene, wavelength, k, n_edges,
                    material_detail, suffix, tx_polarization, active_rx_polarization, receiver_axis, execution,
                ) if first_edge_states["n_states"] > 0 else zero_field
                per_first_edge_fields.append(first_edge_field)
            order_edge_fields.append(per_edge_fields)
            order_first_edge_fields.append(per_first_edge_fields)

    return {
        "edge_cache": edge_cache,
        "edge_data": edge_data,
        "solver_metadata": solver_metadata,
        "state_audit": _build_state_audit(state_arrays, edge_data) if edge_data is not None else _empty_state_audit(max_order),
        "order_fields": order_fields,
        "order_edge_fields": order_edge_fields,
        "order_first_edge_fields": order_first_edge_fields,
    }


def compute_diffraction_field(
    X,
    Y,
    rx_z,
    tx_pos,
    scene,
    wavelength,
    k,
    reflection_detail=None,
    max_diffractions=1,
    reflection_n_rays=0,
    reflection_max_bounces=0,
    reflection_coef=0.7,
    reflection_mode="2d",
    grid=None,
    grid_data=None,
    return_components=False,
    return_per_edge=True,
    return_solver_metadata=False,
    return_state_audit=False,
    diffraction_material=None,
    use_scene_materials=False,
    total_state_budget_per_order=None,
    inserted_state_budget_per_order=None,
    max_inserted_reflections_per_path=None,
    tx_polarization=(1.0, 0.0, 0.0),
    rx_polarization=None,
    rx_pos=None,
    receiver_axis="z",
    edge_anchor_coordinate=None,
    execution=None,
    solver_mode="accuracy",
    memory_profile="default",
):
    """
    Compute diffraction field using unified edge states.

    Path families covered by this implementation:
    - Exact for the currently selected diffraction-edge set: S -> D
    - Approximate reflection-prefix diffraction: R^n -> D
    - Approximate higher-order diffraction up to max_diffractions: S/R -> D -> D
    - Approximate reflection suffix after the last diffraction: ... -> D -> R^n

    Args:
        diffraction_material: Optional explicit wedge-face material override.
            When omitted, direct diffraction uses the default dielectric
            wedge-face material unless `use_scene_materials=True`.
        use_scene_materials: When True, specified `Structure.material` values
            drive per-face Fresnel wedge coefficients. This is enabled by
            default through `TraceConfig`.
    """
    execution = coerce_diffraction_execution(execution)
    EvalSync.sync()
    total_start = time.perf_counter()
    receiver_axis = _resolve_receiver_axis(receiver_axis, grid)
    rx_pos = _resolve_receiver_positions(X=X, Y=Y, rx_z=rx_z, rx_pos=rx_pos, grid=grid)
    receiver_plane_position = _resolve_receiver_plane_position(rx_z=rx_z, grid=grid)
    receiver_tiles = resolve_receiver_tiles(
        grid=grid,
        plane_position=receiver_plane_position,
        grid_data=grid_data,
        receiver_positions=rx_pos,
    )
    n_rx = dr.width(rx_pos.x)
    max_order = max(1, int(max_diffractions))
    anchor_coordinate = float(rx_z if edge_anchor_coordinate is None else edge_anchor_coordinate)

    effective_inserted_reflection_budget = (
        max(0, max_order - 1)
        if max_inserted_reflections_per_path is None
        else max(0, int(max_inserted_reflections_per_path))
    )
    inserted_reflection_enabled = (
        scene is not None
        and reflection_n_rays > 0
        and reflection_max_bounces > 0
        and max_order > 1
        and effective_inserted_reflection_budget > 0
    )
    material_detail = diffraction_material
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
    suffix = ReflectionSuffixConfig(
        n_rays=reflection_n_rays,
        max_bounces=reflection_max_bounces,
        coef=reflection_coef,
        mode=reflection_mode,
        detail=reflection_detail,
        grid=grid,
        grid_data=grid_data,
        rx_z=rx_z,
    )
    timing_report = {
        "state_preparation_seconds": 0.0,
        "utd_accumulation_seconds": 0.0,
        "suffix_seconds": 0.0,
        "postprocess_seconds": 0.0,
        "total_seconds": 0.0,
    }
    EvalSync.sync()
    t0 = time.perf_counter()
    edge_cache, edge_data, state_arrays, path_budget_report = _prepare_diffraction_state_arrays(
        tx_pos,
        anchor_coordinate,
        scene,
        wavelength,
        k,
        reflection_detail,
        material_detail,
        reflection_n_rays,
        reflection_max_bounces,
        reflection_coef,
        reflection_mode,
        max_order,
        use_scene_materials=use_scene_materials,
        total_state_budget_per_order=total_state_budget_per_order,
        inserted_state_budget_per_order=inserted_state_budget_per_order,
        max_inserted_reflections_per_path=effective_inserted_reflection_budget,
        retain_cold_metadata=bool(return_state_audit),
        tx_polarization=tx_polarization,
        solver_mode=solver_mode,
        memory_profile=memory_profile,
        collect_timing=bool(return_solver_metadata or return_components),
    )
    EvalSync.barrier(edge_data, state_arrays)
    timing_report["state_preparation_seconds"] = time.perf_counter() - t0
    solver_metadata = _finalize_solver_metadata(
        _build_solver_metadata(
            scene=scene,
            max_diffractions=max_order,
            reflection_detail=reflection_detail,
            reflection_n_rays=reflection_n_rays,
            reflection_max_bounces=reflection_max_bounces,
            reflected_suffix_enabled=suffix.enabled,
            inserted_reflection_enabled=inserted_reflection_enabled,
            max_inserted_reflections_per_path=effective_inserted_reflection_budget,
            total_state_budget_per_order=total_state_budget_per_order,
            inserted_state_budget_per_order=inserted_state_budget_per_order,
            path_budget_report=path_budget_report,
        ),
        scene=scene,
        material_detail=material_detail,
        use_scene_materials=use_scene_materials,
        execution=execution,
        rx_polarization=rx_polarization,
        active_rx_polarization=active_rx_polarization,
        receiver_axis=receiver_axis,
    )
    builder_timing = path_budget_report.get("timing")
    if isinstance(builder_timing, dict):
        timing_report["state_preparation_breakdown"] = dict(builder_timing)
    order_reports = path_budget_report.get("per_order")
    if isinstance(order_reports, list):
        timing_report["state_preparation_order_reports"] = list(order_reports)
    candidate_backend = path_budget_report.get("higher_order_candidate_backend")
    if candidate_backend is not None:
        timing_report["higher_order_candidate_backend"] = candidate_backend
    if edge_data is None:
        EvalSync.sync()
        timing_report["total_seconds"] = time.perf_counter() - total_start
        solver_metadata["timing"] = timing_report
        zero_real = dr.zeros(wt.Float, n_rx)
        zero_imag = dr.zeros(wt.Float, n_rx)
        zero_summary = _diffraction_summary_payload(
            solver_metadata=solver_metadata,
            n_point_sources=0,
            n_edge_states=0,
            state_audit=_empty_state_audit(max_order) if return_state_audit else None,
        )
        if return_components:
            zero_field = wt.Complex2f(zero_real, zero_imag)
            zero_vector = vector_zero(n_rx)
            return zero_real, zero_imag, [], {
                "a_direct": zero_field,
                "a_multi": zero_field,
                "polarization_direct": zero_vector,
                "polarization_multi": zero_vector,
                "jones_direct": tangential_jones(zero_vector, axis=receiver_axis),
                "jones_multi": tangential_jones(zero_vector, axis=receiver_axis),
                "n_point_sources": 0,
                "n_edge_states": 0,
                "solver_metadata": solver_metadata,
                "state_audit": _empty_state_audit(max_order) if return_state_audit else None,
            }
        if return_solver_metadata:
            return zero_real, zero_imag, [], zero_summary
        return zero_real, zero_imag, []

    if state_arrays["n_states"] == 0:
        EvalSync.sync()
        timing_report["total_seconds"] = time.perf_counter() - total_start
        solver_metadata["timing"] = timing_report
        zero_real = dr.zeros(wt.Float, n_rx)
        zero_imag = dr.zeros(wt.Float, n_rx)
        zero_field = wt.Complex2f(zero_real, zero_imag)
        zero_vector = vector_zero(n_rx)
        per_edge_list = []
        if return_per_edge:
            per_edge_list = [(zero_real, zero_imag) for _ in range(edge_data["n_edges"])]
        zero_summary = _diffraction_summary_payload(
            solver_metadata=solver_metadata,
            n_point_sources=_count_point_sources(reflection_detail),
            n_edge_states=0,
            state_audit=_build_state_audit(state_arrays, edge_data) if return_state_audit else None,
        )
        if return_components:
            return zero_real, zero_imag, per_edge_list, {
                "a_direct": zero_field,
                "a_multi": zero_field,
                "polarization_direct": zero_vector,
                "polarization_multi": zero_vector,
                "jones_direct": tangential_jones(zero_vector, axis=receiver_axis),
                "jones_multi": tangential_jones(zero_vector, axis=receiver_axis),
                "n_point_sources": _count_point_sources(reflection_detail),
                "n_edge_states": 0,
                "solver_metadata": solver_metadata,
                "state_audit": _build_state_audit(state_arrays, edge_data) if return_state_audit else None,
            }
        if return_solver_metadata:
            return zero_real, zero_imag, per_edge_list, zero_summary
        return zero_real, zero_imag, per_edge_list

    if return_solver_metadata and not return_components and not return_per_edge:
        EvalSync.sync()
        t0 = time.perf_counter()
        total, scalar_timing = _accumulate_state_subset_field(
            state_arrays,
            rx_pos,
            scene,
            wavelength,
            k,
            edge_data["n_edges"],
            material_detail,
            suffix,
            tx_polarization=tx_polarization,
            rx_polarization=active_rx_polarization,
            receiver_axis=receiver_axis,
            execution=execution,
            return_timing=True,
            receiver_tiles=receiver_tiles,
        )
        timing_report["utd_accumulation_seconds"] = float(scalar_timing["utd_accumulation_seconds"])
        timing_report["suffix_seconds"] = float(scalar_timing["suffix_seconds"])
        timing_report["postprocess_seconds"] = float(scalar_timing["postprocess_seconds"])
        EvalSync.sync()
        timing_report["total_seconds"] = time.perf_counter() - total_start
        solver_metadata["timing"] = timing_report
        return total.real, total.imag, [], _diffraction_summary_payload(
            solver_metadata=solver_metadata,
            n_point_sources=_count_point_sources(reflection_detail),
            n_edge_states=state_arrays["n_states"],
            state_audit=_build_state_audit(state_arrays, edge_data) if return_state_audit else None,
        )

    EvalSync.sync()
    t0 = time.perf_counter()
    direct_total, multi_total, direct_vector_total, multi_vector_total, per_edge_list = _accumulate_edge_states_to_receivers(
        state_arrays,
        rx_pos,
        k,
        edge_data["n_edges"],
        return_per_edge,
        scene=scene,
        wavelength=wavelength,
        material_detail=material_detail,
        rx_polarization=active_rx_polarization,
        receiver_axis=receiver_axis,
        execution=execution,
        receiver_tiles=receiver_tiles if native_extension_available() else None,
    )
    EvalSync.barrier(direct_vector_total, multi_vector_total)
    timing_report["utd_accumulation_seconds"] = time.perf_counter() - t0

    if suffix.enabled:
        EvalSync.sync()
        t0 = time.perf_counter()
        reflected_suffix, reflected_suffix_vector = trace_reflected_suffix_from_edge_states(
            state_arrays=state_arrays,
            suffix=suffix,
            scene=scene,
            wavelength=wavelength,
            k=k,
            tx_polarization=tx_polarization,
            execution=execution,
            receiver_tiles=receiver_tiles,
        )
        reflected_suffix = eval_complex(reflected_suffix)
        reflected_suffix_vector = vector_eval(reflected_suffix_vector)
        EvalSync.barrier(reflected_suffix, reflected_suffix_vector)
        timing_report["suffix_seconds"] = time.perf_counter() - t0
        multi_total = eval_complex(multi_total + reflected_suffix)
        multi_vector_total = vector_eval({
            "x": multi_vector_total["x"] + reflected_suffix_vector["x"],
            "y": multi_vector_total["y"] + reflected_suffix_vector["y"],
            "z": multi_vector_total["z"] + reflected_suffix_vector["z"],
        })

    EvalSync.sync()
    t0 = time.perf_counter()
    total_vector = vector_eval({
        "x": direct_vector_total["x"] + multi_vector_total["x"],
        "y": direct_vector_total["y"] + multi_vector_total["y"],
        "z": direct_vector_total["z"] + multi_vector_total["z"],
    })
    direct_total = eval_complex(
        scalarize_tangential_jones(
            tangential_jones(direct_vector_total, axis=receiver_axis),
            active_rx_polarization,
            axis=receiver_axis,
        )
    )
    multi_total = eval_complex(
        scalarize_tangential_jones(
            tangential_jones(multi_vector_total, axis=receiver_axis),
            active_rx_polarization,
            axis=receiver_axis,
        )
    )
    total = eval_complex(
        scalarize_tangential_jones(
            tangential_jones(total_vector, axis=receiver_axis),
            active_rx_polarization,
            axis=receiver_axis,
        )
    )

    EvalSync.and_sync(total, direct_total, multi_total)
    timing_report["postprocess_seconds"] = time.perf_counter() - t0
    EvalSync.sync()
    timing_report["total_seconds"] = time.perf_counter() - total_start
    solver_metadata["timing"] = timing_report

    if return_components:
        return total.real, total.imag, per_edge_list, {
            "a_direct": direct_total,
            "a_multi": multi_total,
            "polarization_direct": direct_vector_total,
            "polarization_multi": multi_vector_total,
            "jones_direct": tangential_jones(direct_vector_total, axis=receiver_axis),
            "jones_multi": tangential_jones(multi_vector_total, axis=receiver_axis),
            "n_point_sources": _count_point_sources(reflection_detail),
            "n_edge_states": state_arrays["n_states"],
            "solver_metadata": solver_metadata,
            "state_audit": _build_state_audit(state_arrays, edge_data) if return_state_audit else None,
        }
    if return_solver_metadata:
        return total.real, total.imag, per_edge_list, _diffraction_summary_payload(
            solver_metadata=solver_metadata,
            n_point_sources=_count_point_sources(reflection_detail),
            n_edge_states=state_arrays["n_states"],
            state_audit=_build_state_audit(state_arrays, edge_data) if return_state_audit else None,
        )
    return total.real, total.imag, per_edge_list


__all__ = [
    "_accumulate_state_subset_field",
    "compute_diffraction_field",
    "compute_diffraction_order_breakdown",
]
