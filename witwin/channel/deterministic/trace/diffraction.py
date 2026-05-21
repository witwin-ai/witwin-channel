"""Public diffraction phase entry for the deterministic radiomap solver."""

import time

import drjit as dr

from ..config import ReflectionSuffixConfig
from ..kernels.utd import utd_accumulate_forward
from witwin.channel.core.runtime import Material, Rx, Tx, Wave
from witwin.channel.core.physics import polarization
from witwin.channel.core.numerics.arrays import complex_zero, eval_complex, scalar
from ..diffraction import accumulation as diffraction_accumulation
from ..diffraction import builders
from ..diffraction.forward import ForwardEval


def _zero_components(n_rx: int, *, reason: str, metadata: dict[str, object] | None = None):
    zero = complex_zero(n_rx)
    zero_vector = polarization.vector_zero(n_rx)
    return {
        "a_direct": zero,
        "a_multi": zero,
        "a_total": zero,
        "polarization_direct": zero_vector,
        "polarization_multi": zero_vector,
        "polarization": zero_vector,
        "metadata": {
            "diffraction_skipped": True,
            "diffraction_skip_reason": str(reason),
            **dict(metadata or {}),
        },
    }


def _edge_anchor_coordinate(ctx) -> float:
    sample_grid = ctx.sample_grid
    if getattr(sample_grid, "axis", "z") == "z":
        return float(sample_grid.position)
    return float(scalar(ctx.runtime.tx.position.z))


def trace(
    *,
    ctx,
    reflection_detail,
    state_layout: str,
):
    effective = ctx.solver_controls["effective"]
    max_diffractions = int(effective["max_diffractions"])
    preserve_candidate_topology = bool(ctx.grad_preserving) and (
        not ctx.config.enable_rd_diffraction or max_diffractions <= 1
    )
    state_start = time.perf_counter()
    raw_collections = diffraction_accumulation.trace_diffraction_raw_collections(
        runtime=ctx.runtime,
        scene=ctx.scene,
        config=ctx.config,
        solver_controls=ctx.solver_controls,
        spec=ctx.spec,
        reflection_detail=reflection_detail,
        state_layout=state_layout,
        preserve_higher_order_candidate_topology=preserve_candidate_topology,
    )
    state_preparation_seconds = time.perf_counter() - state_start
    accumulation_start = time.perf_counter()
    vector, accumulation_metadata = (
        diffraction_accumulation.baseline_matched_isotropic_diffraction_vector(
            diffraction_raw_collections=raw_collections,
            scene=ctx.scene,
            config=ctx.config,
            n_rx=ctx.n_rx,
            receiver_axis=ctx.spec.axis if ctx.spec.axis is not None else "z",
            sample_grid=ctx.sample_grid,
            ray_mode=ctx.spec.ray_mode,
            return_metadata=True,
        )
    )
    accumulation_seconds = time.perf_counter() - accumulation_start
    state_counts = []
    builder_reports = []
    for raw in raw_collections:
        state_arrays = raw.get("state_arrays")
        state_counts.append(
            0 if state_arrays is None else int(state_arrays["n_states"])
        )
        if "builder_report" in raw:
            builder_reports.append(dict(raw["builder_report"]))
    return {
        "vector": vector,
        "metadata": {
            "state_preparation_seconds": float(state_preparation_seconds),
            "accumulation_seconds": float(accumulation_seconds),
            "raw_collection_count": int(len(raw_collections)),
            "state_counts": tuple(state_counts),
            "state_count_total": int(sum(state_counts)),
            "state_count_max": int(max(state_counts, default=0)),
            "builder_reports": tuple(builder_reports),
            **dict(accumulation_metadata),
        },
    }


def accumulate_components(
    *,
    state_arrays,
    edge_data,
    rx: Rx,
    tx: Tx,
    scene,
    wave: Wave,
    material: Material,
    suffix: ReflectionSuffixConfig,
    execution,
    receiver_axis: str,
    ray_mode: str,
    shadow_support_cutoff_db=None,
):
    n_rx = int(dr.width(rx.positions.x))
    if state_arrays is None or int(state_arrays["n_states"]) <= 0:
        return _zero_components(
            n_rx,
            reason="no_diffraction_states",
            metadata={"n_edge_states": 0},
        )

    _, _, direct_vector_total, multi_vector_total, _ = utd_accumulate_forward(
        state_arrays,
        rx,
        tx,
        edge_data["n_edges"] if edge_data is not None else 0,
        False,
        scene=scene,
        wave=wave,
        material=material,
        receiver_axis=str(receiver_axis),
        execution=execution,
        select_diffraction_point=str(ray_mode) != "2d",
        prefilter_visibility=str(ray_mode) == "2d",
        shadow_support_cutoff_db=shadow_support_cutoff_db,
        return_scalar=False,
    )

    if suffix.enabled:
        _, reflected_suffix_vector = ForwardEval.trace_suffix(
            state_arrays=state_arrays,
            suffix=suffix,
            scene=scene,
            wave=wave,
            tx=tx,
            execution=execution,
        )
        multi_vector_total = polarization.vector_add(
            multi_vector_total,
            reflected_suffix_vector,
        )

    active_rx_polarization = rx.effective_polarization(tx)
    direct_vector_total = polarization.vector_eval(direct_vector_total)
    multi_vector_total = polarization.vector_eval(multi_vector_total)
    total_vector = polarization.vector_eval(
        polarization.vector_add(direct_vector_total, multi_vector_total)
    )
    direct_total = eval_complex(
        polarization.scalarize_tangential_jones(
            polarization.jones_tangential(direct_vector_total, axis=receiver_axis),
            active_rx_polarization,
            axis=receiver_axis,
        )
    )
    multi_total = eval_complex(
        polarization.scalarize_tangential_jones(
            polarization.jones_tangential(multi_vector_total, axis=receiver_axis),
            active_rx_polarization,
            axis=receiver_axis,
        )
    )
    total = eval_complex(
        polarization.scalarize_tangential_jones(
            polarization.jones_tangential(total_vector, axis=receiver_axis),
            active_rx_polarization,
            axis=receiver_axis,
        )
    )
    return {
        "a_direct": direct_total,
        "a_multi": multi_total,
        "a_total": total,
        "polarization_direct": direct_vector_total,
        "polarization_multi": multi_vector_total,
        "polarization": total_vector,
        "metadata": {
            "diffraction_skipped": False,
            "n_edge_states": int(state_arrays["n_states"]),
            "n_edges": 0 if edge_data is None else int(edge_data["n_edges"]),
            "reflection_suffix_enabled": bool(suffix.enabled),
            "reflection_suffix_budget": {
                "n_rays": int(suffix.n_rays),
                "max_bounces": int(suffix.max_bounces),
            },
        },
    }


def trace_components(*, ctx, reflection_detail, return_audit: bool = False):
    n_rx = int(ctx.n_rx)
    effective = ctx.solver_controls["effective"]
    max_diffractions = int(effective["max_diffractions"])
    max_inserted_reflections = effective["max_inserted_reflections_per_path"]
    max_inserted_reflections = (
        max(0, max_diffractions - 1)
        if max_inserted_reflections is None
        else max(0, int(max_inserted_reflections))
    )
    if max_diffractions <= 0:
        return _zero_components(n_rx, reason="max_diffractions_disabled")
    if ctx.scene is None or getattr(ctx.scene, "n_diffraction_edges", 0) <= 0:
        return _zero_components(n_rx, reason="no_diffraction_edges")

    mixed_reflection_detail = reflection_detail if ctx.config.enable_rd_diffraction else None
    reflection_n_rays = (
        int(effective["reflection_n_rays"]) if ctx.config.enable_rd_diffraction else 0
    )
    reflection_max_bounces = (
        int(effective["reflection_max_bounces"]) if ctx.config.enable_rd_diffraction else 0
    )
    edge_cache, edge_data, state_arrays = builders.prepare(
        ctx.runtime.tx,
        _edge_anchor_coordinate(ctx),
        ctx.scene,
        ctx.runtime.wave,
        mixed_reflection_detail,
        ctx.runtime.diffraction,
        reflection_n_rays,
        reflection_max_bounces,
        ctx.runtime.reflection,
        ctx.spec.ray_mode,
        max_diffractions,
        total_state_budget_per_order=effective["diffraction_state_budget"],
        inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
        max_inserted_reflections_per_path=max_inserted_reflections,
        retain_lineage_state=True,
        solver_mode=ctx.solver_controls["selected"],
        memory_profile=effective["memory_profile"],
        state_layout="full",
        preserve_higher_order_candidate_topology=(
            bool(ctx.grad_preserving)
            and (not ctx.config.enable_rd_diffraction or max_diffractions <= 1)
        ),
    )
    suffix = ReflectionSuffixConfig(
        n_rays=reflection_n_rays,
        max_bounces=reflection_max_bounces,
        coef=1.0,
        mode=ctx.spec.ray_mode,
        detail=reflection_detail,
        grid=ctx.sample_grid,
        grid_data=ctx.sample_grid.get_coordinates(),
        rx_z=ctx.sample_grid.position,
    )
    payload = accumulate_components(
        state_arrays=state_arrays,
        edge_data=edge_data if edge_data is not None else edge_cache.get("edge_data"),
        rx=ctx.runtime.rx,
        tx=ctx.runtime.tx,
        scene=ctx.scene,
        wave=ctx.runtime.wave,
        material=ctx.runtime.diffraction,
        suffix=suffix,
        execution=ctx.config.diffraction_execution,
        receiver_axis=ctx.spec.axis,
        ray_mode=ctx.spec.ray_mode,
        shadow_support_cutoff_db=ctx.spec.shadow_support_cutoff_db,
    )
    edge_policy = getattr(ctx.scene, "_edge_policy", None)
    payload["metadata"].update({
        "max_diffractions": max_diffractions,
        "enable_rd_diffraction": bool(ctx.config.enable_rd_diffraction),
        "execution": ctx.config.diffraction_execution.to_dict(),
        "edge_selection_mode": getattr(edge_policy, "edge_selection_mode", "vertical_only"),
        "edge_diffraction": getattr(
            edge_policy,
            "edge_diffraction",
            getattr(edge_policy, "boundary_edge_policy", "exclude") == "half_plane",
        ),
        "boundary_edge_policy": getattr(edge_policy, "boundary_edge_policy", "exclude"),
        "finite_edge_mode": "finite_wedge",
        "path_budget_policy": {
            "total_state_budget_per_order": effective["diffraction_state_budget"],
            "inserted_state_budget_per_order": effective["inserted_reflection_state_budget"],
        },
        "mixed_chain_budget": {
            "max_inserted_reflections_per_path": max_inserted_reflections,
        },
    })
    if return_audit:
        payload["state_audit"] = {
            "n_edge_states": int(state_arrays["n_states"]) if state_arrays is not None else 0,
        }
    return payload


def accumulate_coherent(
    *,
    state_arrays,
    edge_data,
    sample_grid=None,
    rx: Rx,
    tx: Tx,
    scene,
    wave: Wave,
    material: Material,
    suffix: ReflectionSuffixConfig,
    execution,
    return_vector: bool = False,
    return_scalar: bool = True,
    receiver_axis: str | None = None,
    ray_mode: str = "3d",
    shadow_support_cutoff_db=None,
):
    n_rx = int(dr.width(rx.positions.x))
    zero = complex_zero(n_rx)
    zero_vector = polarization.vector_zero(n_rx) if return_vector else None
    if state_arrays is None or int(state_arrays["n_states"]) <= 0:
        return (zero if return_scalar else None), zero_vector

    resolved_receiver_axis = (
        str(receiver_axis) if receiver_axis is not None else str(sample_grid.axis)
    )
    direct_total, multi_total, direct_vector_total, multi_vector_total, _ = (
        utd_accumulate_forward(
            state_arrays,
            rx,
            tx,
            edge_data["n_edges"],
            False,
            scene=scene,
            wave=wave,
            material=material,
            receiver_axis=resolved_receiver_axis,
            execution=execution,
            select_diffraction_point=str(ray_mode) != "2d",
            prefilter_visibility=str(ray_mode) == "2d",
            shadow_support_cutoff_db=shadow_support_cutoff_db,
            return_scalar=return_scalar,
        )
    )

    if suffix.enabled:
        reflected_suffix, reflected_suffix_vector = ForwardEval.trace_suffix(
            state_arrays=state_arrays,
            suffix=suffix,
            scene=scene,
            wave=wave,
            tx=tx,
            execution=execution,
        )
        if return_scalar:
            multi_total = multi_total + reflected_suffix
        multi_vector_total = {
            axis: multi_vector_total[axis] + reflected_suffix_vector[axis]
            for axis in ("x", "y", "z")
        }

    total = direct_total + multi_total if return_scalar else None
    total_vector = (
        None
        if not return_vector
        else {
            axis: direct_vector_total[axis] + multi_vector_total[axis]
            for axis in ("x", "y", "z")
        }
    )
    return total, total_vector


__all__ = ["accumulate_coherent", "accumulate_components", "trace", "trace_components"]
