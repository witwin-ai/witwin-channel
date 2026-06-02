from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping

import drjit as dr
import witwin as wt

from ...profiler import capture_cuda_memory_report
from ...orchestration import ResolvedTraceConfig
from .. import backend as rm_backend
from .. import diagnostics as rm_diag
from .. import metadata as rm_metadata
from ..grid import RadioMapGrid
from ..monitor import RadioMapMonitor
from ..payload import RadioMapPayload
from ....scene import Scene
from ....trace.materials import (
    material_source_label,
    reflection_material_omega,
    reflection_model_label,
)
from ....utils import scalar
from ....utils.drjit_ops import EvalSync
from . import common as mc_common
from . import ad_support as mc_ad
from .custom_op import (
    MC_COMPONENT_NAMES,
    detached_workload,
    trace_with_custom_op,
)
from . import diffraction as mc_diff
from . import reflection as mc_reflection
from .native import native_monte_carlo_ad_available


@dataclass(slots=True)
class _PrimalTrace:
    grid: RadioMapGrid
    component_power: dict[str, object]
    weighted_diagnostics: dict[str, object]
    metadata: dict[str, object]
    noise_power: float
    timing: dict[str, float] | None
    reflection_detail: Mapping[str, object] | None
    diffraction_edge_indices: object | None
    los_tape: mc_ad.LosTape | None
    reflection_tape: mc_ad.ReflectionTape | None
    diffraction_tape: mc_ad.DiffractionTape | None
    diffraction_total_length_weight: float


def _trace_monte_carlo_primal(
    tx_pos,
    monitor: RadioMapMonitor,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls: Mapping[str, object],
    *,
    reflection_detail: Mapping[str, object] | None = None,
    radio_map_accumulation_backend: str = "auto",
    return_timing: bool = False,
    resolved_ad_mode: bool,
    ad_backend: str,
    loop_mode: str,
    collect_ad_tapes: bool = False,
):
    effective = solver_controls["effective"]
    if int(effective["max_diffractions"]) > 1:
        raise RuntimeError(
            "sampling_mode='monte_carlo' currently supports only direct first-order diffraction."
        )

    grid = RadioMapGrid.from_monitor(monitor, default_cell_size=config.cell_size)
    resolved_accumulation_backend = rm_backend._resolve_radio_map_accumulation_backend(
        requested_backend=radio_map_accumulation_backend,
        monitor=monitor,
        grid=grid,
        config=config,
        tx_pos=tx_pos,
        scene=scene,
    )

    n_rx = int(grid.n_cells)
    weighted_diagnostics = rm_diag._empty_radio_map_diagnostics(n_rx)
    samples_per_tx = int(monitor.samples_per_tx)
    cell_area = float(grid.cell_size[0] * grid.cell_size[1])
    diffraction_path_gain_scale = wt.Float(
        (float(config.wavelength) / (4.0 * math.pi)) ** 2 / cell_area
    )
    rr_depth = None if monitor.rr_depth is None else int(monitor.rr_depth)
    rr_prob = float(monitor.rr_prob if monitor.rr_prob is not None else 1.0)
    stop_threshold_linear = mc_common._stop_threshold_linear(monitor.stop_threshold)
    material_omega = reflection_material_omega(config.wavelength)
    timing = {
        "los_seconds": 0.0,
        "reflection_seconds": 0.0,
        "diffraction_seconds": 0.0,
        "state_preparation_seconds": 0.0,
        "scatter_seconds": 0.0,
    } if return_timing else None

    total_start = None
    if return_timing:
        EvalSync.sync()
        total_start = time.perf_counter()

    ray_sampling_metadata = mc_common._sionna_full_sphere_sampling_metadata(
        axis=str(grid.axis),
        plane_position=float(grid.position),
        tx_pos=tx_pos,
    )
    solid_angle_per_ray = mc_common._solid_angle_per_ray(ray_sampling_metadata, samples_per_tx)
    reflection_runtime_backend = {
        "implementation": "tx_emitted_rays_drjit_symbolic_loop_plus_scatter_reduce",
        "cell_scatter_backend": "drjit_scatter_reduce",
        "point_evaluation_backend": "direct_image_source_field",
        "ray_distribution": str(ray_sampling_metadata.get("selected_ray_sampling", "")),
        "source_field_contract": "sionna_iso_v_implicit_basis",
        "reflection_model": reflection_model_label(
            scene,
            config.reflection_material,
            use_scene_materials=config.use_scene_materials_for_reflection,
        ),
        "material_source": material_source_label(
            scene,
            config.reflection_material,
            use_scene_materials=config.use_scene_materials_for_reflection,
        ),
        "loop_mode": str(loop_mode),
    }
    collect_diffraction_wedges = (
        int(effective["max_diffractions"]) > 0
    )
    cuda_memory_report = capture_cuda_memory_report(
        release_reclaimable_caches=collect_diffraction_wedges,
    )
    batch_plan = mc_common._resolve_monte_carlo_batch_plan(
        samples_per_tx=samples_per_tx,
        diffraction_state_count=0,
        cuda_memory_report=cuda_memory_report,
    )
    ray_batch_size = int(batch_plan["ray_batch_size"])

    accepted_hit_counts = {
        "los": wt.UInt32(0),
        "reflection": wt.UInt32(0),
        "diffraction": wt.UInt32(0),
    }
    path_tape_store = (
        mc_reflection.PathTapeStore(
            samples_per_tx=samples_per_tx,
            max_bounces=int(effective["reflection_max_bounces"]),
        )
        if collect_ad_tapes
        else None
    )
    diffraction_tape_store = None
    diffraction_total_length_weight = 0.0

    direct_tx_diffraction_state_store = None
    if collect_diffraction_wedges:
        direct_tx_diffraction_state_store = mc_diff.DirectTxDiffractionStateStore.for_scene(scene)
    for batch_start in range(0, samples_per_tx, ray_batch_size):
        current_batch_size = min(ray_batch_size, samples_per_tx - batch_start)
        ray_index = dr.arange(wt.UInt32, current_batch_size) + wt.UInt32(batch_start)
        ray_dir = mc_common._generate_sionna_full_sphere_directions(
            samples_per_tx,
            ray_index=ray_index,
        )
        batch_los_hits, batch_reflection_hits, direct_tx_diffraction_state_store = (
            mc_reflection.trace_reflection(
                scene=scene,
                grid=grid,
                tx_pos=tx_pos,
                ray_index=ray_index,
                ray_dir=ray_dir,
                config=config,
                solid_angle_per_ray=float(solid_angle_per_ray),
                cell_area=float(cell_area),
                max_bounces=int(effective["reflection_max_bounces"]),
                seed=int(monitor.seed),
                rr_depth=rr_depth,
                rr_prob=rr_prob,
                stop_threshold_linear=stop_threshold_linear,
                material_omega=material_omega,
                weighted_diagnostics=weighted_diagnostics,
                collect_diffraction_wedges=collect_diffraction_wedges,
                direct_tx_diffraction_state_store=direct_tx_diffraction_state_store,
                path_tape_store=path_tape_store,
                loop_mode=loop_mode,
            )
        )
        accepted_hit_counts["los"] += batch_los_hits
        accepted_hit_counts["reflection"] += batch_reflection_hits

    runtime_reuse = {
        "cache_mode": "disabled",
        "state_preparation_hits": 0,
        "state_preparation_misses": 0,
        "state_layout": "depth0_in_loop_state_store",
    }
    state_pool = {
        "total": 0,
        "kept": 0,
        "threshold_pruned": 0,
        "roulette_pruned": 0,
    }
    diffraction_runtime_backend = mc_diff._direct_tx_diffraction_runtime_backend(
        implementation="disabled",
        wedge_discovery_backend="disabled",
    )
    diffraction_states = None
    diffraction_edge_indices = None

    if collect_diffraction_wedges:
        stored_state_count = (
            0
            if direct_tx_diffraction_state_store is None
            else direct_tx_diffraction_state_store.count()
        )
        diffraction_states = (
            None
            if direct_tx_diffraction_state_store is None
            else direct_tx_diffraction_state_store.state_arrays
        )
        diffraction_edge_indices = (
            None
            if diffraction_states is None
            else dr.gather(
                wt.Int32,
                diffraction_states.edge_index,
                dr.arange(wt.UInt32, int(stored_state_count)),
            )
        )
        if diffraction_states is not None:
            diffraction_states.set_stored_count(stored_state_count)
        state_pool = {
            "total": int(stored_state_count),
            "kept": int(stored_state_count),
            "threshold_pruned": 0,
            "roulette_pruned": 0,
        }
        diffraction_runtime_backend = mc_diff._direct_tx_diffraction_runtime_backend(
            implementation="depth0_in_loop_wedge_state_store_plus_keller_cone_symbolic_loop_scatter_reduce",
            wedge_discovery_backend="depth0_direct_hit_in_loop_triangle_surface_edge_candidates_unique_state_store",
        )
        diffraction_runtime_backend["loop_mode"] = str(loop_mode)

    cuda_memory_report = capture_cuda_memory_report(
        release_reclaimable_caches=collect_diffraction_wedges,
    )
    n_diffraction_states = 0 if diffraction_states is None else diffraction_states.width()
    batch_plan = mc_common._resolve_monte_carlo_batch_plan(
        samples_per_tx=samples_per_tx,
        diffraction_state_count=n_diffraction_states,
        cuda_memory_report=cuda_memory_report,
    )

    if n_diffraction_states > 0:
        sampler = mc_diff.LengthProportionalStateSampler.from_line_length(
            diffraction_states.line_lengths()
        )
        if sampler is not None:
            diffraction_total_length_weight = float(
                scalar(sampler.total_length_weight(samples_per_tx=samples_per_tx))
            )
            diffraction_tape_store = (
                mc_diff.DiffractionTapeStore.for_samples(samples_per_tx)
                if collect_ad_tapes
                else None
            )
            plane_normal = mc_common._axis_unit_normal(str(grid.axis))
            diffraction_batch_size = max(1, int(batch_plan["diffraction_batch_size"]))
            if return_timing and timing is not None:
                EvalSync.sync()
                t0 = time.perf_counter()
            accepted_hit_counts["diffraction"] += mc_diff._trace_diffraction_batches_symbolic(
                scene=scene,
                grid=grid,
                diffraction_states=diffraction_states,
                sampler=sampler,
                diffraction_batch_size=diffraction_batch_size,
                diffraction_batch_count=int(batch_plan["diffraction_batch_count"]),
                samples_per_tx=samples_per_tx,
                seed=int(monitor.seed),
                k=config.k,
                wavelength=config.wavelength,
                plane_normal=plane_normal,
                diffraction_path_gain_scale=diffraction_path_gain_scale,
                weighted_diagnostics=weighted_diagnostics,
                diffraction_tape_store=diffraction_tape_store,
                loop_mode=loop_mode,
            )
            if return_timing and timing is not None:
                EvalSync.sync()
                timing["diffraction_seconds"] += time.perf_counter() - t0

    accepted_hit_counts_scalar = {
        component: int(scalar(value))
        for component, value in accepted_hit_counts.items()
    }
    for component in MC_COMPONENT_NAMES:
        if accepted_hit_counts_scalar[component] <= 0:
            accepted_hit_counts_scalar[component] = int(
                scalar(
                    dr.count(weighted_diagnostics["incoherent"][component] > wt.Float(0.0))
                )
            )

    rm_diag._finalize_radio_map_component_totals(weighted_diagnostics)
    noise_power, noise_power_source = rm_metadata._resolve_noise_power(scene, monitor)
    path_counts = {
        "los": int(accepted_hit_counts_scalar["los"]),
        "reflection": int(accepted_hit_counts_scalar["reflection"]),
        "diffraction": int(accepted_hit_counts_scalar["diffraction"]),
        "total": int(sum(int(value) for value in accepted_hit_counts_scalar.values())),
    }
    metadata = {
        "sampling_mode": "monte_carlo",
        "receiver_sampling": {
            "monitor_name": monitor.name,
            "monitor_kind": monitor.kind,
            "sampling_mode": "monte_carlo",
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
            "quadrature_mode": None,
            "samples_per_cell": None,
            "sample_offsets_local": (),
            "samples_per_tx": int(monitor.samples_per_tx),
            "seed": int(monitor.seed),
            "strategy": "tx_emitted_rays",
        },
        "ray_sampling": {
            "samples_per_tx": int(monitor.samples_per_tx),
            "seed": int(monitor.seed),
            "batch_size": int(batch_plan["ray_batch_size"]),
            "batch_count": int(batch_plan["ray_batch_count"]),
            "batch_policy": str(batch_plan["ray_policy"]),
            "diffraction_batch_size": int(batch_plan["diffraction_batch_size"]),
            "diffraction_batch_count": int(batch_plan["diffraction_batch_count"]),
            "diffraction_batch_policy": str(batch_plan["diffraction_policy"]),
            "scatter_safe_batch_cap": int(batch_plan["scatter_safe_batch_cap"]),
            "distribution": str(ray_sampling_metadata.get("selected_ray_sampling", "")),
            "sequence": str(ray_sampling_metadata.get("sampling_sequence", "")),
            "solid_angle_per_ray_sr": float(solid_angle_per_ray),
        },
        "metric_contract": {
            "path_gain": "tx_emitted_ray_driven_monte_carlo_cell_average_of_matched_isotropic_incoherent_power",
            "rss": "tx_power_times_path_gain",
            "sinr": "rss_over_noise_plus_other_tx_rss",
        },
        "metric": monitor.metric,
        "combine_mode": monitor.combine_mode,
        "receiver_model": monitor.receiver_model,
        "tx_power": float(monitor.tx_power),
        "noise_power": float(noise_power),
        "noise_power_source": str(noise_power_source),
        "path_counts": path_counts,
        "runtime_reuse": {
            "diffraction_state_prep_cache": {
                "mode": str(runtime_reuse["cache_mode"]),
                "hits": int(runtime_reuse["state_preparation_hits"]),
                "misses": int(runtime_reuse["state_preparation_misses"]),
                "state_layout": str(runtime_reuse["state_layout"]),
            },
        },
        "accumulation_backend": {
            "requested": str(monitor.accumulation_backend),
            "resolved": str(resolved_accumulation_backend),
            "cell_accumulation_mode": "direct_in_loop_scatter",
        },
        "solver_mode": solver_controls,
        "execution_intent": dict(solver_controls["execution_intent"]),
        "runtime_backends": {
            "reflection": dict(reflection_runtime_backend),
            "diffraction": dict(diffraction_runtime_backend),
            "suffix": {
                "requested_backend": "disabled_for_monte_carlo",
                "resolved_backend": "disabled_for_monte_carlo",
                "implementation": "disabled",
            },
        },
        "monte_carlo": {
            "ad_mode": bool(resolved_ad_mode),
            "ad_backend": str(ad_backend),
            "primal_loop_mode": str(loop_mode),
            "tx_accumulation_transport_mode": (
                "fixed_tape_full_replay_central_difference"
                if resolved_ad_mode
                else "disabled"
            ),
            "tx_accumulation_transport_step": (
                float(mc_ad._MC_TX_TRANSPORT_FD_STEP)
                if resolved_ad_mode
                else 0.0
            ),
            "tape_layout_version": (
                "single_solver_native_sparse_coeff_tape_v3"
                if resolved_ad_mode
                else "disabled"
            ),
            "samples_per_tx": int(monitor.samples_per_tx),
            "seed": int(monitor.seed),
            "rr_depth": rr_depth,
            "rr_prob": float(rr_prob),
            "stop_threshold_db": float(monitor.stop_threshold or 0.0),
            "accepted_hit_counts": dict(accepted_hit_counts_scalar),
            "state_pool": dict(state_pool),
        },
    }
    if return_timing and timing is not None and total_start is not None:
        EvalSync.sync()
        timing["total_seconds"] = time.perf_counter() - total_start

    los_tape = None
    reflection_tape = None
    diffraction_tape = None
    if collect_ad_tapes:
        max_bounces = int(effective["reflection_max_bounces"])
        if path_tape_store is None:
            los_tape = mc_ad.empty_los_tape()
            reflection_tape = mc_ad.empty_reflection_tape(max_bounces)
        else:
            tape_payload = path_tape_store.finalize()
            los_tape = mc_ad.LosTape(
                ray_dir=tape_payload["los"]["ray_dir"],
                cell_idx=tape_payload["los"]["cell_idx"],
                transport_ray_dir=tape_payload["los"]["transport_ray_dir"],
                transport_blocker_prim_idx=tape_payload["los"]["transport_blocker_prim_idx"],
            )
            reflection_tape = mc_ad.ReflectionTape(
                initial_ray_dir=tape_payload["reflection"]["initial_ray_dir"],
                blocker_dist=tape_payload["reflection"]["blocker_dist"],
                cell_idx=tape_payload["reflection"]["cell_idx"],
                depth=tape_payload["reflection"]["depth"],
                prim_index_by_bounce=tape_payload["reflection"]["prim_index_by_bounce"],
                transport_initial_ray_dir=tape_payload["reflection"]["transport_initial_ray_dir"],
                transport_depth=tape_payload["reflection"]["transport_depth"],
                transport_blocker_prim_idx=tape_payload["reflection"]["transport_blocker_prim_idx"],
                transport_prim_index_by_bounce=tape_payload["reflection"]["transport_prim_index_by_bounce"],
            )
        if diffraction_tape_store is None:
            diffraction_tape = mc_ad.empty_diffraction_tape()
        else:
            diffraction_payload = diffraction_tape_store.finalize()
            diffraction_tape = mc_ad.DiffractionTape(
                edge_index=diffraction_payload["edge_index"],
                edge_fraction=diffraction_payload["edge_fraction"],
                cone_sample=diffraction_payload["cone_sample"],
                cell_idx=diffraction_payload["cell_idx"],
                field_valid=diffraction_payload["field_valid"],
                pole_safe=diffraction_payload["pole_safe"],
                dif_n_p=diffraction_payload["dif_n_p"],
                dif_n_m=diffraction_payload["dif_n_m"],
                sum_n_p=diffraction_payload["sum_n_p"],
                sum_n_m=diffraction_payload["sum_n_m"],
            )

    return _PrimalTrace(
        grid=grid,
        component_power={
            component: weighted_diagnostics["incoherent"][component]
            for component in MC_COMPONENT_NAMES
        },
        weighted_diagnostics=weighted_diagnostics,
        metadata=metadata,
        noise_power=float(noise_power),
        timing=timing,
        reflection_detail=reflection_detail,
        diffraction_edge_indices=diffraction_edge_indices,
        los_tape=los_tape,
        reflection_tape=reflection_tape,
        diffraction_tape=diffraction_tape,
        diffraction_total_length_weight=float(diffraction_total_length_weight),
    )

def trace(
    tx_pos,
    monitor: RadioMapMonitor,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls: Mapping[str, object],
    *,
    reflection_detail: Mapping[str, object] | None = None,
    persistent_diffraction_state_cache: dict[tuple[object, ...], object] | None = None,
    local_diffraction_state_cache: dict[tuple[object, ...], object] | None = None,
    diffraction_state_cache_key_fn: Callable[[float, object | None], tuple[object, ...]] | None = None,
    radio_map_accumulation_backend: str = "auto",
    return_timing: bool = False,
    return_reflection_detail: bool = False,
):
    # Trace the transmitter-driven Monte Carlo radio-map path on a single implementation.
    del persistent_diffraction_state_cache
    del local_diffraction_state_cache
    del diffraction_state_cache_key_fn
    grad_sensitive_workload = rm_backend._radio_map_grad_sensitive_workload(
        config,
        tx_pos=tx_pos,
        scene=scene,
    )
    resolved_ad_mode = (
        bool(grad_sensitive_workload)
        if monitor.ad is None
        else bool(monitor.ad)
    )

    if resolved_ad_mode:
        if not native_monte_carlo_ad_available():
            raise RuntimeError(
                "RadioMapMonitor(sampling_mode='monte_carlo', ad=True) requires the Monte Carlo "
                "native AD kernels. Rebuild the witwin.channel native extension."
            )
        payload, resolved_reflection_detail = trace_with_custom_op(
            tx_pos=tx_pos,
            monitor=monitor,
            scene=scene,
            config=config,
            solver_controls=solver_controls,
            reflection_detail=reflection_detail,
            radio_map_accumulation_backend=radio_map_accumulation_backend,
            return_timing=return_timing,
            grad_sensitive_workload=grad_sensitive_workload,
            trace_primal_fn=_trace_monte_carlo_primal,
        )
    else:
        if grad_sensitive_workload:
            detached = detached_workload(scene, tx_pos, config)
            tx_pos = detached["tx_pos"]
            scene = detached["scene"]
        primal_state = _trace_monte_carlo_primal(
            tx_pos,
            monitor,
            scene,
            config,
            solver_controls,
            reflection_detail=reflection_detail,
            radio_map_accumulation_backend=radio_map_accumulation_backend,
            return_timing=return_timing,
            resolved_ad_mode=False,
            ad_backend="disabled",
            loop_mode="symbolic",
            collect_ad_tapes=False,
        )
        payload = RadioMapPayload(
            monitor=monitor,
            grid=primal_state.grid,
            weighted_diagnostics=primal_state.weighted_diagnostics,
            metadata=primal_state.metadata,
            path_gain=primal_state.weighted_diagnostics["incoherent"]["total"],
            rss=primal_state.weighted_diagnostics["incoherent"]["total"] * float(monitor.tx_power),
            sinr=rm_diag._single_tx_sinr(
                primal_state.weighted_diagnostics["incoherent"]["total"] * float(monitor.tx_power),
                noise_power=float(primal_state.noise_power),
            ),
            tx_pos=tx_pos,
            noise_power=float(primal_state.noise_power),
            sample_payload_positions=(),
            timing=primal_state.timing,
        )
        resolved_reflection_detail = primal_state.reflection_detail

    if return_reflection_detail:
        return payload, resolved_reflection_detail
    return payload


__all__ = ["trace"]
