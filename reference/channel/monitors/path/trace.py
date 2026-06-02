"""PathMonitor-specific solver orchestration."""

from __future__ import annotations

import time
from typing import Callable, Mapping

import numpy as np
import drjit as dr
import witwin as wt

from .monitor import PathMonitor
from ..common import group_positions_by_z_coordinate
from .result import PathResult
from ...scene import Scene
from ...trace.diffraction.api import _finalize_solver_metadata
from ...trace.diffraction.builders import (
    _build_solver_metadata,
    _prepare_diffraction_state_arrays,
)
from ...trace.materials import ReflectionTraceDetail
from ...trace.diffraction.state import path_export_state_layout
from .collectors import (
    collect_diffraction_state_paths,
    collect_los_paths,
    collect_reflection_paths,
)
from ...types import InteractionType
from ...utils import scalar
from ...utils.polarization import effective_rx_polarization
from ..orchestration import ResolvedTraceConfig
from ..profiler import capture_cuda_memory_report


def _gather_positions(positions, index_array: np.ndarray):
    safe_index = wt.UInt32(index_array.astype(np.uint32, copy=False))
    return wt.Point3f(
        dr.gather(wt.Float, positions.x, safe_index),
        dr.gather(wt.Float, positions.y, safe_index),
        dr.gather(wt.Float, positions.z, safe_index),
    )


def _raw_path_count(raw: Mapping[str, object]) -> int:
    return int(dr.width(raw["rx_index"]))


def _remap_raw_rx_index(raw: dict[str, object], group_indices: np.ndarray):
    if _raw_path_count(raw) == 0:
        return
    mapping = wt.UInt32(group_indices.astype(np.uint32, copy=False))
    raw["rx_index"] = dr.gather(wt.UInt32, mapping, raw["rx_index"])


def _build_path_trace_metadata(
    *,
    monitor: PathMonitor,
    solver_controls: Mapping[str, object],
    reflection_detail: ReflectionTraceDetail | None,
    diffraction_groups,
    rx_positions,
    config: ResolvedTraceConfig,
    return_timing: bool,
    timing,
    path_counts: Mapping[str, int],
    runtime_reuse: Mapping[str, object],
) -> dict[str, object]:
    active_rx_polarization = effective_rx_polarization(
        config.rx_polarization,
        config.tx_polarization,
    )
    metadata: dict[str, object] = {
        "receiver_sampling": {
            "monitor_name": monitor.name,
            "monitor_kind": monitor.kind,
            "receiver_count": int(dr.width(rx_positions.x)),
            "positions_shape": (int(dr.width(rx_positions.x)), 3),
            "ray_mode": monitor.ray_mode,
            "max_num_paths_requested": monitor.max_num_paths,
            "return_geometry": monitor.return_geometry,
        },
        "polarization_transport": {
            "enabled": True,
            "tx_polarization": config.tx_polarization,
            "rx_polarization": active_rx_polarization,
            "rx_polarization_source": (
                "explicit"
                if config.rx_polarization is not None
                else "default_from_tx_polarization"
            ),
            "result_basis": "receiver_polarization_projected_to_arrival_ray",
        },
        "solver_mode": solver_controls,
        "execution_intent": dict(solver_controls["execution_intent"]),
        "interaction_type_codes": {
            "none": InteractionType.NONE,
            "reflection": InteractionType.REFLECTION,
            "diffraction": InteractionType.DIFFRACTION,
            "transmission_reserved": InteractionType.TRANSMISSION,
            "scattering_reserved": InteractionType.SCATTERING,
        },
        "angle_convention": {
            "theta_reference": "zenith_from_+z",
            "phi_reference": "azimuth_from_+x_toward_+y",
            "aod_direction": "tx_to_first_interaction_or_rx",
            "aoa_direction": "last_interaction_to_rx",
        },
        "path_counts": dict(path_counts),
        "diffraction_groups": tuple(diffraction_groups),
        "runtime_reuse": dict(runtime_reuse),
    }
    if reflection_detail is not None:
        metadata["reflection_sampling"] = dict(
            reflection_detail.get("reflection_sampling", {})
        )
    if return_timing and timing is not None:
        metadata["timing"] = dict(timing)
    metadata["performance_memory"] = {
        "torch_cuda": capture_cuda_memory_report(),
    }
    return metadata


def trace_path_monitor(
    tx_pos,
    monitor: PathMonitor,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls: Mapping[str, object],
    *,
    reflection_detail: ReflectionTraceDetail | Mapping[str, object] | None = None,
    persistent_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None = None,
    local_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] | None = None,
    diffraction_state_cache_key_fn: Callable[[float, ReflectionTraceDetail | None], tuple[object, ...]] | None = None,
    verbose: bool = False,
    return_timing: bool = False,
) -> tuple[PathResult, ReflectionTraceDetail]:
    """Run LoS, reflection, and diffraction for a PathMonitor."""

    del verbose
    timing = {} if return_timing else None
    effective = solver_controls["effective"]
    rx_positions = monitor.positions

    if return_timing:
        t0 = time.perf_counter()
    los_raw = collect_los_paths(
        scene=scene,
        rx_positions=rx_positions,
        tx_pos=tx_pos,
        wavelength=config.wavelength,
        k=config.k,
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
    )
    if return_timing:
        timing["los"] = time.perf_counter() - t0

    if return_timing:
        t0 = time.perf_counter()
    reflection_raw, reflection_detail = collect_reflection_paths(
        scene=scene,
        rx_positions=rx_positions,
        tx_pos=tx_pos,
        wavelength=config.wavelength,
        k=config.k,
        n_rays=effective["reflection_n_rays"],
        max_reflections=effective["reflection_max_bounces"],
        mode=monitor.ray_mode,
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        reflection_coef=config.reflection_coef,
        min_ray_contribution_threshold=config.min_ray_contribution_threshold,
        reflection_relative_permittivity=config.reflection_relative_permittivity,
        reflection_conductivity=config.reflection_conductivity,
        reflection_material=config.reflection_material,
        use_scene_materials=config.use_scene_materials_for_reflection,
        return_geometry=monitor.return_geometry,
        reflection_detail=reflection_detail,
    )
    if return_timing:
        timing["reflection"] = time.perf_counter() - t0

    diffraction_raw_collections = []
    diffraction_group_metadata = []
    state_prep_cache_hits = 0
    state_prep_cache_misses = 0
    state_prep_cache_mode = "disabled"
    if return_timing:
        t0 = time.perf_counter()
    if effective["max_diffractions"] > 0:
        mixed_reflection_detail = (
            reflection_detail if config.enable_rd_diffraction else None
        )
        reflection_n_rays = (
            effective["reflection_n_rays"] if config.enable_rd_diffraction else 0
        )
        reflection_max_bounces = (
            effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0
        )
        selected_diffraction_state_cache = local_diffraction_state_cache
        if mixed_reflection_detail is None and persistent_diffraction_state_cache is not None:
            selected_diffraction_state_cache = persistent_diffraction_state_cache
            state_prep_cache_mode = "persistent"
        elif local_diffraction_state_cache is not None:
            state_prep_cache_mode = "local"
        for receiver_z, group_indices in group_positions_by_z_coordinate(rx_positions):
            group_positions = _gather_positions(rx_positions, group_indices)
            path_collection_stats = {}
            cache_key = None
            prepared_state_group = None
            if diffraction_state_cache_key_fn is not None:
                cache_key = diffraction_state_cache_key_fn(receiver_z, mixed_reflection_detail)
            if (
                selected_diffraction_state_cache is not None
                and cache_key is not None
            ):
                prepared_state_group = selected_diffraction_state_cache.get(cache_key)
            cache_hit = prepared_state_group is not None
            if cache_hit:
                state_prep_cache_hits += 1
            else:
                (
                    edge_cache,
                    edge_data,
                    state_arrays,
                    path_budget_report,
                ) = _prepare_diffraction_state_arrays(
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
                    inserted_state_budget_per_order=effective[
                        "inserted_reflection_state_budget"
                    ],
                    max_inserted_reflections_per_path=effective[
                        "max_inserted_reflections_per_path"
                    ],
                    retain_cold_metadata=True,
                    use_scene_materials=config.use_scene_materials_for_diffraction,
                    tx_polarization=config.tx_polarization,
                    solver_mode=solver_controls["selected"],
                    memory_profile=effective["memory_profile"],
                    state_layout="path_export_reduced",
                )
                prepared_state_group = (
                    edge_cache,
                    edge_data,
                    state_arrays,
                    path_budget_report,
                )
                if (
                    selected_diffraction_state_cache is not None
                    and cache_key is not None
                ):
                    selected_diffraction_state_cache[cache_key] = prepared_state_group
                    state_prep_cache_misses += 1
            (
                edge_cache,
                edge_data,
                state_arrays,
                path_budget_report,
            ) = prepared_state_group
            raw = collect_diffraction_state_paths(
                state_arrays=state_arrays,
                edge_data=(
                    edge_data if edge_data is not None else edge_cache.get("edge_data")
                ),
                scene=scene,
                rx_positions=group_positions,
                tx_pos=tx_pos,
                wavelength=config.wavelength,
                k=config.k,
                tx_polarization=config.tx_polarization,
                rx_polarization=config.rx_polarization,
                material_detail=config.diffraction_material,
                return_geometry=monitor.return_geometry,
                stats=path_collection_stats,
            )
            _remap_raw_rx_index(raw, group_indices)
            diffraction_raw_collections.append(raw)
            active_rx_polarization = effective_rx_polarization(
                config.rx_polarization,
                config.tx_polarization,
            )
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
                    inserted_state_budget_per_order=effective[
                        "inserted_reflection_state_budget"
                    ],
                    path_budget_report=path_budget_report,
                ),
                scene=scene,
                material_detail=config.diffraction_material,
                use_scene_materials=config.use_scene_materials_for_diffraction,
                execution=config.diffraction_execution,
                rx_polarization=config.rx_polarization,
                active_rx_polarization=active_rx_polarization,
                receiver_axis="z",
            )
            solver_metadata["receiver_model"] = "discrete_positions_grouped_by_shared_z"
            solver_metadata["receiver_grouping_axis"] = "z"
            diffraction_group_metadata.append(
                {
                    "receiver_height_z": float(receiver_z),
                    "receiver_count": int(group_indices.shape[0]),
                    "receiver_indices": tuple(int(value) for value in group_indices.tolist()),
                    "grouping_axis": "z",
                    "grouping_rule": "shared_z_height_slice",
                    "state_prep_cache_hit": bool(cache_hit),
                    "state_prep_cache_mode": state_prep_cache_mode,
                    "state_layout": path_export_state_layout(state_arrays) or "full",
                    "n_edge_states": int(state_arrays["n_states"]),
                    "n_paths": int(raw["metadata"].get("n_paths", 0)),
                    "path_collection": path_collection_stats,
                    "solver_metadata": solver_metadata,
                }
            )
    if return_timing:
        timing["diffraction"] = time.perf_counter() - t0

    path_counts = {
        "los": int(los_raw["metadata"].get("n_paths", 0)),
        "reflection": int(reflection_raw["metadata"].get("n_paths", 0)),
        "diffraction": int(
            sum(raw["metadata"].get("n_paths", 0) for raw in diffraction_raw_collections)
        ),
    }
    metadata = _build_path_trace_metadata(
        monitor=monitor,
        solver_controls=solver_controls,
        reflection_detail=reflection_detail,
        diffraction_groups=diffraction_group_metadata,
        rx_positions=rx_positions,
        config=config,
        return_timing=return_timing,
        timing=timing,
        path_counts=path_counts,
        runtime_reuse={
            "diffraction_state_prep_cache": {
                "mode": state_prep_cache_mode,
                "hits": int(state_prep_cache_hits),
                "misses": int(state_prep_cache_misses),
                "state_layout": (
                    path_export_state_layout(diffraction_raw_collections[0]["state_arrays"])
                    if diffraction_raw_collections
                    and diffraction_raw_collections[0].get("payload_kind") == "diffraction_state_refs_v1"
                    else "full"
                ),
            },
        },
    )
    return (
        PathResult.from_raw_collections(
            name=monitor.name,
            num_rx=int(dr.width(rx_positions.x)),
            max_num_paths=monitor.max_num_paths,
            tx_pos=(scalar(tx_pos.x), scalar(tx_pos.y), scalar(tx_pos.z)),
            rx_positions=rx_positions,
            frequency=float(config.frequency),
            wavelength=float(config.wavelength),
            raw_collections=[los_raw, reflection_raw, *diffraction_raw_collections],
            return_geometry=monitor.return_geometry,
            metadata=metadata,
        ),
        reflection_detail,
    )


__all__ = ["trace_path_monitor"]
