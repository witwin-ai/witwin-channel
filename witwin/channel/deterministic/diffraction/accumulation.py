"""Receiver-side diffraction state grouping and native coherent accumulation."""

from __future__ import annotations

import drjit as dr
import math
import numpy as np
from witwin.channel.core.scene import Scene
from witwin.channel.deterministic import types as wt

from ..config import ReflectionSuffixConfig, ResolvedTraceConfig, SolveSpec
from witwin.channel.core.runtime import Rx
from witwin.channel.core.physics.polarization import vector_zero
from . import builders


_MEMORY_SAFE_RECEIVER_TILE_SIZE = 32768


def gather_group_positions(positions, group_indices: np.ndarray):
    safe_index = wt.UInt32(group_indices.astype(np.uint32, copy=False))
    return wt.Point3f(
        dr.gather(wt.Float, positions.x, safe_index),
        dr.gather(wt.Float, positions.y, safe_index),
        dr.gather(wt.Float, positions.z, safe_index),
    )


def group_positions_by_z_coordinate(positions) -> list[tuple[float, np.ndarray]]:
    z_values = np.asarray(positions.z, dtype=np.float32)
    if z_values.size == 0:
        return []
    quantized_z = np.rint(z_values * 1e6).astype(np.int64)
    ordered_groups = []
    for group_key in np.unique(quantized_z):
        group_indices = np.nonzero(quantized_z == group_key)[0].astype(np.int64, copy=False)
        ordered_groups.append((float(z_values[group_indices[0]]), group_indices))
    return ordered_groups


def receiver_position_groups(*, runtime, spec) -> list[tuple[float, object, wt.UInt32]]:
    positions = runtime.rx.positions
    n_rx = int(dr.width(positions.x))
    if n_rx == 0:
        return []

    if (
        str(getattr(spec, "axis", "")).lower() == "z"
        and str(getattr(spec, "surface_mode", "")).lower().endswith("axis_aligned")
    ):
        return [(
            float(getattr(spec, "position")),
            positions,
            dr.arange(wt.UInt32, n_rx),
        )]

    return [
        (
            receiver_z,
            gather_group_positions(positions, group_indices),
            wt.UInt32(group_indices.astype(np.uint32, copy=False)),
        )
        for receiver_z, group_indices in group_positions_by_z_coordinate(positions)
    ]


def receiver_tile_plan(config: ResolvedTraceConfig, *, n_rx: int) -> dict[str, object]:
    n_rx = max(0, int(n_rx))
    if n_rx == 0:
        return {"enabled": False, "tile_size": 0, "tile_count": 0}
    if str(getattr(config, "memory_profile", "default")) != "memory_safe":
        return {"enabled": False, "tile_size": n_rx, "tile_count": 1}

    requested_tile_size = int(
        getattr(config, "receiver_tile_size", _MEMORY_SAFE_RECEIVER_TILE_SIZE)
    )
    tile_size = max(1, min(n_rx, requested_tile_size))
    tile_count = int(math.ceil(n_rx / tile_size))
    return {
        "enabled": tile_count > 1,
        "tile_size": tile_size,
        "tile_count": tile_count,
    }


def _gather_receiver_tile(positions, receiver_index_map, *, start: int, count: int):
    tile_idx = dr.arange(wt.UInt32, int(count)) + wt.UInt32(int(start))
    tile_positions = wt.Point3f(
        dr.gather(wt.Float, positions.x, tile_idx),
        dr.gather(wt.Float, positions.y, tile_idx),
        dr.gather(wt.Float, positions.z, tile_idx),
    )
    tile_receiver_index_map = dr.gather(wt.UInt32, receiver_index_map, tile_idx)
    return tile_positions, tile_receiver_index_map


def baseline_matched_isotropic_diffraction_vector(
    *,
    diffraction_raw_collections,
    scene: Scene,
    config: ResolvedTraceConfig,
    n_rx: int,
    receiver_axis: str,
    sample_grid=None,
    ray_mode: str = "3d",
    return_metadata: bool = False,
    rx_positions=None,
):
    """Accumulate prepared diffraction states onto receiver positions.

    ``rx_positions`` (full-width, indexed by ``receiver_index_map``) lets one
    state preparation be replayed against multiple quadrature sample sets.
    When omitted, each collection's build-time positions are used.
    """
    from witwin.channel.deterministic.trace import diffraction
    diffraction_vector_coherent = vector_zero(n_rx)
    tile_reports = []
    used_rayd_exact_grid_accumulation = False
    for raw in diffraction_raw_collections:
        receiver_index_map = raw.get("receiver_index_map")
        state_arrays = raw.get("state_arrays")
        if rx_positions is None:
            local_positions = raw.get("rx_positions")
        elif receiver_index_map is not None:
            index = wt.UInt32(receiver_index_map)
            local_positions = wt.Point3f(
                dr.gather(wt.Float, rx_positions.x, index),
                dr.gather(wt.Float, rx_positions.y, index),
                dr.gather(wt.Float, rx_positions.z, index),
            )
        else:
            local_positions = None
        if receiver_index_map is None or local_positions is None or state_arrays is None:
            continue
        local_rx_count = int(dr.width(local_positions.x))
        plan = receiver_tile_plan(config, n_rx=local_rx_count)
        # Reflected diffraction suffix (D->R...) splats directly onto the full
        # grid, so it requires an axis-aligned grid covering this collection
        # in a single tile.
        suffix = ReflectionSuffixConfig()
        suffix_budget = raw.get("suffix_budget") or {}
        if (
            bool(getattr(config, "enable_rd_diffraction", False))
            and sample_grid is not None
            and int(plan["tile_count"]) <= 1
            and local_rx_count == int(sample_grid.n_cells)
            and int(suffix_budget.get("n_rays", 0)) > 0
            and int(suffix_budget.get("max_bounces", 0)) > 0
        ):
            suffix = ReflectionSuffixConfig(
                n_rays=int(suffix_budget["n_rays"]),
                max_bounces=int(suffix_budget["max_bounces"]),
                coef=1.0,
                mode=str(ray_mode),
                detail=raw.get("reflection_detail"),
                grid=sample_grid,
                grid_data=sample_grid.get_coordinates(),
                rx_z=sample_grid.position,
            )
        accumulate_mode = str(getattr(config.diffraction_execution, "accumulate_primal", "auto"))
        rayd_exact_auto_allowed = (
            accumulate_mode == "auto"
            and sample_grid is not None
            and config.shadow_support_cutoff_db is None
        )
        rayd_exact_grid_accumulation = accumulate_mode == "rayd_exact_coherent" or (
            rayd_exact_auto_allowed
            and diffraction._rayd_exact_coherent_auto_supported(
                state_arrays=state_arrays,
                sample_grid=sample_grid,
                scene=scene,
                tx=raw["runtime"].tx,
                suffix=suffix,
                shadow_support_cutoff_db=config.shadow_support_cutoff_db,
                allow_rayd_exact_coherent_auto=True,
            )
        )
        if rayd_exact_grid_accumulation:
            used_rayd_exact_grid_accumulation = True
            plan = {"enabled": False, "tile_size": local_rx_count, "tile_count": 1}
        tile_reports.append({
            "receiver_count": int(local_rx_count),
            "tile_size": int(plan["tile_size"]),
            "tile_count": int(plan["tile_count"]),
            "enabled": bool(plan["enabled"]),
        })
        tile_size = int(plan["tile_size"])
        if tile_size <= 0:
            continue

        def accumulate_tile(tile_positions, tile_receiver_index_map) -> None:
            _, local_vector = diffraction.accumulate_coherent(
                state_arrays=state_arrays,
                edge_data=raw.get("edge_data"),
                sample_grid=sample_grid,
                rx=Rx(
                    positions=tile_positions,
                    polarization=config.rx_polarization,
                ),
                tx=raw["runtime"].tx,
                scene=scene,
                wave=raw["runtime"].wave,
                material=raw["runtime"].diffraction,
                suffix=suffix,
                execution=config.diffraction_execution,
                return_vector=True,
                return_scalar=False,
                receiver_axis=str(receiver_axis),
                ray_mode=ray_mode,
                shadow_support_cutoff_db=config.shadow_support_cutoff_db,
                allow_rayd_exact_coherent_auto=rayd_exact_auto_allowed,
            )
            if local_vector is not None:
                for axis in ("x", "y", "z"):
                    dr.scatter_reduce(
                        dr.ReduceOp.Add,
                        diffraction_vector_coherent[axis].real,
                        local_vector[axis].real,
                        tile_receiver_index_map,
                    )
                    dr.scatter_reduce(
                        dr.ReduceOp.Add,
                        diffraction_vector_coherent[axis].imag,
                        local_vector[axis].imag,
                        tile_receiver_index_map,
                    )

        if not bool(plan["enabled"]):
            accumulate_tile(local_positions, receiver_index_map)
            continue

        for tile_start in range(0, local_rx_count, tile_size):
            tile_count = min(tile_size, local_rx_count - tile_start)
            tile_positions, tile_receiver_index_map = _gather_receiver_tile(
                local_positions,
                receiver_index_map,
                start=tile_start,
                count=tile_count,
            )
            accumulate_tile(tile_positions, tile_receiver_index_map)
    if not return_metadata:
        return diffraction_vector_coherent
    metadata = {
        "receiver_tiling_enabled": any(report["enabled"] for report in tile_reports),
        "receiver_tile_size": (
            max((int(report["tile_size"]) for report in tile_reports), default=0)
        ),
        "receiver_tile_count": int(sum(int(report["tile_count"]) for report in tile_reports)),
        "receiver_tile_reports": tuple(tile_reports),
    }
    if (
        str(getattr(config.diffraction_execution, "accumulate_primal", "auto"))
        == "rayd_exact_coherent"
        or used_rayd_exact_grid_accumulation
    ):
        metadata.update({
            "implementation": "rayd_accum_dfr_coherent_direct_exact",
            "coherence": "complex_vector_sum",
            "estimator": "exact_state_receiver_sum",
            "ad_contract": "primal_non_ad_only",
        })
    return diffraction_vector_coherent, metadata


def trace_diffraction_raw_collections(
    *,
    runtime,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls,
    spec: SolveSpec,
    reflection_detail,
    state_layout: str,
    preserve_higher_order_candidate_topology: bool = False,
):
    effective = solver_controls["effective"]
    if effective["max_diffractions"] <= 0:
        return []

    mixed_reflection_detail = reflection_detail if config.enable_rd_diffraction else None
    reflection_n_rays = effective["reflection_n_rays"] if config.enable_rd_diffraction else 0
    reflection_max_bounces = (
        effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0
    )
    raw_collections = []

    for receiver_z, group_positions, receiver_index_map in receiver_position_groups(
        runtime=runtime,
        spec=spec,
    ):
        group_runtime = runtime.with_rx(
            group_positions,
            polarization=config.rx_polarization,
        )
        edge_cache, edge_data, state_arrays, builder_report = builders.prepare(
            group_runtime.tx,
            receiver_z,
            scene,
            group_runtime.wave,
            mixed_reflection_detail,
            group_runtime.diffraction,
            reflection_n_rays,
            reflection_max_bounces,
            group_runtime.reflection,
            effective["max_diffractions"],
            total_state_budget_per_order=effective["diffraction_state_budget"],
            inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
            max_inserted_reflections_per_path=effective["max_inserted_reflections_per_path"],
            retain_lineage_state=True,
            solver_mode=solver_controls["selected"],
            memory_profile=effective["memory_profile"],
            state_layout=state_layout,
            preserve_higher_order_candidate_topology=bool(
                preserve_higher_order_candidate_topology
            ),
            return_report=True,
        )
        raw_collections.append({
            "runtime": group_runtime,
            "receiver_index_map": receiver_index_map,
            "rx_positions": group_positions,
            "state_arrays": state_arrays,
            "edge_data": edge_data if edge_data is not None else edge_cache.get("edge_data"),
            "builder_report": builder_report,
            "reflection_detail": mixed_reflection_detail,
            "suffix_budget": {
                "n_rays": int(reflection_n_rays),
                "max_bounces": int(reflection_max_bounces),
            },
        })
    return raw_collections


__all__ = [
    "baseline_matched_isotropic_diffraction_vector",
    "receiver_tile_plan",
    "receiver_position_groups",
    "trace_diffraction_raw_collections",
]
