"""Public solve entry for the deterministic radiomap package."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import time

import drjit as dr
from witwin.channel.core.scene import ReceiverGrid, Scene, Transmitter

from . import config as cfg, types as wt
from .config import Config, SolveSpec
from witwin.channel.core.grid import Grid, GridSpec
from .grid_ops import NativeGrid, build_grid
from .trace import diffraction, los, reflection
from .diffraction import postprocessing
from .diffraction.state import PATH_EXPORT_REDUCED_STATE_LAYOUT
from .reflection.common import grad_sensitive_workload
from witwin.channel.core.runtime import TraceContext, TraceCtx, assert_scene_materials_complete
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.physics.polarization import vector_power, vector_zero
from witwin.channel.core.results import (
    DEFAULT_RAY_MODE,
    RadioMapFieldPayload,
    RadioMapResult,
    RayMode,
    coordinates_from_grid,
    normalize_ray_mode,
    stack_radiomap_results,
)
from witwin.channel.core.numerics.tensors import to_float_tensor, to_int_tensor, to_mapping_proxy


ComplexVector = dict[str, wt.Complex2f]


_TIMING_KEYS = (
    "grid_resolution_seconds",
    "los_trace_seconds",
    "reflection_trace_seconds",
    "diffraction_state_preparation_seconds",
    "diffraction_accumulation_seconds",
    "shadow_boundary_correction_seconds",
    "result_shaping_seconds",
    "total_solve_seconds",
)


def _new_timing() -> dict[str, float]:
    return {key: 0.0 for key in _TIMING_KEYS}


def _scene_summary(scene: Scene, config: Config) -> dict[str, object]:
    tri_data = getattr(scene, "tri_data", None)
    policy = config.edge_policy
    return {
        "n_structures": int(len(getattr(scene, "structures", ()) or ())),
        "n_triangles": None if tri_data is None else int(tri_data["n_triangles"]),
        "n_diffraction_edges": scene.diffraction_edge_count(edge_policy=policy),
        "edge_selection_mode": policy.edge_selection_mode,
        "edge_diffraction": policy.edge_diffraction,
        "boundary_edge_policy": policy.boundary_edge_policy,
        "device": getattr(scene, "device", None),
    }


def _build_diffraction_metadata(
    prep_metadata: Mapping[str, object],
    sample_metadata: list[dict[str, object]],
) -> dict[str, object]:
    receiver_tile_count = 0
    receiver_tile_size = 0
    receiver_tiling_enabled = False
    for metadata in sample_metadata:
        receiver_tile_count += int(metadata.get("receiver_tile_count", 0))
        receiver_tile_size = max(
            receiver_tile_size,
            int(metadata.get("receiver_tile_size", 0)),
        )
        receiver_tiling_enabled = receiver_tiling_enabled or bool(
            metadata.get("receiver_tiling_enabled", False)
        )
    return {
        "samples": tuple(sample_metadata),
        "sample_count": int(len(sample_metadata)),
        "raw_collection_count": int(prep_metadata.get("raw_collection_count", 0)),
        "state_counts": tuple(prep_metadata.get("state_counts", ())),
        "state_count_total": int(prep_metadata.get("state_count_total", 0)),
        "state_count_max": int(prep_metadata.get("state_count_max", 0)),
        "builder_reports": tuple(prep_metadata.get("builder_reports", ())),
        "receiver_tiling_enabled": bool(receiver_tiling_enabled),
        "receiver_tile_size": int(receiver_tile_size),
        "receiver_tile_count": int(receiver_tile_count),
    }


def _build_metadata(
    *,
    scene: Scene,
    resolved: cfg.ResolvedTraceConfig,
    rm_config: Config,
    spec: SolveSpec,
    grid: Grid,
    solver_controls: Mapping[str, object],
    timing: Mapping[str, float],
    diffraction_metadata: Mapping[str, object],
    shadow_boundary_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    shadow_boundary_metadata = (
        {}
        if shadow_boundary_payload is None
        else dict(shadow_boundary_payload.get("metadata", {}))
    )
    return {
        "scene_summary": _scene_summary(scene, rm_config),
        "grid": {
            "surface_mode": str(grid.surface_mode),
            "grid_shape": (int(grid.grid_shape[0]), int(grid.grid_shape[1])),
            "cell_size": (float(grid.cell_size[0]), float(grid.cell_size[1])),
            "quadrature_mode": str(spec.quadrature_mode),
            "samples_per_cell": int(spec.samples_per_cell),
        },
        "solver_controls": dict(solver_controls),
        "array_model": {
            "synthetic_array": bool(rm_config.synthetic_array),
            "assumption": "synthetic_array=True traces endpoint centers and assumes far-field plane-wave phase across the aperture; use explicit mode for near-field or large apertures.",
        },
        "runtime_backends": {
            "reflection_field_backend": str(rm_config.tuning.reflection_field_backend),
            "reflection_transition": {
                "mode": str(resolved.reflection_transition_mode),
                "resolved_backend": {
                    "hard": "hard",
                    "f_weight_reference": "reference_pair_replay",
                    "f_weight_native": "native_cuda_f_weight",
                }[str(resolved.reflection_transition_mode)],
                "boundary_radius_wavelengths": float(
                    resolved.reflection_f_weight_boundary_radius_wavelengths
                ),
                "max_edges_per_slot": int(resolved.reflection_f_weight_max_edges_per_slot),
            },
            "reflection_secondary_visibility": {
                "mode": str(resolved.reflection_secondary_visibility_mode),
                "resolved_backend": {
                    "hard": "hard",
                    "f_weight": "reference_segment_f_weight",
                }[str(resolved.reflection_secondary_visibility_mode)],
            },
            "diffraction_execution": resolved.diffraction_execution.to_dict(),
        },
        "diffraction": dict(diffraction_metadata),
        "shadow_boundary_correction": {
            "enabled": bool(spec.shadow_boundary_correction),
            "applied": shadow_boundary_payload is not None,
            "backend": (
                shadow_boundary_metadata.get("resolved_backend")
                or shadow_boundary_metadata.get("backend")
                or ("matched_isb_correction" if shadow_boundary_payload is not None else "none")
            ),
            "requested_backend": str(spec.shadow_boundary_backend),
            "statistics": shadow_boundary_metadata,
        },
        "performance_timing": {
            str(key): float(value) for key, value in dict(timing).items()
        },
    }


def _resolve_noise_power(scene: Scene, spec: SolveSpec) -> float:
    if spec.noise_power is not None:
        return float(spec.noise_power)
    for key in ("noise_power", "thermal_noise_power"):
        value = getattr(scene, key, None)
        if value is not None:
            return float(value)
    return 0.0


def _single_tx_sinr(rss: wt.Float, *, noise_power: float) -> wt.Float:
    n_rx = int(dr.width(rss))
    if noise_power > 0.0:
        return rss / float(noise_power)
    inf = dr.full(wt.Float, float("inf"), n_rx)
    return dr.select(rss > 0.0, inf, dr.zeros(wt.Float, n_rx))


def _empty_components(n_rx: int) -> dict[str, object]:
    return {
        "vector_coherent": {
            "los": vector_zero(n_rx),
            "reflection": vector_zero(n_rx),
            "diffraction": vector_zero(n_rx),
            "total": vector_zero(n_rx),
        },
        # Quadrature-averaged per-component power: sum_i w_i |E_comp,i|^2.
        # Same incoherent-in-cell semantics as path_gain, unlike the
        # vector_coherent payload which is the quadrature-mean field.
        "power": {
            "los": dr.zeros(wt.Float, int(n_rx)),
            "reflection": dr.zeros(wt.Float, int(n_rx)),
            "diffraction": dr.zeros(wt.Float, int(n_rx)),
        },
        "path_gain": dr.zeros(wt.Float, int(n_rx)),
    }


def _add_sample(
    components: dict[str, object],
    *,
    sample_weight: float,
    los_field_vector: ComplexVector,
    reflection_vector_coherent: ComplexVector,
    diffraction_vector_coherent: ComplexVector,
) -> dict[str, object]:
    w = float(sample_weight)
    vc = components["vector_coherent"]
    power = components["power"]
    vc["los"] = {
        axis: vc["los"][axis] + los_field_vector[axis] * w for axis in ("x", "y", "z")
    }
    vc["reflection"] = {
        axis: vc["reflection"][axis] + reflection_vector_coherent[axis] * w
        for axis in ("x", "y", "z")
    }
    vc["diffraction"] = {
        axis: vc["diffraction"][axis] + diffraction_vector_coherent[axis] * w
        for axis in ("x", "y", "z")
    }
    power["los"] = power["los"] + vector_power(los_field_vector) * w
    power["reflection"] = power["reflection"] + vector_power(reflection_vector_coherent) * w
    power["diffraction"] = power["diffraction"] + vector_power(diffraction_vector_coherent) * w
    total_v = {
        axis: los_field_vector[axis] + reflection_vector_coherent[axis] + diffraction_vector_coherent[axis]
        for axis in ("x", "y", "z")
    }
    components["path_gain"] = components["path_gain"] + vector_power(total_v) * w
    return components


def _finalize_totals(components: dict[str, object]) -> dict[str, object]:
    vc = components["vector_coherent"]
    vc["total"] = {
        axis: vc["los"][axis] + vc["reflection"][axis] + vc["diffraction"][axis]
        for axis in ("x", "y", "z")
    }
    return components


def _apply_shadow_boundary_correction(
    components: dict[str, object],
    shadow_boundary_payload: Mapping[str, object],
    *,
    samples_per_cell: int,
) -> None:
    # The overwrite below replaces the quadrature-averaged path_gain with the
    # power of the corrected mean field. That is only equivalent for a single
    # center sample; resolve_trace_config enforces quadrature_mode='center'
    # for shadow_boundary_correction, and this assert keeps the coupling local.
    if int(samples_per_cell) != 1:
        raise RuntimeError(
            "shadow boundary correction requires a single center quadrature sample."
        )
    correction_vector = shadow_boundary_payload["vector_coherent"]
    vc = components["vector_coherent"]
    vc["diffraction"] = {
        axis: vc["diffraction"][axis] + correction_vector[axis]
        for axis in ("x", "y", "z")
    }
    vc["total"] = {
        axis: vc["total"][axis] + correction_vector[axis] for axis in ("x", "y", "z")
    }
    components["power"]["diffraction"] = vector_power(vc["diffraction"])
    components["path_gain"] = vector_power(vc["total"])


def _component_maps(components: dict[str, object]) -> Mapping[str, wt.Float]:
    power = components["power"]
    return {
        "los": power["los"],
        "reflection": power["reflection"],
        "diffraction": power["diffraction"],
        "path_gain": components["path_gain"],
    }


def solve(
    *,
    scene: Scene,
    transmitter: str | Transmitter | list[str | Transmitter] | tuple[str | Transmitter, ...],
    receiver: str | ReceiverGrid,
    config: Config | None = None,
) -> RadioMapResult:
    """Compute a deterministic radio map using ``scene.frequency``."""
    return _solve(
        scene=scene,
        transmitter=transmitter,
        receiver=receiver,
        config=config,
        ray_mode=DEFAULT_RAY_MODE,
    )


def _solve(
    *,
    scene: Scene,
    transmitter: str | Transmitter | list[str | Transmitter] | tuple[str | Transmitter, ...],
    receiver: str | ReceiverGrid,
    config: Config | None = None,
    ray_mode: str = DEFAULT_RAY_MODE,
) -> RadioMapResult:
    resolved_ray_mode: RayMode = normalize_ray_mode(ray_mode)
    total_start = time.perf_counter()
    timing = _new_timing()
    assert_scene_materials_complete(scene)
    grid_start = time.perf_counter()
    rm_config = Config() if config is None else config
    frequency = scene.frequency
    if frequency is None:
        raise ValueError("deterministic.solve requires Scene.frequency.")
    if isinstance(transmitter, (list, tuple)):
        if not transmitter:
            raise ValueError("deterministic.solve transmitter list must not be empty.")
        results = tuple(
            _solve(
                scene=scene,
                transmitter=item,
                receiver=receiver,
                config=rm_config,
                ray_mode=resolved_ray_mode,
            )
            for item in transmitter
        )
        return stack_radiomap_results(results, noise_power=results[0].noise_power)
    resolved = cfg.resolve_trace_config(frequency=frequency, config=rm_config)

    tx_endpoint = scene.transmitter(transmitter)
    tx_pos = tx_endpoint.position
    resolved = replace(resolved, tx_polarization=tx_endpoint.polarization or (1.0, 0.0, 0.0))

    rx_endpoint = scene.receiver(receiver)
    if not isinstance(rx_endpoint, ReceiverGrid):
        raise TypeError("deterministic.solve receiver endpoint must be a ReceiverGrid.")
    grid = rx_endpoint
    resolved = replace(resolved, rx_polarization=rx_endpoint.polarization)

    solver_controls = cfg.resolve_solver_controls(
        rm_config,
        execution_intent="coherent",
        max_diffractions_override=int(rm_config.max_diffraction_order),
    )
    scene.diffraction_edge_count(edge_policy=rm_config.edge_policy)
    spec = SolveSpec.from_public(grid=grid, config=rm_config, ray_mode=resolved_ray_mode)

    runtime = TraceContext.from_config(tx_pos=tx_pos, config=resolved)
    grid = build_grid(spec, default_cell_size=resolved.cell_size)
    n_rx = int(grid.n_cells)
    grad_sensitive = (
        grid.surface_mode == "axis_aligned"
        and grad_sensitive_workload(tx=runtime.tx, scene=scene)
    )
    timing["grid_resolution_seconds"] += time.perf_counter() - grid_start
    components = _empty_components(n_rx)
    diffraction_sample_metadata: list[dict[str, object]] = []

    rayd_exact_auto_candidate = (
        str(resolved.diffraction_execution.accumulate_primal) == "auto"
        and grid.surface_mode == "axis_aligned"
        and int(resolved.max_diffractions) >= 1
        and resolved.shadow_support_cutoff_db is None
    )
    needs_native_grid = (
        grad_sensitive
        or str(resolved.diffraction_execution.accumulate_primal) == "rayd_exact_coherent"
        or rayd_exact_auto_candidate
        # Reflected diffraction suffix (D->R...) splats onto the grid directly.
        or (
            bool(resolved.enable_rd_diffraction)
            and grid.surface_mode == "axis_aligned"
            and int(resolved.reflection_n_rays) > 0
            and int(resolved.reflection_max_bounces) > 0
        )
    )

    # Path discovery and diffraction state preparation are rx-independent:
    # run them once against the cell centers and replay the prepared states
    # against every quadrature sample set below.
    center_runtime = runtime.with_rx(
        grid.cell_centers,
        polarization=resolved.rx_polarization,
    )
    center_ctx = TraceCtx(
        scene=scene,
        runtime=center_runtime,
        sample_grid=(
            NativeGrid.from_grid(grid, sample_index=0) if needs_native_grid else None
        ),
        config=resolved,
        spec=spec,
        solver_controls=solver_controls,
        grad_preserving=bool(grad_sensitive),
        n_rx=n_rx,
    )
    reflection_start = time.perf_counter()
    reflection_detail = reflection.discover(ctx=center_ctx)
    timing["reflection_trace_seconds"] += time.perf_counter() - reflection_start
    diffraction_raw_collections, diffraction_prep_metadata = diffraction.prepare(
        ctx=center_ctx,
        reflection_detail=reflection_detail,
        state_layout=PATH_EXPORT_REDUCED_STATE_LAYOUT,
    )
    timing["diffraction_state_preparation_seconds"] += float(
        diffraction_prep_metadata.get("state_preparation_seconds", 0.0)
    )

    for sample_set in grid.sample_sets:
        sample_grid = (
            NativeGrid.from_grid(grid, sample_index=sample_set.index)
            if needs_native_grid
            else None
        )
        sample_runtime = runtime.with_rx(
            sample_set.positions,
            polarization=resolved.rx_polarization,
        )
        ctx = TraceCtx(
            scene=scene,
            runtime=sample_runtime,
            sample_grid=sample_grid,
            config=resolved,
            spec=spec,
            solver_controls=solver_controls,
            grad_preserving=bool(grad_sensitive),
            n_rx=n_rx,
        )

        los_start = time.perf_counter()
        los_vector = los.trace_vector(scene=scene, runtime=sample_runtime)
        timing["los_trace_seconds"] += time.perf_counter() - los_start
        reflection_start = time.perf_counter()
        reflection_payload = reflection.trace(ctx=ctx, reflection_detail=reflection_detail)
        reflection_field = reflection_payload["field"]
        reflection_detail = reflection_payload["detail"]
        timing["reflection_trace_seconds"] += time.perf_counter() - reflection_start
        diffraction_field = diffraction.trace(
            ctx=ctx,
            raw_collections=diffraction_raw_collections,
        )
        diffraction_metadata = dict(diffraction_field.get("metadata", {}))
        diffraction_sample_metadata.append(diffraction_metadata)
        timing["diffraction_accumulation_seconds"] += float(
            diffraction_metadata.get("accumulation_seconds", 0.0)
        )

        _add_sample(
            components,
            sample_weight=sample_set.weight,
            los_field_vector=los_vector,
            reflection_vector_coherent=reflection_field["vector"],
            diffraction_vector_coherent=diffraction_field["vector"],
        )

    _finalize_totals(components)
    shadow_boundary_runtime = runtime.with_rx(
        grid.cell_centers,
        polarization=resolved.rx_polarization,
    )
    shadow_boundary_start = time.perf_counter()
    shadow_boundary_payload = postprocessing.trace_shadow_boundary_correction(
        spec=spec, grid=grid, scene=scene, runtime=shadow_boundary_runtime,
        components=components,
    )
    if shadow_boundary_payload is not None:
        _apply_shadow_boundary_correction(
            components,
            shadow_boundary_payload,
            samples_per_cell=int(spec.samples_per_cell),
        )
    timing["shadow_boundary_correction_seconds"] += (
        time.perf_counter() - shadow_boundary_start
    )
    path_gain = components["path_gain"]
    tx_power = float(tx_endpoint.power) if tx_endpoint is not None else float(spec.tx_power)
    rss = path_gain * tx_power
    noise_power = _resolve_noise_power(scene, spec)
    sinr = _single_tx_sinr(rss, noise_power=noise_power)

    shape_start = time.perf_counter()
    grid_shape = (int(grid.grid_shape[0]), int(grid.grid_shape[1]))
    cell_tensor_shape = (grid_shape[1], grid_shape[0])
    tx_tensor_shape = (1, *cell_tensor_shape)
    surface = grid.surface_descriptor()
    tx_position = runtime.tx.position
    coords = coordinates_from_grid(
        grid,
        sample_positions=tuple(sample.positions for sample in grid.sample_sets),
    )
    result_components = to_mapping_proxy({
        str(name): to_float_tensor(value, shape=tx_tensor_shape)
        for name, value in _component_maps(components).items()
    })
    path_gain_tensor = to_float_tensor(path_gain, shape=tx_tensor_shape)
    rss_tensor = to_float_tensor(rss, shape=tx_tensor_shape)
    sinr_tensor = to_float_tensor(sinr, shape=tx_tensor_shape)
    best_tx_index = to_int_tensor(
        dr.zeros(wt.Int32, cell_tensor_shape[0] * cell_tensor_shape[1]),
        shape=cell_tensor_shape,
    )
    tx_pos_tuple = (scalar(tx_position.x), scalar(tx_position.y), scalar(tx_position.z))
    timing["result_shaping_seconds"] += time.perf_counter() - shape_start
    timing["total_solve_seconds"] = time.perf_counter() - total_start
    metadata = _build_metadata(
        scene=scene,
        resolved=resolved,
        rm_config=rm_config,
        spec=spec,
        grid=grid,
        solver_controls=solver_controls,
        timing=timing,
        diffraction_metadata=_build_diffraction_metadata(
            diffraction_prep_metadata,
            diffraction_sample_metadata,
        ),
        shadow_boundary_payload=shadow_boundary_payload,
    )
    metadata["transmitter_count"] = 1
    return RadioMapResult(
        name=str(spec.name),
        kind=str(spec.kind),
        metric=str(spec.metric),
        solver="deterministic",
        grid_shape=grid_shape,
        cell_size=(float(grid.cell_size[0]), float(grid.cell_size[1])),
        surface=to_mapping_proxy(surface),
        coords=coords,
        path_gain=path_gain_tensor,
        rss=rss_tensor,
        sinr=sinr_tensor,
        best_tx_index=best_tx_index,
        tx_association_map=best_tx_index,
        tx_pos=tx_pos_tuple,
        tx_power=tx_power,
        noise_power=float(noise_power),
        metadata=to_mapping_proxy(metadata),
        field=RadioMapFieldPayload(
            vector_coherent=to_mapping_proxy(components["vector_coherent"]),
        ),
        power=None,
        components=result_components,
    )


__all__ = ["solve"]
