"""Benchmark Phase 0 cell-state memory pressure across field and path workloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping
try:
    from ._benchmark_runtime import benchmark_environment_report
    from ._multipath_scaling_common import (
        flush_gpu_caches,
        measure_phase,
        memory_snapshot,
    )
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import benchmark_environment_report
    from _multipath_scaling_common import flush_gpu_caches, measure_phase, memory_snapshot

import drjit as dr
import witwin as wt
from tests.main.plot_multipath_components import (
    CUBE1_BASE_CENTER,
    MULTIPATH_RELATIVE_PERMITTIVITY,
    TRACE_BOUNDS,
    TX_POS,
    build_scene_for_cube1_x,
)
from witwin.channel import FieldMonitor, PathMonitor, Tracer
from witwin.channel.monitors.field.trace import trace_field_monitor_total_only
from witwin.channel.monitors.path.collectors import (
    collect_diffraction_state_paths,
    collect_los_paths,
    collect_reflection_paths,
)
from witwin.channel.monitors.path.trace import (
    _gather_positions,
    _receiver_groups,
    _remap_raw_rx_index,
)
from witwin.channel.monitors.orchestration import resolve_solver_controls
from witwin.channel.monitors.profiler import capture_cuda_memory_report
from witwin.channel.trace.diffraction.api import _finalize_solver_metadata
from witwin.channel.trace.diffraction.builders import (
    _build_solver_metadata,
    _prepare_diffraction_state_arrays,
)
from witwin.channel.utils.polarization import effective_rx_polarization
def _jsonable_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable_copy(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_copy(item) for item in value]
    return value


def _make_tracer(
    scene,
    *,
    reflection_n_rays: int,
    reflection_max_bounces: int,
    max_diffractions: int,
    solver_mode: str,
    memory_profile: str,
) -> Tracer:
    return Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=reflection_n_rays,
        reflection_max_bounces=reflection_max_bounces,
        reflection_coef=0.8,
        reflection_relative_permittivity=MULTIPATH_RELATIVE_PERMITTIVITY,
        enable_rd_diffraction=True,
        max_diffractions=max_diffractions,
        solver_mode=solver_mode,
        memory_profile=memory_profile,
    )


def _make_field_monitor(grid_size: int) -> FieldMonitor:
    return FieldMonitor(
        "cell_state_field",
        axis="z",
        position=TX_POS[2],
        bounds=TRACE_BOUNDS,
        grid_size=grid_size,
    )


def _make_path_monitor(*, tracer: Tracer, grid_size: int) -> PathMonitor:
    field_monitor = _make_field_monitor(grid_size)
    field = field_monitor.to_field(
        tracer.wavelength,
        default_resolution=tracer.resolution_wavelength,
    )
    return PathMonitor(
        "cell_state_paths",
        positions=field.receivers,
        max_diffractions=tracer.config.trace.max_diffractions,
        return_geometry=False,
    )


def _setup_case(case: Mapping[str, Any]) -> dict[str, Any]:
    scene = build_scene_for_cube1_x(CUBE1_BASE_CENTER[0])
    tracer = _make_tracer(
        scene,
        reflection_n_rays=int(case["reflection_n_rays"]),
        reflection_max_bounces=int(case["reflection_max_bounces"]),
        max_diffractions=int(case["max_diffractions"]),
        solver_mode=str(case["solver_mode"]),
        memory_profile=str(case["memory_profile"]),
    )
    tx_pos = wt.Point3f(*TX_POS)
    if case["kind"] == "field":
        monitor = _make_field_monitor(int(case["grid_size"]))
        solver_controls = resolve_solver_controls(
            tracer.config.trace,
            execution_intent="field_scalar_only",
        )
    else:
        monitor = _make_path_monitor(tracer=tracer, grid_size=int(case["grid_size"]))
        solver_controls = resolve_solver_controls(
            tracer.config.trace,
            execution_intent="path_export",
        )
    return {
        "scene": scene,
        "tracer": tracer,
        "tx_pos": tx_pos,
        "monitor": monitor,
        "solver_controls": solver_controls,
        "config": dict(case),
    }


def _summarize_field_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}))
    return {
        "payload_kind": str(payload.get("payload_kind", "field_total_only")),
        "execution_intent": _jsonable_copy(metadata.get("execution_intent", {})),
        "solver_mode": _jsonable_copy(metadata.get("solver_mode", {})),
        "state_memory_profile": _jsonable_copy(metadata.get("state_memory_profile", {})),
        "performance_guardrails": _jsonable_copy(metadata.get("performance_guardrails", {})),
        "performance_timing": _jsonable_copy(metadata.get("performance_timing", {})),
        "performance_memory": _jsonable_copy(metadata.get("performance_memory", {})),
    }


def _summarize_path_payload(path_result) -> dict[str, Any]:
    metadata = dict(getattr(path_result, "metadata", {}) or {})
    diffraction_groups = []
    for group in metadata.get("diffraction_groups", ()):
        solver_metadata = dict(group.get("solver_metadata", {}))
        diffraction_groups.append(
            {
                "receiver_height_z": float(group.get("receiver_height_z", 0.0)),
                "receiver_count": int(group.get("receiver_count", 0)),
                "n_edge_states": int(group.get("n_edge_states", 0)),
                "n_paths": int(group.get("n_paths", 0)),
                "state_memory_profile": _jsonable_copy(
                    solver_metadata.get("state_memory_profile", {})
                ),
                "path_collection": _jsonable_copy(group.get("path_collection", {})),
            }
        )
    return {
        "receiver_count": int(path_result.num_rx),
        "max_num_paths": int(path_result.max_num_paths),
        "path_counts": _jsonable_copy(metadata.get("path_counts", {})),
        "execution_intent": _jsonable_copy(metadata.get("execution_intent", {})),
        "solver_mode": _jsonable_copy(metadata.get("solver_mode", {})),
        "performance_memory": _jsonable_copy(metadata.get("performance_memory", {})),
        "timing": _jsonable_copy(metadata.get("timing", {})),
        "diffraction_groups": diffraction_groups,
    }


def _run_field_case(case_ctx: Mapping[str, Any]) -> dict[str, Any]:
    payload, _ = trace_field_monitor_total_only(
        case_ctx["tx_pos"],
        case_ctx["monitor"],
        case_ctx["scene"],
        case_ctx["tracer"]._resolved_trace_config,
        case_ctx["solver_controls"],
        verbose=False,
        return_timing=True,
        return_diffraction_audit=False,
    )
    return _summarize_field_payload(payload)


def _run_path_case(case_ctx: Mapping[str, Any]) -> dict[str, Any]:
    config = case_ctx["tracer"]._resolved_trace_config
    solver_controls = case_ctx["solver_controls"]
    effective = solver_controls["effective"]
    monitor = case_ctx["monitor"]
    scene = case_ctx["scene"]
    tx_pos = case_ctx["tx_pos"]
    rx_positions = monitor.positions

    los_raw = collect_los_paths(
        scene=scene,
        rx_positions=rx_positions,
        tx_pos=tx_pos,
        wavelength=config.wavelength,
        k=config.k,
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
    )
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
        return_geometry=False,
        reflection_detail=None,
    )

    diffraction_groups = []
    path_counts = {
        "los": int(los_raw["metadata"].get("n_paths", 0)),
        "reflection": int(reflection_raw["metadata"].get("n_paths", 0)),
        "diffraction": 0,
    }
    reflection_n_rays = effective["reflection_n_rays"] if config.enable_rd_diffraction else 0
    reflection_max_bounces = effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0
    mixed_reflection_detail = reflection_detail if config.enable_rd_diffraction else None

    if effective["max_diffractions"] > 0:
        for receiver_z, group_indices in _receiver_groups(rx_positions):
            group_positions = _gather_positions(rx_positions, group_indices)
            edge_cache, edge_data, state_arrays, path_budget_report = _prepare_diffraction_state_arrays(
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
                use_scene_materials=config.use_scene_materials_for_diffraction,
                tx_polarization=config.tx_polarization,
                solver_mode=solver_controls["selected"],
                memory_profile=effective["memory_profile"],
            )
            path_collection_stats = {}
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
                stats=path_collection_stats,
            )
            _remap_raw_rx_index(raw, group_indices)
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
                receiver_axis="z",
            )
            n_paths = int(raw["metadata"].get("n_paths", 0))
            path_counts["diffraction"] += n_paths
            diffraction_groups.append(
                {
                    "receiver_height_z": float(receiver_z),
                    "receiver_count": int(group_indices.shape[0]),
                    "n_edge_states": int(state_arrays["n_states"]),
                    "n_paths": n_paths,
                    "state_memory_profile": _jsonable_copy(
                        solver_metadata.get("state_memory_profile", {})
                    ),
                    "path_collection": _jsonable_copy(path_collection_stats),
                }
            )

    return {
        "receiver_count": int(monitor.num_rx),
        "max_num_paths": None,
        "path_counts": path_counts,
        "execution_intent": _jsonable_copy(solver_controls["execution_intent"]),
        "solver_mode": _jsonable_copy(solver_controls),
        "performance_memory": {
            "torch_cuda": capture_cuda_memory_report(),
        },
        "timing": {},
        "diffraction_groups": diffraction_groups,
    }


def _run_case(case: Mapping[str, Any]) -> dict[str, Any]:
    flush_gpu_caches()
    case_ctx, setup_phase = measure_phase("setup", lambda: _setup_case(case))
    if case["kind"] == "field":
        summary, trace_phase = measure_phase("trace", lambda: _run_field_case(case_ctx))
    else:
        summary, trace_phase = measure_phase("trace", lambda: _run_path_case(case_ctx))
    return {
        "name": str(case["name"]),
        "kind": str(case["kind"]),
        "config": dict(case),
        "setup_seconds": float(setup_phase["seconds"]),
        "trace_seconds": float(trace_phase["seconds"]),
        "phase_metrics": {
            "setup": setup_phase,
            "trace": trace_phase,
        },
        "summary": summary,
        "memory_final": memory_snapshot(),
    }


DEFAULT_CASES = (
    {
        "name": "dense_field",
        "kind": "field",
        "grid_size": 512,
        "reflection_n_rays": 10000,
        "reflection_max_bounces": 3,
        "max_diffractions": 2,
        "solver_mode": "accuracy",
        "memory_profile": "default",
    },
    {
        "name": "high_diffractions",
        "kind": "field",
        "grid_size": 256,
        "reflection_n_rays": 10000,
        "reflection_max_bounces": 3,
        "max_diffractions": 4,
        "solver_mode": "accuracy",
        "memory_profile": "default",
    },
    {
        "name": "high_reflection_rays",
        "kind": "field",
        "grid_size": 256,
        "reflection_n_rays": 40000,
        "reflection_max_bounces": 3,
        "max_diffractions": 2,
        "solver_mode": "accuracy",
        "memory_profile": "default",
    },
    {
        "name": "path_export",
        "kind": "path",
        "grid_size": 128,
        "reflection_n_rays": 10000,
        "reflection_max_bounces": 3,
        "max_diffractions": 2,
        "solver_mode": "accuracy",
        "memory_profile": "default",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[case["name"] for case in DEFAULT_CASES],
        help="Subset of benchmark case names to run.",
    )
    parser.add_argument(
        "--memory-profile",
        choices=("default", "memory_safe"),
        default="default",
        help="Override the configured memory profile for every case.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full benchmark payload as JSON.",
    )
    args = parser.parse_args()

    selected_names = set(args.cases)
    cases = []
    for case in DEFAULT_CASES:
        if case["name"] not in selected_names:
            continue
        overridden = dict(case)
        overridden["memory_profile"] = str(args.memory_profile)
        cases.append(overridden)
    if not cases:
        raise ValueError("No benchmark cases were selected.")

    payload = {
        "benchmark": "cell_state_memory_phase0",
        "runtime_environment": benchmark_environment_report(),
        "cases": [_run_case(case) for case in cases],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        f"Runtime: module={payload['runtime_environment'].get('channel_module_file', 'n/a')} "
        f"native={payload['runtime_environment'].get('native_extension_available', 'n/a')} "
        f"cuda_runtime_version={payload['runtime_environment'].get('cuda_runtime_version', 'n/a')}"
    )
    for case in payload["cases"]:
        trace_peak = case["phase_metrics"]["trace"]["memory_after"]["drjit_allocator"].get(
            "device_peak",
            "n/a",
        )
        print(
            f"[{case['name']}] kind={case['kind']} setup={case['setup_seconds']:.3f}s "
            f"trace={case['trace_seconds']:.3f}s trace_peak={trace_peak}"
        )


if __name__ == "__main__":
    main()
