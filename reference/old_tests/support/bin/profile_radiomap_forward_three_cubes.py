"""Profile a single three-cube radiomap forward run and summarize Dr.Jit kernels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from ._benchmark_runtime import benchmark_environment_report
    from ._multipath_scaling_common import flush_gpu_caches, sync_gpu
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import benchmark_environment_report
    from _multipath_scaling_common import flush_gpu_caches, sync_gpu

import drjit as dr
import numpy as np

import witwin as wt
from tests.main.plot_multipath_components import CUBE1_BASE_CENTER
from tests.main.plot_radiomap_sionna_three_cubes import (
    DEFAULT_BOUNDS,
    DEFAULT_GRID_SIZE,
    DEFAULT_N_RAYS,
    DEFAULT_PLANE_Z,
    DEFAULT_TX_POS,
    DEFAULT_WITWIN_COMBINE_MODE,
    DEFAULT_WITWIN_EDGE_SELECTION_MODE,
    DEFAULT_WITWIN_PROFILE,
    DEFAULT_WITWIN_RECEIVER_MODEL,
    DEFAULT_WITWIN_SHADOW_BOUNDARY_MODE,
    WITWIN_PROFILES,
    _build_comparison_scene,
    _resolve_witwin_profile,
)
from witwin.channel import RadioMapMonitor, Tracer

DEFAULT_OUTPUT_JSON = (
    Path(__file__).resolve().parent.parent / "output" / "radiomap_forward_three_cubes_profile.json"
)


def _enum_name(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if "." in text:
        return text.rsplit(".", maxsplit=1)[-1]
    return text


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _float_stat_grid(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _value_stats(values: Any) -> dict[str, Any]:
    grid = _float_stat_grid(values)
    flat = grid.reshape(-1)
    if flat.size == 0:
        return {
            "shape": [int(value) for value in grid.shape],
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "finite": True,
        }
    return {
        "shape": [int(value) for value in grid.shape],
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "p50": float(np.percentile(flat, 50.0)),
        "p95": float(np.percentile(flat, 95.0)),
        "p99": float(np.percentile(flat, 99.0)),
        "finite": bool(np.isfinite(flat).all()),
    }


def _component_metric_summary(result, *, combine_mode: str) -> dict[str, Any]:
    component_metrics = (
        getattr(result, "coherent_power", None)
        if str(combine_mode) == "coherent"
        else getattr(result, "incoherent", None)
    )
    if component_metrics is None:
        return {"path_gain": _value_stats(result.path_gain)}
    folded_diffraction = component_metrics.get("diffraction")
    raw_diffraction = (
        folded_diffraction
        if folded_diffraction is None
        else component_metrics.get("raw_diffraction", folded_diffraction)
    )
    summary = {
        "path_gain": _value_stats(result.path_gain),
    }
    for name in ("los", "reflection"):
        if name in component_metrics:
            summary[name] = _value_stats(component_metrics[name])
    if raw_diffraction is not None:
        summary["raw_diffraction"] = _value_stats(raw_diffraction)
    if folded_diffraction is not None:
        summary["folded_diffraction"] = _value_stats(
            component_metrics.get("folded_diffraction", folded_diffraction)
        )
    return summary


def _kernel_record(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": _enum_name(entry.get("type", "unknown")),
        "size": int(entry.get("size", 0) or 0),
        "execution_time_ms": float(entry.get("execution_time", 0.0) or 0.0),
        "operation_count": int(entry.get("operation_count", 0) or 0),
        "hash": str(entry.get("hash", "")),
    }


def _summarize_kernel_history(
    history: list[dict[str, Any]],
    *,
    top_k: int,
    small_kernel_max_size: int,
    small_kernel_min_count: int,
) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    repeated_small = Counter()
    records = [_kernel_record(entry) for entry in history]
    for record in records:
        type_name = str(record["type"])
        bucket = by_type.setdefault(
            type_name,
            {
                "count": 0,
                "size_sum": 0,
                "size_max": 0,
                "execution_time_ms_sum": 0.0,
                "small_count_le_4": 0,
                "small_count_le_128": 0,
                "small_count_le_1024": 0,
                "large_count_gt_1m": 0,
            },
        )
        size = int(record["size"])
        bucket["count"] += 1
        bucket["size_sum"] += size
        bucket["size_max"] = max(int(bucket["size_max"]), size)
        bucket["execution_time_ms_sum"] += float(record["execution_time_ms"])
        if size <= 4:
            bucket["small_count_le_4"] += 1
        if size <= 128:
            bucket["small_count_le_128"] += 1
        if size <= 1024:
            bucket["small_count_le_1024"] += 1
        if size > (1 << 20):
            bucket["large_count_gt_1m"] += 1
        if size <= int(small_kernel_max_size):
            small_key = (
                type_name,
                size,
                int(record["operation_count"]),
                str(record["hash"]),
            )
            repeated_small[small_key] += 1

    top_by_execution = sorted(
        records,
        key=lambda item: (float(item["execution_time_ms"]), int(item["size"])),
        reverse=True,
    )[: int(top_k)]
    top_by_size = sorted(
        records,
        key=lambda item: (int(item["size"]), float(item["execution_time_ms"])),
        reverse=True,
    )[: int(top_k)]
    frequent_small_kernels = []
    for (type_name, size, operation_count, hash_value), count in repeated_small.most_common():
        if int(count) < int(small_kernel_min_count):
            continue
        frequent_small_kernels.append(
            {
                "type": str(type_name),
                "size": int(size),
                "operation_count": int(operation_count),
                "hash": str(hash_value),
                "count": int(count),
            }
        )
        if len(frequent_small_kernels) >= int(top_k):
            break

    return {
        "total_count": int(len(records)),
        "by_type": by_type,
        "top_by_execution_ms": top_by_execution,
        "top_by_size": top_by_size,
        "frequent_small_kernels": frequent_small_kernels,
        "small_kernel_max_size": int(small_kernel_max_size),
        "small_kernel_min_count": int(small_kernel_min_count),
    }


def _build_scene():
    cube1_x = float(CUBE1_BASE_CENTER[0])
    return _build_comparison_scene(
        cube1_x,
        edge_selection_mode=DEFAULT_WITWIN_EDGE_SELECTION_MODE,
    )


def _run_forward(
    *,
    grid_size: int,
    bounds,
    plane_z: float,
    tx_pos,
    n_rays: int,
    max_diffractions: int,
    combine_mode: str,
    receiver_model: str,
    accumulation_backend: str,
    shadow_boundary_mode: str,
    shadow_support_cutoff_db: float | None,
) -> tuple[Any, dict[str, float]]:
    total_t0 = time.perf_counter()
    scene_t0 = time.perf_counter()
    scene = _build_scene()
    sync_gpu()
    scene_build_seconds = time.perf_counter() - scene_t0

    setup_t0 = time.perf_counter()
    monitor = RadioMapMonitor(
        "profile_rm",
        axis="z",
        position=float(plane_z),
        bounds=bounds,
        grid_shape=(int(grid_size), int(grid_size)),
        metric="path_gain",
        combine_mode=str(combine_mode),
        quadrature_mode="center",
        receiver_model=str(receiver_model),
        accumulation_backend=str(accumulation_backend),
        ray_mode="3d",
        max_diffractions=int(max_diffractions),
        shadow_boundary_mode=str(shadow_boundary_mode),
        shadow_support_cutoff_db=shadow_support_cutoff_db,
    )
    tracer = Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=int(n_rays),
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=int(max_diffractions),
    )
    sync_gpu()
    monitor_tracer_setup_seconds = time.perf_counter() - setup_t0

    trace_t0 = time.perf_counter()
    trace_output = tracer.trace(wt.Point3f(*tx_pos), monitor=monitor, verbose=False)
    result = trace_output.monitor(monitor.name) if hasattr(trace_output, "monitor") else trace_output
    dr.eval(result.path_gain)
    sync_gpu()
    forward_seconds = time.perf_counter() - trace_t0
    end_to_end_seconds = time.perf_counter() - total_t0
    return result, {
        "scene_build_seconds": float(scene_build_seconds),
        "monitor_tracer_setup_seconds": float(monitor_tracer_setup_seconds),
        "forward_seconds": float(forward_seconds),
        "end_to_end_seconds": float(end_to_end_seconds),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--n-rays", type=int, default=DEFAULT_N_RAYS)
    parser.add_argument("--max-diffractions", type=int, default=2)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--tx-x", type=float, default=float(DEFAULT_TX_POS[0]))
    parser.add_argument("--tx-y", type=float, default=float(DEFAULT_TX_POS[1]))
    parser.add_argument("--tx-z", type=float, default=float(DEFAULT_TX_POS[2]))
    parser.add_argument("--xmin", type=float, default=float(DEFAULT_BOUNDS[0][0]))
    parser.add_argument("--xmax", type=float, default=float(DEFAULT_BOUNDS[0][1]))
    parser.add_argument("--ymin", type=float, default=float(DEFAULT_BOUNDS[1][0]))
    parser.add_argument("--ymax", type=float, default=float(DEFAULT_BOUNDS[1][1]))
    parser.add_argument(
        "--witwin-profile",
        choices=tuple(WITWIN_PROFILES.keys()),
        default=DEFAULT_WITWIN_PROFILE,
    )
    parser.add_argument("--witwin-combine-mode", type=str, default=DEFAULT_WITWIN_COMBINE_MODE)
    parser.add_argument("--witwin-receiver-model", type=str, default=DEFAULT_WITWIN_RECEIVER_MODEL)
    parser.add_argument(
        "--accumulation-backend",
        choices=("baseline", "cell_accumulation"),
        default="baseline",
    )
    parser.add_argument(
        "--witwin-shadow-boundary-mode",
        type=str,
        default=DEFAULT_WITWIN_SHADOW_BOUNDARY_MODE,
    )
    parser.add_argument(
        "--witwin-shadow-support-cutoff-db",
        type=float,
        default=None,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--small-kernel-max-size", type=int, default=1024)
    parser.add_argument("--small-kernel-min-count", type=int, default=8)
    parser.add_argument("--kernel-history", dest="kernel_history", action="store_true")
    parser.add_argument("--no-kernel-history", dest="kernel_history", action="store_false")
    parser.set_defaults(kernel_history=True)
    parser.add_argument("--flush-gpu-caches", dest="flush_gpu_caches", action="store_true")
    parser.add_argument("--no-flush-gpu-caches", dest="flush_gpu_caches", action="store_false")
    parser.set_defaults(flush_gpu_caches=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--write-default-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bounds = (
        (float(args.xmin), float(args.xmax)),
        (float(args.ymin), float(args.ymax)),
    )
    tx_pos = (float(args.tx_x), float(args.tx_y), float(args.tx_z))
    (
        resolved_profile,
        resolved_profile_label,
        resolved_combine_mode,
        resolved_receiver_model,
        resolved_shadow_boundary_mode,
    ) = _resolve_witwin_profile(
        profile=str(args.witwin_profile),
        combine_mode=str(args.witwin_combine_mode),
        receiver_model=str(args.witwin_receiver_model),
        shadow_boundary_mode=str(args.witwin_shadow_boundary_mode),
    )

    if bool(args.flush_gpu_caches):
        flush_gpu_caches()

    history: list[dict[str, Any]] = []
    if bool(args.kernel_history):
        with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
            dr.kernel_history_clear()
            result, timing = _run_forward(
                grid_size=int(args.grid_size),
                bounds=bounds,
                plane_z=float(args.plane_z),
                tx_pos=tx_pos,
                n_rays=int(args.n_rays),
                max_diffractions=int(args.max_diffractions),
                combine_mode=resolved_combine_mode,
                receiver_model=resolved_receiver_model,
                accumulation_backend=str(args.accumulation_backend),
                shadow_boundary_mode=resolved_shadow_boundary_mode,
                shadow_support_cutoff_db=args.witwin_shadow_support_cutoff_db,
            )
            history = list(dr.kernel_history())
    else:
        result, timing = _run_forward(
            grid_size=int(args.grid_size),
            bounds=bounds,
            plane_z=float(args.plane_z),
            tx_pos=tx_pos,
            n_rays=int(args.n_rays),
            max_diffractions=int(args.max_diffractions),
            combine_mode=resolved_combine_mode,
            receiver_model=resolved_receiver_model,
            accumulation_backend=str(args.accumulation_backend),
            shadow_boundary_mode=resolved_shadow_boundary_mode,
            shadow_support_cutoff_db=args.witwin_shadow_support_cutoff_db,
        )

    metadata = dict(getattr(result, "metadata", {}) or {})
    diffraction_runtime = dict(metadata.get("runtime_backends", {}).get("diffraction", {}) or {})
    full_pair_count = int(diffraction_runtime.get("full_pair_count", 0) or 0)
    peak_pair_count_estimate = int(diffraction_runtime.get("peak_pair_count_estimate", 0) or 0)
    summary = {
        "environment": benchmark_environment_report(),
        "scenario": {
            "grid_size": int(args.grid_size),
            "bounds": _jsonable(bounds),
            "plane_z": float(args.plane_z),
            "tx_pos": _jsonable(tx_pos),
            "n_rays": int(args.n_rays),
            "max_diffractions": int(args.max_diffractions),
            "witwin_profile": str(resolved_profile),
            "witwin_profile_label": str(resolved_profile_label),
            "witwin_combine_mode": str(resolved_combine_mode),
            "witwin_receiver_model": str(resolved_receiver_model),
            "accumulation_backend": str(args.accumulation_backend),
            "witwin_shadow_boundary_mode": str(resolved_shadow_boundary_mode),
            "witwin_shadow_support_cutoff_db": (
                None
                if args.witwin_shadow_support_cutoff_db is None
                else float(args.witwin_shadow_support_cutoff_db)
            ),
        },
        "timing": {
            **{str(key): float(value) for key, value in timing.items()},
            "kernel_history_enabled": bool(args.kernel_history),
        },
        "grid_shape": _jsonable(getattr(result, "grid_shape", ())),
        "tensor_shape": _jsonable(getattr(result, "tensor_shape", ())),
        "metric": str(getattr(result, "metric", "")),
        "path_gain_stats": _value_stats(result.path_gain),
        "component_metric_summary": _component_metric_summary(
            result,
            combine_mode=str(resolved_combine_mode),
        ),
        "path_counts": _jsonable(metadata.get("path_counts", {})),
        "metric_contract": _jsonable(metadata.get("metric_contract", {})),
        "performance_timing": _jsonable(metadata.get("performance_timing", {})),
        "runtime_backends": _jsonable(metadata.get("runtime_backends", {})),
        "diffraction_runtime_metadata": _jsonable(diffraction_runtime),
        "bounded_streaming": {
            "pair_chunk_budget": int(diffraction_runtime.get("pair_chunk_budget", 0) or 0),
            "peak_pair_count_estimate": peak_pair_count_estimate,
            "full_pair_count": full_pair_count,
            "full_pair_materialization_detected": bool(
                full_pair_count > 0 and peak_pair_count_estimate >= full_pair_count
            ),
        },
        "kernel_history": _summarize_kernel_history(
            history,
            top_k=int(args.top_k),
            small_kernel_max_size=int(args.small_kernel_max_size),
            small_kernel_min_count=int(args.small_kernel_min_count),
        ),
    }
    output_json = args.output_json
    if bool(args.write_default_json):
        output_json = DEFAULT_OUTPUT_JSON
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["output_json"] = str(output_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
