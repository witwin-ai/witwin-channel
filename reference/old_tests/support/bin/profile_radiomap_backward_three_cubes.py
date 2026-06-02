"""Profile forward and reverse-mode AD kernel launches for the three-cube radiomap benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from ._benchmark_runtime import benchmark_environment_report
    from ._multipath_scaling_common import flush_gpu_caches, sync_gpu
    from .profile_radiomap_forward_three_cubes import _jsonable, _summarize_kernel_history
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import benchmark_environment_report
    from _multipath_scaling_common import flush_gpu_caches, sync_gpu
    from profile_radiomap_forward_three_cubes import _jsonable, _summarize_kernel_history

import drjit as dr
import witwin as wt

from tests.main.plot_multipath_components import CUBE1_BASE_CENTER, build_scene_for_cube1_x
from tests.main.plot_radiomap_gradients_three_cubes import (
    DEFAULT_ACCUMULATION_BACKEND,
    DEFAULT_COMBINE_MODE,
    DEFAULT_MAX_DIFFRACTIONS,
    DEFAULT_RECEIVER_MODEL,
    DEFAULT_SHADOW_BOUNDARY_MODE,
    _GRAD_FLAGS,
    _loss_weights,
    _make_monitor,
    _make_tracer,
    _scalar_from_drjit,
    parameter_config,
)
from tests.main.plot_radiomap_sionna_three_cubes import (
    DEFAULT_BOUNDS,
    DEFAULT_GRID_SIZE,
    DEFAULT_N_RAYS,
    DEFAULT_PLANE_Z,
    DEFAULT_TX_POS,
)


DEFAULT_OUTPUT_JSON = (
    Path(__file__).resolve().parent.parent / "output" / "radiomap_backward_three_cubes_profile.json"
)


def _enum_name(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if "." in text:
        return text.rsplit(".", maxsplit=1)[-1]
    return text


def _kernel_record_extended(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": _enum_name(entry.get("type", "unknown")),
        "backend": _enum_name(entry.get("backend", "unknown")),
        "size": int(entry.get("size", 0) or 0),
        "execution_time_ms": float(entry.get("execution_time", 0.0) or 0.0),
        "operation_count": int(entry.get("operation_count", 0) or 0),
        "hash": str(entry.get("hash", "")),
        "input_count": int(entry.get("input_count", 0) or 0),
        "output_count": int(entry.get("output_count", 0) or 0),
    }


def _custom_launch_summary(history: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    records = [_kernel_record_extended(entry) for entry in history]
    custom_like = [
        record
        for record in records
        if str(record["type"]) not in {"JIT", "Reduce"}
    ]
    return {
        "count": int(len(custom_like)),
        "top": sorted(
            custom_like,
            key=lambda item: (
                float(item["execution_time_ms"]),
                int(item["size"]),
                int(item["input_count"]),
            ),
            reverse=True,
        )[: int(limit)],
    }


def _resolved_accumulation_backend(requested: str) -> str:
    requested_backend = str(requested)
    if requested_backend == "cell_accumulation":
        return "baseline"
    return requested_backend


def _build_variable(parameter: str, *, tx_pos):
    config = parameter_config(parameter, tx_pos=tx_pos)
    if parameter == "cube1_x":
        cube1_x = wt.Float(config["cube1_x"])
        dr.enable_grad(cube1_x)
        return cube1_x, cube1_x, wt.Point3f(*config["tx_pos"]), config

    tx_x = wt.Float(config["tx_pos"][0])
    dr.enable_grad(tx_x)
    return tx_x, config["cube1_x"], wt.Point3f(tx_x, config["tx_pos"][1], config["tx_pos"][2]), config


def _profile_parameter(
    *,
    parameter: str,
    tx_pos,
    grid_size: int,
    n_rays: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
    top_k: int,
    small_kernel_max_size: int,
    small_kernel_min_count: int,
) -> dict[str, Any]:
    differentiable_var, cube1_x, tx_point, config = _build_variable(parameter, tx_pos=tx_pos)
    requested_backend = str(accumulation_backend)
    resolved_backend_request = _resolved_accumulation_backend(requested_backend)

    scene_t0 = time.perf_counter()
    scene = build_scene_for_cube1_x(cube1_x)
    sync_gpu()
    scene_build_seconds = time.perf_counter() - scene_t0

    setup_t0 = time.perf_counter()
    tracer = _make_tracer(scene, n_rays=n_rays, max_diffractions=max_diffractions)
    monitor = _make_monitor(
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=resolved_backend_request,
        max_diffractions=max_diffractions,
    )
    sync_gpu()
    setup_seconds = time.perf_counter() - setup_t0

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        dr.kernel_history_clear()
        forward_t0 = time.perf_counter()
        trace_output = tracer.trace(tx_point, monitor=monitor, verbose=False)
        result = trace_output.monitor(monitor.name) if hasattr(trace_output, "monitor") else trace_output
        path_gain = result.path_gain
        weights = _loss_weights(path_gain, grid_size)
        loss = dr.sum(path_gain * weights)
        dr.eval(loss)
        sync_gpu()
        forward_seconds = time.perf_counter() - forward_t0
        forward_history = list(dr.kernel_history())

        dr.kernel_history_clear()
        backward_t0 = time.perf_counter()
        dr.backward(loss, flags=_GRAD_FLAGS)
        gradient_value = dr.grad(differentiable_var)
        dr.eval(gradient_value)
        sync_gpu()
        backward_seconds = time.perf_counter() - backward_t0
        backward_history = list(dr.kernel_history())

    metadata = dict(getattr(result, "metadata", {}) or {})
    return {
        "parameter": str(parameter),
        "parameter_config": _jsonable(config),
        "accumulation_backend_requested": requested_backend,
        "accumulation_backend_resolved_request": resolved_backend_request,
        "gradient_value": float(_scalar_from_drjit(gradient_value)),
        "loss_value": float(_scalar_from_drjit(loss)),
        "timing": {
            "scene_build_seconds": float(scene_build_seconds),
            "monitor_tracer_setup_seconds": float(setup_seconds),
            "forward_seconds": float(forward_seconds),
            "backward_seconds": float(backward_seconds),
        },
        "runtime_backends": _jsonable(metadata.get("runtime_backends", {})),
        "path_counts": _jsonable(metadata.get("path_counts", {})),
        "forward_kernel_history": _summarize_kernel_history(
            forward_history,
            top_k=top_k,
            small_kernel_max_size=small_kernel_max_size,
            small_kernel_min_count=small_kernel_min_count,
        ),
        "forward_custom_launches": _custom_launch_summary(forward_history, limit=top_k),
        "backward_kernel_history": _summarize_kernel_history(
            backward_history,
            top_k=top_k,
            small_kernel_max_size=small_kernel_max_size,
            small_kernel_min_count=small_kernel_min_count,
        ),
        "backward_custom_launches": _custom_launch_summary(backward_history, limit=top_k),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", choices=("tx_x", "cube1_x", "both"), default="both")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--n-rays", type=int, default=DEFAULT_N_RAYS)
    parser.add_argument("--max-diffractions", type=int, default=DEFAULT_MAX_DIFFRACTIONS)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--tx-x", type=float, default=float(DEFAULT_TX_POS[0]))
    parser.add_argument("--tx-y", type=float, default=float(DEFAULT_TX_POS[1]))
    parser.add_argument("--tx-z", type=float, default=float(DEFAULT_TX_POS[2]))
    parser.add_argument("--xmin", type=float, default=float(DEFAULT_BOUNDS[0][0]))
    parser.add_argument("--xmax", type=float, default=float(DEFAULT_BOUNDS[0][1]))
    parser.add_argument("--ymin", type=float, default=float(DEFAULT_BOUNDS[1][0]))
    parser.add_argument("--ymax", type=float, default=float(DEFAULT_BOUNDS[1][1]))
    parser.add_argument("--witwin-combine-mode", type=str, default=DEFAULT_COMBINE_MODE)
    parser.add_argument("--witwin-receiver-model", type=str, default=DEFAULT_RECEIVER_MODEL)
    parser.add_argument("--witwin-shadow-boundary-mode", type=str, default=DEFAULT_SHADOW_BOUNDARY_MODE)
    parser.add_argument("--accumulation-backend", type=str, default=DEFAULT_ACCUMULATION_BACKEND)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--small-kernel-max-size", type=int, default=1024)
    parser.add_argument("--small-kernel-min-count", type=int, default=4)
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

    if bool(args.flush_gpu_caches):
        flush_gpu_caches()

    parameters = ("tx_x", "cube1_x") if str(args.parameter) == "both" else (str(args.parameter),)
    results = []
    for parameter in parameters:
        results.append(
            _profile_parameter(
                parameter=parameter,
                tx_pos=tx_pos,
                grid_size=int(args.grid_size),
                n_rays=int(args.n_rays),
                bounds=bounds,
                plane_z=float(args.plane_z),
                combine_mode=str(args.witwin_combine_mode),
                receiver_model=str(args.witwin_receiver_model),
                shadow_boundary_mode=str(args.witwin_shadow_boundary_mode),
                accumulation_backend=str(args.accumulation_backend),
                max_diffractions=int(args.max_diffractions),
                top_k=int(args.top_k),
                small_kernel_max_size=int(args.small_kernel_max_size),
                small_kernel_min_count=int(args.small_kernel_min_count),
            )
        )

    summary = {
        "environment": benchmark_environment_report(),
        "scenario": {
            "grid_size": int(args.grid_size),
            "bounds": _jsonable(bounds),
            "plane_z": float(args.plane_z),
            "tx_pos": _jsonable(tx_pos),
            "n_rays": int(args.n_rays),
            "max_diffractions": int(args.max_diffractions),
            "combine_mode": str(args.witwin_combine_mode),
            "receiver_model": str(args.witwin_receiver_model),
            "shadow_boundary_mode": str(args.witwin_shadow_boundary_mode),
            "accumulation_backend": str(args.accumulation_backend),
            "cube1_x": float(CUBE1_BASE_CENTER[0]),
        },
        "parameters": results,
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
