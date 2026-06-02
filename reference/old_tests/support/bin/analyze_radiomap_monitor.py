"""Diagnostic comparison for RadioMapMonitor modes on the multipath scene."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import witwin as wt
from tests.main.plot_multipath_components import (
    CUBE1_BASE_CENTER,
    TRACE_BOUNDS,
    TX_POS,
    build_scene_for_cube1_x,
)
from witwin.channel import FieldMonitor, RadioMapMonitor, Tracer
from witwin.channel.monitors.radio_map.deterministic.trace import trace_radio_map_monitor
from witwin.channel.monitors.orchestration import resolve_solver_controls
def _as_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _as_complex_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.complex64)


def _max_location(values: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> dict[str, Any]:
    index = np.unravel_index(int(np.argmax(values)), values.shape)
    return {
        "index": [int(index[0]), int(index[1])],
        "x": float(x_grid[index]),
        "y": float(y_grid[index]),
        "value": float(values[index]),
    }


def _timed_call(fn: Callable[[], Any], *, warmup: int) -> tuple[Any, float]:
    result = None
    for _ in range(max(0, int(warmup))):
        result = fn()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    return result, float(elapsed)


def _field_case(*, tracer: Tracer, grid_size: int, warmup: int):
    monitor = FieldMonitor(
        "analysis_field",
        axis="z",
        position=TX_POS[2],
        bounds=TRACE_BOUNDS,
        grid_size=int(grid_size),
        ray_mode="3d",
        ray_sampling="full_sphere",
    )

    def _run():
        return tracer.trace(wt.Point3f(*TX_POS), monitor=monitor, verbose=False).monitor(monitor.name)

    result, elapsed = _timed_call(_run, warmup=warmup)
    total = _as_complex_array(result.field.total)
    los = _as_complex_array(result.field.los)
    reflection = _as_complex_array(result.field.reflection)
    diffraction = _as_complex_array(result.field.diffraction)
    return {
        "elapsed_seconds": elapsed,
        "runtime_backends": dict(result.metadata.get("runtime_backends", {})),
        "total_abs_max": float(np.abs(total).max()),
        "total_power_max": float((np.abs(total) ** 2).max()),
        "component_abs_max": {
            "los": float(np.abs(los).max()),
            "reflection": float(np.abs(reflection).max()),
            "diffraction": float(np.abs(diffraction).max()),
        },
    }


def _radio_map_payload_case(
    *,
    tracer: Tracer,
    grid_size: int,
    combine_mode: str,
    backend: str,
    warmup: int,
):
    monitor = RadioMapMonitor(
        f"analysis_rm_{combine_mode}_{backend}",
        axis="z",
        position=TX_POS[2],
        bounds=TRACE_BOUNDS,
        grid_shape=(int(grid_size), int(grid_size)),
        combine_mode=str(combine_mode),
        ray_mode="3d",
        quadrature_mode="center",
    )
    config = tracer._resolved_trace_config
    solver_controls = resolve_solver_controls(config)

    def _run():
        return trace_radio_map_monitor(
            wt.Point3f(*TX_POS),
            monitor,
            tracer.scene,
            config,
            solver_controls,
            radio_map_accumulation_backend=str(backend),
        )

    payload, elapsed = _timed_call(_run, warmup=warmup)
    tensor_shape = (int(grid_size), int(grid_size))
    path_gain = _as_float_array(payload["metrics"]["path_gain"]).reshape(tensor_shape)
    x_grid = _as_float_array(payload["coords"]["grid_x"]).reshape(tensor_shape)
    y_grid = _as_float_array(payload["coords"]["grid_y"]).reshape(tensor_shape)
    coherent = {
        key: _as_complex_array(value).reshape(tensor_shape)
        for key, value in dict(payload["diagnostics"]["coherent"]).items()
    }
    incoherent = {
        key: _as_float_array(value).reshape(tensor_shape)
        for key, value in dict(payload["diagnostics"]["incoherent"]).items()
    }
    coherent_power = {
        key: _as_float_array(value).reshape(tensor_shape)
        for key, value in dict(payload["diagnostics"]["coherent_power"]).items()
    }
    return {
        "elapsed_seconds": elapsed,
        "path_gain_max": float(path_gain.max()),
        "path_gain_mean": float(path_gain.mean()),
        "path_counts": dict(payload["metadata"].get("path_counts", {})),
        "accumulation_backend": dict(payload["metadata"].get("accumulation_backend", {})),
        "runtime_backends": dict(payload["metadata"].get("runtime_backends", {})),
        "coherent_abs_max": {
            key: float(np.abs(value).max())
            for key, value in coherent.items()
        },
        "incoherent_max": {
            key: float(value.max())
            for key, value in incoherent.items()
        },
        "coherent_power_max": {
            key: float(value.max())
            for key, value in coherent_power.items()
        },
        "diffraction_peak": _max_location(
            coherent_power["diffraction"],
            x_grid,
            y_grid,
        ),
    }


def build_report(*, grid_size: int, n_rays: int, warmup: int) -> dict[str, Any]:
    scene = build_scene_for_cube1_x(CUBE1_BASE_CENTER[0])
    tracer = Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=int(n_rays),
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )
    return {
        "scene": {
            "name": "multipath_components",
            "cube1_x": float(CUBE1_BASE_CENTER[0]),
            "tx_pos": [float(value) for value in TX_POS],
            "bounds": [[float(a), float(b)] for (a, b) in TRACE_BOUNDS],
        },
        "config": {
            "grid_size": int(grid_size),
            "n_rays": int(n_rays),
            "warmup": int(warmup),
        },
        "field_monitor": _field_case(
            tracer=tracer,
            grid_size=grid_size,
            warmup=warmup,
        ),
        "radio_map": {
            "incoherent_baseline": _radio_map_payload_case(
                tracer=tracer,
                grid_size=grid_size,
                combine_mode="incoherent",
                backend="baseline",
                warmup=warmup,
            ),
            "coherent_baseline": _radio_map_payload_case(
                tracer=tracer,
                grid_size=grid_size,
                combine_mode="coherent",
                backend="baseline",
                warmup=warmup,
            ),
            "coherent_native": _radio_map_payload_case(
                tracer=tracer,
                grid_size=grid_size,
                combine_mode="coherent",
                backend="native_coherent",
                warmup=warmup,
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--n-rays", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    report = build_report(
        grid_size=args.grid_size,
        n_rays=args.n_rays,
        warmup=args.warmup,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
