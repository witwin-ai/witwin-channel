"""Shared helpers for multipath scaling stress workers."""

from __future__ import annotations

import contextlib
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

try:
    from ._benchmark_runtime import (
        benchmark_environment_report,
        extract_monitor_performance_timing,
        extract_monitor_runtime_backends,
    )
    from ._multipath_benchmark import capture_drjit_allocator
    from ._paths import REPO_ROOT
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import (
        benchmark_environment_report,
        extract_monitor_performance_timing,
        extract_monitor_runtime_backends,
    )
    from _multipath_benchmark import capture_drjit_allocator
    from _paths import REPO_ROOT

import drjit as dr
import torch

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import witwin as wt
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from tests.main.plot_multipath_components import (
    CUBE1_BASE_CENTER,
    CUBE2_CENTER,
    CUBE3_CENTER,
    CUBE_SIZE,
    MULTIPATH_SCENE_MATERIAL,
    TX_POS,
    make_monitor,
    make_tracer,
)
from witwin.channel.monitors.field.trace import trace_field_monitor_total_only
from witwin.channel.monitors.orchestration import resolve_solver_controls

FORWARD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad
_MOTIF_SPACING = 10.0


def sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def flush_gpu_caches() -> None:
    sync_gpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()
    sync_gpu()


def bytes_text(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def float_scalar(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "__len__") and len(value) == 0:
        return 0.0
    return float(value[0])


def torch_memory_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    device = torch.cuda.current_device()
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    max_allocated = int(torch.cuda.max_memory_allocated(device))
    max_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "available": True,
        "device_index": int(device),
        "device_name": torch.cuda.get_device_name(device),
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "max_allocated_bytes": max_allocated,
        "max_reserved_bytes": max_reserved,
        "allocated": bytes_text(allocated),
        "reserved": bytes_text(reserved),
        "max_allocated": bytes_text(max_allocated),
        "max_reserved": bytes_text(max_reserved),
    }


def memory_snapshot() -> dict[str, Any]:
    return {
        "torch_cuda": torch_memory_snapshot(),
        "drjit_allocator": capture_drjit_allocator(),
    }


def measure_phase(name: str, fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    sync_gpu()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before = memory_snapshot()
    start = time.perf_counter()
    value = fn()
    sync_gpu()
    elapsed = time.perf_counter() - start
    after = memory_snapshot()
    return value, {
        "name": name,
        "seconds": elapsed,
        "memory_before": before,
        "memory_after": after,
    }


@contextlib.contextmanager
def benchmark_timing_mode(*, sync_timing: bool):
    import os

    key = "WITWIN_BENCHMARK_SYNC_TIMING"
    previous = os.environ.get(key)
    if sync_timing:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _motif_offsets(motif_repeats: int) -> list[tuple[float, float]]:
    repeats = int(motif_repeats)
    if repeats <= 0:
        raise ValueError("motif_repeats must be > 0.")
    offsets = [(0.0, 0.0)]
    ring = 1
    while len(offsets) < repeats:
        for dy in range(-ring, ring + 1):
            for dx in range(-ring, ring + 1):
                if max(abs(dx), abs(dy)) != ring:
                    continue
                offsets.append((float(dx) * _MOTIF_SPACING, float(dy) * _MOTIF_SPACING))
                if len(offsets) == repeats:
                    return offsets
        ring += 1
    return offsets


@lru_cache(maxsize=32)
def build_scene_for_motif_repeats(motif_repeats: int):
    meshes = []
    for offset_x, offset_y in _motif_offsets(int(motif_repeats)):
        for center in (CUBE1_BASE_CENTER, CUBE2_CENTER, CUBE3_CENTER):
            shifted_center = (
                center[0] + offset_x,
                center[1] + offset_y,
                center[2],
            )
            meshes.append(
                box_drjit_geometry(
                    center=shifted_center,
                    size=CUBE_SIZE,
                    rotation=None,
                ).to_mesh()
            )
    return build_test_scene(*meshes, material=MULTIPATH_SCENE_MATERIAL)


def scene_metrics(scene, motif_repeats: int) -> dict[str, Any]:
    n_triangles = 0
    if scene.tri_data_gpu is not None:
        n_triangles = int(scene.tri_data_gpu["n_triangles"])
    return {
        "motif_repeats": int(motif_repeats),
        "structures": int(len(scene.structures)),
        "triangles": int(n_triangles),
        "diffraction_edges": int(scene.n_diffraction_edges),
    }


def create_case(*, grid_size: int, n_rays: int, motif_repeats: int) -> dict[str, Any]:
    variable = wt.Float(TX_POS[0])
    dr.enable_grad(variable)
    tx_pos = wt.Point3f(variable, wt.Float(TX_POS[1]), wt.Float(TX_POS[2]))
    scene = build_scene_for_motif_repeats(int(motif_repeats))
    monitor = make_monitor(int(grid_size))
    tracer = make_tracer(scene, int(n_rays))
    solver_controls = resolve_solver_controls(
        tracer.config.trace,
        execution_intent="field_scalar_only",
    )
    return {
        "variable": variable,
        "seed_value": 1.0,
        "scene": scene,
        "scene_metrics": scene_metrics(scene, motif_repeats),
        "monitor": monitor,
        "tracer": tracer,
        "solver_controls": solver_controls,
        "tx_pos": tx_pos,
        "grid_size": int(grid_size),
        "n_rays": int(n_rays),
        "motif_repeats": int(motif_repeats),
    }


def trace_case(case: dict[str, Any]) -> dict[str, Any]:
    payload, _ = trace_field_monitor_total_only(
        case["tx_pos"],
        case["monitor"],
        case["scene"],
        case["tracer"]._resolved_trace_config,
        case["solver_controls"],
        verbose=False,
        return_timing=True,
        return_diffraction_audit=False,
    )
    trace_timing = {} if payload.get("timing") is None else {
        key: float(value) for key, value in payload["timing"].items()
    }
    return {
        "payload": payload,
        "trace_timing": trace_timing,
        "trace_timing_sum_seconds": float(sum(trace_timing.values())),
        "runtime_backends": extract_monitor_runtime_backends(payload),
        "performance_timing": extract_monitor_performance_timing(payload),
    }


def build_loss(trace_payload: dict[str, Any]) -> dict[str, Any]:
    total = trace_payload["payload"]["field"]["total"]
    loss = dr.sum(total.real * total.real + total.imag * total.imag)
    dr.eval(loss)
    return {
        "loss": loss,
        "loss_value": float_scalar(loss),
    }


def print_single_run_summary(result: dict[str, Any], *, diff_phase_name: str) -> None:
    trace_peak = result["phase_metrics"]["trace"]["memory_after"]["drjit_allocator"].get("device_peak", "n/a")
    diff_peak = result["phase_metrics"][diff_phase_name]["memory_after"]["drjit_allocator"].get("device_peak", "n/a")
    print(
        f"grid={result['grid_size']} rays={result['n_rays']} motifs={result['motif_repeats']} "
        f"triangles={result['scene_metrics']['triangles']} receivers={result['n_receivers']}"
    )
    print(
        f"  setup={result['setup_seconds']:.3f}s trace={result['trace_seconds']:.3f}s "
        f"loss={result['loss_seconds']:.3f}s {diff_phase_name}={result[f'{diff_phase_name}_seconds']:.3f}s "
        f"total={result['total_seconds']:.3f}s"
    )
    print(f"  drjit_peak(trace/{diff_phase_name})={trace_peak}/{diff_peak}")


__all__ = [
    "FORWARD_FLAGS",
    "benchmark_environment_report",
    "benchmark_timing_mode",
    "build_loss",
    "create_case",
    "float_scalar",
    "flush_gpu_caches",
    "measure_phase",
    "memory_snapshot",
    "print_single_run_summary",
    "trace_case",
]
