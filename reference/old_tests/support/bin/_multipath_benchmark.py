"""Benchmark helpers for the fixed ``grad_multipath`` bin workload."""

from __future__ import annotations

import contextlib
import io
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

try:
    from ._benchmark_runtime import (
        assert_native_benchmark_support,
        benchmark_environment_report,
        extract_monitor_performance_timing,
        extract_monitor_runtime_backends,
    )
    from ._paths import REPO_ROOT
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import (
        assert_native_benchmark_support,
        benchmark_environment_report,
        extract_monitor_performance_timing,
        extract_monitor_runtime_backends,
    )
    from _paths import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import drjit as dr
import witwin as wt

from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import DEFAULT_VARIANT, FieldMonitor, Tracer

try:
    from ._monitor import DEFAULT_MONITOR_NAME, assert_plane_monitor_result
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _monitor import DEFAULT_MONITOR_NAME, assert_plane_monitor_result

_GRAD_TRACE_CONFIG = {
    "trace": {
        "diffraction_execution": {
            "suffix_dda": "symbolic",
        }
    }
}

_DEVICE_ALLOCATOR_RE = re.compile(
    r"- device\s*:\s*(?P<used>[^/]+)/(?P<reserved>[^ ]+\s+[A-Za-z]+) used \(peak:\s*(?P<peak>[^)]+)\)\."
)
_MEMORY_VALUE_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[KMGT]?i?B|B)\s*$")
_MEMORY_UNIT_SCALE = {
    "B": 1,
    "KiB": 1024,
    "MiB": 1024 ** 2,
    "GiB": 1024 ** 3,
    "TiB": 1024 ** 4,
}


@dataclass(slots=True)
class GradMultipathCase:
    """Concrete scene/tracer setup shared by the manual demo and benchmarks."""

    scene: Any
    scene_c1_only: Any
    tracer: Tracer
    tracer_c1_only: Tracer
    tx_x: wt.Float
    tx_y: wt.Float
    tx_z: wt.Float
    tx_pos: wt.Point3f
    tx_pos_eval: wt.Point3f
    monitor: FieldMonitor
    grid_size: int
    range_xy: tuple[float, float]
    frequency: float


def detached_point3f(point: wt.Point3f) -> wt.Point3f:
    """Return a Point3f view that does not retain the upstream AD graph."""
    return wt.Point3f(
        dr.detach(point.x),
        dr.detach(point.y),
        dr.detach(point.z),
    )


def create_grad_multipath_case() -> GradMultipathCase:
    """Build the exact scene/tracer workload used by ``tests/bin/grad_multipath.py``."""
    cube1 = box_drjit_geometry(center=(-2.5, -3.0, 1.5), size=2.0, rotation=None).to_mesh()
    cube2 = box_drjit_geometry(center=(2.0, 0.5, 1.5), size=2.0, rotation=None).to_mesh()
    cube3 = box_drjit_geometry(center=(-0.5, 3.5, 1.5), size=2.0, rotation=None).to_mesh()

    scene = build_test_scene(cube1, cube2, cube3)
    scene_c1_only = build_test_scene(cube1)

    frequency = 1e9
    tracer = Tracer(
        frequency=frequency,
        scene=scene,
        config=_GRAD_TRACE_CONFIG,
        reflection_n_rays=10000,
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
    )
    tracer_c1_only = Tracer(
        frequency=frequency,
        scene=scene_c1_only,
        config=_GRAD_TRACE_CONFIG,
        reflection_n_rays=10000,
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
    )

    tx_x = wt.Float(0.0)
    tx_y = wt.Float(-5.0)
    tx_z = wt.Float(1.5)
    dr.enable_grad(tx_x)
    dr.enable_grad(tx_y)
    dr.enable_grad(tx_z)
    tx_pos = wt.Point3f(tx_x, tx_y, tx_z)
    tx_pos_eval = detached_point3f(tx_pos)

    grid_size = 256
    range_xy = (-6.0, 6.0)
    monitor = FieldMonitor(
        DEFAULT_MONITOR_NAME,
        axis="z",
        position=1.5,
        bounds=(range_xy, range_xy),
        grid_size=grid_size,
    )
    scene.add_monitor(monitor)
    scene_c1_only.add_monitor(monitor)

    return GradMultipathCase(
        scene=scene,
        scene_c1_only=scene_c1_only,
        tracer=tracer,
        tracer_c1_only=tracer_c1_only,
        tx_x=tx_x,
        tx_y=tx_y,
        tx_z=tx_z,
        tx_pos=tx_pos,
        tx_pos_eval=tx_pos_eval,
        monitor=monitor,
        grid_size=grid_size,
        range_xy=range_xy,
        frequency=frequency,
    )


def _scalar_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(value[0])


def _memory_text_to_bytes(text: str | None) -> int | None:
    if text is None:
        return None
    match = _MEMORY_VALUE_RE.match(text)
    if match is None:
        return None
    value = float(match.group("value"))
    unit = match.group("unit")
    return int(round(value * _MEMORY_UNIT_SCALE[unit]))


def capture_drjit_allocator(*, include_raw: bool = False) -> dict[str, Any]:
    """Capture ``dr.whos()`` and extract the device allocator line."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        dr.whos()
    raw = stream.getvalue()
    match = _DEVICE_ALLOCATOR_RE.search(raw)
    if match is None:
        return {"raw": raw} if include_raw else {}

    used = match.group("used").strip()
    reserved = match.group("reserved").strip()
    peak = match.group("peak").strip()
    allocator = {
        "device_used": used,
        "device_reserved": reserved,
        "device_peak": peak,
        "device_used_bytes": _memory_text_to_bytes(used),
        "device_reserved_bytes": _memory_text_to_bytes(reserved),
        "device_peak_bytes": _memory_text_to_bytes(peak),
    }
    if include_raw:
        allocator["raw"] = raw
    return allocator


def _run_grad_multipath_benchmark_once(*, label: str, verbose: bool) -> dict[str, Any]:
    """Run one measured pass of the fixed forward/backward benchmark."""
    if hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()

    case = create_grad_multipath_case()
    assert_native_benchmark_support(
        benchmark_name="grad_multipath_benchmark",
        reflection_field_backend=case.tracer.config.trace.reflection_field_backend,
        diffraction_execution=case.tracer.config.trace.diffraction_execution,
    )
    result = case.tracer.trace(case.tx_pos, verbose=verbose, return_timing=True)
    assert_plane_monitor_result(result, case.monitor)
    payload = result.primary

    if hasattr(dr, "sync_thread"):
        dr.sync_thread()

    trace_timing = {} if payload.timing is None else {key: float(value) for key, value in payload.timing.items()}
    forward_seconds = float(sum(trace_timing.values()))
    runtime_environment = benchmark_environment_report()
    runtime_backends = extract_monitor_runtime_backends(payload)
    performance_timing = extract_monitor_performance_timing(payload)

    total_field = payload.field.total
    loss = dr.sum(total_field.real * total_field.real + total_field.imag * total_field.imag)

    if hasattr(dr, "sync_thread"):
        dr.sync_thread()
    backward_start = time.perf_counter()
    dr.backward(loss)
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()
    backward_seconds = time.perf_counter() - backward_start

    grad_x = _scalar_float(dr.grad(case.tx_x))
    grad_y = _scalar_float(dr.grad(case.tx_y))
    grad_z = _scalar_float(dr.grad(case.tx_z))
    grad_norm = math.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
    allocator = capture_drjit_allocator()

    return {
        "label": label,
        "frequency_hz": case.frequency,
        "grid_size": case.grid_size,
        "range_xy": list(case.range_xy),
        "reflection_n_rays": case.tracer.reflection_n_rays,
        "reflection_max_bounces": case.tracer.reflection_max_bounces,
        "enable_rd_diffraction": bool(case.tracer.enable_rd_diffraction),
        "forward_seconds": forward_seconds,
        "trace_timing": trace_timing,
        "runtime_environment": runtime_environment,
        "runtime_backends": runtime_backends,
        "performance_timing": performance_timing,
        "backward_seconds": backward_seconds,
        "loss": _scalar_float(loss),
        "tx_grad": {
            "x": grad_x,
            "y": grad_y,
            "z": grad_z,
            "norm": grad_norm,
        },
        "allocator": allocator,
    }


def run_grad_multipath_benchmark(
    *,
    label: str = "manual",
    verbose: bool = False,
    warmup_runs: int = 0,
) -> dict[str, Any]:
    """Run the fixed forward/backward benchmark for the multipath sample workload."""
    for _ in range(max(0, int(warmup_runs))):
        _run_grad_multipath_benchmark_once(label=f"{label}-warmup", verbose=False)
    return _run_grad_multipath_benchmark_once(label=label, verbose=verbose)


def _backend_summary_text(runtime_backends: dict[str, Any]) -> str:
    reflection = runtime_backends.get("reflection", {})
    diffraction = runtime_backends.get("diffraction", {})
    suffix = runtime_backends.get("suffix", {})
    reflection_impl = reflection.get("implementation", reflection.get("resolved_backend", "unknown"))
    diffraction_impl = diffraction.get("implementation", diffraction.get("resolved_primal", "unknown"))
    suffix_impl = suffix.get("implementation", suffix.get("resolved_backend", "unknown"))
    return f"ref={reflection_impl}, dif={diffraction_impl}, suffix={suffix_impl}"


def format_benchmark_summary(result: dict[str, Any]) -> str:
    """Format a compact human-readable benchmark summary."""
    trace_parts = ", ".join(
        f"{key}={value:.3f}s"
        for key, value in result["trace_timing"].items()
    )
    allocator = result["allocator"]
    peak = allocator.get("device_peak", "unknown")
    used = allocator.get("device_used", "unknown")
    backend_summary = _backend_summary_text(result.get("runtime_backends", {}))
    return (
        f"[{result['label']}] forward={result['forward_seconds']:.3f}s "
        f"backward={result['backward_seconds']:.3f}s "
        f"loss={result['loss']:.6e} "
        f"|grad|={result['tx_grad']['norm']:.6e} "
        f"backends[{backend_summary}] "
        f"allocator(device used/peak)={used}/{peak} "
        f"timing[{trace_parts}]"
    )


__all__ = [
    "GradMultipathCase",
    "capture_drjit_allocator",
    "create_grad_multipath_case",
    "detached_point3f",
    "format_benchmark_summary",
    "run_grad_multipath_benchmark",
]
