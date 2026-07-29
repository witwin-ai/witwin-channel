# Copyright Xingyu Chen.
# Benchmarks harness.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any, TypeVar


SCHEMA_NAME = "witwin.channel.performance"
SCHEMA_VERSION = "1.0.0"
T = TypeVar("T")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


@dataclass(frozen=True, slots=True)
class SampleTiming:
    wall_ms: float
    cuda_event_ms: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    first: SampleTiming
    steady: tuple[SampleTiming, ...]
    steady_wall_median_ms: float
    steady_wall_p95_ms: float
    steady_cuda_median_ms: float | None
    steady_cuda_p95_ms: float | None
    memory: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steady"] = [asdict(row) for row in self.steady]
        return payload


def _sync_result(result: Any) -> None:
    import torch

    if hasattr(result, "valid") and getattr(result, "valid") is not None:
        getattr(result, "valid").numel()
    torch.cuda.synchronize()


def _measure_once(operation: Callable[[], T], sync: Callable[[T], None]) -> tuple[T, SampleTiming]:
    import torch

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    result = operation()
    end_event.record()
    sync(result)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return result, SampleTiming(
        wall_ms=wall_ms,
        cuda_event_ms=float(start_event.elapsed_time(end_event)),
    )


def benchmark_operation(
    operation: Callable[[], T],
    *,
    warmup: int = 1,
    repeats: int = 5,
    sync: Callable[[T], None] | None = None,
) -> tuple[T, BenchmarkMeasurement]:
    """Measure first call and steady state with wall/CUDA clocks and peak memory."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the unified Channel benchmark harness requires CUDA")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    synchronize = _sync_result if sync is None else sync

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    device_free_before, device_total = torch.cuda.mem_get_info()
    tracemalloc.start()
    tracemalloc.reset_peak()
    result, first = _measure_once(operation, synchronize)
    for _ in range(warmup):
        result = operation()
        synchronize(result)
    steady_rows: list[SampleTiming] = []
    for _ in range(repeats):
        result, row = _measure_once(operation, synchronize)
        steady_rows.append(row)
    host_current, host_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall = [row.wall_ms for row in steady_rows]
    cuda = [row.cuda_event_ms for row in steady_rows if row.cuda_event_ms is not None]
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    allocated_after = int(torch.cuda.memory_allocated())
    reserved_after = int(torch.cuda.memory_reserved())
    device_free_after, _ = torch.cuda.mem_get_info()
    output_bytes = tensor_bytes(result)
    measurement = BenchmarkMeasurement(
        first=first,
        steady=tuple(steady_rows),
        steady_wall_median_ms=float(statistics.median(wall)),
        steady_wall_p95_ms=float(_percentile(wall, 0.95)),
        steady_cuda_median_ms=float(statistics.median(cuda)) if cuda else None,
        steady_cuda_p95_ms=_percentile(cuda, 0.95),
        memory={
            "persistent_allocated_before_bytes": allocated_before,
            "persistent_reserved_before_bytes": reserved_before,
            "persistent_allocated_after_bytes": allocated_after,
            "persistent_reserved_after_bytes": reserved_after,
            "persistent_growth_excluding_output_bytes": max(
                0, allocated_after - allocated_before - output_bytes
            ),
            "output_bytes": output_bytes,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_temporary_allocated_bytes": max(
                0, peak_allocated - max(allocated_before, allocated_after)
            ),
            "device_total_bytes": int(device_total),
            "device_used_before_bytes": int(device_total - device_free_before),
            "device_used_after_bytes": int(device_total - device_free_after),
            "device_persistent_growth_bytes": max(
                0, int(device_free_before - device_free_after)
            ),
            "host_traced_current_bytes": int(host_current),
            "host_traced_peak_bytes": int(host_peak),
        },
    )
    return result, measurement


def measure_cold_import(
    module: str = "witwin.channel", *, timeout_s: float = 120.0
) -> dict[str, Any]:
    """Measure a source-tree fresh import; this is not an installed-wheel smoke."""

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    core_root = repo_root.parent / "core-radar-architecture-stage1"
    source_paths = [str(core_root), str(repo_root)]
    inherited = env.get("PYTHONPATH")
    if inherited:
        source_paths.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.meta_path=[finder for finder in sys.meta_path "
            "if '_witwin_channel_editable' not in type(finder).__module__]; "
            f"import {module}"
        ),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return {
        "scope": "source_tree",
        "module": module,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stderr": completed.stderr.strip(),
    }


def environment_record() -> dict[str, Any]:
    from witwin.channel.deployment import runtime_diagnostics

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit_sha": commit.stdout.strip() if commit.returncode == 0 else None,
        "python": sys.version,
        "platform": platform.platform(),
        "runtime": runtime_diagnostics(),
    }


def tensor_bytes(value: Any) -> int:
    """Count tensor storage represented by a result without copying it to host."""

    try:
        import torch
    except ImportError:
        return 0
    seen: set[int] = set()

    def visit(item: Any) -> int:
        if isinstance(item, torch.Tensor):
            storage_id = item.untyped_storage().data_ptr()
            if storage_id in seen:
                return 0
            seen.add(storage_id)
            return int(item.untyped_storage().nbytes())
        if isinstance(item, dict):
            return sum(visit(child) for child in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(child) for child in item)
        if is_dataclass(item) and not isinstance(item, type):
            return sum(visit(getattr(item, field.name)) for field in fields(item))
        return 0

    return visit(value)


def versioned_report(*, benchmark: str, scenario: Any, results: Any) -> dict[str, Any]:
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "benchmark": benchmark,
        "scenario": scenario,
        "environment": environment_record(),
        "results": results,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")