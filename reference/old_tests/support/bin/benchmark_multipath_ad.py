"""Benchmark multipath forward, VJP, and JVP timings for selected parameters."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
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

import drjit as dr
import torch
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import witwin as wt
from witwin.channel import DEFAULT_VARIANT
from witwin.channel.monitors.field.trace import trace_field_monitor_total_only
from witwin.channel.monitors.orchestration import resolve_solver_controls
try:
    from ._monitor import assert_plane_monitor_result
    from ._multipath_benchmark import capture_drjit_allocator
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _monitor import assert_plane_monitor_result
    from _multipath_benchmark import capture_drjit_allocator
from tests.main.plot_multipath_components import (
    CUBE1_BASE_CENTER,
    TX_POS,
    build_scene_for_cube1_x,
    make_monitor,
    make_tracer,
)

_FORWARD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad


def _sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def _flush_gpu_caches() -> None:
    _sync_gpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()
    _sync_gpu()


def _bytes_text(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _float_scalar(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "__len__") and len(value) == 0:
        return 0.0
    return float(value[0])


def _torch_memory_snapshot() -> dict[str, Any]:
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
        "allocated": _bytes_text(allocated),
        "reserved": _bytes_text(reserved),
        "max_allocated": _bytes_text(max_allocated),
        "max_reserved": _bytes_text(max_reserved),
    }


def _memory_snapshot() -> dict[str, Any]:
    return {
        "torch_cuda": _torch_memory_snapshot(),
        "drjit_allocator": capture_drjit_allocator(),
    }


def _measure_phase(
    name: str,
    fn: Callable[[], Any],
) -> tuple[Any, dict[str, Any]]:
    _sync_gpu()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before = _memory_snapshot()
    start = time.perf_counter()
    value = fn()
    _sync_gpu()
    elapsed = time.perf_counter() - start
    after = _memory_snapshot()
    return value, {
        "name": name,
        "seconds": elapsed,
        "memory_before": before,
        "memory_after": after,
    }


def _create_case(parameter: str, *, grid_size: int, n_rays: int, workload: str) -> dict[str, Any]:
    if parameter == "cube1_x":
        variable = wt.Float(CUBE1_BASE_CENTER[0])
        dr.enable_grad(variable)
        scene = build_scene_for_cube1_x(variable)
        tx_pos = wt.Point3f(*TX_POS)
        seed_value = 1.0
    elif parameter == "tx_x":
        variable = wt.Float(TX_POS[0])
        dr.enable_grad(variable)
        tx_pos = wt.Point3f(variable, wt.Float(TX_POS[1]), wt.Float(TX_POS[2]))
        scene = build_scene_for_cube1_x(CUBE1_BASE_CENTER[0])
        seed_value = 1.0
    else:
        raise ValueError(f"Unsupported parameter: {parameter}")

    monitor = make_monitor(grid_size)
    tracer = make_tracer(scene, n_rays)
    return {
        "parameter": parameter,
        "variable": variable,
        "seed_value": seed_value,
        "scene": scene,
        "monitor": monitor,
        "tracer": tracer,
        "solver_controls": resolve_solver_controls(
            tracer.config.trace,
            execution_intent="field_scalar_only" if workload == "scalar_loss" else "field",
        ),
        "tx_pos": tx_pos,
        "grid_size": grid_size,
        "n_rays": n_rays,
    }


def _minimal_monitor_payload(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(
        field=SimpleNamespace(total=payload["field"]["total"]),
        metadata=payload.get("metadata", {}),
        timing=payload.get("timing"),
        tx_pos=payload["tx_pos"],
        payload_kind=payload.get("payload_kind", "field_total_only"),
    )


def _trace_case(case: dict[str, Any], *, verbose: bool, workload: str) -> dict[str, Any]:
    if workload == "scalar_loss":
        payload_dict, _ = trace_field_monitor_total_only(
            case["tx_pos"],
            case["monitor"],
            case["scene"],
            case["tracer"]._resolved_trace_config,
            case["solver_controls"],
            verbose=verbose,
            return_timing=True,
            return_diffraction_audit=False,
        )
        payload = _minimal_monitor_payload(payload_dict)
        trace_timing = (
            {} if payload_dict.get("timing") is None else {key: float(value) for key, value in payload_dict["timing"].items()}
        )
        return {
            "payload": payload,
            "trace_timing": trace_timing,
            "trace_timing_sum_seconds": float(sum(trace_timing.values())),
            "runtime_backends": extract_monitor_runtime_backends(payload),
            "performance_timing": extract_monitor_performance_timing(payload),
        }

    result = case["tracer"].trace(
        case["tx_pos"],
        monitor=case["monitor"],
        verbose=verbose,
        return_timing=True,
        return_diffraction_audit=False,
    )
    assert_plane_monitor_result(result, case["monitor"])
    payload = result.primary
    trace_timing = {} if payload.timing is None else {key: float(value) for key, value in payload.timing.items()}
    return {
        "result": result,
        "payload": payload,
        "trace_timing": trace_timing,
        "trace_timing_sum_seconds": float(sum(trace_timing.values())),
        "runtime_backends": extract_monitor_runtime_backends(payload),
        "performance_timing": extract_monitor_performance_timing(payload),
    }


def _build_loss(trace_payload: dict[str, Any]) -> dict[str, Any]:
    total = trace_payload["payload"].field.total
    loss = dr.sum(total.real * total.real + total.imag * total.imag)
    dr.eval(loss)
    return {
        "loss": loss,
        "loss_value": _float_scalar(loss),
    }


def _run_vjp(case: dict[str, Any], loss_payload: dict[str, Any]) -> dict[str, Any]:
    dr.backward(loss_payload["loss"])
    return {
        "parameter_grad": _float_scalar(dr.grad(case["variable"])),
        "loss_grad": _float_scalar(dr.grad(loss_payload["loss"])),
    }


def _run_jvp(case: dict[str, Any], loss_payload: dict[str, Any]) -> dict[str, Any]:
    dr.set_grad(case["variable"], case["seed_value"])
    dr.forward_to(loss_payload["loss"], flags=_FORWARD_FLAGS)
    return {
        "seed": float(case["seed_value"]),
        "loss_jvp": _float_scalar(dr.grad(loss_payload["loss"])),
        "parameter_tangent": _float_scalar(dr.grad(case["variable"])),
    }


def run_single_pass(
    *,
    parameter: str,
    mode: str,
    grid_size: int,
    n_rays: int,
    workload: str,
    verbose_trace: bool,
) -> dict[str, Any]:
    _flush_gpu_caches()

    case, setup_phase = _measure_phase(
        "setup",
        lambda: _create_case(parameter, grid_size=grid_size, n_rays=n_rays, workload=workload),
    )
    assert_native_benchmark_support(
        benchmark_name="multipath_ad_timing",
        reflection_field_backend=case["tracer"].config.trace.reflection_field_backend,
        diffraction_execution=case["tracer"].config.trace.diffraction_execution,
    )
    trace_payload, trace_phase = _measure_phase(
        "trace",
        lambda: _trace_case(case, verbose=verbose_trace, workload=workload),
    )
    loss_payload, loss_phase = _measure_phase(
        "loss",
        lambda: _build_loss(trace_payload),
    )

    if mode == "vjp":
        diff_payload, diff_phase = _measure_phase(
            "vjp",
            lambda: _run_vjp(case, loss_payload),
        )
    elif mode == "jvp":
        diff_payload, diff_phase = _measure_phase(
            "jvp",
            lambda: _run_jvp(case, loss_payload),
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    payload = trace_payload["payload"]
    summary = {
        "parameter": parameter,
        "mode": mode,
        "workload": workload,
        "grid_size": int(grid_size),
        "n_rays": int(n_rays),
        "n_structures": int(len(case["scene"].structures)),
        "n_diffraction_edges": int(case["scene"].n_diffraction_edges),
        "setup_seconds": float(setup_phase["seconds"]),
        "trace_seconds": float(trace_phase["seconds"]),
        "loss_seconds": float(loss_phase["seconds"]),
        "diff_seconds": float(diff_phase["seconds"]),
        "total_seconds": float(
            setup_phase["seconds"] + trace_phase["seconds"] + loss_phase["seconds"] + diff_phase["seconds"]
        ),
        "trace_timing": trace_payload["trace_timing"],
        "trace_timing_sum_seconds": float(trace_payload["trace_timing_sum_seconds"]),
        "runtime_backends": trace_payload["runtime_backends"],
        "performance_timing": trace_payload["performance_timing"],
        "loss": float(loss_payload["loss_value"]),
        "phase_metrics": {
            "setup": setup_phase,
            "trace": trace_phase,
            "loss": loss_phase,
            mode: diff_phase,
        },
        "memory_final": _memory_snapshot(),
        "tx_pos": tuple(float(value) for value in payload.tx_pos),
        "result": diff_payload,
    }
    return summary


def run_benchmark(
    *,
    parameters: list[str],
    modes: list[str],
    grid_size: int,
    n_rays: int,
    repeats: int,
    warmup_runs: int,
    workload: str,
    verbose_trace: bool,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []

    for parameter in parameters:
        for mode in modes:
            for _ in range(max(0, warmup_runs)):
                run_single_pass(
                    parameter=parameter,
                    mode=mode,
                    grid_size=grid_size,
                    n_rays=n_rays,
                    workload=workload,
                    verbose_trace=False,
                )
            for repeat_index in range(max(1, repeats)):
                run = run_single_pass(
                    parameter=parameter,
                    mode=mode,
                    grid_size=grid_size,
                    n_rays=n_rays,
                    workload=workload,
                    verbose_trace=verbose_trace,
                )
                run["repeat_index"] = int(repeat_index)
                runs.append(run)

    return {
        "benchmark": "multipath_ad_timing",
        "runtime_environment": benchmark_environment_report(),
        "parameters": parameters,
        "modes": modes,
        "workload": workload,
        "grid_size": int(grid_size),
        "n_rays": int(n_rays),
        "repeats": int(max(1, repeats)),
        "warmup_runs": int(max(0, warmup_runs)),
        "runs": runs,
    }


def _format_trace_timing(trace_timing: dict[str, float]) -> str:
    if not trace_timing:
        return "n/a"
    return ", ".join(f"{name}={seconds:.3f}s" for name, seconds in trace_timing.items())


def _phase_peak_text(phase: dict[str, Any]) -> str:
    torch_cuda = phase["memory_after"]["torch_cuda"]
    drjit = phase["memory_after"]["drjit_allocator"]
    torch_peak = torch_cuda.get("max_allocated", "n/a")
    drjit_peak = drjit.get("device_peak", "n/a")
    return f"torch_peak={torch_peak}, drjit_peak={drjit_peak}"


def _runtime_backend_text(runtime_backends: dict[str, Any]) -> str:
    reflection = runtime_backends.get("reflection", {})
    diffraction = runtime_backends.get("diffraction", {})
    suffix = runtime_backends.get("suffix", {})
    reflection_impl = reflection.get("implementation", reflection.get("resolved_backend", "unknown"))
    diffraction_impl = diffraction.get("implementation", diffraction.get("resolved_primal", "unknown"))
    suffix_impl = suffix.get("implementation", suffix.get("resolved_backend", "unknown"))
    return f"ref={reflection_impl}, dif={diffraction_impl}, suffix={suffix_impl}"


def print_summary(payload: dict[str, Any]) -> None:
    runtime_environment = payload.get("runtime_environment", {})
    print(
        f"Multipath benchmark: workload={payload.get('workload', 'full_field')}, "
        f"grid={payload['grid_size']}, n_rays={payload['n_rays']}, "
        f"repeats={payload['repeats']}, warmup_runs={payload['warmup_runs']}"
    )
    if runtime_environment:
        print(
            "Runtime: "
            f"module={runtime_environment.get('channel_module_file', 'n/a')} "
            f"variant={runtime_environment.get('backend_variant', 'n/a')} "
            f"native={runtime_environment.get('native_extension_available', 'n/a')} "
            f"cuda_runtime_version={runtime_environment.get('cuda_runtime_version', 'n/a')}"
        )
    for run in payload["runs"]:
        print(
            f"[{run['parameter']}][{run['mode']}][repeat={run['repeat_index']}] "
            f"setup={run['setup_seconds']:.3f}s trace={run['trace_seconds']:.3f}s "
            f"loss={run['loss_seconds']:.3f}s diff={run['diff_seconds']:.3f}s "
            f"total={run['total_seconds']:.3f}s"
        )
        print(f"  trace_internal: {_format_trace_timing(run['trace_timing'])}")
        print(f"  backends: {_runtime_backend_text(run['runtime_backends'])}")
        if run["performance_timing"]:
            print(f"  performance_timing: {json.dumps(run['performance_timing'], ensure_ascii=False, sort_keys=True)}")
        print(f"  result: {json.dumps(run['result'], ensure_ascii=False, sort_keys=True)}")
        print(f"  setup_mem: {_phase_peak_text(run['phase_metrics']['setup'])}")
        print(f"  trace_mem: {_phase_peak_text(run['phase_metrics']['trace'])}")
        print(f"  loss_mem: {_phase_peak_text(run['phase_metrics']['loss'])}")
        print(f"  {run['mode']}_mem: {_phase_peak_text(run['phase_metrics'][run['mode']])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=["tx_x", "cube1_x"],
        choices=("tx_x", "cube1_x"),
        help="Parameters to benchmark.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vjp", "jvp"],
        choices=("vjp", "jvp"),
        help="AD modes to benchmark.",
    )
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--n-rays", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument(
        "--workload",
        choices=("scalar_loss", "full_field"),
        default="scalar_loss",
        help="Benchmark either the total-field-only scalar-loss workload or the historical full-field collection path.",
    )
    parser.add_argument("--verbose-trace", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full benchmark payload as JSON.",
    )
    args = parser.parse_args()

    payload = run_benchmark(
        parameters=list(args.parameters),
        modes=list(args.modes),
        grid_size=args.grid_size,
        n_rays=args.n_rays,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        workload=args.workload,
        verbose_trace=args.verbose_trace,
    )
    print_summary(payload)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved JSON: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
