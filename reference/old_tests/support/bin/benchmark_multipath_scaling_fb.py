"""Stress-test multipath scaling with scalar-loss forward and backward passes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
try:
    from ._benchmark_runtime import assert_native_benchmark_support
    from ._multipath_scaling_cases import DEFAULT_BASELINE, DEFAULT_SWEEPS
    from ._multipath_scaling_common import (
        benchmark_environment_report,
        benchmark_timing_mode,
        build_loss,
        create_case,
        float_scalar,
        flush_gpu_caches,
        measure_phase,
        memory_snapshot,
        print_single_run_summary,
        trace_case,
    )
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import assert_native_benchmark_support
    from _multipath_scaling_cases import DEFAULT_BASELINE, DEFAULT_SWEEPS
    from _multipath_scaling_common import (
        benchmark_environment_report,
        benchmark_timing_mode,
        build_loss,
        create_case,
        float_scalar,
        flush_gpu_caches,
        measure_phase,
        memory_snapshot,
        print_single_run_summary,
        trace_case,
    )

import drjit as dr


def _run_backward(case: dict[str, object], loss_payload: dict[str, object]) -> dict[str, float]:
    dr.backward(loss_payload["loss"])
    return {
        "parameter_grad": float_scalar(dr.grad(case["variable"])),
        "loss_grad": float_scalar(dr.grad(loss_payload["loss"])),
    }


def run_single_pass(*, grid_size: int, n_rays: int, motif_repeats: int) -> dict[str, object]:
    flush_gpu_caches()

    case, setup_phase = measure_phase(
        "setup",
        lambda: create_case(grid_size=grid_size, n_rays=n_rays, motif_repeats=motif_repeats),
    )
    assert_native_benchmark_support(
        benchmark_name="multipath_scaling_stress_fb",
        reflection_field_backend=case["tracer"].config.trace.reflection_field_backend,
        diffraction_execution=case["tracer"].config.trace.diffraction_execution,
    )
    with benchmark_timing_mode(sync_timing=False):
        trace_payload, trace_phase = measure_phase("trace", lambda: trace_case(case))
        loss_payload, loss_phase = measure_phase("loss", lambda: build_loss(trace_payload))
        backward_payload, backward_phase = measure_phase("backward", lambda: _run_backward(case, loss_payload))
    measured_memory_final = memory_snapshot()

    flush_gpu_caches()
    synced_case = create_case(grid_size=grid_size, n_rays=n_rays, motif_repeats=motif_repeats)
    with benchmark_timing_mode(sync_timing=True):
        synced_trace_payload = trace_case(synced_case)

    return {
        "grid_size": int(grid_size),
        "n_receivers": int(grid_size) * int(grid_size),
        "n_rays": int(n_rays),
        "motif_repeats": int(motif_repeats),
        "runtime_environment": benchmark_environment_report(),
        "scene_metrics": case["scene_metrics"],
        "setup_seconds": float(setup_phase["seconds"]),
        "trace_seconds": float(trace_phase["seconds"]),
        "loss_seconds": float(loss_phase["seconds"]),
        "backward_seconds": float(backward_phase["seconds"]),
        "total_seconds": float(
            setup_phase["seconds"] + trace_phase["seconds"] + loss_phase["seconds"] + backward_phase["seconds"]
        ),
        "trace_timing": synced_trace_payload["trace_timing"],
        "trace_timing_sum_seconds": float(synced_trace_payload["trace_timing_sum_seconds"]),
        "runtime_backends": synced_trace_payload["runtime_backends"],
        "performance_timing": synced_trace_payload["performance_timing"],
        "timing_measurement_mode": "synced_sidecar_trace",
        "trace_timing_unsynced": trace_payload["trace_timing"],
        "trace_timing_sum_seconds_unsynced": float(trace_payload["trace_timing_sum_seconds"]),
        "loss": float(loss_payload["loss_value"]),
        "backward_result": backward_payload,
        "phase_metrics": {
            "setup": setup_phase,
            "trace": trace_phase,
            "loss": loss_phase,
            "backward": backward_phase,
        },
        "memory_final": measured_memory_final,
    }


def _run_sweep(name: str, configs: list[dict[str, int]], *, warmup_runs: int) -> dict[str, object]:
    runs = []
    for index, config in enumerate(configs):
        config_text = (
            f"grid={config['grid_size']} rays={config['n_rays']} motifs={config['motif_repeats']}"
        )
        print(f"[{name}][{index + 1}/{len(configs)}] {config_text}")
        for _ in range(max(0, warmup_runs)):
            run_single_pass(**config)
        result = run_single_pass(**config)
        result["sweep_name"] = name
        runs.append(result)
        trace_peak = result["phase_metrics"]["trace"]["memory_after"]["drjit_allocator"].get("device_peak")
        backward_peak = result["phase_metrics"]["backward"]["memory_after"]["drjit_allocator"].get("device_peak")
        print(
            f"  trace={result['trace_seconds']:.3f}s backward={result['backward_seconds']:.3f}s "
            f"trace_peak={trace_peak} backward_peak={backward_peak}"
        )
    return {"name": name, "runs": runs}


def run_scaling_benchmark(*, warmup_runs: int) -> dict[str, object]:
    results = {
        "benchmark": "multipath_scaling_stress_fb",
        "date": time.strftime("%Y-%m-%d"),
        "runtime_environment": benchmark_environment_report(),
        "parameter": "tx_x",
        "workload": "scalar_loss_forward_backward_stress",
        "baseline": dict(DEFAULT_BASELINE),
        "warmup_runs": int(max(0, warmup_runs)),
        "sweeps": {},
    }
    for name, configs in DEFAULT_SWEEPS.items():
        results["sweeps"][name] = _run_sweep(name, configs, warmup_runs=warmup_runs)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--n-rays", type=int, default=10000)
    parser.add_argument("--motif-repeats", type=int, default=1)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Warmup passes before the measured pass.",
    )
    parser.add_argument(
        "--single-process-sweep",
        action="store_true",
        help="Legacy mode: run the entire sweep inside one process. This can retain allocator state between configs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full benchmark payload as JSON.",
    )
    args = parser.parse_args()

    if args.single_process_sweep:
        payload = run_scaling_benchmark(warmup_runs=args.warmup_runs)
    else:
        for _ in range(max(0, args.warmup_runs)):
            run_single_pass(
                grid_size=args.grid_size,
                n_rays=args.n_rays,
                motif_repeats=args.motif_repeats,
            )
        payload = run_single_pass(
            grid_size=args.grid_size,
            n_rays=args.n_rays,
            motif_repeats=args.motif_repeats,
        )
        print_single_run_summary(payload, diff_phase_name="backward")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved JSON: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
