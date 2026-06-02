"""Run the frozen PathMonitor Phase 6 benchmark matrix and rollout gates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any
try:
    from ._benchmark_runtime import benchmark_environment_report
    from ._multipath_scaling_common import flush_gpu_caches
    from ._path_monitor_phase6 import (
        evaluate_geometry_toggle_correctness,
        evaluate_mixed_monitor_correctness,
        evaluate_ratio_gate,
        evaluate_warm_cache_trace_many_correctness,
        resolve_path_monitor_diffraction_depth_report,
        summarize_field_payload,
        summarize_path_result,
    )
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import benchmark_environment_report
    from _multipath_scaling_common import flush_gpu_caches
    from _path_monitor_phase6 import (
        evaluate_geometry_toggle_correctness,
        evaluate_mixed_monitor_correctness,
        evaluate_ratio_gate,
        evaluate_warm_cache_trace_many_correctness,
        resolve_path_monitor_diffraction_depth_report,
        summarize_field_payload,
        summarize_path_result,
    )

import drjit as dr
import numpy as np
import torch

from tests._scene_helpers import box_geometry, build_scene
from witwin.channel import FieldMonitor, PathMonitor, Tracer
from witwin.channel.validation import (
    build_single_wedge_case,
    build_triple_wedge_case,
)


def _sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def _timed_repeat(fn, *, warmup: int, repeats: int) -> tuple[object, dict[str, Any]]:
    last_value = None
    for _ in range(int(warmup)):
        last_value = fn()
        _sync_gpu()

    samples_ms: list[float] = []
    for _ in range(int(repeats)):
        _sync_gpu()
        t0 = time.perf_counter()
        last_value = fn()
        _sync_gpu()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    if not samples_ms:
        raise ValueError("repeats must be greater than zero.")

    return last_value, {
        "samples_ms": samples_ms,
        "median_ms": float(median(samples_ms)),
        "mean_ms": float(mean(samples_ms)),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
        "warmup": int(warmup),
        "repeats": int(repeats),
    }


def _line_receivers(*, xs, y: float, z: float) -> torch.Tensor:
    return torch.tensor(
        [[float(x), float(y), float(z)] for x in xs],
        dtype=torch.float32,
    )


def _build_geometry_scene():
    return build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )


def _build_mixed_scene():
    return build_scene(
        box_geometry(center=(0.0, 0.5, 1.5), size=(1.5, 1.5, 3.0)),
        box_geometry(center=(-2.0, 2.0, 1.5), size=(1.5, 1.5, 3.0)),
    )


def _build_depth_pair_case(*, repeats: int, warmup: int) -> dict[str, Any]:
    flush_gpu_caches()
    case = build_triple_wedge_case()
    tracer = Tracer(
        frequency=1e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=3,
    )
    tx = torch.tensor(case.tx_pos, dtype=torch.float32)
    rx_positions = _line_receivers(
        xs=(-5.0, -3.0, -1.0, 1.0, 3.0, 5.0),
        y=case.cut_value,
        z=case.calculation_height,
    )

    default_monitor = PathMonitor("rx", positions=rx_positions)
    explicit_monitor = PathMonitor("rx", positions=rx_positions, max_diffractions=3)
    default_depth = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=default_monitor.max_diffractions,
    )
    explicit_depth = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=explicit_monitor.max_diffractions,
    )

    default_result, default_timing = _timed_repeat(
        lambda: tracer.trace(tx, monitor=default_monitor, verbose=False, return_timing=True),
        warmup=warmup,
        repeats=repeats,
    )
    explicit_result, explicit_timing = _timed_repeat(
        lambda: tracer.trace(tx, monitor=explicit_monitor, verbose=False, return_timing=True),
        warmup=warmup,
        repeats=repeats,
    )

    default_paths = default_result.paths("rx")
    explicit_paths = explicit_result.paths("rx")
    default_diff_count = int(default_paths.metadata.get("path_counts", {}).get("diffraction", 0))
    explicit_diff_count = int(explicit_paths.metadata.get("path_counts", {}).get("diffraction", 0))

    correctness = {
        "passed": (
            int(default_depth["effective_max_diffractions"]) == 1
            and int(explicit_depth["effective_max_diffractions"]) == 3
            and explicit_diff_count >= default_diff_count
        ),
        "default_effective_max_diffractions": int(default_depth["effective_max_diffractions"]),
        "explicit_effective_max_diffractions": int(explicit_depth["effective_max_diffractions"]),
        "default_diffraction_path_count": default_diff_count,
        "explicit_diffraction_path_count": explicit_diff_count,
    }
    performance = evaluate_ratio_gate(
        gate_id="first_order_vs_multi_order_bounded",
        measured_label="default_first_order_path_export",
        measured_ms=float(default_timing["median_ms"]),
        baseline_label="explicit_multi_order_path_export",
        baseline_ms=float(explicit_timing["median_ms"]),
        max_ratio=1.05,
        description=(
            "Default first-order PathMonitor export should stay bounded relative to the "
            "explicit multi-order export on the same triple-wedge workload."
        ),
    )
    performance["correctness"] = correctness
    performance["passed"] = bool(performance["passed"] and correctness["passed"])

    return {
        "benchmarks": {
            "default_first_order_path_export": {
                "scenario": {
                    "scene": case.name,
                    "tx_count": 1,
                    "rx_count": int(rx_positions.shape[0]),
                    "reflection_n_rays": 0,
                    "reflection_max_bounces": 0,
                    "return_geometry": False,
                },
                "timing": default_timing,
                "summary": summarize_path_result(default_paths, depth_report=default_depth),
            },
            "explicit_multi_order_path_export": {
                "scenario": {
                    "scene": case.name,
                    "tx_count": 1,
                    "rx_count": int(rx_positions.shape[0]),
                    "reflection_n_rays": 0,
                    "reflection_max_bounces": 0,
                    "return_geometry": False,
                },
                "timing": explicit_timing,
                "summary": summarize_path_result(explicit_paths, depth_report=explicit_depth),
            },
        },
        "gates": {
            performance["gate_id"]: performance,
        },
    }


def _build_geometry_pair_case(*, repeats: int, warmup: int) -> dict[str, Any]:
    flush_gpu_caches()
    scene = _build_geometry_scene()
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        max_diffractions=0,
    )
    tx = torch.tensor((-3.0, -5.0, 1.5), dtype=torch.float32)
    rx_positions = torch.tensor(
        [
            [-3.0, 5.0, 1.5],
            [-2.0, 4.0, 1.5],
        ],
        dtype=torch.float32,
    )

    off_monitor = PathMonitor(
        "rx",
        positions=rx_positions,
        max_diffractions=0,
        return_geometry=False,
    )
    on_monitor = PathMonitor(
        "rx",
        positions=rx_positions,
        max_diffractions=0,
        return_geometry=True,
    )
    off_depth = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=off_monitor.max_diffractions,
    )
    on_depth = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=on_monitor.max_diffractions,
    )

    off_result, off_timing = _timed_repeat(
        lambda: tracer.trace(tx, monitor=off_monitor, verbose=False, return_timing=True),
        warmup=warmup,
        repeats=repeats,
    )
    on_result, on_timing = _timed_repeat(
        lambda: tracer.trace(tx, monitor=on_monitor, verbose=False, return_timing=True),
        warmup=warmup,
        repeats=repeats,
    )

    off_paths = off_result.paths("rx")
    on_paths = on_result.paths("rx")
    correctness = evaluate_geometry_toggle_correctness(off_paths, on_paths)
    performance = evaluate_ratio_gate(
        gate_id="geometry_off_vs_on_bounded",
        measured_label="geometry_off_path_export",
        measured_ms=float(off_timing["median_ms"]),
        baseline_label="geometry_on_path_export",
        baseline_ms=float(on_timing["median_ms"]),
        max_ratio=1.10,
        description=(
            "Geometry-off path export should stay bounded relative to geometry-on export on "
            "the same reflection workload."
        ),
    )
    performance["correctness"] = correctness
    performance["passed"] = bool(performance["passed"] and correctness["passed"])

    return {
        "benchmarks": {
            "geometry_off_path_export": {
                "scenario": {
                    "scene": "reflection_wall",
                    "tx_count": 1,
                    "rx_count": int(rx_positions.shape[0]),
                    "reflection_n_rays": 8192,
                    "reflection_max_bounces": 1,
                    "return_geometry": False,
                },
                "timing": off_timing,
                "summary": summarize_path_result(off_paths, depth_report=off_depth),
            },
            "geometry_on_path_export": {
                "scenario": {
                    "scene": "reflection_wall",
                    "tx_count": 1,
                    "rx_count": int(rx_positions.shape[0]),
                    "reflection_n_rays": 8192,
                    "reflection_max_bounces": 1,
                    "return_geometry": True,
                },
                "timing": on_timing,
                "summary": summarize_path_result(on_paths, depth_report=on_depth),
            },
        },
        "gates": {
            performance["gate_id"]: performance,
        },
    }


def _build_mixed_monitor_case(*, repeats: int, warmup: int) -> dict[str, Any]:
    flush_gpu_caches()
    scene = _build_mixed_scene()
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=1024,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    tx = torch.tensor((0.0, -4.5, 1.5), dtype=torch.float32)
    field_name = "field_xy"
    path_name = "rx"
    field_monitor = FieldMonitor(
        field_name,
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-1.0, 5.0)),
        grid_size=(24, 24),
    )
    path_monitor = PathMonitor(
        path_name,
        positions=_line_receivers(xs=(-2.5, -0.5, 1.5, 3.0), y=3.5, z=1.5),
        max_diffractions=1,
        return_geometry=False,
    )
    depth_report = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=path_monitor.max_diffractions,
    )

    field_only_result, field_only_timing = _timed_repeat(
        lambda: tracer.trace(tx, monitor=field_monitor, verbose=False, return_timing=True),
        warmup=warmup,
        repeats=repeats,
    )
    path_only_result, path_only_timing = _timed_repeat(
        lambda: tracer.trace(tx, monitor=path_monitor, verbose=False, return_timing=True),
        warmup=warmup,
        repeats=repeats,
    )
    mixed_result, mixed_timing = _timed_repeat(
        lambda: tracer.trace(
            tx,
            monitor=(field_monitor, path_monitor),
            verbose=False,
            return_timing=True,
        ),
        warmup=warmup,
        repeats=repeats,
    )

    correctness = evaluate_mixed_monitor_correctness(
        field_only_result,
        path_only_result,
        mixed_result,
        field_name=field_name,
        path_name=path_name,
    )
    performance = evaluate_ratio_gate(
        gate_id="mixed_monitor_vs_separate_bounded",
        measured_label="mixed_field_path_trace",
        measured_ms=float(mixed_timing["median_ms"]),
        baseline_label="field_only_baseline + path_only_baseline",
        baseline_ms=float(field_only_timing["median_ms"] + path_only_timing["median_ms"]),
        max_ratio=1.10,
        description=(
            "A mixed field-plus-path trace should stay bounded relative to the sum of the "
            "standalone field-only and path-only traces on the same scene."
        ),
    )
    performance["correctness"] = correctness
    performance["passed"] = bool(performance["passed"] and correctness["passed"])

    return {
        "benchmarks": {
            "field_only_baseline": {
                "scenario": {
                    "scene": "mixed_boxes",
                    "tx_count": 1,
                    "grid_size": (24, 24),
                    "reflection_n_rays": 1024,
                    "reflection_max_bounces": 1,
                },
                "timing": field_only_timing,
                "summary": summarize_field_payload(field_only_result.monitor(field_name)),
            },
            "path_only_baseline": {
                "scenario": {
                    "scene": "mixed_boxes",
                    "tx_count": 1,
                    "rx_count": 4,
                    "reflection_n_rays": 1024,
                    "reflection_max_bounces": 1,
                    "return_geometry": False,
                },
                "timing": path_only_timing,
                "summary": summarize_path_result(
                    path_only_result.paths(path_name),
                    depth_report=depth_report,
                ),
            },
            "mixed_field_path_trace": {
                "scenario": {
                    "scene": "mixed_boxes",
                    "tx_count": 1,
                    "grid_size": (24, 24),
                    "rx_count": 4,
                    "reflection_n_rays": 1024,
                    "reflection_max_bounces": 1,
                    "return_geometry": False,
                },
                "timing": mixed_timing,
                "field_summary": summarize_field_payload(mixed_result.monitor(field_name)),
                "path_summary": summarize_path_result(
                    mixed_result.paths(path_name),
                    depth_report=depth_report,
                ),
            },
        },
        "gates": {
            performance["gate_id"]: performance,
        },
    }


def _warm_cache_requests(*, frame_idx: int, tx_positions: list[tuple[float, float, float]]) -> list[dict[str, Any]]:
    base_offsets = (
        (-1.5, 2.0),
        (-0.25, 2.4),
        (1.25, 2.8),
        (2.5, 3.2),
    )
    requests = []
    for tx_idx, tx_pos in enumerate(tx_positions):
        rx_positions = torch.tensor(
            [
                [
                    base_x + 0.12 * frame_idx + 0.05 * tx_idx,
                    base_y + 0.08 * frame_idx - 0.03 * tx_idx,
                    1.5,
                ]
                for base_x, base_y in base_offsets
            ],
            dtype=torch.float32,
        )
        requests.append(
            {
                "tx_pos": torch.tensor(tx_pos, dtype=torch.float32),
                "monitor_overrides": {
                    "rx": {
                        "positions": rx_positions,
                    }
                },
            }
        )
    return requests


def _build_warm_cache_trace_many_case(*, frames: int) -> dict[str, Any]:
    flush_gpu_caches()
    case = build_single_wedge_case()
    tracer = Tracer(
        frequency=1e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=2,
    )
    base_monitor = PathMonitor(
        "rx",
        positions=_line_receivers(xs=(-1.5, -0.25, 1.25, 2.5), y=2.0, z=case.calculation_height),
    )
    depth_report = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=base_monitor.max_diffractions,
    )
    tx_positions = [
        tuple(float(value) for value in case.tx_pos),
        (1.25, -6.0, case.calculation_height),
    ]

    compile_warmup_requests = _warm_cache_requests(frame_idx=0, tx_positions=tx_positions)
    tracer.trace_many(
        compile_warmup_requests,
        monitor=base_monitor,
        verbose=False,
        return_timing=True,
    )
    _sync_gpu()
    tracer._clear_trace_caches()

    frame_payloads = []
    for frame_idx in range(int(frames)):
        requests = _warm_cache_requests(frame_idx=frame_idx, tx_positions=tx_positions)
        _sync_gpu()
        t0 = time.perf_counter()
        results = tracer.trace_many(
            requests,
            monitor=base_monitor,
            verbose=False,
            return_timing=True,
        )
        _sync_gpu()
        frame_total_ms = (time.perf_counter() - t0) * 1000.0
        path_summaries = []
        cache_hits = []
        cache_modes = []
        for result in results:
            paths = result.paths("rx")
            path_summaries.append(summarize_path_result(paths, depth_report=depth_report))
            cache_report = (
                dict(paths.metadata.get("runtime_reuse", {}))
                .get("diffraction_state_prep_cache", {})
            )
            cache_hits.append(int(cache_report.get("hits", 0)))
            cache_modes.append(str(cache_report.get("mode", "")))
        frame_payloads.append(
            {
                "frame_index": int(frame_idx),
                "requests": requests,
                "results": results,
                "frame_total_ms": float(frame_total_ms),
                "per_request_ms": float(frame_total_ms / len(results)),
                "path_summaries": path_summaries,
                "cache_hits": cache_hits,
                "cache_modes": cache_modes,
            }
        )

    cold_per_request_ms = float(frame_payloads[0]["per_request_ms"])
    warm_per_request_samples = [float(item["per_request_ms"]) for item in frame_payloads[1:]]
    warm_median_ms = float(median(warm_per_request_samples)) if warm_per_request_samples else cold_per_request_ms
    correctness = evaluate_warm_cache_trace_many_correctness(
        results=frame_payloads[-1]["results"],
        requests=frame_payloads[-1]["requests"],
        path_name="rx",
    )
    performance = evaluate_ratio_gate(
        gate_id="warm_cache_trace_many_reuse",
        measured_label="warm_cache_trace_many.steady_state_per_request",
        measured_ms=warm_median_ms,
        baseline_label="warm_cache_trace_many.cold_per_request",
        baseline_ms=cold_per_request_ms,
        max_ratio=1.00,
        description=(
            "Steady-state multi-TX trace_many path export with tx/rx overrides should stay "
            "bounded relative to the cold frame once persistent PathMonitor diffraction "
            "state-prep reuse is active."
        ),
    )
    performance["correctness"] = correctness
    performance["passed"] = bool(performance["passed"] and correctness["passed"])

    return {
        "benchmarks": {
            "warm_cache_trace_many": {
                "scenario": {
                    "scene": case.name,
                    "tx_count": len(tx_positions),
                    "rx_count": int(base_monitor.num_rx),
                    "frames": int(frames),
                    "trace_many": True,
                    "tx_rx_override": True,
                    "return_geometry": False,
                },
                "timing": {
                    "frame_total_ms": [float(item["frame_total_ms"]) for item in frame_payloads],
                    "per_request_ms": [float(item["per_request_ms"]) for item in frame_payloads],
                    "cold_per_request_ms": cold_per_request_ms,
                    "warm_steady_state_median_ms": warm_median_ms,
                },
                "summary": {
                    "depth_report": dict(depth_report),
                    "cache_hits_per_frame": [list(item["cache_hits"]) for item in frame_payloads],
                    "cache_modes_per_frame": [list(item["cache_modes"]) for item in frame_payloads],
                    "last_frame_path_summaries": frame_payloads[-1]["path_summaries"],
                },
            },
        },
        "gates": {
            performance["gate_id"]: performance,
        },
    }


def build_phase6_benchmark_payload(*, repeats: int, warmup: int, frames: int) -> dict[str, Any]:
    payload = {
        "benchmark": "path_monitor_phase6",
        "runtime_environment": benchmark_environment_report(),
        "matrix_config": {
            "repeats": int(repeats),
            "warmup": int(warmup),
            "warm_cache_frames": int(frames),
        },
        "benchmarks": {},
        "gates": {},
    }
    for case_builder in (
        lambda: _build_depth_pair_case(repeats=repeats, warmup=warmup),
        lambda: _build_geometry_pair_case(repeats=repeats, warmup=warmup),
        lambda: _build_mixed_monitor_case(repeats=repeats, warmup=warmup),
        lambda: _build_warm_cache_trace_many_case(frames=frames),
    ):
        case_payload = case_builder()
        payload["benchmarks"].update(case_payload["benchmarks"])
        payload["gates"].update(case_payload["gates"])
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    runtime = payload["runtime_environment"]
    print(
        f"Runtime: module={runtime.get('channel_module_file', 'n/a')} "
        f"native={runtime.get('native_extension_available', 'n/a')} "
        f"cuda_runtime_version={runtime.get('cuda_runtime_version', 'n/a')}"
    )
    for benchmark_id, benchmark in payload["benchmarks"].items():
        timing = benchmark["timing"]
        if "median_ms" in timing:
            summary = benchmark.get("summary") or benchmark.get("path_summary") or {}
            depth_report = summary.get("depth_report")
            depth_text = ""
            if isinstance(depth_report, dict):
                depth_text = (
                    f" requested={depth_report['monitor_requested_max_diffractions']} "
                    f"effective={depth_report['effective_max_diffractions']}"
                )
            print(
                f"[{benchmark_id}] median={timing['median_ms']:.2f} ms "
                f"mean={timing['mean_ms']:.2f} ms{depth_text}"
            )
        else:
            print(
                f"[{benchmark_id}] cold_per_request={timing['cold_per_request_ms']:.2f} ms "
                f"warm_median={timing['warm_steady_state_median_ms']:.2f} ms"
            )

    for gate_id, gate in payload["gates"].items():
        status = "PASS" if gate["passed"] else "FAIL"
        ratio = gate.get("observed_ratio")
        ratio_text = "n/a" if ratio is None else f"{float(ratio):.3f}"
        correctness = gate.get("correctness", {})
        correctness_text = "ok" if correctness.get("passed") else ",".join(correctness.get("errors", ())) or "failed"
        print(
            f"[gate:{gate_id}] {status} ratio={ratio_text} "
            f"correctness={correctness_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repeats for the steady-state benchmark cases.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations for the steady-state benchmark cases.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=6,
        help="Number of frames for the warm-cache trace_many benchmark case.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full Phase 6 benchmark payload as JSON.",
    )
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="Exit with status 1 if any rollout gate fails.",
    )
    args = parser.parse_args()

    payload = build_phase6_benchmark_payload(
        repeats=args.repeats,
        warmup=args.warmup,
        frames=args.frames,
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_summary(payload)

    if args.strict_gates and not all(gate["passed"] for gate in payload["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
