"""Benchmark the current RadioMapMonitor baseline workloads."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Mapping
try:
    from ._benchmark_runtime import benchmark_environment_report
    from ._multipath_scaling_common import flush_gpu_caches
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import benchmark_environment_report
    from _multipath_scaling_common import flush_gpu_caches

import drjit as dr
import numpy as np
import torch

import witwin as wt
from tests._scene_helpers import box_geometry, build_scene
from witwin.channel import RadioMapMonitor, Tracer, native_extension_available
from witwin.channel.monitors.radio_map.deterministic.trace import trace_radio_map_monitor
def _sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def _timed_repeat(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> tuple[Any, dict[str, Any]]:
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


def _benchmark_scene():
    return build_scene(
        box_geometry(center=(0.0, 0.0, -2.0), size=0.5),
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )


def _make_tracer(
    *,
    reflection_n_rays: int,
    reflection_max_bounces: int,
    max_diffractions: int,
) -> Tracer:
    return Tracer(
        frequency=1.0e9,
        scene=_benchmark_scene(),
        reflection_n_rays=reflection_n_rays,
        reflection_max_bounces=reflection_max_bounces,
        max_diffractions=max_diffractions,
    )


def _value_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in values.shape],
        "min": float(np.min(values)) if values.size > 0 else 0.0,
        "max": float(np.max(values)) if values.size > 0 else 0.0,
        "mean": float(np.mean(values)) if values.size > 0 else 0.0,
        "finite": bool(np.isfinite(values).all()),
    }


def _summarize_radio_map(payload) -> dict[str, Any]:
    values = payload.metric_tensor().detach().cpu().numpy()
    metadata = dict(payload.metadata)
    return {
        "metric": str(payload.metric),
        "combine_mode": str(payload.combine_mode),
        "receiver_model": str(payload.receiver_model),
        "tx_stack_execution": dict(metadata.get("tx_stack_execution", {})),
        "grid_shape": [int(value) for value in payload.grid_shape],
        "tensor_shape": [int(value) for value in payload.tensor_shape],
        "cell_size": [float(value) for value in payload.cell_size],
        "surface_mode": str(payload.surface.get("surface_mode", "axis_aligned")),
        "quadrature_mode": str(metadata.get("receiver_sampling", {}).get("quadrature_mode", "center")),
        "samples_per_cell": int(metadata.get("receiver_sampling", {}).get("samples_per_cell", 1)),
        "path_counts": dict(metadata.get("path_counts", {})),
        "noise_power_source": str(metadata.get("noise_power_source", "")),
        "aggregate_tx_labels": [str(label) for label in metadata.get("aggregate_tx_labels", ())],
        "accumulation_backend": dict(metadata.get("accumulation_backend", {})),
        "runtime_backends": {
            str(key): dict(value)
            for key, value in dict(metadata.get("runtime_backends", {})).items()
            if isinstance(value, Mapping)
        },
        "value_stats": _value_stats(values),
    }


def _make_gate(*, gate_id: str, passed: bool, detail: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    return gate_id, {
        "gate_id": str(gate_id),
        "passed": bool(passed),
        "detail": dict(detail),
    }


def _axis_aligned_center_case(*, repeats: int, warmup: int) -> dict[str, Any]:
    flush_gpu_caches()
    tracer = _make_tracer(reflection_n_rays=0, reflection_max_bounces=0, max_diffractions=0)
    monitor = RadioMapMonitor(
        "radio_map_axis_center",
        axis="z",
        position=1.5,
        bounds=((-6.0, 6.0), (-6.0, 6.0)),
        grid_shape=(48, 48),
        metric="path_gain",
    )
    tx_pos = wt.Point3f(0.0, -8.0, 1.5)

    result, timing = _timed_repeat(
        lambda: tracer.trace(tx_pos, monitor=monitor, verbose=False),
        warmup=warmup,
        repeats=repeats,
    )
    payload = result.monitor("radio_map_axis_center")
    values = payload.metric_tensor().detach().cpu().numpy()
    gate_id, gate = _make_gate(
        gate_id="axis_aligned_center_metrics_finite",
        passed=bool(np.isfinite(values).all()),
        detail={"metric": payload.metric, "grid_shape": list(payload.grid_shape)},
    )
    return {
        "scenario": {
            "trace_kind": "trace",
            "reflection_n_rays": 0,
            "reflection_max_bounces": 0,
            "max_diffractions": 0,
        },
        "timing": timing,
        "summary": _summarize_radio_map(payload),
        "gates": {gate_id: gate},
    }


def _multipath_case(*, repeats: int, warmup: int) -> dict[str, Any]:
    flush_gpu_caches()
    tracer = _make_tracer(reflection_n_rays=256, reflection_max_bounces=1, max_diffractions=1)
    monitor = RadioMapMonitor(
        "radio_map_multipath",
        axis="z",
        position=1.5,
        bounds=((-6.0, 6.0), (-6.0, 6.0)),
        grid_shape=(20, 20),
        metric="rss",
        tx_power=2.0,
    )
    tx_pos = wt.Point3f(-5.0, 5.0, 1.5)

    result, timing = _timed_repeat(
        lambda: tracer.trace(tx_pos, monitor=monitor, verbose=False),
        warmup=warmup,
        repeats=repeats,
    )
    payload = result.monitor("radio_map_multipath")
    gate_id, gate = _make_gate(
        gate_id="center_sample_count_matches",
        passed=(len(payload.sample_positions()) == 1),
        detail={"sample_positions": len(payload.sample_positions()), "expected": 1},
    )
    return {
        "scenario": {
            "trace_kind": "trace",
            "reflection_n_rays": 256,
            "reflection_max_bounces": 1,
            "max_diffractions": 1,
        },
        "timing": timing,
        "summary": _summarize_radio_map(payload),
        "gates": {gate_id: gate},
    }


def _multi_tx_sinr_case(*, repeats: int, warmup: int) -> dict[str, Any]:
    flush_gpu_caches()
    tracer = _make_tracer(reflection_n_rays=128, reflection_max_bounces=1, max_diffractions=0)
    monitor = RadioMapMonitor(
        "radio_map_sinr",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-2.0, 2.0)),
        grid_shape=(24, 12),
        metric="sinr",
        tx_power=1.5,
        noise_power=1.0e-9,
    )
    requests = [
        {"tx_pos": wt.Point3f(-2.0, -4.0, 1.5), "tx_label": "left"},
        {"tx_pos": wt.Point3f(2.0, -4.0, 1.5), "tx_label": "right"},
    ]

    results, timing = _timed_repeat(
        lambda: tracer.trace_many(requests, monitor=monitor, verbose=False),
        warmup=warmup,
        repeats=repeats,
    )
    left_payload = results[0].monitor("radio_map_sinr")
    association = np.asarray(left_payload.tx_association(), dtype=np.int32)
    gate_id, gate = _make_gate(
        gate_id="multi_tx_association_labels_present",
        passed=(
            tuple(left_payload.metadata.get("aggregate_tx_labels", ())) == ("left", "right")
            and left_payload.metadata.get("tx_stack_execution", {}).get("mode")
            == "trace_many_streaming_post_aggregation"
            and bool(left_payload.metadata.get("tx_stack_execution", {}).get("rss_stack_materialized", True))
            is False
            and int(association.min()) == 0
            and int(association.max()) == 1
        ),
        detail={
            "aggregate_tx_labels": list(left_payload.metadata.get("aggregate_tx_labels", ())),
            "tx_stack_execution": dict(left_payload.metadata.get("tx_stack_execution", {})),
            "association_min": int(association.min()),
            "association_max": int(association.max()),
        },
    )
    sampled_positions = left_payload.sample_metric_positions(
        8,
        tx_association="left",
        seed=7,
        jitter=True,
    ).detach().cpu().numpy()
    return {
        "scenario": {
            "trace_kind": "trace_many",
            "tx_count": 2,
            "reflection_n_rays": 128,
            "reflection_max_bounces": 1,
            "max_diffractions": 0,
        },
        "timing": timing,
        "summary": _summarize_radio_map(left_payload),
        "sample_metric_positions": {
            "count": int(sampled_positions.shape[0]),
            "association_filter": "left",
            "position_bounds_min": [float(value) for value in sampled_positions.min(axis=0)],
            "position_bounds_max": [float(value) for value in sampled_positions.max(axis=0)],
        },
        "gates": {gate_id: gate},
    }


def _oriented_plane_case(*, repeats: int, warmup: int) -> dict[str, Any]:
    flush_gpu_caches()
    tracer = _make_tracer(reflection_n_rays=0, reflection_max_bounces=0, max_diffractions=0)
    monitor = RadioMapMonitor(
        "radio_map_oriented",
        center=(0.0, 0.0, 1.5),
        orientation=(0.2, 0.1, 0.3),
        size=(6.0, 4.0),
        grid_shape=(18, 12),
        metric="path_gain",
        receiver_model="projected_polarized",
    )
    tx_pos = wt.Point3f(-2.0, -6.0, 1.5)

    result, timing = _timed_repeat(
        lambda: tracer.trace(tx_pos, monitor=monitor, verbose=False),
        warmup=warmup,
        repeats=repeats,
    )
    payload = result.monitor("radio_map_oriented")
    positions = payload.sample_metric_positions(16, seed=11, jitter=True)
    center = torch.tensor(payload.surface["center"], dtype=torch.float32, device=positions.device)
    normal = torch.tensor(payload.surface["normal"], dtype=torch.float32, device=positions.device)
    signed_distance = torch.sum((positions - center.unsqueeze(0)) * normal.unsqueeze(0), dim=1)
    max_abs_signed_distance = float(torch.max(torch.abs(signed_distance)).item())
    gate_id, gate = _make_gate(
        gate_id="oriented_samples_stay_on_plane",
        passed=(max_abs_signed_distance <= 1.0e-5),
        detail={"max_abs_signed_distance": max_abs_signed_distance},
    )
    return {
        "scenario": {
            "trace_kind": "trace",
            "surface_mode": "oriented",
            "reflection_n_rays": 0,
            "reflection_max_bounces": 0,
            "max_diffractions": 0,
        },
        "timing": timing,
        "summary": _summarize_radio_map(payload),
        "gates": {gate_id: gate},
    }


def _native_coherent_wall_case(*, repeats: int, warmup: int) -> dict[str, Any]:
    flush_gpu_caches()
    tracer = _make_tracer(reflection_n_rays=4096, reflection_max_bounces=1, max_diffractions=1)
    monitor = RadioMapMonitor(
        "radio_map_native_coherent",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        metric="path_gain",
        combine_mode="coherent",
        receiver_model="projected_polarized",
        ray_mode="3d",
    )
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)

    result, timing = _timed_repeat(
        lambda: tracer.trace(tx_pos, monitor=monitor, verbose=False),
        warmup=warmup,
        repeats=repeats,
    )
    payload = result.monitor("radio_map_native_coherent")
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_coherent",
    )
    baseline_payload = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="baseline",
    )
    baseline_path_gain = np.asarray(
        baseline_payload["metrics"]["path_gain"],
        dtype=np.float32,
    ).reshape(payload.tensor_shape)
    native_path_gain = payload.metric_tensor("path_gain").detach().cpu().numpy()
    max_abs_diff = float(np.max(np.abs(native_path_gain - baseline_path_gain)))
    runtime_backends = dict(payload.metadata.get("runtime_backends", {}))
    reflection_power = np.asarray(payload.coherent_power["reflection"], dtype=np.float32)
    diffraction_power = np.asarray(payload.coherent_power["diffraction"], dtype=np.float32)
    gates = dict(
        [
            _make_gate(
                gate_id="native_coherent_backend_selected",
                passed=(
                    payload.metadata.get("accumulation_backend", {}).get("resolved")
                    == "native_coherent"
                ),
                detail=dict(payload.metadata.get("accumulation_backend", {})),
            ),
            _make_gate(
                gate_id="native_coherent_reflection_and_diffraction_present",
                passed=(
                    float(np.max(reflection_power)) > 0.0
                    and float(np.max(diffraction_power)) > 0.0
                ),
                detail={
                    "path_counts": dict(payload.metadata.get("path_counts", {})),
                    "reflection_power_max": float(np.max(reflection_power)),
                    "diffraction_power_max": float(np.max(diffraction_power)),
                },
            ),
            _make_gate(
                gate_id="native_coherent_matches_baseline",
                passed=bool(
                    np.allclose(
                        native_path_gain,
                        baseline_path_gain,
                        rtol=1.0e-3,
                        atol=5.0e-6,
                    )
                ),
                detail={"max_abs_diff": max_abs_diff},
            ),
            _make_gate(
                gate_id="native_runtime_backends_reported",
                passed=(
                    runtime_backends.get("reflection", {}).get("requested_backend") == "native"
                    and runtime_backends.get("diffraction", {}).get("implementation")
                    == "native_cuda_custom_op"
                    and runtime_backends.get("suffix", {}).get("implementation")
                    == "native_cuda_custom_op"
                ),
                detail={
                    "reflection": dict(runtime_backends.get("reflection", {})),
                    "diffraction": dict(runtime_backends.get("diffraction", {})),
                    "suffix": dict(runtime_backends.get("suffix", {})),
                },
            ),
        ]
    )
    return {
        "scenario": {
            "trace_kind": "trace",
            "reflection_n_rays": 4096,
            "reflection_max_bounces": 1,
            "max_diffractions": 1,
            "combine_mode": "coherent",
            "quadrature_mode": "center",
        },
        "timing": timing,
        "summary": _summarize_radio_map(payload),
        "baseline_parity": {"max_abs_diff": max_abs_diff},
        "gates": gates,
    }


def _baseline_incoherent_wall_case(*, repeats: int, warmup: int) -> dict[str, Any]:
    flush_gpu_caches()
    tracer = _make_tracer(reflection_n_rays=4096, reflection_max_bounces=1, max_diffractions=1)
    monitor = RadioMapMonitor(
        "radio_map_baseline_incoherent_wall",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        metric="path_gain",
        combine_mode="incoherent",
        receiver_model="projected_polarized",
        ray_mode="3d",
    )
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)

    result, timing = _timed_repeat(
        lambda: tracer.trace(tx_pos, monitor=monitor, verbose=False),
        warmup=warmup,
        repeats=repeats,
    )
    payload = result.monitor("radio_map_baseline_incoherent_wall")
    reflection_power = np.asarray(payload.incoherent["reflection"], dtype=np.float32)
    diffraction_power = np.asarray(payload.incoherent["diffraction"], dtype=np.float32)
    gates = dict(
        [
            _make_gate(
                gate_id="baseline_incoherent_reflection_and_diffraction_present",
                passed=(
                    float(np.max(reflection_power)) > 0.0
                    and float(np.max(diffraction_power)) > 0.0
                ),
                detail={
                    "path_counts": dict(payload.metadata.get("path_counts", {})),
                    "reflection_power_max": float(np.max(reflection_power)),
                    "diffraction_power_max": float(np.max(diffraction_power)),
                },
            ),
        ]
    )
    return {
        "scenario": {
            "trace_kind": "trace",
            "reflection_n_rays": 4096,
            "reflection_max_bounces": 1,
            "max_diffractions": 1,
            "combine_mode": "incoherent",
            "quadrature_mode": "center",
            "accumulation_backend": "baseline",
        },
        "timing": timing,
        "summary": _summarize_radio_map(payload),
        "gates": gates,
    }


def _large_incoherent_wall_case(
    *,
    repeats: int,
    warmup: int,
    accumulation_backend: str,
    grid_shape: tuple[int, int],
) -> dict[str, Any]:
    flush_gpu_caches()
    tracer = _make_tracer(reflection_n_rays=4096, reflection_max_bounces=1, max_diffractions=1)
    name = f"radio_map_{accumulation_backend}_{grid_shape[0]}_center"
    monitor = RadioMapMonitor(
        name,
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=grid_shape,
        metric="path_gain",
        combine_mode="incoherent",
        receiver_model="projected_polarized",
        accumulation_backend=accumulation_backend,
        ray_mode="3d",
    )
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)

    result, timing = _timed_repeat(
        lambda: tracer.trace(tx_pos, monitor=monitor, verbose=False),
        warmup=warmup,
        repeats=repeats,
    )
    payload = result.monitor(name)
    summary = _summarize_radio_map(payload)
    reflection_power = np.asarray(payload.incoherent["reflection"], dtype=np.float32)
    diffraction_power = np.asarray(payload.incoherent["diffraction"], dtype=np.float32)
    gates = dict(
        [
            _make_gate(
                gate_id=f"{accumulation_backend}_{grid_shape[0]}_center_reflection_and_diffraction_present",
                passed=(
                    float(np.max(reflection_power)) > 0.0
                    and float(np.max(diffraction_power)) > 0.0
                ),
                detail={
                    "path_counts": dict(payload.metadata.get("path_counts", {})),
                    "reflection_power_max": float(np.max(reflection_power)),
                    "diffraction_power_max": float(np.max(diffraction_power)),
                },
            ),
        ]
    )
    return {
        "scenario": {
            "trace_kind": "trace",
            "reflection_n_rays": 4096,
            "reflection_max_bounces": 1,
            "max_diffractions": 1,
            "combine_mode": "incoherent",
            "quadrature_mode": "center",
            "grid_shape": list(grid_shape),
            "accumulation_backend": accumulation_backend,
        },
        "timing": timing,
        "summary": summary,
        "gates": gates,
    }


def _cell_accumulation_wall_case(*, repeats: int, warmup: int) -> dict[str, Any]:
    flush_gpu_caches()
    tracer = _make_tracer(reflection_n_rays=4096, reflection_max_bounces=1, max_diffractions=1)
    monitor = RadioMapMonitor(
        "radio_map_cell_accumulation",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        metric="path_gain",
        combine_mode="incoherent",
        receiver_model="projected_polarized",
        accumulation_backend="cell_accumulation",
        ray_mode="3d",
    )
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)

    result, timing = _timed_repeat(
        lambda: tracer.trace(tx_pos, monitor=monitor, verbose=False),
        warmup=warmup,
        repeats=repeats,
    )
    payload = result.monitor("radio_map_cell_accumulation")
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_incoherent",
    )
    baseline_payload = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="baseline",
    )
    baseline_path_gain = np.asarray(
        baseline_payload["metrics"]["path_gain"],
        dtype=np.float32,
    ).reshape(payload.tensor_shape)
    native_path_gain = payload.metric_tensor("path_gain").detach().cpu().numpy()
    max_abs_diff = float(np.max(np.abs(native_path_gain - baseline_path_gain)))
    runtime_backends = dict(payload.metadata.get("runtime_backends", {}))
    reflection_power = np.asarray(payload.incoherent["reflection"], dtype=np.float32)
    diffraction_power = np.asarray(payload.incoherent["diffraction"], dtype=np.float32)
    gates = dict(
        [
            _make_gate(
                gate_id="cell_accumulation_backend_selected",
                passed=(
                    payload.metadata.get("accumulation_backend", {}).get("resolved")
                    == "cell_accumulation"
                ),
                detail=dict(payload.metadata.get("accumulation_backend", {})),
            ),
            _make_gate(
                gate_id="cell_accumulation_reflection_and_diffraction_present",
                passed=(
                    float(np.max(reflection_power)) > 0.0
                    and float(np.max(diffraction_power)) > 0.0
                ),
                detail={
                    "path_counts": dict(payload.metadata.get("path_counts", {})),
                    "reflection_power_max": float(np.max(reflection_power)),
                    "diffraction_power_max": float(np.max(diffraction_power)),
                },
            ),
            _make_gate(
                gate_id="cell_accumulation_matches_baseline",
                passed=bool(
                    np.allclose(
                        native_path_gain,
                        baseline_path_gain,
                        rtol=1.0e-4,
                        atol=2.0e-6,
                    )
                ),
                detail={"max_abs_diff": max_abs_diff},
            ),
            _make_gate(
                gate_id="cell_accumulation_runtime_backends_reported",
                passed=(
                    runtime_backends.get("reflection", {}).get("radio_map_scalar_power_backend")
                    == "direct_in_loop_cell_scatter"
                    and runtime_backends.get("reflection", {}).get("pair_replay_backend")
                    == "direct_replay_scalar_power"
                    and runtime_backends.get("diffraction", {}).get("radio_map_scalar_power_backend")
                    == "direct_in_loop_cell_scatter"
                    and runtime_backends.get("diffraction", {}).get("pair_replay_backend")
                    == "direct_state_scalar_power"
                    and runtime_backends.get("suffix", {}).get("implementation") == "disabled"
                ),
                detail={
                    "reflection": dict(runtime_backends.get("reflection", {})),
                    "diffraction": dict(runtime_backends.get("diffraction", {})),
                    "suffix": dict(runtime_backends.get("suffix", {})),
                },
            ),
        ]
    )
    return {
        "scenario": {
            "trace_kind": "trace",
            "reflection_n_rays": 4096,
            "reflection_max_bounces": 1,
            "max_diffractions": 1,
            "combine_mode": "incoherent",
            "quadrature_mode": "center",
        },
        "timing": timing,
        "summary": _summarize_radio_map(payload),
        "baseline_parity": {"max_abs_diff": max_abs_diff},
        "gates": gates,
    }


def _run_suite(
    *,
    repeats: int,
    warmup: int,
    include_large_wall_512: bool = False,
) -> dict[str, Any]:
    cases = {
        "axis_aligned_center_baseline": _axis_aligned_center_case(repeats=repeats, warmup=warmup),
        "multipath": _multipath_case(repeats=repeats, warmup=warmup),
        "multi_tx_sinr": _multi_tx_sinr_case(repeats=repeats, warmup=warmup),
        "oriented_plane": _oriented_plane_case(repeats=repeats, warmup=warmup),
        "baseline_incoherent_wall": _baseline_incoherent_wall_case(
            repeats=repeats,
            warmup=warmup,
        ),
        "cell_accumulation_wall": _cell_accumulation_wall_case(repeats=repeats, warmup=warmup),
    }
    if native_extension_available():
        cases["native_coherent_wall"] = _native_coherent_wall_case(
            repeats=repeats,
            warmup=warmup,
        )
    if include_large_wall_512:
        cases["baseline_incoherent_wall_512_center"] = _large_incoherent_wall_case(
            repeats=repeats,
            warmup=warmup,
            accumulation_backend="baseline",
            grid_shape=(512, 512),
        )
        cases["cell_accumulation_wall_512_center"] = _large_incoherent_wall_case(
            repeats=repeats,
            warmup=warmup,
            accumulation_backend="cell_accumulation",
            grid_shape=(512, 512),
        )
    gates = {}
    for case_name, payload in cases.items():
        for gate_id, gate in payload.get("gates", {}).items():
            gates[f"{case_name}:{gate_id}"] = dict(gate)
    if include_large_wall_512:
        baseline_case = cases["baseline_incoherent_wall_512_center"]
        cell_case = cases["cell_accumulation_wall_512_center"]
        baseline_ms = float(baseline_case["timing"]["median_ms"])
        cell_ms = float(cell_case["timing"]["median_ms"])
        timing_ratio = float(cell_ms / baseline_ms) if baseline_ms > 0.0 else float("inf")
        gates["large_wall_512_center:cell_accumulation_not_slower_than_25pct"] = {
            "gate_id": "large_wall_512_center:cell_accumulation_not_slower_than_25pct",
            "passed": bool(timing_ratio <= 1.25),
            "detail": {
                "baseline_median_ms": baseline_ms,
                "cell_accumulation_median_ms": cell_ms,
                "timing_ratio": timing_ratio,
            },
        }
        baseline_stats = baseline_case["summary"]["value_stats"]
        cell_stats = cell_case["summary"]["value_stats"]
        gates["large_wall_512_center:cell_accumulation_value_stats_finite"] = {
            "gate_id": "large_wall_512_center:cell_accumulation_value_stats_finite",
            "passed": bool(baseline_stats["finite"] and cell_stats["finite"]),
            "detail": {
                "baseline": dict(baseline_stats),
                "cell_accumulation": dict(cell_stats),
            },
        }
    return {
        "environment": benchmark_environment_report(),
        "benchmarks": cases,
        "gates": gates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions per case.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repetitions per case.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="Exit nonzero if any benchmark correctness gate fails.",
    )
    parser.add_argument(
        "--include-large-wall-512",
        action="store_true",
        help="Also run opt-in 512x512 center-sampled baseline-vs-native incoherent wall comparisons.",
    )
    args = parser.parse_args(argv)

    report = _run_suite(
        repeats=args.repeats,
        warmup=args.warmup,
        include_large_wall_512=bool(args.include_large_wall_512),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for case_name, case_payload in report["benchmarks"].items():
            timing = case_payload["timing"]
            summary = case_payload["summary"]
            print(
                f"{case_name}: median={timing['median_ms']:.2f} ms "
                f"metric={summary['metric']} surface={summary['surface_mode']} "
                f"grid={tuple(summary['grid_shape'])}"
            )

    if args.strict_gates:
        failed = [gate_id for gate_id, gate in report["gates"].items() if not gate["passed"]]
        if failed:
            print(json.dumps({"failed_gates": failed}, indent=2, sort_keys=True), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

