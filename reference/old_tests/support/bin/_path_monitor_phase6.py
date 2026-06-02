"""Shared PathMonitor Phase 6 benchmark reporting and gate helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from witwin.channel.monitors.orchestration import resolve_solver_controls

PHASE6_BENCHMARK_CASE_IDS = (
    "default_first_order_path_export",
    "explicit_multi_order_path_export",
    "geometry_off_path_export",
    "geometry_on_path_export",
    "field_only_baseline",
    "path_only_baseline",
    "mixed_field_path_trace",
    "warm_cache_trace_many",
)

PHASE6_GATE_IDS = (
    "first_order_vs_multi_order_bounded",
    "geometry_off_vs_on_bounded",
    "mixed_monitor_vs_separate_bounded",
    "warm_cache_trace_many_reuse",
)


def _to_float_tuple(tx_pos) -> tuple[float, float, float]:
    if hasattr(tx_pos, "tolist"):
        values = tx_pos.tolist()
    else:
        values = tx_pos
    return tuple(float(value) for value in values)


def _path_complex_array(paths) -> np.ndarray:
    return np.asarray(paths.a, dtype=np.complex64)


def _path_tau_array(paths) -> np.ndarray:
    return np.asarray(paths.tau, dtype=np.float32)


def _path_valid_array(paths) -> np.ndarray:
    return np.asarray(paths.valid, dtype=np.bool_)


def _path_num_paths_array(paths) -> np.ndarray:
    return np.asarray(paths.num_paths, dtype=np.int32)


def _path_types_array(paths) -> np.ndarray:
    return np.asarray(paths.types, dtype=np.int32)


def _path_rx_positions_array(paths) -> np.ndarray:
    return np.asarray(paths.rx_positions, dtype=np.float32)


def _field_total_complex_array(payload) -> np.ndarray:
    total = payload.field.total
    return np.asarray(total.real, dtype=np.float32) + 1j * np.asarray(total.imag, dtype=np.float32)


def _resolve_path_payload(result, name: str):
    if isinstance(result, Mapping):
        return result[name]
    return result


def _resolve_field_payload(result, name: str):
    if isinstance(result, Mapping):
        return result[name]
    return result


def resolve_path_monitor_diffraction_depth_report(
    trace_config,
    *,
    requested_max_diffractions: int | None,
) -> dict[str, Any]:
    solver_controls = resolve_solver_controls(
        trace_config,
        execution_intent="path_export",
        max_diffractions_override=requested_max_diffractions,
    )
    effective = dict(solver_controls["effective"])
    requested = None if requested_max_diffractions is None else int(requested_max_diffractions)
    return {
        "monitor_requested_max_diffractions": requested,
        "trace_default_max_diffractions": int(trace_config.max_diffractions),
        "effective_max_diffractions": int(effective["max_diffractions"]),
        "solver_mode": str(solver_controls["selected"]),
        "memory_profile": str(effective["memory_profile"]),
        "execution_intent": dict(solver_controls["execution_intent"]),
    }


def format_path_monitor_diffraction_depth_report(report: Mapping[str, Any]) -> str:
    requested = report.get("monitor_requested_max_diffractions")
    requested_text = "inherit" if requested is None else str(int(requested))
    return (
        f"requested={requested_text} "
        f"effective={int(report['effective_max_diffractions'])} "
        f"tracer_default={int(report['trace_default_max_diffractions'])}"
    )


def summarize_path_result(paths, *, depth_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metadata = dict(getattr(paths, "metadata", {}) or {})
    timing = metadata.get("timing", {}) or {}
    return {
        "num_rx": int(paths.num_rx),
        "max_num_paths": int(paths.max_num_paths),
        "path_counts": dict(metadata.get("path_counts", {})),
        "depth_report": None if depth_report is None else dict(depth_report),
        "runtime_reuse": dict(metadata.get("runtime_reuse", {})),
        "timing_ms": {str(key): float(value) * 1000.0 for key, value in timing.items()},
    }


def summarize_field_payload(payload) -> dict[str, Any]:
    metadata = dict(getattr(payload, "metadata", {}) or {})
    performance_timing = metadata.get("performance_timing", {}) or {}
    return {
        "grid_shape": tuple(int(value) for value in payload.field.total.shape),
        "performance_timing_ms": {
            str(key): float(value) * 1000.0
            for key, value in performance_timing.items()
            if isinstance(value, (float, int))
        },
        "runtime_backends": dict(metadata.get("runtime_backends", {})),
    }


def evaluate_ratio_gate(
    *,
    gate_id: str,
    measured_label: str,
    measured_ms: float,
    baseline_label: str,
    baseline_ms: float,
    max_ratio: float,
    description: str,
) -> dict[str, Any]:
    baseline = float(baseline_ms)
    measured = float(measured_ms)
    ratio = None if baseline <= 0.0 else measured / baseline
    delta_ms = measured - baseline
    delta_percent = None if baseline <= 0.0 else (delta_ms / baseline) * 100.0
    return {
        "gate_id": gate_id,
        "kind": "performance_ratio",
        "description": description,
        "measured_label": measured_label,
        "measured_ms": measured,
        "baseline_label": baseline_label,
        "baseline_ms": baseline,
        "max_ratio": float(max_ratio),
        "observed_ratio": ratio,
        "observed_delta_ms": delta_ms,
        "observed_delta_percent": delta_percent,
        "passed": bool(ratio is not None and ratio <= float(max_ratio)),
    }


def evaluate_path_payload_equality(
    reference_paths,
    candidate_paths,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    errors: list[str] = []
    if int(reference_paths.num_rx) != int(candidate_paths.num_rx):
        errors.append("num_rx")
    if tuple(reference_paths.tx_pos) != tuple(candidate_paths.tx_pos):
        errors.append("tx_pos")
    if _path_complex_array(reference_paths).shape != _path_complex_array(candidate_paths).shape:
        errors.append("a.shape")
    elif not np.allclose(
        _path_complex_array(reference_paths),
        _path_complex_array(candidate_paths),
        rtol=rtol,
        atol=atol,
    ):
        errors.append("a")
    if _path_tau_array(reference_paths).shape != _path_tau_array(candidate_paths).shape:
        errors.append("tau.shape")
    elif not np.allclose(
        _path_tau_array(reference_paths),
        _path_tau_array(candidate_paths),
        rtol=rtol,
        atol=atol,
    ):
        errors.append("tau")
    if not np.array_equal(_path_valid_array(reference_paths), _path_valid_array(candidate_paths)):
        errors.append("valid")
    if not np.array_equal(_path_num_paths_array(reference_paths), _path_num_paths_array(candidate_paths)):
        errors.append("num_paths")
    if not np.array_equal(_path_types_array(reference_paths), _path_types_array(candidate_paths)):
        errors.append("types")
    if not np.allclose(
        _path_rx_positions_array(reference_paths),
        _path_rx_positions_array(candidate_paths),
        rtol=0.0,
        atol=0.0,
    ):
        errors.append("rx_positions")
    return {
        "passed": not errors,
        "errors": errors,
    }


def evaluate_geometry_toggle_correctness(no_geometry_paths, geometry_paths) -> dict[str, Any]:
    base = evaluate_path_payload_equality(no_geometry_paths, geometry_paths)
    errors = list(base["errors"])
    if no_geometry_paths.vertices is not None:
        errors.append("no_geometry.vertices_present")
    if no_geometry_paths.normals is not None:
        errors.append("no_geometry.normals_present")
    if no_geometry_paths.objects is not None:
        errors.append("no_geometry.objects_present")
    if geometry_paths.vertices is None:
        errors.append("geometry.vertices_missing")
    if geometry_paths.normals is None:
        errors.append("geometry.normals_missing")
    if geometry_paths.objects is None:
        errors.append("geometry.objects_missing")
    return {
        "passed": not errors,
        "errors": errors,
    }


def evaluate_mixed_monitor_correctness(
    field_only_result,
    path_only_result,
    mixed_result,
    *,
    field_name: str,
    path_name: str,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    errors: list[str] = []
    path_check = evaluate_path_payload_equality(
        _resolve_path_payload(path_only_result, path_name),
        _resolve_path_payload(mixed_result, path_name),
        atol=atol,
        rtol=rtol,
    )
    errors.extend(f"path.{item}" for item in path_check["errors"])

    standalone_field = _field_total_complex_array(_resolve_field_payload(field_only_result, field_name))
    mixed_field = _field_total_complex_array(_resolve_field_payload(mixed_result, field_name))
    if standalone_field.shape != mixed_field.shape:
        errors.append("field.total.shape")
    elif not np.allclose(standalone_field, mixed_field, rtol=rtol, atol=atol):
        errors.append("field.total")

    return {
        "passed": not errors,
        "errors": errors,
    }


def evaluate_warm_cache_trace_many_correctness(
    *,
    results: Sequence[object],
    requests: Sequence[Mapping[str, Any]],
    path_name: str,
    require_persistent_mode: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    persistent_hit_results = 0
    if len(results) != len(requests):
        errors.append("result_count")
        return {
            "passed": False,
            "errors": errors,
            "persistent_hit_results": 0,
        }

    for index, (result, request) in enumerate(zip(results, requests, strict=True)):
        paths = _resolve_path_payload(result, path_name)
        expected_tx = _to_float_tuple(request["tx_pos"])
        if tuple(paths.tx_pos) != expected_tx:
            errors.append(f"tx_pos[{index}]")

        override_positions = request.get("monitor_overrides", {}).get(path_name, {}).get("positions")
        if override_positions is not None:
            expected_rx = np.asarray(override_positions, dtype=np.float32)
            if not np.allclose(_path_rx_positions_array(paths), expected_rx, rtol=0.0, atol=0.0):
                errors.append(f"rx_positions[{index}]")

        cache_report = (
            dict(paths.metadata.get("runtime_reuse", {}))
            .get("diffraction_state_prep_cache", {})
        )
        if require_persistent_mode and str(cache_report.get("mode", "")) != "persistent":
            errors.append(f"cache_mode[{index}]")
        if int(cache_report.get("hits", 0)) <= 0:
            errors.append(f"cache_hit[{index}]")
        else:
            persistent_hit_results += 1

    return {
        "passed": not errors,
        "errors": errors,
        "persistent_hit_results": int(persistent_hit_results),
    }


__all__ = [
    "PHASE6_BENCHMARK_CASE_IDS",
    "PHASE6_GATE_IDS",
    "evaluate_geometry_toggle_correctness",
    "evaluate_mixed_monitor_correctness",
    "evaluate_path_payload_equality",
    "evaluate_ratio_gate",
    "evaluate_warm_cache_trace_many_correctness",
    "format_path_monitor_diffraction_depth_report",
    "resolve_path_monitor_diffraction_depth_report",
    "summarize_field_payload",
    "summarize_path_result",
]
