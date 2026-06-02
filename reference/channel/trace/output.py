from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from ..monitors.field.result import MonitorResult
from ..monitors.path.result import PathResult
from ..monitors.radio_map import RadioMapResult


TracePayload: TypeAlias = MonitorResult | RadioMapResult | PathResult
TraceOutput: TypeAlias = TracePayload | dict[str, TracePayload]
MonitorPayloadInput: TypeAlias = MonitorResult | RadioMapResult | Mapping[str, object]
PathPayloadInput: TypeAlias = PathResult | Mapping[str, object]


def _resolve_monitor_payload(payload: MonitorPayloadInput) -> MonitorResult | RadioMapResult:
    if isinstance(payload, (MonitorResult, RadioMapResult)):
        return payload
    payload_kind = str(payload.get("kind", "field"))
    if payload_kind == "radio_map":
        return RadioMapResult.from_payload(payload)
    return MonitorResult.from_payload(payload)


def _resolve_path_payload(payload: PathPayloadInput) -> PathResult:
    if isinstance(payload, PathResult):
        return payload
    return PathResult.from_payload(payload)


def finalize_trace_output(
    *,
    monitors: Mapping[str, MonitorPayloadInput] | None = None,
    path_monitors: Mapping[str, PathPayloadInput] | None = None,
    primary_monitor_name: str | None = None,
) -> TraceOutput:
    resolved_monitors = {
        str(name): _resolve_monitor_payload(payload)
        for name, payload in dict(monitors or {}).items()
    }
    resolved_path_monitors = {
        str(name): _resolve_path_payload(payload)
        for name, payload in dict(path_monitors or {}).items()
    }

    total_payloads = len(resolved_monitors) + len(resolved_path_monitors)
    if total_payloads <= 0:
        raise ValueError("Trace produced no monitor payloads.")

    if total_payloads == 1:
        if primary_monitor_name is not None and primary_monitor_name in resolved_monitors:
            return resolved_monitors[primary_monitor_name]
        if len(resolved_monitors) == 1:
            return next(iter(resolved_monitors.values()))
        return next(iter(resolved_path_monitors.values()))

    merged: dict[str, TracePayload] = {}
    if primary_monitor_name is not None and primary_monitor_name in resolved_monitors:
        merged[primary_monitor_name] = resolved_monitors[primary_monitor_name]
    for name, payload in resolved_monitors.items():
        if name != primary_monitor_name:
            merged[name] = payload
    for name, payload in resolved_path_monitors.items():
        merged[name] = payload
    return merged


__all__ = [
    "TraceOutput",
    "TracePayload",
    "finalize_trace_output",
]
