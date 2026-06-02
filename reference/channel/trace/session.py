from __future__ import annotations

from collections.abc import Mapping

from ..monitors import (
    FieldMonitor,
    PathMonitor,
    RadioMapMonitor,
    resolve_field_monitor,
    resolve_path_monitor,
    resolve_radio_map_monitor,
)
from ..monitors.field.result import MonitorResult
from ..monitors.path.result import PathResult
from ..monitors.radio_map import RadioMapResult
from .materials import ReflectionTraceDetail
from .executors.field import FieldTraceExecutor
from .executors.path import PathTraceExecutor
from .executors.radio_map import RadioMapTraceExecutor
from .output import TraceOutput, finalize_trace_output


def normalize_monitor_overrides(
    monitor_overrides: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    if monitor_overrides is None:
        return {}
    if not isinstance(monitor_overrides, Mapping):
        raise TypeError("monitor_overrides must be a mapping of monitor name -> override mapping.")
    normalized = {}
    for name, overrides in monitor_overrides.items():
        if not isinstance(overrides, Mapping):
            raise TypeError(
                "monitor_overrides entries must be mappings of override keys for each monitor."
            )
        normalized[str(name)] = dict(overrides)
    return normalized


def merge_monitor_overrides(
    base_overrides: Mapping[str, Mapping[str, object]] | None,
    request_overrides: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    base = {} if base_overrides is None else dict(base_overrides)
    if request_overrides is None:
        return base
    merged = dict(base)
    for name, overrides in dict(request_overrides).items():
        merged[str(name)] = dict(overrides)
    return merged


class TraceSession:
    def __init__(
        self,
        tracer,
        *,
        tx_pos,
        monitor=None,
        monitor_overrides=None,
        verbose: bool = False,
        return_timing: bool = False,
        return_diffraction_audit: bool = False,
    ):
        self.tracer = tracer
        self.scene = tracer.scene
        self.tx_pos = tx_pos
        self.verbose = bool(verbose)
        self.return_timing = bool(return_timing)
        self.return_diffraction_audit = bool(return_diffraction_audit)
        self.resolved_trace_config = tracer._resolved_trace_config
        self.cache_manager = tracer._trace_caches
        self.cache_manager.refresh()

        self.monitors = self._apply_monitor_overrides(
            monitors=self._resolve_trace_monitors(monitor=monitor),
            monitor_overrides=monitor_overrides,
        )
        self.path_diffraction_local_cache: dict[tuple[object, ...], tuple[object, ...]] = {}
        self.radio_map_diffraction_local_cache: dict[tuple[object, ...], tuple[object, ...]] = {}
        self.persistent_path_diffraction_cache = self.cache_manager.path_persistent_diffraction_state_cache(
            self.tx_pos
        )
        self.persistent_radio_map_diffraction_cache = (
            self.cache_manager.radio_map_persistent_diffraction_state_cache(self.tx_pos)
        )
        self.reflection_detail_cache: dict[tuple[object, ...], ReflectionTraceDetail] = {}
        self.monitor_payloads: dict[str, MonitorResult | RadioMapResult] = {}
        self.path_payloads: dict[str, PathResult] = {}
        self.primary_monitor_name: str | None = None

        self._field_executor = FieldTraceExecutor(self)
        self._path_executor = PathTraceExecutor(self)
        self._radio_map_executor = RadioMapTraceExecutor(self)

    def _resolve_trace_monitors(
        self,
        *,
        monitor,
    ) -> list[FieldMonitor | PathMonitor | RadioMapMonitor]:
        if monitor is not None:
            if isinstance(monitor, (FieldMonitor, PathMonitor, RadioMapMonitor)):
                provided_monitors = [monitor]
            else:
                try:
                    provided_monitors = list(monitor)
                except TypeError as exc:
                    raise TypeError(
                        "trace(monitor=...) expects a FieldMonitor, PathMonitor, RadioMapMonitor, "
                        "or an iterable of monitor instances."
                    ) from exc
                if len(provided_monitors) == 0:
                    raise ValueError("trace(monitor=...) cannot receive an empty iterable.")
        else:
            provided_monitors = self.scene.resolved_monitors()

        if len(provided_monitors) == 0:
            raise ValueError(
                "Tracer.trace() requires monitor=FieldMonitor(...) / PathMonitor(...) / RadioMapMonitor(...) "
                "or at least one monitor attached to the Scene."
            )

        resolved_monitors = []
        for item in provided_monitors:
            if isinstance(item, FieldMonitor):
                resolved_monitors.append(resolve_field_monitor(item))
            elif isinstance(item, PathMonitor):
                resolved_monitors.append(resolve_path_monitor(item))
            elif isinstance(item, RadioMapMonitor):
                resolved_monitors.append(resolve_radio_map_monitor(item))
            else:
                raise TypeError(
                    "trace(monitor=...) expects FieldMonitor, PathMonitor, and RadioMapMonitor instances only."
                )
        return resolved_monitors

    def _apply_monitor_overrides(
        self,
        *,
        monitors: list[FieldMonitor | PathMonitor | RadioMapMonitor],
        monitor_overrides: Mapping[str, Mapping[str, object]] | None,
    ) -> list[FieldMonitor | PathMonitor | RadioMapMonitor]:
        pending = normalize_monitor_overrides(monitor_overrides)
        if not pending:
            return monitors

        resolved_monitors = []
        for resolved_monitor in monitors:
            overrides = pending.pop(str(resolved_monitor.name), None)
            resolved_monitors.append(
                resolved_monitor.with_overrides(**overrides)
                if overrides is not None
                else resolved_monitor
            )

        if pending:
            missing = ", ".join(sorted(pending))
            raise KeyError(f"monitor_overrides referenced unknown monitors: {missing}")
        return resolved_monitors

    def _remember_primary_monitor(self, name: str):
        if self.primary_monitor_name is None:
            self.primary_monitor_name = str(name)

    def _cache_reflection_detail(
        self,
        reflection_key: tuple[object, ...],
        reflection_detail: ReflectionTraceDetail | None,
    ) -> None:
        if reflection_detail is not None:
            self.reflection_detail_cache[reflection_key] = reflection_detail

    def run_monitor(self, monitor: FieldMonitor | PathMonitor | RadioMapMonitor) -> None:
        if isinstance(monitor, FieldMonitor):
            payload, reflection_key, reflection_detail = self._field_executor.run(monitor)
            self.monitor_payloads[monitor.name] = payload
            self._cache_reflection_detail(reflection_key, reflection_detail)
            self._remember_primary_monitor(monitor.name)
            return

        if isinstance(monitor, RadioMapMonitor):
            payload, reflection_key, reflection_detail = self._radio_map_executor.run(monitor)
            self.monitor_payloads[monitor.name] = payload
            self._cache_reflection_detail(reflection_key, reflection_detail)
            self._remember_primary_monitor(monitor.name)
            return

        payload, reflection_key, reflection_detail = self._path_executor.run(monitor)
        self.path_payloads[monitor.name] = payload
        self._cache_reflection_detail(reflection_key, reflection_detail)

    def execute(self) -> TraceSession:
        for monitor in self.monitors:
            self.run_monitor(monitor)
        return self

    def run(self) -> TraceOutput:
        self.execute()
        return self.finalize_output()

    def finalize_output(self) -> TraceOutput:
        return finalize_trace_output(
            monitors=self.monitor_payloads,
            path_monitors=self.path_payloads,
            primary_monitor_name=self.primary_monitor_name,
        )


__all__ = [
    "TraceSession",
    "merge_monitor_overrides",
    "normalize_monitor_overrides",
]
