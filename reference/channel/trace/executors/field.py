from __future__ import annotations

from ...monitors.field.monitor import FieldMonitor
from ...monitors.field.result import MonitorResult
from ...trace.materials import ReflectionTraceDetail
from ...monitors.field.trace import trace_field_monitor
from ...monitors.orchestration import reflection_discovery_key_for_field


class FieldTraceExecutor:
    def __init__(self, session):
        self.session = session

    def run(
        self,
        monitor: FieldMonitor,
    ) -> tuple[MonitorResult | dict[str, object], tuple[object, ...], ReflectionTraceDetail]:
        solver_controls = self.session.cache_manager.resolve_monitor_solver_controls(
            monitor,
            execution_intent="field",
        )
        reflection_key = reflection_discovery_key_for_field(monitor, self.session.tx_pos)
        payload, reflection_detail = trace_field_monitor(
            self.session.tx_pos,
            monitor,
            self.session.scene,
            self.session.resolved_trace_config,
            solver_controls,
            reflection_detail=self.session.reflection_detail_cache.get(reflection_key),
            verbose=self.session.verbose,
            return_timing=self.session.return_timing,
            return_diffraction_audit=self.session.return_diffraction_audit,
        )
        return payload, reflection_key, reflection_detail


__all__ = ["FieldTraceExecutor"]
