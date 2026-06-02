from __future__ import annotations

from ...monitors.radio_map import RadioMapMonitor, RadioMapResult
from ...monitors.radio_map.deterministic.trace import trace_radio_map_monitor
from ...monitors.orchestration import reflection_discovery_key_for_radio_map
from ...trace.materials import ReflectionTraceDetail
from ..cache import radio_map_execution_intent


class RadioMapTraceExecutor:
    def __init__(self, session):
        self.session = session

    def run(
        self,
        monitor: RadioMapMonitor,
    ) -> tuple[RadioMapResult | dict[str, object], tuple[object, ...], ReflectionTraceDetail]:
        solver_controls = self.session.cache_manager.resolve_monitor_solver_controls(
            monitor,
            execution_intent=radio_map_execution_intent(monitor),
        )
        reflection_key = reflection_discovery_key_for_radio_map(monitor)
        payload, reflection_detail = trace_radio_map_monitor(
            self.session.tx_pos,
            monitor,
            self.session.scene,
            self.session.resolved_trace_config,
            solver_controls,
            reflection_detail=self.session.reflection_detail_cache.get(reflection_key),
            persistent_diffraction_state_cache=self.session.persistent_radio_map_diffraction_cache,
            local_diffraction_state_cache=self.session.radio_map_diffraction_local_cache,
            radio_map_accumulation_backend=monitor.accumulation_backend,
            diffraction_state_cache_key_fn=lambda receiver_z, detail, monitor=monitor, solver_controls=solver_controls: self.session.cache_manager.radio_map_diffraction_state_cache_key(
                tx_pos=self.session.tx_pos,
                receiver_z=receiver_z,
                monitor=monitor,
                solver_controls=solver_controls,
                reflection_detail=detail,
            ),
            verbose=self.session.verbose,
            return_timing=self.session.return_timing,
            return_reflection_detail=True,
        )
        return payload, reflection_key, reflection_detail


__all__ = ["RadioMapTraceExecutor"]
