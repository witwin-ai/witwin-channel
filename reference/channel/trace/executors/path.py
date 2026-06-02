from __future__ import annotations

from ...monitors.path.monitor import PathMonitor
from ...monitors.path.result import PathResult
from ...monitors.path.trace import trace_path_monitor
from ...monitors.orchestration import reflection_discovery_key_for_path
from ...trace.materials import ReflectionTraceDetail


class PathTraceExecutor:
    def __init__(self, session):
        self.session = session

    def run(
        self,
        monitor: PathMonitor,
    ) -> tuple[PathResult, tuple[object, ...], ReflectionTraceDetail]:
        solver_controls = self.session.cache_manager.resolve_monitor_solver_controls(
            monitor,
            execution_intent="path_export",
        )
        reflection_key = reflection_discovery_key_for_path(monitor)
        payload, reflection_detail = trace_path_monitor(
            self.session.tx_pos,
            monitor,
            self.session.scene,
            self.session.resolved_trace_config,
            solver_controls,
            reflection_detail=self.session.reflection_detail_cache.get(reflection_key),
            persistent_diffraction_state_cache=self.session.persistent_path_diffraction_cache,
            local_diffraction_state_cache=self.session.path_diffraction_local_cache,
            diffraction_state_cache_key_fn=lambda receiver_z, detail, monitor=monitor, solver_controls=solver_controls: self.session.cache_manager.path_diffraction_state_cache_key(
                tx_pos=self.session.tx_pos,
                receiver_z=receiver_z,
                monitor=monitor,
                solver_controls=solver_controls,
                reflection_detail=detail,
            ),
            verbose=self.session.verbose,
            return_timing=self.session.return_timing,
        )
        return payload, reflection_key, reflection_detail


__all__ = ["PathTraceExecutor"]
