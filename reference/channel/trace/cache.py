from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import drjit as dr

from ..monitors import RadioMapMonitor
from ..monitors.orchestration import TraceExecutionIntent, resolve_solver_controls
from ..trace.diffraction.state import PATH_EXPORT_REDUCED_STATE_LAYOUT
from ..utils import scalar


def point_grad_enabled(point) -> bool:
    if point is None:
        return False
    try:
        return bool(
            dr.grad_enabled(point.x)
            or dr.grad_enabled(point.y)
            or dr.grad_enabled(point.z)
        )
    except Exception:
        return False


def scene_geometry_grad_enabled(scene) -> bool:
    if scene is None:
        return False
    if point_grad_enabled(scene.vertices):
        return True
    tri_data = scene.tri_data_gpu
    if isinstance(tri_data, dict):
        for key in ("v0", "v1", "v2"):
            value = tri_data.get(key)
            if value is not None and point_grad_enabled(value):
                return True
    return False


def scene_material_grad_enabled(scene) -> bool:
    if scene is None:
        return False
    tri_data = scene.tri_data_gpu
    if isinstance(tri_data, dict):
        for key in ("material_eps_r", "material_sigma_e"):
            value = tri_data.get(key)
            if value is None:
                continue
            try:
                if bool(dr.grad_enabled(value)):
                    return True
            except Exception:
                continue
    return False


def mapping_signature(mapping: Mapping[str, object] | None):
    if mapping is None:
        return None
    return tuple(
        sorted(
            (str(key), value if isinstance(value, str) else float(value))
            for key, value in mapping.items()
        )
    )


def radio_map_execution_intent(monitor: RadioMapMonitor) -> TraceExecutionIntent:
    return (
        "radio_map_coherent"
        if str(monitor.combine_mode) == "coherent"
        else "radio_map_incoherent"
    )


@dataclass
class TraceCacheManager:
    scene: object
    trace_config: object
    resolved_trace_config: object
    path_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] = field(
        default_factory=dict
    )
    radio_map_diffraction_state_cache: dict[tuple[object, ...], tuple[object, ...]] = field(
        default_factory=dict
    )
    cached_scene_mesh_version: int = field(init=False)

    def __post_init__(self):
        self.cached_scene_mesh_version = self.scene_mesh_version()

    def scene_mesh_version(self) -> int:
        return int(self.scene._mesh_version)

    def clear(self):
        self.path_diffraction_state_cache.clear()
        self.radio_map_diffraction_state_cache.clear()
        self.cached_scene_mesh_version = self.scene_mesh_version()

    def refresh(self):
        current_mesh_version = self.scene_mesh_version()
        if current_mesh_version != self.cached_scene_mesh_version:
            self.clear()

    def resolve_monitor_solver_controls(
        self,
        monitor,
        *,
        execution_intent: TraceExecutionIntent,
    ) -> dict[str, object]:
        return resolve_solver_controls(
            self.trace_config,
            execution_intent=execution_intent,
            max_diffractions_override=monitor.max_diffractions,
        )

    def persistent_diffraction_state_cache_allowed(self, tx_pos) -> bool:
        if point_grad_enabled(tx_pos):
            return False
        if scene_geometry_grad_enabled(self.scene):
            return False
        if (
            bool(self.trace_config.use_scene_materials_for_diffraction)
            and scene_material_grad_enabled(self.scene)
        ):
            return False
        return True

    def path_persistent_diffraction_state_cache(self, tx_pos):
        if not self.persistent_diffraction_state_cache_allowed(tx_pos):
            return None
        return self.path_diffraction_state_cache

    def radio_map_persistent_diffraction_state_cache(self, tx_pos):
        if not self.persistent_diffraction_state_cache_allowed(tx_pos):
            return None
        return self.radio_map_diffraction_state_cache

    def diffraction_state_cache_key(
        self,
        *,
        tx_pos,
        receiver_z: float,
        ray_mode: str,
        solver_controls,
        reflection_detail,
    ) -> tuple[object, ...]:
        effective = solver_controls["effective"]
        config = self.resolved_trace_config
        return (
            self.scene_mesh_version(),
            str(self.scene.edge_selection_mode),
            str(self.scene.boundary_edge_policy),
            float(self.scene.vertical_ratio),
            float(scalar(tx_pos.x)),
            float(scalar(tx_pos.y)),
            float(scalar(tx_pos.z)),
            float(receiver_z),
            str(ray_mode),
            float(config.wavelength),
            float(config.k),
            float(config.reflection_coef),
            bool(config.enable_rd_diffraction),
            mapping_signature(config.diffraction_material),
            bool(config.use_scene_materials_for_diffraction),
            tuple(float(value) for value in config.tx_polarization),
            str(solver_controls["selected"]),
            int(effective["max_diffractions"]),
            int(effective["reflection_n_rays"]),
            int(effective["reflection_max_bounces"]),
            effective["diffraction_state_budget"],
            effective["inserted_reflection_state_budget"],
            effective["max_inserted_reflections_per_path"],
            str(effective["memory_profile"]),
            PATH_EXPORT_REDUCED_STATE_LAYOUT,
            None if reflection_detail is None else id(reflection_detail),
        )

    def path_diffraction_state_cache_key(
        self,
        *,
        tx_pos,
        receiver_z: float,
        monitor,
        solver_controls,
        reflection_detail,
    ) -> tuple[object, ...]:
        return self.diffraction_state_cache_key(
            tx_pos=tx_pos,
            receiver_z=receiver_z,
            ray_mode=monitor.ray_mode,
            solver_controls=solver_controls,
            reflection_detail=reflection_detail,
        )

    def radio_map_diffraction_state_cache_key(
        self,
        *,
        tx_pos,
        receiver_z: float,
        monitor,
        solver_controls,
        reflection_detail,
    ) -> tuple[object, ...]:
        return self.diffraction_state_cache_key(
            tx_pos=tx_pos,
            receiver_z=receiver_z,
            ray_mode=monitor.ray_mode,
            solver_controls=solver_controls,
            reflection_detail=reflection_detail,
        )


__all__ = [
    "TraceCacheManager",
    "mapping_signature",
    "point_grad_enabled",
    "radio_map_execution_intent",
    "scene_geometry_grad_enabled",
    "scene_material_grad_enabled",
]
