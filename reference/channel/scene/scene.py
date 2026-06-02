from __future__ import annotations

from dataclasses import dataclass

import rayd
import torch
import witwin as wt

from witwin.core import GeometryBase, Material, SceneBase, Structure
from ..config import ChannelConfig, coerce_channel_config
from ..monitors.field import FieldMonitor
from ..monitors.path import PathMonitor
from ..monitors.radio_map import RadioMapMonitor
from ..utils import drjit_to_torch_view

from .runtime_state import SceneRuntime


def _resolve_scene_device(device: str | None) -> str:
    requested = "cuda" if device is None else device
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Channel scenes default to CUDA, but torch.cuda.is_available() is False. "
            "Pass device='cpu' only for scene construction or non-rendering workflows."
        )
    return str(resolved)


def _default_material(material: Material | None) -> Material:
    return material if material is not None else Material()


def _as_point3(value):
    if isinstance(value, wt.Point3f):
        return value
    return wt.Point3f(
        wt.Float(value.x),
        wt.Float(value.y),
        wt.Float(value.z),
    )


def _as_vector3(value):
    if isinstance(value, wt.Vector3f):
        return value
    return wt.Vector3f(
        wt.Float(value.x),
        wt.Float(value.y),
        wt.Float(value.z),
    )


@dataclass(frozen=True)
class _SceneIntersection:
    valid: wt.Bool
    t: wt.Float
    p: wt.Point3f
    n: wt.Vector3f
    geo_n: wt.Vector3f
    prim_index: wt.Int32
    prim_id: wt.Int32
    shape_id: wt.Int32

    def is_valid(self):
        return self.valid


class _ScenePreliminaryIntersection:
    def __init__(self, surface_interaction: _SceneIntersection):
        self._surface_interaction = surface_interaction
        self.t = surface_interaction.t
        self.prim_index = surface_interaction.prim_index
        self.prim_id = surface_interaction.prim_id
        self.shape_id = surface_interaction.shape_id

    def is_valid(self):
        return self._surface_interaction.is_valid()

    def compute_surface_interaction(self, ray=None, flags=None):
        del ray, flags
        return self._surface_interaction


class Scene(SceneBase):
    """Declarative channel scene backed by shared core structures."""

    @classmethod
    def from_sionna(cls, sionna_scene, **kwargs) -> "Scene":
        from .sionna import scene_from_sionna_scene

        return scene_from_sionna_scene(sionna_scene, scene_cls=cls, **kwargs)

    def __init__(
        self,
        *,
        config: ChannelConfig | dict | None = None,
        structures=None,
        monitors=None,
        metadata=None,
        device: str | None = "cuda",
        verbose: bool = False,
        vertical_ratio: float = 0.7,
        edge_selection_mode: str = "vertical_only",
        boundary_edge_policy: str = "exclude",
    ):
        if edge_selection_mode not in {"vertical_only", "all_edges"}:
            raise ValueError(
                f"Unsupported edge_selection_mode '{edge_selection_mode}'. "
                "Supported values are 'vertical_only' and 'all_edges'."
            )
        if boundary_edge_policy not in {"exclude", "half_plane"}:
            raise ValueError(
                f"Unsupported boundary_edge_policy '{boundary_edge_policy}'. "
                "Supported values are 'exclude' and 'half_plane'."
            )
        channel_config = coerce_channel_config(config)
        resolved_device = _resolve_scene_device(device)
        normalized_structures = []
        if structures is not None:
            for structure in structures:
                normalized_structures.append(self._normalize_structure(structure))
            seen_names = set()
            for structure in normalized_structures:
                if structure.name in seen_names:
                    raise ValueError(f"Structure '{structure.name}' already exists.")
                seen_names.add(structure.name)
        normalized_monitors = []
        if monitors is not None:
            for monitor in monitors:
                normalized_monitors.append(self._normalize_monitor(monitor))
            seen_monitor_names = set()
            for monitor in normalized_monitors:
                if monitor.name in seen_monitor_names:
                    raise ValueError(f"Monitor '{monitor.name}' already exists.")
                seen_monitor_names.add(monitor.name)

        super().__init__(
            structures=normalized_structures,
            monitors=normalized_monitors,
            metadata=metadata,
            device=resolved_device,
            verbose=verbose,
        )
        self.vertical_ratio = float(vertical_ratio)
        self.edge_selection_mode = edge_selection_mode
        self.boundary_edge_policy = boundary_edge_policy
        self.config = ChannelConfig(trace=channel_config.trace)
        self._runtime = SceneRuntime(self)
        self._rebuild_runtime()

    @staticmethod
    def _normalize_structure(structure) -> Structure:
        if not isinstance(structure, Structure):
            raise TypeError("Channel Scene structures must be witwin.core.Structure instances.")
        if not isinstance(structure.geometry, GeometryBase):
            raise TypeError("Channel Scene structures must wrap a GeometryBase geometry.")
        if structure.name is None:
            raise ValueError("Channel Scene structures must define a unique name.")
        return structure

    @staticmethod
    def _normalize_monitor(monitor):
        if not isinstance(monitor, (FieldMonitor, PathMonitor, RadioMapMonitor)):
            raise TypeError(
                "Channel Scene monitors must be FieldMonitor, PathMonitor, or RadioMapMonitor instances."
            )
        if monitor.name is None:
            raise ValueError("Channel Scene monitors must define a unique name.")
        return monitor

    def add_structure(self, structure: Structure) -> "Scene":
        normalized = self._normalize_structure(structure)
        if any(existing.name == normalized.name for existing in self.structures):
            raise ValueError(f"Structure '{normalized.name}' already exists.")
        self.structures.append(normalized)
        self._runtime_vertex_override = None
        self._rebuild_runtime()
        return self

    def add_monitor(self, monitor) -> "Scene":
        normalized = self._normalize_monitor(monitor)
        if any(existing.name == normalized.name for existing in self.monitors):
            raise ValueError(f"Monitor '{normalized.name}' already exists.")
        self.monitors.append(normalized)
        return self

    def add_mesh(
        self,
        *,
        name: str,
        geometry: GeometryBase,
        material: Material | None = None,
        metadata=None,
    ) -> "Scene":
        if not isinstance(geometry, GeometryBase):
            raise TypeError("add_mesh requires a GeometryBase instance.")
        return self.add_structure(
            Structure(
                geometry=geometry,
                material=_default_material(material),
                name=name,
                metadata=metadata,
            )
        )

    def clone(self, **overrides) -> "Scene":
        if "finite_edge_policy" in overrides:
            raise TypeError(
                "Scene.clone() no longer accepts finite_edge_policy. "
                "Finite-wedge diffraction is always used."
            )
        return Scene(
            config=overrides.get("config", self.config),
            structures=overrides.get("structures", list(self.structures)),
            monitors=overrides.get("monitors", list(self.monitors)),
            metadata=overrides.get("metadata", dict(self.metadata)),
            device=overrides.get("device", self.device),
            verbose=overrides.get("verbose", self.verbose),
            vertical_ratio=overrides.get("vertical_ratio", self.vertical_ratio),
            edge_selection_mode=overrides.get("edge_selection_mode", self.edge_selection_mode),
            boundary_edge_policy=overrides.get("boundary_edge_policy", self.boundary_edge_policy),
        )

    def to_sionna(self, **kwargs):
        from .sionna import scene_to_sionna_scene

        return scene_to_sionna_scene(self, **kwargs).scene

    def resolved_monitors(self):
        return list(self.monitors)

    def _rebuild_runtime(self):
        self._runtime.rebuild()

    def get_global_diffraction_edge_indices(self):
        return self._runtime.get_global_diffraction_edge_indices()

    def get_adjacent_diffraction_edge_indices_for_triangle(self, prim_idx: int, include_sibling: bool = True):
        return self._runtime.get_adjacent_diffraction_edge_indices_for_triangle(
            prim_idx,
            include_sibling=include_sibling,
        )

    def get_triangle_surface_edge_candidates(self, prim_idx):
        return self._runtime.get_triangle_surface_edge_candidates(prim_idx)

    def ray_test(self, ray, active=True):
        rayd_scene = self._require_rayd_scene()
        if not isinstance(ray, rayd.Ray):
            raise TypeError(f"Scene.ray_test expects rayd.Ray, got {type(ray).__name__}.")
        return wt.Bool(rayd_scene.shadow_test(ray, active=active))

    def ray_intersect(self, ray, active=True, flags=None):
        rayd_scene = self._require_rayd_scene()
        if not isinstance(ray, rayd.Ray):
            raise TypeError(f"Scene.ray_intersect expects rayd.Ray, got {type(ray).__name__}.")
        raw = rayd_scene.intersect(
            ray,
            active=active,
            flags=rayd.RayFlags.All if flags is None else flags,
        )
        prim_index = wt.Int32(raw.prim_id)
        return _SceneIntersection(
            valid=wt.Bool(raw.is_valid()),
            t=wt.Float(raw.t),
            p=_as_point3(raw.p),
            n=_as_vector3(raw.n),
            geo_n=_as_vector3(raw.geo_n),
            prim_index=prim_index,
            prim_id=prim_index,
            shape_id=wt.Int32(raw.shape_id),
        )

    def ray_intersect_preliminary(self, ray, active=True):
        return _ScenePreliminaryIntersection(self.ray_intersect(ray, active=active))

    def nearest_edge(self, query, active=True):
        if self._rayd_scene is None:
            raise RuntimeError("Scene.nearest_edge() requires an active RayD runtime scene.")
        return self._rayd_scene.nearest_edge(query, active=active)

    def set_edge_mask(self, mask):
        if self._rayd_scene is None:
            raise RuntimeError("Scene.set_edge_mask() requires an active RayD runtime scene.")
        self._rayd_scene.set_edge_mask(mask)

    def edge_mask(self):
        if self._rayd_scene is None:
            raise RuntimeError("Scene.edge_mask() requires an active RayD runtime scene.")
        return drjit_to_torch_view(self._rayd_scene.edge_mask(), detach=True, dtype=torch.bool)

    def update_vertices(self, vertices, recompute_edges: bool = True):
        self._runtime.update_vertices(vertices, recompute_edges=recompute_edges)

    def get_edge_data(self, calculation_height, include_projection: bool = True):
        return self._runtime.get_edge_data(
            calculation_height,
            include_projection=include_projection,
        )

    def get_edges_2d(self, height: float) -> list:
        return self.get_edge_data(height, include_projection=True)["edges_2d"]

    def get_corners_2d(self, height: float) -> list:
        return self.get_edge_data(height, include_projection=True)["corners_2d"]

    @property
    def n_diffraction_edges(self) -> int:
        return len(self.vertical_edges)

    def _require_rayd_scene(self):
        if self._rayd_scene is None:
            raise RuntimeError("Scene ray queries require an active RayD runtime scene.")
        return self._rayd_scene


def _runtime_attr(name: str):
    def getter(self):
        return getattr(self._runtime, name)

    def setter(self, value):
        setattr(self._runtime, name, value)

    return property(getter, setter)


for _runtime_name in (
    "vertices",
    "faces",
    "_wedge_backend_source",
    "_wedge_backend_kind",
    "_wedge_backend_error",
    "_rayd_scene",
    "_n_verts",
    "_mesh_center_3d",
    "_mesh_center_2d",
    "_edge_cache",
    "_edge_topology",
    "_runtime_vertex_override",
    "_global_diffraction_edge_indices",
    "vertical_edges",
    "_vertical_edge_id_to_index",
    "tri_data_gpu",
    "_tri_edge_indices",
    "_diffraction_edge_gpu",
    "_triangle_surface_groups",
    "_triangle_surface_group_by_triangle",
    "_triangle_surface_edge_groups",
    "_triangle_surface_data",
    "_triangle_material_data",
    "_face_normals",
    "_edge_runtime_dirty",
    "_mesh_version",
    "edge_selection_summary",
):
    setattr(Scene, _runtime_name, _runtime_attr(_runtime_name))


def _edge_runtime_property(name: str):
    def getter(self):
        from .builder import _ensure_edge_runtime

        _ensure_edge_runtime(self._runtime)
        return getattr(self._runtime, name)

    def setter(self, value):
        setattr(self._runtime, name, value)

    return property(getter, setter)


Scene.vertical_edges = _edge_runtime_property("vertical_edges")
Scene.edge_selection_summary = _edge_runtime_property("edge_selection_summary")
