"""Runtime orchestration, caching, and pipeline functions for wedge building."""

from __future__ import annotations

from weakref import WeakKeyDictionary

import drjit as dr
import witwin as wt

from ...utils.drjit_ops import Gather, mask_count
from .adapters import RayDSceneAdapter, is_rayd_scene_like
from .build import build_wedge_geometry
from .types import (
    EdgeInfoBuffer,
    EdgeTopologyBuffer,
    HeightPlaneAnchorSpec,
    TriangleWedgeMap,
    WedgeAnchorView,
    WedgeBackend,
    WedgeGeometry,
    WedgeGeometryConfig,
    WedgePack,
    WedgeSelection,
    WedgeSelectionConfig,
)


# ---------------------------------------------------------------------------
# Cache registry
# ---------------------------------------------------------------------------

class WedgeRuntimeRegistry:
    def __init__(self):
        self._runtimes = WeakKeyDictionary()

    def get(self, key):
        return self._runtimes.get(key)

    def set(self, key, runtime) -> None:
        self._runtimes[key] = runtime

    def discard(self, key) -> None:
        self._runtimes.pop(key, None)

    def clear(self) -> None:
        self._runtimes = WeakKeyDictionary()


RUNTIME_REGISTRY = WedgeRuntimeRegistry()


# ---------------------------------------------------------------------------
# Pipeline functions (select, anchors, mapping, pack)
# ---------------------------------------------------------------------------

def select_wedges(
    geometry: WedgeGeometry,
    config: WedgeSelectionConfig | None = None,
) -> WedgeSelection:
    config = WedgeSelectionConfig() if config is None else config
    if geometry.n_edges == 0:
        return WedgeSelection.empty(geometry)

    valid_mask = geometry.is_valid & (geometry.length > 0.0) & (geometry.wedge_n > config.min_wedge_n)
    vertical_mask = dr.abs(geometry.edge_dir.z) > config.vertical_ratio
    if config.mode == "vertical_only":
        selected_mask = valid_mask & vertical_mask
    else:
        selected_mask = valid_mask

    selected_idx = dr.compress(selected_mask)
    summary = {
        "total_edges": int(geometry.n_edges),
        "valid_wedges": mask_count(valid_mask),
        "boundary_wedges": mask_count(geometry.is_valid & geometry.is_boundary),
        "interior_wedges": mask_count(geometry.is_valid & ~geometry.is_boundary),
        "edges_matching_vertical_ratio": mask_count(vertical_mask),
        "selected_wedges": mask_count(selected_mask),
        "selected_boundary_wedges": mask_count(selected_mask & geometry.is_boundary),
        "selected_interior_wedges": mask_count(selected_mask & ~geometry.is_boundary),
    }
    return WedgeSelection(
        geometry=geometry,
        selected_idx=selected_idx,
        selected_mask=selected_mask,
        summary=summary,
    )


def build_height_plane_anchors(
    selection: WedgeSelection,
    anchor_spec: HeightPlaneAnchorSpec,
) -> WedgeAnchorView:
    if selection.size() == 0:
        return WedgeAnchorView.empty()

    geometry = selection.geometry
    wedge_idx = selection.selected_idx
    start = Gather.point3(geometry.start, wedge_idx)
    end = Gather.point3(geometry.end, wedge_idx)
    edge_vec = end - start

    dz = end.z - start.z
    horizontal = dr.abs(dz) <= anchor_spec.clamp_epsilon
    t = dr.select(horizontal, wt.Float(0.5), dr.clip((wt.Float(anchor_spec.z) - start.z) / dz, 0.0, 1.0))
    is_clamped = (t <= anchor_spec.clamp_epsilon) | (t >= 1.0 - anchor_spec.clamp_epsilon)
    anchor_pos = start + edge_vec * t

    keep_idx = dr.arange(wt.UInt32, selection.size())
    if dr.width(keep_idx) == 0:
        return WedgeAnchorView(
            wedge_idx=dr.zeros(wt.UInt32, 0),
            anchor_pos=wt.Point3f(dr.zeros(wt.Float, 0), dr.zeros(wt.Float, 0), dr.zeros(wt.Float, 0)),
            anchor_t=dr.zeros(wt.Float, 0),
            is_clamped=dr.zeros(wt.Bool, 0),
            summary={
                "input_selected_wedges": selection.size(),
                "output_anchors": 0,
                "excluded_clamped_anchors": 0,
            },
        )

    summary = {
        "input_selected_wedges": selection.size(),
        "output_anchors": int(dr.width(keep_idx)),
        "excluded_clamped_anchors": 0,
    }
    return WedgeAnchorView(
        wedge_idx=dr.gather(wt.UInt32, wedge_idx, keep_idx),
        anchor_pos=Gather.point3(anchor_pos, keep_idx),
        anchor_t=dr.gather(wt.Float, t, keep_idx),
        is_clamped=dr.gather(wt.Bool, is_clamped, keep_idx),
        summary=summary,
    )


def build_triangle_wedge_map(
    selection: WedgeSelection,
    triangle_edge_indices,
    n_triangles: int,
    local: bool = True,
) -> TriangleWedgeMap:
    if n_triangles == 0:
        return TriangleWedgeMap.empty()

    edge0_global, edge1_global, edge2_global = triangle_edge_indices
    geometry = selection.geometry
    selected_lookup = dr.full(wt.Int32, -1, geometry.n_edges)
    if selection.size() > 0:
        selected_globals = dr.gather(wt.Int32, geometry.global_edge_id, selection.selected_idx)
        if local:
            mapped_values = dr.arange(wt.Int32, selection.size())
        else:
            mapped_values = selected_globals
        dr.scatter(selected_lookup, mapped_values, selected_globals)

    mapped = []
    for edge_global in (edge0_global, edge1_global, edge2_global):
        valid = edge_global >= 0
        safe_idx = dr.select(valid, wt.UInt32(edge_global), wt.UInt32(0))
        mapped_edge = dr.gather(wt.Int32, selected_lookup, safe_idx)
        mapped.append(dr.select(valid, mapped_edge, wt.Int32(-1)))

    return TriangleWedgeMap(
        edge0=wt.Int32(mapped[0]),
        edge1=wt.Int32(mapped[1]),
        edge2=wt.Int32(mapped[2]),
        n_triangles=int(n_triangles),
        n_wedges=selection.size(),
    )


def pack_wedges(selection: WedgeSelection, anchors: WedgeAnchorView) -> WedgePack:
    if anchors.size() == 0:
        return WedgePack.empty()

    geometry = selection.geometry
    wedge_idx = anchors.wedge_idx
    n_wedges = anchors.size()
    length = dr.gather(wt.Float, geometry.length, wedge_idx)
    anchor_t = anchors.anchor_t
    return WedgePack(
        n_wedges=n_wedges,
        pos=anchors.anchor_pos,
        edge_dir=Gather.vector3(geometry.edge_dir, wedge_idx),
        length=length,
        line_min=-anchor_t * length,
        line_max=(wt.Float(1.0) - anchor_t) * length,
        n0=Gather.vector3(geometry.n0, wedge_idx),
        nn=Gather.vector3(geometry.nn, wedge_idx),
        wedge_n=dr.gather(wt.Float, geometry.wedge_n, wedge_idx),
        adjacent_face0=dr.gather(wt.Int32, geometry.face0, wedge_idx),
        adjacent_face1=dr.gather(wt.Int32, geometry.face1, wedge_idx),
        global_idx=dr.gather(wt.Int32, geometry.global_edge_id, wedge_idx),
        local_idx=dr.arange(wt.UInt32, n_wedges),
        is_boundary=dr.gather(wt.Bool, geometry.is_boundary, wedge_idx),
        is_clamped=anchors.is_clamped,
        summary={**selection.summary, **anchors.summary},
    )


# ---------------------------------------------------------------------------
# Runtime class and factory
# ---------------------------------------------------------------------------

def create_wedge_backend(source) -> WedgeBackend:
    source = getattr(source, "_wedge_backend_source", source)
    if isinstance(source, WedgeBackend):
        return source
    if is_rayd_scene_like(source):
        return RayDSceneAdapter(source)
    raise TypeError("Unsupported wedge backend source.")


class WedgeRuntime:
    def __init__(self, backend: WedgeBackend):
        self._backend = backend
        self._geometry_cache: dict[tuple[int, WedgeGeometryConfig], WedgeGeometry] = {}
        self._selection_cache: dict[tuple[int, WedgeGeometryConfig, WedgeSelectionConfig], WedgeSelection] = {}
        self._anchor_cache: dict[tuple[int, WedgeGeometryConfig, WedgeSelectionConfig, HeightPlaneAnchorSpec], WedgeAnchorView] = {}
        self._triangle_map_cache: dict[tuple[int, WedgeGeometryConfig, WedgeSelectionConfig, bool], TriangleWedgeMap] = {}
        self._pack_cache: dict[tuple[int, WedgeGeometryConfig, WedgeSelectionConfig, HeightPlaneAnchorSpec], WedgePack] = {}

    def clear(self) -> None:
        self._geometry_cache.clear()
        self._selection_cache.clear()
        self._anchor_cache.clear()
        self._triangle_map_cache.clear()
        self._pack_cache.clear()

    def geometry(self, config: WedgeGeometryConfig | None = None) -> WedgeGeometry:
        config = WedgeGeometryConfig() if config is None else config
        key = (int(self._backend.edge_version()), config)
        geometry = self._geometry_cache.get(key)
        if geometry is None:
            geometry = build_wedge_geometry(self._backend.edge_info(), self._backend.edge_topology(), config)
            self._geometry_cache[key] = geometry
        return geometry

    def select(
        self,
        geometry_config: WedgeGeometryConfig | None = None,
        selection_config: WedgeSelectionConfig | None = None,
    ) -> WedgeSelection:
        geometry_config = WedgeGeometryConfig() if geometry_config is None else geometry_config
        selection_config = WedgeSelectionConfig() if selection_config is None else selection_config
        key = (int(self._backend.edge_version()), geometry_config, selection_config)
        selection = self._selection_cache.get(key)
        if selection is None:
            selection = select_wedges(self.geometry(geometry_config), selection_config)
            self._selection_cache[key] = selection
        return selection

    def anchors(
        self,
        anchor_spec: HeightPlaneAnchorSpec,
        geometry_config: WedgeGeometryConfig | None = None,
        selection_config: WedgeSelectionConfig | None = None,
    ) -> WedgeAnchorView:
        geometry_config = WedgeGeometryConfig() if geometry_config is None else geometry_config
        selection_config = WedgeSelectionConfig() if selection_config is None else selection_config
        key = (int(self._backend.edge_version()), geometry_config, selection_config, anchor_spec)
        anchors = self._anchor_cache.get(key)
        if anchors is None:
            anchors = build_height_plane_anchors(self.select(geometry_config, selection_config), anchor_spec)
            self._anchor_cache[key] = anchors
        return anchors

    def triangle_map(
        self,
        geometry_config: WedgeGeometryConfig | None = None,
        selection_config: WedgeSelectionConfig | None = None,
        local: bool = True,
    ) -> TriangleWedgeMap:
        geometry_config = WedgeGeometryConfig() if geometry_config is None else geometry_config
        selection_config = WedgeSelectionConfig() if selection_config is None else selection_config
        key = (int(self._backend.edge_version()), geometry_config, selection_config, bool(local))
        triangle_map = self._triangle_map_cache.get(key)
        if triangle_map is None:
            n_triangles = self._backend.n_triangles()
            if n_triangles == 0:
                triangle_map = TriangleWedgeMap.empty()
            else:
                prim_id = dr.arange(wt.Int32, n_triangles)
                triangle_edges = self._backend.triangle_edge_indices(prim_id, global_=True)
                triangle_map = build_triangle_wedge_map(
                    self.select(geometry_config, selection_config),
                    triangle_edges,
                    n_triangles=n_triangles,
                    local=local,
                )
            self._triangle_map_cache[key] = triangle_map
        return triangle_map

    def pack(
        self,
        anchor_spec: HeightPlaneAnchorSpec,
        geometry_config: WedgeGeometryConfig | None = None,
        selection_config: WedgeSelectionConfig | None = None,
    ) -> WedgePack:
        geometry_config = WedgeGeometryConfig() if geometry_config is None else geometry_config
        selection_config = WedgeSelectionConfig() if selection_config is None else selection_config
        key = (int(self._backend.edge_version()), geometry_config, selection_config, anchor_spec)
        packed = self._pack_cache.get(key)
        if packed is None:
            packed = pack_wedges(
                self.select(geometry_config, selection_config),
                self.anchors(anchor_spec, geometry_config, selection_config),
            )
            self._pack_cache[key] = packed
        return packed


def get_scene_wedge_runtime(source) -> WedgeRuntime:
    runtime = RUNTIME_REGISTRY.get(source)
    if runtime is not None:
        return runtime

    backend = create_wedge_backend(source)
    runtime = WedgeRuntime(backend)
    RUNTIME_REGISTRY.set(source, runtime)
    return runtime
