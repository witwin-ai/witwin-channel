from __future__ import annotations

import witwin as wt

from ..utils.drjit_ops import ArrayInit
from .builder import empty_edge_selection_summary, rebuild_runtime, update_vertices
from .runtime_queries import (
    get_adjacent_diffraction_edge_indices_for_triangle,
    get_edge_data,
    get_global_diffraction_edge_indices,
    get_triangle_surface_edge_candidates,
)


class SceneRuntime:
    def __init__(self, scene):
        self.scene = scene
        self.vertices = ArrayInit.empty_point3()
        self.faces = ArrayInit.empty_vector3u()
        self._wedge_backend_source = None
        self._wedge_backend_kind = "rayd"
        self._wedge_backend_error = None
        self._rayd_scene = None
        self._n_verts = 0
        self._mesh_center_3d = wt.Point3f(0.0, 0.0, 0.0)
        self._mesh_center_2d = wt.Vector2f(0.0, 0.0)
        self._edge_cache = {}
        self._edge_topology = []
        self._runtime_vertex_override = None
        self._global_diffraction_edge_indices = []
        self.vertical_edges = []
        self._vertical_edge_id_to_index = {}
        self.tri_data_gpu = None
        self._tri_edge_indices = None
        self._diffraction_edge_gpu = None
        self._triangle_surface_groups = ()
        self._triangle_surface_group_by_triangle = ()
        self._triangle_surface_edge_groups = ()
        self._triangle_surface_data = None
        self._triangle_material_data = None
        self._face_normals = None
        self._edge_runtime_dirty = False
        self._mesh_version = 0
        self.edge_selection_summary = empty_edge_selection_summary()

    def __getattr__(self, name: str):
        return getattr(self.scene, name)

    def rebuild(self):
        rebuild_runtime(self)

    def update_vertices(self, vertices, recompute_edges: bool = True):
        update_vertices(self, vertices, recompute_edges=recompute_edges)

    def get_global_diffraction_edge_indices(self):
        return get_global_diffraction_edge_indices(self)

    def get_adjacent_diffraction_edge_indices_for_triangle(
        self,
        prim_idx: int,
        *,
        include_sibling: bool = True,
    ):
        return get_adjacent_diffraction_edge_indices_for_triangle(
            self,
            prim_idx,
            include_sibling=include_sibling,
        )

    def get_triangle_surface_edge_candidates(self, prim_idx):
        return get_triangle_surface_edge_candidates(self, prim_idx)

    def get_edge_data(self, calculation_height, include_projection: bool = True):
        return get_edge_data(self, calculation_height, include_projection=include_projection)


__all__ = ["SceneRuntime"]
