"""DrJit-backed mesh geometry."""

from __future__ import annotations

import drjit as dr
import torch
from witwin.channel import types as wt

from witwin.core import GeometryBase
from witwin.channel.core.geometry.mesh_buffers import faces_array, mesh_buffer_count, to_point3f, vertices_array


class Mesh(GeometryBase):
    """Explicit mesh geometry that preserves DrJit-backed vertex buffers."""

    kind = "mesh"

    def __init__(self, vertices: wt.Point3f | torch.Tensor, faces: wt.Vector3u | torch.Tensor) -> None:
        super().__init__(position=(0.0, 0.0, 0.0), rotation=None, device="cpu")
        self._vertices = self._validate_vertices(vertices)
        self._faces = self._validate_faces(faces)
        self._scene = None
        self._mesh_id = -1

    @staticmethod
    def _validate_vertices(vertices):
        if isinstance(vertices, wt.Point3f):
            if dr.width(vertices) == 0:
                raise ValueError("vertices must contain at least one vertex.")
            return vertices
        if isinstance(vertices, torch.Tensor):
            if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
                raise ValueError("vertices must have shape (N, 3) with N > 0.")
            return vertices.contiguous()
        vertices_array(vertices)
        return vertices

    @staticmethod
    def _validate_faces(faces):
        if isinstance(faces, wt.Vector3u):
            if dr.width(faces) == 0:
                raise ValueError("faces must contain at least one triangle.")
            return faces
        if isinstance(faces, torch.Tensor):
            if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
                raise ValueError("faces must have shape (M, 3) with M > 0.")
            return faces.to(dtype=torch.int32).contiguous()
        faces_array(faces)
        return faces

    def to_mesh(self, segments: int = 16, *, device: str | None = None) -> tuple:
        del segments
        vertices, faces = self._vertices, self._faces
        if device is not None and isinstance(vertices, torch.Tensor):
            vertices = vertices.to(device=device, dtype=torch.float32).contiguous()
        if device is not None and isinstance(faces, torch.Tensor):
            faces = faces.to(device=device, dtype=torch.int32).contiguous()
        return vertices, faces

    def update_vertices(self, vertices: wt.Point3f | torch.Tensor, sync: bool = True) -> None:
        """Update vertex positions. Pushes to RayD if attached to a scene."""
        vertices = self._validate_vertices(vertices)
        if mesh_buffer_count(vertices) != mesh_buffer_count(self._vertices):
            raise ValueError(
                f"Expected {mesh_buffer_count(self._vertices)} vertices, got {mesh_buffer_count(vertices)}."
            )
        self._vertices = vertices
        scene = self._scene
        if scene is not None and scene._rayd_scene is not None and self._mesh_id >= 0:
            scene._rayd_scene.update_mesh_vertices(self._mesh_id, to_point3f(vertices))
            scene._structure_meshes[self._mesh_id]["vertices"] = vertices
            if sync:
                scene.sync()
