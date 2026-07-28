from __future__ import annotations

import torch

from witwin.channel.scene.resources import refine_edge_geometry
from witwin.channel.scene.resources import RayDSceneResource
from witwin.channel.kernels import geometry as geometry_kernels

_RAYD_EDGE_INFO_PLANE_TOL = 1.34e-5


def diffraction_edge_geometry(records: object) -> tuple[torch.Tensor, ...]:
    return geometry_kernels.mc_diffraction_edge_geometry(
        records.vertices,
        records.faces,
        records.face_normals,
        records.edge_v0,
        records.edge_v1,
        records.face0,
        records.face1,
        plane_tol=_RAYD_EDGE_INFO_PLANE_TOL,
    )


def cached_diffraction_edge_geometry(
    rayd: RayDSceneResource,
    *,
    preserve_imported_edges: bool = False,
) -> tuple[torch.Tensor, ...]:
    cache = rayd.runtime_cache
    cache_key = (
        "mc_imported_diffraction_edge_geometry"
        if preserve_imported_edges
        else "mc_diffraction_edge_geometry"
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    geometry = diffraction_edge_geometry(rayd.edge_records())
    if not preserve_imported_edges:
        geometry = refine_edge_geometry(rayd, geometry)
    cache[cache_key] = geometry
    return geometry
