"""Regression coverage for the RayD-shaped wedge adapter."""

from __future__ import annotations

from types import SimpleNamespace

import witwin as wt
import pytest
import torch

from tests._scene_helpers import box_drjit_geometry, build_scene
from witwin.channel.scene.wedge import HeightPlaneAnchorSpec, WedgeGeometryConfig, WedgeSelectionConfig, get_scene_wedge_runtime
from witwin.channel.scene.wedge.adapters import RayDSceneAdapter, is_rayd_scene_like


def _device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the RayD-shaped wedge adapter test.")
    return torch.device("cuda")


class _FakeRayDScene:
    def __init__(self, device: torch.device):
        self.version = 3
        self.edge_version = 7
        self._device = device

        start = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        end = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        edge = end - start
        normal0 = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        normal1 = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=torch.float32)

        self._edge_info = SimpleNamespace(
            start=start,
            edge=edge,
            end=end,
            length=torch.tensor([1.0], device=device, dtype=torch.float32),
            normal0=normal0,
            normal1=normal1,
            is_boundary=torch.tensor([False], device=device, dtype=torch.bool),
            shape_id=torch.tensor([0], device=device, dtype=torch.int32),
            local_edge_id=torch.tensor([0], device=device, dtype=torch.int32),
            global_edge_id=torch.tensor([0], device=device, dtype=torch.int32),
        )
        self._edge_topology = SimpleNamespace(
            v0=torch.tensor([0], device=device, dtype=torch.int32),
            v1=torch.tensor([1], device=device, dtype=torch.int32),
            face0_local=torch.tensor([0], device=device, dtype=torch.int32),
            face1_local=torch.tensor([1], device=device, dtype=torch.int32),
            face0_global=torch.tensor([0], device=device, dtype=torch.int32),
            face1_global=torch.tensor([1], device=device, dtype=torch.int32),
            opposite_vertex0=torch.tensor([2], device=device, dtype=torch.int32),
            opposite_vertex1=torch.tensor([3], device=device, dtype=torch.int32),
        )
        self._mesh_face_offsets = torch.tensor([0, 2], device=device, dtype=torch.int32)
        self._mesh_edge_offsets = torch.tensor([0, 1], device=device, dtype=torch.int32)

    def edge_info(self):
        return self._edge_info

    def edge_topology(self):
        return self._edge_topology

    def triangle_edge_indices(self, prim_id, global_=True):
        del global_
        prim_id = prim_id.to(device=self._device, dtype=torch.int32).reshape(-1)
        edge0 = torch.where(prim_id >= 0, torch.zeros_like(prim_id), torch.full_like(prim_id, -1))
        edge1 = torch.full_like(prim_id, -1)
        edge2 = torch.full_like(prim_id, -1)
        return edge0, edge1, edge2

    def edge_adjacent_faces(self, edge_id, global_=True):
        del global_
        edge_id = edge_id.to(device=self._device, dtype=torch.int32).reshape(-1)
        face0 = torch.where(edge_id >= 0, torch.zeros_like(edge_id), torch.full_like(edge_id, -1))
        face1 = torch.where(edge_id >= 0, torch.ones_like(edge_id), torch.full_like(edge_id, -1))
        return face0, face1

    def mesh_face_offsets(self):
        return self._mesh_face_offsets

    def mesh_edge_offsets(self):
        return self._mesh_edge_offsets

    def intersect(self, ray, active=True):
        return {"ray": ray, "active": active}

    def shadow_test(self, ray, active=True):
        return {"ray": ray, "active": active}


@pytest.mark.gpu
def test_rayd_adapter_builds_wedge_geometry_and_pack():
    scene = _FakeRayDScene(_device())

    assert is_rayd_scene_like(scene)

    runtime = get_scene_wedge_runtime(scene)
    assert isinstance(runtime._backend, RayDSceneAdapter)

    geometry_cfg = WedgeGeometryConfig(boundary_policy="exclude")
    selection_cfg = WedgeSelectionConfig(mode="vertical_only", vertical_ratio=0.7)
    anchor_spec = HeightPlaneAnchorSpec(z=0.5)

    geometry = runtime.geometry(geometry_cfg)
    selection = runtime.select(geometry_cfg, selection_cfg)
    anchors = runtime.anchors(anchor_spec, geometry_cfg, selection_cfg)
    packed = runtime.pack(anchor_spec, geometry_cfg, selection_cfg)
    triangle_map = runtime.triangle_map(geometry_cfg, selection_cfg, local=True)

    assert geometry.n_edges == 1
    assert selection.size() == 1
    assert anchors.size() == 1
    assert packed.n_wedges == 1
    assert float(packed.line_min[0]) < 0.0
    assert float(packed.line_max[0]) > 0.0
    assert abs(float((packed.line_max - packed.line_min)[0]) - float(packed.length[0])) < 1.0e-6
    assert triangle_map.n_triangles == 2
    assert triangle_map.n_wedges == 1


@pytest.mark.gpu
def test_channel_scene_can_route_wedge_runtime_through_rayd_backend():
    scene = build_scene(
        box_drjit_geometry(center=(0.0, 0.0, 1.5), size=2.0),
        device="cuda",
        edge_selection_mode="vertical_only",
        boundary_edge_policy="exclude",
    )

    runtime = get_scene_wedge_runtime(scene)
    edge_cache = scene.get_edge_data(1.5)

    assert scene._wedge_backend_kind == "rayd"
    assert scene._wedge_backend_source is scene._rayd_scene
    assert isinstance(runtime._backend, RayDSceneAdapter)
    assert edge_cache["edge_data"] is not None
    assert edge_cache["edge_data"]["n_edges"] > 0
    assert len(edge_cache["diffraction_points"]) == edge_cache["edge_data"]["n_edges"]

