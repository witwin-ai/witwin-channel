"""Regression coverage for multi-mesh shared-edge indexing in channel_scene."""

from __future__ import annotations

import numpy as np
import pytest
import witwin.channel as wt
from witwin.channel.core.scene import Scene as ChannelScene
from witwin.channel.core.scene.edge_policy import EdgePolicy
from witwin.core import Box, Material, Structure
pytestmark = pytest.mark.gpu


def _build_three_cube_scene() -> ChannelScene:
    centers = (
        (-2.5, -3.0, 1.5),
        (2.0, 0.5, 1.5),
        (-0.5, 3.5, 1.5),
    )
    material = Material(eps_r=4.0, sigma_e=0.0)
    structures = [
        Structure(
            name=f"cube_{index}",
            geometry=Box(position=center, size=(2.0, 2.0, 2.0), device="cuda"),
            material=material,
        )
        for index, center in enumerate(centers)
    ]
    return ChannelScene(
        structures=structures,
        device="cuda",
    )


def test_channel_scene_globalizes_multi_mesh_shared_vertices_for_surface_groups():
    scene = _build_three_cube_scene()
    edge_policy = EdgePolicy(edge_selection_mode="all_edges")
    assert scene.diffraction_edge_count(edge_policy=edge_policy) == 36
    geometry = scene._wedge_geometry
    selection = scene._wedge_selection
    surface_data = scene._triangle_surface_data

    assert geometry is not None
    assert selection is not None
    assert surface_data is not None

    shape_id = np.asarray(geometry.shape_id, dtype=np.int32)
    v0 = np.asarray(geometry.v0, dtype=np.int32)
    v1 = np.asarray(geometry.v1, dtype=np.int32)
    selected_mask = np.asarray(selection.selected_mask, dtype=bool)
    for mesh_index, expected_offset in enumerate((0, 8, 16)):
        mesh_mask = shape_id == mesh_index
        mesh_vertices = np.unique(np.concatenate([v0[mesh_mask], v1[mesh_mask]])).tolist()
        assert mesh_vertices == list(range(expected_offset, expected_offset + 8))
        assert int(selected_mask[mesh_mask].sum()) == 12

    group_id = np.asarray(surface_data["group_id"], dtype=np.int32)
    group_size = np.asarray(surface_data["group_size"], dtype=np.int32)
    surface_edge_size = np.asarray(surface_data["surface_edge_size"], dtype=np.int32)

    unique_groups, group_counts = np.unique(group_id, return_counts=True)
    assert unique_groups.size == 18
    assert np.array_equal(group_counts, np.full(18, 2, dtype=np.int32))
    assert np.array_equal(group_size, np.full(36, 2, dtype=np.int32))
    assert np.array_equal(surface_edge_size, np.full(36, 4, dtype=np.int32))
    assert int(surface_data["max_surface_edge_count"]) == 4


def test_adapted_scene_exposes_full_surface_edge_candidates_for_each_cube():
    adapted = _build_three_cube_scene()
    adapted.diffraction_edge_count(edge_policy=EdgePolicy(edge_selection_mode="all_edges"))

    for prim_idx in (wt.UInt32(0), wt.UInt32(12), wt.UInt32(24)):
        candidates = adapted.get_triangle_surface_edge_candidates(prim_idx)
        count = int(np.asarray(candidates["count"], dtype=np.int32).reshape(-1)[0])
        assert count == 4
