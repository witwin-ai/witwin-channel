from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import drjit as dr
import witwin.channel as wt

from witwin.channel.core.scene.wedge import WedgeConfig, WedgeOps, WedgeSelection


def _tolist(value) -> list:
    return np.asarray(value).reshape(-1).tolist()


def _sample_edge_inputs():
    edge_info = SimpleNamespace(
        start=wt.Point3f(wt.Float([0.0, 0.0]), wt.Float([0.0, 0.0]), wt.Float([0.0, 0.0])),
        end=wt.Point3f(wt.Float([0.0, 2.0]), wt.Float([0.0, 0.0]), wt.Float([2.0, 0.0])),
        edge=wt.Vector3f(wt.Float([0.0, 2.0]), wt.Float([0.0, 0.0]), wt.Float([2.0, 0.0])),
        length=wt.Float([2.0, 2.0]),
        normal0=wt.Vector3f(wt.Float([1.0, 0.0]), wt.Float([0.0, 1.0]), wt.Float([0.0, 0.0])),
        normal1=wt.Vector3f(wt.Float([0.0, -1.0]), wt.Float([1.0, 0.0]), wt.Float([0.0, 0.0])),
        is_boundary=wt.Bool([True, False]),
        shape_id=wt.Int32([0, 0]),
        local_edge_id=wt.Int32([0, 1]),
        global_edge_id=wt.Int32([10, 11]),
    )
    edge_topology = SimpleNamespace(
        face0_global=wt.Int32([0, 1]),
        face1_global=wt.Int32([-1, 2]),
        v0_global=wt.Int32([0, 1]),
        v1_global=wt.Int32([2, 3]),
    )
    return edge_info, edge_topology


def test_wedge_ops_build_geometry_select_and_pack_boundary_edges() -> None:
    edge_info, edge_topology = _sample_edge_inputs()
    config = WedgeConfig(boundary_policy="half_plane", vertical_only=True, vertical_ratio=0.7)

    geometry = WedgeOps.build_geometry(edge_info, edge_topology, config)
    selection = WedgeOps.select(geometry, config)
    midpoint = WedgeOps.build_midpoint_anchors(selection)
    anchors = WedgeOps.build_anchors(selection, 1.0)
    tri_map = WedgeOps.build_triangle_map(selection, 3)
    pack = WedgeOps.pack(selection, anchors)

    assert geometry.n_edges == 2
    assert _tolist(geometry.is_valid) == [True, True]
    assert selection.size() == 1
    assert _tolist(selection.selected_idx) == [0]
    assert _tolist(midpoint.anchor_t) == [0.5]
    assert _tolist(anchors.anchor_t) == [0.5]
    assert _tolist(pack.pos.z) == [1.0]
    assert _tolist(pack.adjacent_face0) == [0]
    assert _tolist(pack.adjacent_face1) == [-1]
    assert _tolist(pack.global_idx) == [10]
    assert _tolist(pack.is_boundary) == [True]
    assert _tolist(pack.line_min) == [-1.0]
    assert _tolist(pack.line_max) == [1.0]
    assert _tolist(tri_map.edge0) == [0, -1, -1]
    assert _tolist(tri_map.edge1) == [-1, -1, -1]
    assert _tolist(tri_map.edge2) == [-1, -1, -1]


def test_wedge_ops_build_triangle_map_returns_empty_slots_for_empty_selection() -> None:
    edge_info, edge_topology = _sample_edge_inputs()
    config = WedgeConfig(boundary_policy="exclude", vertical_only=False)
    geometry = WedgeOps.build_geometry(edge_info, edge_topology, config)
    empty_selection = WedgeSelection(
        geometry=geometry,
        selected_idx=dr.zeros(wt.UInt32, 0),
        selected_mask=dr.zeros(wt.Bool, geometry.n_edges),
    )

    tri_map = WedgeOps.build_triangle_map(empty_selection, 4)

    assert tri_map.n_triangles == 4
    assert tri_map.n_wedges == 0
    assert _tolist(tri_map.edge0) == [-1, -1, -1, -1]
    assert _tolist(tri_map.edge1) == [-1, -1, -1, -1]
    assert _tolist(tri_map.edge2) == [-1, -1, -1, -1]
