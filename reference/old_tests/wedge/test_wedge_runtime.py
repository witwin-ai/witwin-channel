"""Smoke coverage for the backend-neutral wedge runtime."""

from __future__ import annotations

import drjit as dr
import pytest

from tests._scene_helpers import box_drjit_geometry, build_scene
from witwin.channel.scene.wedge import (
    HeightPlaneAnchorSpec,
    WedgeGeometryConfig,
    WedgeSelectionConfig,
    get_scene_wedge_runtime,
)


@pytest.mark.gpu
def test_wedge_runtime_compiles_geometry_pack_and_triangle_map():
    scene = build_scene(
        box_drjit_geometry(center=(0.0, 0.0, 1.5), size=2.0),
        device="cuda",
        edge_selection_mode="vertical_only",
        boundary_edge_policy="exclude",
    )

    runtime = get_scene_wedge_runtime(scene)
    geometry = runtime.geometry(WedgeGeometryConfig(boundary_policy="exclude"))
    selection = runtime.select(
        WedgeGeometryConfig(boundary_policy="exclude"),
        WedgeSelectionConfig(mode="vertical_only", vertical_ratio=0.7),
    )
    anchors = runtime.anchors(
        HeightPlaneAnchorSpec(z=1.5),
        WedgeGeometryConfig(boundary_policy="exclude"),
        WedgeSelectionConfig(mode="vertical_only", vertical_ratio=0.7),
    )
    packed = runtime.pack(
        HeightPlaneAnchorSpec(z=1.5),
        WedgeGeometryConfig(boundary_policy="exclude"),
        WedgeSelectionConfig(mode="vertical_only", vertical_ratio=0.7),
    )
    triangle_map = runtime.triangle_map(
        WedgeGeometryConfig(boundary_policy="exclude"),
        WedgeSelectionConfig(mode="vertical_only", vertical_ratio=0.7),
        local=True,
    )

    assert geometry.n_edges > 0
    assert selection.size() > 0
    assert anchors.size() == packed.n_wedges
    assert triangle_map.n_triangles == int(dr.width(scene.faces.x))
    assert int(dr.width(packed.global_idx)) == packed.n_wedges
    assert int(dr.width(packed.line_min)) == packed.n_wedges
    assert int(dr.width(packed.line_max)) == packed.n_wedges
    assert float(dr.max(dr.abs((packed.line_max - packed.line_min) - packed.length))[0]) < 1.0e-6


@pytest.mark.gpu
def test_height_plane_finite_wedge_keeps_all_selected_wedges():
    scene = build_scene(
        box_drjit_geometry(center=(0.0, 0.0, 1.5), size=2.0),
        device="cuda",
        edge_selection_mode="vertical_only",
        boundary_edge_policy="exclude",
    )

    runtime = get_scene_wedge_runtime(scene)
    geometry_cfg = WedgeGeometryConfig(boundary_policy="exclude")
    selection_cfg = WedgeSelectionConfig(mode="vertical_only", vertical_ratio=0.7)
    selection = runtime.select(geometry_cfg, selection_cfg)
    anchors = runtime.anchors(
        HeightPlaneAnchorSpec(z=1.5),
        geometry_cfg,
        selection_cfg,
    )

    assert selection.size() > 0
    assert anchors.size() == selection.size()
    assert anchors.summary["input_selected_wedges"] == selection.size()
    assert anchors.summary["output_anchors"] == selection.size()
    assert anchors.summary["excluded_clamped_anchors"] == 0
