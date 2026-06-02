"""Regression tests for explicit boundary/open-edge diffraction policy."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

import drjit as dr

from witwin.channel import FieldMonitor, Tracer
from witwin.channel.scene import Scene
from witwin.channel.utils import scalar
from tests._scene_helpers import build_scene
def _build_open_wall_mesh():
    vertices = wt.Point3f(
        wt.Float(-1.0, 1.0, -1.0, 1.0),
        wt.Float(0.0, 0.0, 0.0, 0.0),
        wt.Float(0.0, 0.0, 3.0, 3.0),
    )
    faces = wt.Vector3u(
        wt.UInt32(0, 0),
        wt.UInt32(1, 3),
        wt.UInt32(3, 2),
    )
    return vertices, faces


def _build_non_sibling_open_wall_mesh():
    vertices = wt.Point3f(
        wt.Float(-1.0, 1.0, -1.0, 1.0, 4.0, 5.0, 4.5),
        wt.Float(0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 4.0),
        wt.Float(0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0),
    )
    faces = wt.Vector3u(
        wt.UInt32(0, 4, 0),
        wt.UInt32(1, 5, 3),
        wt.UInt32(3, 6, 2),
    )
    return vertices, faces


def test_boundary_edges_are_excluded_by_default():
    vertices, faces = _build_open_wall_mesh()
    scene = build_scene((vertices, faces))

    edge_cache = scene.get_edge_data(1.5)

    assert scene.boundary_edge_policy == "exclude"
    assert scene.edge_selection_summary["boundary_vertical_edges"] == 2
    assert scene.edge_selection_summary["included_boundary_edges"] == 0
    assert scene.edge_selection_summary["excluded_boundary_edges"] == 2
    assert edge_cache["edge_data"] is None
    assert scene.get_global_diffraction_edge_indices() == ()
    assert scene.get_adjacent_diffraction_edge_indices_for_triangle(0) == ()


def test_half_plane_policy_includes_boundary_edges_consistently():
    vertices, faces = _build_open_wall_mesh()
    scene = build_scene((vertices, faces), boundary_edge_policy="half_plane")

    edge_cache = scene.get_edge_data(1.5)
    edge_data = edge_cache["edge_data"]

    assert scene.edge_selection_summary["boundary_vertical_edges"] == 2
    assert scene.edge_selection_summary["included_boundary_edges"] == 2
    assert scene.edge_selection_summary["excluded_boundary_edges"] == 0
    assert edge_cache["boundary_edge_policy"] == "half_plane"
    assert edge_data is not None
    assert edge_data["n_edges"] == 2
    assert len(scene.get_global_diffraction_edge_indices()) == 2
    assert len(scene.get_adjacent_diffraction_edge_indices_for_triangle(0)) == 2
    assert scalar(dr.max(dr.abs(edge_data["wedge_n"] - 2.0))) < 1e-6


def test_tracer_metadata_reports_boundary_edge_policy():
    vertices, faces = _build_open_wall_mesh()
    scene = build_scene((vertices, faces), boundary_edge_policy="half_plane")
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    )
    monitor = FieldMonitor(
        "open_wall_plane",
        axis="z",
        position=1.5,
        bounds=((-3.0, 3.0), (-3.0, 3.0)),
        grid_size=12,
    )

    result = tracer.trace(
        tx_pos=wt.Point3f(0.0, -4.0, 1.5),
        monitor=monitor,
        verbose=False,
    )

    metadata = result.primary.metadata

    assert metadata["boundary_edge_policy"] == "half_plane"
    assert metadata["edge_selection_summary"]["included_boundary_edges"] == 2
    assert metadata["edge_selection_summary"]["excluded_boundary_edges"] == 0


def test_surface_group_edges_do_not_depend_on_triangle_sibling_order():
    vertices, faces = _build_non_sibling_open_wall_mesh()
    scene = build_scene((vertices, faces), boundary_edge_policy="half_plane")

    adjacent_edges = scene.get_adjacent_diffraction_edge_indices_for_triangle(0)

    assert len(adjacent_edges) == 2
    assert adjacent_edges == scene.get_adjacent_diffraction_edge_indices_for_triangle(2)


def test_update_vertices_with_lazy_edge_refresh_keeps_edge_runtime_consistent():
    vertices, faces = _build_open_wall_mesh()
    scene = build_scene((vertices, faces), boundary_edge_policy="half_plane")

    scene.update_vertices(
        wt.Point3f(vertices.x + 0.25, vertices.y, vertices.z),
        recompute_edges=False,
    )

    assert scene._edge_runtime_dirty is True
    edge_cache = scene.get_edge_data(1.5)
    assert scene._edge_runtime_dirty is False
    assert edge_cache["edge_data"] is not None


