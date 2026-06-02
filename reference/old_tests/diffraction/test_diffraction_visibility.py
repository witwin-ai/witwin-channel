"""Regression tests for hard diffraction visibility gating."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

import drjit as dr

from tests._scene_helpers import box_geometry, build_scene
from witwin.channel.trace.diffraction import (
    _build_tx_first_order_state_arrays,
    _point_inside_closed_mesh_mask,
    _segment_visibility_mask,
)


def build_cube_scene():
    cube = box_geometry(center=(0.0, 0.0, 1.5), size=2.0)
    return build_scene(cube)


def test_segment_visibility_blocks_through_obstacle():
    scene = build_cube_scene()
    start = wt.Point3f(wt.Float(-3.0), wt.Float(0.0), wt.Float(1.5))
    end = wt.Point3f(wt.Float(3.0), wt.Float(0.0), wt.Float(1.5))
    visible = _segment_visibility_mask(start, end, scene)
    assert not bool(visible[0]), "Expected cube to block segment that passes through its interior"


def test_segment_visibility_keeps_clear_path():
    scene = build_cube_scene()
    start = wt.Point3f(wt.Float(-3.0), wt.Float(3.0), wt.Float(1.5))
    end = wt.Point3f(wt.Float(3.0), wt.Float(3.0), wt.Float(1.5))
    visible = _segment_visibility_mask(start, end, scene)
    assert bool(visible[0]), "Expected clear segment outside cube footprint to remain visible"


def test_point_inside_closed_mesh_mask_detects_cube_interior():
    scene = build_cube_scene()
    probe_points = wt.Point3f(
        wt.Float([0.0, 3.0]),
        wt.Float([0.0, 0.0]),
        wt.Float([1.5, 1.5]),
    )
    inside = _point_inside_closed_mesh_mask(probe_points, scene)

    assert bool(inside[0]), "Expected cube center to be classified as inside the closed mesh"
    assert not bool(inside[1]), "Expected exterior probe point to stay outside"


def test_first_order_states_keep_visible_exterior_edges():
    scene = build_cube_scene()
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    wavelength = 299792458.0 / 1e9
    k = 2.0 * dr.pi / wavelength
    tx = wt.Point3f(-4.0, 0.0, 1.5)

    unfiltered = _build_tx_first_order_state_arrays(tx, edge_data, wavelength, k, scene=None)
    filtered = _build_tx_first_order_state_arrays(tx, edge_data, wavelength, k, scene=scene)

    assert unfiltered["n_states"] == 2
    assert filtered["n_states"] == unfiltered["n_states"]


