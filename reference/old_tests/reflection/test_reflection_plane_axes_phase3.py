"""Phase 3 regression coverage for x/y-normal reflection accumulation."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import drjit as dr
import numpy as np
import pytest
import witwin as wt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._scene_helpers import box_geometry, build_scene
from witwin.channel import DrJitMesh, Field
from witwin.channel.utils import to_point3f, to_vector3u
from witwin.channel.trace import compute_reflection_field
from witwin.channel.monitors.field.grid_reflection import prepare_plane_intersections
FREQUENCY = 1e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * math.pi / WAVELENGTH
GRID_SIZE = (25, 25)
BOUNDS = ((-6.0, 6.0), (-6.0, 6.0))
TX_BASE = (-1.0, -5.0, 4.5)
CEILING_CENTER = (0.0, 0.0, 6.0)
CEILING_SIZE = (8.0, 10.0, 0.25)
TX_POLARIZATION_X = (1.0, 0.0, 0.0)
TX_POLARIZATION_Y = (0.0, 1.0, 0.0)


def _rotate_mesh_y(mesh_geometry, degrees: float) -> DrJitMesh:
    vertices, faces = mesh_geometry.to_mesh()
    transform = wt.Transform4f().rotate(wt.Vector3f(0.0, 1.0, 0.0), wt.Float(degrees))
    return DrJitMesh(transform @ to_point3f(vertices), to_vector3u(faces))


def _rotate_mesh_x(mesh_geometry, degrees: float) -> DrJitMesh:
    vertices, faces = mesh_geometry.to_mesh()
    transform = wt.Transform4f().rotate(wt.Vector3f(1.0, 0.0, 0.0), wt.Float(degrees))
    return DrJitMesh(transform @ to_point3f(vertices), to_vector3u(faces))


def _rotate_point_y_90(point):
    x, y, z = point
    return (z, y, -x)


def _rotate_point_x_neg_90(point):
    x, y, z = point
    return (x, z, -y)


def _vector_magnitude_matrix(detail, field: Field):
    total = np.zeros((field.size[1], field.size[0]), dtype=np.float64)
    for axis_name in ("x", "y", "z"):
        component = detail["polarization_field_total"][axis_name]
        magnitude = (
            np.asarray(component.real, dtype=np.float64) ** 2
            + np.asarray(component.imag, dtype=np.float64) ** 2
        ).reshape(field.size[1], field.size[0])
        total += magnitude
    return np.sqrt(total)


def _assert_vector_magnitude_rms_close(actual, expected, *, max_rms_db):
    assert actual.shape == expected.shape
    mask = (expected > expected.max() * 1e-1) | (actual > actual.max() * 1e-1)
    actual_db = 20.0 * np.log10(np.maximum(actual, 1e-20))
    expected_db = 20.0 * np.log10(np.maximum(expected, 1e-20))
    rms_db = float(np.sqrt(np.mean((actual_db[mask] - expected_db[mask]) ** 2)))
    assert rms_db < max_rms_db


def _run_single_reflection(scene, field: Field, tx_pos, *, tx_polarization, n_rays):
    coords = field.get_coordinates()
    _, _, detail = compute_reflection_field(
        grid=field,
        rx_z=field.position,
        tx_pos=wt.Point3f(*tx_pos),
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=n_rays,
        max_reflections=1,
        mode="3d",
        reflection_coef=0.82,
        tx_polarization=tx_polarization,
        use_scene_materials=False,
        grid_data=coords,
    )
    return detail


@pytest.mark.gpu
def test_x_normal_reflection_matches_rotated_z_normal_reference():
    base_scene = build_scene(box_geometry(center=CEILING_CENTER, size=CEILING_SIZE))
    base_field = Field(bounds=BOUNDS, size=(16, 16), axis="z", position=1.5)
    base_detail = _run_single_reflection(
        base_scene,
        base_field,
        TX_BASE,
        tx_polarization=TX_POLARIZATION_Y,
        n_rays=16384,
    )

    rotated_scene = build_scene(_rotate_mesh_y(box_geometry(center=CEILING_CENTER, size=CEILING_SIZE), 90.0))
    rotated_field = Field(
        bounds=(BOUNDS[1], BOUNDS[0]),
        size=(16, 16),
        axis="x",
        position=1.5,
    )
    rotated_detail = _run_single_reflection(
        rotated_scene,
        rotated_field,
        _rotate_point_y_90(TX_BASE),
        tx_polarization=TX_POLARIZATION_Y,
        n_rays=65536,
    )

    assert rotated_detail["dda_stats"]["backend"] == "ray_plane_scatter"
    assert rotated_detail["source_paths_per_bounce"][0]["n_paths"] > 0

    base_magnitude = _vector_magnitude_matrix(base_detail, base_field)
    rotated_magnitude = _vector_magnitude_matrix(rotated_detail, rotated_field)
    _assert_vector_magnitude_rms_close(
        rotated_magnitude,
        base_magnitude[:, ::-1].T,
        max_rms_db=0.5,
    )


@pytest.mark.gpu
def test_y_normal_reflection_matches_rotated_z_normal_reference():
    base_scene = build_scene(box_geometry(center=CEILING_CENTER, size=CEILING_SIZE))
    base_field = Field(bounds=BOUNDS, size=(16, 16), axis="z", position=1.5)
    base_detail = _run_single_reflection(
        base_scene,
        base_field,
        TX_BASE,
        tx_polarization=TX_POLARIZATION_X,
        n_rays=16384,
    )

    rotated_scene = build_scene(_rotate_mesh_x(box_geometry(center=CEILING_CENTER, size=CEILING_SIZE), -90.0))
    rotated_field = Field(
        bounds=BOUNDS,
        size=(16, 16),
        axis="y",
        position=1.5,
    )
    rotated_detail = _run_single_reflection(
        rotated_scene,
        rotated_field,
        _rotate_point_x_neg_90(TX_BASE),
        tx_polarization=TX_POLARIZATION_X,
        n_rays=65536,
    )

    assert rotated_detail["dda_stats"]["backend"] == "ray_plane_scatter"
    assert rotated_detail["source_paths_per_bounce"][0]["n_paths"] > 0

    base_magnitude = _vector_magnitude_matrix(base_detail, base_field)
    rotated_magnitude = _vector_magnitude_matrix(rotated_detail, rotated_field)
    _assert_vector_magnitude_rms_close(
        rotated_magnitude,
        base_magnitude[::-1, :],
        max_rms_db=0.5,
    )


def test_prepare_plane_intersections_handles_parallel_and_away_rays():
    field = Field(bounds=((-2.0, 2.0), (-2.0, 2.0)), size=(8, 8), axis="x", position=1.0)
    ray_origin = wt.Point3f(
        wt.Float([0.0, 2.0, 0.0]),
        wt.Float([0.0, 0.0, 0.0]),
        wt.Float([0.0, 0.0, 0.0]),
    )
    ray_dir = wt.Vector3f(
        wt.Float([0.0, 1.0, 1.0]),
        wt.Float([1.0, 0.0, 0.5]),
        wt.Float([0.0, 0.0, 0.25]),
    )
    intersections = prepare_plane_intersections(
        grid=field,
        ray_origin=ray_origin,
        ray_dir=ray_dir,
        active=dr.full(wt.Bool, True, 3),
        blocker_dist=wt.Float([10.0, 10.0, 10.0]),
        plane_position=field.position,
    )

    parallel = np.asarray(intersections["parallel"], dtype=bool)
    away = np.asarray(intersections["points_away"], dtype=bool)
    valid = np.asarray(intersections["valid"], dtype=bool)

    np.testing.assert_array_equal(parallel, np.array([True, False, False]))
    np.testing.assert_array_equal(away, np.array([False, True, False]))
    np.testing.assert_array_equal(valid, np.array([False, False, True]))
