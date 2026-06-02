"""Phase 5 regression coverage for 3D reflection sampling and scatter optimization."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import drjit as dr
import numpy as np
import pytest
import witwin as wt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin.channel.monitors.field.grid_reflection as scatter_module
from tests._scene_helpers import box_geometry, build_scene
from witwin.channel import DrJitMesh, Field, FieldMonitor, Tracer
from witwin.channel.utils.raygen import generate_cone_directions, generate_hemisphere_directions
from witwin.channel.utils import to_point3f, to_vector3u
from witwin.channel.trace import compute_reflection_field
FREQUENCY = 1e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * math.pi / WAVELENGTH
TRACE_BOUNDS = ((-6.0, 6.0), (-2.0, 8.0))


def _build_wall_scene():
    wall = box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0))
    return build_scene(wall)


def _rotate_mesh_y(mesh_geometry, degrees: float) -> DrJitMesh:
    vertices, faces = mesh_geometry.to_mesh()
    transform = wt.Transform4f().rotate(wt.Vector3f(0.0, 1.0, 0.0), wt.Float(degrees))
    return DrJitMesh(transform @ to_point3f(vertices), to_vector3u(faces))


def _rotate_point_y_90(point):
    x, y, z = point
    return (z, y, -x)


def _field_magnitude(field):
    real = np.asarray(field.real, dtype=np.float64)
    imag = np.asarray(field.imag, dtype=np.float64)
    return np.sqrt(real * real + imag * imag)


@pytest.mark.gpu
def test_phase5_direction_generators_respect_requested_domains():
    hemisphere_normal = wt.Vector3f(0.0, 0.0, 1.0)
    hemisphere_dirs = generate_hemisphere_directions(2048, hemisphere_normal)
    hemisphere_dot = np.asarray(dr.dot(hemisphere_dirs, hemisphere_normal), dtype=np.float64)
    hemisphere_norm = np.asarray(dr.norm(hemisphere_dirs), dtype=np.float64)

    cone_axis = wt.Vector3f(0.0, 1.0, 1.0)
    cone_axis = cone_axis / dr.norm(cone_axis)
    half_angle = math.radians(18.0)
    cone_dirs = generate_cone_directions(2048, cone_axis, half_angle)
    cone_dot = np.asarray(dr.dot(cone_dirs, cone_axis), dtype=np.float64)
    cone_norm = np.asarray(dr.norm(cone_dirs), dtype=np.float64)

    assert hemisphere_dot.min() >= -1e-6
    np.testing.assert_allclose(hemisphere_norm, 1.0, atol=1e-6)
    assert cone_dot.min() >= math.cos(half_angle) - 1e-6
    np.testing.assert_allclose(cone_norm, 1.0, atol=1e-6)


@pytest.mark.gpu
def test_plane_monitor_suggested_ray_mode_and_auto_sampling_metadata():
    monitor = FieldMonitor(
        "phase5_monitor",
        axis="z",
        position=1.5,
        bounds=TRACE_BOUNDS,
        grid_size=(24, 20),
        ray_mode="3d",
        ray_sampling="auto",
    )
    field = monitor.to_field(WAVELENGTH)

    assert monitor.suggested_ray_mode(wt.Point3f(0.0, 0.0, 1.6)) == "2d"
    assert monitor.suggested_ray_mode(wt.Point3f(0.0, 0.0, 5.0)) == "3d"

    _, _, near_detail = compute_reflection_field(
        grid=field,
        rx_z=monitor.position,
        tx_pos=wt.Point3f(0.0, 0.0, 1.6),
        scene=None,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=0,
        max_reflections=1,
        mode="3d",
        ray_sampling="auto",
    )
    _, _, far_detail = compute_reflection_field(
        grid=field,
        rx_z=monitor.position,
        tx_pos=wt.Point3f(0.0, 0.0, 5.0),
        scene=None,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=0,
        max_reflections=1,
        mode="3d",
        ray_sampling="auto",
    )

    assert near_detail["dda_stats"]["selected_ray_sampling"] == "full_sphere"
    assert far_detail["dda_stats"]["selected_ray_sampling"] == "hemisphere_facing_monitor"


@pytest.mark.gpu
def test_tracer_reports_explicit_hemisphere_sampling_metadata():
    scene = _build_wall_scene()
    monitor = FieldMonitor(
        "phase5_trace_monitor",
        axis="z",
        position=1.5,
        bounds=TRACE_BOUNDS,
        grid_size=(24, 20),
        ray_mode="3d",
        ray_sampling="hemisphere",
    )
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        max_diffractions=0,
    )

    result = tracer.trace(wt.Point3f(-3.0, -5.0, 1.5), monitor=monitor, verbose=False)
    sampling = result.primary.metadata["reflection_sampling"]

    assert sampling["requested_ray_sampling"] == "hemisphere"
    assert sampling["selected_ray_sampling"] == "hemisphere_facing_monitor"
    assert sampling["recommended_ray_count_multiplier_vs_2d"] == pytest.approx(5.0)
    assert result.primary.metadata["receiver_sampling"]["ray_sampling"] == "hemisphere"


@pytest.mark.gpu
def test_hemisphere_sampling_matches_full_sphere_on_single_wall_scene():
    scene = _build_wall_scene()
    field = Field(bounds=TRACE_BOUNDS, size=(24, 20), axis="z", position=1.5)
    coords = field.get_coordinates()
    tx = wt.Point3f(-3.0, -5.0, 1.5)

    full_sphere, _, full_detail = compute_reflection_field(
        grid=field,
        rx_z=field.position,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=16384,
        max_reflections=1,
        mode="3d",
        ray_sampling="full_sphere",
        reflection_coef=0.8,
        grid_data=coords,
    )
    hemisphere, _, hemisphere_detail = compute_reflection_field(
        grid=field,
        rx_z=field.position,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=8192,
        max_reflections=1,
        mode="3d",
        ray_sampling="hemisphere",
        reflection_coef=0.8,
        grid_data=coords,
    )

    full_mag = _field_magnitude(full_sphere)
    hemisphere_mag = _field_magnitude(hemisphere)
    mask = (full_mag > full_mag.max() * 1e-3) | (hemisphere_mag > hemisphere_mag.max() * 1e-3)
    full_db = 20.0 * np.log10(np.maximum(full_mag, 1e-20))
    hemisphere_db = 20.0 * np.log10(np.maximum(hemisphere_mag, 1e-20))
    rms_db = float(np.sqrt(np.mean((full_db[mask] - hemisphere_db[mask]) ** 2)))

    assert full_detail["dda_stats"]["selected_ray_sampling"] == "full_sphere"
    assert hemisphere_detail["dda_stats"]["selected_ray_sampling"] == "hemisphere_facing_monitor"
    assert rms_db < 0.5


@pytest.mark.gpu
def test_chunked_scatter_matches_unchunked_x_normal_reflection(monkeypatch):
    base_scene = build_scene(
        _rotate_mesh_y(box_geometry(center=(0.0, 0.0, 6.0), size=(8.0, 10.0, 0.25)), 90.0)
    )
    field = Field(bounds=(( -2.0, 8.0), (-6.0, 6.0)), size=(16, 16), axis="x", position=1.5)
    coords = field.get_coordinates()
    tx = wt.Point3f(*_rotate_point_y_90((-1.0, -5.0, 4.5)))

    reference_field, _, reference_detail = compute_reflection_field(
        grid=field,
        rx_z=field.position,
        tx_pos=tx,
        scene=base_scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=8192,
        max_reflections=1,
        mode="3d",
        ray_sampling="full_sphere",
        reflection_coef=0.82,
        tx_polarization=(0.0, 1.0, 0.0),
        grid_data=coords,
    )

    monkeypatch.setattr(scatter_module, "_SCATTER_CHUNK_RAY_THRESHOLD", 1)
    monkeypatch.setattr(scatter_module, "_SCATTER_CHUNK_SIZE", 64)

    chunked_field, _, chunked_detail = compute_reflection_field(
        grid=field,
        rx_z=field.position,
        tx_pos=tx,
        scene=base_scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=8192,
        max_reflections=1,
        mode="3d",
        ray_sampling="full_sphere",
        reflection_coef=0.82,
        tx_polarization=(0.0, 1.0, 0.0),
        grid_data=coords,
    )

    np.testing.assert_allclose(
        np.asarray(chunked_field.real, dtype=np.float64),
        np.asarray(reference_field.real, dtype=np.float64),
        rtol=1e-4,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(chunked_field.imag, dtype=np.float64),
        np.asarray(reference_field.imag, dtype=np.float64),
        rtol=1e-4,
        atol=1e-7,
    )
    assert reference_detail["dda_stats"]["per_bounce"][0]["scatter_chunks_used"] == 1
    assert chunked_detail["dda_stats"]["per_bounce"][0]["chunked_scatter"] is True
    assert chunked_detail["dda_stats"]["per_bounce"][0]["scatter_chunks_used"] > 1
