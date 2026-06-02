"""Phase 1 regression coverage for z-normal 3D reflection sampling."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import witwin as wt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._scene_helpers import box_geometry, build_scene
from witwin.channel import Field, FieldMonitor, Tracer
from witwin.channel.trace import compute_reflection_field
FREQUENCY = 1e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * math.pi / WAVELENGTH
TRACE_BOUNDS = ((-6.0, 6.0), (-2.0, 8.0))


def _build_wall_scene():
    wall = box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0))
    return build_scene(wall)


def _build_ceiling_scene():
    ceiling = box_geometry(center=(0.0, 0.0, 6.5), size=(10.0, 10.0, 0.25))
    return build_scene(ceiling)


def _field_magnitude(field):
    real = np.asarray(field.real, dtype=np.float64)
    imag = np.asarray(field.imag, dtype=np.float64)
    return np.sqrt(real * real + imag * imag)


@pytest.mark.gpu
def test_reflection_3d_matches_2d_on_single_wall_scene():
    scene = _build_wall_scene()
    field = Field(bounds=TRACE_BOUNDS, size=(24, 20))
    coords = field.get_coordinates()
    tx = wt.Point3f(-3.0, -5.0, 1.5)

    ref_2d, _, detail_2d = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=1024,
        max_reflections=1,
        mode="2d",
        reflection_coef=0.8,
        grid_data=coords,
    )
    ref_3d, _, detail_3d = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=16384,
        max_reflections=1,
        mode="3d",
        reflection_coef=0.8,
        grid_data=coords,
    )

    mag_2d = _field_magnitude(ref_2d)
    mag_3d = _field_magnitude(ref_3d)
    mask = (mag_2d > mag_2d.max() * 1e-3) | (mag_3d > mag_3d.max() * 1e-3)
    db_2d = 20.0 * np.log10(np.maximum(mag_2d, 1e-20))
    db_3d = 20.0 * np.log10(np.maximum(mag_3d, 1e-20))
    rms_db = float(np.sqrt(np.mean((db_2d[mask] - db_3d[mask]) ** 2)))

    assert detail_2d["dda_stats"]["ray_mode"] == "2d"
    assert detail_3d["dda_stats"]["ray_mode"] == "3d"
    assert detail_3d["dda_stats"]["projected_direction_norm_threshold"] == 0.0
    assert rms_db < 0.1


@pytest.mark.gpu
def test_tracer_reports_3d_reflection_sampling_stats_and_skips_low_projection_rays():
    scene = _build_ceiling_scene()
    monitor = FieldMonitor(
        "phase1_monitor",
        axis="z",
        position=1.5,
        bounds=TRACE_BOUNDS,
        grid_size=(24, 20),
        ray_mode="3d",
    )
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=4096,
        reflection_max_bounces=2,
        min_ray_contribution_threshold=0.25,
        max_diffractions=1,
    )

    result = tracer.trace(
        wt.Point3f(-1.0, -5.0, 4.5),
        monitor=monitor,
        verbose=False,
        return_diffraction_audit=False,
    )
    payload = result.primary
    sampling = payload.metadata["reflection_sampling"]
    total_real = np.asarray(payload.field.total.real, dtype=np.float64)
    total_imag = np.asarray(payload.field.total.imag, dtype=np.float64)

    assert np.isfinite(total_real).all()
    assert np.isfinite(total_imag).all()
    assert sampling["backend"] == "dda_planar_grid"
    assert sampling["ray_mode"] == "3d"
    assert sampling["projected_direction_norm_threshold"] == pytest.approx(0.25)
    assert sampling["recommended_ray_count_multiplier_vs_2d"] == pytest.approx(10.0)
    assert len(sampling["per_bounce"]) >= 1
    assert any(stat["skipped_low_projection_rays"] > 0 for stat in sampling["per_bounce"])
    assert all(stat["segment_rays"] >= stat["dda_candidate_rays"] for stat in sampling["per_bounce"])
    assert all(stat["max_dt_x"] >= 0.0 for stat in sampling["per_bounce"])
    assert all(stat["max_dt_y"] >= 0.0 for stat in sampling["per_bounce"])
