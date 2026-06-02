"""Regression tests for chunked Cartesian receiver accumulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import witwin as wt

from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import FieldMonitor, Tracer
from witwin.channel.trace.diffraction import constants as diffraction_common


def build_scene():
    cube1 = box_drjit_geometry(center=(-2.0, -2.5, 1.5), size=2.0, rotation=None).to_mesh()
    cube2 = box_drjit_geometry(center=(1.8, 0.8, 1.5), size=2.0, rotation=None).to_mesh()
    return build_test_scene(cube1, cube2)


def field_to_numpy(field):
    return np.asarray(field.real) + 1j * np.asarray(field.imag)


@pytest.mark.gpu
def test_chunked_cartesian_accumulation_matches_unchunked(monkeypatch):
    scene = build_scene()
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=512,
        reflection_max_bounces=2,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )
    monitor = FieldMonitor(
        "chunk_regression_plane",
        axis="z",
        position=1.5,
        bounds=((-5.0, 5.0), (-5.0, 5.0)),
        grid_size=24,
    )
    tx = wt.Point3f(0.0, -4.5, 1.5)

    monkeypatch.setattr(diffraction_common, "CARTESIAN_PAIR_CHUNK_BUDGET", 1 << 30)
    baseline = tracer.trace(tx, monitor=monitor, verbose=False).primary

    monkeypatch.setattr(diffraction_common, "CARTESIAN_PAIR_CHUNK_BUDGET", 256)
    chunked = tracer.trace(tx, monitor=monitor, verbose=False).primary

    for field_name in ("reflection", "diffraction", "total"):
        baseline_np = field_to_numpy(getattr(baseline.field, field_name))
        chunked_np = field_to_numpy(getattr(chunked.field, field_name))
        assert np.allclose(chunked_np, baseline_np, rtol=1e-5, atol=1e-6), field_name
