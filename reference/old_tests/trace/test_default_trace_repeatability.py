"""Regression coverage for repeated forward traces on the default GPU path."""

import sys
from pathlib import Path

import numpy as np
import pytest
import witwin as wt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import FieldMonitor, Tracer


CUBE1_BASE_CENTER = (-2.5, -3.0, 1.5)
CUBE2_CENTER = (2.0, 0.5, 1.5)
CUBE3_CENTER = (-0.5, 3.5, 1.5)
CUBE_SIZE = 2.0
TX_POS = (0.0, -5.0, 1.5)
TRACE_BOUNDS = ((-6.0, 6.0), (-6.0, 6.0))


def _build_scene():
    cube1 = box_drjit_geometry(
        center=(CUBE1_BASE_CENTER[0], CUBE1_BASE_CENTER[1], CUBE1_BASE_CENTER[2]),
        size=CUBE_SIZE,
    ).to_mesh()
    cube2 = box_drjit_geometry(center=CUBE2_CENTER, size=CUBE_SIZE).to_mesh()
    cube3 = box_drjit_geometry(center=CUBE3_CENTER, size=CUBE_SIZE).to_mesh()
    return build_test_scene(cube1, cube2, cube3)


@pytest.mark.gpu
def test_default_forward_trace_is_repeatable():
    scene = _build_scene()
    monitor = FieldMonitor(
        "repeatability_monitor",
        axis="z",
        position=TX_POS[2],
        bounds=TRACE_BOUNDS,
        grid_size=64,
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=320,
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )

    result_a = tracer.trace(wt.Point3f(*TX_POS), monitor=monitor, verbose=False, return_diffraction_audit=False)
    result_b = tracer.trace(wt.Point3f(*TX_POS), monitor=monitor, verbose=False, return_diffraction_audit=False)

    total_a = result_a.primary.field.total
    total_b = result_b.primary.field.total
    real_a = np.asarray(total_a.real, dtype=np.float64)
    imag_a = np.asarray(total_a.imag, dtype=np.float64)
    real_b = np.asarray(total_b.real, dtype=np.float64)
    imag_b = np.asarray(total_b.imag, dtype=np.float64)

    assert np.allclose(real_a, real_b, rtol=2e-6, atol=1e-8)
    assert np.allclose(imag_a, imag_b, rtol=2e-6, atol=1e-8)
    assert result_a.primary.metadata["execution"] == {
        "accumulate_primal": "drjit",
        "accumulate_jvp": "drjit_replay",
        "accumulate_backward": "drjit_replay",
        "suffix_dda": "symbolic",
        "suffix_russian_roulette": False,
    }
