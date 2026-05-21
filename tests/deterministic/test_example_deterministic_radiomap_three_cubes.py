"""Smoke test mirroring examples/deterministic_radiomap_three_cubes.py.

Locks in the deterministic radiomap solver's forward workflow and the
shadow-boundary-correction toggle that the notebook exercises. AD/JVP
gradient coverage for the deterministic solver lives in
``test_deterministic_material_gradients.py`` -- the example's own
``gradient()`` plumbing is intentionally not tested here because it is
auxiliary code, not part of the public solver contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.deterministic_radiomap_three_cubes import ThreeCubeExperiment


@pytest.fixture(scope="module")
def tiny_experiment():
    return ThreeCubeExperiment(
        grid_shape=(32, 32),
        forward_num_samples=128,
        gradient_num_samples=128,
        max_bounces=2,
        max_diffraction_order=0,
        shadow_boundary_correction=False,
        seed=7,
    )


def test_deterministic_three_cube_forward_components_are_finite(tiny_experiment):
    snapshot = tiny_experiment.forward()
    assert snapshot.path_gain.shape == (32, 32)
    assert np.isfinite(snapshot.path_gain).all()
    assert float(snapshot.path_gain.max()) > 0.0
    for name in ("los", "reflection", "diffraction"):
        component = snapshot.components[name]
        assert component.shape == snapshot.path_gain.shape
        assert np.isfinite(component).all()
    assert float(snapshot.components["los"].max()) > 0.0


def test_deterministic_three_cube_shadow_boundary_toggle_is_finite(tiny_experiment):
    comparison = tiny_experiment.shadow_boundary_correction_comparison()
    for snapshot in (comparison.with_correction, comparison.without_correction):
        assert snapshot.path_gain.shape == (32, 32)
        assert np.isfinite(snapshot.path_gain).all()

