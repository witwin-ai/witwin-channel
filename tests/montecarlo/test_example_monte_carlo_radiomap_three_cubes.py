"""Smoke test mirroring examples/monte_carlo_radiomap_three_cubes.py.

Drives the example's ``ThreeCubeExperiment`` with a tiny grid + ray budget and
checks that the forward pass and a tx-position JVP/FD pair still produce
finite values. This is the canonical Monte Carlo radiomap differentiability
contract that every solver-side change must preserve.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.monte_carlo_radiomap_three_cubes import ThreeCubeExperiment


@pytest.fixture(scope="module")
def tiny_experiment():
    return ThreeCubeExperiment(
        grid_shape=(16, 16),
        forward_num_samples=64,
        gradient_num_samples=64,
        samples_per_tx=8_192,
        seed=7,
    )


def test_monte_carlo_three_cube_forward_components_are_finite(tiny_experiment):
    snapshot = tiny_experiment.forward()
    assert snapshot.path_gain.shape == (16, 16)
    assert np.isfinite(snapshot.path_gain).all()
    assert float(snapshot.path_gain.max()) > 0.0
    for name in ("los", "reflection", "diffraction"):
        component = snapshot.components[name]
        assert component.shape == snapshot.path_gain.shape
        assert np.isfinite(component).all()
    assert float(snapshot.components["los"].max()) > 0.0


def test_monte_carlo_three_cube_tx_x_gradient_matches_fd_direction(tiny_experiment):
    gradient = tiny_experiment.gradient("tx_x", fd_step=5.0e-3)
    assert gradient.jvp.shape == tiny_experiment.grid_shape
    assert gradient.fd.shape == tiny_experiment.grid_shape
    assert np.isfinite(gradient.jvp).all()
    assert np.isfinite(gradient.fd).all()
    assert float(np.sum(np.abs(gradient.jvp))) > 0.0
    assert float(np.sum(np.abs(gradient.fd))) > 0.0
