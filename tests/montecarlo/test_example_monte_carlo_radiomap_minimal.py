"""Smoke test mirroring examples/monte_carlo_radiomap_minimal.py.

Locks in the public Monte Carlo radiomap solve entrypoint: build a Scene with
endpoints, call montecarlo.solve(...), and read path_gain.
"""

from __future__ import annotations

import numpy as np
from witwin.channel.core.scene import ReceiverGrid, Scene, Transmitter
from witwin.core import Box, Material, Structure
from witwin.channel.montecarlo import Config, IntegratorOptions, solve


def test_monte_carlo_radiomap_minimal_wall_path_gain_is_finite_and_nonzero():
    scene = Scene(
        structures=[
            Structure(
                name="wall",
                geometry=Box(
                    position=(0.0, 0.0, 1.5),
                    size=(0.25, 4.0, 3.0),
                    device="cuda",
                ),
                material=Material(eps_r=4.0, sigma_e=0.0),
            ),
        ],
        transmitters=[
            Transmitter("tx", (-2.0, 0.0, 1.5)),
        ],
        receivers=[
            ReceiverGrid(
                "rm",
                axis="z",
                position=1.5,
                bounds=((-3.0, 3.0), (-3.0, 3.0)),
                grid_shape=(16, 16),
            ),
        ],
        frequency=3.5e9,
        device="cuda",
    )
    config = Config(
        num_samples=64,
        max_bounces=1,
        max_diffraction_order=0,
        integrator_options=IntegratorOptions(
            integrator="basic",
            samples_per_tx=4096,
            accumulation_backend="auto",
            seed=7,
        ),
    )
    result = solve(
        scene=scene,
        transmitter="tx",
        receiver="rm",
        config=config,
    )

    path_gain = np.asarray(result.path_gain, dtype=np.float32)
    assert path_gain.shape == (1, 16, 16)
    assert np.isfinite(path_gain).all()
    assert float(path_gain.max()) > 0.0
    assert float(path_gain.min()) >= 0.0

    los = np.asarray(result.incoherent["los"], dtype=np.float32)
    reflection = np.asarray(result.incoherent["reflection"], dtype=np.float32)
    assert los.shape == path_gain.shape == reflection.shape
    assert float(los.max()) > 0.0
