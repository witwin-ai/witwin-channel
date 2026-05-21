"""Smoke test mirroring examples/path_solver_minimal.py.

Locks in the public path-solver workflow that the example demonstrates:
build a Scene with scene-owned endpoints and consume CIR/CFR/type-filtering
on the result.
"""

from __future__ import annotations

import numpy as np
import torch
import witwin.channel as wc


def test_path_solver_minimal_wall_paths_have_finite_cir_and_cfr():
    scene = wc.Scene(
        structures=[
            wc.Structure(
                name="wall",
                geometry=wc.Box(
                    position=(0.0, 0.0, 1.5),
                    size=(0.25, 4.0, 3.0),
                    device="cuda",
                ),
                material=wc.Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        transmitters=[
            wc.Transmitter(
                name="tx",
                position=wc.Point3f(-2.0, -1.0, 1.5),
            )
        ],
        receivers=[
            wc.Receiver(name="rx0", position=(-2.0, 1.0, 1.5)),
            wc.Receiver(name="rx1", position=(-1.5, 1.5, 1.5)),
        ],
        frequency=3.5e9,
        device="cuda",
    )
    result = wc.path.solve(
        scene=scene,
        transmitter="tx",
        receiver=["rx0", "rx1"],
        config=wc.path.Config(
            num_samples=64,
            max_bounces=1,
            max_diffraction_order=0,
            max_num_paths=4,
            return_geometry=True,
            edge_policy=wc.EdgePolicy(edge_selection_mode="all_edges"),
        ),
    )

    coeff, delay = result.cir()
    subcarriers = torch.linspace(3.49e9, 3.51e9, 4)
    response = result.cfr(subcarriers)
    reflection_paths = result.filter_by_type(wc.path.InteractionType.REFLECTION)

    assert tuple(coeff.shape) == tuple(result.coeff_shape)
    assert tuple(delay.shape) == tuple(result.path_shape)
    assert torch.isfinite(coeff.real).all() and torch.isfinite(coeff.imag).all()
    assert torch.isfinite(delay).all()
    assert tuple(response.shape)[-1] == int(subcarriers.numel())
    assert torch.isfinite(response.real).all() and torch.isfinite(response.imag).all()

    num_paths = np.asarray(result.num_paths)
    assert num_paths.shape[0] == 2
    assert int(num_paths.sum()) >= 1

    reflection_counts = np.asarray(reflection_paths.num_paths)
    assert reflection_counts.shape == num_paths.shape
    assert int(reflection_counts.sum()) >= 1

    path_counts = result.metadata["path_counts"]
    assert "reflection" in path_counts
