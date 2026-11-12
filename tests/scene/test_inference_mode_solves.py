# Copyright Xingyu Chen.
# Tests that every radiomap estimator and the path solver run under torch.inference_mode.

"""Tests that every radiomap estimator and the path solver run under torch.inference_mode."""

from __future__ import annotations

import pytest
import torch
from witwin.core import PhysicalMaterial, Scene

from tests.accuracy.scenes import FREQ, antenna, grid, surface

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Channel solves need CUDA")


def _wall_scene() -> Scene:
    v = [[5.0, -4.0, -1.0], [5.0, 4.0, -1.0], [5.0, -4.0, 4.0], [5.0, 4.0, 4.0]]
    wall = surface(v, [[0, 2, 1], [1, 2, 3]], PhysicalMaterial(eps_r=4.0, sigma_e=0.01))
    return Scene(structures=[wall], endpoints=[antenna("tx", (0.0, 0.0, 1.5)),
                                               grid((8.0, -1.0, 1.5), (2, 2), 1.0)])


def _solves():
    from witwin.channel import path, radiomap

    components = {"los", "reflection", "transmission"}
    return {
        "coherent": lambda scene: radiomap.solve(
            scene, radiomap.Coherent(max_depth=2, components=components), reference_frequency_hz=FREQ),
        "stochastic": lambda scene: radiomap.solve(
            scene, radiomap.Stochastic(samples=1 << 12, max_depth=2, components=components),
            reference_frequency_hz=FREQ),
        "path": lambda scene: path.solve(
            scene, path.Config(max_depth=2, components=components), reference_frequency_hz=FREQ),
    }


@pytest.mark.parametrize("name", ["coherent", "stochastic", "path"])
@pytest.mark.parametrize("built_inside", [False, True], ids=["scene_outside", "scene_inside"])
def test_solve_under_inference_mode_matches_the_plain_solve(name, built_inside):
    solve = _solves()[name]
    reference = solve(_wall_scene())
    if built_inside:
        # Scene edits are tracked through version counters, which inference tensors lack.
        with torch.inference_mode(), pytest.raises(ValueError, match="outside torch.inference_mode"):
            solve(_wall_scene())
        return
    scene = _wall_scene()
    with torch.inference_mode():
        result = solve(scene)
        again = solve(scene)
    for key in ("path_gain", "a"):
        expected = getattr(reference, key, None)
        if isinstance(expected, torch.Tensor):
            assert torch.equal(getattr(result, key), expected), key
            assert torch.equal(getattr(again, key), expected), key
