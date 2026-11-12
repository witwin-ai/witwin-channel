# Copyright Xingyu Chen.
# Tests array and directional-pattern rejection before any CUDA work.

from __future__ import annotations

import re

import pytest
import torch

from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel import radiomap
from witwin.channel.experimental import bdpt
from witwin.channel.radiomap import Coherent
from witwin.core import Scene


def test_array_fails_before_cuda() -> None:
    scene = Scene(
        structures=[],
        endpoints=[
            make_transmitter(
                position=torch.zeros(3),
                element_positions=torch.tensor([[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]),
            ),
            make_receiver(position=torch.tensor([1.0, 0.0, 0.0])),
        ],
    )

    with pytest.raises(ValueError, match=re.escape("radiomap.Coherent does not support antenna arrays")):
        radiomap.solve(
            scene,
            Coherent(components={"los"}),
            reference_frequency_hz=1.0e9,
        )


@pytest.mark.parametrize(
    ("solve", "config", "name"),
    [
        (radiomap.solve, radiomap.Coherent(components={"los"}), "radiomap.Coherent"),
        (radiomap.solve, radiomap.Stochastic(samples=1, components={"los"}), "radiomap.Stochastic"),
        (bdpt.solve, bdpt.Config(samples=1, components={"los"}), "experimental.bdpt"),
    ],
)
def test_directional_pattern_error_names_the_public_solver(solve, config, name) -> None:
    scene = Scene(
        structures=[],
        endpoints=[
            make_transmitter(position=torch.zeros(3), pattern="vertical"),
            make_receiver(position=torch.tensor([1.0, 0.0, 0.0])),
        ],
    )

    with pytest.raises(ValueError, match=re.escape(f"{name} does not support directional antenna patterns")):
        solve(scene, config, reference_frequency_hz=1.0e9)
