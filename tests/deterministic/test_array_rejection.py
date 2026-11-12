# Copyright Xingyu Chen.
# Tests array rejection.

from __future__ import annotations

import pytest
import torch

from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel.radiomap import Coherent
from witwin.channel import radiomap
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

    with pytest.raises(ValueError, match="does not support antenna arrays"):
        radiomap.solve(
            scene,
            Coherent(components={"los"}),
            reference_frequency_hz=1.0e9,
        )