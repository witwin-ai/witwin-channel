from __future__ import annotations

import pytest
import torch

from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel.deterministic import Config
from witwin.channel import deterministic
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
        deterministic.solve(
            scene,
            Config(components={"los"}),
            reference_frequency_hz=1.0e9,
        )
