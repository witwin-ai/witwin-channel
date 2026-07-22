from __future__ import annotations

import pytest
import torch

from witwin.channel import AntennaArray, ReceiverPoint, Scene, Transmitter
from witwin.channel.deterministic import Config
from witwin.channel.deterministic import solver


def test_array_fails_before_cuda() -> None:
    scene = Scene(
        structures=[],
        transmitters=[
            Transmitter(
                position=torch.zeros(3), array=AntennaArray.ula(2, 0.1)
            )
        ],
        receivers=[ReceiverPoint(position=torch.tensor([1.0, 0.0, 0.0]))],
        frequency=1.0e9,
    )

    with pytest.raises(ValueError, match="does not support antenna arrays"):
        solver.solve(scene, Config(components={"los"}))
