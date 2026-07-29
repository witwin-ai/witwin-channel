# Copyright Xingyu Chen.
# Tests array rejection.

from __future__ import annotations

import pytest
import torch

from witwin.core import AntennaPattern, Scene
from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel.montecarlo import bdpt as bdpt_solver
from witwin.channel.montecarlo.bdpt import Config


def _scene(feature: str) -> Scene:
    tx_kwargs: dict[str, object] = {}
    rx_kwargs: dict[str, object] = {}
    if feature == "array":
        tx_kwargs["element_positions"] = torch.tensor(
            [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]
        )
    elif feature == "pattern":
        rx_kwargs["pattern"] = AntennaPattern("vertical")
    elif feature == "precoding":
        tx_kwargs["weights"] = torch.ones(1, dtype=torch.complex64)
    elif feature == "combining":
        rx_kwargs["weights"] = torch.ones(1, dtype=torch.complex64)
    return Scene(
        structures=[],
        endpoints=[
            make_transmitter(position=torch.zeros(3), **tx_kwargs),
            make_receiver(position=torch.tensor([1.0, 0.0, 0.0]), **rx_kwargs),
        ],
    )


@pytest.mark.parametrize("feature", ["array", "pattern", "precoding", "combining"])
def test_unsupported_endpoint_features_fail_before_cuda(
    monkeypatch: pytest.MonkeyPatch, feature: str
) -> None:
    cuda_calls = 0

    def unexpected_cuda() -> bool:
        nonlocal cuda_calls
        cuda_calls += 1
        raise AssertionError("CUDA capability queried before endpoint preflight")

    monkeypatch.setattr(bdpt_solver.torch.cuda, "is_available", unexpected_cuda)

    with pytest.raises(ValueError, match="does not support"):
        bdpt_solver.solve(
            _scene(feature),
            Config(samples=1, components={"los"}),
            reference_frequency_hz=1.0e9,
        )

    assert cuda_calls == 0