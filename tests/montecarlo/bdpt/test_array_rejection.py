from __future__ import annotations

import pytest
import torch

from witwin.channel_native import (
    AntennaArray,
    AntennaPattern,
    ReceiverPoint,
    Scene,
    Transmitter,
)
from witwin.channel_native.montecarlo.bdpt import Config
from witwin.channel_native.montecarlo.bdpt import solver


def _scene(feature: str) -> Scene:
    tx_kwargs: dict[str, object] = {}
    rx_kwargs: dict[str, object] = {}
    if feature == "array":
        tx_kwargs["array"] = AntennaArray.ula(2, 0.1)
    elif feature == "pattern":
        rx_kwargs["pattern"] = AntennaPattern("vertical")
    elif feature == "precoding":
        tx_kwargs["precoding"] = torch.ones(1, dtype=torch.complex64)
    elif feature == "combining":
        rx_kwargs["combining"] = torch.ones(1, dtype=torch.complex64)
    return Scene(
        structures=[],
        transmitters=[Transmitter(position=torch.zeros(3), **tx_kwargs)],
        receivers=[ReceiverPoint(position=torch.tensor([1.0, 0.0, 0.0]), **rx_kwargs)],
        frequency=1.0e9,
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

    monkeypatch.setattr(solver.torch.cuda, "is_available", unexpected_cuda)

    with pytest.raises(ValueError, match="does not support"):
        solver.solve(_scene(feature), Config(samples=1, components={"los"}))

    assert cuda_calls == 0
