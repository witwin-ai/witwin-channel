import pytest
import torch

from witwin.channel_native import ReceiverPoint, Scene, Transmitter
from witwin.channel_native.core.kernels.metadata import validate_metadata
from witwin.channel_native.montecarlo.basic import Config, solve


def _scene() -> Scene:
    return Scene(
        structures=[],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([1.0, 0.0, 0.0]))],
        frequency=3.0e9,
    )


def test_basic_solver_metadata_reports_counts_and_capabilities():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic solver")

    result = solve(_scene(), Config(samples=32, seed=123, diagnostics=True))

    validate_metadata(result.metadata["kernel"])
    assert result.metadata["seed"] == 123
    assert result.metadata["samples"] == 32
    assert result.metadata["path_count"] == 32
    assert result.metadata["valid_contribution_count"] == 32
    assert result.metadata["raydn"]["reflection"] is False
    assert result.metadata["raydn"]["diffraction"] is False
    assert result.metadata["components"]["los"] == "enabled"
    assert result.metadata["components"]["reflection"] == "capability-disabled"
    assert result.metadata["components"]["diffraction"] == "capability-disabled"
    assert result.diagnostics is not None
