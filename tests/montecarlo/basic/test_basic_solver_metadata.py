import pytest
import torch

from witwin.channel_native import ReceiverPoint, Scene, Transmitter
from witwin.channel_native.core.kernels.extension import build_info
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

    raydn_native = build_info()["uses_raydn_native"]
    config = (
        Config(samples=32, seed=123, diagnostics=True)
        if raydn_native
        else Config(samples=32, seed=123, diagnostics=True, components={"los"})
    )
    result = solve(_scene(), config)

    validate_metadata(result.metadata["kernel"])
    assert result.metadata["seed"] == 123
    assert result.metadata["samples"] == 32
    assert result.metadata["path_count"] == 32
    assert result.metadata["valid_contribution_count"] == 32
    assert result.metadata["raydn"]["reflection"] is raydn_native
    assert result.metadata["raydn"]["diffraction"] is raydn_native
    assert result.metadata["components"]["los"] == "enabled"
    assert result.metadata["components"]["reflection"] == ("enabled" if raydn_native else "not_requested")
    assert result.metadata["components"]["diffraction"] == ("enabled" if raydn_native else "not_requested")
    assert result.metadata["edge_policy"] == {
        "edge_selection_mode": "all_edges",
        "edge_diffraction": True,
        "boundary_edge_policy": "half_plane",
    }
    assert result.metadata["kernel"]["scheduling_strategy"] in {"native_cuda", "native_fused"}
    assert result.metadata["kernel"]["ad_status"] == "none"
    assert result.diagnostics is not None
