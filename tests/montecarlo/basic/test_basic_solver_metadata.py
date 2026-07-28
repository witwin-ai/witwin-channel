import pytest
import torch

from tests.support.core_world import make_receiver, make_transmitter
from witwin.core import Scene
from witwin.channel.deployment import build_info
from witwin.channel.runtime import validate_metadata
from witwin.channel.montecarlo.basic import Config, solve


def _scene() -> Scene:
    return Scene(
        structures=[],
        endpoints=[
            make_transmitter(torch.tensor([0.0, 0.0, 0.0])),
            make_receiver(torch.tensor([1.0, 0.0, 0.0])),
        ],
    )


def test_basic_solver_metadata_reports_counts_and_capabilities():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic solver")

    rayd_native = build_info()["uses_rayd_native"]
    config = (
        Config(samples=32, seed=123, diagnostics=True)
        if rayd_native
        else Config(samples=32, seed=123, diagnostics=True, components={"los"})
    )
    result = solve(_scene(), config, reference_frequency_hz=3.0e9)

    validate_metadata(result.metadata["kernel"])
    assert result.metadata["seed"] == 123
    assert result.metadata["samples"] == 32
    assert result.metadata["path_count"] == 32
    assert result.metadata["contribution_capacity"] == 32
    assert "valid_contribution_count" not in result.metadata
    assert result.metadata["rayd"]["reflection"] is rayd_native
    assert result.metadata["rayd"]["diffraction"] is rayd_native
    assert result.metadata["components"]["los"] == "enabled"
    assert result.metadata["components"]["reflection"] == (
        "enabled" if rayd_native else "not_requested"
    )
    assert result.metadata["components"]["diffraction"] == (
        "enabled" if rayd_native else "not_requested"
    )
    assert result.metadata["edge_policy"] == {
        "edge_selection_mode": "all_edges",
        "edge_diffraction": True,
        "boundary_edge_policy": "half_plane",
    }
    assert result.metadata["kernel"]["scheduling_strategy"] in {
        "native_cuda",
        "native_fused",
    }
    assert result.metadata["kernel"]["ad_status"] == "none"
    assert result.diagnostics is not None
