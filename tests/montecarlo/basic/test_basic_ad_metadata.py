import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.core.kernels.metadata import validate_metadata
from witwin.channel_native.montecarlo.basic import Config, solve


def test_basic_primal_metadata_keeps_ad_unsupported_by_default():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic metadata")

    result = solve(empty_space_los_scene(), Config(samples=64, components={"los"}))

    validate_metadata(result.metadata["kernel"])
    assert result.metadata["kernel"]["ad_status"] == "unsupported"
    assert result.metadata["kernel"]["tape_bytes"] == 0


def test_basic_vjp_metadata_reports_fixed_topology_ad():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic metadata")

    result = solve(empty_space_los_scene(), Config(samples=64, components={"los"}, ad_mode="vjp"))

    validate_metadata(result.metadata["kernel"])
    assert result.metadata["kernel"]["ad_status"] == "vjp"
    assert result.metadata["fixed_topology"] is True
    assert result.metadata["requires_fixed_seed"] is True
    assert result.metadata["ad_mode"] == "vjp"
