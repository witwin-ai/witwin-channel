import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.core.kernels.metadata import validate_metadata
from witwin.channel_native.montecarlo.bdpt import Config, solve


def test_bdpt_metadata_reports_contract_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT metadata")

    config = Config(samples=32, seed=123, sample_streams=2, components={"los"}, diagnostics=True)
    result = solve(empty_space_los_scene(), config)

    assert result.metadata["samples"] == 32
    assert result.metadata["seed"] == 123
    assert result.metadata["sample_streams"] == 2
    assert result.metadata["mis"] == "power_heuristic"
    assert result.metadata["path_counts_by_strategy"]["los"] == 32 * 2 * 2
    assert (
        result.metadata["valid_contribution_count"]
        == result.metadata["path_counts_by_strategy"]["los"] * result.path_gain.shape[1]
    )
    assert result.metadata["components"]["los"] == "enabled"
    assert result.metadata["native_capabilities"]["cuda"] is True
    assert result.metadata["variance"] is True
    assert result.metadata["ad_status"] == "none"
    validate_metadata(result.metadata["kernel"])
    assert result.metadata["kernel"]["ad_status"] == "none"
    assert result.diagnostics is not None
