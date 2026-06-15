import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.core.kernels.metadata import validate_metadata
from witwin.channel_native.montecarlo.basic import Config, solve


def test_basic_primal_metadata_reports_no_ad_by_default():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic metadata")

    result = solve(empty_space_los_scene(), Config(samples=64, components={"los"}))

    validate_metadata(result.metadata["kernel"])
    assert result.metadata["kernel"]["ad_status"] == "none"
    assert result.metadata["kernel"]["tape_bytes"] == 0


@pytest.mark.parametrize("ad_mode", ["vjp", "jvp"])
def test_basic_metadata_rejects_non_primal_ad_modes(ad_mode):
    with pytest.raises(ValueError, match="ad_mode"):
        Config(samples=64, components={"los"}, ad_mode=ad_mode)
