import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel.core.kernels.metadata import validate_metadata
from witwin.channel.montecarlo.basic import Config, solve


def test_basic_primal_metadata_reports_no_ad_by_default():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic metadata")

    result = solve(
        empty_space_los_scene(),
        Config(samples=64, components={"los"}),
        reference_frequency_hz=3.0e9,
    )

    validate_metadata(result.metadata["kernel"])
    assert result.metadata["kernel"]["ad_status"] == "none"
    assert result.metadata["kernel"]["tape_bytes"] == 0
    assert result.metadata["kernel"]["backward_launch_count"] == 0
    assert result.metadata["kernel"]["jvp_launch_count"] == 0


@pytest.mark.parametrize("ad_mode", ["vjp", "jvp"])
def test_basic_metadata_reports_registered_ad_companions(ad_mode):
    """Plan 07 AD-3: counters report the companion launches actually wired.

    A point-receiver LoS solve registers exactly one AD Function (the LoS
    path-gain matrix), whose companion is one native launch; reverse mode
    retains its saved endpoint tensors as tape, forward mode retains nothing
    past the solve.
    """

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic metadata")

    result = solve(
        empty_space_los_scene(),
        Config(samples=64, components={"los"}, ad_mode=ad_mode),
        reference_frequency_hz=3.0e9,
    )

    kernel = result.metadata["kernel"]
    validate_metadata(kernel)
    assert kernel["ad_status"] == ad_mode
    if ad_mode == "vjp":
        assert kernel["backward_launch_count"] == 1
        assert kernel["jvp_launch_count"] == 0
        # tape = saved tx positions (2x3) + tx power (2) + rx positions (2x3).
        assert kernel["tape_bytes"] == 14 * 4
    else:
        assert kernel["backward_launch_count"] == 0
        assert kernel["jvp_launch_count"] == 1
        assert kernel["tape_bytes"] == 0
