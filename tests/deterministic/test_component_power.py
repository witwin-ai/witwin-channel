import pytest
import torch

from tests.support.scenes import (
    empty_space_los_scene,
    same_side_wall_reflection_scene,
)
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.deterministic import Config, solve


def test_transmission_and_scattering_are_zero_contribution_plumbing():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the deterministic solver")

    baseline = solve(empty_space_los_scene(), Config(components={"los"}))
    result = solve(
        empty_space_los_scene(),
        Config(components={"los", "transmission", "scattering"}),
    )

    # (a) requesting the new components validates and (b) they carry zeros, so
    # the total gain is unchanged from the los-only baseline.
    for name in ("transmission", "scattering"):
        assert name in result.component_power
        assert torch.count_nonzero(result.component_power[name]) == 0
    torch.testing.assert_close(result.path_gain, baseline.path_gain)
    # (c) metadata reports a truthful requested-but-empty status.
    assert result.metadata["components"]["transmission"] == "enabled_no_paths"
    assert result.metadata["components"]["scattering"] == "enabled_no_paths"
    assert result.metadata["components"]["los"] == "enabled"


def test_deterministic_config_rejects_unknown_component():
    with pytest.raises(ValueError, match="components"):
        Config(components={"los", "teleportation"})


def test_single_component_power_matches_total_for_reflection_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic reflection")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    result = solve(same_side_wall_reflection_scene(), Config(components={"reflection"}, coherent=False))

    torch.testing.assert_close(result.component_power["reflection"], result.path_gain, rtol=5.0e-4, atol=1.0e-8)
    assert torch.count_nonzero(result.component_power["los"]) == 0
    assert torch.count_nonzero(result.component_power["diffraction"]) == 0
