import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.kernels.metadata import validate_metadata
from witwin.channel_native.path import Config, solve


def test_path_solver_metadata_reports_counts_and_capabilities():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    result = solve(empty_space_los_scene(), Config(components={"los"}, diagnostics=True))

    validate_metadata(result.metadata["kernel"])
    assert result.metadata["path_count"] == 4
    assert result.metadata["valid_contribution_count"] == 4
    assert result.metadata["counts"] == {
        "path_count": 4,
        "valid_path_count": 4,
    }
    path_native = build_info()["uses_path_native"]
    raydn_native = build_info()["uses_raydn_native"]
    assert result.metadata["capability"]["path_native"] is path_native
    assert result.metadata["capability"]["raydn_native"] is raydn_native
    assert result.metadata["components"]["los"] == "enabled"
    assert result.metadata["components"]["reflection"] == "disabled"
    assert result.metadata["components"]["diffraction"] == "disabled"
    assert result.metadata["kernel"]["primitive"] == "path_solver"
    assert result.metadata["kernel"]["launch_count"] == (1 if path_native else 0)
    assert result.metadata["kernel"]["fusion_debt"] is (not path_native)
    assert result.metadata["kernel"]["ad_status"] == "unsupported"
    assert result.diagnostics is not None
