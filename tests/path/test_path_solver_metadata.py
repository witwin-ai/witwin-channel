# Copyright Xingyu Chen.
# Tests path solver metadata.

import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel.deployment import build_info
from witwin.channel.runtime import validate_metadata
from witwin.channel.path import Config, solve


def test_path_solver_metadata_reports_counts_and_capabilities():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    result = solve(
        empty_space_los_scene(),
        Config(components={"los"}),
        reference_frequency_hz=3.0e9,
    )

    validate_metadata(result.metadata["kernel"])
    assert result.metadata["path_count"] == 4
    path_native = build_info()["uses_path_native"]
    rayd_native = build_info()["uses_rayd_native"]
    assert result.metadata["native_capabilities"]["path_native"] is path_native
    assert result.metadata["native_capabilities"]["rayd_native"] is rayd_native
    assert result.metadata["components"]["los"] == "enabled"
    assert result.metadata["components"]["reflection"] == "not_requested"
    assert result.metadata["components"]["diffraction"] == "not_requested"
    assert result.metadata["kernel"]["primitive"] == "path_solver"
    assert result.metadata["kernel"]["launch_count"] == 1
    assert result.metadata["kernel"]["scheduling_strategy"] == "native_cuda"
    assert "fusion_debt" not in result.metadata["kernel"]
    assert result.metadata["kernel"]["ad_status"] == "none"

@pytest.mark.parametrize("discovery,complete", [("auto", True), ("exhaustive", True), ("rays", False)])
def test_path_solver_metadata_reports_discovery_sources_and_completeness(discovery, complete):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")
    from tests.support.scenes import same_side_wall_reflection_scene

    result = solve(
        same_side_wall_reflection_scene(),
        Config(components={"los", "reflection"}, max_depth=2, discovery=discovery),
        reference_frequency_hz=3.0e9,
    )
    source = {"auto": "image_tree"}.get(discovery, discovery)
    assert result.metadata["discovery"] == {
        "requested": discovery,
        "sources": {"1": source, "2": source},
        "complete": complete,
    }
    assert result.metadata["complete"] is complete


def test_path_solver_metadata_is_complete_without_reflection():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")
    result = solve(empty_space_los_scene(), Config(components={"los"}), reference_frequency_hz=3.0e9)
    assert result.metadata["complete"] is True
    assert result.metadata["discovery"] == {"requested": "auto", "sources": {}, "complete": True}
