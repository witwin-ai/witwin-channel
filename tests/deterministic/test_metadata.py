import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.deterministic import Config, solve


def test_metadata_reports_deterministic_solver_decisions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic metadata")

    result = solve(empty_space_los_scene(), Config(max_depth=0, components={"los"}, diagnostics=True))
    metadata = result.metadata

    assert metadata["coherent"] is True
    assert metadata["accumulation_strategy"] == "coherent"
    assert metadata["components"]["los"] == "enabled"
    assert metadata["components"]["reflection"] == "not_requested"
    assert metadata["counts"]["path_count"] == 4
    assert metadata["counts"]["components"]["los"] == 4
    assert metadata["kernel"]["scheduling_strategy"] == "native_cuda"
    assert metadata["kernel"]["ad_status"] == "none"
    assert metadata["kernel"]["launch_count"] >= 1
    assert set(metadata["capability"]) >= {"raydn_native", "path_native", "cuda_available", "optix_available"}


def test_metadata_rejects_requested_components_that_depth_would_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic metadata")

    with pytest.raises(RuntimeError, match="reflection requires max_depth"):
        solve(empty_space_los_scene(), Config(max_depth=0, components={"los", "reflection", "diffraction"}, coherent=False))


def test_diagnostics_report_native_launches_and_path_planning():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic diagnostics")

    result = solve(empty_space_los_scene(), Config(max_depth=0, components={"los"}, diagnostics=True))

    assert result.diagnostics is not None
    assert result.diagnostics["native_launch_count"] >= 1
    assert result.diagnostics["visibility_rejection_count"] == 0
    assert result.diagnostics["selected_edge_count"] == 0
    assert result.diagnostics["accumulation_mode"] == "coherent"
    assert result.diagnostics["path_planning"] == {
        "max_paths": None,
        "candidate_count": 4,
        "guardrail_count": 0,
        "truncated": False,
    }
