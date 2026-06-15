import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.deterministic import Config, solve


def test_fixed_topology_ad_status_reports_no_ad_for_primal_solve():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic AD metadata")

    result = solve(empty_space_los_scene(), Config(max_depth=0, components={"los"}))

    assert result.metadata["kernel"]["ad_status"] == "none"
