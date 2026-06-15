import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.montecarlo.bdpt import Config, solve


def test_bdpt_fixed_seed_reproduces_outputs_metadata_and_exports():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT seed stability")

    scene = empty_space_los_scene()
    config = Config(samples=64, seed=2026, components={"los"}, export_paths=True, max_exported_paths=8)

    first = solve(scene, config)
    second = solve(scene, config)

    assert first.metadata["path_counts_by_strategy"] == second.metadata["path_counts_by_strategy"]
    assert first.metadata["valid_contribution_count"] == second.metadata["valid_contribution_count"]
    torch.testing.assert_close(first.path_gain, second.path_gain, rtol=1.0e-6, atol=1.0e-12)
    assert first.path_samples is not None
    assert second.path_samples is not None
    torch.testing.assert_close(first.path_samples.contribution, second.path_samples.contribution)
    torch.testing.assert_close(first.path_samples.pdf, second.path_samples.pdf)
