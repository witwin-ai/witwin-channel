import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel.montecarlo.basic import Config, solve


def test_basic_solver_fixed_seed_is_stable_for_counts_and_outputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic seed stability")

    scene = empty_space_los_scene()
    config = Config(samples=512, seed=2026)

    first = solve(scene, config)
    second = solve(scene, config)

    assert first.metadata["path_count"] == second.metadata["path_count"]
    assert (
        first.metadata["contribution_capacity"]
        == second.metadata["contribution_capacity"]
    )
    torch.testing.assert_close(first.path_gain, second.path_gain, rtol=0.0, atol=0.0)


def test_basic_solver_seed_is_reported_in_metadata():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic seed stability")

    result = solve(empty_space_los_scene(), Config(samples=64, seed=99))

    assert result.metadata["seed"] == 99
