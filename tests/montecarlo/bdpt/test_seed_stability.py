import pytest
import torch

from tests.support.scenes import empty_space_los_scene, same_side_wall_reflection_scene
from witwin.channel_native import ReceiverGrid
from witwin.channel_native.core.kernels.extension import build_info
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


def test_bdpt_delta_reflection_is_seed_invariant_and_reproducible():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT seed stability")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    grid = ReceiverGrid(
        origin=torch.tensor([0.0, -1.0, 0.0]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )
    scene = same_side_wall_reflection_scene().add(grid)
    first = solve(scene, Config(samples=2048, seed=17, components={"reflection"}))
    repeated = solve(scene, Config(samples=2048, seed=17, components={"reflection"}))
    changed = solve(scene, Config(samples=2048, seed=18, components={"reflection"}))

    torch.testing.assert_close(first.path_gain, repeated.path_gain, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.path_gain, changed.path_gain, rtol=0.0, atol=0.0)
