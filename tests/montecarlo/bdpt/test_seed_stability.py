import pytest
import torch

from tests.support.scenes import empty_space_los_scene, same_side_wall_reflection_scene
from tests.support.core_world import make_receiver_grid
from witwin.core import ReceiverGrid, Scene
from witwin.channel.core.kernels.extension import build_info
from witwin.channel.montecarlo.bdpt import Config, solve


def _with_grid(scene: Scene, grid: ReceiverGrid) -> Scene:
    return scene.with_endpoints(
        (
            *tuple(endpoint for endpoint in scene.endpoints if endpoint.role == "tx"),
            grid,
        )
    )


def test_bdpt_fixed_seed_reproduces_outputs_metadata_and_exports():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT seed stability")

    scene = empty_space_los_scene()
    config = Config(
        samples=64,
        seed=2026,
        components={"los"},
        export_paths=True,
        max_exported_paths=8,
    )

    first = solve(scene, config, reference_frequency_hz=3.0e9)
    second = solve(scene, config, reference_frequency_hz=3.0e9)

    assert (
        first.metadata["path_counts_by_strategy"]
        == second.metadata["path_counts_by_strategy"]
    )
    assert (
        first.metadata["valid_contribution_count"]
        == second.metadata["valid_contribution_count"]
    )
    torch.testing.assert_close(
        first.path_gain, second.path_gain, rtol=1.0e-6, atol=1.0e-12
    )
    assert first.path_samples is not None
    assert second.path_samples is not None
    torch.testing.assert_close(
        first.path_samples.contribution, second.path_samples.contribution
    )
    torch.testing.assert_close(first.path_samples.pdf, second.path_samples.pdf)


def test_bdpt_delta_reflection_is_seed_invariant_and_reproducible():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT seed stability")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    grid = make_receiver_grid(
        origin=torch.tensor([0.0, -1.0, 0.0]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )
    scene = _with_grid(same_side_wall_reflection_scene(), grid)
    first = solve(
        scene,
        Config(samples=2048, seed=17, components={"reflection"}),
        reference_frequency_hz=3.0e9,
    )
    repeated = solve(
        scene,
        Config(samples=2048, seed=17, components={"reflection"}),
        reference_frequency_hz=3.0e9,
    )
    changed = solve(
        scene,
        Config(samples=2048, seed=18, components={"reflection"}),
        reference_frequency_hz=3.0e9,
    )

    torch.testing.assert_close(first.path_gain, repeated.path_gain, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.path_gain, changed.path_gain, rtol=0.0, atol=0.0)
