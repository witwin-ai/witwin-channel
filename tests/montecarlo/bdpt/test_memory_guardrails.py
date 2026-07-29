# Copyright Xingyu Chen.
# Tests memory guardrails.

import pytest
import torch

from tests.support.scenes import empty_space_los_scene, same_side_wall_reflection_scene
from tests.support.core_world import make_receiver_grid
from witwin.core import ReceiverGrid, Scene
from witwin.channel.montecarlo.bdpt import Config, solve


def _with_grid(scene: Scene, grid: ReceiverGrid) -> Scene:
    return scene.with_endpoints(
        (
            *tuple(endpoint for endpoint in scene.endpoints if endpoint.role == "tx"),
            grid,
        )
    )


def test_bdpt_radiomap_peak_memory_stays_bounded():
    """The LoS connection table must not materialize
 light_count x rx rows (91 B x samples x cells grew to gigabytes)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT memory measurement")

    grid = make_receiver_grid(
        origin=torch.tensor([0.0, -1.0, 0.0]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(64, 64),
        spacing=(0.1, 0.1),
    )
    scene = _with_grid(same_side_wall_reflection_scene(), grid)
    config = Config(samples=4096, seed=0)

    solve(scene, config, reference_frequency_hz=3.0e9)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    solve(scene, config, reference_frequency_hz=3.0e9)
    torch.cuda.synchronize()

    assert torch.cuda.max_memory_allocated() < 300 * 2**20


def test_bdpt_path_export_memory_guardrail_fails_before_large_allocation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT memory guardrail")

    config = Config(
        samples=32,
        components={"los"},
        export_paths=True,
        max_exported_paths=10_000_000,
        workspace_limit_bytes=1024,
    )

    with pytest.raises(RuntimeError, match="workspace.*BDPT"):
        solve(empty_space_los_scene(), config, reference_frequency_hz=3.0e9)