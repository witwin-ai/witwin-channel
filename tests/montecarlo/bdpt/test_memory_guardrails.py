import pytest
import torch

from tests.support.scenes import empty_space_los_scene, same_side_wall_reflection_scene
from witwin.channel import ReceiverGrid
from witwin.channel.montecarlo.bdpt import Config, solve


def test_bdpt_radiomap_peak_memory_stays_bounded():
    """Guards audit P-1/P-5: the LoS connection table must not materialize
    light_count x rx rows (91 B x samples x cells grew to gigabytes)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT memory measurement")

    grid = ReceiverGrid(
        origin=torch.tensor([0.0, -1.0, 0.0]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(64, 64),
        spacing=(0.1, 0.1),
    )
    scene = same_side_wall_reflection_scene().add(grid)
    config = Config(samples=4096, seed=0)

    solve(scene, config)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    solve(scene, config)
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
        solve(empty_space_los_scene(), config)
