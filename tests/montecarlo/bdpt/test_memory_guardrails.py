import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.montecarlo.bdpt import Config, solve


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
