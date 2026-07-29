# Copyright Xingyu Chen.
# Tests solver cuda requirement.

import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel.deterministic import Config, solve


def test_solve_requires_cuda_before_native_solver_work():
    if torch.cuda.is_available():
        pytest.skip("CUDA requirement is only observable on non-CUDA hosts")

    with pytest.raises(
        RuntimeError, match="witwin.channel.deterministic requires CUDA"
    ):
        solve(empty_space_los_scene(), Config(), reference_frequency_hz=3.0e9)