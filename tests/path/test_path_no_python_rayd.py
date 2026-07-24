import sys

import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel.path import Config, solve


def test_path_solver_does_not_import_python_rayd_or_drjit():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    sys.modules.pop("rayd", None)
    sys.modules.pop("drjit", None)

    solve(
        empty_space_los_scene(),
        Config(components={"los"}),
        reference_frequency_hz=3.0e9,
    )

    assert "rayd" not in sys.modules
    assert "drjit" not in sys.modules
