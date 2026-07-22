import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel.montecarlo.bdpt import Config, solve


def test_bdpt_solver_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="bdpt requires CUDA"):
        solve(empty_space_los_scene(), Config(components={"los"}))
