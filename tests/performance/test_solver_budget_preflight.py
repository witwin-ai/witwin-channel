from __future__ import annotations

import importlib

import pytest

from tests.support.scenes import empty_space_los_scene
from witwin.channel.core.memory_budget import MemoryBudgetError
from witwin.channel.montecarlo.basic import Config as BasicConfig
from witwin.channel.montecarlo.bdpt import Config as BDPTConfig


@pytest.mark.parametrize(
    ("module_name", "config"),
    [
        (
            "witwin.channel.montecarlo.basic.solver",
            BasicConfig(samples=1, components={"los"}, workspace_limit_bytes=0),
        ),
        (
            "witwin.channel.montecarlo.bdpt.solver",
            BDPTConfig(samples=1, components={"los"}, workspace_limit_bytes=0),
        ),
    ],
)
def test_budget_failure_precedes_cuda_native_and_tensor_work(
    monkeypatch: pytest.MonkeyPatch, module_name: str, config: object
) -> None:
    solver = importlib.import_module(module_name)
    calls = {"cuda": 0, "native": 0, "allocation": 0}

    def unexpected(kind: str):
        def fail(*args: object, **kwargs: object) -> object:
            calls[kind] += 1
            raise AssertionError(f"{kind} work started before budget enforcement")

        return fail

    monkeypatch.setattr(solver.torch.cuda, "is_available", unexpected("cuda"))
    monkeypatch.setattr(solver, "build_info", unexpected("native"))
    if module_name.endswith("basic.solver"):
        monkeypatch.setattr(solver, "make_cuda_generator", unexpected("allocation"))
    else:
        monkeypatch.setattr(solver, "transmitter_tensors", unexpected("allocation"))

    with pytest.raises(MemoryBudgetError, match="before launch"):
        solver.solve(empty_space_los_scene(), config)

    assert calls == {"cuda": 0, "native": 0, "allocation": 0}
