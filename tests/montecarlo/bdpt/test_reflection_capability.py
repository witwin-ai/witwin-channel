# Copyright Xingyu Chen.
# Tests reflection capability.

import pytest
import torch

from tests.support.scenes import single_wall_reflection_scene
from witwin.channel.montecarlo.bdpt import Config, solve
import witwin.channel.montecarlo.bdpt as bdpt_solver


def test_bdpt_reflection_errors_when_capability_missing(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection capability")

    monkeypatch.setattr(
        bdpt_solver,
        "build_info",
        lambda: {
            "uses_rayd_native": False,
            "cuda_available": True,
            "optix_available": False,
        },
    )

    with pytest.raises(RuntimeError, match="reflection.*RayD"):
        solve(
            single_wall_reflection_scene(),
            Config(components={"reflection"}),
            reference_frequency_hz=3.0e9,
        )


def test_bdpt_los_only_skips_reflection_capability(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection capability")

    monkeypatch.setattr(
        bdpt_solver,
        "build_info",
        lambda: {
            "uses_rayd_native": False,
            "cuda_available": True,
            "optix_available": False,
        },
    )

    result = solve(
        single_wall_reflection_scene(),
        Config(components={"los"}),
        reference_frequency_hz=3.0e9,
    )

    assert result.metadata["components"]["reflection"] == "not_requested"