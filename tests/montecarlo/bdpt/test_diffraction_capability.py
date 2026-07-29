# Copyright Xingyu Chen.
# Tests diffraction capability.

import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel.montecarlo.bdpt import Config, solve
import witwin.channel.montecarlo.bdpt as bdpt_solver


def test_bdpt_diffraction_errors_when_capability_missing(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction capability")

    monkeypatch.setattr(
        bdpt_solver,
        "build_info",
        lambda: {
            "uses_rayd_native": False,
            "cuda_available": True,
            "optix_available": False,
        },
    )

    with pytest.raises(RuntimeError, match="diffraction.*RayD"):
        solve(
            wedge_diffraction_scene(),
            Config(components={"diffraction"}),
            reference_frequency_hz=3.0e9,
        )


def test_bdpt_los_only_skips_diffraction_capability(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction capability")

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
        wedge_diffraction_scene(),
        Config(components={"los"}),
        reference_frequency_hz=3.0e9,
    )

    assert result.metadata["components"]["diffraction"] == "not_requested"


def test_bdpt_diffraction_errors_when_order_disabled():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction capability")

    with pytest.raises(
        RuntimeError, match="diffraction requires max_diffraction_order"
    ):
        solve(
            wedge_diffraction_scene(),
            Config(components={"diffraction"}, max_diffraction_order=0),
            reference_frequency_hz=3.0e9,
        )