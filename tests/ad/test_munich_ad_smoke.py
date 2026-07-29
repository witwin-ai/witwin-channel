# Copyright Xingyu Chen.
# Tests munich ad smoke.

"""Tests munich ad smoke."""

from __future__ import annotations

import pytest
import torch

from tests.support.bin.benchmark_munich_deterministic_native_vs_original import (
    DEFAULT_MUNICH_XML,
    _load_scene,
    _parser,
)
from witwin.core import Scene
from tests.support.core_world import make_transmitter
from witwin.channel.deterministic import Config, solve
from witwin.channel.scene import compile as compile_scene

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Munich AD smoke"
)

_GRID_SIZE = 16
_FREQUENCY_HZ = 2.4e9


def _munich_scene() -> Scene:
    args = _parser().parse_args(["--grid-size", str(_GRID_SIZE), "--max-depth", "1"])
    return _load_scene(args)


def _config() -> Config:
    return Config(
        max_depth=1,
        components={"los", "reflection"},
        coherent=False,
        return_field=False,
        export_paths=False,
        ad_mode="vjp",
    )


def _assert_live_gradient(grad: torch.Tensor) -> None:
    assert grad is not None
    assert bool(torch.isfinite(grad).all()), "Munich AD gradient carries NaN/Inf"
    assert float(grad.abs().max()) > 0.0, "Munich AD gradient is identically zero"


def test_munich_material_backward_is_finite_and_nonzero():
    if not DEFAULT_MUNICH_XML.exists():
        pytest.skip("Munich reference scene is not available")
    scene = _munich_scene()
    leaf = compile_scene(
        scene,
        reference_frequency_hz=_FREQUENCY_HZ,
    ).materials.eps_r
    leaf.requires_grad_(True)
    try:
        result = solve(
            scene,
            _config(),
            reference_frequency_hz=_FREQUENCY_HZ,
        )
        result.path_gain.sum().backward()
        _assert_live_gradient(leaf.grad)
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None


def test_munich_transmitter_backward_is_finite_and_nonzero():
    if not DEFAULT_MUNICH_XML.exists():
        pytest.skip("Munich reference scene is not available")
    base = _munich_scene()
    transmitter = next(endpoint for endpoint in base.endpoints if endpoint.role == "tx")
    leaf = (
        transmitter.position.detach()
        .to(device="cuda", dtype=torch.float32)
        .clone()
        .requires_grad_(True)
    )
    scene = Scene(
        structures=list(base.structures),
        endpoints=[
            make_transmitter(position=leaf, power_w=float(transmitter.power_w)),
            *[endpoint for endpoint in base.endpoints if endpoint.role == "rx"],
        ],
        metadata=base.metadata,
    )
    result = solve(
        scene,
        _config(),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    result.path_gain.sum().backward()
    _assert_live_gradient(leaf.grad)