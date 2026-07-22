"""Real-scene AD smoke (plan 07 AD-4 final gate).

After the analytic fixtures, one reduced-Munich deterministic solve must
survive ad_mode="vjp" end to end: one backward through a material store leaf
and one through a live transmitter position, each yielding finite, nonzero,
NaN-free gradients. Kept CI-sized: depth 1, a 16x16 radiomap slice, los +
reflection only.

Follows the suite's CUDA gating convention (skipif, like every tests/ad
module); additionally skips when the Munich reference scene asset is not
checked out.
"""

from __future__ import annotations

import pytest
import torch

from tests.support.bin.benchmark_munich_deterministic_native_vs_original import (
    DEFAULT_MUNICH_XML,
    _load_scene,
    _parser,
)
from witwin.channel import Scene, Transmitter
from witwin.channel.deterministic import Config, solve

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Munich AD smoke"
)

_GRID_SIZE = 16


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
    leaf = scene.compile().materials.eps_r
    leaf.requires_grad_(True)
    try:
        result = solve(scene, _config())
        result.path_gain.sum().backward()
        _assert_live_gradient(leaf.grad)
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None


def test_munich_transmitter_backward_is_finite_and_nonzero():
    if not DEFAULT_MUNICH_XML.exists():
        pytest.skip("Munich reference scene is not available")
    base = _munich_scene()
    transmitter = base.transmitters[0]
    leaf = (
        transmitter.position.detach()
        .to(device="cuda", dtype=torch.float32)
        .clone()
        .requires_grad_(True)
    )
    scene = Scene(
        structures=list(base.structures),
        transmitters=[
            Transmitter(position=leaf, power_w=float(transmitter.power_w))
        ],
        receivers=list(base.receivers),
        frequency=base.frequency,
        metadata=base.metadata,
    )
    result = solve(scene, _config())
    result.path_gain.sum().backward()
    _assert_live_gradient(leaf.grad)
