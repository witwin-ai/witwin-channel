# Copyright Xingyu Chen.
# Tests ad errors.

import pytest
import torch

from witwin.channel.radiomap import Coherent
from witwin.channel.radiomap import solve
from tests.support.scenes import same_side_wall_reflection_scene


@pytest.mark.parametrize("ad_mode", ["vjp", "jvp"])
def test_deterministic_accepts_fixed_topology_ad_modes(ad_mode):
    assert Coherent(ad_mode=ad_mode).ad_mode == ad_mode


def test_deterministic_rejects_unknown_ad_modes():
    with pytest.raises(ValueError, match="ad_mode"):
        Coherent(ad_mode="forward")


def test_coherent_reflection_uses_complex_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic runtime validation")

    result = solve(
        same_side_wall_reflection_scene(),
        Coherent(components={"reflection"}, combine="field"),
        reference_frequency_hz=3.0e9,
    )

    torch.testing.assert_close(
        result.path_gain, result.field.abs().square(), rtol=2.0e-4, atol=1.0e-10
    )