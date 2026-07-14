import pytest
import torch

from witwin.channel_native.deterministic import Config
from witwin.channel_native.deterministic import solve
from tests.support.scenes import same_side_wall_reflection_scene


@pytest.mark.parametrize("ad_mode", ["vjp", "jvp"])
def test_deterministic_accepts_fixed_topology_ad_modes(ad_mode):
    assert Config(ad_mode=ad_mode).ad_mode == ad_mode


def test_deterministic_rejects_unknown_ad_modes():
    with pytest.raises(RuntimeError, match="ad_mode"):
        Config(ad_mode="forward")


def test_coherent_reflection_uses_complex_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic runtime validation")

    result = solve(same_side_wall_reflection_scene(), Config(components={"reflection"}, coherent=True))

    torch.testing.assert_close(result.path_gain, result.field.abs().square(), rtol=2.0e-4, atol=1.0e-10)
