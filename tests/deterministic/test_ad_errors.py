import pytest
import torch

from witwin.channel_native.deterministic import Config
from witwin.channel_native.deterministic import solve
from tests.support.scenes import same_side_wall_reflection_scene


@pytest.mark.parametrize("ad_mode", ["vjp", "jvp", "forward"])
def test_deterministic_rejects_unsupported_ad_modes(ad_mode):
    with pytest.raises(RuntimeError, match="deterministic fixed-topology AD is not enabled"):
        Config(ad_mode=ad_mode)


def test_coherent_reflection_uses_complex_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic runtime validation")

    result = solve(same_side_wall_reflection_scene(), Config(components={"reflection"}, coherent=True))

    torch.testing.assert_close(result.path_gain, result.field.abs().square(), rtol=2.0e-4, atol=1.0e-10)
