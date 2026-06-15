import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.montecarlo.basic import Config, solve


def test_basic_ad_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="ad_mode"):
        Config(ad_mode="reverse")


@pytest.mark.parametrize("ad_mode", ["vjp", "jvp"])
def test_basic_ad_config_rejects_non_primal_modes(ad_mode):
    with pytest.raises(ValueError, match="ad_mode"):
        Config(ad_mode=ad_mode)
