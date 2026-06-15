import pytest

from witwin.channel_native.montecarlo.basic import Config


@pytest.mark.parametrize("ad_mode", ["vjp", "jvp"])
def test_basic_fixed_topology_ad_modes_are_not_enabled(ad_mode):
    with pytest.raises(ValueError, match="ad_mode"):
        Config(samples=128, components={"los"}, ad_mode=ad_mode)
