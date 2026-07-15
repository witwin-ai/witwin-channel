import pytest

from witwin.channel_native.montecarlo.basic import Config


@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_basic_fixed_topology_ad_modes_are_accepted(ad_mode):
    # Plan 07 AD-3: montecarlo.basic differentiates its incoherent power map
    # (materials from the compiled store, frequency, LoS endpoints).
    # Solve-time rejection of pending components is covered by
    # tests/montecarlo/basic/test_basic_ad_errors.py.
    assert Config(samples=128, components={"los"}, ad_mode=ad_mode).ad_mode == ad_mode
