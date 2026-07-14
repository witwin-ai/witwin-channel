import pytest

from witwin.channel_native.montecarlo.basic import Config


@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_basic_fixed_topology_ad_modes_are_accepted(ad_mode):
    # Plan 07 AD-3: montecarlo.basic differentiates its incoherent power map
    # (materials from the compiled store, frequency, LoS endpoints).
    assert Config(samples=128, components={"los"}, ad_mode=ad_mode).ad_mode == ad_mode


@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
@pytest.mark.parametrize("component", ["scattering"])
def test_basic_ad_modes_reject_pending_components_at_solve(ad_mode, component):
    # The config stays permissive (component availability is a solve-time
    # question); the solver rejects the one pending map (scattering) before
    # any launch. Diffraction differentiates since plan 07 AD-4b.
    config = Config(
        samples=128, components={"los", component}, max_depth=1, ad_mode=ad_mode
    )
    assert component in config.components
