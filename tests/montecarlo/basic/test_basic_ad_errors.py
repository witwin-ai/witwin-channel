import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.montecarlo.basic import Config, solve


def test_basic_ad_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="ad_mode"):
        Config(ad_mode="reverse")


@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_basic_ad_config_accepts_fixed_topology_modes(ad_mode):
    assert Config(ad_mode=ad_mode).ad_mode == ad_mode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
@pytest.mark.parametrize("component", ["scattering"])
def test_basic_ad_solve_rejects_pending_components(ad_mode, component):
    # Diffraction gained its AD companions with plan 07 AD-4b (see
    # tests/ad/test_mc_basic_ad.py); the Kirchhoff scattering map remains
    # the one explicitly rejected component.
    scene = empty_space_los_scene()
    config = Config(
        samples=64,
        components={"los", component},
        max_depth=1,
        ad_mode=ad_mode,
    )
    with pytest.raises(RuntimeError, match=component):
        solve(scene, config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_basic_ad_solve_rejects_reflection_depth_over_ad_cap(ad_mode, monkeypatch):
    # The reflection radiomap AD companions hard-cap contribution_depth in
    # the native kernels; the solver must name that cap and reject the
    # configuration at solve() time, before any forward launch (not
    # mid-backward). The monkeypatched trace entry proves no forward ran.
    from witwin.channel_native.core.kernels.ops import mc_reflection_ad_max_depth
    from witwin.channel_native.montecarlo.basic import raydn_components

    depth_cap = mc_reflection_ad_max_depth()
    assert depth_cap >= 1

    def _forbidden_forward(*args, **kwargs):
        raise AssertionError(
            "reflection forward launched before the AD depth-cap rejection"
        )

    monkeypatch.setattr(
        raydn_components, "raydn_trace_reflections_forward", _forbidden_forward
    )
    scene = empty_space_los_scene()
    config = Config(
        samples=64,
        components={"reflection"},
        max_depth=depth_cap + 1,
        ad_mode=ad_mode,
    )
    with pytest.raises(RuntimeError, match="reflection AD depth cap"):
        solve(scene, config)
