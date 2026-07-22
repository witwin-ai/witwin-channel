import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel.montecarlo.basic import Config, solve


def test_basic_ad_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="ad_mode"):
        Config(ad_mode="reverse")


@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_basic_ad_config_accepts_fixed_topology_modes(ad_mode):
    assert Config(ad_mode=ad_mode).ad_mode == ad_mode


# ADR-015 Part A enabled Kirchhoff scattering AD in the MC-basic solver, so the
# former "pending component" rejection guard is gone; there are no remaining
# AD-unsupported components to reject. Enabled-path coverage lives in
# tests/ad/test_mc_basic_scattering_ad.py (acceptance protocol point 5).


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_basic_ad_solve_rejects_reflection_depth_over_ad_cap(ad_mode, monkeypatch):
    # The reflection radiomap AD companions hard-cap contribution_depth in
    # the native kernels; the solver must name that cap and reject the
    # configuration at solve() time, before any forward launch (not
    # mid-backward). The monkeypatched trace entry proves no forward ran.
    from witwin.channel.montecarlo.basic.kernels.maps import (
        mc_reflection_ad_max_depth,
    )
    from witwin.channel.propagation.geometry.kernels import (
        bridge as geometry_bridge,
    )

    depth_cap = mc_reflection_ad_max_depth()
    assert depth_cap >= 1

    def _forbidden_forward(*args, **kwargs):
        raise AssertionError(
            "reflection forward launched before the AD depth-cap rejection"
        )

    monkeypatch.setattr(
        geometry_bridge, "rayd_trace_reflections_forward", _forbidden_forward
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
