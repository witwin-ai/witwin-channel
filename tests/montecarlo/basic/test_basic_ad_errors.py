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
@pytest.mark.parametrize("component", ["diffraction", "scattering"])
def test_basic_ad_solve_rejects_pending_components(ad_mode, component):
    scene = empty_space_los_scene()
    config = Config(
        samples=64,
        components={"los", component},
        max_depth=1,
        ad_mode=ad_mode,
    )
    with pytest.raises(RuntimeError, match=component):
        solve(scene, config)
