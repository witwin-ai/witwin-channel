import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.montecarlo.basic import Config, solve


def test_basic_ad_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="ad_mode"):
        Config(ad_mode="reverse")


def test_basic_ad_requires_fixed_topology():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic fixed-topology AD")

    with pytest.raises(RuntimeError, match="fixed_topology"):
        solve(empty_space_los_scene(), Config(ad_mode="vjp", fixed_topology=False))


def test_basic_ad_requires_fixed_seed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic fixed-topology AD")

    with pytest.raises(RuntimeError, match="fixed seed"):
        solve(empty_space_los_scene(), Config(ad_mode="vjp", requires_fixed_seed=False))


def test_basic_ad_rejects_reflection_and_diffraction_topology():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic fixed-topology AD")

    with pytest.raises(RuntimeError, match="LoS-only"):
        solve(empty_space_los_scene(), Config(ad_mode="vjp", components={"los", "reflection"}))
