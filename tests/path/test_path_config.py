import pytest

from witwin.channel_native.path import Config


def test_path_config_defaults_are_explicit():
    config = Config()

    assert config.max_depth == 1
    assert config.components == frozenset({"los", "reflection", "diffraction"})
    assert config.max_paths is None
    assert config.sort_key == "receiver_transmitter_depth_component"
    assert config.diagnostics is False
    assert config.require_reflection is False
    assert config.require_diffraction is False
    assert config.ad_mode == "none"


def test_path_config_validates_inputs():
    with pytest.raises(ValueError, match="max_depth"):
        Config(max_depth=-1)

    with pytest.raises(ValueError, match="components"):
        Config(components={"scatter"})

    with pytest.raises(ValueError, match="max_paths"):
        Config(max_paths=0)

    with pytest.raises(ValueError, match="sort_key"):
        Config(sort_key="random")

    with pytest.raises(ValueError, match="ad_mode"):
        Config(ad_mode="vjp")
