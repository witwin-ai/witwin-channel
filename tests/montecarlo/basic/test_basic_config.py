import pytest

from witwin.channel.montecarlo.basic import Config


def test_basic_config_defaults_are_explicit():
    config = Config()

    assert config.samples == 4096
    assert config.max_depth == 1
    assert config.seed == 0
    assert config.components == frozenset({"los", "reflection", "diffraction"})
    assert config.diagnostics is False


def test_basic_config_validates_samples_and_components():
    with pytest.raises(ValueError, match="samples"):
        Config(samples=0)

    with pytest.raises(ValueError, match="components"):
        Config(components={"scatter"})


def test_basic_config_rejects_negative_workspace_limit():
    with pytest.raises(ValueError, match="workspace_limit_bytes"):
        Config(workspace_limit_bytes=-1)
