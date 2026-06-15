import pytest

from witwin.channel_native.montecarlo.basic import Config


def test_basic_config_defaults_are_explicit():
    config = Config()

    assert config.samples == 4096
    assert config.max_depth == 1
    assert config.seed == 0
    assert config.components == frozenset({"los", "reflection", "diffraction"})
    assert config.accumulation_strategy == "atomic_add"
    assert config.diagnostics is False


def test_basic_config_validates_samples_and_components():
    with pytest.raises(ValueError, match="samples"):
        Config(samples=0)

    with pytest.raises(ValueError, match="components"):
        Config(components={"scatter"})


def test_basic_config_rejects_unknown_accumulation_strategy():
    with pytest.raises(ValueError, match="accumulation_strategy"):
        Config(accumulation_strategy="python_loop")
