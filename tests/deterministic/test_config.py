import pytest

from witwin.channel_native.deterministic import Config


def test_config_defaults_match_public_contract():
    config = Config()

    assert config.max_depth == 1
    assert config.max_diffraction_order == 1
    assert config.components == frozenset({"los", "reflection", "diffraction"})
    assert config.coherent is True
    assert config.return_field is True
    assert config.export_paths is False
    assert config.max_paths is None
    assert config.sort_key == "receiver_transmitter_depth_component"
    assert config.diagnostics is False
    assert config.require_reflection is False
    assert config.require_diffraction is False
    assert config.ad_mode == "none"


def test_config_normalizes_component_iterables_to_frozenset():
    config = Config(components=["los", "reflection"])

    assert config.components == frozenset({"los", "reflection"})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": -1}, "max_depth must be non-negative"),
        ({"max_diffraction_order": -1}, "max_diffraction_order must be 0 or 1"),
        ({"max_diffraction_order": 2}, "max_diffraction_order above 1"),
        ({"components": set()}, "components must be a non-empty subset"),
        ({"components": {"los", "scatter"}}, "components must be a non-empty subset"),
        ({"max_paths": 0}, "max_paths must be positive"),
        ({"sort_key": "unstable"}, "sort_key must be one of"),
        ({"ad_mode": "vjp"}, "deterministic fixed-topology AD is not enabled"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises((RuntimeError, ValueError), match=message):
        Config(**kwargs)
