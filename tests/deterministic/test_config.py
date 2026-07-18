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
    assert config.max_paths_scope == "global"
    assert config.sort_key == "receiver_transmitter_depth_component"
    assert config.diagnostics is False
    assert config.ad_mode == "none"
    # Coupled reflection-diffraction is opt-in (ADR-011); the default 1M limit
    # matches the path solver.
    assert config.coupled_paths is False
    assert config.coupled_candidate_limit == 1_000_000


def test_config_normalizes_component_iterables_to_frozenset():
    config = Config(components=["los", "reflection"])

    assert config.components == frozenset({"los", "reflection"})


def test_coupled_paths_accepts_reflection_diffraction_depth_two():
    config = Config(
        components={"los", "reflection", "diffraction"},
        max_depth=2,
        coupled_paths=True,
    )

    assert config.coupled_paths is True
    assert config.coupled_candidate_limit == 1_000_000


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"coupled_paths": True, "max_depth": 1},
            "coupled reflection-diffraction paths require max_depth >= 2",
        ),
        (
            {
                "coupled_paths": True,
                "max_depth": 2,
                "components": {"los", "reflection"},
            },
            "coupled paths require both reflection and diffraction",
        ),
        (
            {
                "coupled_paths": True,
                "max_depth": 2,
                "components": {"los", "diffraction"},
            },
            "coupled paths require both reflection and diffraction",
        ),
        ({"coupled_candidate_limit": 0}, "coupled_candidate_limit must be positive"),
        (
            {"coupled_candidate_limit": 2_000_000},
            "coupled_candidate_limit cannot exceed the hard limit",
        ),
    ],
)
def test_coupled_paths_config_validation(kwargs, message):
    with pytest.raises((RuntimeError, ValueError), match=message):
        Config(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": -1}, "max_depth must be non-negative"),
        ({"max_diffraction_order": -1}, "max_diffraction_order must be 0 or 1"),
        ({"max_diffraction_order": 2}, "max_diffraction_order above 1"),
        ({"components": set()}, "components must be a non-empty subset"),
        ({"components": {"los", "scatter"}}, "components must be a non-empty subset"),
        ({"max_paths": 0}, "max_paths must be positive"),
        ({"max_paths_scope": "receiver"}, "max_paths_scope must be"),
        ({"sort_key": "unstable"}, "sort_key must be one of"),
        ({"ad_mode": "forward"}, "deterministic ad_mode must be one of"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises((RuntimeError, ValueError), match=message):
        Config(**kwargs)
