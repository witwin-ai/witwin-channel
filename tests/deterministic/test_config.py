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
    # ISB boundary taper (ADR-017) is DEFAULT-OFF; the width default is the
    # projection-validated 0.5 but is inert while the switch is off.
    assert config.isb_boundary_taper is False
    assert config.isb_boundary_taper_width == 0.5


def test_isb_boundary_taper_accepts_width_bounds():
    for width in (0.25, 0.5, 1.0, 4.0):
        config = Config(isb_boundary_taper=True, isb_boundary_taper_width=width)
        assert config.isb_boundary_taper is True
        assert config.isb_boundary_taper_width == width


@pytest.mark.parametrize(
    ("width", "message"),
    [
        (0.0, r"isb_boundary_taper_width must be in \(0, 4\]"),
        (-0.5, r"isb_boundary_taper_width must be in \(0, 4\]"),
        (4.0001, r"isb_boundary_taper_width must be in \(0, 4\]"),
        (10.0, r"isb_boundary_taper_width must be in \(0, 4\]"),
    ],
)
def test_isb_boundary_taper_width_validation(width, message):
    # The width bound is validated regardless of the on/off flag so a bad width
    # is rejected at construction, not silently ignored while off.
    with pytest.raises(ValueError, match=message):
        Config(isb_boundary_taper_width=width)


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


@pytest.mark.parametrize(
    "field",
    (
        "path_capacity_per_pair",
        "diffraction_state_capacity",
        "reflection_candidate_capacity_per_pair",
    ),
)
def test_config_rejects_retired_capacity_fields(field):
    with pytest.raises(TypeError, match=rf"unexpected keyword argument '{field}'"):
        Config(**{field: 1})
