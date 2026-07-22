import pytest

from witwin.channel.path import Config


def test_path_config_defaults_are_explicit():
    config = Config()

    assert config.max_depth == 1
    assert config.coupled_paths is False
    assert config.components == frozenset({"los", "reflection", "diffraction"})
    assert config.max_paths is None
    assert config.max_paths_scope == "per_pair"
    assert config.coupled_candidate_limit == 1_000_000
    assert config.ad_mode == "none"
    # ISB boundary taper (ADR-017) is DEFAULT-OFF with the projection-validated
    # width default; the switch never defaults on.
    assert config.isb_boundary_taper is False
    assert config.isb_boundary_taper_width == 0.5


def test_path_config_isb_boundary_taper_width_bounds():
    for width in (0.25, 0.5, 1.0, 4.0):
        assert Config(
            isb_boundary_taper=True, isb_boundary_taper_width=width
        ).isb_boundary_taper_width == width
    for width in (0.0, -1.0, 4.5):
        with pytest.raises(ValueError, match=r"isb_boundary_taper_width must be in"):
            Config(isb_boundary_taper_width=width)


def test_path_config_validates_inputs():
    with pytest.raises(ValueError, match="max_depth"):
        Config(max_depth=-1)

    with pytest.raises(ValueError, match="components"):
        Config(components={"scatter"})

    with pytest.raises(ValueError, match="max_paths"):
        Config(max_paths=0)
    with pytest.raises(ValueError, match="max_paths_scope"):
        Config(max_paths_scope="global")
    with pytest.raises(ValueError, match="coupled_candidate_limit"):
        Config(coupled_candidate_limit=0)
    with pytest.raises(ValueError, match="hard limit"):
        Config(coupled_candidate_limit=1_000_001)

    with pytest.raises(ValueError, match="ad_mode"):
        Config(ad_mode="forward")
    # Fixed-topology material/frequency AD (plan 07 AD-1).
    assert Config(ad_mode="vjp").ad_mode == "vjp"
    assert Config(ad_mode="jvp").ad_mode == "jvp"


@pytest.mark.parametrize(
    "field",
    (
        "path_capacity_per_pair",
        "diffraction_state_capacity",
        "reflection_candidate_capacity_per_pair",
    ),
)
def test_path_config_rejects_retired_capacity_fields(field):
    with pytest.raises(TypeError, match=rf"unexpected keyword argument '{field}'"):
        Config(**{field: 1})


def test_path_config_rejects_invalid_coupled_requests_before_solve():
    with pytest.raises(RuntimeError, match="max_depth >= 2"):
        Config(
            max_depth=1,
            components={"reflection", "diffraction"},
            coupled_paths=True,
        )
    with pytest.raises(RuntimeError, match="both reflection and diffraction"):
        Config(max_depth=2, components={"reflection"}, coupled_paths=True)
