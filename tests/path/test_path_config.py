import pytest

from witwin.channel_native.path import Config


def test_path_config_defaults_are_explicit():
    config = Config()

    assert config.max_depth == 1
    assert config.coupled_paths is False
    assert config.components == frozenset({"los", "reflection", "diffraction"})
    assert config.max_paths is None
    assert config.max_paths_scope == "per_pair"
    assert config.coupled_candidate_limit == 1_000_000
    assert config.sort_key == "receiver_transmitter_depth_component"
    assert config.diagnostics is False
    assert config.ad_mode == "none"


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

    with pytest.raises(ValueError, match="sort_key"):
        Config(sort_key="random")

    with pytest.raises(ValueError, match="ad_mode"):
        Config(ad_mode="vjp")


def test_path_config_rejects_invalid_coupled_requests_before_solve():
    with pytest.raises(RuntimeError, match="max_depth >= 2"):
        Config(
            max_depth=1,
            components={"reflection", "diffraction"},
            coupled_paths=True,
        )
    with pytest.raises(RuntimeError, match="both reflection and diffraction"):
        Config(max_depth=2, components={"reflection"}, coupled_paths=True)
