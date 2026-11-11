# Copyright Xingyu Chen.
# Tests that EnumeratedOptions is the single owner of the shared enumerated solver options.

from __future__ import annotations

from dataclasses import fields

import pytest

from witwin.channel import deterministic, path
from witwin.channel.propagation import consumer
from witwin.channel.propagation.enumerated import EnumeratedOptions

_SHARED = (
    "max_depth", "components", "max_paths", "max_paths_scope", "ad_mode", "coupled_paths",
    "coupled_candidate_limit", "scattering_samples_per_m2", "scattering_max_paths_per_pair",
    "scattering_power_threshold", "scattering_chain_max_depth", "scattering_chain_samples_per_m2",
    "scattering_chain_max_rows", "isb_boundary_taper", "isb_boundary_taper_width", "discovery",
)
_DETERMINISTIC_TERMINAL = (
    "coherent", "return_field", "export_paths", "sort_key", "diagnostics", "scattering_coherent",
)


def _names(config_type: type) -> tuple[str, ...]:
    return tuple(item.name for item in fields(config_type))


def test_shared_fields_have_one_owner():
    assert _names(EnumeratedOptions) == _SHARED
    assert issubclass(path.Config, EnumeratedOptions)
    assert issubclass(deterministic.Config, EnumeratedOptions)
    assert _names(path.Config) == _SHARED
    assert _names(deterministic.Config) == _SHARED + _DETERMINISTIC_TERMINAL
    for config_type in (path.Config, deterministic.Config):
        own = {name for name in vars(config_type).get("__annotations__", {}) if not name.startswith("_")}
        # A subclass may only re-default max_paths_scope; every other shared field is inherited.
        assert own & set(_SHARED) <= {"max_paths_scope"}


def test_entries_keep_their_max_paths_scope_rules():
    assert EnumeratedOptions().max_paths_scope == "global"
    assert deterministic.Config().max_paths_scope == "global"
    assert path.Config().max_paths_scope == "per_pair"
    with pytest.raises(ValueError, match="path max_paths_scope"):
        path.Config(max_paths_scope="global")
    assert deterministic.Config(max_paths_scope="per_pair").max_paths_scope == "per_pair"


@pytest.mark.parametrize("config_type", [EnumeratedOptions, path.Config, deterministic.Config])
def test_shared_validation_runs_for_every_entry(config_type):
    with pytest.raises(ValueError, match="max_depth must be in"):
        config_type(max_depth=6, components={"reflection"})
    with pytest.raises(ValueError, match="ad_mode must be one of"):
        config_type(ad_mode="forward")
    with pytest.raises(ValueError, match="discovery must be one of"):
        config_type(discovery="images")
    with pytest.raises(ValueError, match="max_depth >= 2"):
        config_type(components={"reflection", "diffraction"}, coupled_paths=True)
    with pytest.raises(TypeError, match="max_diffraction_order"):
        config_type(max_diffraction_order=0)
    assert config_type(components=["los"]).components == frozenset({"los"})


def test_consumer_builds_engine_options_without_a_private_config():
    assert not hasattr(consumer, "_ConsumerConfig")
