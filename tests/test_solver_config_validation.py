# Copyright Xingyu Chen.
# Tests that the four solver Configs refuse bad field types and values the same way.

from __future__ import annotations

import math

import numpy as np
import pytest
from witwin.core import Scene

from tests.support.solvers import solver as _module

SOLVERS = ("deterministic", "path", "montecarlo.basic", "montecarlo.bdpt")


def _float_fields(config_type: type) -> list[str]:
    import dataclasses

    return [
        field.name
        for field in dataclasses.fields(config_type)
        if str(field.type) in {"float", "float | None"}
    ]


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("value", [1.5, True, "2", None, 2.0])
def test_integer_field_rejects_non_integers_with_type_error(solver: str, value: object) -> None:
    config_type = _module(solver).Config
    with pytest.raises(TypeError, match=rf"{config_type.__name__}\.max_depth must be an int, got"):
        config_type(max_depth=value)


@pytest.mark.parametrize("solver", SOLVERS)
def test_integer_field_accepts_integer_like_values(solver: str) -> None:
    config = _module(solver).Config(max_depth=np.int64(2))
    assert config.max_depth == 2
    assert type(config.max_depth) is int


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_every_float_field_rejects_non_finite_values(solver: str, value: float) -> None:
    config_type = _module(solver).Config
    names = _float_fields(config_type)
    if not names:
        pytest.skip("this solver has no float fields")
    for name in names:
        with pytest.raises(ValueError, match=rf"{config_type.__name__}\.{name} must be finite, got"):
            config_type(**{name: value})


@pytest.mark.parametrize("solver", SOLVERS)
def test_bool_field_rejects_integers(solver: str) -> None:
    name = "diagnostics" if solver == "montecarlo.basic" else "coupled_paths"
    config_type = _module(solver).Config
    with pytest.raises(TypeError, match=rf"{config_type.__name__}\.{name} must be True or False, got 1"):
        config_type(**{name: 1})


@pytest.mark.parametrize("solver", SOLVERS)
def test_bad_ad_mode_is_a_value_error_in_every_solver(solver: str) -> None:
    with pytest.raises(ValueError, match=r"ad_mode must be one of \['jvp', 'none', 'vjp'\], got 'x'"):
        _module(solver).Config(ad_mode="x")


@pytest.mark.parametrize("solver", SOLVERS)
def test_components_must_be_a_collection_not_a_string(solver: str) -> None:
    with pytest.raises(TypeError, match="components must be a set, frozenset, tuple or list"):
        _module(solver).Config(components="los")


@pytest.mark.parametrize("solver", ["deterministic", "path", "montecarlo.bdpt"])
def test_chain_depth_cap_names_the_solver_and_range(solver: str) -> None:
    with pytest.raises(ValueError, match=r"max_depth must be in \[0, 5\] when components include \['reflection'\], got 6"):
        _module(solver).Config(max_depth=6, components={"los", "reflection"})


def test_bdpt_power_heuristic_beta_must_be_finite() -> None:
    from witwin.channel.experimental.bdpt import Config

    with pytest.raises(ValueError, match="power_heuristic_beta must be finite"):
        Config(power_heuristic_beta=math.inf)


def test_nan_scattering_density_no_longer_silently_disables_scattering() -> None:
    from witwin.channel.radiomap import Coherent as Config

    with pytest.raises(ValueError, match="scattering_samples_per_m2 must be finite"):
        Config(scattering_samples_per_m2=math.nan)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("other", SOLVERS)
def test_solve_refuses_another_solvers_config(solver: str, other: str) -> None:
    if solver == other or _module(solver).entry == _module(other).entry:
        return  # radiomap.solve accepts both of its estimator configs
    wrong = _module(other).Config()
    entry = _module(solver).entry.replace(".", r"\.")
    with pytest.raises(TypeError, match=rf"{entry}\.solve\(\) requires config to be"):
        _module(solver).solve(Scene(), wrong, reference_frequency_hz=3.5e9)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solve_refuses_a_non_config(solver: str) -> None:
    with pytest.raises(TypeError, match="requires config to be"):
        _module(solver).solve(Scene(), {"max_depth": 1}, reference_frequency_hz=3.5e9)


@pytest.mark.parametrize("solver", ["deterministic", "path"])
def test_discovery_accepts_the_three_sources_and_refuses_others(solver: str) -> None:
    config_type = _module(solver).Config
    for value in ("auto", "exhaustive", "rays"):
        assert config_type(discovery=value).discovery == value
    assert config_type().discovery == "auto"
    for bad in ("tree", "Auto", ""):
        with pytest.raises(ValueError, match="discovery must be one of"):
            config_type(discovery=bad)
