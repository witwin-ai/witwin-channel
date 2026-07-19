"""Phase D material acceptance across all four public solver families."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest
import torch

from tests.support.phase_d_acceptance import (
    SOLVERS,
    SolverAdapter,
    dispersive_multilayer,
    rough_scene,
    transmission_scene,
)
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.physics.oracle import layer_stack_rt

_FREQUENCIES_HZ = (2.0e9, 12.0e9)
_SEEDS = (3, 7, 11)


def _require_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA torch is required")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native material transport is not built")


def _adapter(name: str) -> SolverAdapter:
    return next(adapter for adapter in SOLVERS if adapter.name == name)


@lru_cache
def _transmission_ratio(adapter: SolverAdapter, frequency_hz: float, *, reverse=False):
    wall = adapter.transmission(transmission_scene(frequency_hz, reverse=reverse))
    empty = adapter.transmission(transmission_scene(frequency_hz, empty=True))
    assert abs(empty) > 0.0
    return wall / empty


@lru_cache
def _scattering(solver_name: str, reverse: bool, seed: int) -> float:
    return _adapter(solver_name).scattering(rough_scene(reverse=reverse), seed)


def _relative_gap(left: float, right: float) -> float:
    return abs(left - right) / max(
        abs(left), abs(right), torch.finfo(torch.float64).tiny
    )


def test_dispersive_multilayer_oracle_changes_and_is_passive() -> None:
    """The test material itself has a large, physical two-frequency signal."""

    material = dispersive_multilayer()
    coefficients = [
        layer_stack_rt(material.layer_parameters(frequency), 1.0, frequency)
        for frequency in _FREQUENCIES_HZ
    ]
    for coefficient in coefficients:
        values = (
            complex(coefficient.r_te),
            complex(coefficient.t_te),
            float(coefficient.R_te),
            float(coefficient.T_te),
            float(coefficient.A_te),
        )
        assert np.isfinite(values).all()
        assert all(0.0 <= value <= 1.0 + 1.0e-12 for value in values[2:])
        assert sum(values[2:]) == pytest.approx(1.0, abs=1.0e-12)

    assert abs(complex(coefficients[0].t_te) - complex(coefficients[1].t_te)) > 0.1
    assert abs(float(coefficients[0].T_te) - float(coefficients[1].T_te)) > 0.1


@pytest.mark.parametrize("solver_name", [adapter.name for adapter in SOLVERS])
@pytest.mark.parametrize("frequency_hz", _FREQUENCIES_HZ)
def test_multilayer_transmission_is_finite_passive_and_matches_oracle(
    solver_name: str, frequency_hz: float
) -> None:
    _require_native()
    adapter = _adapter(solver_name)
    ratio = _transmission_ratio(adapter, frequency_hz)
    observed_power = (
        abs(ratio) ** 2 if adapter.field_domain == "complex" else float(ratio)
    )
    oracle = layer_stack_rt(
        dispersive_multilayer().layer_parameters(frequency_hz), 1.0, frequency_hz
    )

    assert math.isfinite(observed_power)
    assert 0.0 <= observed_power <= 1.0 + 1.0e-4
    assert observed_power == pytest.approx(float(oracle.T_te), rel=1.0e-3, abs=1.0e-8)
    if adapter.field_domain == "complex":
        # Solvers propagate over the straight geometric path, so the thin-sheet
        # interaction de-embeds the same thickness of vacuum propagation. This
        # also makes a multilayer vacuum wall exactly transparent.
        thickness_m = sum(layer.thickness_m for layer in dispersive_multilayer().layers)
        vacuum_phase = np.exp(
            1.0j * 2.0 * np.pi * frequency_hz * thickness_m / 299_792_458.0
        )
        expected_field = complex(oracle.t_te) * vacuum_phase
        assert ratio == pytest.approx(expected_field, rel=2.0e-3, abs=2.0e-4)


@pytest.mark.parametrize("solver_name", [adapter.name for adapter in SOLVERS])
def test_multilayer_transmission_changes_with_frequency_and_is_reciprocal(
    solver_name: str,
) -> None:
    _require_native()
    adapter = _adapter(solver_name)
    forward = [_transmission_ratio(adapter, frequency) for frequency in _FREQUENCIES_HZ]
    reverse = [
        _transmission_ratio(adapter, frequency, reverse=True)
        for frequency in _FREQUENCIES_HZ
    ]

    if adapter.field_domain == "complex":
        assert abs(forward[0] - forward[1]) > 0.1
        for lhs, rhs in zip(forward, reverse):
            assert lhs == pytest.approx(rhs, rel=3.0e-3, abs=3.0e-4)
    else:
        assert abs(float(forward[0]) - float(forward[1])) > 0.1
        for lhs, rhs in zip(forward, reverse):
            assert float(lhs) == pytest.approx(float(rhs), rel=2.0e-3, abs=1.0e-8)


def test_four_solver_transmission_domains_agree() -> None:
    """Field solvers agree in complex space; MC solvers agree in power space."""

    _require_native()
    for frequency in _FREQUENCIES_HZ:
        ratios = {
            adapter.name: _transmission_ratio(adapter, frequency) for adapter in SOLVERS
        }
        path_power = abs(ratios["path"]) ** 2
        assert ratios["deterministic"] == pytest.approx(
            ratios["path"], rel=2.0e-3, abs=2.0e-4
        )
        assert float(ratios["mc_basic"]) == pytest.approx(
            path_power, rel=2.0e-3, abs=1.0e-8
        )
        assert float(ratios["bdpt"]) == pytest.approx(
            path_power, rel=2.0e-3, abs=1.0e-8
        )


@pytest.mark.parametrize("solver_name", [adapter.name for adapter in SOLVERS])
def test_rough_scattering_is_finite_positive_and_reciprocal(solver_name: str) -> None:
    _require_native()
    adapter = _adapter(solver_name)
    seeds = (0,) if adapter.field_domain == "complex" else _SEEDS
    forward = [_scattering(solver_name, False, seed) for seed in seeds]
    reverse = [_scattering(solver_name, True, seed) for seed in seeds]

    assert all(math.isfinite(value) and value > 0.0 for value in forward + reverse)
    forward_mean = float(np.mean(forward))
    reverse_mean = float(np.mean(reverse))
    tolerance = 0.04 if adapter.field_domain == "complex" else 0.30
    assert _relative_gap(forward_mean, reverse_mean) <= tolerance


def test_four_solver_rough_scattering_agrees_at_each_capability_level() -> None:
    _require_native()
    path = _scattering("path", False, 0)
    deterministic = _scattering("deterministic", False, 0)
    basic = np.mean([_scattering("mc_basic", False, seed) for seed in _SEEDS])
    bdpt = np.mean([_scattering("bdpt", False, seed) for seed in _SEEDS])

    # Path/Deterministic expose the same polarized quadrature, while Basic and
    # BDPT expose stochastic aggregate power. Cross-domain equality is not a
    # public contract: Basic's scattering map is explicitly unpolarized.
    assert _relative_gap(path, deterministic) <= 0.08
    assert _relative_gap(float(basic), float(bdpt)) <= 0.30
