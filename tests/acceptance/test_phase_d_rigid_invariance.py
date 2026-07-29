# Copyright Xingyu Chen.
# Tests rigid-transform invariance.

"""Tests rigid-transform invariance."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tests.support.phase_d_acceptance import (
    SOLVERS,
    SolverAdapter,
    rigid_transform,
    rough_scene,
    transmission_scene,
)
from witwin.channel.deployment import build_info

_FREQUENCY_HZ = 2.0e9
_SEEDS = (3, 7, 11)


def _require_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA torch is required")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native rigid-transform acceptance is not built")


def _adapter(name: str) -> SolverAdapter:
    return next(adapter for adapter in SOLVERS if adapter.name == name)


def _ratio(adapter: SolverAdapter, *, transformed: bool):
    wall = transmission_scene()
    empty = transmission_scene(empty=True)
    if transformed:
        wall = rigid_transform(wall)
        empty = rigid_transform(empty)
    return adapter.transmission(wall, _FREQUENCY_HZ) / adapter.transmission(
        empty, _FREQUENCY_HZ
    )


def _relative_gap(left: float, right: float) -> float:
    return abs(left - right) / max(
        abs(left), abs(right), torch.finfo(torch.float64).tiny
    )


def test_rigid_transform_fixture_preserves_relative_geometry() -> None:
    scene = rough_scene()
    transformed = rigid_transform(scene)
    transmitter = next(
        endpoint for endpoint in scene.endpoints if endpoint.role == "tx"
    )
    receiver = next(endpoint for endpoint in scene.endpoints if endpoint.role == "rx")
    transformed_transmitter = next(
        endpoint for endpoint in transformed.endpoints if endpoint.role == "tx"
    )
    transformed_receiver = next(
        endpoint for endpoint in transformed.endpoints if endpoint.role == "rx"
    )
    original_vertices = scene.structures[0].geometry.vertices
    transformed_vertices = transformed.structures[0].geometry.vertices

    torch.testing.assert_close(
        torch.cdist(original_vertices, original_vertices),
        torch.cdist(transformed_vertices, transformed_vertices),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    original_delta = receiver.origin - transmitter.position
    transformed_delta = transformed_receiver.origin - transformed_transmitter.position
    torch.testing.assert_close(
        torch.linalg.vector_norm(original_delta),
        torch.linalg.vector_norm(transformed_delta),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        torch.dot(receiver.polarization, original_delta),
        torch.dot(transformed_receiver.polarization, transformed_delta),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


@pytest.mark.parametrize("solver_name", ["path", "deterministic"])
def test_field_solver_transmission_is_rigid_transform_invariant(
    solver_name: str,
) -> None:
    _require_native()
    adapter = _adapter(solver_name)
    assert _ratio(adapter, transformed=True) == pytest.approx(
        _ratio(adapter, transformed=False), rel=5.0e-4, abs=5.0e-5
    )


@pytest.mark.parametrize("solver_name", ["path", "deterministic"])
def test_field_solver_scattering_power_is_rigid_transform_invariant(
    solver_name: str,
) -> None:
    _require_native()
    adapter = _adapter(solver_name)
    baseline = adapter.scattering(rough_scene(), 0, _FREQUENCY_HZ)
    transformed = adapter.scattering(rigid_transform(rough_scene()), 0, _FREQUENCY_HZ)

    assert math.isfinite(baseline) and baseline > 0.0
    assert math.isfinite(transformed) and transformed > 0.0
    assert _relative_gap(baseline, transformed) <= 3.0e-3


@pytest.mark.parametrize("solver_name", ["mc_basic", "bdpt"])
def test_power_solver_transmission_is_rigid_transform_invariant(
    solver_name: str,
) -> None:
    _require_native()
    adapter = _adapter(solver_name)
    baseline = float(_ratio(adapter, transformed=False))
    transformed = float(_ratio(adapter, transformed=True))
    assert baseline == pytest.approx(transformed, rel=5.0e-3, abs=1.0e-8)


@pytest.mark.parametrize("solver_name", ["mc_basic", "bdpt"])
def test_power_solver_scattering_is_statistically_rigid_transform_invariant(
    solver_name: str,
) -> None:
    _require_native()
    adapter = _adapter(solver_name)
    baseline = [
        adapter.scattering(rough_scene(), seed, _FREQUENCY_HZ) for seed in _SEEDS
    ]
    transformed_scene = rigid_transform(rough_scene())
    transformed = [
        adapter.scattering(transformed_scene, seed, _FREQUENCY_HZ) for seed in _SEEDS
    ]

    assert all(math.isfinite(value) and value > 0.0 for value in baseline + transformed)
    assert _relative_gap(float(np.mean(baseline)), float(np.mean(transformed))) <= 0.30