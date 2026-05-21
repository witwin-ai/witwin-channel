from __future__ import annotations

import math

import drjit as dr
import numpy as np
import pytest

from witwin.channel.deterministic import types as wt
from witwin.channel.deterministic.reflection.boundary import ReflectionBoundarySupport
from witwin.channel.deterministic.reflection.f_weight import reflection_transition_weights


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _bool_scalar(value) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _support(*, distance, valid: bool = True) -> ReflectionBoundarySupport:
    return ReflectionBoundarySupport(
        edge_idx=wt.Int32(0 if valid else -1),
        global_edge_idx=wt.Int32(7 if valid else -1),
        edge_pos=wt.Point3f(0.0, 0.0, 0.0),
        edge_dir=wt.Vector3f(0.0, 1.0, 0.0),
        edge_v0=wt.Point3f(0.0, -1.0, 0.0),
        edge_v1=wt.Point3f(0.0, 1.0, 0.0),
        edge_length=wt.Float(2.0 if valid else 0.0),
        distance=wt.Float(distance),
        adjacent_face0=wt.Int32(0 if valid else -1),
        adjacent_face1=wt.Int32(1 if valid else -1),
        n0=wt.Vector3f(0.0, 0.0, 1.0),
        n_face_n=wt.Vector3f(-1.0, 0.0, 0.0),
        valid=wt.Bool(valid),
    )


def _weights(distance, *, primary_side: bool, valid_support: bool = True):
    return reflection_transition_weights(
        hit_p=wt.Point3f(distance, 0.0, 0.0),
        previous_point=wt.Point3f(distance, 0.0, -1.0),
        next_point=wt.Point3f(distance, 0.0, 1.0),
        primary_plane_point=wt.Point3f(0.5, 0.0, 0.0),
        primary_plane_normal=wt.Vector3f(0.0, 0.0, 1.0),
        edge_support=_support(distance=abs(distance), valid=valid_support),
        wave_k=wt.Float(2.0 * math.pi / 0.1),
        primary_side_mask=wt.Bool(primary_side),
    )


def test_reflection_f_weight_returns_unit_primary_when_no_boundary_support() -> None:
    weights = _weights(0.5, primary_side=True, valid_support=False)

    assert _scalar(weights.primary_weight.real) == pytest.approx(1.0)
    assert _scalar(weights.primary_weight.imag) == pytest.approx(0.0)
    assert _scalar(weights.adjacent_weight.real) == pytest.approx(0.0)
    assert _scalar(weights.adjacent_weight.imag) == pytest.approx(0.0)
    assert _bool_scalar(weights.adjacent_valid) is False


def test_reflection_f_weight_zeroes_primary_outside_surface_without_boundary_support() -> None:
    weights = _weights(-0.5, primary_side=False, valid_support=False)

    assert _scalar(weights.primary_weight.real) == pytest.approx(0.0)
    assert _scalar(weights.primary_weight.imag) == pytest.approx(0.0)
    assert _bool_scalar(weights.adjacent_valid) is False


def test_reflection_f_weight_zeroes_primary_at_boundary() -> None:
    weights = _weights(0.0, primary_side=True)

    assert _scalar(weights.primary_weight.real) == pytest.approx(0.0, abs=1e-6)
    assert _scalar(weights.primary_weight.imag) == pytest.approx(0.0, abs=1e-6)
    assert _bool_scalar(weights.adjacent_valid) is False


def test_reflection_f_weight_sweep_stays_finite_from_boundary_to_interior() -> None:
    previous_magnitude = 0.0
    for distance in (0.0, 0.005, 0.01, 0.02, 0.04):
        weights = _weights(distance, primary_side=True)
        value = complex(_scalar(weights.primary_weight.real), _scalar(weights.primary_weight.imag))
        magnitude = abs(value)
        assert math.isfinite(value.real)
        assert math.isfinite(value.imag)
        assert magnitude < 2.0
        assert magnitude + 1e-5 >= previous_magnitude
        previous_magnitude = magnitude


def test_reflection_f_weight_emits_adjacent_weight_only_on_shadow_side() -> None:
    weights = _weights(-0.03, primary_side=False)

    assert _scalar(weights.primary_weight.real) == pytest.approx(0.0)
    assert _scalar(weights.primary_weight.imag) == pytest.approx(0.0)
    assert _bool_scalar(weights.adjacent_valid) is True
    assert abs(complex(_scalar(weights.adjacent_weight.real), _scalar(weights.adjacent_weight.imag))) > 0.0
    assert _scalar(weights.adjacent_plane_normal.x) == pytest.approx(-1.0)


def test_reflection_f_weight_has_finite_near_boundary_gradient() -> None:
    distance = wt.Float(0.02)
    dr.enable_grad(distance)
    weights = _weights(distance, primary_side=True)
    loss = weights.primary_weight.real * weights.primary_weight.real + weights.primary_weight.imag * weights.primary_weight.imag
    dr.backward(loss)

    grad = _scalar(dr.grad(distance))
    assert math.isfinite(grad)
