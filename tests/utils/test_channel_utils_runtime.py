"""Unit tests for the shared scene-aware helpers in ``witwin.channel.core.runtime``.

These exercise the helpers in isolation against a duck-typed scene fixture so
they can run without spinning up ``witwin.channel.core.scene`` or the solvers.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import drjit as dr
import pytest
import witwin.channel as wt

from witwin.channel.core.numerics.constants import SPEED_OF_LIGHT
from witwin.channel.core.runtime import (
    assert_scene_materials_complete,
    point_grad_enabled,
    scene_geometry_grad_enabled,
    scene_material_grad_enabled,
)
from witwin.channel.core.physics.wave_math import material_angular_frequency


# ---------------------------------------------------------------------------
# Duck-typed scene helpers
# ---------------------------------------------------------------------------


def _make_scene(tri_data=None, vertices=None):
    """Build a minimal scene stub matching the duck-typed contract."""
    return SimpleNamespace(
        _merged_vertices=lambda: vertices,
        _triangle_runtime=lambda: tri_data,
    )


def _point(value: float):
    return wt.Point3f(wt.Float(value), wt.Float(value), wt.Float(value))


# ---------------------------------------------------------------------------
# material_angular_frequency
# ---------------------------------------------------------------------------


def test_material_angular_frequency_matches_definition():
    wavelength = 0.005
    expected = 2.0 * math.pi * SPEED_OF_LIGHT / wavelength
    omega = material_angular_frequency(wt.Float(wavelength))
    assert float(omega[0]) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# assert_scene_materials_complete
# ---------------------------------------------------------------------------


def test_assert_scene_materials_complete_none_scene_is_noop():
    assert_scene_materials_complete(None)


def test_assert_scene_materials_complete_no_triangle_runtime_is_noop():
    assert_scene_materials_complete(_make_scene(tri_data=None))


def test_assert_scene_materials_complete_missing_flag_raises():
    scene = _make_scene(tri_data={})
    with pytest.raises(RuntimeError, match="material_specified"):
        assert_scene_materials_complete(scene)


def test_assert_scene_materials_complete_incomplete_table_raises():
    specified = wt.Bool([True, False, True])
    scene = _make_scene(tri_data={"material_specified": specified})
    with pytest.raises(RuntimeError, match="every triangle"):
        assert_scene_materials_complete(scene)


def test_assert_scene_materials_complete_full_table_passes():
    specified = wt.Bool([True, True, True])
    scene = _make_scene(tri_data={"material_specified": specified})
    assert_scene_materials_complete(scene)


# ---------------------------------------------------------------------------
# point_grad_enabled
# ---------------------------------------------------------------------------


def test_point_grad_enabled_returns_false_for_none():
    assert point_grad_enabled(None) is False


def test_point_grad_enabled_returns_false_when_no_grad():
    point = _point(1.0)
    assert point_grad_enabled(point) is False


def test_point_grad_enabled_returns_true_when_grad_enabled():
    point = _point(1.0)
    dr.enable_grad(point.x)
    try:
        assert point_grad_enabled(point) is True
    finally:
        dr.disable_grad(point.x)


# ---------------------------------------------------------------------------
# scene_geometry_grad_enabled / scene_material_grad_enabled
# ---------------------------------------------------------------------------


def test_scene_geometry_grad_enabled_none_scene_is_false():
    assert scene_geometry_grad_enabled(None) is False


def test_scene_geometry_grad_enabled_returns_true_for_grad_vertices():
    vertices = _point(0.5)
    dr.enable_grad(vertices.y)
    try:
        scene = _make_scene(tri_data=None, vertices=vertices)
        assert scene_geometry_grad_enabled(scene) is True
    finally:
        dr.disable_grad(vertices.y)


def test_scene_geometry_grad_enabled_picks_up_triangle_payload():
    v0 = _point(0.1)
    dr.enable_grad(v0.z)
    try:
        scene = _make_scene(tri_data={"v0": v0, "v1": None, "v2": None})
        assert scene_geometry_grad_enabled(scene) is True
    finally:
        dr.disable_grad(v0.z)


def test_scene_geometry_grad_enabled_returns_false_when_nothing_tracks():
    scene = _make_scene(tri_data={"v0": _point(0.0), "v1": _point(0.0), "v2": _point(0.0)},
                        vertices=_point(0.0))
    assert scene_geometry_grad_enabled(scene) is False


def test_scene_material_grad_enabled_none_scene_is_false():
    assert scene_material_grad_enabled(None) is False


def test_scene_material_grad_enabled_picks_up_grad_array():
    eps_r = wt.Float([1.0, 2.0, 3.0])
    dr.enable_grad(eps_r)
    try:
        scene = _make_scene(tri_data={"material_eps_r": eps_r, "material_sigma_e": None})
        assert scene_material_grad_enabled(scene) is True
    finally:
        dr.disable_grad(eps_r)


def test_scene_material_grad_enabled_returns_false_without_grad():
    scene = _make_scene(tri_data={
        "material_eps_r": wt.Float([1.0]),
        "material_sigma_e": wt.Float([0.0]),
    })
    assert scene_material_grad_enabled(scene) is False
