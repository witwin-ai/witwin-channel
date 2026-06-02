from __future__ import annotations

import drjit as dr
import numpy as np
import pytest

from witwin.channel.core.scene import Scene
from witwin.channel.core.scene import scene as scene_module
from witwin.channel.montecarlo import types as wt
from witwin.core import Material, Mesh, Structure


def _bool_scalar(value) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _bool_list(value) -> list[bool]:
    return [bool(v) for v in np.asarray(value).reshape(-1)]


def _open_wall_scene() -> Scene:
    mesh = Mesh(
        vertices=(
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 3.0),
            (1.0, 0.0, 3.0),
        ),
        faces=((0, 1, 3), (0, 3, 2)),
        recenter=False,
        device="cpu",
    )
    return Scene(
        structures=[Structure(geometry=mesh, material=Material(), name="open_wall")],
        device="cpu",
    )


@pytest.mark.gpu
def test_segment_visible_uses_rayd_ignore_without_python_refire_loop() -> None:
    scene = _open_wall_scene()
    start = wt.Point3f(0.5, -1.0, 1.5)
    end = wt.Point3f(0.5, 1.0, 1.5)

    blocked = scene.segment_visible(start, end)
    dr.eval(blocked)
    assert _bool_scalar(blocked) is False

    def fail_refire_loop(*args, **kwargs):
        raise AssertionError("Python segment-visible re-fire loop should not run")

    scene.intersect_rays_raw_with_prim = fail_refire_loop
    scene.intersect_rays_with_prim = fail_refire_loop

    visible = scene.segment_visible(start, end, ignore_prim_idx=wt.Int32(0), max_ignored_hits=2)
    dr.eval(visible)
    assert _bool_scalar(visible) is True


@pytest.mark.gpu
def test_segment_visible_native_ignore_preserves_surface_group_semantics() -> None:
    scene = _open_wall_scene()
    group_id = scene.triangle_group_id(wt.Int32(0))
    start = wt.Point3f(0.5, -1.0, 1.5)
    end = wt.Point3f(0.5, 1.0, 1.5)

    visible = scene.segment_visible(start, end, ignore_surface_group_idx=group_id, max_ignored_hits=2)
    dr.eval(visible)
    assert _bool_scalar(visible) is True


@pytest.mark.gpu
def test_segment_visible_chunks_large_native_ignore_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = _open_wall_scene()
    group_id = scene.triangle_group_id(wt.Int32(0))
    start = wt.Point3f(wt.Float([0.5, 1.5]), wt.Float([-1.0, -1.0]), wt.Float([1.5, 1.5]))
    end = wt.Point3f(wt.Float([0.5, 1.5]), wt.Float([1.0, -2.0]), wt.Float([1.5, 1.5]))

    monkeypatch.setattr(scene_module, "_NATIVE_IGNORE_MAX_ENTRIES", 1)
    visible = scene.segment_visible(start, end, ignore_surface_group_idx=group_id)
    dr.eval(visible)

    assert _bool_list(visible) == [True, True]


@pytest.mark.gpu
def test_segment_visible_rejects_unsupported_structure_ignore_without_fallback() -> None:
    scene = _open_wall_scene()
    start = wt.Point3f(0.5, -1.0, 1.5)
    end = wt.Point3f(0.5, 1.0, 1.5)

    with pytest.raises(ValueError, match="ignore_structure_idx"):
        scene.segment_visible(start, end, ignore_structure_idx=wt.Int32(0))


@pytest.mark.gpu
def test_segment_visible_accepts_ignore_lists_past_legacy_rayd_limit() -> None:
    scene = _open_wall_scene()
    start = wt.Point3f(0.5, -1.0, 1.5)
    end = wt.Point3f(0.5, 1.0, 1.5)

    visible = scene.segment_visible(start, end, ignore_prim_idx=tuple(wt.Int32(0) for _ in range(9)))
    dr.eval(visible)
    assert _bool_scalar(visible) is True


@pytest.mark.gpu
def test_segment_pair_visible_uses_rayd_without_segment_visible_fallback() -> None:
    scene = _open_wall_scene()
    start = wt.Point3f(0.5, -1.0, 1.5)
    end = wt.Point3f(0.5, 1.0, 1.5)
    end_offset = wt.Point3f(1.5, -1.0, 1.5)

    def fail_segment_visible(*args, **kwargs):
        raise AssertionError("segment_pair_visible should not combine segment_visible fallbacks")

    scene.segment_visible = fail_segment_visible
    visible = scene.segment_pair_visible(start, end, end_offset)
    dr.eval(visible)
    assert _bool_scalar(visible) is False


@pytest.mark.gpu
def test_segment_chain_visible_uses_rayd_segment_ignores_without_single_segment_fallback() -> None:
    scene = _open_wall_scene()
    wall_group = scene.triangle_group_id(wt.Int32(0))
    points = (
        wt.Point3f(wt.Float([0.5, 1.5]), wt.Float([-1.0, -1.0]), wt.Float([1.5, 1.5])),
        wt.Point3f(wt.Float([0.5, 1.5]), wt.Float([1.0, -2.0]), wt.Float([1.5, 1.5])),
    )

    blocked = scene.segment_chain_visible(points)
    dr.eval(blocked)
    assert _bool_list(blocked) == [False, True]

    def fail_segment_visible(*args, **kwargs):
        raise AssertionError("segment_chain_visible should not loop through segment_visible")

    scene.segment_visible = fail_segment_visible
    visible = scene.segment_chain_visible(points, ignore_surface_group_idx_per_segment=((wall_group,),))
    dr.eval(visible)
    assert _bool_list(visible) == [True, True]


@pytest.mark.gpu
def test_segment_chain_visible_interleaves_multi_segment_ignore_table_for_rayd() -> None:
    scene = _open_wall_scene()
    wall_group = scene.triangle_group_id(wt.Int32(0))
    points = (
        wt.Point3f(wt.Float([0.5, 1.5]), wt.Float([-1.0, -1.0]), wt.Float([1.5, 1.5])),
        wt.Point3f(wt.Float([0.5, 1.5]), wt.Float([1.0, -2.0]), wt.Float([1.5, 1.5])),
        wt.Point3f(wt.Float([1.5, 1.5]), wt.Float([1.0, -3.0]), wt.Float([1.5, 1.5])),
    )

    blocked = scene.segment_chain_visible(points)
    ignored = scene.segment_chain_visible(
        points,
        ignore_surface_group_idx_per_segment=((wall_group,), None),
    )
    dr.eval(blocked, ignored)
    assert _bool_list(blocked) == [False, True]
    assert _bool_list(ignored) == [True, True]


@pytest.mark.gpu
def test_segment_chain_visible_accepts_ignore_lists_past_legacy_rayd_limit() -> None:
    scene = _open_wall_scene()
    points = (
        wt.Point3f(0.5, -1.0, 1.5),
        wt.Point3f(0.5, 1.0, 1.5),
    )

    visible = scene.segment_chain_visible(
        points,
        ignore_prim_idx_per_segment=(tuple(wt.Int32(0) for _ in range(9)),),
    )
    dr.eval(visible)
    assert _bool_scalar(visible) is True
