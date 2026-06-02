from __future__ import annotations

import witwin as wt
import pytest
import witwin.channel.scene.builder as scene_builder

from tests._scene_helpers import box_drjit_geometry, build_scene
def _hit_ray():
    return wt.Ray(
        wt.Point3f([0.0], [0.0], [5.0]),
        wt.Vector3f([0.0], [0.0], [-1.0]),
    )


@pytest.mark.gpu
def test_scene_builds_rayd_query_and_wedge_runtime_by_default():
    geometry = box_drjit_geometry(center=(0.0, 0.0, 1.0), size=2.0)
    ray = _hit_ray()

    scene_rayd = build_scene(geometry, device="cuda")
    assert scene_rayd._wedge_backend_kind == "rayd"
    assert scene_rayd._rayd_scene is not None

    si_rayd = scene_rayd.ray_intersect(ray)

    assert si_rayd.is_valid()[0]


@pytest.mark.gpu
def test_rayd_backend_failure_raises_without_fallback(monkeypatch):
    geometry = box_drjit_geometry(center=(0.0, 0.0, 1.0), size=2.0)

    def _fail_build(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("rayd build failed")

    monkeypatch.setattr(scene_builder, "build_rayd_scene", _fail_build)

    with pytest.raises(RuntimeError, match="Failed to build the RayD runtime scene"):
        build_scene(geometry, device="cuda")


@pytest.mark.gpu
def test_empty_scene_still_uses_rayd_runtime_without_fallback():
    scene = build_scene(device="cuda")
    ray = _hit_ray()

    assert scene._wedge_backend_kind == "rayd"
    assert scene._rayd_scene is not None
    assert not scene.ray_intersect(ray).is_valid()[0]
