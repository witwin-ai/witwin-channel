from __future__ import annotations

import drjit as dr
import witwin as wt
import pytest
from tests._scene_helpers import box_drjit_geometry, build_scene
def _hit_ray():
    return wt.Ray(
        wt.Point3f([0.0], [0.0], [5.0]),
        wt.Vector3f([0.0], [0.0], [-1.0]),
    )


@pytest.mark.gpu
def test_rayd_query_backend_returns_expected_hit_distance_and_point():
    geometry = box_drjit_geometry(center=(0.0, 0.0, 1.0), size=2.0)
    scene_rd = build_scene(geometry, device="cuda")
    ray = _hit_ray()

    si_rd = scene_rd.ray_intersect(ray)

    assert scene_rd._rayd_scene is not None
    assert si_rd.is_valid()[0]
    assert float(si_rd.t[0]) == pytest.approx(3.0, abs=1e-5)
    assert float(si_rd.p.z[0]) == pytest.approx(2.0, abs=1e-5)
    assert scene_rd.ray_test(ray)[0]


@pytest.mark.gpu
def test_rayd_query_backend_preliminary_intersection_returns_surface_interaction():
    scene = build_scene(
        box_drjit_geometry(center=(0.0, 0.0, 1.0), size=2.0),
        device="cuda",
    )
    ray = _hit_ray()

    preliminary = scene.ray_intersect_preliminary(ray)
    surface = preliminary.compute_surface_interaction(ray)

    assert scene._rayd_scene is not None
    assert preliminary.is_valid()[0]
    assert surface.is_valid()[0]
    assert abs(float(preliminary.t[0]) - float(surface.t[0])) < 1e-6
    assert dr.width(surface.p.x) == 1

