from __future__ import annotations

import drjit as dr
import pytest
import witwin as wt
from tests._scene_helpers import box_drjit_geometry, build_scene
@pytest.mark.gpu
def test_query_backend_preserves_drjit_translation_gradient_on_initial_build():
    cube_x = wt.Float(0.0)
    dr.enable_grad(cube_x)

    geometry = box_drjit_geometry(center=wt.Point3f(cube_x, 0.0, 1.0), size=2.0)
    scene = build_scene(geometry, device="cuda")
    ray = wt.Ray(
        wt.Point3f(-5.0, 0.0, 1.0),
        wt.Vector3f(1.0, 0.0, 0.0),
    )

    intersection = scene.ray_intersect(ray)
    assert intersection.is_valid()[0]

    dr.set_grad(cube_x, 1.0)
    dr.forward_to(intersection.t, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)

    assert float(dr.grad(intersection.t)[0]) == pytest.approx(1.0, abs=1e-6)
