from __future__ import annotations

import numpy as np

from witwin.channel.deterministic import types as wt
from witwin.channel.deterministic.reflection import epc


def _bool_list(value) -> list[bool]:
    return [bool(v) for v in np.asarray(value).reshape(-1)]


class _RaydSurfaceScene:
    def __init__(self) -> None:
        self.calls = 0

    def intersect_rays_raw_with_prim(self, ray_origin, ray_dir, active, *, tmax=None):
        del ray_origin, ray_dir
        self.calls += 1
        assert tmax is not None
        return active, wt.Float([1.0e-3, 1.0e-3]), wt.UInt32([4, 7])

    def triangle_group_id(self, prim_idx_i32):
        return wt.Int32(
            np.where(
                np.asarray(prim_idx_i32) == 7,
                99,
                10,
            )
        )


def test_surface_contains_uses_rayd_group_hit_for_large_surface_groups() -> None:
    scene = _RaydSurfaceScene()
    hit_p = wt.Point3f([0.0, 1.0], [0.0, 0.0], [0.0, 0.0])
    prim_idx = wt.Int32([3, 3])
    valid_prim = wt.Bool([True, True])
    tri_data = {
        "v0": wt.Point3f([0.0], [0.0], [0.0]),
        "v1": wt.Point3f([0.0], [0.0], [0.0]),
        "v2": wt.Point3f([0.0], [0.0], [0.0]),
    }
    tri_surface_data = {"max_group_size": 219}

    contains = epc.surface_contains(
        hit_p,
        prim_idx,
        tri_data,
        tri_surface_data,
        valid_prim,
        scene=scene,
        plane_normal=wt.Vector3f([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]),
    )

    assert scene.calls >= 1
    assert _bool_list(contains) == [True, False]
