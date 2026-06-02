from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import rayd
import torch
import witwin.channel as wt

from witwin.channel.core.scene.builder import SceneBuilder


def _triangle_mesh_dict() -> dict:
    return {
        "vertices": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "faces": torch.tensor([[0, 1, 2]], dtype=torch.int32),
    }


def test_build_rayd_scene_constructs_queryable_runtime() -> None:
    rayd_scene = SceneBuilder.build_rayd_scene([_triangle_mesh_dict()])
    ray = rayd.RayAD(wt.Point3f(0.2, 0.2, 1.0), wt.Vector3f(0.0, 0.0, -1.0))

    assert np.asarray(rayd_scene.shadow_test(ray)).reshape(-1).tolist() == [True]


def test_configure_runtime_backends_rebuilds_and_updates_vertices() -> None:
    mesh = _triangle_mesh_dict()
    runtime = SimpleNamespace(_structure_meshes=[mesh], _rayd_scene=None)
    ray = rayd.RayAD(wt.Point3f(0.2, 0.2, 1.0), wt.Vector3f(0.0, 0.0, -1.0))

    SceneBuilder.configure_runtime_backends(runtime)
    assert np.asarray(runtime._rayd_scene.shadow_test(ray)).reshape(-1).tolist() == [True]

    moved_vertices = mesh["vertices"] + torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)
    SceneBuilder.configure_runtime_backends(runtime, update_mesh_id=0, vertices=moved_vertices)

    assert np.asarray(runtime._rayd_scene.shadow_test(ray)).reshape(-1).tolist() == [False]
