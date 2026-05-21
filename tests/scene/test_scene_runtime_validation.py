from __future__ import annotations

import numpy as np
import pytest
import rayd
import torch
import witwin.channel as wt

from witwin.channel.core.scene import Mesh, ReceiverGrid, Scene, Transmitter
from witwin.channel.core.scene.edge_policy import EdgePolicy
from witwin.core import Material, Structure


def _tolist(value) -> list:
    return np.asarray(value).reshape(-1).tolist()


def test_scene_rejects_invalid_modes_and_duplicate_structure_names(triangle_mesh: Mesh) -> None:
    with pytest.raises(TypeError, match="edge_selection_mode"):
        Scene(device="cpu", edge_selection_mode="invalid")

    with pytest.raises(TypeError, match="boundary_edge_policy"):
        Scene(device="cpu", boundary_edge_policy="invalid")

    duplicate = [
        Structure(geometry=triangle_mesh, material=Material(name="m1"), name="dup"),
        Structure(geometry=Mesh(*triangle_mesh.to_mesh()), material=Material(name="m2"), name="dup"),
    ]
    with pytest.raises(ValueError, match="already exists"):
        Scene(structures=duplicate, device="cpu")


def test_scene_ray_queries_material_queries_and_sync(triangle_scene: Scene, triangle_mesh: Mesh) -> None:
    ray = rayd.Ray(wt.Point3f(0.2, 0.2, 1.0), wt.Vector3f(0.0, 0.0, -1.0))
    material = triangle_scene.triangle_material(wt.UInt32(0))
    intersection = triangle_scene.ray_intersect(ray)

    assert _tolist(triangle_scene.ray_test(ray)) == [True]
    assert _tolist(intersection.is_valid()) == [True]
    assert _tolist(intersection.prim_id) == [0]
    assert _tolist(material["eps_r"]) == pytest.approx([2.5])
    assert _tolist(material["sigma_e"]) == pytest.approx([0.1])
    assert _tolist(material["specified"]) == [True]
    assert _tolist(triangle_scene.gather_structure_indices(wt.UInt32(0))) == [0]

    triangle_mesh.update_vertices(
        torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        sync=True,
    )

    assert _tolist(triangle_scene.ray_test(ray)) == [False]
    assert triangle_scene._mesh_version >= 2


def test_scene_edge_cache_clone_and_add(wall_scene: Scene, wall_vertices: torch.Tensor) -> None:
    edge_policy = EdgePolicy(boundary_edge_policy="half_plane")
    first = wall_scene._wedge_pack_at(1.5, edge_policy=edge_policy)
    second = wall_scene._wedge_pack_at(1.5, edge_policy=edge_policy)

    assert wall_scene.diffraction_edge_count(edge_policy=edge_policy) == 4
    assert first is second
    assert first is not None
    assert _tolist(first.pos.z) == [0.0, 1.5, 1.5, 3.0]

    wall_scene.structures[0].geometry.update_vertices(
        wall_vertices + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        sync=True,
    )
    third = wall_scene._wedge_pack_at(1.5, edge_policy=edge_policy)
    clone = wall_scene.clone(device="cpu")

    assert third is not first
    assert clone is not wall_scene
    assert len(clone.structures) == 1
    with pytest.raises(TypeError, match="vertical_ratio"):
        wall_scene.clone(device="cpu", vertical_ratio=0.5)

    appended = Scene(device="cpu")
    appended.add(
        Structure(
            geometry=Mesh(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor([[0, 1, 2]], dtype=torch.int32),
            ),
            material=Material(name="triangle-material", eps_r=3.0, sigma_e=0.3),
            name="triangle",
        )
    )

    assert [structure.name for structure in appended.structures] == ["triangle"]


def test_scene_stores_named_transmitter_and_receiver_grid() -> None:
    tx = Transmitter(
        name="tx",
        position=(0.0, 0.0, 2.0),
        polarization=(0.0, 1.0, 0.0),
        power=2.5,
    )
    grid = ReceiverGrid(
        name="map",
        axis="z",
        position=1.0,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        grid_shape=(4, 4),
        polarization=None,
    )
    scene = Scene(structures=[], transmitters=[tx], receivers=[grid], device="cpu")

    assert scene.transmitter("tx") is tx
    assert scene.receiver("map") is grid


def test_empty_scene_returns_default_material_and_structure_indices() -> None:
    empty = Scene(structures=[], device="cpu")
    material = empty.triangle_material(wt.UInt32(0))
    structure_idx = empty.gather_structure_indices(wt.UInt32([0, 1]))

    assert _tolist(material["eps_r"]) == [1.0]
    assert _tolist(material["sigma_e"]) == [0.0]
    assert _tolist(material["specified"]) == [False]
    assert _tolist(material["structure_idx"]) == [-1]
    assert _tolist(material["valid"]) == [False]
    assert _tolist(structure_idx) == [-1, -1]
