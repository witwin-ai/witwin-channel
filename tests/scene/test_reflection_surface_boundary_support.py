from __future__ import annotations

import numpy as np
import pytest

import witwin.channel as wt
from witwin.channel.core.scene import EdgePolicy, Scene
from witwin.channel.deterministic.reflection.boundary import nearest_surface_boundary_edge
from witwin.core import Material, Mesh, Structure


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _int_scalar(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _bool_scalar(value) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _point_tuple(value) -> tuple[float, float, float]:
    return (_scalar(value.x), _scalar(value.y), _scalar(value.z))


def _open_wall_scene() -> Scene:
    mesh = Mesh(
        vertices=(
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 3.0),
            (1.0, 0.0, 3.0),
        ),
        faces=((0, 1, 3), (0, 3, 2)),
        device="cpu",
    )
    return Scene(
        structures=[Structure(geometry=mesh, material=Material(), name="open_wall")],
        device="cpu",
    )


def _two_face_wedge_scene() -> Scene:
    mesh = Mesh(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        faces=((0, 1, 2), (0, 3, 1)),
        device="cpu",
    )
    return Scene(
        structures=[Structure(geometry=mesh, material=Material(), name="wedge")],
        device="cpu",
    )


def _edge_policy() -> EdgePolicy:
    return EdgePolicy(
        edge_diffraction=True,
        boundary_edge_policy="half_plane",
        edge_selection_mode="all_edges",
    )


def test_reflection_boundary_support_is_disabled_in_hard_mode() -> None:
    scene = _open_wall_scene()
    edge_policy = _edge_policy()
    scene.diffraction_edge_count(edge_policy=edge_policy)

    support = nearest_surface_boundary_edge(
        scene=scene,
        prim_idx=wt.Int32(0),
        hit_p=wt.Point3f(-0.99, 0.0, 0.0),
        mode="hard",
        wavelength=0.1,
        boundary_radius_wavelengths=2.0,
        edge_policy=edge_policy,
    )

    assert _bool_scalar(support.valid) is False
    assert _int_scalar(support.edge_idx) == -1


def test_reflection_boundary_support_is_invalid_without_edge_runtime() -> None:
    scene = _open_wall_scene()
    edge_policy = EdgePolicy(edge_diffraction=False)
    assert scene.diffraction_edge_count(edge_policy=edge_policy) == 0

    support = nearest_surface_boundary_edge(
        scene=scene,
        prim_idx=wt.Int32(0),
        hit_p=wt.Point3f(-0.99, 0.0, 0.0),
        mode="f_weight_reference",
        wavelength=0.1,
        boundary_radius_wavelengths=2.0,
        edge_policy=edge_policy,
    )

    assert _bool_scalar(support.valid) is False
    assert _int_scalar(support.edge_idx) == -1


def test_reflection_boundary_support_uses_surface_group_edges_not_internal_diagonal() -> None:
    scene = _open_wall_scene()
    edge_policy = _edge_policy()
    assert scene.diffraction_edge_count(edge_policy=edge_policy) == 4

    support = nearest_surface_boundary_edge(
        scene=scene,
        prim_idx=wt.Int32(0),
        hit_p=wt.Point3f(0.0, 0.0, 0.0),
        mode="f_weight_reference",
        wavelength=0.1,
        boundary_radius_wavelengths=0.5,
        edge_policy=edge_policy,
    )

    assert _bool_scalar(support.valid) is False
    assert _int_scalar(support.edge_idx) == -1


def test_reflection_boundary_support_returns_nearest_boundary_edge_payload() -> None:
    scene = _open_wall_scene()
    edge_policy = _edge_policy()
    scene.diffraction_edge_count(edge_policy=edge_policy)

    support = nearest_surface_boundary_edge(
        scene=scene,
        prim_idx=wt.Int32(1),
        hit_p=wt.Point3f(-0.99, 0.0, 0.0),
        mode="f_weight_reference",
        wavelength=0.1,
        boundary_radius_wavelengths=2.0,
        edge_policy=edge_policy,
    )

    assert _bool_scalar(support.valid) is True
    assert _int_scalar(support.edge_idx) >= 0
    assert _scalar(support.distance) == pytest.approx(0.01, abs=1e-5)

    edge_v0 = _point_tuple(support.edge_v0)
    edge_v1 = _point_tuple(support.edge_v1)
    assert edge_v0[0] == pytest.approx(-1.0)
    assert edge_v1[0] == pytest.approx(-1.0)
    assert sorted((edge_v0[2], edge_v1[2])) == pytest.approx([-1.5, 1.5])


def test_reflection_boundary_support_returns_adjacent_faces_for_wedge_edge() -> None:
    scene = _two_face_wedge_scene()
    edge_policy = _edge_policy()
    scene.diffraction_edge_count(edge_policy=edge_policy)

    support = nearest_surface_boundary_edge(
        scene=scene,
        prim_idx=wt.Int32(0),
        hit_p=wt.Point3f(0.0, -0.49, -0.5),
        mode="f_weight_reference",
        wavelength=0.1,
        boundary_radius_wavelengths=2.0,
        edge_policy=edge_policy,
    )

    assert _bool_scalar(support.valid) is True
    assert _scalar(support.distance) < 0.02
    assert {_int_scalar(support.adjacent_face0), _int_scalar(support.adjacent_face1)} == {0, 1}
