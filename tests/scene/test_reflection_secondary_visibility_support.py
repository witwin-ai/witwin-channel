from __future__ import annotations

import numpy as np
import pytest

import witwin.channel as wt
from witwin.channel.core.scene import EdgePolicy, Scene
from witwin.channel.deterministic.reflection.secondary_visibility import (
    nearest_blocker_silhouette_edge,
)
from witwin.core import Material, Mesh, Structure


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _int_scalar(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _bool_scalar(value) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _edge_policy() -> EdgePolicy:
    return EdgePolicy(
        edge_diffraction=True,
        boundary_edge_policy="half_plane",
        edge_selection_mode="all_edges",
    )


def _single_blocker_scene() -> Scene:
    mesh = Mesh(
        vertices=(
            (-1.0, 0.0, -1.0),
            (1.0, 0.0, -1.0),
            (-1.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
        ),
        faces=((0, 1, 3), (0, 3, 2)),
        device="cpu",
    )
    return Scene(
        structures=[Structure(geometry=mesh, material=Material(), name="blocker")],
        device="cpu",
    )


def test_secondary_visibility_support_is_disabled_in_hard_mode() -> None:
    scene = _single_blocker_scene()
    edge_policy = _edge_policy()
    scene.diffraction_edge_count(edge_policy=edge_policy)

    support = nearest_blocker_silhouette_edge(
        scene=scene,
        hit_p=wt.Point3f(0.0, -1.0, 0.0),
        rx_pos=wt.Point3f(0.0, 1.0, 0.0),
        primary_surface_group=wt.Int32(-1),
        mode="hard",
        wavelength=1.0,
        boundary_radius_wavelengths=2.0,
        edge_policy=edge_policy,
    )

    assert _bool_scalar(support.is_occluded) is False
    assert _bool_scalar(support.valid) is False
    assert _int_scalar(support.blocker_prim_idx) == -1


def test_secondary_visibility_support_reports_clear_segment() -> None:
    scene = _single_blocker_scene()
    edge_policy = _edge_policy()
    scene.diffraction_edge_count(edge_policy=edge_policy)

    support = nearest_blocker_silhouette_edge(
        scene=scene,
        hit_p=wt.Point3f(2.0, -1.0, 0.0),
        rx_pos=wt.Point3f(2.0, 1.0, 0.0),
        primary_surface_group=wt.Int32(-1),
        mode="f_weight",
        wavelength=1.0,
        boundary_radius_wavelengths=2.0,
        edge_policy=edge_policy,
    )

    assert _bool_scalar(support.is_occluded) is False
    assert _bool_scalar(support.valid) is False
    assert _int_scalar(support.blocker_prim_idx) == -1


def test_secondary_visibility_support_reports_off_edge_blocker() -> None:
    scene = _single_blocker_scene()
    edge_policy = _edge_policy()
    scene.diffraction_edge_count(edge_policy=edge_policy)

    support = nearest_blocker_silhouette_edge(
        scene=scene,
        hit_p=wt.Point3f(0.0, -1.0, 0.0),
        rx_pos=wt.Point3f(0.0, 1.0, 0.0),
        primary_surface_group=wt.Int32(-1),
        mode="f_weight",
        wavelength=1.0,
        boundary_radius_wavelengths=2.0,
        edge_policy=edge_policy,
    )

    assert _bool_scalar(support.is_occluded) is True
    assert _bool_scalar(support.valid) is True
    assert _int_scalar(support.blocker_prim_idx) >= 0
    assert _int_scalar(support.silhouette_edge_idx) >= 0
    assert _scalar(support.gamma) == pytest.approx(1.0, abs=1e-4)
    assert _scalar(support.effective_L) > 0.0


def test_secondary_visibility_support_reports_grazing_silhouette() -> None:
    scene = _single_blocker_scene()
    edge_policy = _edge_policy()
    scene.diffraction_edge_count(edge_policy=edge_policy)

    support = nearest_blocker_silhouette_edge(
        scene=scene,
        hit_p=wt.Point3f(0.99, -1.0, 0.0),
        rx_pos=wt.Point3f(0.99, 1.0, 0.0),
        primary_surface_group=wt.Int32(-1),
        mode="f_weight",
        wavelength=1.0,
        boundary_radius_wavelengths=2.0,
        edge_policy=edge_policy,
    )

    assert _bool_scalar(support.is_occluded) is True
    assert _bool_scalar(support.valid) is True
    assert _scalar(support.gamma) < 0.02
