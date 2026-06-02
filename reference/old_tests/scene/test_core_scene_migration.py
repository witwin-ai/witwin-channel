"""Acceptance coverage for the declarative channel scene model."""

from __future__ import annotations

import pytest
import witwin as wt

import numpy as np
import drjit as dr
from tests._scene_helpers import box_drjit_geometry, box_geometry, mesh_from_geometry
from witwin.channel import Material, FieldMonitor, Scene, Structure, Tracer
def _make_drjit_scene():
    left = box_drjit_geometry(center=(-1.8, -1.2, 1.5), size=2.0)
    right = box_drjit_geometry(center=(1.8, 1.2, 1.5), size=2.0)
    return left, right, Scene(
        structures=[
            Structure(geometry=left, material=Material(eps_r=5.0), name="left"),
            Structure(geometry=right, material=Material(eps_r=5.0), name="right"),
        ],
        device="cuda",
    )


def _make_core_scene():
    return Scene(
        structures=[
            Structure(
                geometry=mesh_from_geometry(box_geometry(center=(-1.8, -1.2, 1.5), size=2.0), device="cuda"),
                material=Material(eps_r=5.0),
                name="left",
            ),
            Structure(
                geometry=mesh_from_geometry(box_geometry(center=(1.8, 1.2, 1.5), size=2.0), device="cuda"),
                material=Material(eps_r=5.0),
                name="right",
            ),
        ],
        device="cuda",
    )


@pytest.mark.gpu
@pytest.mark.acceptance
def test_core_scene_matches_drjit_scene_runtime_topology():
    _, _, drjit_scene = _make_drjit_scene()
    core_scene = _make_core_scene()

    assert dr_width(drjit_scene.vertices) == dr_width(core_scene.vertices)
    assert dr_width(drjit_scene.faces) == dr_width(core_scene.faces)
    assert drjit_scene.n_diffraction_edges == core_scene.n_diffraction_edges
    assert drjit_scene.edge_selection_summary == core_scene.edge_selection_summary


@pytest.mark.gpu
def test_scene_legacy_raw_constructor_is_removed():
    with pytest.raises(TypeError):
        Scene(
            box_geometry(center=(-1.8, -1.2, 1.5), size=2.0),
            box_geometry(center=(1.8, 1.2, 1.5), size=2.0),
        )
    assert not hasattr(Scene, "from_meshes")


@pytest.mark.gpu
def test_box_drjit_geometry_translation_preserves_ad():
    cube1_x = wt.Float(-2.5)
    dr.enable_grad(cube1_x)

    geometry = box_drjit_geometry(center=wt.Point3f(cube1_x, -3.0, 1.5), size=2.0)
    vertices, _ = geometry.to_mesh()
    objective = dr.sum(vertices.x)

    dr.backward(objective)

    assert float(dr.grad(cube1_x)[0]) == pytest.approx(8.0, abs=1e-6)


@pytest.mark.gpu
@pytest.mark.acceptance
def test_core_scene_update_vertices_refreshes_runtime_caches():
    scene = _make_core_scene()
    original_z = np.asarray(scene.vertices.z, dtype=np.float32)
    updated_vertices = wt.Point3f(
        scene.vertices.x,
        scene.vertices.y,
        scene.vertices.z + 0.125,
    )

    scene.update_vertices(updated_vertices)

    assert scene.tri_data_gpu is not None
    assert scene._mesh_version >= 2
    assert np.allclose(np.asarray(scene.vertices.z, dtype=np.float32), original_z + 0.125)


@pytest.mark.gpu
@pytest.mark.acceptance
@pytest.mark.validation
def test_tracer_trace_matches_between_drjit_and_core_scenes():
    _, _, drjit_scene = _make_drjit_scene()
    core_scene = _make_core_scene()

    tracer_drjit = Tracer(
        frequency=1e9,
        scene=drjit_scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    )
    tracer_core = Tracer(
        frequency=1e9,
        scene=core_scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    )

    tx = wt.Point3f(0.0, -4.0, 1.5)
    monitor = FieldMonitor(
        "comparison_plane",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_size=8,
    )
    drjit_result = tracer_drjit.trace(tx_pos=tx, monitor=monitor, verbose=False)
    core_result = tracer_core.trace(tx_pos=tx, monitor=monitor, verbose=False)

    drjit_total = np.asarray(drjit_result.primary.field.total.real) + 1j * np.asarray(drjit_result.primary.field.total.imag)
    core_total = np.asarray(core_result.primary.field.total.real) + 1j * np.asarray(core_result.primary.field.total.imag)

    assert drjit_total.shape == core_total.shape
    assert np.max(np.abs(drjit_total - core_total)) < 1e-6


def dr_width(value) -> int:
    import drjit as dr

    return int(dr.width(value))

