"""Tracer initialization smoke tests."""

from __future__ import annotations

import pytest

from tests._scene_helpers import box_geometry, mesh_from_geometry
from witwin.channel import Material, Scene, Structure, Tracer


def _mesh_from_cube():
    return mesh_from_geometry(box_geometry(center=(0.0, 0.0, 2.0), size=4.0), device="cuda")


@pytest.mark.gpu
def test_tracer_initializes_from_core_declarative_scene():
    scene = Scene(
        structures=[
            Structure(
                geometry=_mesh_from_cube(),
                material=Material(eps_r=5.0),
                name="obstacle",
            )
        ],
        device="cuda",
    )

    tracer = Tracer(frequency=1e9, scene=scene)

    assert tracer.scene is scene
    assert tracer.wavelength > 0.0
    assert tracer.scene.n_diffraction_edges > 0

