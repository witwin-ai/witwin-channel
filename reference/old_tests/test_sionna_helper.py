from __future__ import annotations

import pytest
from witwin.channel import (
    Box,
    Material,
    Mesh,
    Scene,
    Structure,
    load_sionna_rt,
    scene_from_sionna_scene,
    scene_to_sionna_scene,
)
from tests._scene_helpers import box_drjit_geometry, build_scene
def _simple_scene(*, include_disabled: bool = False, mu_r: float = 1.0) -> Scene:
    shared = Material(name="shared-material", eps_r=4.0, mu_r=mu_r, sigma_e=0.2)
    structures = [
        Structure(
            geometry=Box(position=(0.0, 0.0, 1.0), size=(2.0, 2.0, 2.0), device="cpu"),
            material=shared,
            name="box-left",
        ),
        Structure(
            geometry=Box(position=(3.0, 0.0, 1.0), size=(1.5, 1.5, 1.5), device="cpu"),
            material=shared,
            name="box-right",
        ),
    ]
    if include_disabled:
        structures.append(
            Structure(
                geometry=Box(position=(6.0, 0.0, 1.0), size=(1.0, 1.0, 1.0), device="cpu"),
                material=Material(name="disabled-material", eps_r=2.0),
                name="box-disabled",
                enabled=False,
            )
        )
    return Scene(structures=structures, device="cpu")


def test_load_sionna_rt_prefers_local_reference():
    result = load_sionna_rt(prefer_local=True)
    assert hasattr(result.rt, "Scene")
    assert result.source == "local_reference"
    assert result.source_root is not None
    assert (result.source_root / "sionna" / "rt").exists()


def test_scene_to_sionna_scene_converts_structures_and_reuses_materials():
    scene = _simple_scene()
    sionna_scene = scene.to_sionna()
    result = scene_to_sionna_scene(scene)

    assert len(result.scene.objects) == 2
    assert len(result.scene.mi_scene.shapes()) == 2
    assert len(sionna_scene.objects) == 2
    assert set(result.structure_name_map.keys()) == {"box-left", "box-right"}
    assert result.structure_material_map["box-left"] == result.structure_material_map["box-right"]

    left_name = result.structure_name_map["box-left"]
    left_object = result.scene.objects[left_name]
    radio_material = left_object.radio_material
    assert float(radio_material.relative_permittivity[0]) == pytest.approx(4.0)
    assert float(radio_material.conductivity[0]) == pytest.approx(0.2)
    assert float(radio_material.thickness[0]) == pytest.approx(0.1)


def test_scene_to_sionna_scene_skips_disabled_and_converts_mesh_geometry():
    mesh = Mesh(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        faces=[[0, 1, 2]],
        position=(2.0, 3.0, 1.0),
        recenter=False,
        device="cpu",
    )
    scene = Scene(
        structures=[
            Structure(
                geometry=mesh,
                material=Material(name="mesh-material", eps_r=3.0, sigma_e=0.05),
                name="triangle",
            ),
            Structure(
                geometry=Box(position=(6.0, 0.0, 1.0), size=(1.0, 1.0, 1.0), device="cpu"),
                material=Material(name="disabled-material", eps_r=2.0),
                name="box-disabled",
                enabled=False,
            ),
        ],
        device="cpu",
    )

    result = scene_to_sionna_scene(scene)

    assert set(result.structure_name_map.keys()) == {"triangle"}
    assert len(result.scene.objects) == 1
    triangle_name = result.structure_name_map["triangle"]
    bbox = result.scene.objects[triangle_name].mi_mesh.bbox()
    assert float(bbox.min.x) == pytest.approx(2.0)
    assert float(bbox.min.y) == pytest.approx(3.0)
    assert float(bbox.min.z) == pytest.approx(1.0)


def test_scene_to_sionna_scene_accepts_transposed_drjit_mesh_layout():
    scene = build_scene(
        box_drjit_geometry(center=(0.0, 0.0, 1.0), size=2.0, rotation=None, device="cpu"),
        device="cpu",
        material=Material(name="drjit-mesh-material", eps_r=3.0, sigma_e=0.05),
    )

    result = scene_to_sionna_scene(scene)

    assert len(result.scene.objects) == 1
    structure_name = result.structure_name_map["structure_0"]
    bbox = result.scene.objects[structure_name].mi_mesh.bbox()
    assert float(bbox.min.x) == pytest.approx(-1.0)
    assert float(bbox.max.x) == pytest.approx(1.0)
    assert float(bbox.min.z) == pytest.approx(0.0)
    assert float(bbox.max.z) == pytest.approx(2.0)


def test_scene_to_sionna_scene_reports_mu_r_mismatch_and_helper_wrapper_works():
    scene = _simple_scene(mu_r=1.5)
    with pytest.raises(ValueError, match="mu_r"):
        scene_to_sionna_scene(scene)

    result = scene_to_sionna_scene(scene, strict_mu_r=False)
    assert result.warnings
    assert "mu_r=1.5" in result.warnings[0]


def test_scene_from_sionna_scene_round_trips_meshes_and_materials():
    original = _simple_scene()
    sionna_scene = original.to_sionna()

    restored = Scene.from_sionna(sionna_scene, device="cpu")

    assert len(restored.structures) == 2
    assert restored.structures[0].name == "box-left"
    assert restored.structures[1].name == "box-right"
    assert restored.structures[0].material is restored.structures[1].material
    assert restored.structures[0].material.name == "shared-material"
    assert restored.structures[0].material.eps_r == pytest.approx(4.0)
    assert restored.structures[0].material.sigma_e == pytest.approx(0.2)

    mesh = restored.structures[0].geometry
    vertices, faces = mesh.to_mesh()
    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)


def test_scene_from_sionna_scene_function_matches_classmethod():
    source = _simple_scene()
    sionna_scene = source.to_sionna()

    from_function = scene_from_sionna_scene(sionna_scene, device="cpu")
    from_method = Scene.from_sionna(sionna_scene, device="cpu")

    assert [structure.name for structure in from_function.structures] == [
        structure.name for structure in from_method.structures
    ]
