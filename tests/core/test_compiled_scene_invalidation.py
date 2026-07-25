import torch

from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel.scene import compile as compile_scene
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

_REFERENCE_FREQUENCY_HZ = 3.5e9


def _scene() -> Scene:
    geometry = Mesh(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    material = PhysicalMaterial(eps_r=2.0)
    return Scene(
        structures=[Structure(geometry, material)],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 1.0])),
            make_receiver(position=torch.tensor([1.0, 0.0, 1.0])),
        ],
    )


def test_channel_compile_separates_geometry_materials_assignments():
    compiled = compile_scene(
        _scene(), reference_frequency_hz=_REFERENCE_FREQUENCY_HZ
    )

    assert compiled.geometry.vertices.shape == (3, 3)
    assert compiled.materials.eps_r.shape == (1,)
    assert compiled.assignments.face_material_id.shape == (1,)
    assert compiled.geometry_version == compiled.geometry.version
    assert compiled.material_version == compiled.materials.version
    assert compiled.assignment_version == compiled.assignments.version


def test_scene_edit_versions_and_frequency_request_are_independent():
    scene = _scene()
    structure = scene.structures[0]
    replacement_geometry = Mesh(
        torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        structure.geometry.faces,
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    vertex_scene = scene.with_structure_geometry(
        structure.structure_id,
        replacement_geometry,
        topology_changed=False,
    )
    replacement_material = PhysicalMaterial(
        eps_r=3.0,
        material_id=structure.material_id,
    )
    material_scene = scene.with_material(
        structure.material_id, replacement_material
    )

    assert vertex_scene.geometry_version > scene.geometry_version
    assert vertex_scene.material_version == scene.material_version
    assert material_scene.material_version > scene.material_version
    assert material_scene.geometry_version == scene.geometry_version

    base = compile_scene(
        scene, reference_frequency_hz=_REFERENCE_FREQUENCY_HZ
    )
    changed_frequency = compile_scene(
        scene, reference_frequency_hz=5.0e9
    )
    assert changed_frequency.source is scene
    assert changed_frequency.geometry is base.geometry
    assert changed_frequency.materials is not base.materials
    assert not hasattr(scene, "with_frequency")
