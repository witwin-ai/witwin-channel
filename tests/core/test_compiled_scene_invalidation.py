import torch

from witwin.channel_native import ReceiverPoint, Scene, Structure, Transmitter
from witwin.channel_native.core.materials import Dielectric


def _scene() -> Scene:
    return Scene(
        structures=[
            Structure(
                vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                material=Dielectric(eps_r=2.0),
            )
        ],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 1.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([1.0, 0.0, 1.0]))],
        frequency=3.5e9,
    )


def test_scene_compile_separates_geometry_materials_assignments():
    compiled = _scene().compile()

    assert compiled.geometry.vertices.shape == (3, 3)
    assert compiled.materials.eps_r.shape == (1,)
    assert compiled.assignments.face_material_id.shape == (1,)
    assert compiled.geometry_version == compiled.geometry.version
    assert compiled.material_version == compiled.materials.version
    assert compiled.assignment_version == compiled.assignments.version


def test_scene_edit_versions_are_independent():
    scene = _scene()

    vertex_scene = scene.with_structure_vertices(
        0,
        torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    material_scene = scene.with_structure_material(0, Dielectric(eps_r=3.0))
    frequency_scene = scene.with_frequency(5.0e9)

    assert vertex_scene.compile().geometry.version > scene.compile().geometry.version
    assert vertex_scene.compile().materials.version == scene.compile().materials.version
    assert material_scene.compile().materials.version > scene.compile().materials.version
    assert material_scene.compile().geometry.version == scene.compile().geometry.version
    assert frequency_scene.compile().materials.version > scene.compile().materials.version
    assert frequency_scene.compile().geometry.version == scene.compile().geometry.version
