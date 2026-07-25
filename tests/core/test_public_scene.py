import torch

from tests.support.core_world import (
    make_receiver,
    make_receiver_grid,
    make_transmitter,
)
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure


def test_public_scene_objects_capture_structured_inputs():
    vertices = torch.zeros((3, 3), dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    material = PhysicalMaterial(eps_r=2.5)
    mesh = Mesh(
        vertices,
        faces,
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    structure = Structure(mesh, material, name="wall")
    transmitter = make_transmitter(
        position=torch.tensor([0.0, 0.0, 1.0]),
        polarization=torch.tensor([0.0, 0.0, 1.0]),
        power_w=2.0,
    )
    receiver = make_receiver(
        position=torch.tensor([1.0, 0.0, 1.0]),
        polarization=torch.tensor([0.0, 0.0, 1.0]),
    )

    scene = Scene(
        structures=[structure],
        endpoints=[transmitter, receiver],
    )

    assert scene.structures == (structure,)
    assert scene.endpoints == (transmitter, receiver)
    assert tuple(endpoint.role for endpoint in scene.endpoints) == ("tx", "rx")
    assert mesh.vertices is vertices
    assert mesh.faces is faces
    assert not hasattr(scene, "frequency")


def test_scene_endpoints_preserve_explicit_polarization():
    tx_polarization = torch.tensor([2.0, 0.0, 0.0])
    rx_polarization = torch.tensor([0.0, -3.0, 0.0])
    tx = make_transmitter(
        position=torch.zeros(3), polarization=tx_polarization
    )
    rx = make_receiver(
        position=torch.ones(3), polarization=rx_polarization
    )

    assert tx.polarization is tx_polarization
    assert rx.polarization is rx_polarization


def test_receiver_grid_expands_points_in_row_major_order():
    grid = make_receiver_grid(
        origin=torch.tensor([0.0, 0.0, 1.0]),
        x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=(2, 3),
        spacing=(0.5, 2.0),
    )

    points = grid.points()

    assert points.shape == (6, 3)
    torch.testing.assert_close(points[0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(points[1], torch.tensor([0.0, 2.0, 1.0]))
    torch.testing.assert_close(points[3], torch.tensor([0.5, 0.0, 1.0]))


def test_materials_expose_canonical_samples():
    dielectric = PhysicalMaterial(eps_r=2.5, mu_r=1.1)
    lossy = PhysicalMaterial(eps_r=4.0, sigma_e=0.02)
    conductor = PhysicalMaterial.perfect_conductor(gain=0.75)

    dielectric_sample = dielectric.evaluate_static()
    lossy_sample = lossy.evaluate_static()
    conductor_sample = conductor.evaluate_static()
    assert dielectric_sample.eps_r == 2.5
    assert dielectric_sample.mu_r == 1.1
    assert lossy_sample.sigma_e == 0.02
    assert conductor_sample.eps_r == 1.0
    assert conductor.gain == 0.75
    assert conductor.capabilities().perfect_conductor


def test_scene_immutable_endpoint_update_workflow():
    scene = Scene()
    transmitter = make_transmitter(position=torch.tensor([0.0, 0.0, 0.0]))
    receiver = make_receiver(position=torch.tensor([1.0, 0.0, 0.0]))

    updated = scene.with_endpoints([transmitter, receiver])

    assert scene.endpoints == ()
    assert updated.endpoints == (transmitter, receiver)
