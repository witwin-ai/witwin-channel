import pytest
import torch

from witwin.channel_native import ReceiverGrid, ReceiverPoint, Scene, Structure, Transmitter
from witwin.channel_native.core.materials import Dielectric, LossyDielectric, PerfectConductor


def test_public_scene_objects_capture_structured_inputs():
    vertices = torch.zeros((3, 3), dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    material = Dielectric(eps_r=2.5)
    structure = Structure(vertices=vertices, faces=faces, material=material, name="wall")
    transmitter = Transmitter(position=torch.tensor([0.0, 0.0, 1.0]), power_w=2.0)
    receiver = ReceiverPoint(position=torch.tensor([1.0, 0.0, 1.0]))

    scene = Scene(
        structures=[structure],
        transmitters=[transmitter],
        receivers=[receiver],
        frequency=3.5e9,
    )

    assert scene.structures == (structure,)
    assert scene.transmitters == (transmitter,)
    assert scene.receivers == (receiver,)
    assert scene.frequency == 3.5e9


def test_receiver_grid_expands_points_in_row_major_order():
    grid = ReceiverGrid(
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


def test_materials_compile_scalar_parameters():
    dielectric = Dielectric(eps_r=2.5, mu_r=1.1)
    lossy = LossyDielectric(eps_r=4.0, sigma_e=0.02)
    conductor = PerfectConductor(gain=0.75)

    assert dielectric.parameters() == {
        "eps_r": 2.5,
        "mu_r": 1.1,
        "sigma_e": 0.0,
        "gain": 1.0,
        "model_id": 1,
    }
    assert lossy.parameters()["sigma_e"] == 0.02
    assert conductor.parameters()["model_id"] == 2
    assert conductor.parameters()["gain"] == 0.75


def test_scene_requires_at_least_one_receiver():
    with pytest.raises(ValueError, match="at least one receiver"):
        Scene(structures=[], transmitters=[], receivers=[], frequency=1.0)
