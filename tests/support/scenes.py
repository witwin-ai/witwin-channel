from __future__ import annotations

import torch

from witwin.channel_native import ReceiverPoint, Scene, Structure, Transmitter
from witwin.channel_native.core.materials import Dielectric, PerfectConductor


def empty_space_los_scene() -> Scene:
    return Scene(
        structures=[],
        transmitters=[
            Transmitter(position=torch.tensor([0.0, 0.0, 0.0]), power_w=2.0),
            Transmitter(position=torch.tensor([0.0, 2.0, 0.0]), power_w=0.5),
        ],
        receivers=[
            ReceiverPoint(position=torch.tensor([3.0, 4.0, 0.0])),
            ReceiverPoint(position=torch.tensor([6.0, 8.0, 0.0])),
        ],
        frequency=3.0e9,
    )


def single_wall_reflection_scene() -> Scene:
    wall = Structure(
        vertices=torch.tensor(
            [
                [2.5, -2.0, -1.0],
                [2.5, 2.0, -1.0],
                [2.5, -2.0, 2.0],
                [2.5, 2.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=Dielectric(eps_r=4.0, sigma_e=0.01),
        name="single-wall",
        surface_id=1,
    )
    return Scene(
        structures=[wall],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([5.0, 0.0, 0.0]))],
        frequency=3.0e9,
    )


def same_side_wall_reflection_scene() -> Scene:
    wall = Structure(
        vertices=torch.tensor(
            [
                [2.5, -3.0, -1.0],
                [2.5, 3.0, -1.0],
                [2.5, -3.0, 2.0],
                [2.5, 3.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=Dielectric(eps_r=4.0, sigma_e=0.01),
        name="same-side-wall",
        surface_id=4,
    )
    return Scene(
        structures=[wall],
        transmitters=[Transmitter(position=torch.tensor([0.0, -1.0, 0.5]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 1.0, 0.5]))],
        frequency=3.0e9,
    )


def wedge_diffraction_scene() -> Scene:
    face_a = Structure(
        vertices=torch.tensor(
            [
                [2.0, 0.0, -1.0],
                [2.0, 0.0, 2.0],
                [2.0, 2.0, -1.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2]]),
        material=PerfectConductor(),
        name="wedge-a",
        surface_id=2,
    )
    face_b = Structure(
        vertices=torch.tensor(
            [
                [2.0, 0.0, -1.0],
                [2.0, 0.0, 2.0],
                [4.0, 0.0, -1.0],
            ]
        ),
        faces=torch.tensor([[0, 2, 1]]),
        material=PerfectConductor(),
        name="wedge-b",
        surface_id=3,
    )
    return Scene(
        structures=[face_a, face_b],
        transmitters=[Transmitter(position=torch.tensor([0.0, -1.0, 0.5]))],
        receivers=[ReceiverPoint(position=torch.tensor([3.0, 1.0, 0.5]))],
        frequency=3.0e9,
    )


def coupled_wall_wedge_scene() -> Scene:
    """Analytic z=0 reflector plus the convex edge through (2, y, 2)."""

    vertices = torch.tensor(
        [
            [-5.0, -5.0, 0.0],
            [5.0, -5.0, 0.0],
            [5.0, 5.0, 0.0],
            [-5.0, 5.0, 0.0],
            [2.0, -1.0, 2.0],
            [2.0, 1.0, 2.0],
            [4.0, -1.0, 2.0],
            [4.0, 1.0, 2.0],
            [2.0, -1.0, 4.0],
            [2.0, 1.0, 4.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [4, 6, 7], [4, 7, 5], [4, 5, 9], [4, 9, 8]],
        dtype=torch.int32,
    )
    return Scene(
        structures=[
            Structure(
                vertices=vertices,
                faces=faces,
                material=PerfectConductor(),
                surface_id=0,
            )
        ],
        transmitters=[Transmitter(position=torch.tensor([0.0, -2.0, 1.0]))],
        receivers=[ReceiverPoint(position=torch.tensor([0.0, 2.0, 5.0]))],
        frequency=3.0e9,
    )
