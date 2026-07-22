from __future__ import annotations

import torch

from witwin.channel import ReceiverPoint, Scene, Structure, Transmitter
from witwin.channel.core.materials import (
    Dielectric,
    Layer,
    PerfectConductor,
    PhysicalSurface,
    Roughness,
    SurfaceAssignment,
)
from witwin.channel.core.objects import planar_uv


def rough_wall_structure(
    x_m: float,
    *,
    rms_height_m: float,
    corr_length_m: float,
    half_size: float = 2.0,
    eps_r: float = 4.0,
    sigma_e: float = 0.01,
    thickness_m: float = 0.1,
    phase_screen: object | None = None,
    with_uv: bool = False,
    name: str = "rough-wall",
    surface_id: int = 1,
) -> Structure:
    """Axis-aligned wall in the x = ``x_m`` plane with front-surface roughness.

    ``rms_height_m == 0`` compiles as a smooth wall with the same layer stack
    (scatter_model_id 0). ``phase_screen`` wraps the material in a
    ``SurfaceAssignment``; ``with_uv`` adds a planar unit-square UV chart
    (u along +y, v along +z), required for realization_coherent screens.
    """

    vertices = torch.tensor(
        [
            [x_m, -half_size, -half_size],
            [x_m, half_size, -half_size],
            [x_m, -half_size, half_size],
            [x_m, half_size, half_size],
        ]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32)
    roughness = (
        None
        if rms_height_m <= 0.0
        else Roughness(
            rms_height_m=rms_height_m,
            corr_length_x_m=corr_length_m,
            corr_length_y_m=corr_length_m,
        )
    )
    material: object = PhysicalSurface(
        layers=(Layer(thickness_m=thickness_m, eps_r=eps_r, sigma_e=sigma_e),),
        roughness_front=roughness,
        name=name,
    )
    if phase_screen is not None:
        material = SurfaceAssignment(material=material, phase_screen=phase_screen)
    kwargs = {}
    if with_uv:
        kwargs = {
            "uv": planar_uv(
                vertices,
                axis_u=torch.tensor([0.0, 1.0, 0.0]),
                axis_v=torch.tensor([0.0, 0.0, 1.0]),
                origin=torch.tensor([x_m, -half_size, -half_size]),
                scale=1.0 / (2.0 * half_size),
            ),
            "face_uv": faces.clone(),
        }
    return Structure(
        vertices=vertices,
        faces=faces,
        material=material,
        name=name,
        surface_id=surface_id,
        **kwargs,
    )


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


def transmission_wall_structure(
    x_m: float,
    material: object,
    *,
    name: str = "wall",
    surface_id: int = 1,
    half_size: float = 4.0,
) -> Structure:
    """Axis-aligned thin-sheet wall in the x = ``x_m`` plane (normal +x)."""

    return Structure(
        vertices=torch.tensor(
            [
                [x_m, -half_size, -half_size],
                [x_m, half_size, -half_size],
                [x_m, -half_size, half_size],
                [x_m, half_size, half_size],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=material,
        name=name,
        surface_id=surface_id,
    )


def wedge_diffraction_scene(
    material: object | None = None,
    *,
    tx: torch.Tensor | None = None,
    rx: torch.Tensor | None = None,
    frequency: float | torch.Tensor = 3.0e9,
) -> Scene:
    wedge_material = PerfectConductor() if material is None else material
    face_a = Structure(
        vertices=torch.tensor(
            [
                [2.0, 0.0, -1.0],
                [2.0, 0.0, 2.0],
                [2.0, 2.0, -1.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2]]),
        material=wedge_material,
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
        material=wedge_material,
        name="wedge-b",
        surface_id=3,
    )
    return Scene(
        structures=[face_a, face_b],
        transmitters=[
            Transmitter(
                position=torch.tensor([0.0, -1.0, 0.5]) if tx is None else tx
            )
        ],
        receivers=[
            ReceiverPoint(
                position=torch.tensor([3.0, 1.0, 0.5]) if rx is None else rx
            )
        ],
        frequency=frequency,
    )


def coupled_wall_wedge_scene(
    material: object | None = None,
    *,
    tx: torch.Tensor | None = None,
    rx: torch.Tensor | None = None,
    frequency: float | torch.Tensor = 3.0e9,
) -> Scene:
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
                material=PerfectConductor() if material is None else material,
                surface_id=0,
            )
        ],
        transmitters=[
            Transmitter(
                position=torch.tensor([0.0, -2.0, 1.0]) if tx is None else tx
            )
        ],
        receivers=[
            ReceiverPoint(
                position=torch.tensor([0.0, 2.0, 5.0]) if rx is None else rx
            )
        ],
        frequency=frequency,
    )
