# Copyright Xingyu Chen.
# Implements scenes.

from __future__ import annotations

import torch

from witwin.core import (
    MaterialLayer,
    PhysicalMaterial,
    Scene,
    Structure,
    SurfaceRoughness,
)

from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_transmitter,
    planar_uv,
)


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
        else SurfaceRoughness(
            rms_height_m=rms_height_m,
            correlation_length_x_m=corr_length_m,
            correlation_length_y_m=corr_length_m,
        )
    )
    material = PhysicalMaterial(
        layers=(
            MaterialLayer(
                thickness_m=thickness_m,
                eps_r=eps_r,
                sigma_e=sigma_e,
            ),
        ),
        roughness_front=roughness,
        name=name,
    )
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
    return make_mesh_structure(
        vertices=vertices,
        faces=faces,
        material=material,
        name=name,
        surface_id=surface_id,
        phase_screen=phase_screen,
        **kwargs,
    )


def empty_space_los_scene() -> Scene:
    return Scene(
        structures=[],
        endpoints=[
            make_transmitter(torch.tensor([0.0, 0.0, 0.0]), power_w=2.0),
            make_transmitter(torch.tensor([0.0, 2.0, 0.0]), power_w=0.5),
            make_receiver(torch.tensor([3.0, 4.0, 0.0])),
            make_receiver(torch.tensor([6.0, 8.0, 0.0])),
        ],
    )


def single_wall_reflection_scene() -> Scene:
    wall = make_mesh_structure(
        vertices=torch.tensor(
            [
                [2.5, -2.0, -1.0],
                [2.5, 2.0, -1.0],
                [2.5, -2.0, 2.0],
                [2.5, 2.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
        name="single-wall",
        surface_id=1,
    )
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(torch.tensor([0.0, 0.0, 0.0])),
            make_receiver(torch.tensor([5.0, 0.0, 0.0])),
        ],
    )


def same_side_wall_reflection_scene() -> Scene:
    wall = make_mesh_structure(
        vertices=torch.tensor(
            [
                [2.5, -3.0, -1.0],
                [2.5, 3.0, -1.0],
                [2.5, -3.0, 2.0],
                [2.5, 3.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
        name="same-side-wall",
        surface_id=4,
    )
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(torch.tensor([0.0, -1.0, 0.5])),
            make_receiver(torch.tensor([0.0, 1.0, 0.5])),
        ],
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

    return make_mesh_structure(
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
    wedge_material = (
        PhysicalMaterial.perfect_conductor() if material is None else material
    )
    face_a = make_mesh_structure(
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
    face_b = make_mesh_structure(
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
        endpoints=[
            make_transmitter(
                torch.tensor([0.0, -1.0, 0.5]) if tx is None else tx
            ),
            make_receiver(
                torch.tensor([3.0, 1.0, 0.5]) if rx is None else rx
            ),
        ],
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
            make_mesh_structure(
                vertices=vertices,
                faces=faces,
                material=(
                    PhysicalMaterial.perfect_conductor()
                    if material is None
                    else material
                ),
                surface_id=0,
            )
        ],
        endpoints=[
            make_transmitter(
                torch.tensor([0.0, -2.0, 1.0]) if tx is None else tx
            ),
            make_receiver(
                torch.tensor([0.0, 2.0, 5.0]) if rx is None else rx
            ),
        ],
    )