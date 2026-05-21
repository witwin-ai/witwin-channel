"""Shared fixtures for scene-level unit tests."""

from __future__ import annotations

import pytest
import torch

from witwin.channel.core.scene import Mesh, Scene
from witwin.core import Material, Structure


@pytest.fixture
def triangle_vertices() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def triangle_faces() -> torch.Tensor:
    return torch.tensor([[0, 1, 2]], dtype=torch.int32)


@pytest.fixture
def triangle_mesh(triangle_vertices: torch.Tensor, triangle_faces: torch.Tensor) -> Mesh:
    return Mesh(vertices=triangle_vertices, faces=triangle_faces)


@pytest.fixture
def triangle_structure(triangle_mesh: Mesh) -> Structure:
    return Structure(
        geometry=triangle_mesh,
        material=Material(name="triangle-material", eps_r=2.5, sigma_e=0.1),
        name="triangle",
    )


@pytest.fixture
def triangle_scene(triangle_structure: Structure) -> Scene:
    return Scene(structures=[triangle_structure], device="cpu")


@pytest.fixture
def wall_vertices() -> torch.Tensor:
    return torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 3.0],
            [1.0, 0.0, 3.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def wall_faces() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 3],
            [0, 3, 2],
        ],
        dtype=torch.int32,
    )


@pytest.fixture
def wall_mesh(wall_vertices: torch.Tensor, wall_faces: torch.Tensor) -> Mesh:
    return Mesh(vertices=wall_vertices, faces=wall_faces)


@pytest.fixture
def wall_scene(wall_mesh: Mesh) -> Scene:
    return Scene(
        structures=[
            Structure(
                geometry=wall_mesh,
                material=Material(name="wall-material", eps_r=4.0, sigma_e=0.2),
                name="wall",
            )
        ],
        device="cpu",
    )
