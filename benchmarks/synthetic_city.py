# Copyright Xingyu Chen.
# Build the fixed synthetic box-building cities used by the baseline, budget and accuracy suites.

"""Build the fixed synthetic box-building cities used by the baseline, budget and accuracy suites."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from witwin.core import AntennaState, Mesh, PhysicalMaterial, ReceiverGrid, Structure
from witwin.core.identity import new_antenna_id

FREQUENCY_HZ = 3.5e9
CONCRETE = PhysicalMaterial(eps_r=5.24, sigma_e=0.123)
PITCH_M = 40.0
WIDTH_M = 22.0
WELD_TOLERANCE_M = 1e-4


@dataclass(frozen=True, slots=True)
class City:
    structures: list[Structure]
    extent_m: float
    triangle_count: int
    vertex_count: int


def _box(x0: float, y0: float, x1: float, y1: float, h: float) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
         [x0, y0, h], [x1, y0, h], [x1, y1, h], [x0, y1, h]],
        dtype=torch.float64,
    )
    faces = torch.tensor(
        [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7],
         [2, 7, 6], [3, 0, 4], [3, 4, 7], [4, 5, 6], [4, 6, 7]]
    )
    return vertices, faces


def _subdivide(vertices: torch.Tensor, faces: torch.Tensor, levels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split every triangle into four at its edge midpoints, ``levels`` times (midpoints not shared)."""
    for _ in range(levels):
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        n, m = vertices.shape[0], faces.shape[0]
        vertices = torch.cat([vertices, (a + b) / 2, (b + c) / 2, (c + a) / 2])
        iab = torch.arange(n, n + m)
        ibc = iab + m
        ica = ibc + m
        faces = torch.cat([
            torch.stack([faces[:, 0], iab, ica], 1),
            torch.stack([iab, faces[:, 1], ibc], 1),
            torch.stack([ica, ibc, faces[:, 2]], 1),
            torch.stack([iab, ibc, ica], 1),
        ])
    return vertices, faces


def _weld(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge vertices closer than the weld tolerance so neighbouring triangles share them."""
    keys = torch.round(vertices / WELD_TOLERANCE_M).to(torch.int64)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    return unique.to(torch.float64) * WELD_TOLERANCE_M, inverse[faces]


def city(n_side: int = 4, *, subdiv: int = 0, welded: bool = True, seed: int = 0) -> City:
    """``n_side`` x ``n_side`` concrete boxes 10-40 m tall on a ground plane, merged into one mesh.

    ``subdiv`` levels of 4-way subdivision keep the geometry identical; ``welded`` shares the new
    midpoint vertices between neighbouring triangles (unwelded leaves T-junction-free duplicates).
    """

    generator = torch.Generator().manual_seed(seed)
    extent = n_side * PITCH_M
    parts = [(
        torch.tensor([[0, 0, 0], [extent, 0, 0], [extent, extent, 0], [0, extent, 0]],
                     dtype=torch.float64),
        torch.tensor([[0, 1, 2], [0, 2, 3]]),
    )]
    for i in range(n_side):
        for j in range(n_side):
            height = 10.0 + 30.0 * float(torch.rand((), generator=generator))
            x0 = i * PITCH_M + (PITCH_M - WIDTH_M) / 2
            y0 = j * PITCH_M + (PITCH_M - WIDTH_M) / 2
            parts.append(_box(x0, y0, x0 + WIDTH_M, y0 + WIDTH_M, height))
    vertex_parts, face_parts, offset = [], [], 0
    for vertices, faces in parts:
        vertices, faces = _subdivide(vertices, faces, subdiv)
        vertex_parts.append(vertices)
        face_parts.append(faces + offset)
        offset += vertices.shape[0]
    vertices, faces = torch.cat(vertex_parts), torch.cat(face_parts)
    if welded and subdiv > 0:
        vertices, faces = _weld(vertices, faces)
    mesh = Mesh(vertices.to(torch.float32), faces.to(torch.int32), recenter=False,
                fill_mode="surface", topology_diagnostics=False)
    structure = Structure(mesh, CONCRETE, name="city", surface_id=1)
    return City([structure], extent, int(faces.shape[0]), int(vertices.shape[0]))


def transmitter(extent_m: float, z: float = 25.0) -> AntennaState:
    """Transmitter above the street crossing nearest the city centre."""
    c = round(extent_m / 2 / PITCH_M) * PITCH_M
    return AntennaState(new_antenna_id(), "tx", torch.tensor([c, c, z]), power_w=1.0)


def receiver_grid(extent_m: float, n: int, z: float = 1.5) -> ReceiverGrid:
    """``n`` x ``n`` receiver grid covering the whole city at height ``z``."""
    spacing = extent_m / n
    return ReceiverGrid(
        new_antenna_id(), torch.tensor([spacing / 2, spacing / 2, z]), torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]), (n, n), (spacing, spacing),
    )


def receiver_points(extent_m: float, n: int, z: float = 1.5, seed: int = 1) -> list[AntennaState]:
    """``n`` point receivers at seeded positions on street centre lines (outdoors) at height ``z``."""
    generator = torch.Generator().manual_seed(seed)
    streets = int(round(extent_m / PITCH_M))
    out = []
    centre = round(extent_m / 2 / PITCH_M) * PITCH_M
    for index in range(n):
        along = float(torch.rand((), generator=generator)) * extent_m
        street = int(torch.randint(0, streets + 1, (), generator=generator)) * PITCH_M
        x, y = (along, street) if index % 2 == 0 else (street, along)
        if index == 0:  # the first receiver is down the transmitter's street, in line of sight
            x, y = centre + 0.75 * PITCH_M, centre
        out.append(AntennaState(new_antenna_id(), "rx", torch.tensor([x, y, z])))
    return out
