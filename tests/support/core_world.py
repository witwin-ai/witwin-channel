# Copyright Xingyu Chen.
# Implements core world.

from __future__ import annotations

import torch

from witwin.core import AntennaState, Mesh, ReceiverGrid, Structure
from witwin.core.identity import new_antenna_id


def make_transmitter(
    position,
    *,
    orientation=None,
    polarization=None,
    element_positions=None,
    weights=None,
    synthetic_array: bool = True,
    pattern: str = "isotropic",
    power_w=1.0,
) -> AntennaState:
    return AntennaState(
        new_antenna_id(),
        "tx",
        position,
        orientation=orientation,
        polarization=polarization,
        element_positions=element_positions,
        weights=weights,
        synthetic_array=synthetic_array,
        pattern=pattern,
        power_w=power_w,
    )


def make_receiver(
    position,
    *,
    orientation=None,
    polarization=None,
    element_positions=None,
    weights=None,
    synthetic_array: bool = True,
    pattern: str = "isotropic",
) -> AntennaState:
    return AntennaState(
        new_antenna_id(),
        "rx",
        position,
        orientation=orientation,
        polarization=polarization,
        element_positions=element_positions,
        weights=weights,
        synthetic_array=synthetic_array,
        pattern=pattern,
    )


def make_receiver_grid(
    *,
    origin,
    x_axis,
    y_axis,
    shape,
    spacing,
    orientation=None,
    polarization=None,
    element_positions=None,
    weights=None,
    synthetic_array: bool = True,
    pattern: str = "isotropic",
) -> ReceiverGrid:
    return ReceiverGrid(
        new_antenna_id(),
        origin,
        x_axis,
        y_axis,
        shape,
        spacing,
        orientation=orientation,
        polarization=polarization,
        element_positions=element_positions,
        weights=weights,
        synthetic_array=synthetic_array,
        pattern=pattern,
    )


def make_mesh_structure(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    material,
    recenter: bool = False,
    fill_mode: str = "surface",
    **kwargs,
) -> Structure:
    geometry = Mesh(
        vertices,
        faces,
        recenter=recenter,
        fill_mode=fill_mode,
        topology_diagnostics=False,
    )
    return Structure(geometry, material, **kwargs)


def planar_uv(
    vertices: torch.Tensor,
    *,
    axis_u: torch.Tensor,
    axis_v: torch.Tensor,
    origin: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    relative = vertices - origin.to(device=vertices.device, dtype=vertices.dtype)
    return torch.stack(
        (
            relative @ axis_u.to(device=vertices.device, dtype=vertices.dtype),
            relative @ axis_v.to(device=vertices.device, dtype=vertices.dtype),
        ),
        dim=-1,
    ) * scale