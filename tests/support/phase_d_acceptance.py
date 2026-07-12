from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal

import torch

from tests.support.scenes import rough_wall_structure, transmission_wall_structure
from witwin.channel_native import ReceiverGrid, ReceiverPoint, Scene, Transmitter
from witwin.channel_native.core.antenna import orientation_matrix
from witwin.channel_native.core.materials import DebyeModel, Layer, PhysicalSurface
from witwin.channel_native.deterministic import Config as DeterministicConfig
from witwin.channel_native.deterministic import solve as solve_deterministic
from witwin.channel_native.montecarlo.basic import Config as BasicConfig
from witwin.channel_native.montecarlo.basic import solve as solve_basic
from witwin.channel_native.montecarlo.bdpt import Config as BDPTConfig
from witwin.channel_native.montecarlo.bdpt import solve as solve_bdpt
from witwin.channel_native.path import Config as PathConfig
from witwin.channel_native.path import InteractionType, solve as solve_path


@dataclass(frozen=True, slots=True)
class SolverAdapter:
    name: str
    field_domain: Literal["complex", "power"]
    transmission: Callable[[Scene], complex | float]
    scattering: Callable[[Scene, int], float]


def dispersive_multilayer() -> PhysicalSurface:
    return PhysicalSurface(
        layers=(
            Layer(
                thickness_m=0.035,
                eps_model=DebyeModel(
                    eps_inf=2.1,
                    delta_eps=2.4,
                    tau_s=18.0e-12,
                    sigma_dc=0.015,
                ),
            ),
            Layer(thickness_m=0.055, eps_r=5.2, sigma_e=0.025),
        ),
        name="phase-d-dispersive-stack",
    )


def transmission_scene(
    frequency_hz: float, *, reverse: bool = False, empty: bool = False
) -> Scene:
    tx = torch.tensor([0.0, 0.0, 0.0])
    rx = torch.tensor([5.0, 0.0, 0.0])
    if reverse:
        tx, rx = rx, tx
    structures = []
    if not empty:
        structures = [
            transmission_wall_structure(
                2.5,
                dispersive_multilayer(),
                name="phase-d-wall",
                surface_id=41,
            )
        ]
    return Scene(
        structures=structures,
        transmitters=[
            Transmitter(position=tx, polarization=torch.tensor([0.0, 0.0, 1.0]))
        ],
        receivers=[
            ReceiverGrid(
                origin=rx,
                x_axis=torch.tensor([0.0, 1.0, 0.0]),
                y_axis=torch.tensor([0.0, 0.0, 1.0]),
                shape=(1, 1),
                spacing=(1.0, 1.0),
                polarization=torch.tensor([0.0, 0.0, 1.0]),
            )
        ],
        frequency=frequency_hz,
    )


def rough_scene(*, reverse: bool = False) -> Scene:
    # Near-normal but non-degenerate geometry keeps the Basic solver's
    # unpolarized response comparable with the explicitly polarized solvers.
    tx_position = torch.tensor([0.0, -0.25, 0.0])
    rx_position = torch.tensor([0.0, 0.25, 0.0])
    if reverse:
        tx_position, rx_position = rx_position, tx_position
    return Scene(
        structures=[
            rough_wall_structure(
                2.5,
                rms_height_m=0.015,
                corr_length_m=0.15,
                half_size=2.0,
                name="phase-d-rough-wall",
                surface_id=42,
            )
        ],
        transmitters=[
            Transmitter(
                position=tx_position, polarization=torch.tensor([0.0, 0.0, 1.0])
            )
        ],
        receivers=[
            ReceiverGrid(
                origin=rx_position,
                x_axis=torch.tensor([0.0, 1.0, 0.0]),
                y_axis=torch.tensor([0.0, 0.0, 1.0]),
                shape=(1, 1),
                spacing=(1.0, 1.0),
                polarization=torch.tensor([0.0, 0.0, 1.0]),
            )
        ],
        frequency=3.0e9,
    )


def rigid_transform(scene: Scene) -> Scene:
    rotation = orientation_matrix(torch.tensor([0.47, -0.31, 0.23]))
    translation = torch.tensor([7.0, -4.0, 2.5])

    def point(value: torch.Tensor) -> torch.Tensor:
        return rotation @ value + translation

    structures = [
        replace(structure, vertices=structure.vertices @ rotation.T + translation)
        for structure in scene.structures
    ]
    transmitters = [
        replace(
            transmitter,
            position=point(transmitter.position),
            polarization=rotation @ transmitter.polarization,
        )
        for transmitter in scene.transmitters
    ]
    receivers = []
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            receivers.append(
                replace(
                    receiver,
                    position=point(receiver.position),
                    polarization=rotation @ receiver.polarization,
                )
            )
        else:
            receivers.append(
                replace(
                    receiver,
                    origin=point(receiver.origin),
                    x_axis=rotation @ receiver.x_axis,
                    y_axis=rotation @ receiver.y_axis,
                    polarization=rotation @ receiver.polarization,
                )
            )
    return Scene(
        structures=structures,
        transmitters=transmitters,
        receivers=receivers,
        frequency=scene.frequency,
        metadata=scene.metadata,
    )


def _path_transmission(scene: Scene) -> complex:
    component = "los" if not scene.structures else "transmission"
    result = solve_path(scene, PathConfig(max_depth=1, components={component}))
    values = result.a[result.valid].reshape(-1)
    assert values.numel() == 1
    return complex(values[0].item())


def _deterministic_transmission(scene: Scene) -> complex:
    component = "los" if not scene.structures else "transmission"
    result = solve_deterministic(
        scene, DeterministicConfig(max_depth=1, components={component})
    )
    return complex(result.component_fields[component].reshape(-1)[0].item())


def _basic_transmission(scene: Scene) -> float:
    component = "los" if not scene.structures else "transmission"
    result = solve_basic(
        scene, BasicConfig(samples=512, max_depth=1, components={component})
    )
    return float(result.component_power[component])


def _bdpt_transmission(scene: Scene) -> float:
    component = "los" if not scene.structures else "transmission"
    result = solve_bdpt(
        scene, BDPTConfig(samples=512, max_depth=1, components={component})
    )
    return float(result.component_power[component])


def _path_scattering(scene: Scene, seed: int) -> float:
    del seed
    result = solve_path(
        scene,
        PathConfig(
            max_depth=1,
            components={"scattering"},
            scattering_samples_per_m2=32.0,
        ),
    )
    scattering = result.valid & (
        result.interaction_type == int(InteractionType.SCATTERING)
    ).any(dim=-1)
    return float(result.a[scattering].abs().square().sum())


def _deterministic_scattering(scene: Scene, seed: int) -> float:
    del seed
    result = solve_deterministic(
        scene,
        DeterministicConfig(
            max_depth=1,
            components={"scattering"},
            scattering_samples_per_m2=64.0,
        ),
    )
    return float(result.component_power["scattering"])


def _basic_scattering(scene: Scene, seed: int) -> float:
    result = solve_basic(
        scene, BasicConfig(samples=16_384, seed=seed, components={"scattering"})
    )
    return float(result.component_power["scattering"])


def _bdpt_scattering(scene: Scene, seed: int) -> float:
    result = solve_bdpt(
        scene,
        BDPTConfig(
            samples=32_768,
            seed=seed,
            max_depth=2,
            components={"scattering"},
        ),
    )
    return float(result.component_power["scattering"])


SOLVERS = (
    SolverAdapter("path", "complex", _path_transmission, _path_scattering),
    SolverAdapter(
        "deterministic",
        "complex",
        _deterministic_transmission,
        _deterministic_scattering,
    ),
    SolverAdapter("mc_basic", "power", _basic_transmission, _basic_scattering),
    SolverAdapter("bdpt", "power", _bdpt_transmission, _bdpt_scattering),
)


__all__ = [
    "SOLVERS",
    "SolverAdapter",
    "dispersive_multilayer",
    "rigid_transform",
    "rough_scene",
    "transmission_scene",
]
