from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Literal

import torch

from tests.support.scenes import rough_wall_structure, transmission_wall_structure
from witwin.core import (
    AntennaState,
    MaterialLayer,
    Mesh,
    PhysicalMaterial,
    ReceiverGrid,
    Scene,
    antenna_orientation_matrix,
)
from tests.support.core_world import (
    make_receiver,
    make_receiver_grid,
    make_transmitter,
)
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as solve_deterministic
from witwin.channel.montecarlo.basic import Config as BasicConfig
from witwin.channel.montecarlo.basic import solve as solve_basic
from witwin.channel.montecarlo.bdpt import Config as BDPTConfig
from witwin.channel.montecarlo.bdpt import solve as solve_bdpt
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import InteractionType, solve as solve_path


@dataclass(frozen=True, slots=True)
class SolverAdapter:
    name: str
    field_domain: Literal["complex", "power"]
    transmission: Callable[[Scene, float], complex | float]
    scattering: Callable[[Scene, int, float], float]


@dataclass(frozen=True, slots=True)
class _DebyeDispersion:
    eps_inf: float
    delta_eps: float
    tau_s: float
    sigma_dc: float = 0.0

    def complex_eps(self, frequency_hz):
        omega = 2.0 * math.pi * frequency_hz
        return (
            self.eps_inf
            + self.delta_eps / (1.0 + 1j * omega * self.tau_s)
            - 1j * self.sigma_dc / (omega * 8.854_187_812_8e-12)
        )


def dispersive_multilayer() -> PhysicalMaterial:
    return PhysicalMaterial(
        layers=(
            MaterialLayer(
                thickness_m=0.035,
                dispersion=_DebyeDispersion(
                    eps_inf=2.1,
                    delta_eps=2.4,
                    tau_s=18.0e-12,
                    sigma_dc=0.015,
                ),
            ),
            MaterialLayer(thickness_m=0.055, eps_r=5.2, sigma_e=0.025),
        ),
        name="phase-d-dispersive-stack",
    )


def transmission_scene(*, reverse: bool = False, empty: bool = False) -> Scene:
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
        endpoints=[
            make_transmitter(
                position=tx,
                polarization=torch.tensor([0.0, 0.0, 1.0]),
            ),
            make_receiver_grid(
                origin=rx,
                x_axis=torch.tensor([0.0, 1.0, 0.0]),
                y_axis=torch.tensor([0.0, 0.0, 1.0]),
                shape=(1, 1),
                spacing=(1.0, 1.0),
                polarization=torch.tensor([0.0, 0.0, 1.0]),
            ),
        ],
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
        endpoints=[
            make_transmitter(
                position=tx_position, polarization=torch.tensor([0.0, 0.0, 1.0])
            ),
            make_receiver_grid(
                origin=rx_position,
                x_axis=torch.tensor([0.0, 1.0, 0.0]),
                y_axis=torch.tensor([0.0, 0.0, 1.0]),
                shape=(1, 1),
                spacing=(1.0, 1.0),
                polarization=torch.tensor([0.0, 0.0, 1.0]),
            ),
        ],
    )


def rigid_transform(scene: Scene) -> Scene:
    orientation = torch.tensor([0.47, -0.31, 0.23])
    rotation = antenna_orientation_matrix(orientation, reference=orientation)
    translation = torch.tensor([7.0, -4.0, 2.5])

    def point(value: torch.Tensor) -> torch.Tensor:
        return rotation @ value + translation

    structures = []
    for structure in scene.structures:
        geometry = structure.geometry
        if not isinstance(geometry, Mesh):
            raise TypeError("Phase D rigid-transform fixtures require Mesh geometry.")
        structures.append(
            replace(
                structure,
                geometry=Mesh(
                    geometry.vertices @ rotation.T + translation,
                    geometry.faces,
                    recenter=False,
                    fill_mode="surface",
                    topology_diagnostics=False,
                ),
            )
        )
    transmitters = [
        make_transmitter(
            position=point(transmitter.position),
            orientation=transmitter.orientation,
            polarization=rotation @ transmitter.polarization,
            element_positions=transmitter.element_positions,
            weights=transmitter.weights,
            synthetic_array=transmitter.synthetic_array,
            pattern=transmitter.pattern,
            power_w=transmitter.power_w,
        )
        for transmitter in scene.endpoints
        if transmitter.role == "tx"
    ]
    receivers = []
    for receiver in scene.endpoints:
        if receiver.role != "rx":
            continue
        if isinstance(receiver, AntennaState) and not isinstance(
            receiver, ReceiverGrid
        ):
            receivers.append(
                make_receiver(
                    position=point(receiver.position),
                    orientation=receiver.orientation,
                    polarization=rotation @ receiver.polarization,
                    element_positions=receiver.element_positions,
                    weights=receiver.weights,
                    synthetic_array=receiver.synthetic_array,
                    pattern=receiver.pattern,
                )
            )
        else:
            receivers.append(
                make_receiver_grid(
                    origin=point(receiver.origin),
                    x_axis=rotation @ receiver.x_axis,
                    y_axis=rotation @ receiver.y_axis,
                    shape=receiver.shape,
                    spacing=receiver.spacing,
                    orientation=receiver.orientation,
                    polarization=rotation @ receiver.polarization,
                    element_positions=receiver.element_positions,
                    weights=receiver.weights,
                    synthetic_array=receiver.synthetic_array,
                    pattern=receiver.pattern,
                )
            )
    return Scene(
        structures=structures,
        endpoints=[*transmitters, *receivers],
        metadata=scene.metadata,
    )


def _path_transmission(scene: Scene, reference_frequency_hz: float) -> complex:
    component = "los" if not scene.structures else "transmission"
    result = solve_path(
        scene,
        PathConfig(max_depth=1, components={component}),
        reference_frequency_hz=reference_frequency_hz,
    )
    values = result.a[result.valid].reshape(-1)
    assert values.numel() == 1
    return complex(values[0].item())


def _deterministic_transmission(scene: Scene, reference_frequency_hz: float) -> complex:
    component = "los" if not scene.structures else "transmission"
    result = solve_deterministic(
        scene,
        DeterministicConfig(max_depth=1, components={component}),
        reference_frequency_hz=reference_frequency_hz,
    )
    return complex(result.component_fields[component].reshape(-1)[0].item())


def _basic_transmission(scene: Scene, reference_frequency_hz: float) -> float:
    component = "los" if not scene.structures else "transmission"
    result = solve_basic(
        scene,
        BasicConfig(samples=512, max_depth=1, components={component}),
        reference_frequency_hz=reference_frequency_hz,
    )
    return float(result.component_power[component])


def _bdpt_transmission(scene: Scene, reference_frequency_hz: float) -> float:
    component = "los" if not scene.structures else "transmission"
    result = solve_bdpt(
        scene,
        BDPTConfig(samples=512, max_depth=1, components={component}),
        reference_frequency_hz=reference_frequency_hz,
    )
    return float(result.component_power[component])


def _path_scattering(scene: Scene, seed: int, reference_frequency_hz: float) -> float:
    del seed
    result = solve_path(
        scene,
        PathConfig(
            max_depth=1,
            components={"scattering"},
            scattering_samples_per_m2=32.0,
        ),
        reference_frequency_hz=reference_frequency_hz,
    )
    scattering = result.valid & (
        result.interaction_type == int(InteractionType.SCATTERING)
    ).any(dim=-1)
    return float(result.a[scattering].abs().square().sum())


def _deterministic_scattering(
    scene: Scene, seed: int, reference_frequency_hz: float
) -> float:
    del seed
    result = solve_deterministic(
        scene,
        DeterministicConfig(
            max_depth=1,
            components={"scattering"},
            scattering_samples_per_m2=64.0,
        ),
        reference_frequency_hz=reference_frequency_hz,
    )
    return float(result.component_power["scattering"])


def _basic_scattering(scene: Scene, seed: int, reference_frequency_hz: float) -> float:
    result = solve_basic(
        scene,
        BasicConfig(samples=16_384, seed=seed, components={"scattering"}),
        reference_frequency_hz=reference_frequency_hz,
    )
    return float(result.component_power["scattering"])


def _bdpt_scattering(scene: Scene, seed: int, reference_frequency_hz: float) -> float:
    result = solve_bdpt(
        scene,
        BDPTConfig(
            samples=32_768,
            seed=seed,
            max_depth=2,
            components={"scattering"},
        ),
        reference_frequency_hz=reference_frequency_hz,
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
