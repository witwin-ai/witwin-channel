"""Typed scattering resources owned by :class:`CompiledScene`."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.materials.models import Roughness
from witwin.channel_native.scattering.phase_screen import PhaseScreenRuntime
from witwin.channel_native.scattering.tables import KirchhoffTable
from witwin.channel_native.scene.stores.assignments import AssignmentStore
from witwin.channel_native.scene.stores.materials import MaterialStore

__all__ = [
    "KirchhoffRuntimeResources",
    "PhaseScreenRuntimeResources",
    "RoughMaterialRuntime",
    "ScatteringResourceKey",
    "build_kirchhoff_resources",
    "build_phase_screen_resources",
]


@dataclass(frozen=True, slots=True)
class ScatteringResourceKey:
    """Identity of device resources derived from one compiled scene."""

    material_cache_token: str
    assignment_version: int
    device: torch.device


@dataclass(frozen=True, slots=True)
class RoughMaterialRuntime:
    """Per-material scattering runtime: table plus layer-stack inputs."""

    material_index: int
    table: KirchhoffTable
    layers: tuple[tuple[float, float, float, float], ...]
    roughness: Roughness


@dataclass(frozen=True, slots=True)
class KirchhoffRuntimeResources:
    """Kirchhoff tables and matching per-material runtime records."""

    key: ScatteringResourceKey
    tables: dict[int, KirchhoffTable]
    materials: dict[int, RoughMaterialRuntime]


@dataclass(frozen=True, slots=True)
class PhaseScreenRuntimeResources:
    """GPU phase-screen runtimes for one compiled scene resource key."""

    key: ScatteringResourceKey
    runtimes: dict[int, PhaseScreenRuntime]


def build_kirchhoff_resources(
    store: MaterialStore, key: ScatteringResourceKey
) -> KirchhoffRuntimeResources:
    """Build all rough-material resources without publishing partial state."""

    # Keep the supported package-level instrumentation seam lazy.
    from witwin.channel_native import scattering

    tables: dict[int, KirchhoffTable] = {}
    materials: dict[int, RoughMaterialRuntime] = {}
    for index in range(int(store.material_id.numel())):
        if int(store.scatter_model_id[index]) != 1:
            continue
        offset = int(store.layer_offset[index])
        count = int(store.layer_count[index])
        layers = tuple(
            (
                float(store.layer_thickness_m[row]),
                float(store.layer_eps_r[row]),
                float(store.layer_sigma_e[row]),
                float(store.layer_mu_r[row]),
            )
            for row in range(offset, offset + count)
        )
        roughness = Roughness(
            rms_height_m=float(store.rough_sigma_h_m[index]),
            corr_length_x_m=float(store.rough_corr_x_m[index]),
            corr_length_y_m=float(store.rough_corr_y_m[index]),
            principal_axis_rad=float(store.rough_axis_rad[index]),
        )
        table = scattering.build_kirchhoff_table(
            roughness, layers, store.frequency_hz, device=key.device
        )
        tables[index] = table
        materials[index] = RoughMaterialRuntime(
            material_index=index,
            table=table,
            layers=layers,
            roughness=roughness,
        )
    return KirchhoffRuntimeResources(key=key, tables=tables, materials=materials)


def build_phase_screen_resources(
    assignments: AssignmentStore, key: ScatteringResourceKey
) -> PhaseScreenRuntimeResources:
    """Build all phase-screen resources without publishing partial state."""

    from witwin.channel_native import scattering

    runtimes = {
        index: scattering.PhaseScreenRuntime(screen, device=key.device)
        for index, screen in assignments.structure_phase_screens.items()
    }
    return PhaseScreenRuntimeResources(key=key, runtimes=runtimes)


# Preserve the pre-migration import/pickle owner exposed by the MC module.
RoughMaterialRuntime.__module__ = "witwin.channel_native.montecarlo.scattering_events"
