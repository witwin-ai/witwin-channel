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
    "KirchhoffTableStack",
    "PhaseScreenRuntimeResources",
    "RoughMaterialRuntime",
    "ScatteringResourceKey",
    "build_kirchhoff_resources",
    "build_kirchhoff_table_stack",
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
class KirchhoffTableStack:
    """Stacked per-material Kirchhoff tables for the native ensemble kernel.

    ``material_slot`` maps a face material id to its slot (``-1`` when the
    material carries no Kirchhoff table). ``f_te_flat`` / ``f_tm_flat`` are the
    concatenated table values; ``table_offset`` and ``table_dims``
    (``[nti, npi, nto, npo]``) locate each slot's block so heterogeneous
    isotropic (``npi == 1``) and anisotropic (``npi == 64``) tables coexist.
    """

    material_slot: torch.Tensor
    f_te_flat: torch.Tensor
    f_tm_flat: torch.Tensor
    table_offset: torch.Tensor
    table_dims: torch.Tensor


@dataclass(frozen=True, slots=True)
class KirchhoffRuntimeResources:
    """Kirchhoff tables and matching per-material runtime records."""

    key: ScatteringResourceKey
    tables: dict[int, KirchhoffTable]
    materials: dict[int, RoughMaterialRuntime]
    stack: KirchhoffTableStack


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
    stack = build_kirchhoff_table_stack(
        tables, int(store.material_id.numel()), key.device
    )
    return KirchhoffRuntimeResources(
        key=key, tables=tables, materials=materials, stack=stack
    )


def build_kirchhoff_table_stack(
    tables: dict[int, KirchhoffTable],
    material_count: int,
    device: torch.device,
) -> KirchhoffTableStack:
    """Stack per-material tables into flat device buffers (ADR-010 op 1)."""

    material_slot = torch.full(
        (material_count,), -1, dtype=torch.int32, device=device
    )
    fte_parts: list[torch.Tensor] = []
    ftm_parts: list[torch.Tensor] = []
    offsets: list[int] = []
    dims: list[list[int]] = []
    running = 0
    for slot, material_index in enumerate(sorted(tables)):
        material_slot[material_index] = slot
        table = tables[material_index]
        fte = table.f_te.to(device=device, dtype=torch.float32).reshape(-1).contiguous()
        ftm = table.f_tm.to(device=device, dtype=torch.float32).reshape(-1).contiguous()
        fte_parts.append(fte)
        ftm_parts.append(ftm)
        offsets.append(running)
        dims.append([int(size) for size in table.f_te.shape])
        running += int(fte.numel())
    if fte_parts:
        f_te_flat = torch.cat(fte_parts)
        f_tm_flat = torch.cat(ftm_parts)
        table_offset = torch.tensor(offsets, dtype=torch.int64, device=device)
        table_dims = torch.tensor(dims, dtype=torch.int32, device=device)
    else:
        f_te_flat = torch.zeros((0,), dtype=torch.float32, device=device)
        f_tm_flat = torch.zeros((0,), dtype=torch.float32, device=device)
        table_offset = torch.zeros((0,), dtype=torch.int64, device=device)
        table_dims = torch.zeros((0, 4), dtype=torch.int32, device=device)
    return KirchhoffTableStack(
        material_slot=material_slot,
        f_te_flat=f_te_flat,
        f_tm_flat=f_tm_flat,
        table_offset=table_offset,
        table_dims=table_dims,
    )


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


# Preserve the import/pickle owner exposed by the MC events module.
RoughMaterialRuntime.__module__ = (
    "witwin.channel_native.montecarlo.events.scattering"
)
