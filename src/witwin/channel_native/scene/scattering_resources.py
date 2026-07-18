"""Typed scattering resources owned by :class:`CompiledScene`."""

from __future__ import annotations

from dataclasses import dataclass, replace

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
        # The float64 numpy build runs unchanged (host float() reads are the
        # sanctioned compile-time island). When any roughness/layer store tensor
        # participates in AD, route f_te/f_tm through the native build adjoint so
        # the resident table values keep a graph to those leaves (ADR-015 Part
        # C); otherwise today's path is bitwise identical.
        table = scattering.build_kirchhoff_table(
            roughness, layers, store.frequency_hz, device=key.device
        )
        table = _maybe_differentiate_table(store, index, offset, count, table, key)
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


def _maybe_differentiate_table(
    store: MaterialStore,
    index: int,
    offset: int,
    count: int,
    table: KirchhoffTable,
    key: ScatteringResourceKey,
) -> KirchhoffTable:
    """Attach the native build adjoint when store leaves participate in AD.

    ``store.frequency_hz`` is a host float, so a store-driven build carries no
    differentiable frequency leaf here (the AD wrapper still handles frequency
    when a caller supplies a live frequency tensor); the store trigger is the
    roughness/layer slices. When nothing requires grad the numpy table is
    returned unchanged (bitwise primal path).
    """

    from witwin.channel_native.runtime.autograd_contracts import _ad_geometry_live

    # Store parameter tensors are host-side float32 metadata; the native build
    # adjoint runs on device, so move the differentiable leaves to key.device.
    # The copy is differentiable, so gradients still accumulate on the original
    # store leaves.
    sigma_h = store.rough_sigma_h_m[index].to(key.device)
    corr_x = store.rough_corr_x_m[index].to(key.device)
    corr_y = store.rough_corr_y_m[index].to(key.device)
    thickness = store.layer_thickness_m[offset : offset + count].to(key.device)
    eps_r = store.layer_eps_r[offset : offset + count].to(key.device)
    sigma_e = store.layer_sigma_e[offset : offset + count].to(key.device)
    mu_r = store.layer_mu_r[offset : offset + count].to(key.device)
    frequency = torch.tensor(
        store.frequency_hz, dtype=torch.float32, device=key.device
    )
    if not _ad_geometry_live(sigma_h, corr_x, corr_y, thickness, eps_r, sigma_e, frequency):
        return table

    from witwin.channel_native.scattering.kernels.table_build_ad import (
        kirchhoff_table_build_ad,
    )

    f_te, f_tm = kirchhoff_table_build_ad(
        sigma_h,
        corr_x,
        corr_y,
        thickness,
        eps_r,
        sigma_e,
        mu_r,
        frequency,
        table=table,
    )
    return replace(table, f_te=f_te, f_tm=f_tm)


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
