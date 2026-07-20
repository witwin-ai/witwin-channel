"""Typed scattering resources owned by :class:`CompiledScene`."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from witwin.channel_native.materials.models import PhaseScreen, Roughness
from witwin.channel_native.scattering.phase_screen import PhaseScreenRuntime
from witwin.channel_native.scattering.tables import KirchhoffTable, MAX_RMS_SLOPE
from witwin.channel_native.scene.kernels.rayd_scene import RayDSceneResource
from witwin.channel_native.scene.stores.assignments import AssignmentStore
from witwin.channel_native.scene.stores.materials import MaterialStore

__all__ = [
    "KirchhoffRuntimeResources",
    "KirchhoffTableStack",
    "PhaseScreenResourceKey",
    "PhaseScreenRuntimeResources",
    "PhaseScreenStructureResource",
    "RoughMaterialRuntime",
    "ScatteringResourceKey",
    "build_kirchhoff_resources",
    "build_kirchhoff_table_stack",
    "build_phase_screen_resources",
    "realization_phase_screens",
]


@dataclass(frozen=True, slots=True)
class ScatteringResourceKey:
    """Identity of device resources derived from one compiled scene."""

    material_cache_token: str
    assignment_version: int
    device: torch.device


@dataclass(frozen=True, slots=True)
class PhaseScreenResourceKey:
    """Identity of immutable phase-screen resources for one compiled scene."""

    material_cache_token: str
    geometry_version: int
    assignment_version: int
    phase_screen_token: tuple[tuple[object, ...], ...]
    structure_uv_presence: tuple[tuple[bool, bool], ...]
    rayd_scene_identity: int
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


def _require_phase_screen_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tensor.device != device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(f"phase-screen {name} must be {dtype} {shape} on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"phase-screen {name} must be contiguous")


@dataclass(frozen=True, slots=True)
class PhaseScreenStructureResource:
    """Immutable scene-owned inputs for one realization phase screen.

    Endpoint-, frequency-, and solver-config-dependent patch subdivision and
    visibility are intentionally absent.  They remain solve-plan state.
    """

    structure_index: int
    material_index: int
    face_range: tuple[int, int]
    first_face: int
    face_count: int
    uv_vertex_count: int
    runtime: PhaseScreenRuntime
    uv_vertices: torch.Tensor
    face_uv: torch.Tensor
    uv_tris: torch.Tensor
    face_areas_m2: torch.Tensor
    uv_world_scale_m: float
    rms_slope: float

    def __post_init__(self) -> None:
        start, stop = self.face_range
        if self.structure_index < 0 or self.material_index < 0:
            raise ValueError("phase-screen structure/material indices must be non-negative")
        if start < 0 or stop < start or self.first_face != start:
            raise ValueError("phase-screen face_range/first_face is inconsistent")
        if self.face_count != stop - start:
            raise ValueError("phase-screen face_count must match face_range")
        if self.uv_vertex_count < 0:
            raise ValueError("phase-screen uv_vertex_count must be non-negative")
        device = self.runtime.heights_m.device
        expected = (
            ("uv_vertices", self.uv_vertices, torch.float32, (self.uv_vertex_count, 2)),
            ("face_uv", self.face_uv, torch.int64, (self.face_count, 3)),
            ("uv_tris", self.uv_tris, torch.float32, (self.face_count, 3, 2)),
            ("face_areas_m2", self.face_areas_m2, torch.float32, (self.face_count,)),
        )
        for name, tensor, dtype, shape in expected:
            _require_phase_screen_tensor(
                name, tensor, dtype=dtype, shape=shape, device=device
            )
        if not math.isfinite(self.uv_world_scale_m) or self.uv_world_scale_m <= 0.0:
            raise ValueError("phase-screen uv_world_scale_m must be finite and positive")
        if not math.isfinite(self.rms_slope) or self.rms_slope < 0.0:
            raise ValueError("phase-screen rms_slope must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PhaseScreenRuntimeResources:
    """Immutable per-structure phase-screen resources for a compiled scene."""

    key: PhaseScreenResourceKey
    structures: dict[int, PhaseScreenStructureResource]


def realization_phase_screens(
    materials: MaterialStore, assignments: AssignmentStore
) -> dict[int, PhaseScreen]:
    """Resolve mutually exclusive realization assignments at the scene boundary."""

    screens: dict[int, PhaseScreen] = {}
    for structure_index, screen in sorted(
        assignments.structure_phase_screens.items()
    ):
        material_index = int(assignments.structure_material_id[structure_index])
        rough = int(materials.scatter_model_id[material_index]) == 1
        if screen.mode == "realization_coherent":
            screens[structure_index] = screen
        elif rough:
            raise RuntimeError(
                "scattering_mode_conflict: structure "
                f"{structure_index} combines a PhaseScreen(mode='ensemble_bsdf') "
                "with a Kirchhoff-rough material; realization and ensemble "
                "models must never be summed for one surface (contract 6.7.3). "
                "Use mode='realization_coherent' to replace the ensemble lobe, "
                "or drop the phase screen to keep the material ensemble table"
            )
        else:
            raise RuntimeError(
                "PhaseScreen mode 'ensemble_bsdf' requires a Kirchhoff-rough "
                "material to define the ensemble statistics; assign a rough "
                "PhysicalSurface or use mode='realization_coherent'"
            )
    return screens


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
    materials: MaterialStore,
    assignments: AssignmentStore,
    rayd: RayDSceneResource,
    key: PhaseScreenResourceKey,
) -> PhaseScreenRuntimeResources:
    """Build immutable phase-screen resources without publishing partial state.

    This is called lazily by the first phase-screen consumer.  The build owns
    only scene-static data; patch subdivision and visibility remain solve-plan
    work because they depend on endpoints, frequency, or solver configuration.
    """

    from witwin.channel_native import scattering

    _require_phase_screen_rayd_identity(rayd, key)
    if not assignments.structure_phase_screens:
        return PhaseScreenRuntimeResources(key=key, structures={})
    if not rayd.available:
        rayd.require_resource()

    mesh_tensors = rayd.mesh_tensors
    if len(mesh_tensors) != int(assignments.structure_material_id.shape[0]):
        raise RuntimeError(
            "phase-screen scene resource mesh/structure count is inconsistent"
        )
    if len(key.structure_uv_presence) != len(mesh_tensors):
        raise RuntimeError(
            "phase-screen scene resource UV-presence contract is inconsistent"
        )
    face_ranges: list[tuple[int, int]] = []
    running = 0
    for mesh in mesh_tensors:
        if len(mesh) < 4:
            raise RuntimeError("phase-screen scene resource mesh layout is incomplete")
        face_count = int(mesh[1].shape[0])
        face_ranges.append((running, running + face_count))
        running += face_count

    records = rayd.edge_records()
    if running != int(records.faces.shape[0]):
        raise RuntimeError("phase-screen scene resource face ranges are inconsistent")

    screens = sorted(realization_phase_screens(materials, assignments).items())
    runtimes = {
        structure_index: scattering.PhaseScreenRuntime(screen, device=key.device)
        for structure_index, screen in screens
    }
    resources: dict[int, PhaseScreenStructureResource] = {}
    for structure_index, screen in screens:
        material_index = int(assignments.structure_material_id[structure_index])

        mesh_vertices, mesh_faces, mesh_uv, mesh_face_uv = mesh_tensors[
            structure_index
        ][:4]
        del mesh_vertices
        face_range = face_ranges[structure_index]
        face_count = face_range[1] - face_range[0]
        runtime = runtimes[structure_index]
        uv_vertex_count = int(mesh_uv.shape[0])
        _require_phase_screen_uv_presence(key, structure_index)
        if face_count == 0:
            resources[structure_index] = PhaseScreenStructureResource(
                structure_index=structure_index,
                material_index=material_index,
                face_range=face_range,
                first_face=face_range[0],
                face_count=0,
                uv_vertex_count=0,
                runtime=runtime,
                uv_vertices=torch.empty(
                    (0, 2), device=key.device, dtype=torch.float32
                ),
                face_uv=torch.empty(
                    (0, 3), device=key.device, dtype=torch.int64
                ),
                uv_tris=torch.empty(
                    (0, 3, 2), device=key.device, dtype=torch.float32
                ),
                face_areas_m2=torch.empty(
                    (0,), device=key.device, dtype=torch.float32
                ),
                uv_world_scale_m=1.0,
                rms_slope=0.0,
            )
            continue
        if uv_vertex_count == 0 or tuple(mesh_uv.shape[1:]) != (2,):
            raise RuntimeError("phase-screen structure UV shape is inconsistent")
        if tuple(mesh_faces.shape) != (face_count, 3):
            raise RuntimeError("phase-screen structure face shape is inconsistent")
        if tuple(mesh_face_uv.shape) != (face_count, 3):
            raise RuntimeError("phase-screen structure face_uv shape is inconsistent")

        uv_vertices = mesh_uv.to(device=key.device, dtype=torch.float32).contiguous()
        face_uv = mesh_face_uv.to(device=key.device, dtype=torch.int64).contiguous()
        # Structure construction already checks face_uv value bounds on the
        # host.  Here F_s/P_s and the resident shapes are checked without a
        # device-to-host value transfer in the lazy solve boundary.
        uv_tris = uv_vertices[face_uv].contiguous()
        first_face = face_range[0]
        faces = records.faces[first_face : face_range[1]].to(torch.int64)
        tri = records.vertices[faces]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        face_areas_m2 = (
            0.5
            * torch.linalg.vector_norm(torch.cross(e1, e2, dim=-1), dim=-1)
        ).contiguous()
        uv_e1 = uv_tris[:, 1] - uv_tris[:, 0]
        uv_e2 = uv_tris[:, 2] - uv_tris[:, 0]
        uv_areas = 0.5 * (
            uv_e1[:, 0] * uv_e2[:, 1] - uv_e1[:, 1] * uv_e2[:, 0]
        ).abs()
        uv_world_scale_m = float(
            torch.sqrt(face_areas_m2.sum() / uv_areas.sum().clamp_min(1.0e-20))
        )
        rms_slope = _phase_screen_rms_slope(
            runtime, uv_world_scale_m, structure_index
        )
        resources[structure_index] = PhaseScreenStructureResource(
            structure_index=structure_index,
            material_index=material_index,
            face_range=face_range,
            first_face=first_face,
            face_count=face_count,
            uv_vertex_count=uv_vertex_count,
            runtime=runtime,
            uv_vertices=uv_vertices,
            face_uv=face_uv,
            uv_tris=uv_tris,
            face_areas_m2=face_areas_m2,
            uv_world_scale_m=uv_world_scale_m,
            rms_slope=rms_slope,
        )
    return PhaseScreenRuntimeResources(key=key, structures=resources)


def _require_phase_screen_rayd_identity(
    rayd: RayDSceneResource, key: PhaseScreenResourceKey
) -> None:
    if id(rayd) != key.rayd_scene_identity:
        raise RuntimeError(
            "phase-screen resource key does not own the supplied RayD scene"
        )


def _require_phase_screen_uv_presence(
    key: PhaseScreenResourceKey, structure_index: int
) -> None:
    uv_present, face_uv_present = key.structure_uv_presence[structure_index]
    if not uv_present or not face_uv_present:
        raise RuntimeError(
            "realization_coherent phase screen requires structure UV "
            f"(structure {structure_index} has none); contract section 6"
        )


def _phase_screen_rms_slope(
    runtime: PhaseScreenRuntime, uv_scale_m: float, structure_index: int
) -> float:
    """Validate the static tangent-plane applicability guard exactly once."""

    if uv_scale_m <= 0.0 or not math.isfinite(uv_scale_m):
        raise RuntimeError(
            "phase_screen_geometry_limit_exceeded: structure "
            f"{structure_index} has a degenerate UV-to-world scale"
        )
    heights = runtime.heights_m
    rows, cols = heights.shape
    slope_u = (heights[:, 1:] - heights[:, :-1]) * (cols / uv_scale_m)
    slope_v = (heights[1:, :] - heights[:-1, :]) * (rows / uv_scale_m)
    mean_square_u = 0.0 if slope_u.numel() == 0 else float(slope_u.square().mean())
    mean_square_v = 0.0 if slope_v.numel() == 0 else float(slope_v.square().mean())
    rms_slope = math.sqrt(mean_square_u + mean_square_v)
    if rms_slope > MAX_RMS_SLOPE:
        raise RuntimeError(
            "phase_screen_geometry_limit_exceeded: structure "
            f"{structure_index} phase screen RMS slope {rms_slope:.3g} exceeds "
            f"the tangent-plane limit {MAX_RMS_SLOPE:g}; heights of this "
            "magnitude change occlusion/silhouettes and cannot be represented "
            "as a pure phase screen"
        )
    return rms_slope


# Preserve the import/pickle owner exposed by the MC events module.
RoughMaterialRuntime.__module__ = (
    "witwin.channel_native.montecarlo.events.scattering"
)
