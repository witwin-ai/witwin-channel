# Copyright Xingyu Chen.
# Immutable native resources a compiled scene owns.

"""Immutable native resources a compiled scene owns.

This module is the single owner of everything a :class:`CompiledScene` holds
that is not a store: the typed RayD scene/BVH lifetime, the diffraction edge
policy and the scene-policy refinement of the exported edge geometry, the
lazily built Kirchhoff and phase-screen resources, and - since the root
``scattering`` module was merged into it - the compile-time definition of the
Kirchhoff table and the phase-screen runtime those resources are made of.

They were four modules that only ever called each other. The RayD resource is
the thing the edge refinement reads its records from and the thing the
phase-screen build takes its mesh tensors from, and the edge policy exists only
to be cached on that resource at compile time, so a reader following one
resource had to walk four files to see one lifetime. The root ``scattering``
module was the fifth: it read like "the scattering owner" while
:mod:`witwin.channel.interactions.scattering` owns per-solve path evaluation,
and everything it held was compile-time resource construction that this module
already caches.

Nothing here computes RF physics. The RayD facade validates a contract and
dispatches a native symbol; the edge refinement is scene-policy row selection
over already-exported geometry; the resource builders cache scene-static data
that the consumers used to recompute per solve, with their exception and
numerical order preserved. Moving that retained static construction across the
native boundary requires its own accepted ADR.

The merged compile-time section at the bottom of this file is kept visibly
separated behind its own banner, because it is the only part of the module that
runs float64 NumPy on the host. That is the sanctioned compile-time table
construction, not a production backend, and the banner records which side of
the CPU-compute policy line every name below it sits on.

The stores this module annotates against live in
:mod:`witwin.channel.scene.compiler`, which imports this module for the RayD
lifetime. The store names are needed for typing only, so they are imported
under ``TYPE_CHECKING`` and the runtime dependency stays one-way.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from witwin.core import PhaseScreen, SurfaceRoughness

from witwin.channel.constants import C0
from witwin.channel.materials import layer_stack_rt
from witwin.channel.runtime import mc_pack_vec3, required_symbol as _required_native_op

if TYPE_CHECKING:
    from witwin.channel.scene.compiler import AssignmentStore, MaterialStore

__all__ = [
    "DEFAULT_EDGE_POLICY",
    "EdgePolicy",
    "KirchhoffRuntimeResources",
    "KirchhoffTable",
    "KirchhoffTableStack",
    "PhaseScreenResourceKey",
    "PhaseScreenRuntime",
    "PhaseScreenRuntimeResources",
    "PhaseScreenStructureResource",
    "RayDEdgeRecords",
    "RayDSceneResource",
    "RoughMaterialRuntime",
    "ScatteringResourceKey",
    "_empty_tensor",
    "_mesh_flags",
    "build_kirchhoff_resources",
    "build_kirchhoff_table",
    "build_kirchhoff_table_stack",
    "build_phase_screen_resources",
    "build_scene_from_structures",
    "eval_bsdf",
    "generate_gaussian_realization",
    "patch_phase_integral",
    "pdf",
    "pdf_reverse",
    "rayd_scene_create",
    "rayd_scene_edge_records",
    "realization_phase_screens",
    "realization_seed",
    "refine_edge_geometry",
    "resolve_scene_edge_policy",
    "sample_directions",
]


def rayd_scene_create(
    vertices: list[torch.Tensor],
    faces: list[torch.Tensor],
    uv: list[torch.Tensor],
    face_uv: list[torch.Tensor],
    to_world_left: list[torch.Tensor],
    to_world_right: list[torch.Tensor],
    mesh_flags: list[int],
) -> object:
    resource = _required_native_op("rayd_scene_create")(
        vertices,
        faces,
        uv,
        face_uv,
        to_world_left,
        to_world_right,
        mesh_flags,
    )
    if resource is None or not bool(getattr(resource, "available", False)):
        raise RuntimeError("_channel.rayd_scene_create returned an invalid resource")
    return resource


def rayd_scene_edge_records(resource: object) -> tuple[torch.Tensor, ...]:
    out = _required_native_op("rayd_scene_edge_records")(resource)
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel.rayd_scene_edge_records must return a tensor sequence"
        )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class RayDEdgeRecords:
    vertices: torch.Tensor
    faces: torch.Tensor
    face_normals: torch.Tensor
    edge_v0: torch.Tensor
    edge_v1: torch.Tensor
    face0: torch.Tensor
    face1: torch.Tensor
    shape_id: torch.Tensor
    local_edge_id: torch.Tensor
    opposite: torch.Tensor


@dataclass(frozen=True, slots=True)
class RayDSceneResource:
    """Typed, owning wrapper for a native RayD scene resource."""

    resource: object | None = None
    mesh_tensors: tuple[tuple[torch.Tensor, ...], ...] = ()
    reason: str | None = None
    runtime_cache: dict[str, object] = field(
        default_factory=dict, compare=False, repr=False
    )

    @property
    def available(self) -> bool:
        return self.resource is not None and bool(
            getattr(self.resource, "available", False)
        )

    def require_resource(self) -> object:
        if not self.available:
            reason = "unknown" if self.reason is None else self.reason
            raise RuntimeError(f"RayD native scene is unavailable: {reason}")
        assert self.resource is not None
        return self.resource

    def edge_records(self) -> RayDEdgeRecords:
        cached = self.runtime_cache.get("edge_records")
        if cached is not None:
            return cached  # type: ignore[return-value]
        values = rayd_scene_edge_records(self.require_resource())
        if len(values) != 12:
            raise RuntimeError(
                f"RayD edge_records returned {len(values)} tensors, expected 12"
            )
        face_normals = mc_pack_vec3(values[2], values[3], values[4])
        records = RayDEdgeRecords(
            vertices=values[0],
            faces=values[1],
            face_normals=face_normals,
            edge_v0=values[5],
            edge_v1=values[6],
            face0=values[7],
            face1=values[8],
            shape_id=values[9],
            local_edge_id=values[10],
            opposite=values[11],
        )
        self.runtime_cache["edge_records"] = records
        return records


def _empty_tensor(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _mesh_flags(*, use_face_normals: bool, edges_enabled: bool, dynamic: bool) -> int:
    flags = 0
    if use_face_normals:
        flags |= 1
    if edges_enabled:
        flags |= 2
    if dynamic:
        flags |= 4
    return flags


def build_scene_from_structures(structures: tuple[object, ...]) -> RayDSceneResource:
    """Build a typed RayD native scene from Channel structures.

    This function uses the RayD native core source-linked into
    `_channel`. It does not import a RayD Python package or dispatcher.
    """

    if not structures:
        return RayDSceneResource(reason="scene has no structures")
    if not torch.cuda.is_available():
        return RayDSceneResource(reason="CUDA is unavailable")

    device = torch.device("cuda")
    vertices: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []
    uv: list[torch.Tensor] = []
    face_uv: list[torch.Tensor] = []
    to_world_left: list[torch.Tensor] = []
    to_world_right: list[torch.Tensor] = []
    mesh_flags: list[int] = []
    keepalive: list[tuple[torch.Tensor, ...]] = []

    for structure in structures:
        mesh_vertices = structure.vertices.to(
            device=device, dtype=torch.float32
        ).contiguous()
        mesh_faces = structure.faces.to(device=device, dtype=torch.int32).contiguous()
        # Structures that carry a UV parametrization forward it to the native
        # mesh (RayD carries UV end-to-end); structures without UV keep the
        # empty per-mesh tensors, preserving the pre-UV behavior exactly.
        structure_uv = getattr(structure, "uv", None)
        structure_face_uv = getattr(structure, "face_uv", None)
        if structure_uv is not None and structure_face_uv is not None:
            mesh_uv = structure_uv.to(device=device, dtype=torch.float32).contiguous()
            mesh_face_uv = structure_face_uv.to(
                device=device, dtype=torch.int32
            ).contiguous()
        else:
            mesh_uv = _empty_tensor((0, 2), dtype=torch.float32, device=device)
            mesh_face_uv = _empty_tensor((0, 3), dtype=torch.int32, device=device)
        mesh_to_world_left = _empty_tensor((0, 4), dtype=torch.float32, device=device)
        mesh_to_world_right = _empty_tensor((0, 4), dtype=torch.float32, device=device)
        vertices.append(mesh_vertices)
        faces.append(mesh_faces)
        uv.append(mesh_uv)
        face_uv.append(mesh_face_uv)
        to_world_left.append(mesh_to_world_left)
        to_world_right.append(mesh_to_world_right)
        mesh_flags.append(
            _mesh_flags(use_face_normals=False, edges_enabled=True, dynamic=False)
        )
        keepalive.append(
            (
                mesh_vertices,
                mesh_faces,
                mesh_uv,
                mesh_face_uv,
                mesh_to_world_left,
                mesh_to_world_right,
            )
        )

    resource = rayd_scene_create(
        vertices,
        faces,
        uv,
        face_uv,
        to_world_left,
        to_world_right,
        mesh_flags,
    )
    return RayDSceneResource(resource=resource, mesh_tensors=tuple(keepalive))


_BOUNDARY_EDGE_POLICIES = {"exclude", "half_plane"}
_EDGE_SELECTION_MODES = {"vertical_only", "all_edges"}


@dataclass(slots=True)
class EdgePolicy:
    vertical_ratio: float = 0.7
    edge_selection_mode: str = "all_edges"
    edge_diffraction: bool | None = True
    boundary_edge_policy: str | None = None

    def __post_init__(self) -> None:
        mode = str(self.edge_selection_mode)
        if mode not in _EDGE_SELECTION_MODES:
            raise ValueError(f"edge_selection_mode must be one of {sorted(_EDGE_SELECTION_MODES)}")
        requested = None if self.edge_diffraction is None else bool(self.edge_diffraction)
        boundary = (
            "half_plane" if requested is not False else "exclude"
        ) if self.boundary_edge_policy is None else str(self.boundary_edge_policy)
        if boundary not in _BOUNDARY_EDGE_POLICIES:
            raise ValueError(f"boundary_edge_policy must be one of {sorted(_BOUNDARY_EDGE_POLICIES)}")
        resolved = boundary == "half_plane"
        if requested is not None and requested != resolved:
            raise ValueError(
                f"edge_diffraction={requested!r} conflicts with boundary_edge_policy={boundary!r}"
            )
        self.vertical_ratio = float(self.vertical_ratio)
        self.edge_selection_mode = mode
        self.edge_diffraction = resolved
        self.boundary_edge_policy = boundary

    @property
    def vertical_only(self) -> bool:
        return self.edge_selection_mode == "vertical_only"

    @property
    def cache_key(self) -> tuple[float, str, str]:
        return (self.vertical_ratio, self.edge_selection_mode, self.boundary_edge_policy or "half_plane")


DEFAULT_EDGE_POLICY = EdgePolicy()


# Scene-policy filtering and cross-structure merging for diffraction edges.
#
# The native edge-geometry kernel selects every interior wedge and every
# boundary half-plane edge regardless of the scene's edge policy, and exports
# one boundary record per structure even when two structures share the same
# geometric edge. The refinement below rewrites the exported geometry tuple so
# that:
#
# - the scene's ``EdgePolicy`` (``vertical_only`` filter,
#   ``boundary_edge_policy``) is actually enforced for path generation (audit
#   DF-4), and
# - boundary edges shared between two structures are merged into a single wedge
#   record with the correct exterior angle instead of two duplicate half-plane
#   records that double count the diffracted field (audit D-6).

_ENDPOINT_QUANTIZATION = 1.0e-4
_NORMAL_COS_TOL = 1.0 - 1.0e-5
_TWO_PI = 2.0 * math.pi


def resolve_scene_edge_policy(scene: object) -> EdgePolicy:
    policy = getattr(scene, "metadata", {}).get("imported_edge_policy")
    return policy if isinstance(policy, EdgePolicy) else DEFAULT_EDGE_POLICY


def _lexicographic_min_first(p0: torch.Tensor, p1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    le = (
        (p0[:, 0] < p1[:, 0])
        | ((p0[:, 0] == p1[:, 0]) & (p0[:, 1] < p1[:, 1]))
        | ((p0[:, 0] == p1[:, 0]) & (p0[:, 1] == p1[:, 1]) & (p0[:, 2] <= p1[:, 2]))
    ).unsqueeze(1)
    return torch.where(le, p0, p1), torch.where(le, p1, p0)


def _duplicate_boundary_pairs(
    records: object,
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group boundary edges by quantized endpoints.

    Returns (first_of_pair, second_of_pair, extra_duplicates); the extras are
    third-or-later records of a group and are simply deselected.
    """

    device = candidate.device
    empty = torch.empty((0,), device=device, dtype=torch.long)
    index = candidate.nonzero(as_tuple=False).squeeze(1)
    if int(index.numel()) < 2:
        return empty, empty, empty
    v0 = records.edge_v0[index].to(dtype=torch.long)
    v1 = records.edge_v1[index].to(dtype=torch.long)
    p0 = torch.round(records.vertices[v0] / _ENDPOINT_QUANTIZATION).to(dtype=torch.long)
    p1 = torch.round(records.vertices[v1] / _ENDPOINT_QUANTIZATION).to(dtype=torch.long)
    first, second = _lexicographic_min_first(p0, p1)
    key = torch.cat([first, second], dim=1)
    _, inverse, counts = torch.unique(key, dim=0, return_inverse=True, return_counts=True)
    order = torch.argsort(inverse, stable=True)
    sorted_inverse = inverse[order]
    is_first = torch.ones_like(sorted_inverse, dtype=torch.bool)
    is_first[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
    is_second = torch.zeros_like(is_first)
    is_second[1:] = is_first[:-1] & (sorted_inverse[1:] == sorted_inverse[:-1])
    in_pair = counts[sorted_inverse] >= 2
    pair_first = index[order[is_first & in_pair]]
    pair_second = index[order[is_second]]
    extras = index[order[~is_first & ~is_second]]
    return pair_first, pair_second, extras


def refine_edge_geometry(
    rayd: object,
    geometry: tuple[torch.Tensor, ...],
    *,
    policy: EdgePolicy | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return the geometry tuple with policy filtering and shared-edge merges."""

    (selected, edge_pos, edge_dir, lengths, line_min, line_max, n0, n1, face0, face1, exterior_angle) = geometry
    if int(selected.numel()) == 0:
        return geometry
    records = rayd.edge_records()
    if policy is None:
        policy = rayd.runtime_cache.get("edge_policy")
    if not isinstance(policy, EdgePolicy):
        policy = DEFAULT_EDGE_POLICY

    selected = selected.clone()
    n1 = n1.clone()
    face1 = face1.clone()
    exterior_angle = exterior_angle.clone()

    boundary = records.face1 < 0
    pair_first, pair_second, extras = _duplicate_boundary_pairs(records, selected & boundary)
    if int(pair_first.numel()) > 0:
        normal_a = n0[pair_first]
        normal_b = n0[pair_second]
        normal_dot = (normal_a * normal_b).sum(dim=1).clamp(-1.0, 1.0)
        interior_angle = torch.arccos((-normal_dot).clamp(-1.0, 1.0))
        merged_exterior = _TWO_PI - interior_angle
        coplanar = normal_dot.abs() >= _NORMAL_COS_TOL
        keep = (merged_exterior > math.pi * (1.0 + 1.0e-6)) & ~coplanar
        selected[pair_first] = keep
        selected[pair_second] = False
        n1[pair_first] = normal_b
        face1[pair_first] = face0[pair_second]
        exterior_angle[pair_first] = merged_exterior
        boundary = boundary.clone()
        boundary[pair_first] = False
    if int(extras.numel()) > 0:
        selected[extras] = False

    if policy.boundary_edge_policy != "half_plane":
        selected &= ~boundary
    if policy.vertical_only:
        v0 = records.edge_v0.to(dtype=torch.long)
        v1 = records.edge_v1.to(dtype=torch.long)
        delta = records.vertices[v1] - records.vertices[v0]
        length = delta.norm(dim=1).clamp_min(1.0e-6)
        selected &= (delta[:, 2].abs() / length) > float(policy.vertical_ratio)

    return (selected, edge_pos, edge_dir, lengths, line_min, line_max, n0, n1, face0, face1, exterior_angle)


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
    roughness: SurfaceRoughness


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
        roughness = SurfaceRoughness(
            rms_height_m=float(store.rough_sigma_h_m[index]),
            correlation_length_x_m=float(store.rough_corr_x_m[index]),
            correlation_length_y_m=float(store.rough_corr_y_m[index]),
            principal_axis_rad=float(store.rough_axis_rad[index]),
        )
        # The float64 numpy build runs unchanged (host float() reads are the
        # sanctioned compile-time island). When any roughness/layer store tensor
        # participates in AD, route f_te/f_tm through the native build adjoint so
        # the resident table values keep a graph to those leaves (ADR-015 Part
        # C); otherwise today's path is bitwise identical.
        table = build_kirchhoff_table(
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

    from witwin.channel.runtime import _ad_geometry_live

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

    from witwin.channel.kernels.scattering import (
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
        structure_index: PhaseScreenRuntime(screen, device=key.device)
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


# ==========================================================================
# COMPILE-TIME SCATTERING RESOURCES  (merged from witwin/channel/scattering.py)
# ==========================================================================
#
# CPU-COMPUTE POLICY LINE.  Everything ABOVE this banner is device-resident
# resource lifetime, validation and packing.  Everything BELOW it is offline
# compile-time construction: ``build_kirchhoff_table`` and
# ``generate_gaussian_realization`` run float64 NumPy on the HOST.  CLAUDE.md
# sanctions exactly that and nothing more: offline/compile-time table
# construction is not a production numerical backend and must not grow into
# production hot-path physics.  These run once per compiled scene, upload
# once, and are never re-entered per solve.
#
# The runtime side of the same subject stays on the production side of the
# line: ``eval_bsdf``, ``sample_directions``, ``pdf`` and ``pdf_reverse`` are
# thin facades over ``witwin.channel.kernels.scattering``, so every production
# eval/sample/PDF/event-budget operation requires native CUDA.  Nothing below
# this banner is a Torch or CPU production alternative, and nothing below it
# may grow into per-solve physics.
#
# Per-solve path evaluation is owned by
# ``witwin.channel.interactions.scattering`` and lives nowhere in this file.
#
# The physics contract this section implements, verbatim from the module it
# was merged from:
#
#
# Tables are precomputed once in float64 at scene compile, uploaded once, and
# all production eval/sample/PDF/event-budget operations require native CUDA.
# PyTorch and CPU implementations are not production runtime alternatives.
#
# Kirchhoff ensemble BSDF tables (contract sections 6, 7.2)
# ---------------------------------------------------------
#
# Conventions (fixed here, used by every consumer):
#
# - The table lives in the LOCAL roughness frame: ``z`` is the mean-plane
#   normal, ``x``/``y`` are the roughness principal axes (``corr_length_x_m``
#   along ``x``). Callers rotate world directions into this frame using the
#   stored ``principal_axis_rad`` before evaluating.
# - ``wi``/``wo`` are unit vectors POINTING AWAY from the surface
#   (``z > 0``): ``wi`` toward the source, ``wo`` toward the observation
#   direction. The incident propagation direction is ``-wi``, so the
#   scattering wave-vector transfer is ``q = k_s - k_i = k0*(wo + wi)`` with
#   normal component ``q_n = k0*(cos_theta_o + cos_theta_i) > 0``.
# - ``f_te``/``f_tm`` are co-pol POWER BSDF values per steradian, defined so
#   that ``sum_hemisphere f * |cos_theta_o| dOmega = R_diff`` for each
#   incidence bin (exact on the table's own outgoing grid, by construction).
#
# Raw lobe shape (before normalization)::
#
#     f_raw_q(wi, wo) = |q|^4 / (16*pi^2 * q_n^2 * cos_theta_i * cos_theta_o)
#                       * |r_q_stack(cos_theta_h)|^2 * I(q_par, q_n)
#
# where ``I`` is the production Beckmann series implemented in this module
# and the polarized Fresnel factor is evaluated at the SPECULAR-EQUIVALENT
# local incidence angle onto the half vector
# ``h = normalize(wo - wi_dir) = normalize(wo + wi)``::
#
#     cos_theta_h = wi . h = wo . h = (1 + wi . wo) / |wo + wi|
#
# Documented choice: ``wi . h`` is the angle between the incident ray and
# the micro-plane normal that specularly maps ``wi`` to ``wo``; it equals
# ``cos_theta_i`` at the lobe peak, so the kernel reproduces the exact
# stack reflectance ``R_bar_q(theta_i)`` there and tracks its angular /
# Fabry-Perot structure across the lobe. (Evaluating at ``|h . n|`` instead
# would sample the stack at NORMAL incidence for every near-specular pair
# and misses the budget by ``R_bar(theta_i)/R_bar(0)``  -  measured up to 4x  -
# so it cannot satisfy the normalization tolerance band.) ``wi . h`` is
# exactly reciprocal, as are ``q`` and the geometry prefactor.
#
# Derivation of the prefactor (standard scalar Kirchhoff): the tangent-plane
# aperture integral gives ``E_s = (j*k0/(4*pi*R)) * F * r *
# Int exp(-j*q.x - j*q_n*h) dA`` with the Kirchhoff geometry factor
# ``F = |q|^2/(k0*q_n)`` (Beckmann's ``F`` rescaled so a smooth plate
# conserves energy: ``F -> 2*cos_theta_i`` at specular). Ensemble-averaging
# the diffuse part gives ``<|Int|^2> = A * I(q)``, and dividing the scattered
# power density by ``A * cos_theta_i * cos_theta_o`` (BSDF flux convention)
# yields the expression above. Its hemispheric energy: substituting
# ``d^2 q_par = k0^2 * cos_theta_o dOmega_o`` and Parseval
# ``(1/(2*pi)^2) Int I(q_par; q_n) d^2 q_par = 1 - exp(-g)`` shows
# ``Int f_raw*cos dOmega -> R_bar*(1 - C_r^2) = R_diff`` near specular
# (``exp(-g) = C_r^2`` at ``q_n = 2*k0*cos_theta_i``). Residual horizon,
# Fresnel-variation and grid errors are removed by reciprocal symmetric matrix
# balancing. Its diagonal factor acts on both incident and outgoing states, so
# the final production table retains Helmholtz reciprocity while matching every
# directional diffuse-energy budget.
#
# Energy accounting is exact on the discrete outgoing grid: the hemisphere
# sum of ``f * cos * dOmega`` over the table bins equals ``r_diff`` bitwise
# (up to float32 rounding), which is also the measure the sampling CDFs use.
#
# Precompute runs in float64 numpy on CPU; the frozen table holds float32
# torch tensors on the requested device. Runtime eval/sample/pdf are pure
# batched torch (GPU-first).
#
# Realization-coherent phase screens (contract section 6, plan 6.7)
# -----------------------------------------------------------------
#
# Heights NEVER displace geometry: RayD intersects the mean plane and the
# sampled metric height ``h(u, v)`` enters only through the complex phasor
# ``exp(-j*q_n*h)`` with ``q_n = (k_s - k_i) . n_hat`` (``e^{+j w t}`` /
# ``e^{-j k r}`` conventions, matching ``core.field_state.PHASE_CONVENTION``
# and the CPU oracle). Footprint averaging must happen on the PHASOR, never
# on heights: ``E[exp(-j*q_n*h)] != exp(-j*q_n*E[h])``.
#
# Texture convention: ``height[iy, ix]`` samples the UV point
# ``u = (ix + 0.5)/W``, ``v = (iy + 0.5)/H`` (texel centers). Bilinear
# interpolation between texel centers with EDGE CLAMP at the borders (no
# wrap): UV coordinates outside the half-texel margin reuse the border texel
# value. Implemented with explicit gathers (not ``grid_sample``) so the
# border behavior is exact and documented.
# ==========================================================================

# Grid resolution fixed by the implementation contract (section 6).
N_COS_THETA_I = 32
# An anisotropic reciprocal table must use the same directional state grid on
# both sides of f(wi, wo).  A coarser incidence azimuth grid followed by a
# per-incidence normalization cannot preserve reciprocity.  64 keeps the
# contract's outgoing resolution and makes the discrete transport matrix
# square, so symmetric balancing below can enforce both invariants exactly.
N_PHI_I_ANISO = 64
N_COS_THETA_O = 32
N_PHI_O = 64

# Applicability guards (contract section 6): tangent-plane approximation
# needs k0*l >= ~6 and moderate RMS slope sqrt(2)*sigma_h/l <= 0.5.
MIN_K0_CORR_LENGTH = 6.0
MAX_RMS_SLOPE = 0.5


def _kirchhoff_diffuse_lobe_series(
    q_par_x, q_par_y, q_n, sigma_h, lx, ly, n_terms: int = 64
):
    """Production Beckmann series for the Gaussian-correlation lobe."""

    qx, qy, qn = np.broadcast_arrays(
        np.asarray(q_par_x, dtype=np.float64),
        np.asarray(q_par_y, dtype=np.float64),
        np.asarray(q_n, dtype=np.float64),
    )
    g = (qn * float(sigma_h)) ** 2
    rho2 = (qx * lx) ** 2 + (qy * ly) ** 2
    m_flat = np.arange(1, n_terms + 1, dtype=np.float64)
    shape = (n_terms,) + (1,) * g.ndim
    m = m_flat.reshape(shape)
    log_fact = np.cumsum(np.log(m_flat)).reshape(shape)
    with np.errstate(divide="ignore"):
        log_g = np.log(g)
    log_term = m * log_g - log_fact - np.log(m) - rho2 / (4.0 * m) - g
    series = np.exp(log_term).sum(axis=0)
    result = np.pi * lx * ly * series
    return result if result.ndim else float(result)

@dataclass(frozen=True, slots=True)
class KirchhoffTable:
    """Precomputed Kirchhoff ensemble BSDF for one rough material.

    All tensors are float32 on one device. Axis tensors hold CELL CENTERS:
    ``cos_theta_i``/``cos_theta_o`` are uniform in cos over (0, 1],
    ``phi_i``/``phi_o`` uniform over [0, 2*pi). ``phi_i`` has one entry for
    isotropic roughness (``lx == ly``).
    """

    # Axes (cell centers).
    cos_theta_i: torch.Tensor  # [Nti]
    phi_i: torch.Tensor  # [Nphi_i] (1 for isotropic)
    cos_theta_o: torch.Tensor  # [Nto]
    phi_o: torch.Tensor  # [Npo]
    # Co-pol power BSDF channels [Nti, Nphi_i, Nto, Npo].
    f_te: torch.Tensor
    f_tm: torch.Tensor
    # Diffuse reflection budgets per incidence bin [Nti, Nphi_i].
    r_diff_te: torch.Tensor
    r_diff_tm: torch.Tensor
    r_diff_unpol: torch.Tensor
    # Symmetric matrix-balance factor a(w) [Nti, Nphi_i, 2], channel order
    # TE/TM.  The final table is f(wi,wo)=a(wi)*f_sym(wi,wo)*a(wo), which
    # preserves reciprocity while enforcing every row's energy budget.
    normalization_applied: torch.Tensor
    # Sampling tables built from the UNPOLARIZED mean lobe:
    # per-bin probability mass and the per-solid-angle density (mass/dOmega),
    # marginal CDF over cos_theta_o, conditional CDF over phi_o.
    sample_density: torch.Tensor  # [Nti, Nphi_i, Nto, Npo]
    marginal_cdf: torch.Tensor  # [Nti, Nphi_i, Nto]
    conditional_cdf: torch.Tensor  # [Nti, Nphi_i, Nto, Npo]
    # Domain metadata / validity flags.
    frequency_hz: float
    k0: float
    sigma_h_m: float
    corr_x_m: float
    corr_y_m: float
    principal_axis_rad: float
    anisotropic: bool
    k0_l_min: float
    rms_slope_max: float
    tangent_plane_ok: bool
    slope_ok: bool
    reciprocity_error: float
    # ADR-015 Part C differentiable-build intermediates. The float64 numpy
    # build is unchanged bit-for-bit; these are the exact f32 downcasts of the
    # structural quantities the native table-build adjoint recomputes against
    # (no f32 recompute drift). ``pre_balance_lobe_*`` are the reciprocity-
    # symmetrized raw lobes ``S`` BEFORE the diagonal energy balance
    # (``F = a S a``); the balance factors ``a`` and the diffuse budgets are
    # already exposed as ``normalization_applied`` and ``r_diff_te``/
    # ``r_diff_tm``. All default to ``None`` so a table built without the AD
    # path (e.g. a bare numpy import) carries no extra state.
    pre_balance_lobe_te: torch.Tensor | None = None  # [Nti, Npi, Nto, Npo]
    pre_balance_lobe_tm: torch.Tensor | None = None  # [Nti, Npi, Nto, Npo]

    @property
    def device(self) -> torch.device:
        return self.f_te.device

    @property
    def bin_solid_angle(self) -> float:
        """Solid angle of one outgoing bin: d(cos_theta) * d(phi)."""

        return (1.0 / N_COS_THETA_O) * (2.0 * math.pi / N_PHI_O)


def _cos_centers(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) + 0.5) / n


def _phi_centers(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) + 0.5) * (2.0 * np.pi / n)


def _stack_power_reflectances(
    layers: Sequence[tuple], cos_theta: np.ndarray, frequency_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(|r_te_stack|^2, |r_tm_stack|^2)`` at real incidence cosines."""

    rt = layer_stack_rt(layers, cos_theta, frequency_hz)
    return np.abs(rt.r_te) ** 2, np.abs(rt.r_tm) ** 2


def _raw_lobe_grid(
    layers: Sequence[tuple],
    frequency_hz: float,
    k0: float,
    sigma_h: float,
    lx: float,
    ly: float,
    n_terms: int,
    inc_cos: np.ndarray,
    inc_phi: np.ndarray,
    out_cos: np.ndarray,
    out_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw (un-normalized) TE/TM lobe on an (incidence x outgoing) grid.

    Returns arrays ``[len(inc_cos), len(inc_phi), len(out_cos),
    len(out_phi)]``. Used twice per build: once on the table axes and once
    with incidence/outgoing axes swapped (the exact grid-resample of
    ``f(wo, wi)``  -  the analytic kernel is reciprocal, so re-evaluating at
    the swapped nodes is the lossless way to symmetrize; interpolating the
    coarse ``phi_i`` axis would smear the specular peak instead).
    """

    sin_inc = np.sqrt(np.maximum(0.0, 1.0 - inc_cos**2))
    sin_out = np.sqrt(np.maximum(0.0, 1.0 - out_cos**2))
    wo_x = sin_out[:, None] * np.cos(out_phi)[None, :]
    wo_y = sin_out[:, None] * np.sin(out_phi)[None, :]
    wo_z = np.broadcast_to(out_cos[:, None], wo_x.shape)
    f_te = np.empty((inc_cos.shape[0], inc_phi.shape[0], *wo_x.shape))
    f_tm = np.empty_like(f_te)
    for ti in range(inc_cos.shape[0]):
        for pi_idx in range(inc_phi.shape[0]):
            wi_x = sin_inc[ti] * np.cos(inc_phi[pi_idx])
            wi_y = sin_inc[ti] * np.sin(inc_phi[pi_idx])
            wi_z = inc_cos[ti]
            # q = k_s - k_i = k0*(wo + wi); components already in the
            # roughness principal frame (the table's local frame).
            qx = k0 * (wo_x + wi_x)
            qy = k0 * (wo_y + wi_y)
            qn = k0 * (wo_z + wi_z)
            lobe = _kirchhoff_diffuse_lobe_series(
                qx, qy, qn, sigma_h, lx, ly, n_terms=n_terms
            )
            # Specular-equivalent local incidence: cos_theta_h = wi.h with
            # h || (wo + wi) (module docstring); |wo + wi| = |q|/k0.
            q_sq = qx**2 + qy**2 + qn**2
            wi_dot_wo = wo_x * wi_x + wo_y * wi_y + wo_z * wi_z
            cos_h = np.clip((1.0 + wi_dot_wo) * k0 / np.sqrt(q_sq), 1e-6, 1.0)
            rr_te, rr_tm = _stack_power_reflectances(
                layers, cos_h.reshape(-1), frequency_hz
            )
            # Standard Kirchhoff geometry prefactor (see module docstring):
            # |q|^4 / (16*pi^2*q_n^2*cos_i*cos_o); reduces to k0^2/(4*pi^2)
            # at the specular direction and is symmetric under wi <-> wo.
            prefactor = q_sq**2 / (16.0 * np.pi**2 * qn**2 * wi_z * wo_z)
            shape = prefactor * lobe
            f_te[ti, pi_idx] = shape * rr_te.reshape(shape.shape)
            f_tm[ti, pi_idx] = shape * rr_tm.reshape(shape.shape)
    return f_te, f_tm


def _symmetric_energy_balance(
    symmetric_lobe: np.ndarray,
    target: np.ndarray,
    cos_o: np.ndarray,
    *,
    isotropic: bool,
    tolerance: float = 1e-11,
    max_iterations: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance a reciprocal nonnegative kernel without breaking symmetry.

    Finds positive diagonal factors ``a`` such that ``F_ij=a_i*S_ij*a_j``
    and every cosine-weighted row integral of ``F`` equals ``target_i``.
    This is the symmetric Sinkhorn fixed point with a half-step damping;
    unlike one-sided row normalization, reciprocity is invariant at every
    iteration.
    """

    n_ti, n_pi, n_to, n_po = symmetric_lobe.shape
    if n_ti != n_to:
        raise ValueError("reciprocal balance requires identical cos grids")
    d_omega = (1.0 / n_to) * (2.0 * np.pi / n_po)
    outgoing_weight = np.broadcast_to(
        cos_o[:, None] * d_omega, (n_to, n_po)
    ).reshape(-1)

    if isotropic:
        # Rotation invariance collapses the incident azimuth state.  The
        # reverse state factor depends only on cos(theta_o), while the full
        # relative-azimuth lobe remains in the row integral.
        s = symmetric_lobe[:, 0]
        rhs = target[:, 0]
        factor = np.ones(n_ti, dtype=np.float64)
        active = rhs > 0.0
        for _ in range(max_iterations):
            weighted = outgoing_weight * np.repeat(factor, n_po)
            denom = (s * weighted[None, :, None].reshape(1, n_to, n_po)).sum(
                axis=(1, 2)
            )
            ratio = np.ones_like(factor)
            ratio[active] = rhs[active] / np.maximum(
                factor[active] * denom[active], 1e-300
            )
            factor *= np.sqrt(ratio)
            factor[~active] = 0.0
            if np.max(np.abs(np.log(np.maximum(ratio[active], 1e-300))), initial=0.0) < tolerance:
                break
        else:
            raise ValueError("symmetric Kirchhoff energy balance did not converge")
        balanced = s * factor[:, None, None] * factor[None, :, None]
        return balanced[:, None], factor[:, None]

    if n_pi != n_po:
        raise ValueError("anisotropic reciprocal balance requires identical phi grids")
    states = n_ti * n_pi
    s = symmetric_lobe.reshape(states, states)
    rhs = target.reshape(states)
    weights = np.repeat(cos_o * d_omega, n_po)
    factor = np.ones(states, dtype=np.float64)
    active = rhs > 0.0
    for _ in range(max_iterations):
        denom = s @ (weights * factor)
        ratio = np.ones_like(factor)
        ratio[active] = rhs[active] / np.maximum(
            factor[active] * denom[active], 1e-300
        )
        factor *= np.sqrt(ratio)
        factor[~active] = 0.0
        if np.max(np.abs(np.log(np.maximum(ratio[active], 1e-300))), initial=0.0) < tolerance:
            break
    else:
        raise ValueError("symmetric Kirchhoff energy balance did not converge")
    balanced = s * factor[:, None] * factor[None, :]
    return balanced.reshape(symmetric_lobe.shape), factor.reshape(n_ti, n_pi)


def build_kirchhoff_table(
    roughness,
    layers: Sequence[tuple],
    frequency_hz: float,
    device: torch.device | str = "cuda",
) -> KirchhoffTable:
    """Precompute the Kirchhoff ensemble BSDF table for one material.

    ``roughness`` is a :class:`witwin.core.SurfaceRoughness`
    (or any object with the same fields); ``layers`` is the oracle layer
    list ``[(thickness_m, eps_r, sigma_e, mu_r), ...]`` in incidence order.
    Raises when the surface is outside the Kirchhoff applicability domain
    (``kirchhoff_domain_exceeded``) or reciprocal energy balancing fails to
    converge.
    """

    frequency_hz = float(frequency_hz)
    sigma_h = float(roughness.rms_height_m)
    lx = float(roughness.correlation_length_x_m)
    ly = float(roughness.correlation_length_y_m)
    axis_rad = float(roughness.principal_axis_rad)
    k0 = 2.0 * math.pi * frequency_hz / C0

    l_min = min(lx, ly)
    k0_l_min = k0 * l_min
    rms_slope_max = math.sqrt(2.0) * sigma_h / l_min
    tangent_plane_ok = k0_l_min >= MIN_K0_CORR_LENGTH
    slope_ok = rms_slope_max <= MAX_RMS_SLOPE
    if not (tangent_plane_ok and slope_ok):
        raise ValueError(
            "kirchhoff_domain_exceeded: ensemble Kirchhoff requires "
            f"k0*corr_length >= {MIN_K0_CORR_LENGTH:g} "
            f"(got {k0_l_min:.3g}) and RMS slope sqrt(2)*sigma_h/l <= "
            f"{MAX_RMS_SLOPE:g} (got {rms_slope_max:.3g})"
        )

    anisotropic = lx != ly
    cos_i = _cos_centers(N_COS_THETA_I)
    phi_i = _phi_centers(N_PHI_I_ANISO) if anisotropic else np.zeros(1)
    cos_o = _cos_centers(N_COS_THETA_O)
    phi_o = _phi_centers(N_PHI_O)
    n_pi = phi_i.shape[0]

    # 1) Smooth-stack budgets on the incidence grid (complex128 oracle).
    r_bar_te, r_bar_tm = _stack_power_reflectances(layers, cos_i, frequency_hz)
    # 2) Coherent part: |r_stack * C_r|^2 = R_bar * C_r^2.
    c_r = np.exp(-2.0 * (k0 * cos_i * sigma_h) ** 2)
    r_coh_te = r_bar_te * c_r**2
    r_coh_tm = r_bar_tm * c_r**2
    # 3) Diffuse budgets (>= 0 since C_r <= 1; max() guards fp rounding).
    r_diff_te = np.maximum(0.0, r_bar_te - r_coh_te)
    r_diff_tm = np.maximum(0.0, r_bar_tm - r_coh_tm)

    # 4) Raw lobe on the 4D grid.
    g_max = (2.0 * k0 * sigma_h) ** 2
    n_terms = int(max(64, g_max + 12.0 * math.sqrt(g_max) + 16.0))
    f_raw_te, f_raw_tm = _raw_lobe_grid(
        layers, frequency_hz, k0, sigma_h, lx, ly, n_terms,
        cos_i, phi_i, cos_o, phi_o,
    )

    # 5) Reciprocity symmetrization BEFORE normalization: f(wo, wi) is
    # obtained by exact re-evaluation on the swapped grid nodes (see
    # _raw_lobe_grid), then averaged with the forward evaluation. The
    # residual measures genuine wi<->wo asymmetry of the implementation
    # (the kernel is analytically reciprocal), so anything above float
    # rounding is a bug.
    swap_te, swap_tm = _raw_lobe_grid(
        layers, frequency_hz, k0, sigma_h, lx, ly, n_terms,
        cos_o, phi_o, cos_i, phi_i,
    )
    swap_te = np.transpose(swap_te, (2, 3, 0, 1))
    swap_tm = np.transpose(swap_tm, (2, 3, 0, 1))
    reciprocity_error = 0.0
    for forward, swapped in ((f_raw_te, swap_te), (f_raw_tm, swap_tm)):
        peak = float(forward.max())
        if peak > 0.0:
            err = float(np.abs(forward - swapped).max() / peak)
            reciprocity_error = max(reciprocity_error, err)
    if reciprocity_error >= 1e-3:
        raise ValueError(
            "kirchhoff table reciprocity error "
            f"{reciprocity_error:.3e} exceeds 1e-3 after symmetrization"
        )
    f_sym_te = 0.5 * (f_raw_te + swap_te)
    f_sym_tm = 0.5 * (f_raw_tm + swap_tm)

    # ADR-015 Part C: snapshot the pre-balance symmetrized lobes S before the
    # in-place diagonal energy balance below overwrites f_sym. The native
    # table-build adjoint consumes these (with a, r_diff) as its saved
    # intermediates; the numpy primal is unaffected.
    pre_balance_lobe_te = f_sym_te.copy()
    pre_balance_lobe_tm = f_sym_tm.copy()

    # 6) Symmetric energy balance on the discrete directional state matrix.
    # A one-sided row scale would make the energy exact but destroy
    # f(wi,wo)==f(wo,wi).  Diagonal scaling on both arguments preserves the
    # already-symmetric raw kernel and satisfies every row budget jointly.
    d_omega = (1.0 / N_COS_THETA_O) * (2.0 * np.pi / N_PHI_O)
    weight = cos_o[None, None, :, None] * d_omega  # broadcast over [.., to, po]
    r_diff_te_grid = np.broadcast_to(r_diff_te[:, None], (N_COS_THETA_I, n_pi)).copy()
    r_diff_tm_grid = np.broadcast_to(r_diff_tm[:, None], (N_COS_THETA_I, n_pi)).copy()
    scales = np.ones((N_COS_THETA_I, n_pi, 2))
    channels = (
        (0, f_sym_te, r_diff_te_grid),
        (1, f_sym_tm, r_diff_tm_grid),
    )
    for channel, f_sym, r_diff in channels:
        balanced, factor = _symmetric_energy_balance(
            f_sym, r_diff, cos_o, isotropic=not anisotropic
        )
        f_sym[...] = balanced
        integral = (f_sym * weight).sum(axis=(2, 3))
        active = r_diff > 0.0
        relative_error = np.zeros_like(integral)
        relative_error[active] = np.abs(integral[active] - r_diff[active]) / r_diff[active]
        if relative_error.max(initial=0.0) > 2e-9:
            raise ValueError(
                "symmetric Kirchhoff balance failed energy tolerance: "
                f"max relative error {relative_error.max():.3e}"
            )
        # Store the one-direction diagonal factor.  Unlike the obsolete
        # one-sided row scale, its magnitude alone is not a shape-error
        # metric: the physical correction on a pair is a(wi)*a(wo), and the
        # factors are jointly constrained by all directional energy rows.
        scales[:, :, channel] = factor

    # 7) Sampling tables from the UNPOLARIZED mean lobe.
    f_unpol = 0.5 * (f_sym_te + f_sym_tm)
    mass = f_unpol * weight  # [ti, pi, to, po] probability mass per bin
    total = mass.sum(axis=(2, 3), keepdims=True)
    uniform = np.full_like(mass, 1.0 / (N_COS_THETA_O * N_PHI_O))
    mass = np.where(total > 0.0, mass / np.where(total > 0.0, total, 1.0), uniform)
    density = mass / d_omega
    marginal = mass.sum(axis=3)  # [ti, pi, to]
    marginal_cdf = np.cumsum(marginal, axis=2)
    marginal_cdf /= marginal_cdf[..., -1:]
    cond = np.where(
        marginal[..., None] > 0.0,
        mass / np.where(marginal[..., None] > 0.0, marginal[..., None], 1.0),
        1.0 / N_PHI_O,
    )
    conditional_cdf = np.cumsum(cond, axis=3)
    conditional_cdf /= conditional_cdf[..., -1:]

    device = torch.device(device)

    def as32(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

    return KirchhoffTable(
        cos_theta_i=as32(cos_i),
        phi_i=as32(phi_i),
        cos_theta_o=as32(cos_o),
        phi_o=as32(phi_o),
        f_te=as32(f_sym_te),
        f_tm=as32(f_sym_tm),
        r_diff_te=as32(r_diff_te_grid),
        r_diff_tm=as32(r_diff_tm_grid),
        r_diff_unpol=as32(0.5 * (r_diff_te_grid + r_diff_tm_grid)),
        normalization_applied=as32(scales),
        sample_density=as32(density),
        marginal_cdf=as32(marginal_cdf),
        conditional_cdf=as32(conditional_cdf),
        frequency_hz=frequency_hz,
        k0=k0,
        sigma_h_m=sigma_h,
        corr_x_m=lx,
        corr_y_m=ly,
        principal_axis_rad=axis_rad,
        anisotropic=anisotropic,
        k0_l_min=k0_l_min,
        rms_slope_max=rms_slope_max,
        tangent_plane_ok=tangent_plane_ok,
        slope_ok=slope_ok,
        reciprocity_error=reciprocity_error,
        pre_balance_lobe_te=as32(pre_balance_lobe_te),
        pre_balance_lobe_tm=as32(pre_balance_lobe_tm),
    )


def eval_bsdf(
    table: KirchhoffTable,
    valid: torch.Tensor,
    wi: torch.Tensor,
    wo: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (multilinear) lookup of ``(f_te, f_tm)`` for batched pairs.

    ``wi``/``wo`` are [N, 3] local-frame unit vectors pointing away from the
    surface. Directions below the horizon return 0.
    """

    from witwin.channel.kernels.scattering import (
        scattering_table_eval,
    )

    return scattering_table_eval(
        valid.contiguous(), wi.contiguous(), wo.contiguous(), table.f_te, table.f_tm
    )


def sample_directions(
    table: KirchhoffTable,
    valid: torch.Tensor,
    wi: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample outgoing directions by CDF inversion; returns ``(wo, pdf)``.

    Uses the nearest incidence bin (the same convention :func:`pdf` uses, so
    the sampler and its density are exactly consistent). ``u1`` inverts the
    marginal CDF over ``cos_theta_o``; ``u2`` the conditional CDF over
    ``phi_o``; both are linearly remapped inside the selected bin, i.e. the
    sampled density is piecewise constant per outgoing bin.
    """

    from witwin.channel.kernels.scattering import (
        scattering_table_sample,
    )

    uniforms = torch.stack((u1, u2), dim=1).contiguous()
    out = scattering_table_sample(
        valid.contiguous(),
        wi.contiguous(),
        uniforms,
        table.marginal_cdf,
        table.conditional_cdf,
        table.sample_density,
    )
    return out["wo"], out["pdf_forward"]


def pdf(
    table: KirchhoffTable,
    valid: torch.Tensor,
    wi: torch.Tensor,
    wo: torch.Tensor,
) -> torch.Tensor:
    """Solid-angle sampling density of :func:`sample_directions`.

    Piecewise constant per outgoing bin and exactly consistent with the
    sampler (same nearest incidence bin, same mass table); integrates to 1
    over the hemisphere by construction. Zero below the horizon.
    """

    from witwin.channel.kernels.scattering import scattering_table_pdf

    return scattering_table_pdf(
        valid.contiguous(),
        wi.contiguous(),
        wo.contiguous(),
        table.sample_density,
        reverse=False,
    )


def pdf_reverse(
    table: KirchhoffTable,
    valid: torch.Tensor,
    wo: torch.Tensor,
    wi: torch.Tensor,
) -> torch.Tensor:
    """Reverse-direction PDF: the SAME table evaluated with swapped args.

    BDPT evaluates the reverse strategy density by treating the outgoing
    direction as the incidence direction (contract section 5); no separate
    reverse table exists.
    """

    from witwin.channel.kernels.scattering import scattering_table_pdf

    return scattering_table_pdf(
        valid.contiguous(),
        wi.contiguous(),
        wo.contiguous(),
        table.sample_density,
        reverse=True,
    )


class PhaseScreenRuntime:
    """GPU height texture + phasor sampling for one surface realization."""

    def __init__(self, screen: PhaseScreen, device: torch.device | str = "cuda"):
        if not isinstance(screen, PhaseScreen):
            raise ValueError("screen must be a PhaseScreen")
        device = torch.device(device)
        # Metric heights [m]: stored texture * scale + offset, float32.
        heights = screen.height.to(device=device, dtype=torch.float32)
        self.heights_m = (
            heights * float(screen.height_scale_m) + float(screen.height_offset_m)
        ).contiguous()
        self.screen = screen
        self.device = device

    def sample_height(self, uv: torch.Tensor) -> torch.Tensor:
        """Bilinear metric height [m] at ``uv`` ([..., 2], u right, v down).

        Manual gather with edge clamp (see module docstring for the texel
        convention).
        """

        if uv.shape[-1] != 2:
            raise ValueError("uv must have trailing dimension 2")
        h_rows, w_cols = self.heights_m.shape
        uv = uv.to(device=self.device, dtype=torch.float32)
        # Edge clamp: clamp the CONTINUOUS texel coordinate into the span of
        # texel centers before flooring, so UV outside the half-texel margin
        # reproduces the border texel exactly (no border interpolation).
        tx = (uv[..., 0] * w_cols - 0.5).clamp(0.0, float(w_cols - 1))
        ty = (uv[..., 1] * h_rows - 0.5).clamp(0.0, float(h_rows - 1))
        x0 = torch.floor(tx)
        y0 = torch.floor(ty)
        wx = tx - x0
        wy = ty - y0
        ix0 = x0.long()
        iy0 = y0.long()
        ix1 = (ix0 + 1).clamp(max=w_cols - 1)
        iy1 = (iy0 + 1).clamp(max=h_rows - 1)
        flat = self.heights_m.reshape(-1)

        def tex(iy: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
            return flat[iy * w_cols + ix]

        top = tex(iy0, ix0) * (1.0 - wx) + tex(iy0, ix1) * wx
        bot = tex(iy1, ix0) * (1.0 - wx) + tex(iy1, ix1) * wx
        return top * (1.0 - wy) + bot * wy

    def phasor(self, uv: torch.Tensor, q_n: torch.Tensor | float) -> torch.Tensor:
        """Complex64 phase-screen factor ``exp(-j*q_n*h(u, v))``."""

        h = self.sample_height(uv)
        if not isinstance(q_n, torch.Tensor):
            q_n = torch.as_tensor(q_n, dtype=torch.float32, device=self.device)
        phase = -(q_n * h)
        return torch.polar(torch.ones_like(phase), phase).to(torch.complex64)


def realization_seed(scene_seed: int, surface_id: int, realization_id: int) -> int:
    """Deterministic 64-bit seed for ``(scene_seed, surface_id, realization_id)``.

    SplitMix64-style avalanche over the packed inputs so nearby ids give
    decorrelated seeds while the mapping stays reproducible across runs and
    platforms (pure integer arithmetic, no RNG state involved).
    """

    mask = (1 << 64) - 1
    z = (int(scene_seed) & mask)
    for salt in (int(surface_id), int(realization_id)):
        z = (z + 0x9E3779B97F4A7C15 + (salt & mask)) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z = z ^ (z >> 31)
    return z


def generate_gaussian_realization(
    roughness: SurfaceRoughness,
    extent_m: tuple[float, float] | float,
    resolution: tuple[int, int] | int,
    seed: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Periodic Gaussian random height field with Gaussian correlation.

    FFT spectral synthesis of the section-6.1 correlation
    ``C(x, y) = sigma_h^2 * exp(-(x/lx)^2 - (y/ly)^2)`` whose PSD (plan
    section 6.3 Fourier convention, ``(1/(2*pi)^2) Int W d^2q = sigma_h^2``)
    is ``W(qx, qy) = pi*sigma_h^2*lx*ly*exp(-(qx^2*lx^2 + qy^2*ly^2)/4)``.

    Normalization: on a periodic domain ``Lx x Ly`` each Fourier mode gets
    variance ``E[|h_k|^2] = W(q_k)/(Lx*Ly)`` (the Riemann sum of the PSD
    integral over the discrete mode lattice ``dq = 2*pi/L``). Modes are
    drawn as independent circular complex Gaussians and the real part is
    taken, which halves per-mode variance, so amplitudes carry a
    compensating ``sqrt(2)``. Total variance then approximates ``sigma_h^2``
    up to spectral truncation at Nyquist: for ``dx << l`` and ``L >> l`` the
    sampled RMS height matches ``sigma_h`` within a few percent.

    ``extent_m``/``resolution`` are ``(x, y)`` pairs (scalars broadcast).
    Returns float32 heights [m] of shape ``(ny, nx)`` on ``device``
    (row = v/y axis, matching :class:`PhaseScreenRuntime`).
    """

    sigma_h = float(roughness.rms_height_m)
    lx = float(roughness.correlation_length_x_m)
    ly = float(roughness.correlation_length_y_m)
    if isinstance(extent_m, (int, float)):
        extent_m = (float(extent_m), float(extent_m))
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    length_x, length_y = float(extent_m[0]), float(extent_m[1])
    nx, ny = int(resolution[0]), int(resolution[1])
    if length_x <= 0.0 or length_y <= 0.0 or nx < 2 or ny < 2:
        raise ValueError("extent_m must be positive and resolution >= 2")

    qx = 2.0 * np.pi * np.fft.fftfreq(nx, d=length_x / nx)
    qy = 2.0 * np.pi * np.fft.fftfreq(ny, d=length_y / ny)
    psd = (
        np.pi
        * sigma_h**2
        * lx
        * ly
        * np.exp(-((qx[None, :] * lx) ** 2 + (qy[:, None] * ly) ** 2) / 4.0)
    )
    amplitude = np.sqrt(2.0 * psd / (length_x * length_y))
    rng = np.random.default_rng(int(seed) & ((1 << 64) - 1))
    xi = (rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))) / math.sqrt(2.0)
    # h(x_m) = Re sum_k A_k xi_k exp(+j q_k . x_m); numpy ifft2 divides by
    # nx*ny, so multiply it back to get the plain Fourier-series sum.
    field = np.fft.ifft2(amplitude * xi) * (nx * ny)
    heights = np.real(field)
    return torch.from_numpy(np.ascontiguousarray(heights, dtype=np.float32)).to(
        torch.device(device)
    )


def patch_phase_integral(
    runtime: PhaseScreenRuntime,
    patch_vertices: torch.Tensor,
    uv_vertices: torch.Tensor,
    k_i_vec: torch.Tensor,
    k_s_vec: torch.Tensor,
    frequency_hz: float,
    n_quad: int = 16,
) -> torch.Tensor:
    """Triangle-domain quadrature of the Kirchhoff phase integral (GPU).

    Evaluates ``sum_T Int_T exp(-j*(k_s - k_i).x) * exp(-j*q_n*h(u, v)) dA``
    over triangles ``patch_vertices`` [T, 3, 3] with matching UVs
    ``uv_vertices`` [T, 3, 2] (a single triangle may omit the leading dim).
    Positions stay on the mean plane; heights enter only the phase, matching
    ``oracle.phase_screen_patch_integral`` for the same height field.

    Quadrature: Duffy-mapped tensor-product Gauss-Legendre with ``n_quad``
    points per axis  -  the unit square ``(xi, eta)`` maps to barycentric
    ``(a, b) = (xi, eta*(1 - xi))`` with Jacobian ``(1 - xi)``, so
    refinement in ``n_quad`` converges to the exact triangle integral.
    Returns a 0-dim complex64 tensor on the runtime device.
    """

    device = runtime.device
    tri = patch_vertices.to(device=device, dtype=torch.float32)
    uv = uv_vertices.to(device=device, dtype=torch.float32)
    if tri.ndim == 2:
        tri = tri.unsqueeze(0)
    if uv.ndim == 2:
        uv = uv.unsqueeze(0)
    if tri.shape[1:] != (3, 3) or uv.shape[1:] != (3, 2) or tri.shape[0] != uv.shape[0]:
        raise ValueError("patch_vertices must be [T, 3, 3] and uv_vertices [T, 3, 2]")
    k_i = k_i_vec.to(device=device, dtype=torch.float32).reshape(3)
    k_s = k_s_vec.to(device=device, dtype=torch.float32).reshape(3)
    k0 = 2.0 * math.pi * float(frequency_hz) / C0
    for name, vec in (("k_i_vec", k_i), ("k_s_vec", k_s)):
        if abs(float(torch.linalg.vector_norm(vec)) - k0) > 1e-5 * k0:
            raise ValueError(f"|{name}| does not match 2*pi*frequency_hz/c0")
    q = k_s - k_i

    # Per-triangle mean-plane frame: edges, normal, area.
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    normal = torch.cross(e1, e2, dim=1)
    double_area = torch.linalg.vector_norm(normal, dim=1)
    n_hat = normal / double_area.unsqueeze(1).clamp(min=1e-30)
    q_n = n_hat @ q  # [T] normal wavenumber transfer per triangle

    # Duffy-mapped Gauss points on the unit square (float64 nodes -> f32).
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    xi = torch.from_numpy(0.5 * (nodes + 1.0)).to(device=device, dtype=torch.float32)
    w1 = torch.from_numpy(0.5 * weights).to(device=device, dtype=torch.float32)
    a = xi[:, None].expand(n_quad, n_quad)  # barycentric a = xi
    b = xi[None, :] * (1.0 - xi[:, None])  # barycentric b = eta*(1 - xi)
    w2d = (w1[:, None] * w1[None, :]) * (1.0 - xi[:, None])  # Duffy Jacobian
    a = a.reshape(-1)
    b = b.reshape(-1)
    w2d = w2d.reshape(-1)

    # Quadrature positions/UVs: x = p0 + a*e1 + b*e2 (same barycentric
    # interpolation for UV), batched [T, Q, ...].
    pos = (
        tri[:, 0, None, :]
        + a[None, :, None] * e1[:, None, :]
        + b[None, :, None] * e2[:, None, :]
    )
    uv_pts = (
        uv[:, 0, None, :]
        + a[None, :, None] * (uv[:, 1] - uv[:, 0])[:, None, :]
        + b[None, :, None] * (uv[:, 2] - uv[:, 0])[:, None, :]
    )
    heights = runtime.sample_height(uv_pts)  # [T, Q]
    phase = pos @ q + q_n[:, None] * heights
    phasor = torch.polar(torch.ones_like(phase), -phase)
    # dA = 2*Area * da db; the simplex measure is carried by w2d.
    contrib = (phasor * w2d[None, :]).sum(dim=1) * double_area.to(phasor.real.dtype)
    return contrib.sum().to(torch.complex64)