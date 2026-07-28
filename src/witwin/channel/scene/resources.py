"""Immutable native resources a compiled scene owns.

This module is the single owner of everything a :class:`CompiledScene` holds
that is not a store: the typed RayD scene/BVH lifetime, the diffraction edge
policy and the scene-policy refinement of the exported edge geometry, and the
lazily built Kirchhoff and phase-screen resources.

They were four modules that only ever called each other. The RayD resource is
the thing the edge refinement reads its records from and the thing the
phase-screen build takes its mesh tensors from, and the edge policy exists only
to be cached on that resource at compile time, so a reader following one
resource had to walk four files to see one lifetime.

Nothing here computes RF physics. The RayD facade validates a contract and
dispatches a native symbol; the edge refinement is scene-policy row selection
over already-exported geometry; the resource builders cache scene-static data
that the consumers used to recompute per solve, with their exception and
numerical order preserved. Moving that retained static construction across the
native boundary requires its own accepted ADR.

The stores this module annotates against live in
:mod:`witwin.channel.scene.compiler`, which imports this module for the RayD
lifetime. The store names are needed for typing only, so they are imported
under ``TYPE_CHECKING`` and the runtime dependency stays one-way.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING

import torch

from witwin.core import PhaseScreen, SurfaceRoughness

from witwin.channel.runtime import mc_pack_vec3, required_symbol as _required_native_op
from witwin.channel.scattering import (
    MAX_RMS_SLOPE,
    KirchhoffTable,
    PhaseScreenRuntime,
)

if TYPE_CHECKING:
    from witwin.channel.scene.compiler import AssignmentStore, MaterialStore

__all__ = [
    "DEFAULT_EDGE_POLICY",
    "EdgePolicy",
    "KirchhoffRuntimeResources",
    "KirchhoffTableStack",
    "PhaseScreenResourceKey",
    "PhaseScreenRuntimeResources",
    "PhaseScreenStructureResource",
    "RayDEdgeRecords",
    "RayDSceneResource",
    "RoughMaterialRuntime",
    "ScatteringResourceKey",
    "_empty_tensor",
    "_mesh_flags",
    "build_kirchhoff_resources",
    "build_kirchhoff_table_stack",
    "build_phase_screen_resources",
    "build_scene_from_structures",
    "rayd_scene_create",
    "rayd_scene_edge_records",
    "realization_phase_screens",
    "refine_edge_geometry",
    "resolve_scene_edge_policy",
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
    policy = getattr(scene, "metadata", {}).get("sionna_import_edge_policy")
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

    # Keep the supported package-level instrumentation seam lazy.
    from witwin.channel import scattering

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

    from witwin.channel import scattering

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
