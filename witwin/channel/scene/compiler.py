# Copyright Xingyu Chen.
# Compile canonical Core world contracts into Channel runtime resources.

"""Compile canonical Core world contracts into Channel runtime resources."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
import hashlib
import math
from threading import RLock
from typing import TypeVar
import weakref

import numpy as np
import torch

from witwin.core import (
    MaterialLayer,
    Mesh,
    PhaseScreen,
    PhysicalMaterial,
    Scene,
    SceneSnapshot,
    Structure,
    quat_from_euler,
    quat_to_rotation_matrix,
)
from witwin.core.material import VACUUM_PERMITTIVITY

from witwin.channel.materials import (
    DIELECTRIC_MODEL_ID,
    MATERIAL_ABI_VERSION,
    PEC_EFFECTIVE_SIGMA_E,
    PEC_MODEL_ID,
)
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel.runtime import (
    bdpt_zero_matrix,
    mc_receiver_grid_points,
    mc_transmitter_tensors,
    requested_trace_backend,
)
from witwin.channel.scene.endpoints import (
    ReceiverGrid,
    ReceiverPoint,
    SolverScene,
    vector3_tuple,
)
from witwin.channel.scene.resources import (
    KirchhoffRuntimeResources,
    KirchhoffTable,
    PhaseScreenResourceKey,
    PhaseScreenRuntimeResources,
    RayDSceneResource,
    RoughMaterialRuntime,
    ScatteringResourceKey,
    build_kirchhoff_resources,
    build_phase_screen_resources,
    build_scene_from_structures,
)


def require_tensor(
    name: str, tensor: torch.Tensor, *, dtype: torch.dtype, ndim: int,
    trailing_shape: tuple[int, ...] = (),
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if trailing_shape and tuple(tensor.shape[-len(trailing_shape) :]) != trailing_shape:
        raise ValueError(f"{name} must end with shape {trailing_shape}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@dataclass(frozen=True, slots=True)
class GeometryStore:
    vertices: torch.Tensor
    faces: torch.Tensor
    face_normals: torch.Tensor
    edges: torch.Tensor
    edge_adj_faces: torch.Tensor
    edge_param_range: torch.Tensor
    face_structure_id: torch.Tensor
    face_surface_id: torch.Tensor
    face_primitive_id: torch.Tensor
    structure_uv_presence: tuple[tuple[bool, bool], ...]
    version: int

    def __post_init__(self) -> None:
        require_tensor("vertices", self.vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,))
        require_tensor("faces", self.faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
        require_tensor(
            "face_normals", self.face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        require_tensor("edges", self.edges, dtype=torch.int32, ndim=2, trailing_shape=(2,))
        require_tensor(
            "edge_adj_faces", self.edge_adj_faces, dtype=torch.int32, ndim=2, trailing_shape=(2,)
        )
        require_tensor(
            "edge_param_range",
            self.edge_param_range,
            dtype=torch.float32,
            ndim=2,
            trailing_shape=(2,),
        )
        require_tensor("face_structure_id", self.face_structure_id, dtype=torch.int64, ndim=1)
        require_tensor("face_surface_id", self.face_surface_id, dtype=torch.int64, ndim=1)
        require_tensor("face_primitive_id", self.face_primitive_id, dtype=torch.int64, ndim=1)
        if self.faces.shape[0] != self.face_normals.shape[0]:
            raise ValueError("face_normals length must match faces")
        if self.faces.shape[0] != self.face_structure_id.shape[0]:
            raise ValueError("face_structure_id length must match faces")
        if self.faces.shape[0] != self.face_surface_id.shape[0]:
            raise ValueError("face_surface_id length must match faces")
        if self.faces.shape[0] != self.face_primitive_id.shape[0]:
            raise ValueError("face_primitive_id length must match faces")
        if self.edges.shape[0] != self.edge_adj_faces.shape[0]:
            raise ValueError("edge_adj_faces length must match edges")
        if self.edges.shape[0] != self.edge_param_range.shape[0]:
            raise ValueError("edge_param_range length must match edges")
        if any(
            type(uv_present) is not bool or type(face_uv_present) is not bool
            for uv_present, face_uv_present in self.structure_uv_presence
        ):
            raise ValueError("structure_uv_presence entries must contain bool values")


@dataclass(frozen=True, slots=True)
class MaterialStore:
    material_id: torch.Tensor
    eps_r: torch.Tensor
    mu_r: torch.Tensor
    sigma_e: torch.Tensor
    gain: torch.Tensor
    model_id: torch.Tensor
    thickness_m: torch.Tensor
    scattering_coefficient: torch.Tensor
    xpd_coefficient: torch.Tensor
    # ABI v3: flat CSR layer stack over all M materials (L total layers).
    layer_offset: torch.Tensor
    layer_count: torch.Tensor
    layer_thickness_m: torch.Tensor
    layer_eps_r: torch.Tensor
    layer_sigma_e: torch.Tensor
    layer_mu_r: torch.Tensor
    # ABI v3: front-surface roughness statistics (sigma_h == 0 means smooth).
    rough_sigma_h_m: torch.Tensor
    rough_corr_x_m: torch.Tensor
    rough_corr_y_m: torch.Tensor
    rough_axis_rad: torch.Tensor
    # ABI v3: 0=thin_sheet, 1=closed_volume / 0=smooth, 1=kirchhoff_ensemble.
    geometry_mode_id: torch.Tensor
    scatter_model_id: torch.Tensor
    material_keys: tuple[str, ...]
    frequency_hz: float
    abi_version: int
    cache_token: str
    version: int
    # Keys of records whose material law changes with the carrier frequency
    # (records are frozen at the primal frequency at compile time). Consumed
    # by the material and frequency derivatives explicit-failure check for frequency AD.
    frequency_dependent: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_tensor("material_id", self.material_id, dtype=torch.int64, ndim=1)
        require_tensor("eps_r", self.eps_r, dtype=torch.float32, ndim=1)
        require_tensor("mu_r", self.mu_r, dtype=torch.float32, ndim=1)
        require_tensor("sigma_e", self.sigma_e, dtype=torch.float32, ndim=1)
        require_tensor("gain", self.gain, dtype=torch.float32, ndim=1)
        require_tensor("model_id", self.model_id, dtype=torch.int32, ndim=1)
        require_tensor("thickness_m", self.thickness_m, dtype=torch.float32, ndim=1)
        require_tensor(
            "scattering_coefficient",
            self.scattering_coefficient,
            dtype=torch.float32,
            ndim=1,
        )
        require_tensor(
            "xpd_coefficient", self.xpd_coefficient, dtype=torch.float32, ndim=1
        )
        require_tensor("layer_offset", self.layer_offset, dtype=torch.int32, ndim=1)
        require_tensor("layer_count", self.layer_count, dtype=torch.int32, ndim=1)
        require_tensor(
            "layer_thickness_m", self.layer_thickness_m, dtype=torch.float32, ndim=1
        )
        require_tensor("layer_eps_r", self.layer_eps_r, dtype=torch.float32, ndim=1)
        require_tensor("layer_sigma_e", self.layer_sigma_e, dtype=torch.float32, ndim=1)
        require_tensor("layer_mu_r", self.layer_mu_r, dtype=torch.float32, ndim=1)
        require_tensor(
            "rough_sigma_h_m", self.rough_sigma_h_m, dtype=torch.float32, ndim=1
        )
        require_tensor(
            "rough_corr_x_m", self.rough_corr_x_m, dtype=torch.float32, ndim=1
        )
        require_tensor(
            "rough_corr_y_m", self.rough_corr_y_m, dtype=torch.float32, ndim=1
        )
        require_tensor(
            "rough_axis_rad", self.rough_axis_rad, dtype=torch.float32, ndim=1
        )
        require_tensor(
            "geometry_mode_id", self.geometry_mode_id, dtype=torch.int32, ndim=1
        )
        require_tensor(
            "scatter_model_id", self.scatter_model_id, dtype=torch.int32, ndim=1
        )
        lengths = {
            self.material_id.shape[0],
            self.eps_r.shape[0],
            self.mu_r.shape[0],
            self.sigma_e.shape[0],
            self.gain.shape[0],
            self.model_id.shape[0],
            self.thickness_m.shape[0],
            self.scattering_coefficient.shape[0],
            self.xpd_coefficient.shape[0],
            self.layer_offset.shape[0],
            self.layer_count.shape[0],
            self.rough_sigma_h_m.shape[0],
            self.rough_corr_x_m.shape[0],
            self.rough_corr_y_m.shape[0],
            self.rough_axis_rad.shape[0],
            self.geometry_mode_id.shape[0],
            self.scatter_model_id.shape[0],
            len(self.material_keys),
        }
        if len(lengths) != 1:
            raise ValueError("material tensors must have the same length")
        layer_lengths = {
            self.layer_thickness_m.shape[0],
            self.layer_eps_r.shape[0],
            self.layer_sigma_e.shape[0],
            self.layer_mu_r.shape[0],
        }
        if len(layer_lengths) != 1:
            raise ValueError("layer tensors must have the same length")
        total_layers = self.layer_thickness_m.shape[0]
        counts = self.layer_count.to(dtype=torch.int64)
        offsets = self.layer_offset.to(dtype=torch.int64)
        if self.layer_count.numel():
            if bool((counts < 1).any()):
                raise ValueError("layer_count must be >= 1 for every material")
            expected_offsets = torch.cumsum(counts, dim=0) - counts
            if not torch.equal(offsets, expected_offsets):
                raise ValueError("layer_offset must be the exclusive scan of layer_count")
        if int(counts.sum()) != total_layers:
            raise ValueError("layer_count must sum to the layer tensor length")
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if self.abi_version != 3:
            raise ValueError("MaterialStore requires material ABI version 3")
        if self.material_id.numel() and bool((self.material_id < 0).any()):
            raise ValueError("material_id must be non-negative")
        if torch.unique(self.material_id).numel() != self.material_id.numel():
            raise ValueError("material_id must be unique")
        if len(set(self.material_keys)) != len(self.material_keys):
            raise ValueError("material_keys must be unique")


@dataclass(frozen=True, slots=True)
class AssignmentStore:
    assignment_id: torch.Tensor
    material_id: torch.Tensor
    structure_id: torch.Tensor
    surface_id: torch.Tensor
    face_material_id: torch.Tensor
    edge_material_id0: torch.Tensor
    edge_material_id1: torch.Tensor
    surface_material_id: torch.Tensor
    structure_material_id: torch.Tensor
    num_faces: int
    num_edges: int
    version: int
    # Per-structure phase-screen bindings from the Core Structure assignment.
    structure_phase_screens: dict[int, PhaseScreen] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_tensor("assignment_id", self.assignment_id, dtype=torch.int64, ndim=1)
        require_tensor("material_id", self.material_id, dtype=torch.int64, ndim=1)
        require_tensor("structure_id", self.structure_id, dtype=torch.int64, ndim=1)
        require_tensor("surface_id", self.surface_id, dtype=torch.int64, ndim=1)
        require_tensor("face_material_id", self.face_material_id, dtype=torch.int32, ndim=1)
        require_tensor("edge_material_id0", self.edge_material_id0, dtype=torch.int32, ndim=1)
        require_tensor("edge_material_id1", self.edge_material_id1, dtype=torch.int32, ndim=1)
        require_tensor("surface_material_id", self.surface_material_id, dtype=torch.int32, ndim=1)
        require_tensor(
            "structure_material_id", self.structure_material_id, dtype=torch.int32, ndim=1
        )
        if self.face_material_id.shape[0] != self.num_faces:
            raise ValueError("face_material_id length must match num_faces")
        if self.edge_material_id0.shape[0] != self.num_edges:
            raise ValueError("edge_material_id0 length must match num_edges")
        if self.edge_material_id1.shape[0] != self.num_edges:
            raise ValueError("edge_material_id1 length must match num_edges")
        num_structures = self.structure_material_id.shape[0]
        if not (
            self.assignment_id.shape[0]
            == self.material_id.shape[0]
            == self.structure_id.shape[0]
            == self.surface_id.shape[0]
            == num_structures
        ):
            raise ValueError(
                "stable assignment/material/structure IDs must have one row "
                "per structure"
            )
        for name, values in (
            ("assignment_id", self.assignment_id),
            ("structure_id", self.structure_id),
        ):
            if torch.unique(values).numel() != values.numel():
                raise ValueError(f"{name} must be unique")
        for index, screen in self.structure_phase_screens.items():
            if not isinstance(index, int) or not 0 <= index < num_structures:
                raise ValueError(
                    "structure_phase_screens keys must be structure indices in "
                    f"[0, {num_structures})"
                )
            if not isinstance(screen, PhaseScreen):
                raise ValueError("structure_phase_screens values must be PhaseScreen")


def _validated_time(time_s: float | torch.Tensor | None) -> float | torch.Tensor | None:
    """Normalize a compiled snapshot instant without reading a tensor."""

    if isinstance(time_s, torch.Tensor):
        if time_s.ndim != 0 or not time_s.dtype.is_floating_point:
            raise TypeError("time_s must be a scalar floating-point tensor")
        return time_s
    if time_s is None or isinstance(time_s, float):
        return time_s
    if isinstance(time_s, int) and not isinstance(time_s, bool):
        return float(time_s)
    raise TypeError("time_s must be a float, a scalar tensor, or None")


@dataclass(slots=True)
class CompiledScene:
    source: Scene | SceneSnapshot
    structures: tuple[object, ...]
    geometry: GeometryStore
    materials: MaterialStore
    assignments: AssignmentStore
    rayd: RayDSceneResource
    reference_frequency_hz: float | torch.Tensor
    reference_frequency_revision: int | None
    topology_version: int
    geometry_version: int
    material_version: int
    assignment_version: int
    # The SceneSnapshot instant this runtime was compiled from, or None for a
    # plain Scene. Reporting and cross-consumer correlation only: it records
    # which world instant a CompiledScene is, and it never gates a call. The
    # four version domains above are the freshness authority (world-version validation).
    time_s: float | torch.Tensor | None = None
    enumerated_penetration_scene_diagonal_m: float = 0.0
    montecarlo_penetration_scene_diagonal_m: float = 0.0
    # Lazy scattering caches (built on first access so smooth scenes pay no
    # compile cost). A CompiledScene instance is already cache-keyed by the
    # material cache_token in Scene.compile, so instance-level caching is
    # consistent with the material cache.
    _kirchhoff_resources_cache: KirchhoffRuntimeResources | None = field(
        default=None, repr=False, compare=False
    )
    _phase_screen_resources_cache: PhaseScreenRuntimeResources | None = field(
        default=None, repr=False, compare=False
    )
    # (source-state signature, tables). The signature is re-derived on every
    # call so a scene leaf marked AFTER the cache was populated invalidates it
    # instead of being silently severed from the graph.
    _fixed_reevaluation_tables_cache: tuple[tuple, dict] | None = field(
        default=None, repr=False, compare=False
    )
    # The structure, material and assignment state this runtime was compiled
    # from, and the RayD refit generation it was built at; None for a runtime
    # assembled by hand, which has no source state to be stale against.
    _world: _WorldState | None = field(default=None, repr=False, compare=False)
    _rayd_generation: int = field(default=0, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, (Scene, SceneSnapshot)):
            raise TypeError("source must be a witwin.core Scene or SceneSnapshot")
        frequency = self.reference_frequency_hz
        if isinstance(frequency, torch.Tensor):
            if frequency.ndim != 0 or not frequency.dtype.is_floating_point:
                raise TypeError(
                    "reference_frequency_hz must be a scalar floating-point tensor"
                )
            if self.reference_frequency_revision is None:
                raise TypeError(
                    "tensor reference frequency requires a mutation revision"
                )
        elif not isinstance(frequency, float):
            raise TypeError("reference_frequency_hz must be a float or tensor")
        elif self.reference_frequency_revision is not None:
            raise TypeError("scalar reference frequency has no mutation revision")
        self.time_s = _validated_time(self.time_s)
        for name in (
            "enumerated_penetration_scene_diagonal_m",
            "montecarlo_penetration_scene_diagonal_m",
        ):
            value = getattr(self, name)
            if not isinstance(value, float):
                raise TypeError(f"{name} must be a float")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def stale_domain(self) -> str | None:
        """Name the world domain that changed in the source since compilation, or ``None``.

        Compares the structure, material and assignment state recorded at
        compile time with the source's current state, using host attributes only
        (no device work, no synchronization). The scene's own antennas are not
        part of it: no compiled resource depends on them. Returns
        ``"geometry_version"`` as well when the shared RayD scene was refit for a
        newer compile of the same, edited, structures.
        """

        recorded = getattr(self, "_world", None)
        if recorded is None:
            return None
        moved = recorded.moved_domain(world_state(self.source))
        if moved is not None:
            return moved
        if self._rayd_generation != self.rayd.generation:
            return "geometry_version"
        return None

    def __reduce_ex__(self, protocol: object) -> object:
        raise TypeError(
            "a CompiledScene owns a native RayD scene on one GPU and cannot be pickled or "
            "copied; pass the witwin.core Scene and call witwin.channel.scene.compile() again"
        )

    def __deepcopy__(self, memo: object) -> CompiledScene:
        self.__reduce_ex__(None)
        raise AssertionError("unreachable")

    @property
    def dropped_faces(self) -> dict[str, int]:
        """Faces removed at compile: ``degenerate`` (repeated vertex index) and ``duplicate``."""

        return {
            "degenerate": sum(
                int(getattr(structure, "dropped_degenerate_faces", 0))
                for structure in self.structures
            ),
            "duplicate": sum(
                int(getattr(structure, "dropped_duplicate_faces", 0))
                for structure in self.structures
            ),
        }

    def require_reference_frequency(self, reference_frequency_hz: float | torch.Tensor) -> None:
        """Reject a request/compile frequency mismatch before native work."""

        compiled = self.reference_frequency_hz
        if isinstance(compiled, torch.Tensor):
            if reference_frequency_hz is not compiled:
                raise ValueError(
                    "reference_frequency_hz does not match the compiled tensor "
                    "identity"
                )
            if int(reference_frequency_hz._version) != self.reference_frequency_revision:
                raise ValueError(
                    "reference_frequency_hz changed after scene compilation"
                )
            return
        if isinstance(reference_frequency_hz, torch.Tensor):
            raise ValueError(
                "reference_frequency_hz tensor does not match the compiled scalar"
            )
        if float(reference_frequency_hz).hex() != compiled.hex():
            raise ValueError(
                "reference_frequency_hz does not exactly match CompiledScene"
            )

    def _fixed_reevaluation_table_sources(self) -> tuple[torch.Tensor, ...]:
        """Every tensor the fixed-replay tables are built from.

        The vertex table concatenates the live structure vertex tensors (or the
        native table when the scene has no structures); the material bundle is
        an index_select over the material and assignment stores. Enumerating the
        stores by dataclass field keeps this a superset of whatever the bundle
        builder reads, so a new store field cannot silently escape the liveness
        signature below.
        """

        sources: list[torch.Tensor] = [self.geometry.vertices]
        for structure in self.structures:
            vertices = getattr(structure, "vertices", None)
            if isinstance(vertices, torch.Tensor):
                sources.append(vertices)
        for store in (self.materials, self.assignments):
            for store_field in fields(store):
                value = getattr(store, store_field.name)
                if isinstance(value, torch.Tensor):
                    sources.append(value)
        return tuple(sources)

    def _fixed_reevaluation_table_state(self) -> tuple[tuple[bool, bool, bool, int], ...]:
        """Host-only autograd/mutation signature of the replay-table sources.

        Reads tensor attributes only - ``requires_grad``, the presence of a
        ``grad_fn`` or a forward-AD tangent, and the mutation ``_version``. It
        touches no device memory, launches nothing, and never synchronizes, so
        it is cheap enough to re-derive on every replay.
        """

        state: list[tuple[bool, bool, bool, int]] = []
        for source in self._fixed_reevaluation_table_sources():
            state.append(
                (
                    bool(source.requires_grad),
                    source.grad_fn is not None,
                    torch.autograd.forward_ad.unpack_dual(source).tangent is not None,
                    int(source._version),
                )
            )
        return tuple(state)

    def fixed_reevaluation_tables(self) -> dict[str, object]:
        """Scene-static vertex and material tables for fixed-topology replay.

        Built lazily per instance following the scene-static cache pattern:
        both tables depend only on this CompiledScene's immutable stores and
        structure tensors, and any world change produces a new instance through
        the version-token compile cache. The cache is bypassed whenever a table
        tensor participates in autograd - a cached graph node would be freed by
        the first backward and fail the second - so differentiable-material or
        differentiable-mesh loops pay the staging exactly as before, while the
        primal per-frame replay (the Radar Doppler shape) stages once.

        The bypass is decided per call, not once at population time: the cache
        is keyed on the host-only liveness/mutation signature of the source
        tensors, so marking ``materials.eps_r`` or a structure's vertices after
        a primal replay has already warmed the cache rebuilds the tables and
        keeps that leaf on the graph. Serving a stale cached table there would
        drop the leaf's gradient silently, which the AD capability record
        forbids.
        """

        signature = self._fixed_reevaluation_table_state()
        cached = self._fixed_reevaluation_tables_cache
        if cached is not None and cached[0] == signature:
            return cached[1]

        from witwin.channel.materials import face_material_field_bundle
        from witwin.channel.scene import endpoints

        tables: dict[str, object] = {
            "vertices": endpoints.scene_vertex_table(self, self),
            "material": face_material_field_bundle(
                self, device=self.geometry.vertices.device
            ),
        }

        def _participates(value: object) -> bool:
            if not isinstance(value, torch.Tensor):
                return False
            if value.requires_grad or value.grad_fn is not None:
                return True
            return (
                torch.autograd.forward_ad.unpack_dual(value).tangent is not None
            )

        graph_bearing = _participates(tables["vertices"]) or any(
            _participates(value) for value in tables["material"].values()
        )
        self._fixed_reevaluation_tables_cache = (
            None if graph_bearing else (signature, tables)
        )
        return tables

    @property
    def kirchhoff_resources(self) -> KirchhoffRuntimeResources:
        """Kirchhoff BSDF tables per material index (scatter_model_id == 1).

        Built lazily from the MaterialStore CSR layers and roughness fields
        at the store frequency; raises ``kirchhoff_domain_exceeded`` for
        out-of-domain roughness (never silently degrades).
        """

        key = self._scattering_resource_key()
        if (
            self._kirchhoff_resources_cache is None
            or self._kirchhoff_resources_cache.key != key
        ):
            self._kirchhoff_resources_cache = build_kirchhoff_resources(
                self.materials, key
            )
        return self._kirchhoff_resources_cache

    @property
    def phase_screen_resources(self) -> PhaseScreenRuntimeResources:
        """Immutable per-structure phase-screen resources, built lazily."""

        key = self._phase_screen_resource_key()
        if (
            self._phase_screen_resources_cache is None
            or self._phase_screen_resources_cache.key != key
        ):
            self._phase_screen_resources_cache = build_phase_screen_resources(
                self.materials, self.assignments, self.rayd, key
            )
        return self._phase_screen_resources_cache

    @property
    def kirchhoff_tables(self) -> dict[int, KirchhoffTable]:
        """Compatibility view of the typed Kirchhoff table resources."""

        return self.kirchhoff_resources.tables

    @property
    def rough_material_runtimes(self) -> dict[int, RoughMaterialRuntime]:
        """Typed rough-material runtimes sharing the cached table objects."""

        return self.kirchhoff_resources.materials

    def _scattering_resource_key(self) -> ScatteringResourceKey:
        device = torch.device("cuda")
        return ScatteringResourceKey(
            material_cache_token=self.materials.cache_token,
            assignment_version=self.assignment_version,
            device=device,
        )

    def _phase_screen_resource_key(self) -> PhaseScreenResourceKey:
        device = torch.device("cuda")
        return PhaseScreenResourceKey(
            material_cache_token=self.materials.cache_token,
            geometry_version=self.geometry_version,
            assignment_version=self.assignment_version,
            phase_screen_token=self._phase_screen_assignment_token(),
            structure_uv_presence=self.geometry.structure_uv_presence,
            # Pure Python wrapper identity for cache invalidation. This is not
            # a native pointer or an integer scene handle; CompiledScene keeps
            # the owning RayDSceneResource alive for the cache lifetime.
            rayd_scene_identity=id(self.rayd),
            device=device,
        )

    def _phase_screen_assignment_token(self) -> tuple[tuple[object, ...], ...]:
        """Mutation-aware identity without reading resident values to the host."""

        tokens: list[tuple[object, ...]] = []
        for structure_index, screen in sorted(
            self.assignments.structure_phase_screens.items()
        ):
            height = screen.height
            if isinstance(height, torch.Tensor):
                height_token: tuple[object, ...] = (
                    "tensor",
                    id(height),
                    int(height._version),
                    tuple(height.shape),
                    height.dtype,
                    height.device,
                )
            else:
                height_token = (
                    "sequence",
                    tuple(tuple(float(value) for value in row) for row in height),
                )
            tokens.append((structure_index, id(screen), *height_token))
        return tuple(tokens)


_ResultT = TypeVar("_ResultT")
_GEOMETRY_MODE_IDS = {"thin_sheet": 0, "volumetric": 1}
_SCATTER_MODEL_SMOOTH = 0
_SCATTER_MODEL_KIRCHHOFF = 1


@dataclass(frozen=True, slots=True)
class _RuntimeStructure:
    source: Structure
    vertices: torch.Tensor
    faces: torch.Tensor
    uv: torch.Tensor | None
    face_uv: torch.Tensor | None
    # Authored local index (int64, host) of every kept face, or None when no face was dropped.
    kept_face_index: torch.Tensor | None = None
    dropped_degenerate_faces: int = 0
    dropped_duplicate_faces: int = 0

    @property
    def name(self) -> str:
        return self.source.name or ""

    @property
    def material(self):
        return self.source.material

    @property
    def phase_screen(self) -> PhaseScreen | None:
        return self.source.phase_screen

    @property
    def structure_id(self) -> int:
        assert self.source.structure_id is not None
        return int(self.source.structure_id)

    @property
    def assignment_id(self) -> int:
        assert self.source.assignment_id is not None
        return int(self.source.assignment_id)

    @property
    def material_id(self) -> int:
        assert self.source.material_id is not None
        return int(self.source.material_id)


# ---------------------------------------------------------------------------
# World state: what a compiled runtime depends on, read from host attributes.
# ---------------------------------------------------------------------------
#
# The Core version domains also count the scene's own antennas, which no
# compiled resource depends on: endpoints are read live from the source scene
# when a solver binds it, and the propagation consumer takes them from its
# endpoint batches. Channel therefore keys its runtimes on its own state of the
# structures, materials and assignments. Every token is built from host-side
# attributes only - Python object identity, scalar fields, and per tensor its
# identity, mutation counter, storage address and ``requires_grad`` - so it
# launches nothing, reads no device memory and never synchronizes. An in-place
# edit advances the mutation counter, a ``tensor.data`` swap moves the storage
# address, and an attribute reassignment changes the identity.

_ATOMIC_STATE_TYPES = frozenset(
    {type(None), bool, int, float, complex, str, bytes}
)
_FIELD_NAMES: dict[type, tuple[str, ...] | None] = {}


def _field_names(kind: type) -> tuple[str, ...] | None:
    names = _FIELD_NAMES.get(kind, ())
    if names != ():
        return names
    if is_dataclass(kind):
        resolved: tuple[str, ...] | None = tuple(item.name for item in fields(kind))
    else:
        slots = tuple(
            name
            for klass in kind.__mro__
            for name in getattr(klass, "__slots__", ())
            if name not in ("__weakref__", "__dict__")
        )
        resolved = slots or None
    _FIELD_NAMES[kind] = resolved
    return resolved


def _append_state(value: object, out: list[object], seen: set[int]) -> None:
    """Append the host-visible mutation state of ``value`` to ``out``."""

    kind = type(value)
    if kind in _ATOMIC_STATE_TYPES:
        out.append(value)
        return
    if isinstance(value, torch.Tensor):
        out.append((id(value), value._version, value.data_ptr(), value.requires_grad))
        return
    marker = id(value)
    if marker in seen:
        out.append(("seen", marker))
        return
    seen.add(marker)
    if kind is tuple or kind is list:
        out.append(len(value))  # type: ignore[arg-type]
        for item in value:  # type: ignore[attr-defined]
            _append_state(item, out, seen)
        return
    if isinstance(value, Mapping):
        out.append(len(value))
        for key, item in value.items():
            out.append(key if type(key) in _ATOMIC_STATE_TYPES else id(key))
            _append_state(item, out, seen)
        return
    out.append((kind, marker))
    instance_dict = getattr(value, "__dict__", None)
    if instance_dict is not None and not is_dataclass(kind):
        for name, item in instance_dict.items():
            if not name.endswith("_cache"):
                out.append(name)
                _append_state(item, out, seen)
        return
    names = _field_names(kind)
    if names is None:
        return
    for name in names:
        _append_state(getattr(value, name, None), out, seen)


def _state(value: object) -> tuple[object, ...]:
    out: list[object] = []
    _append_state(value, out, set())
    return tuple(out)


def _tensor_state(value: torch.Tensor | None) -> tuple[object, ...] | None:
    if value is None:
        return None
    return (id(value), value._version, value.data_ptr(), value.requires_grad)


def _geometry_object_state(geometry: object) -> tuple[object, ...]:
    """State of everything a geometry's world vertices come from (faces excluded)."""

    values = getattr(geometry, "__dict__", None)
    if values is None:
        return _state(geometry)
    out: list[object] = [type(geometry), id(geometry)]
    seen: set[int] = {id(geometry)}
    for name, value in values.items():
        if name == "_faces_tensor" or name.endswith("_cache"):
            continue
        out.append(name)
        _append_state(value, out, seen)
    return tuple(out)


@dataclass(frozen=True, slots=True, eq=False)
class _SourcePlan:
    """The fixed shape of one frozen Core scene or snapshot, derived once per object."""

    # (structure, rigid_motion, deformation) of every enabled structure, in order.
    rows: tuple[tuple[Structure, object | None, object | None], ...]
    # Per enabled structure: stable IDs, primitive IDs and geometry identity.
    topology_static: tuple[tuple[object, ...], ...]
    # Per enabled structure: the IDs and phase-screen identity of its assignment.
    assignment_static: tuple[tuple[object, ...], ...]
    materials: tuple[tuple[int, object], ...]
    edge_policy: object


_PLAN_LOCK = RLock()
_PLANS: dict[int, tuple[weakref.ref, _SourcePlan]] = {}


def _source_plan(scene_or_snapshot: Scene | SceneSnapshot) -> _SourcePlan:
    key = id(scene_or_snapshot)
    with _PLAN_LOCK:
        hit = _PLANS.get(key)
        if hit is not None and hit[0]() is scene_or_snapshot:
            return hit[1]
    rows = tuple(item for item in _source_structures(scene_or_snapshot) if item[0].enabled)
    topology_static = tuple(
        (
            int(structure.structure_id),
            int(structure.surface_id),
            None if structure.primitive_ids is None else tuple(structure.primitive_ids),
            type(structure.geometry),
            id(structure.geometry),
        )
        for structure, _, _ in rows
    )
    assignment_static = tuple(
        (
            int(structure.structure_id),
            int(structure.surface_id),
            int(structure.assignment_id),
            int(structure.material_id),
            id(structure.phase_screen),
        )
        for structure, _, _ in rows
    )
    materials: dict[int, object] = {}
    for structure, _, _ in rows:
        materials[int(structure.material_id)] = structure.material
    plan = _SourcePlan(
        rows=rows,
        topology_static=topology_static,
        assignment_static=assignment_static,
        materials=tuple(sorted(materials.items())),
        edge_policy=scene_or_snapshot.metadata.get("imported_edge_policy"),
    )

    def _forget(_: object, key: int = key) -> None:
        with _PLAN_LOCK:
            entry = _PLANS.get(key)
            if entry is not None and entry[0]() is None:
                del _PLANS[key]

    with _PLAN_LOCK:
        _PLANS[key] = (weakref.ref(scene_or_snapshot, _forget), plan)
    return plan


@dataclass(frozen=True, slots=True, eq=False)
class _WorldState:
    """Channel's view of one Core world: the four domains its runtimes depend on.

    The domains mirror the Core version domains restricted to structures,
    materials and assignments; endpoints are not part of any of them.
    """

    plan: _SourcePlan
    # Per enabled structure: the face and face-UV table states.
    topology_rows: tuple[tuple[object, ...], ...]
    # Per enabled structure: geometry, UV, rigid motion and deformation states.
    geometry_rows: tuple[tuple[object, ...], ...]
    # Snapshot instant, dynamic sources and the edge policy.
    geometry_extra: tuple[object, ...]
    material: tuple[object, ...]
    assignment: tuple[object, ...]

    def rayd_key(self) -> tuple[object, ...]:
        """Everything the RayD scene, geometry tables and diagonals are built from."""

        return (
            self.plan.topology_static,
            self.topology_rows,
            self.geometry_rows,
            self.geometry_extra,
        )

    def layout_key(self) -> tuple[object, ...]:
        """What must hold for an in-place refit: the same objects with the same faces and UVs."""

        return (
            self.plan.topology_static,
            self.topology_rows,
            tuple(
                (row[0][1], row[1], id(motion), id(deformation))
                for row, (_, motion, deformation) in zip(self.geometry_rows, self.plan.rows)
            ),
            self.geometry_extra,
        )

    def material_key(self) -> tuple[object, ...]:
        return (self.plan.materials, self.material)

    def assignment_key(self) -> tuple[object, ...]:
        return (
            self.plan.topology_static,
            self.topology_rows,
            self.plan.assignment_static,
            tuple(material_id for material_id, _ in self.plan.materials),
            self.assignment,
        )

    def moved_domain(self, current: _WorldState) -> str | None:
        """Name the Core-style domain that differs from ``current``, or ``None``."""

        if (
            self.plan.topology_static != current.plan.topology_static
            or self.topology_rows != current.topology_rows
        ):
            return "topology_version"
        if self.material_key() != current.material_key():
            return "material_version"
        if (
            self.plan.assignment_static != current.plan.assignment_static
            or self.assignment != current.assignment
        ):
            return "assignment_version"
        if (
            self.geometry_rows != current.geometry_rows
            or self.geometry_extra != current.geometry_extra
        ):
            return "geometry_version"
        return None

    def same_world(self, current: _WorldState) -> bool:
        return (
            self.geometry_rows == current.geometry_rows
            and self.topology_rows == current.topology_rows
            and self.material == current.material
            and self.assignment == current.assignment
            and self.geometry_extra == current.geometry_extra
            and (
                self.plan is current.plan
                or (
                    self.plan.topology_static == current.plan.topology_static
                    and self.plan.assignment_static == current.plan.assignment_static
                    and self.plan.materials == current.plan.materials
                )
            )
        )


def _translated_state_error(error: RuntimeError) -> Exception:
    """Restate the two tensor kinds whose mutation state cannot be read."""

    if "doesn't have storage" in str(error) or "Cannot access data pointer" in str(error):
        from witwin.channel.scene.endpoints import FUNCTORCH_MESSAGE

        return NotImplementedError(FUNCTORCH_MESSAGE)
    if "Inference tensors" not in str(error):
        return error
    return ValueError(
        "the scene holds tensors created under torch.inference_mode(); Channel tracks "
        "scene edits through tensor version counters, which inference tensors do not "
        "have. Build the scene outside torch.inference_mode() (solving inside it is fine)"
    )


def world_state(scene_or_snapshot: Scene | SceneSnapshot) -> _WorldState:
    """Read the structure, material and assignment state of a Core world on the host."""

    try:
        return _world_state(scene_or_snapshot)
    except RuntimeError as error:
        translated = _translated_state_error(error)
        if translated is error:
            raise
        raise translated from error


def _world_state(scene_or_snapshot: Scene | SceneSnapshot) -> _WorldState:
    plan = _source_plan(scene_or_snapshot)
    topology_rows: list[tuple[object, ...]] = []
    geometry_rows: list[tuple[object, ...]] = []
    assignment: list[object] = []
    for structure, motion, deformation in plan.rows:
        geometry = structure.geometry
        topology_rows.append(
            (
                _tensor_state(getattr(geometry, "_faces_tensor", None)),
                _tensor_state(structure.face_uv),
            )
        )
        geometry_rows.append(
            (
                _geometry_object_state(geometry),
                _tensor_state(structure.uv),
                None if motion is None else _state(motion),
                None if deformation is None else _state(deformation),
            )
        )
        screen = structure.phase_screen
        if screen is not None:
            assignment.append(_state(screen))
    if isinstance(scene_or_snapshot, SceneSnapshot):
        geometry_extra: tuple[object, ...] = (
            _state(scene_or_snapshot.time_s),
            _state(scene_or_snapshot._dynamic_sources),
            scene_or_snapshot._dynamic_revision,
            _state(plan.edge_policy),
        )
    else:
        geometry_extra = (_state(plan.edge_policy),)
    material: list[object] = []
    for _, material_object in plan.materials:
        material.append(_state(material_object))
    return _WorldState(
        plan=plan,
        topology_rows=tuple(topology_rows),
        geometry_rows=tuple(geometry_rows),
        geometry_extra=geometry_extra,
        material=tuple(material),
        assignment=tuple(assignment),
    )


# Compiled runtimes kept for reuse are bounded by the device bytes they hold. The
# newest runtime is always kept, so one scene larger than the bound still reuses.
_CACHE_BYTE_LIMIT = 1 << 30
_CACHE_ENTRY_LIMIT = 64


def _tensor_bytes(value: object, seen: set[int], depth: int = 0) -> int:
    """Device bytes held by the tensors in ``value``, each storage counted once."""

    if isinstance(value, torch.Tensor):
        if value.device.type != "cuda":
            return 0
        storage = value.untyped_storage()
        key = storage.data_ptr()
        if key in seen:
            return 0
        seen.add(key)
        return int(storage.nbytes())
    if depth > 3:
        return 0
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item, seen, depth + 1) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen, depth + 1) for item in value.values())
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _tensor_bytes(getattr(value, item.name), seen, depth + 1) for item in fields(value)
        )
    return 0


def _resident_bytes(entries: Iterable[CompiledScene]) -> int:
    """Device bytes the cached runtimes keep alive, each shared resource counted once."""

    seen_storages: set[int] = set()
    seen_rayd: set[int] = set()
    total = 0
    for compiled in entries:
        if id(compiled.rayd) not in seen_rayd:
            seen_rayd.add(id(compiled.rayd))
            total += compiled.rayd.native_bytes
            total += _tensor_bytes(compiled.rayd.runtime_cache, seen_storages)
        total += _tensor_bytes(
            (compiled.geometry, compiled.materials, compiled.assignments), seen_storages
        )
        total += _tensor_bytes(
            tuple(structure.vertices for structure in compiled.structures), seen_storages
        )
    return total


class _CompileRegistry:
    """Channel-owned runtime cache: one slot per source, frequency, device and backend.

    A slot holds the runtime compiled from the source's current world. A
    recompile of the same source after its world changed replaces every slot of
    that source at once, so an edited scene never keeps its previous runtimes
    alive. Least recently used runtimes are dropped while the cache holds more
    than ``_CACHE_BYTE_LIMIT`` device bytes or ``_CACHE_ENTRY_LIMIT`` runtimes.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._compiled: OrderedDict[tuple[object, ...], CompiledScene] = OrderedDict()

    def get(
        self, key: tuple[object, ...], source: object, state: _WorldState,
    ) -> CompiledScene | None:
        with self._lock:
            value = self._compiled.get(key)
            if (
                value is None
                or value.source is not source
                or value._world is None
                or value._rayd_generation != value.rayd.generation
                or not value._world.same_world(state)
            ):
                return None
            self._compiled.move_to_end(key)
            return value

    def candidates(self, runtime_device: object) -> tuple[CompiledScene, ...]:
        """Current cached runtimes built on ``runtime_device``, most recent first."""

        with self._lock:
            return tuple(
                compiled
                for key, compiled in reversed(self._compiled.items())
                if key[2] == runtime_device
                and compiled._rayd_generation == compiled.rayd.generation
            )

    def put(self, key: tuple[object, ...], compiled: CompiledScene) -> None:
        with self._lock:
            source_id = key[0]
            for stale in [
                other
                for other, value in self._compiled.items()
                if other[0] == source_id
                and value._world is not None
                and compiled._world is not None
                and not value._world.same_world(compiled._world)
            ]:
                del self._compiled[stale]
            self._compiled[key] = compiled
            self._compiled.move_to_end(key)
            while len(self._compiled) > 1 and (
                len(self._compiled) > _CACHE_ENTRY_LIMIT
                or _resident_bytes(self._compiled.values()) > _CACHE_BYTE_LIMIT
            ):
                self._compiled.popitem(last=False)

    def drop_rayd(self, rayd: RayDSceneResource) -> None:
        """Forget every runtime built on ``rayd`` (its geometry is about to be refit)."""

        with self._lock:
            for key in [key for key, value in self._compiled.items() if value.rayd is rayd]:
                del self._compiled[key]

    def resident_bytes(self) -> int:
        with self._lock:
            return _resident_bytes(self._compiled.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._compiled)

    def clear(self) -> None:
        with self._lock:
            self._compiled.clear()


_REGISTRY = _CompileRegistry()


def clear_compile_cache() -> None:
    """Clear the bounded Channel runtime registry (primarily for tests)."""

    _REGISTRY.clear()


def _frequency_value(reference_frequency_hz: float | torch.Tensor) -> float:
    if isinstance(reference_frequency_hz, torch.Tensor):
        if (
            reference_frequency_hz.ndim != 0
            or not reference_frequency_hz.dtype.is_floating_point
        ):
            raise TypeError(
                "reference_frequency_hz must be a scalar floating-point tensor"
            )
        value = float(reference_frequency_hz.detach())
    else:
        value = float(reference_frequency_hz)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("reference_frequency_hz must be finite and positive")
    return value


def _frequency_identity(
    reference_frequency_hz: float | torch.Tensor, value: float,
) -> tuple[object, ...]:
    if isinstance(reference_frequency_hz, torch.Tensor):
        return (
            "tensor",
            id(reference_frequency_hz),
            int(reference_frequency_hz._version),
            reference_frequency_hz.device,
            reference_frequency_hz.dtype,
            value.hex(),
        )
    return ("float", value.hex())


def _versions(scene_or_snapshot: Scene | SceneSnapshot) -> tuple[int, int, int, int]:
    return (
        int(scene_or_snapshot.topology_version),
        int(scene_or_snapshot.geometry_version),
        int(scene_or_snapshot.material_version),
        int(scene_or_snapshot.assignment_version),
    )


def _readable_versions(scene_or_snapshot: Scene | SceneSnapshot) -> tuple[int, int, int, int]:
    """Read the four version domains, restating the two tensor kinds that cannot report one."""

    try:
        return _versions(scene_or_snapshot)
    except RuntimeError as error:
        translated = _translated_state_error(error)
        if translated is error:
            raise
        raise translated from error


def _source_structures(
    scene_or_snapshot: Scene | SceneSnapshot,
) -> tuple[tuple[Structure, object | None, object | None], ...]:
    if isinstance(scene_or_snapshot, Scene):
        return tuple((structure, None, None) for structure in scene_or_snapshot.structures)
    return tuple(
        (state.structure, state.rigid_motion, state.deformation)
        for state in scene_or_snapshot.structures
    )


# RayD emits one diffraction wedge per pair of faces sharing an edge and refuses
# an edge shared by more faces than this.
_MAX_EDGE_INCIDENT_FACES = 256
_INT32_MAX = 2**31 - 1
# Largest vertex count whose sorted index triples pack into one int64 key.
_SINGLE_KEY_VERTEX_LIMIT = 2**21 - 1


def _structure_label(index: int, structure: Structure) -> str:
    name = f" {structure.name!r}" if structure.name else ""
    return f"structure {index}{name}"


def _require_index_table(
    label: str, name: str, table: torch.Tensor, row_count: int, target: str,
) -> None:
    """Reject an index table entry outside ``[0, row_count)`` before any native work."""

    if table.dtype == torch.bool or table.dtype.is_floating_point or table.dtype.is_complex:
        raise TypeError(f"{label}: {name} must be an integer tensor, got {table.dtype}")
    if row_count > _INT32_MAX:
        raise ValueError(
            f"{label}: {row_count} {target} exceed the 32-bit index range "
            f"(at most {_INT32_MAX})"
        )
    if table.numel() == 0:
        return
    host = table.detach().to(device="cpu", dtype=torch.int64)
    bad = (host < 0) | (host >= row_count)
    if bool(bad.any()):
        row = int(bad.any(dim=1).nonzero()[0, 0])
        value = int(host[row][bad[row]][0])
        raise ValueError(
            f"{label}: {name}[{row}] holds index {value}, outside the valid range "
            f"[0, {row_count}) of its {row_count} {target}"
        )


def _require_finite_table(label: str, name: str, table: torch.Tensor) -> None:
    if table.numel() and not bool(torch.isfinite(table.detach()).all()):
        row = int((~torch.isfinite(table.detach())).reshape(table.shape[0], -1).any(dim=1)
                  .nonzero()[0, 0])
        raise ValueError(f"{label}: {name}[{row}] is not finite (NaN or infinite coordinate)")


def _face_cleanup(
    label: str, faces: torch.Tensor, vertex_count: int,
) -> tuple[torch.Tensor | None, int, int]:
    """Return the kept-face mask (None when all faces are kept) and the dropped counts.

    A face that repeats a vertex index has zero area by construction; a face whose
    vertex index set equals an earlier face's (in either winding) is an exact
    duplicate. Both are decided from indices only, so the result does not change
    when the same mesh is deformed. An edge shared by more distinct faces than the
    edge topology supports is rejected here with the structure and edge named.
    """

    host = faces.detach().to(device="cpu", dtype=torch.int64).numpy()
    face_count = int(host.shape[0])
    if face_count == 0:
        return None, 0, 0
    first, second, third = host[:, 0], host[:, 1], host[:, 2]
    low = np.minimum(np.minimum(first, second), third)
    high = np.maximum(np.maximum(first, second), third)
    middle = first + second + third - low - high
    degenerate = (low == middle) | (middle == high)
    # A stable sort keeps the earliest face first in every group of equal index
    # sets, so every later member of a group is a duplicate.
    if vertex_count <= _SINGLE_KEY_VERTEX_LIMIT:
        face_key = (low * vertex_count + middle) * vertex_count + high
        order = np.argsort(face_key, kind="stable")
        grouped = face_key[order]
        repeat = np.zeros(face_count, dtype=bool)
        repeat[1:] = grouped[1:] == grouped[:-1]
    else:
        order = np.lexsort((high, middle, low))
        repeat = np.zeros(face_count, dtype=bool)
        repeat[1:] = (
            (low[order][1:] == low[order][:-1])
            & (middle[order][1:] == middle[order][:-1])
            & (high[order][1:] == high[order][:-1])
        )
    duplicate = np.zeros(face_count, dtype=bool)
    duplicate[order[repeat]] = True
    duplicate &= ~degenerate
    keep = ~(degenerate | duplicate)
    if keep.any():
        low, middle, high = low[keep], middle[keep], high[keep]
        edge_keys = np.sort(
            np.concatenate(
                (
                    low * vertex_count + middle,
                    middle * vertex_count + high,
                    low * vertex_count + high,
                )
            )
        )
        starts = np.flatnonzero(np.diff(edge_keys, prepend=-1) != 0)
        counts = np.diff(np.append(starts, edge_keys.shape[0]))
        worst = int(counts.argmax())
        if int(counts[worst]) > _MAX_EDGE_INCIDENT_FACES:
            v0, v1 = divmod(int(edge_keys[starts[worst]]), vertex_count)
            raise ValueError(
                f"{label}: the edge between vertices {v0} and {v1} is shared by "
                f"{int(counts[worst])} distinct faces; at most "
                f"{_MAX_EDGE_INCIDENT_FACES} faces may share one edge. Split the "
                "fan into several structures or weld it into fewer faces"
            )
    dropped_degenerate = int(degenerate.sum())
    dropped_duplicate = int(duplicate.sum())
    if dropped_degenerate == 0 and dropped_duplicate == 0:
        return None, 0, 0
    if not bool(keep.any()):
        raise ValueError(
            f"{label}: every face repeats a vertex index or duplicates an earlier "
            "face; the mesh has no face with area"
        )
    return torch.from_numpy(keep), dropped_degenerate, dropped_duplicate


def _validated_mesh(
    label: str, vertices: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor | None,
    face_uv: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, int, int]:
    """Validate one structure's tables and return its kept faces and drop counts."""

    _require_finite_table(label, "vertices", vertices)
    if uv is not None and face_uv is not None:
        if face_uv.ndim != 2 or tuple(face_uv.shape) != tuple(faces.shape):
            raise ValueError(f"{label}: face_uv must have one (3,) row per mesh face")
        _require_finite_table(label, "uv", uv)
    keep, dropped_degenerate, dropped_duplicate = _checked_topology(
        label, faces, int(vertices.shape[0]), face_uv, None if uv is None else int(uv.shape[0])
    )
    if keep is None:
        return faces, face_uv, None, 0, 0
    kept_face_index = keep.nonzero().flatten()
    faces = faces[keep.to(device=faces.device)]
    if face_uv is not None:
        face_uv = face_uv[keep.to(device=face_uv.device)]
    return faces, face_uv, kept_face_index, dropped_degenerate, dropped_duplicate


_TOPOLOGY_CHECK_LIMIT = 64
_TOPOLOGY_CHECKS: OrderedDict[tuple[object, ...], tuple[object, ...]] = OrderedDict()
_TOPOLOGY_CHECK_LOCK = RLock()


def _tensor_token(value: torch.Tensor | None) -> tuple[object, ...] | None:
    if value is None:
        return ()
    if value.is_inference():
        return None
    return (id(value), int(value._version), value.data_ptr(), tuple(value.shape), value.device)


def _checked_topology(
    label: str, faces: torch.Tensor, vertex_count: int, face_uv: torch.Tensor | None,
    uv_count: int | None,
) -> tuple[torch.Tensor | None, int, int]:
    """Range-check the face and face-UV tables and decide which faces to keep.

    The result depends only on the index tables and the vertex/UV counts, so it is
    remembered per (tensor identity, mutation version) and an unchanged mesh pays
    the host pass once rather than on every recompile.
    """

    faces_token = _tensor_token(faces)
    uv_token = _tensor_token(face_uv)
    key = (
        None
        if faces_token is None or uv_token is None
        else (faces_token, uv_token, vertex_count, uv_count)
    )
    if key is not None:
        with _TOPOLOGY_CHECK_LOCK:
            hit = _TOPOLOGY_CHECKS.get(key)
            if hit is not None and hit[0]() is faces and hit[1]() is (face_uv if face_uv is not None else faces):
                _TOPOLOGY_CHECKS.move_to_end(key)
                return hit[2]  # type: ignore[return-value]
    _require_index_table(label, "faces", faces, vertex_count, "vertices")
    if face_uv is not None and uv_count is not None:
        _require_index_table(label, "face_uv", face_uv, uv_count, "uv rows")
    result = _face_cleanup(label, faces, vertex_count)
    if key is not None:
        with _TOPOLOGY_CHECK_LOCK:
            _TOPOLOGY_CHECKS[key] = (
                weakref.ref(faces),
                weakref.ref(face_uv if face_uv is not None else faces),
                result,
            )
            while len(_TOPOLOGY_CHECKS) > _TOPOLOGY_CHECK_LIMIT:
                _TOPOLOGY_CHECKS.popitem(last=False)
    return result


def _rebound_structures(
    structures: tuple[_RuntimeStructure, ...], state: _WorldState,
) -> tuple[_RuntimeStructure, ...]:
    """Reuse runtime structures for another source whose structures have the same geometry."""

    if all(
        runtime.source is row[0] for runtime, row in zip(structures, state.plan.rows)
    ):
        return structures
    return tuple(
        runtime if runtime.source is row[0] else replace(runtime, source=row[0])
        for runtime, row in zip(structures, state.plan.rows)
    )


def _runtime_structures(
    scene_or_snapshot: Scene | SceneSnapshot, state: _WorldState,
    donors: dict[tuple[object, ...], _RuntimeStructure],
) -> tuple[_RuntimeStructure, ...]:
    """World-space meshes of the enabled structures.

    A structure whose geometry object and state equal one a cached runtime
    already holds reuses that runtime's validated world vertices and faces
    instead of evaluating and validating its mesh again.
    """

    runtime: list[_RuntimeStructure] = []
    row_index = -1
    for structure_index, (structure, rigid_motion, deformation) in enumerate(
        _source_structures(scene_or_snapshot)
    ):
        if not structure.enabled:
            continue
        row_index += 1
        donor = donors.get(
            (
                id(structure.geometry),
                state.topology_rows[row_index],
                state.geometry_rows[row_index],
            )
        )
        if donor is not None:
            runtime.append(
                donor if donor.source is structure else replace(donor, source=structure)
            )
            continue
        label = _structure_label(structure_index, structure)
        geometry = structure.geometry
        vertices, faces = geometry.to_mesh()
        if not isinstance(vertices, torch.Tensor) or not isinstance(faces, torch.Tensor):
            raise TypeError("Core GeometrySpec.to_mesh() must return torch tensors")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Core mesh vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Core mesh faces must have shape (F, 3)")
        if deformation is not None:
            if not isinstance(geometry, Mesh):
                raise TypeError(
                    "Channel snapshot deformation currently requires "
                    "witwin.core.Mesh geometry"
                )
            local_vertices = (
                deformation.vertices
                if deformation.vertices is not None
                else geometry.vertices + deformation.offsets
            )
            if local_vertices.shape != geometry.vertices.shape:
                raise ValueError(
                    "snapshot deformation must match the authored local mesh"
                )
            if geometry.recenter:
                bounds_min = local_vertices.amin(dim=0)
                bounds_max = local_vertices.amax(dim=0)
                local_vertices = local_vertices - 0.5 * (bounds_min + bounds_max)
            local_vertices = local_vertices * geometry.scale.to(
                device=local_vertices.device, dtype=local_vertices.dtype
            )
            authored_rotation_value = geometry.rotation.to(
                device=local_vertices.device,
                dtype=local_vertices.dtype,
            )
            authored_rotation = quat_to_rotation_matrix(
                authored_rotation_value
                if authored_rotation_value.shape == (4,)
                else quat_from_euler(
                    authored_rotation_value[0],
                    authored_rotation_value[1],
                    authored_rotation_value[2],
                    device=authored_rotation_value.device,
                    dtype=authored_rotation_value.dtype,
                )
            )
            vertices = (
                local_vertices @ authored_rotation.T
                + geometry.position.to(
                    device=local_vertices.device,
                    dtype=local_vertices.dtype,
                )
            )
        if rigid_motion is not None:
            if rigid_motion.rotation is not None:
                rigid_rotation_value = rigid_motion.rotation
                if not isinstance(rigid_rotation_value, torch.Tensor):
                    rigid_rotation_value = torch.as_tensor(
                        rigid_rotation_value,
                        device=vertices.device,
                        dtype=vertices.dtype,
                    )
                rotation = quat_to_rotation_matrix(
                    rigid_rotation_value
                    if rigid_rotation_value.shape == (4,)
                    else quat_from_euler(
                        rigid_rotation_value[2],
                        rigid_rotation_value[1],
                        rigid_rotation_value[0],
                        device=rigid_rotation_value.device,
                        dtype=rigid_rotation_value.dtype,
                    )
                )
                vertices = vertices @ rotation.T
            if rigid_motion.translation is not None:
                vertices = vertices + rigid_motion.translation
        if (
            structure.primitive_ids is not None
            and len(structure.primitive_ids) != int(faces.shape[0])
        ):
            raise ValueError(
                "Core Structure.primitive_ids must contain one ID per mesh face"
            )
        uv = structure.uv
        face_uv = structure.face_uv
        if (uv is None) != (face_uv is None):
            raise ValueError("Core mesh uv and face_uv must be provided together")
        faces, face_uv, kept_face_index, dropped_degenerate, dropped_duplicate = (
            _validated_mesh(label, vertices, faces, uv, face_uv)
        )
        runtime.append(
            _RuntimeStructure(
                source=structure,
                vertices=vertices.to(dtype=torch.float32).contiguous(),
                faces=faces.to(dtype=torch.int32).contiguous(),
                uv=(
                    None
                    if uv is None
                    else uv.to(dtype=torch.float32).contiguous()
                ),
                face_uv=(
                    None
                    if face_uv is None
                    else face_uv.to(dtype=torch.int32).contiguous()
                ),
                kept_face_index=kept_face_index,
                dropped_degenerate_faces=dropped_degenerate,
                dropped_duplicate_faces=dropped_duplicate,
            )
        )
    return tuple(runtime)


def _compile_penetration_scene_diagonals(
    structures: tuple[_RuntimeStructure, ...], *, rayd: RayDSceneResource,
) -> tuple[float, float]:
    if not structures:
        return 0.0, 0.0
    records = rayd.edge_records()
    vertices = records.vertices
    enumerated = float(
        (vertices.max(dim=0).values - vertices.min(dim=0).values).norm().detach()
    )
    authored = torch.cat(
        tuple(
            structure.vertices.to(device=vertices.device, dtype=vertices.dtype)
            for structure in structures
        ),
        dim=0,
    )
    montecarlo = float(
        (authored.amax(dim=0) - authored.amin(dim=0)).norm().detach()
    )
    return enumerated, montecarlo


def _face_counts(structures: tuple[_RuntimeStructure, ...]) -> torch.Tensor:
    return torch.tensor(
        [int(structure.faces.shape[0]) for structure in structures], dtype=torch.int64
    )


def _face_id_columns(
    structures: tuple[_RuntimeStructure, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-face structure, surface and stable primitive IDs, built as whole columns.

    A face's default primitive ID is ``structure_id << 32 | authored local index``;
    authored ``primitive_ids`` replace it, and dropped faces keep their authored
    local index through ``kept_face_index``.
    """

    counts = _face_counts(structures)
    structure_ids = torch.tensor(
        [structure.structure_id for structure in structures], dtype=torch.int64
    )
    surface_ids = torch.tensor(
        [int(structure.source.surface_id) for structure in structures], dtype=torch.int64
    )
    face_structure_id = structure_ids.repeat_interleave(counts)
    face_surface_id = surface_ids.repeat_interleave(counts)
    offsets = torch.cumsum(counts, dim=0) - counts
    total = int(counts.sum())
    local = torch.arange(total, dtype=torch.int64) - offsets.repeat_interleave(counts)
    primitive = (face_structure_id << 32) | local
    for structure, offset, count in zip(structures, offsets.tolist(), counts.tolist()):
        authored = structure.source.primitive_ids
        kept = structure.kept_face_index
        if authored is None and kept is None:
            continue
        if authored is None:
            values = (int(structure.structure_id) << 32) | kept
        else:
            values = torch.tensor(authored, dtype=torch.int64)
            if kept is not None:
                values = values.index_select(0, kept)
        primitive[offset : offset + count] = values
    return face_structure_id, face_surface_id, primitive


def _compile_geometry(
    structures: tuple[_RuntimeStructure, ...], version: int, *, rayd: RayDSceneResource,
) -> GeometryStore:
    if not structures:
        empty_vertices = torch.empty((0, 3), dtype=torch.float32)
        empty_faces = torch.empty((0, 3), dtype=torch.int32)
        empty_edges = torch.empty((0, 2), dtype=torch.int32)
        empty_ids = torch.empty((0,), dtype=torch.int64)
        return GeometryStore(
            vertices=empty_vertices,
            faces=empty_faces,
            face_normals=empty_vertices,
            edges=empty_edges,
            edge_adj_faces=empty_edges,
            edge_param_range=torch.empty((0, 2), dtype=torch.float32),
            face_structure_id=empty_ids,
            face_surface_id=empty_ids,
            face_primitive_id=empty_ids,
            structure_uv_presence=(),
            version=version,
        )

    records = rayd.edge_records()
    structure_ids, surface_ids, primitive_ids = _face_id_columns(structures)
    return GeometryStore(
        vertices=records.vertices,
        faces=records.faces,
        face_normals=records.face_normals,
        edges=topology_kernels.core_pack_int2(records.edge_v0, records.edge_v1),
        edge_adj_faces=topology_kernels.core_pack_int2(records.face0, records.face1),
        edge_param_range=bdpt_zero_matrix(
            records.vertices, rows=records.edge_v0.shape[0], cols=2
        ),
        face_structure_id=structure_ids,
        face_surface_id=surface_ids,
        face_primitive_id=primitive_ids,
        structure_uv_presence=tuple(
            (structure.uv is not None, structure.face_uv is not None)
            for structure in structures
        ),
        version=version,
    )


def _real(value):
    if isinstance(value, complex):
        return value.real
    if isinstance(value, torch.Tensor) and value.is_complex():
        return value.real
    return value


def _sigma_from_eps(value, frequency_hz: float):
    if isinstance(value, complex):
        imaginary = value.imag
    elif isinstance(value, torch.Tensor) and value.is_complex():
        imaginary = value.imag
    else:
        imaginary = 0.0
    return -imaginary * (
        2.0 * math.pi * frequency_hz * VACUUM_PERMITTIVITY
    )


def _material_record(material: PhysicalMaterial, frequency_hz: float) -> dict[str, object]:
    if material.roughness_back is not None and (
        material.roughness_front is None
        or material.roughness_back is not material.roughness_front
    ):
        raise RuntimeError(
            "Channel ABI v3 cannot encode distinct back-side roughness"
        )
    model_id = (
        PEC_MODEL_ID
        if getattr(material, "conductor_model", None) == "perfect"
        else DIELECTRIC_MODEL_ID
    )
    layers = material.layers or (
        MaterialLayer(
            thickness_m=material.thickness_m,
            eps_r=material.eps_r,
            sigma_e=material.sigma_e,
            mu_r=material.mu_r,
            dispersion=material.dispersion,
        ),
    )
    layer_rows: list[tuple[object, object, object, object]] = []
    for layer in layers:
        layer_sample = layer.evaluate_at_frequency(frequency_hz)
        layer_rows.append(
            (
                layer.thickness_m,
                _real(layer_sample.eps_r),
                (
                    PEC_EFFECTIVE_SIGMA_E
                    if model_id == PEC_MODEL_ID
                    else _sigma_from_eps(layer_sample.eps_r, frequency_hz)
                ),
                layer_sample.mu_r,
            )
        )
    first_layer = layer_rows[0]
    roughness = material.roughness_front
    roughness_row: tuple[object, object, object, object]
    if roughness is None:
        roughness_row = (0.0, 0.0, 0.0, 0.0)
    else:
        roughness_row = (
            roughness.rms_height_m,
            roughness.correlation_length_x_m,
            roughness.correlation_length_y_m,
            roughness.principal_axis_rad,
        )
    return {
        # ABI v3 scalar Fresnel view is the first physical layer. The complete
        # stack remains in the CSR payload below.
        "eps_r": first_layer[1],
        "mu_r": first_layer[3],
        "sigma_e": first_layer[2],
        "gain": material.gain,
        "model_id": model_id,
        "thickness_m": first_layer[0],
        "scattering_coefficient": material.scattering_coefficient,
        "xpd_coefficient": material.xpd_coefficient,
        "layers": tuple(layer_rows),
        "roughness": roughness_row,
        "geometry_mode_id": _GEOMETRY_MODE_IDS[material.geometry_mode],
        "scatter_model_id": (
            _SCATTER_MODEL_KIRCHHOFF
            if roughness is not None
            else _SCATTER_MODEL_SMOOTH
        ),
    }


def _unique_materials(
    structures: tuple[_RuntimeStructure, ...],
) -> tuple[tuple[int, PhysicalMaterial], ...]:
    by_id: dict[int, PhysicalMaterial] = {}
    for structure in structures:
        material = structure.material
        if not isinstance(material, PhysicalMaterial):
            raise TypeError(
                "Channel compilation requires witwin.core.PhysicalMaterial"
            )
        existing = by_id.get(structure.material_id)
        if existing is not None and existing is not material:
            raise ValueError("one Core MaterialId must identify one material object")
        by_id[structure.material_id] = material
    return tuple((key, by_id[key]) for key in sorted(by_id))


def _material_cache_token(
    material_version: int, material_ids: tuple[int, ...], frequency_hz: float,
) -> str:
    payload = repr(
        (MATERIAL_ABI_VERSION, material_version, material_ids, frequency_hz.hex())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stack(values: list[object], *, dtype: torch.dtype) -> torch.Tensor:
    tensor_values = [value for value in values if isinstance(value, torch.Tensor)]
    if tensor_values:
        device = tensor_values[0].device
        if any(value.device != device for value in tensor_values[1:]):
            raise ValueError("compiled material tensor leaves must share one device")
        rows = [
            (
                value.to(dtype=dtype)
                if isinstance(value, torch.Tensor)
                else torch.tensor(value, dtype=dtype, device=device)
            )
            for value in values
        ]
        return torch.stack(rows).contiguous()
    return torch.tensor(values, dtype=dtype)


def _compile_materials(
    materials: tuple[tuple[int, PhysicalMaterial], ...], frequency_hz: float, version: int,
) -> MaterialStore:
    if not materials:
        materials = ((0, PhysicalMaterial(material_id=0, name="vacuum")),)
    material_ids = tuple(material_id for material_id, _ in materials)
    material_specs = tuple(material for _, material in materials)
    records = tuple(
        _material_record(material, frequency_hz) for material in material_specs
    )
    layer_rows = [row for record in records for row in record["layers"]]
    offsets: list[int] = []
    counts: list[int] = []
    offset = 0
    for record in records:
        rows = record["layers"]
        offsets.append(offset)
        counts.append(len(rows))
        offset += len(rows)

    def column(name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return _stack([record[name] for record in records], dtype=dtype)

    def layer_column(index: int) -> torch.Tensor:
        return _stack([row[index] for row in layer_rows], dtype=torch.float32)

    def rough_column(index: int) -> torch.Tensor:
        return _stack(
            [record["roughness"][index] for record in records],
            dtype=torch.float32,
        )

    return MaterialStore(
        material_id=torch.tensor(material_ids, dtype=torch.int64),
        eps_r=column("eps_r"),
        mu_r=column("mu_r"),
        sigma_e=column("sigma_e"),
        gain=column("gain"),
        model_id=column("model_id", torch.int32),
        thickness_m=column("thickness_m"),
        scattering_coefficient=column("scattering_coefficient"),
        xpd_coefficient=column("xpd_coefficient"),
        layer_offset=torch.tensor(offsets, dtype=torch.int32),
        layer_count=torch.tensor(counts, dtype=torch.int32),
        layer_thickness_m=layer_column(0),
        layer_eps_r=layer_column(1),
        layer_sigma_e=layer_column(2),
        layer_mu_r=layer_column(3),
        rough_sigma_h_m=rough_column(0),
        rough_corr_x_m=rough_column(1),
        rough_corr_y_m=rough_column(2),
        rough_axis_rad=rough_column(3),
        geometry_mode_id=column("geometry_mode_id", torch.int32),
        scatter_model_id=column("scatter_model_id", torch.int32),
        material_keys=tuple(
            f"material:{material_id}:{material.name or 'unnamed'}"
            for material_id, material in zip(material_ids, material_specs)
        ),
        frequency_hz=frequency_hz,
        abi_version=MATERIAL_ABI_VERSION,
        cache_token=_material_cache_token(version, material_ids, frequency_hz),
        version=version,
        frequency_dependent=tuple(
            f"material:{material_id}:{material.name or 'unnamed'}"
            for material_id, material in zip(material_ids, material_specs)
            if material.capabilities().dispersive
        ),
    )


def _compile_assignments(
    structures: tuple[_RuntimeStructure, ...], *, material_row_by_id: dict[int, int],
    geometry: GeometryStore, version: int,
) -> AssignmentStore:
    phase_screens: dict[int, PhaseScreen] = {}
    for structure_index, structure in enumerate(structures):
        if structure.phase_screen is not None:
            phase_screens[structure_index] = structure.phase_screen
    structure_rows = [
        material_row_by_id[structure.material_id] for structure in structures
    ]
    face_rows_tensor = (
        torch.tensor(structure_rows, dtype=torch.int32).repeat_interleave(
            _face_counts(structures)
        )
        if structures
        else torch.empty((0,), dtype=torch.int32)
    )
    adjacency = geometry.edge_adj_faces
    edge_face_rows = face_rows_tensor.to(device=adjacency.device)
    if adjacency.shape[0]:
        face0 = adjacency[:, 0].to(dtype=torch.int64)
        face1 = adjacency[:, 1].to(dtype=torch.int64)
        edge_material_id0 = edge_face_rows.index_select(0, face0.clamp_min(0))
        edge_material_id1 = torch.where(
            face1 >= 0,
            edge_face_rows.index_select(0, face1.clamp_min(0)),
            edge_material_id0,
        )
    else:
        edge_material_id0 = torch.empty(
            (0,), dtype=torch.int32, device=adjacency.device
        )
        edge_material_id1 = torch.empty(
            (0,), dtype=torch.int32, device=adjacency.device
        )
    return AssignmentStore(
        assignment_id=torch.tensor(
            [structure.assignment_id for structure in structures],
            dtype=torch.int64,
        ),
        material_id=torch.tensor(
            [structure.material_id for structure in structures],
            dtype=torch.int64,
        ),
        structure_id=torch.tensor(
            [structure.structure_id for structure in structures],
            dtype=torch.int64,
        ),
        surface_id=torch.tensor(
            [int(structure.source.surface_id) for structure in structures],
            dtype=torch.int64,
        ),
        face_material_id=face_rows_tensor,
        edge_material_id0=edge_material_id0,
        edge_material_id1=edge_material_id1,
        surface_material_id=torch.tensor(structure_rows, dtype=torch.int32),
        structure_material_id=torch.tensor(structure_rows, dtype=torch.int32),
        num_faces=int(geometry.faces.shape[0]),
        num_edges=int(geometry.edges.shape[0]),
        version=version,
        structure_phase_screens=phase_screens,
    )


def _participates_in_ad(value: torch.Tensor) -> bool:
    if value.requires_grad or value.grad_fn is not None:
        return True
    return torch.autograd.forward_ad.unpack_dual(value).tangent is not None


def _reusable_structures(structures: tuple[_RuntimeStructure, ...]) -> bool:
    """Whether runtime structures may serve another compile (no autograd graph to share)."""

    return not any(_participates_in_ad(structure.vertices) for structure in structures)


def _structure_donors(
    candidates: tuple[CompiledScene, ...],
) -> dict[tuple[object, ...], _RuntimeStructure]:
    """Graph-free runtime structures of cached runtimes, keyed by geometry object and state."""

    donors: dict[tuple[object, ...], _RuntimeStructure] = {}
    for candidate in candidates:
        world = candidate._world
        if world is None:
            continue
        for structure, topology, geometry in zip(
            candidate.structures, world.topology_rows, world.geometry_rows
        ):
            if geometry[2] is None and geometry[3] is None and not _participates_in_ad(
                structure.vertices
            ):
                donors.setdefault((id(structure.source.geometry), topology, geometry), structure)
    return donors


def _same_faces(left: _RuntimeStructure, right: _RuntimeStructure) -> bool:
    if left.vertices.shape != right.vertices.shape or left.faces.shape != right.faces.shape:
        return False
    if (left.kept_face_index is None) != (right.kept_face_index is None):
        return False
    if left.kept_face_index is not None and not torch.equal(
        left.kept_face_index, right.kept_face_index
    ):
        return False
    if left.faces is right.faces:
        return True
    if left.faces.device != right.faces.device:
        return False
    return bool(torch.equal(left.faces, right.faces))


def _changed_meshes(
    donor: CompiledScene, state: _WorldState, structures: tuple[_RuntimeStructure, ...],
) -> list[int] | None:
    """Indices of the meshes whose vertices moved, or None when an in-place refit cannot apply."""

    world = donor._world
    if (
        world is None
        or not donor.rayd.available
        or len(donor.structures) != len(structures)
        or world.layout_key() != state.layout_key()
    ):
        return None
    changed: list[int] = []
    for index, (old, new) in enumerate(zip(donor.structures, structures)):
        if world.geometry_rows[index] == state.geometry_rows[index]:
            continue
        if not _same_faces(old, new):
            return None
        changed.append(index)
    return changed


def _refit_or_build(
    structures: tuple[_RuntimeStructure, ...], state: _WorldState,
    candidates: tuple[CompiledScene, ...],
) -> RayDSceneResource:
    """Refit a cached RayD scene in place when only vertices moved; otherwise build one.

    A refit candidate holds the very same geometry objects with the same faces
    and UVs, so every runtime that shares its RayD scene was compiled from a
    world that has since been edited in place and is dropped from the cache. A
    mesh is created refittable the first time its vertices change without a
    topology change: that compile rebuilds the scene anyway, and later vertex
    changes of that mesh then refit instead of rebuilding.
    """

    for donor in candidates:
        changed = _changed_meshes(donor, state, structures)
        if not changed:
            continue
        rayd = donor.rayd
        if all(rayd.dynamic_meshes[index] for index in changed):
            _REGISTRY.drop_rayd(rayd)
            rayd.refit(changed, [structures[index].vertices for index in changed])
            return rayd
        dynamic = list(rayd.dynamic_meshes)
        for index in changed:
            dynamic[index] = True
        return build_scene_from_structures(structures, dynamic=dynamic)
    return build_scene_from_structures(structures)


def compile(
    scene_or_snapshot: Scene | SceneSnapshot, *, reference_frequency_hz: float | torch.Tensor,
) -> CompiledScene:
    """Build or reuse the sole Channel runtime for a Core world contract."""

    if not isinstance(scene_or_snapshot, (Scene, SceneSnapshot)):
        raise TypeError("scene_or_snapshot must be a witwin.core Scene or SceneSnapshot")
    frequency_value = _frequency_value(reference_frequency_hz)
    state = world_state(scene_or_snapshot)
    trace_backend = requested_trace_backend()
    runtime_device = torch.cuda.current_device() if torch.cuda.is_available() else "no-cuda"
    key = (
        id(scene_or_snapshot),
        _frequency_identity(reference_frequency_hz, frequency_value),
        runtime_device,
        trace_backend,
        MATERIAL_ABI_VERSION,
    )
    cached = _REGISTRY.get(key, scene_or_snapshot, state)
    if cached is not None:
        return cached

    topology_version, geometry_version, material_version, assignment_version = (
        _readable_versions(scene_or_snapshot)
    )
    reusable = tuple(
        candidate
        for candidate in _REGISTRY.candidates(runtime_device)
        if candidate.rayd.requested_trace_backend == trace_backend
    )
    rayd_key = state.rayd_key()
    donor = next(
        (
            candidate
            for candidate in reusable
            if candidate._world is not None and candidate._world.rayd_key() == rayd_key
        ),
        None,
    )
    if donor is not None:
        rayd = donor.rayd
        geometry = donor.geometry
        diagonals = (
            donor.enumerated_penetration_scene_diagonal_m,
            donor.montecarlo_penetration_scene_diagonal_m,
        )
        structures = (
            _rebound_structures(donor.structures, state)
            if _reusable_structures(donor.structures)
            else _runtime_structures(scene_or_snapshot, state, {})
        )
    else:
        structures = _runtime_structures(
            scene_or_snapshot, state, _structure_donors(reusable)
        )
        if structures and torch.cuda.is_available():
            rayd = _refit_or_build(structures, state, reusable)
        else:
            rayd = build_scene_from_structures(structures)
        if "edge_policy" not in rayd.runtime_cache:
            from witwin.channel.scene.resources import resolve_scene_edge_policy

            rayd.runtime_cache["edge_policy"] = resolve_scene_edge_policy(scene_or_snapshot)
        geometry = _compile_geometry(structures, geometry_version, rayd=rayd)
        diagonals = _compile_penetration_scene_diagonals(structures, rayd=rayd)

    material_key = state.material_key()
    materials = next(
        (
            candidate.materials
            for candidate in reusable
            if candidate._world is not None
            and candidate.materials.frequency_hz == frequency_value
            and candidate._world.material_key() == material_key
        ),
        None,
    )
    if materials is None:
        materials = _compile_materials(
            _unique_materials(structures), frequency_value, material_version
        )
    assignment_key = state.assignment_key()
    assignments = next(
        (
            candidate.assignments
            for candidate in reusable
            if candidate.geometry is geometry
            and candidate._world is not None
            and candidate._world.assignment_key() == assignment_key
        ),
        None,
    )
    if assignments is None:
        material_row_by_id = {
            int(material_id): row
            for row, material_id in enumerate(materials.material_id.tolist())
        }
        assignments = _compile_assignments(
            structures,
            material_row_by_id=material_row_by_id,
            geometry=geometry,
            version=assignment_version,
        )

    compiled = CompiledScene(
        source=scene_or_snapshot,
        structures=structures,
        geometry=geometry,
        materials=materials,
        assignments=assignments,
        rayd=rayd,
        reference_frequency_hz=(
            reference_frequency_hz
            if isinstance(reference_frequency_hz, torch.Tensor)
            else frequency_value
        ),
        reference_frequency_revision=(
            int(reference_frequency_hz._version)
            if isinstance(reference_frequency_hz, torch.Tensor)
            else None
        ),
        topology_version=topology_version,
        geometry_version=geometry_version,
        material_version=material_version,
        assignment_version=assignment_version,
        # A SceneSnapshot carries the world instant it was taken at; a plain
        # Scene has no time. Recorded verbatim, so a tensor time keeps its
        # identity and never costs a host read.
        time_s=getattr(scene_or_snapshot, "time_s", None),
        enumerated_penetration_scene_diagonal_m=diagonals[0],
        montecarlo_penetration_scene_diagonal_m=diagonals[1],
        _world=state,
        _rayd_generation=rayd.generation,
    )
    _REGISTRY.put(key, compiled)
    return compiled


def record_dropped_faces(result: _ResultT, compiled: CompiledScene) -> _ResultT:
    """Add ``metadata["dropped_faces"]`` to a solver result when compile removed faces."""

    dropped = compiled.dropped_faces
    if any(dropped.values()):
        result.metadata["dropped_faces"] = dict(dropped)  # type: ignore[attr-defined]
    return result


LIGHT_SPEED_M_PER_S = 299_792_458.0


def _frequency_scalar(scene: SolverScene) -> float:
    """Detached scalar carrier for non-differentiable consumers.

    Topology discovery and metadata never differentiate with respect to
    frequency (fixed-topology contract), so detach before float to keep AD
    solves with a requires_grad tensor frequency warning-free. The field
    evaluation seam must NOT use this helper: it forwards the live tensor so
    the frequency stays on the autograd graph.
    """

    frequency = scene.frequency
    if isinstance(frequency, torch.Tensor):
        return float(frequency.detach())
    return float(frequency)


def receiver_grid_points(grid: ReceiverGrid, *, reference: torch.Tensor) -> torch.Tensor:
    return mc_receiver_grid_points(
        reference,
        origin=vector3_tuple(grid.origin),
        x_axis=vector3_tuple(grid.x_axis),
        y_axis=vector3_tuple(grid.y_axis),
        shape=grid.shape,
        spacing=grid.spacing,
    )


def host_vec3_tensor(flat_positions: tuple[float, ...]) -> torch.Tensor:
    powers = tuple(1.0 for _ in range(len(flat_positions) // 3))
    return mc_transmitter_tensors(flat_positions, powers)["positions"]


def receiver_positions(
    scene: object, *, device: torch.device, reference: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        len(scene.receivers) == 1
        and isinstance(scene.receivers[0], ReceiverGrid)
        and reference is not None
    ):
        return receiver_grid_points(scene.receivers[0], reference=reference)
    blocks: list[torch.Tensor] = []
    grid_reference = reference

    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            block = receiver.position.reshape(1, 3).to(device=device)
            blocks.append(block)
            if grid_reference is None:
                grid_reference = block
        elif isinstance(receiver, ReceiverGrid):
            if grid_reference is None:
                grid_reference = host_vec3_tensor(())
            blocks.append(receiver_grid_points(receiver, reference=grid_reference))
        else:
            raise TypeError(f"receiver type is not accepted: {type(receiver)!r}")
    if not blocks:
        return host_vec3_tensor(())
    if len(blocks) == 1:
        return blocks[0]
    return topology_kernels.path_concat_vec3(blocks)


def transmitter_positions(
    scene: object, *, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not scene.transmitters:
        exported = mc_transmitter_tensors((), ())
        return exported["positions"], exported["power"]
    positions = torch.stack(
        tuple(transmitter.position for transmitter in scene.transmitters)
    ).to(device=device, dtype=torch.float32)
    powers = torch.stack(
        tuple(
            power.reshape(())
            if isinstance((power := transmitter.power_w), torch.Tensor)
            else positions.new_tensor(float(power))
            for transmitter in scene.transmitters
        )
    ).to(device=device, dtype=torch.float32)
    return positions, powers


def transmitter_polarizations_as_stored(scene: object, *, device: torch.device) -> torch.Tensor:
    """Return transmitter polarizations with the scene's stored dtype and layout."""

    if not scene.transmitters:
        return host_vec3_tensor(())
    return torch.stack(
        tuple(transmitter.polarization for transmitter in scene.transmitters)
    ).to(
        device=device
    )


__all__ = ["clear_compile_cache", "compile"]
