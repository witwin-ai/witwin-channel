from __future__ import annotations

import math
from dataclasses import dataclass, field
import torch
from witwin.core import Scene, SceneSnapshot

from witwin.channel.scattering.tables import KirchhoffTable
from witwin.channel.scene.kernels.rayd_scene import RayDSceneResource
from witwin.channel.scene.scattering_resources import (
    KirchhoffRuntimeResources,
    PhaseScreenResourceKey,
    PhaseScreenRuntimeResources,
    RoughMaterialRuntime,
    ScatteringResourceKey,
    build_kirchhoff_resources,
    build_phase_screen_resources,
)
from witwin.channel.scene.stores.assignments import AssignmentStore
from witwin.channel.scene.stores.geometry import GeometryStore
from witwin.channel.scene.stores.materials import MaterialStore


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
    _fixed_reevaluation_tables_cache: dict | None = field(
        default=None, repr=False, compare=False
    )

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
        for name in (
            "enumerated_penetration_scene_diagonal_m",
            "montecarlo_penetration_scene_diagonal_m",
        ):
            value = getattr(self, name)
            if not isinstance(value, float):
                raise TypeError(f"{name} must be a float")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def require_reference_frequency(
        self, reference_frequency_hz: float | torch.Tensor
    ) -> None:
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

    def fixed_reevaluation_tables(self) -> dict[str, object]:
        """Scene-static vertex and material tables for fixed-topology replay.

        Built lazily per instance following the Plan-13 scene-static pattern:
        both tables depend only on this CompiledScene's immutable stores and
        structure tensors, and any world change produces a new instance through
        the version-token compile cache. The cache is bypassed whenever a table
        tensor participates in autograd - a cached graph node would be freed by
        the first backward and fail the second - so differentiable-material or
        differentiable-mesh loops pay the staging exactly as before, while the
        primal per-frame replay (the Radar Doppler shape) stages once.
        """

        if self._fixed_reevaluation_tables_cache is not None:
            return self._fixed_reevaluation_tables_cache

        from witwin.channel.materials.encoding import face_material_field_bundle
        from witwin.channel.scene import ad_geometry

        tables: dict[str, object] = {
            "vertices": ad_geometry.scene_vertex_table(self, self),
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
        if not graph_bearing:
            self._fixed_reevaluation_tables_cache = tables
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
