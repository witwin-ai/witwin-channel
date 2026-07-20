from __future__ import annotations

from dataclasses import dataclass, field

import torch

from witwin.channel_native.scattering.tables import KirchhoffTable
from witwin.channel_native.scene.kernels.rayd_scene import RayDSceneResource
from witwin.channel_native.scene.scattering_resources import (
    KirchhoffRuntimeResources,
    PhaseScreenResourceKey,
    PhaseScreenRuntimeResources,
    RoughMaterialRuntime,
    ScatteringResourceKey,
    build_kirchhoff_resources,
    build_phase_screen_resources,
)
from witwin.channel_native.scene.stores.assignments import AssignmentStore
from witwin.channel_native.scene.stores.geometry import GeometryStore
from witwin.channel_native.scene.stores.materials import MaterialStore


@dataclass(slots=True)
class CompiledScene:
    geometry: GeometryStore
    materials: MaterialStore
    assignments: AssignmentStore
    rayd: RayDSceneResource
    geometry_version: int
    material_version: int
    assignment_version: int
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


CompiledScene.__module__ = "witwin.channel_native.core.runtime.compiled_scene"
