from __future__ import annotations

from dataclasses import dataclass, field

import torch

from witwin.channel_native.scattering.phase_screen import PhaseScreenRuntime
from witwin.channel_native.scattering.tables import KirchhoffTable
from witwin.channel_native.scene.kernels.rayd_scene import RayDNScene
from witwin.channel_native.scene.scattering_resources import (
    KirchhoffRuntimeResources,
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
    raydn: RayDNScene
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
        """Phase-screen runtimes per structure index (lazy, GPU textures)."""

        key = self._scattering_resource_key()
        if (
            self._phase_screen_resources_cache is None
            or self._phase_screen_resources_cache.key != key
        ):
            self._phase_screen_resources_cache = build_phase_screen_resources(
                self.assignments, key
            )
        return self._phase_screen_resources_cache

    @property
    def kirchhoff_tables(self) -> dict[int, KirchhoffTable]:
        """Compatibility view of the typed Kirchhoff table resources."""

        return self.kirchhoff_resources.tables

    @property
    def phase_screen_runtimes(self) -> dict[int, PhaseScreenRuntime]:
        """Compatibility view of the typed phase-screen resources."""

        return self.phase_screen_resources.runtimes

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


CompiledScene.__module__ = "witwin.channel_native.core.runtime.compiled_scene"
