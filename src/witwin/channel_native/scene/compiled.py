from __future__ import annotations

from dataclasses import dataclass, field

import torch

from witwin.channel_native.scene.stores.assignments import AssignmentStore
from witwin.channel_native.scene.stores.geometry import GeometryStore
from witwin.channel_native.scene.stores.materials import MaterialStore
from witwin.channel_native.scene.kernels.rayd_scene import RayDNScene


@dataclass(slots=True)
class CompiledScene:
    geometry: GeometryStore
    materials: MaterialStore
    assignments: AssignmentStore
    raydn: RayDNScene
    workspace: object | None
    geometry_version: int
    material_version: int
    assignment_version: int
    # Lazy scattering caches (built on first access so smooth scenes pay no
    # compile cost). A CompiledScene instance is already cache-keyed by the
    # material cache_token in Scene.compile, so instance-level caching is
    # consistent with the material cache.
    _kirchhoff_tables_cache: dict[int, object] | None = field(
        default=None, repr=False, compare=False
    )
    _phase_screen_runtimes_cache: dict[int, object] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def kirchhoff_tables(self) -> dict[int, object]:
        """Kirchhoff BSDF tables per material index (scatter_model_id == 1).

        Built lazily from the MaterialStore CSR layers and roughness fields
        at the store frequency; raises ``kirchhoff_domain_exceeded`` for
        out-of-domain roughness (never silently degrades).
        """

        if self._kirchhoff_tables_cache is None:
            from witwin.channel_native.core.materials import Roughness
            from witwin.channel_native.scattering import build_kirchhoff_table

            device = "cuda" if torch.cuda.is_available() else "cpu"
            store = self.materials
            tables: dict[int, object] = {}
            for index in range(int(store.material_id.numel())):
                if int(store.scatter_model_id[index]) != 1:
                    continue
                offset = int(store.layer_offset[index])
                count = int(store.layer_count[index])
                layers = [
                    (
                        float(store.layer_thickness_m[row]),
                        float(store.layer_eps_r[row]),
                        float(store.layer_sigma_e[row]),
                        float(store.layer_mu_r[row]),
                    )
                    for row in range(offset, offset + count)
                ]
                roughness = Roughness(
                    rms_height_m=float(store.rough_sigma_h_m[index]),
                    corr_length_x_m=float(store.rough_corr_x_m[index]),
                    corr_length_y_m=float(store.rough_corr_y_m[index]),
                    principal_axis_rad=float(store.rough_axis_rad[index]),
                )
                tables[index] = build_kirchhoff_table(
                    roughness, layers, store.frequency_hz, device=device
                )
            self._kirchhoff_tables_cache = tables
        return self._kirchhoff_tables_cache

    @property
    def phase_screen_runtimes(self) -> dict[int, object]:
        """Phase-screen runtimes per structure index (lazy, GPU textures)."""

        if self._phase_screen_runtimes_cache is None:
            from witwin.channel_native.scattering import PhaseScreenRuntime

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._phase_screen_runtimes_cache = {
                index: PhaseScreenRuntime(screen, device=device)
                for index, screen in self.assignments.structure_phase_screens.items()
            }
        return self._phase_screen_runtimes_cache


CompiledScene.__module__ = "witwin.channel_native.core.runtime.compiled_scene"
