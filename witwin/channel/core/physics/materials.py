"""Shared scene-material resolution for radiomap solvers.

Both ``witwin.channel.deterministic`` and ``witwin.channel.montecarlo`` resolve per-triangle
materials through the same Scene contract (``Scene.triangle_material`` +
``Scene._triangle_runtime``). This module centralizes that lookup, plus the
so the solvers do not duplicate table-lookup logic.

Solver-specific glue stays inside the owning solver package.
"""

from __future__ import annotations

from dataclasses import dataclass

import drjit as dr
from witwin.channel import types as wt


@dataclass(slots=True)
class FaceMaterial:
    """Per-face Fresnel material parameters for a diffraction wedge."""
    eta_r: object
    sigma: object
    gain: object
    use_fresnel: object
    mu_r: object = 1.0


def scene_has_material_table(scene) -> bool:
    """True when ``scene`` exposes a triangle material runtime table."""
    return scene is not None and scene._triangle_runtime() is not None


def resolve_surface_material(
    *,
    scene,
    prim_idx,
    default_gain: float,
    valid_mask=None,
) -> FaceMaterial:
    """Resolve per-triangle Fresnel material parameters.

    Materials must come from the scene's triangle material table. Solver-local
    material fallback is intentionally not supported.
    """
    prim_idx_i32 = wt.Int32(prim_idx)
    width = int(dr.width(prim_idx_i32))
    if valid_mask is None:
        valid_mask = dr.full(wt.Bool, True, width)

    if not scene_has_material_table(scene):
        raise RuntimeError(
            "Surface material resolution requires a scene material table. "
            "Attach witwin.core.Material to every scene structure."
        )

    triangle_material = scene.triangle_material(prim_idx_i32, valid_mask=valid_mask)
    return FaceMaterial(
        eta_r=triangle_material["eps_r"],
        sigma=triangle_material["sigma_e"],
        gain=dr.full(wt.Float, float(default_gain), width),
        use_fresnel=triangle_material["valid"],
        mu_r=triangle_material["mu_r"],
    )


__all__ = [
    "FaceMaterial",
    "resolve_surface_material",
    "scene_has_material_table",
]
