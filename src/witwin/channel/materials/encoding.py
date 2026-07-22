from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any

from .kernels.functional import mc_face_material_tensors
from .models import PEC_EFFECTIVE_SIGMA_E, PEC_MODEL_ID

if TYPE_CHECKING:
    Scene = CompiledScene = Any


def face_material_tensors(
    scene_or_compiled: "Scene | CompiledScene",
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    compiled = (
        scene_or_compiled
        if hasattr(scene_or_compiled, "materials")
        else scene_or_compiled.compile()
    )
    materials = compiled.materials
    assignments = compiled.assignments
    material_eps_r = materials.eps_r.to(device=device, dtype=torch.float32).contiguous()
    material_sigma_e = materials.sigma_e.to(device=device, dtype=torch.float32)
    # The Fresnel kernels only see (eps_r, sigma_e, mu_r); realize the PEC
    # limit through an effective conductivity.
    material_sigma_e = torch.where(
        materials.model_id.to(device=material_sigma_e.device) == PEC_MODEL_ID,
        material_sigma_e.clamp_min(PEC_EFFECTIVE_SIGMA_E),
        material_sigma_e,
    ).contiguous()
    material_mu_r = materials.mu_r.to(device=device, dtype=torch.float32).contiguous()
    face_material_id = assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    exported = mc_face_material_tensors(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id,
    )
    return (
        exported["eps_r"],
        exported["sigma_e"],
        exported["mu_r"],
        exported["gain"],
        exported["valid"],
    )


def face_material_thickness(
    scene_or_compiled: "Scene | CompiledScene",
    *,
    device: torch.device,
) -> torch.Tensor:
    """Expand Sionna/ITU slab thickness to the global face layout."""

    compiled = (
        scene_or_compiled
        if hasattr(scene_or_compiled, "materials")
        else scene_or_compiled.compile()
    )
    material_thickness = compiled.materials.thickness_m.to(
        device=device,
        dtype=torch.float32,
    )
    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    )
    return material_thickness.index_select(0, face_material_id).contiguous()


def face_material_field_bundle(
    scene_or_compiled: "Scene | CompiledScene", *, device: torch.device
) -> dict[str, torch.Tensor]:
    """Return the complete per-face finite-slab field operator inputs."""

    compiled = (
        scene_or_compiled
        if hasattr(scene_or_compiled, "materials")
        else scene_or_compiled.compile()
    )
    materials = compiled.materials
    material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    )
    sigma_e = materials.sigma_e.to(device=device, dtype=torch.float32)
    sigma_e = torch.where(
        materials.model_id.to(device=device) == PEC_MODEL_ID,
        sigma_e.clamp_min(PEC_EFFECTIVE_SIGMA_E),
        sigma_e,
    )

    def per_face(values: torch.Tensor) -> torch.Tensor:
        return (
            values.to(device=device, dtype=torch.float32)
            .index_select(0, material_id)
            .contiguous()
        )

    def material_level(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return values.to(device=device, dtype=dtype).contiguous()

    return {
        "eps_r": per_face(materials.eps_r),
        "sigma_e": per_face(sigma_e),
        "mu_r": per_face(materials.mu_r),
        "gain": per_face(materials.gain),
        "thickness": per_face(materials.thickness_m),
        "material_id": material_id.to(dtype=torch.int32).contiguous(),
        "model_id": materials.model_id.to(device=device, dtype=torch.int32)
        .index_select(0, material_id)
        .contiguous(),
        "valid": torch.ones((material_id.shape[0],), device=device, dtype=torch.bool),
        # ABI v3 material-level views: faces index the CSR via material_id,
        # layers are never expanded per face.
        "layer_offset": material_level(materials.layer_offset, torch.int32),
        "layer_count": material_level(materials.layer_count, torch.int32),
        "layer_thickness_m": material_level(materials.layer_thickness_m, torch.float32),
        "layer_eps_r": material_level(materials.layer_eps_r, torch.float32),
        "layer_sigma_e": material_level(materials.layer_sigma_e, torch.float32),
        "layer_mu_r": material_level(materials.layer_mu_r, torch.float32),
        "rough_sigma_h_m": material_level(materials.rough_sigma_h_m, torch.float32),
        "rough_corr_x_m": material_level(materials.rough_corr_x_m, torch.float32),
        "rough_corr_y_m": material_level(materials.rough_corr_y_m, torch.float32),
        "rough_axis_rad": material_level(materials.rough_axis_rad, torch.float32),
        "geometry_mode_id": material_level(materials.geometry_mode_id, torch.int32),
        "scatter_model_id": material_level(materials.scatter_model_id, torch.int32),
    }
