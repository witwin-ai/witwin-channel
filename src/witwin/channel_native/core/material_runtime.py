from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scene import Scene


def _material_parameters(material) -> dict[str, float | int]:
    if hasattr(material, "parameters"):
        return material.parameters()
    return {
        "eps_r": 1.0,
        "mu_r": 1.0,
        "sigma_e": 0.0,
        "gain": 1.0,
        "model_id": 0,
    }


def face_material_tensors(
    scene: "Scene",
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eps_r: list[torch.Tensor] = []
    sigma_e: list[torch.Tensor] = []
    mu_r: list[torch.Tensor] = []
    gain: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    for structure in scene.structures:
        face_count = int(structure.faces.shape[0])
        params = _material_parameters(structure.material)
        eps_r.append(torch.full((face_count,), float(params.get("eps_r", 1.0)), device=device, dtype=torch.float32))
        sigma_e.append(
            torch.full((face_count,), float(params.get("sigma_e", 0.0)), device=device, dtype=torch.float32)
        )
        mu_r.append(torch.full((face_count,), float(params.get("mu_r", 1.0)), device=device, dtype=torch.float32))
        gain.append(torch.full((face_count,), float(params.get("gain", 1.0)), device=device, dtype=torch.float32))
        valid.append(torch.ones((face_count,), device=device, dtype=torch.bool))
    if not eps_r:
        return (
            torch.ones((1,), device=device, dtype=torch.float32),
            torch.zeros((1,), device=device, dtype=torch.float32),
            torch.ones((1,), device=device, dtype=torch.float32),
            torch.ones((1,), device=device, dtype=torch.float32),
            torch.ones((1,), device=device, dtype=torch.bool),
        )
    return (
        torch.cat(eps_r).contiguous(),
        torch.cat(sigma_e).contiguous(),
        torch.cat(mu_r).contiguous(),
        torch.cat(gain).contiguous(),
        torch.cat(valid).contiguous(),
    )
