"""Reference Torch rough-reflection C_r factor and application (ADR-010 op 3).

The previous production implementation of
``propagation/fields/evaluation.py::_rough_reflection_factor`` plus the
Python-side application of the real factor onto the four reflection field
outputs. Test-only: MUST NOT be imported from production packages.
"""

from __future__ import annotations

import math

import torch

_C0 = 299792458.0


def rough_reflection_factor(
    positions: torch.Tensor,
    normals: torch.Tensor,
    source: torch.Tensor,
    sigma_b: torch.Tensor,
    rough_b: torch.Tensor,
    replaced: torch.Tensor,
    frequency_hz: float | torch.Tensor,
) -> torch.Tensor:
    """C_r = prod_b exp(-2*(k0*cos_b*sigma_b)^2) on rough bounces (1 else).

    ``cos_b = |dot(seg_dir_b, n_b)|`` with ``seg_dir_b`` the unit direction of
    the incoming segment (``pos_b - prev_b``, ``prev_0 = source``). Rows flagged
    ``replaced`` are zeroed (contract 6.7.3). Mirrors the removed production
    expression order exactly.
    """

    depth = positions.shape[1]
    prev = torch.cat((source.unsqueeze(1), positions[:, : depth - 1]), dim=1)
    seg = positions - prev
    seg_dir = seg / torch.linalg.vector_norm(seg, dim=-1, keepdim=True).clamp_min(
        1.0e-9
    )
    cos_b = (seg_dir * normals).sum(-1).abs()
    k0 = 2.0 * math.pi * frequency_hz / _C0
    attenuation = torch.exp(-2.0 * (k0 * cos_b * sigma_b).square())
    c_r = torch.where(rough_b, attenuation, torch.ones_like(attenuation))
    factor = c_r.prod(dim=1)
    factor = torch.where(replaced, torch.zeros_like(factor), factor)
    return factor


def rough_reflection_scale(
    field_vector: torch.Tensor,
    coefficient: torch.Tensor,
    path_field: torch.Tensor,
    path_gain: torch.Tensor,
    positions: torch.Tensor,
    normals: torch.Tensor,
    source: torch.Tensor,
    sigma_b: torch.Tensor,
    rough_b: torch.Tensor,
    replaced: torch.Tensor,
    frequency_hz: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply the real C_r factor onto the four reflection field outputs."""

    factor = rough_reflection_factor(
        positions, normals, source, sigma_b, rough_b, replaced, frequency_hz
    )
    scale = factor.to(torch.float32)
    return {
        "field_vector": field_vector * scale[:, None],
        "coefficient": coefficient * scale,
        "path_field": path_field * scale,
        "path_gain": path_gain * scale.square(),
        "factor": factor,
    }
