from __future__ import annotations

import torch


def normalize_vec3(values: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    return values / torch.linalg.vector_norm(
        values, dim=-1, keepdim=True
    ).clamp_min(eps)
