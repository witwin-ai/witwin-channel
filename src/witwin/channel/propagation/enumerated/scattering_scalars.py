"""AD-scalar construction for deterministic scattering rows (ADR-014/015).

Frequency-dependent radiometric scalars used by the ensemble and realization
scattering paths. In AD mode these become Torch scalars so frequency gradients
flow through the ensemble radiometric ``coef`` and the realization ``k0`` /
amplitude scale; in the primal path the callers keep their plain Python-float
scalars, so this module carries only the differentiable branch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from witwin.channel.physics.conventions import C0

if TYPE_CHECKING:
    from witwin.channel.scene.models import Scene


def frequency_tensor(scene: Scene, device: torch.device) -> torch.Tensor:
    """Scene carrier frequency as a 0-d float32 CUDA tensor for AD scalars.

    A ``requires_grad`` scene frequency keeps its autograd graph so frequency
    gradients flow through the radiometric ``coef`` / ``k0`` scalars; a plain
    Python-float frequency becomes a constant scalar tensor.
    """

    frequency = scene.frequency
    if isinstance(frequency, torch.Tensor):
        return frequency.to(device=device, dtype=torch.float32)
    return torch.tensor(float(frequency), device=device, dtype=torch.float32)


def ensemble_coef_scale(
    scene: Scene, device: torch.device, *, ad_enabled: bool
) -> torch.Tensor | None:
    """AD radiometric ``coef`` scale for ensemble rows, or ``None`` when AD is off.

    ADR-014: the radiometric scale becomes a Torch scalar so frequency gradients
    flow through the ensemble rows (their only frequency dependence). Ensemble
    rows are zero-phase power rows, so nothing else is differentiable w.r.t.
    frequency.
    """

    if not ad_enabled:
        return None
    return (C0 / frequency_tensor(scene, device)) ** 2 / (4.0 * math.pi) ** 2


def realization_scalars(
    scene: Scene, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AD ``(frequency_t, k0_t, amplitude_scale_t)`` for realization rows.

    ADR-014: ``k0`` and the outer amplitude scale become Torch scalars so
    frequency gradients flow through the coherent phase, the Kirchhoff prefactor
    and the radiometric normalization; ``frequency_t`` also threads into the
    differentiable EM layer stack.
    """

    frequency_t = frequency_tensor(scene, device)
    k0_t = 2.0 * math.pi * frequency_t / C0
    amplitude_scale_t = (C0 / frequency_t) / (4.0 * math.pi)
    return frequency_t, k0_t, amplitude_scale_t
