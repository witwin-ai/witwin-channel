"""Shared specular-transmission helpers for the Monte Carlo solvers.

Two evaluation contexts share the per-wall layer-stack algebra (implementation
contract section 4):

- Endpoint connection (``straight_transmission_chains``): the Tx->Rx segment
  is marched straight through every thin_sheet wall it crosses. Each wall is
  evaluated at the straight-line incidence angle and contributes its
  smooth-stack POWER transmittance. Specular transmission never bends the ray
  (parallel-plate exit), so pure-transmission chains are exactly this straight
  segment; the evaluation is exact for vacuum and index-matched walls and a
  documented small-angle-error approximation otherwise.
- Shooting (BDPT light-subpath continuation): handled by the native
  ``bdpt_transmitted_light_subpath_state`` kernel with the exact lateral exit
  offset; this module only supplies the seeded event-selection utilities.
"""

from __future__ import annotations

from typing import Any

import torch

from witwin.channel_native.core.kernels.ops import (
    bdpt_intersect_forward,
    em_layer_stack_eval,
)


_MIN_EPSILON_M = 1.0e-6
_RELATIVE_EPSILON = 1.0e-6
_EVENT_PROBABILITY_FLOOR = 0.05

_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MUL0 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL1 = 0x94D049BB133111EB
_MASK64 = (1 << 64) - 1

_LAYER_CSR_FIELDS = (
    "layer_offset",
    "layer_count",
    "layer_thickness_m",
    "layer_eps_r",
    "layer_sigma_e",
    "layer_mu_r",
)


def layer_csr_view(material_bundle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Material-level CSR layer tensors from a face_material_field_bundle."""

    return {name: material_bundle[name] for name in _LAYER_CSR_FIELDS}


def scene_diagonal_m(scene: Any) -> float:
    """Structure bounding-box diagonal for scale-aware ray epsilons."""

    minimum: torch.Tensor | None = None
    maximum: torch.Tensor | None = None
    for structure in scene.structures:
        vertices = structure.vertices
        low = vertices.amin(dim=0)
        high = vertices.amax(dim=0)
        minimum = low if minimum is None else torch.minimum(minimum, low)
        maximum = high if maximum is None else torch.maximum(maximum, high)
    if minimum is None or maximum is None:
        return 0.0
    return float((maximum - minimum).norm())


def scale_aware_epsilon(
    position: torch.Tensor, *, scene_diagonal: float
) -> torch.Tensor:
    """Per-row restart offset ``max(|p|*1e-6, diag*1e-6, 1e-6 m)`` (contract
    section 4)."""

    floor = max(_MIN_EPSILON_M, float(scene_diagonal) * _RELATIVE_EPSILON)
    return (position.abs().amax(dim=-1) * _RELATIVE_EPSILON).clamp_min(floor)


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_MUL0) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_MUL1) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def event_selection_seed(seed: int, tx_index: int, depth: int) -> int:
    """Deterministic per-(seed, tx, depth) generator seed via the native
    splitmix64 mixing pattern (matches bdpt_subpaths.cu)."""

    mixed = _splitmix64(int(seed))
    mixed = _splitmix64(mixed ^ (((int(tx_index) + 1) * 0xD1B54A32D192ED03) & _MASK64))
    mixed = _splitmix64(mixed ^ (((int(depth) + 1) * 0x8CB92BA72F3D8DD7) & _MASK64))
    return mixed & ((1 << 63) - 1)


def event_uniforms(
    count: int, *, seed: int, tx_index: int, depth: int, device: torch.device
) -> torch.Tensor:
    """Reproducible per-sample uniforms for reflect/transmit event selection."""

    generator = torch.Generator(device=device)
    generator.manual_seed(event_selection_seed(seed, tx_index, depth))
    return torch.rand((int(count),), device=device, generator=generator)


def unpolarized_power_budgets(
    stack: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """(R_eff, T_eff) as the unpolarized TE/TM mean of the smooth-stack power
    budgets. The mean is acceptable for EVENT PROBABILITIES only; the selected
    branch's kernel applies the exact polarized Jones coefficients."""

    r_eff = 0.5 * (stack["cap_R_te"] + stack["cap_R_tm"])
    t_eff = 0.5 * (stack["cap_T_te"] + stack["cap_T_tm"])
    return r_eff, t_eff


def transmission_event_probability(
    r_eff: torch.Tensor,
    t_eff: torch.Tensor,
    *,
    floor: float = _EVENT_PROBABILITY_FLOOR,
) -> torch.Tensor:
    """Event probability p_t = T/(R+T) with a minimum-probability floor when
    both budgets are nonzero (plan section 7.1). Absorption 1-R-T terminates
    implicitly through the field magnitudes; there is no absorption event."""

    total = (r_eff + t_eff).clamp_min(1.0e-12)
    p_t = (t_eff / total).clamp(floor, 1.0 - floor)
    p_t = torch.where(t_eff <= 0.0, torch.zeros_like(p_t), p_t)
    return torch.where(
        (r_eff <= 0.0) & (t_eff > 0.0), torch.ones_like(p_t), p_t
    )


def straight_transmission_chains(
    raydn: Any,
    origins: torch.Tensor,
    targets: torch.Tensor,
    *,
    face_material_id: torch.Tensor,
    layer_csr: dict[str, torch.Tensor],
    frequency_hz: float,
    max_depth: int,
    scene_diagonal: float,
) -> dict[str, torch.Tensor]:
    """March straight origin->target segments through up to ``max_depth``
    thin_sheet walls and accumulate the per-wall power transmittance product.

    Returns per-row tensors: ``transmittance`` (product of unpolarized wall
    power transmittances; zero when blocked), ``wall_count`` (int32
    penetrations), and ``penetrated`` (bool: at least one wall crossed AND the
    target was reached within the depth budget). Rows with more walls than
    ``max_depth`` or an invalid wall material are truthfully blocked
    (transmittance zero), never approximated.
    """

    device = origins.device
    count = int(origins.shape[0])
    handle = raydn.require_handle()
    delta = targets - origins
    distance = delta.norm(dim=-1)
    direction = (delta / distance.clamp_min(_MIN_EPSILON_M)[:, None]).contiguous()
    # Stop marching just short of the target so a receiver sitting on a
    # surface does not count its own support as a wall.
    remaining = (
        distance - scale_aware_epsilon(targets, scene_diagonal=scene_diagonal)
    ).clamp_min(0.0)
    origin = origins.contiguous().clone()
    transmittance = torch.ones((count,), device=device, dtype=torch.float32)
    wall_count = torch.zeros((count,), device=device, dtype=torch.int32)
    blocked = torch.zeros((count,), device=device, dtype=torch.bool)
    active = remaining > 0.0

    for depth in range(int(max_depth) + 1):
        if not bool(active.any()):
            break
        hit = bdpt_intersect_forward(
            handle,
            origin.contiguous(),
            direction,
            remaining.contiguous(),
            active.contiguous(),
        )
        prim = hit["global_prim_id"]
        hit_mask = active & (prim >= 0) & (hit["t"] > 0.0) & (hit["t"] <= remaining)
        if not bool(hit_mask.any()):
            break
        if depth == int(max_depth):
            # Walls remain beyond the penetration budget: truthful blocking.
            blocked |= hit_mask
            break
        rows = torch.nonzero(hit_mask, as_tuple=False).flatten()
        row_prim = prim.index_select(0, rows).to(torch.int64)
        row_material = face_material_id.index_select(0, row_prim)
        material_ok = row_material >= 0
        row_direction = direction.index_select(0, rows)
        cos_theta = (
            (row_direction * hit["n"].index_select(0, rows))
            .sum(dim=-1)
            .abs()
            .clamp(_MIN_EPSILON_M, 1.0)
        )
        stack = em_layer_stack_eval(
            cos_theta.contiguous(),
            row_material.clamp_min(0).contiguous(),
            layer_csr["layer_offset"],
            layer_csr["layer_count"],
            layer_csr["layer_thickness_m"],
            layer_csr["layer_eps_r"],
            layer_csr["layer_sigma_e"],
            layer_csr["layer_mu_r"],
            frequency_hz=float(frequency_hz),
        )
        _r_eff, t_eff = unpolarized_power_budgets(stack)
        t_eff = torch.where(material_ok, t_eff, torch.zeros_like(t_eff))
        transmittance[rows] = transmittance.index_select(0, rows) * t_eff
        wall_count[rows] = wall_count.index_select(0, rows) + 1
        blocked[rows] = blocked.index_select(0, rows) | ~material_ok
        row_point = hit["p"].index_select(0, rows)
        epsilon = scale_aware_epsilon(row_point, scene_diagonal=scene_diagonal)
        origin[rows] = row_point + row_direction * epsilon[:, None]
        row_remaining = (
            remaining.index_select(0, rows)
            - hit["t"].index_select(0, rows)
            - epsilon
        )
        remaining[rows] = row_remaining
        next_active = torch.zeros_like(active)
        next_active[rows] = (row_remaining > 0.0) & material_ok & (t_eff > 0.0)
        active = next_active

    transmittance = torch.where(
        blocked, torch.zeros_like(transmittance), transmittance
    )
    penetrated = (wall_count >= 1) & ~blocked
    return {
        "transmittance": transmittance,
        "wall_count": wall_count,
        "penetrated": penetrated,
        "distance": distance,
    }
