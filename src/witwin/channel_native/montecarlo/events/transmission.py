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

from witwin.channel_native.materials.kernels.autograd import em_layer_stack_ad
from witwin.channel_native.materials.kernels.functional import em_layer_stack_eval
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.runtime.autograd_contracts import _ad_frequency_value


_MIN_EPSILON_M = 1.0e-6
_RELATIVE_EPSILON = 1.0e-6
_EVENT_PROBABILITY_FLOOR = 0.05
# sin^2(theta) below which the plane of incidence is treated as degenerate
# (normal incidence); matches the scattering event-glue convention.
_DEGENERATE_SIN_SQ = 1.0e-12

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


def incident_te_tm_fractions(
    direction: torch.Tensor,
    normal: torch.Tensor,
    polarization: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Incident TE/TM power fractions ``(f_te, f_tm)`` of a polarized ray.

    ADR-020. ``direction`` is the propagation direction into the wall,
    ``normal`` the wall normal (orientation-independent: the s/p basis squares
    both projections), ``polarization`` the incident unit polarization vector.
    The local plane-of-incidence basis is ``s = normalize(direction x normal)``
    (TE) and ``p = s x direction`` (TM), matching the native layer-stack and the
    scattering event glue (``te_tm_incident_power``). The returned fractions
    partition the transverse incident power onto the wall's TE/TM axes and sum
    to one for any fully transverse polarization. At normal incidence the plane
    of incidence is degenerate; there TE and TM transmittances coincide, so the
    split is irrelevant and falls back to the unpolarized halves.
    """

    s = torch.linalg.cross(direction, normal)
    s_norm_sq = (s * s).sum(dim=-1, keepdim=True)
    degenerate = s_norm_sq <= _DEGENERATE_SIN_SQ
    s = torch.where(
        degenerate, torch.zeros_like(s), s * torch.rsqrt(s_norm_sq.clamp_min(1.0e-30))
    )
    p = torch.linalg.cross(s, direction)
    p_te = (polarization * s).sum(dim=-1) ** 2
    p_tm = (polarization * p).sum(dim=-1) ** 2
    total = p_te + p_tm
    safe = total > _DEGENERATE_SIN_SQ
    half = torch.full_like(p_te, 0.5)
    inv_total = torch.where(safe, 1.0 / total.clamp_min(1.0e-30), torch.zeros_like(total))
    f_te = torch.where(safe, p_te * inv_total, half)
    f_tm = torch.where(safe, p_tm * inv_total, half)
    return f_te, f_tm


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
    polarization: torch.Tensor,
    frequency_hz: float | torch.Tensor,
    frequency_value: float | None = None,
    max_depth: int,
    scene_diagonal: float,
    ad: bool = False,
    ledger: object | None = None,
) -> dict[str, torch.Tensor]:
    """March straight origin->target segments through up to ``max_depth``
    thin_sheet walls and accumulate the per-wall power transmittance product.

    ADR-020: each wall's power transmittance is the Jones-derived TE/TM power
    ``f_te * cap_T_te + f_tm * cap_T_tm`` projected on the incident
    ``polarization`` (a ``(3,)`` or ``(count, 3)`` unit vector), sharing the
    full-Jones layer-stack model with the deterministic/Path solvers rather than
    the polarization-agnostic TE/TM mean. The estimator stays power-domain (the
    radiomap accumulates total transmitted power, so the projection is on the
    incident polarization only, without a receiver-antenna projection).

    Returns per-row tensors: ``transmittance`` (product of polarized wall power
    transmittances; zero when blocked), ``wall_count`` (int32 penetrations), and
    ``penetrated`` (bool: at least one wall crossed AND the target was reached
    within the depth budget). Rows with more walls than ``max_depth`` or an
    invalid wall material are truthfully blocked (transmittance zero), never
    approximated.
    """

    device = origins.device
    count = int(origins.shape[0])
    if ad and frequency_value is None:
        # One host read of a tensor frequency for the whole march; every
        # per-wall em_layer_stack_ad below reuses this scalar (audit M3).
        frequency_value = _ad_frequency_value(frequency_hz)
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
        hit = geometry_bridge.bdpt_intersect_forward(
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
        if ad:
            # Plan 07 AD-3: same native stack kernel behind an autograd
            # Function, so the CSR layer leaves and the frequency keep their
            # gradients through the per-wall transmittance.
            if ledger is not None:
                ledger.add(
                    cos_theta,
                    row_material,
                    layer_csr["layer_offset"],
                    layer_csr["layer_count"],
                    layer_csr["layer_thickness_m"],
                    layer_csr["layer_eps_r"],
                    layer_csr["layer_sigma_e"],
                    layer_csr["layer_mu_r"],
                )
            stack = em_layer_stack_ad(
                cos_theta.contiguous(),
                row_material.clamp_min(0).contiguous(),
                layer_csr["layer_offset"],
                layer_csr["layer_count"],
                layer_csr["layer_thickness_m"],
                layer_csr["layer_eps_r"],
                layer_csr["layer_sigma_e"],
                layer_csr["layer_mu_r"],
                frequency=frequency_hz,
                frequency_value=frequency_value,
            )
        else:
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
        # ADR-020: polarized power transmittance projected on the incident
        # polarization, sharing the deterministic full-Jones model. The plane
        # of incidence is orientation-independent so the raw hit normal is fine.
        f_te, f_tm = incident_te_tm_fractions(
            row_direction,
            hit["n"].index_select(0, rows),
            polarization,
        )
        t_eff = f_te * stack["cap_T_te"] + f_tm * stack["cap_T_tm"]
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
