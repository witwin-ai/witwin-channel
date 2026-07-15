"""Shared Kirchhoff rough-surface scattering helpers for the MC solvers.

Extends the wave-2 two-way {reflect, transmit} event machinery
(:mod:`witwin.channel_native.montecarlo.transmission`) to the three-way
{reflect, scatter, transmit} selection of plan section 7.1 at hits on rough
faces (``scatter_model_id == 1``), and hosts the pure-torch pieces both MC
solvers share: local roughness frames, TE/TM incident power decomposition,
per-material Kirchhoff table sampling/eval, and the BDPT scattering NEE
connection-row builder.

Measure and normalization conventions (documented once, used everywhere):

- Event probabilities are proportional to the ``scattering.energy``
  budgets ``(R_coh, R_diff, T_bar)`` at the hit incidence with the same
  minimum-probability floor pattern as the wave-2 two-way split; the
  selected branch's POWER is divided by its probability (fields by
  ``sqrt(p)``), keeping the estimator unbiased. Smooth faces
  (``scatter_model_id != 1``) keep the wave-2 two-way logic bit-identically:
  their scatter probability is exactly zero and their transmit probability
  still comes from the native stack budgets, so the single selection
  uniform partitions identically.
- Scattering contributions are POWER-ONLY (ensemble-average Kirchhoff
  BSDF): the scattered field carries phase 0 as a placeholder that only
  ever enters ``|field|^2``; no random phase is assigned (plan section 7.3).
- Specular reflection and delta transmission stay discrete events;
  Kirchhoff scattering is a continuous solid-angle density. The scattered
  subpath stores ``pdf_forward *= pdf(wo|wi)`` and ``pdf_reverse *=
  pdf(wi|wo)`` from the SAME reciprocal table with swapped arguments
  (contract section 5).
- v1 depth rule: scattering is single-bounce. A scattered subpath connects
  to the receivers (NEE) and terminates; reflection/transmission never
  follow a scattering event in v1.

NEE contribution derivation (the BDPT connection convention is
``contribution = source_power * |field|^2 * (lambda/(4*pi*L))^2 / N`` with
no 1/r stored in the field, i.e. an isotropic-source picture where the
power flux density at unfolded distance ``r1`` is
``source_power * |F|^2 / (4*pi*r1^2)``):

    g_scatter = Int_wall dA |F|^2 cos_i f_K(wi,wo) cos_o
                * (lambda/(4*pi))^2 / (r1^2 * r2^2)

Monte Carlo over the tx launch directions (uniform sphere, pdf 1/(4*pi))
converts solid angle to wall area over the UNFOLDED prefix distance,
``dA = r1^2 dOmega / cos_i`` (specular prefixes preserve the image-source
solid-angle measure), so per hitting ray both ``r1^2`` and ``cos_i``
cancel:

    row = source_power * (P_te*f_te + P_tm*f_tm) * cos_o
          * (lambda/(4*pi))^2 * 4*pi / (r2^2 * p_scatter * N)

where ``P_te/P_tm`` are the incident Jones powers in the local s/p basis
(the same basis the table's co-pol kernels use) and the division by
``p_scatter`` accounts for NEE being evaluated only on scatter-selected
rows. Cross-polarization is not tracked in v1 (contract section 6): the
receiver is treated as polarization-matched to the scattered power, which
keeps BDPT consistent with the unpolarized MC basic map.

MIS: the NEE connection is the ONLY strategy that produces a
tx->...->scatter->rx path in v1 (the directional continuation terminates
without a sensor-intersection strategy, and a point/cell-center sensor has
zero probability under the continuous direction density), so the balance
heuristic over {directional sample, connection} degenerates to weight 1
for the connection rows. The directional density converted to the
connection measure is still recorded in the row ``pdf`` for diagnostics
and future multi-bounce MIS.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from witwin.channel_native.core.tensor_math import normalize_vec3

from witwin.channel_native.core.kernels.ops import (
    scattering_event_probabilities,
    scattering_table_sample,
)
from witwin.channel_native.core.materials import Roughness
from witwin.channel_native.scattering import tables as kirchhoff_tables

from witwin.channel_native.core.material_runtime import face_material_field_bundle
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge

from .transmission import (
    _EVENT_PROBABILITY_FLOOR,
    _splitmix64,
    event_selection_seed,
    scale_aware_epsilon,
    scene_diagonal_m,
)

LIGHT_SPEED_M_PER_S = 299_792_458.0

__all__ = [
    "MASK_SCATTERING",
    "SCATTERING_COMPONENT_ID",
    "SCATTERING_EVENT_TYPE",
    "RoughMaterialRuntime",
    "eval_bsdf_rows",
    "local_frames",
    "solid_angle_to_area_jacobian",
    "local_to_world",
    "rough_material_runtimes",
    "sample_scatter_directions",
    "scatter_direction_uniforms",
    "scattered_subpath_state",
    "scattering_map_matrix",
    "scattering_nee_connection_samples",
    "te_tm_incident_power",
    "three_way_rough_probabilities",
    "world_to_local",
]

# Contract section 1: component_mask bit and exclusive path-class id.
MASK_SCATTERING = 16
SCATTERING_COMPONENT_ID = 6
# Subpath event_type ids: 0=source, 1=specular reflection, 2=delta
# transmission (native kernels), 3=continuous Kirchhoff scattering.
SCATTERING_EVENT_TYPE = 3

# Distinct seed stream for the two direction-sampling uniforms so the event
# SELECTION uniforms stay bit-identical to the wave-2 stream.
_DIRECTION_SEED_SALT = 0x5CA77E12D5B7A31F
_MASK63 = (1 << 63) - 1
_DEGENERATE_SIN_SQ = 1.0e-12


@dataclass(frozen=True, slots=True)
class RoughMaterialRuntime:
    """Per-material scattering runtime: table plus oracle-layer inputs."""

    material_index: int
    table: Any  # KirchhoffTable
    layers: tuple[tuple[float, float, float, float], ...]
    roughness: Roughness


def rough_material_runtimes(compiled: Any) -> dict[int, RoughMaterialRuntime]:
    """Kirchhoff runtimes per material index (``scatter_model_id == 1``).

    Reads the CSR layers and roughness fields from the compiled
    MaterialStore; the tables come from the compiled-scene lazy cache and
    raise ``kirchhoff_domain_exceeded`` for out-of-domain roughness.
    """

    store = compiled.materials
    runtimes: dict[int, RoughMaterialRuntime] = {}
    for index, table in compiled.kirchhoff_tables.items():
        offset = int(store.layer_offset[index])
        count = int(store.layer_count[index])
        layers = tuple(
            (
                float(store.layer_thickness_m[row]),
                float(store.layer_eps_r[row]),
                float(store.layer_sigma_e[row]),
                float(store.layer_mu_r[row]),
            )
            for row in range(offset, offset + count)
        )
        roughness = Roughness(
            rms_height_m=float(store.rough_sigma_h_m[index]),
            corr_length_x_m=float(store.rough_corr_x_m[index]),
            corr_length_y_m=float(store.rough_corr_y_m[index]),
            principal_axis_rad=float(store.rough_axis_rad[index]),
        )
        runtimes[index] = RoughMaterialRuntime(
            material_index=int(index),
            table=table,
            layers=layers,
            roughness=roughness,
        )
    return runtimes


def scatter_direction_uniforms(
    count: int, *, seed: int, tx_index: int, depth: int, device: torch.device
) -> torch.Tensor:
    """Reproducible ``[count, 2]`` uniforms for Kirchhoff direction sampling.

    Derived like the wave-2 event-selection seeds but salted into a distinct
    stream, so the selection uniforms of
    :func:`witwin.channel_native.montecarlo.transmission.event_uniforms`
    are untouched (bit-identical smooth-face behavior).
    """

    generator = torch.Generator(device=device)
    salted = _splitmix64(
        event_selection_seed(seed, tx_index, depth) ^ _DIRECTION_SEED_SALT
    )
    generator.manual_seed(salted & _MASK63)
    return torch.rand((int(count), 2), device=device, generator=generator)


def three_way_rough_probabilities(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    material_bundle: dict[str, torch.Tensor],
    stack: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    floor: float = _EVENT_PROBABILITY_FLOOR,
) -> dict[str, torch.Tensor]:
    """Rough-face three-way event probabilities (plan section 7.1).

    Returns row tensors ``p_scatter``, ``p_transmit``, ``r_coh_amplitude``
    (the coherent attenuation ``C_r`` for the reflect branch field) and the
    bool ``rough`` mask. Non-rough rows get zeros / False; the caller keeps
    the wave-2 two-way probabilities for them so smooth faces stay
    bit-identical.

    Probability construction per rough row: raw shares proportional to
    ``(R_coh, R_diff, T_bar)``; every event with a strictly positive budget
    is floored at ``floor`` and the triple renormalized. Events with a zero
    budget keep probability zero (never selected, so no division by their
    probability ever happens).
    """

    return scattering_event_probabilities(
        cos_theta.contiguous(),
        material_id.to(torch.int32).contiguous(),
        stack["cap_R_te"],
        stack["cap_R_tm"],
        stack["cap_T_te"],
        stack["cap_T_tm"],
        material_bundle["rough_sigma_h_m"],
        material_bundle["scatter_model_id"],
        frequency_hz=float(frequency_hz),
        probability_floor=float(floor),
    )


def local_frames(
    normal: torch.Tensor, axis_rad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic tangent frame ``(t1, t2)`` of unit normals ``[N, 3]``.

    ``t1`` is the table's local x axis (roughness principal axis): the
    stable perpendicular reference (z axis unless the normal is nearly
    vertical, then x) rotated by ``axis_rad`` about the normal. Isotropic
    tables are rotation-invariant so the reference choice only matters for
    anisotropic roughness; anchoring the principal axis to this
    deterministic reference is the documented v1 convention (the same
    degenerate-axis pattern as the polarization basis helpers).
    """

    reference = torch.where(
        (normal[:, 2].abs() < 0.9)[:, None],
        torch.tensor([0.0, 0.0, 1.0], device=normal.device),
        torch.tensor([1.0, 0.0, 0.0], device=normal.device),
    )
    t1 = torch.linalg.cross(reference, normal)
    t1 = normalize_vec3(t1)
    t2 = torch.linalg.cross(normal, t1)
    cos_a = torch.cos(axis_rad)[:, None]
    sin_a = torch.sin(axis_rad)[:, None]
    t1_rot = cos_a * t1 + sin_a * t2
    t2_rot = -sin_a * t1 + cos_a * t2
    return t1_rot, t2_rot


def world_to_local(
    w: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, normal: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        (
            (w * t1).sum(dim=-1),
            (w * t2).sum(dim=-1),
            (w * normal).sum(dim=-1),
        ),
        dim=-1,
    )


def local_to_world(
    w: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, normal: torch.Tensor
) -> torch.Tensor:
    return w[:, 0:1] * t1 + w[:, 1:2] * t2 + w[:, 2:3] * normal


def solid_angle_to_area_jacobian(
    cosine: torch.Tensor, distance: torch.Tensor
) -> torch.Tensor:
    """Return ``|cos(theta)|/r^2`` without folding it into a proposal PDF."""

    return cosine.clamp_min(0.0) / distance.clamp_min(1.0e-6).square()


def te_tm_incident_power(
    field_real: torch.Tensor,
    field_imag: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Incident Jones powers ``(P_te, P_tm)`` in the local s/p basis.

    ``direction`` is the propagation direction into the surface, ``normal``
    the mean-plane normal flipped toward the incident side. ``s =
    normalize(direction x normal)`` (contract section 2) with a
    deterministic substitute basis at normal incidence,
    ``p = s x direction``. Any
    longitudinal residual of the field is dropped by the projection (the
    Jones carriers are transverse by construction).
    """

    s = torch.linalg.cross(direction, normal)
    s_norm_sq = (s * s).sum(dim=-1, keepdim=True)
    substitute, _ = local_frames(
        direction, torch.zeros(direction.shape[0], device=direction.device)
    )
    s = torch.where(
        s_norm_sq > _DEGENERATE_SIN_SQ,
        normalize_vec3(s),
        substitute,
    )
    p = torch.linalg.cross(s, direction)
    p_te = (field_real * s).sum(dim=-1) ** 2 + (field_imag * s).sum(dim=-1) ** 2
    p_tm = (field_real * p).sum(dim=-1) ** 2 + (field_imag * p).sum(dim=-1) ** 2
    return p_te, p_tm


def _grouped_rows(
    material_id: torch.Tensor, runtimes: dict[int, RoughMaterialRuntime]
) -> list[tuple[RoughMaterialRuntime, torch.Tensor]]:
    groups = []
    for index, runtime in runtimes.items():
        rows = torch.nonzero(material_id == index, as_tuple=False).flatten()
        if int(rows.numel()):
            groups.append((runtime, rows))
    return groups


def sample_scatter_directions(
    material_id: torch.Tensor,
    wi_local: torch.Tensor,
    uniforms: torch.Tensor,
    runtimes: dict[int, RoughMaterialRuntime],
) -> dict[str, torch.Tensor]:
    """Per-material Kirchhoff CDF sampling of local outgoing directions.

    Returns ``wo_local`` plus the continuous forward density
    ``pdf(wo|wi)`` and the swapped-argument reverse density ``pdf(wi|wo)``
    (contract section 5). Rows without a rough material keep zeros.
    """

    count = int(material_id.shape[0])
    device = wi_local.device
    wo_local = torch.zeros((count, 3), device=device, dtype=torch.float32)
    pdf_forward = torch.zeros((count,), device=device, dtype=torch.float32)
    pdf_reverse = torch.zeros((count,), device=device, dtype=torch.float32)
    for runtime, rows in _grouped_rows(material_id, runtimes):
        wi_rows = wi_local.index_select(0, rows).contiguous()
        sampled = scattering_table_sample(
            wi_rows,
            uniforms.index_select(0, rows).contiguous(),
            runtime.table.marginal_cdf,
            runtime.table.conditional_cdf,
            runtime.table.sample_density,
        )
        wo_rows = sampled["wo"]
        wo_local[rows] = wo_rows
        pdf_forward[rows] = sampled["pdf_forward"]
        pdf_reverse[rows] = sampled["pdf_reverse"]
    return {
        "wo_local": wo_local,
        "pdf_forward": pdf_forward,
        "pdf_reverse": pdf_reverse,
    }


def eval_bsdf_rows(
    material_id: torch.Tensor,
    wi_local: torch.Tensor,
    wo_local: torch.Tensor,
    runtimes: dict[int, RoughMaterialRuntime],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched ``(f_te, f_tm)`` lookups grouped by rough material."""

    count = int(material_id.shape[0])
    device = wi_local.device
    f_te = torch.zeros((count,), device=device, dtype=torch.float32)
    f_tm = torch.zeros((count,), device=device, dtype=torch.float32)
    for runtime, rows in _grouped_rows(material_id, runtimes):
        te_rows, tm_rows = kirchhoff_tables.eval_bsdf(
            runtime.table,
            wi_local.index_select(0, rows).contiguous(),
            wo_local.index_select(0, rows).contiguous(),
        )
        f_te[rows] = te_rows
        f_tm[rows] = tm_rows
    return f_te, f_tm


def scattering_nee_connection_samples(
    raydn: Any,
    sensor: dict[str, torch.Tensor],
    runtimes: dict[int, Any],
    *,
    position: torch.Tensor,
    normal: torch.Tensor,
    frame_t1: torch.Tensor,
    frame_t2: torch.Tensor,
    wi_local: torch.Tensor,
    p_te: torch.Tensor,
    p_tm: torch.Tensor,
    p_scatter: torch.Tensor,
    material_id: torch.Tensor,
    source_power: torch.Tensor,
    tx_id: torch.Tensor,
    light_depth: torch.Tensor,
    path_length_at_vertex: torch.Tensor,
    frequency_hz: float,
    samples: int,
    scene_diagonal: float,
) -> dict[str, torch.Tensor] | None:
    """NEE connection rows from scatter-selected vertices (component 6).

    Assembled in torch instead of the native endpoint-connection kernel
    because scattering needs a PER-SENSOR BSDF re-evaluation f_K(wi, w_rx)
    and the 1/(r1^2*r2^2) double-spreading geometry, neither of which the
    delta-event connection kernel (single per-vertex field, unfolded
    1/(r1+r2)^2 spreading, sampled-direction hemisphere gate) can express.

    Row value (derivation in the module docstring):
        source_power * (P_te*f_te + P_tm*f_tm) * cos_o
        * (lambda/(4*pi))^2 * 4*pi / (r2^2 * p_scatter * N)
    The unfolded prefix distance r1 and cos_theta_i cancel against the
    solid-angle-to-area Jacobian of the uniform-sphere launch density.
    mis_weight is 1: the connection is the only strategy with nonzero
    density for a scattered vertex in v1 (single-bounce termination, point
    sensors); the directional density converted to the connection measure
    is recorded in ``pdf`` for diagnostics.
    """

    vertex_count = int(position.shape[0])
    sensor_count = int(sensor["origin"].shape[0])
    if vertex_count == 0 or sensor_count == 0:
        return None
    wavelength = LIGHT_SPEED_M_PER_S / float(frequency_hz)
    # Light-major layout (vertex * sensor_count + sensor), matching the
    # native endpoint connection tables.
    rx_origin = sensor["origin"]  # [R, 3]
    delta = rx_origin[None, :, :] - position[:, None, :]  # [S, R, 3]
    r2 = delta.norm(dim=-1).clamp_min(1.0e-6)
    wo_world = delta / r2[..., None]
    cos_o = (wo_world * normal[:, None, :]).sum(dim=-1)
    front = cos_o > 0.0

    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(vertex_count * sensor_count, *t.shape[2:])

    def expand_rows(t: torch.Tensor) -> torch.Tensor:
        return t[:, None].expand(vertex_count, sensor_count, *t.shape[1:])

    wo_flat = flat(wo_world).contiguous()
    wo_local = torch.stack(
        (
            (wo_flat * flat(expand_rows(frame_t1))).sum(dim=-1),
            (wo_flat * flat(expand_rows(frame_t2))).sum(dim=-1),
            (wo_flat * flat(expand_rows(normal))).sum(dim=-1),
        ),
        dim=-1,
    )
    wi_rows = flat(expand_rows(wi_local)).contiguous()
    material_rows = flat(expand_rows(material_id)).contiguous()
    f_te, f_tm = eval_bsdf_rows(material_rows, wi_rows, wo_local.contiguous(), runtimes)

    p_te_rows = flat(expand_rows(p_te))
    p_tm_rows = flat(expand_rows(p_tm))
    power_kernel = p_te_rows * f_te + p_tm_rows * f_tm
    amplitude = wavelength / (4.0 * math.pi)
    geometry = (
        flat(cos_o).clamp_min(0.0)
        * (amplitude * amplitude)
        * (4.0 * math.pi)
        / (flat(r2) ** 2)
    )
    p_s_rows = flat(expand_rows(p_scatter)).clamp_min(1.0e-4)
    contribution = (
        flat(expand_rows(source_power))
        * power_kernel
        * geometry
        / (p_s_rows * float(max(1, int(samples))))
    )

    rx_id_rows = sensor["rx_id"][None, :].expand(vertex_count, sensor_count)
    valid = (
        flat(front)
        & flat(expand_rows(tx_id >= 0))
        & (flat(rx_id_rows) >= 0)
        & flat(sensor["valid"][None, :].expand(vertex_count, sensor_count))
        & (contribution > 0.0)
    )
    if not bool(valid.any()):
        return None

    # Visibility: offset the surface start point along the connection
    # direction with the contract scale-aware epsilon so the rough face
    # never occludes its own connection.
    epsilon = scale_aware_epsilon(
        flat(expand_rows(position)), scene_diagonal=scene_diagonal
    )
    start = flat(expand_rows(position)) + wo_flat * epsilon[:, None]
    end = flat(rx_origin[None, :, :].expand(vertex_count, sensor_count, 3)).contiguous()
    visible = geometry_bridge.raydn_visibility_forward(
        raydn.require_handle(),
        start.contiguous(),
        end,
        valid.contiguous(),
    )[0]
    valid = valid & visible
    contribution = torch.where(valid, contribution, torch.zeros_like(contribution))

    # Store only the sampled proposal density. The solid-angle-to-area
    # Jacobian is a geometry conversion and is deliberately kept out of the
    # public ``pdf`` diagnostic.
    pdf_omega = torch.zeros_like(contribution)
    for index, runtime in runtimes.items():
        rows = torch.nonzero(material_rows == index, as_tuple=False).flatten()
        if int(rows.numel()) == 0:
            continue
        pdf_omega[rows] = kirchhoff_tables.pdf(
            runtime.table,
            wi_rows.index_select(0, rows),
            wo_local.index_select(0, rows).contiguous(),
        )
    pdf_dir = p_s_rows * pdf_omega
    mis_weight = torch.where(
        valid, torch.ones_like(contribution), torch.zeros_like(contribution)
    )

    tx_rows = flat(expand_rows(tx_id)).to(torch.int32)
    rx_rows = flat(rx_id_rows).to(torch.int32)
    grid_rows = flat(
        sensor["grid_linear_id"][None, :].expand(vertex_count, sensor_count)
    ).to(torch.int32)
    depth_rows = flat(expand_rows(light_depth)).to(torch.int32)
    zero_depth = torch.zeros_like(depth_rows)
    component = torch.full_like(tx_rows, SCATTERING_COMPONENT_ID)
    path_length = flat(expand_rows(path_length_at_vertex)) + flat(r2)
    block = {
        "topology": torch.stack((tx_rows, rx_rows, component, depth_rows), dim=1),
        "contribution": contribution,
        "pdf": torch.where(valid, pdf_dir, torch.zeros_like(pdf_dir)),
        "mis_weight": mis_weight,
        "component_id": component,
        "valid": valid,
        "tx_id": tx_rows,
        "rx_id": rx_rows,
        "grid_linear_id": grid_rows,
        "light_depth": depth_rows,
        "sensor_depth": zero_depth,
        "path_length_m": path_length,
    }
    return {key: value.contiguous() for key, value in block.items()}


def scattered_subpath_state(
    state: dict[str, torch.Tensor],
    hit: dict[str, torch.Tensor],
    *,
    choose_scatter: torch.Tensor,
    normal: torch.Tensor,
    frame_t1: torch.Tensor,
    frame_t2: torch.Tensor,
    wi_local: torch.Tensor,
    p_te: torch.Tensor,
    p_tm: torch.Tensor,
    p_scatter: torch.Tensor,
    material_id: torch.Tensor,
    runtimes: dict[int, Any],
    uniforms: torch.Tensor,
    scene_diagonal: float,
) -> dict[str, torch.Tensor]:
    """Continued light subpath after a Kirchhoff scattering event.

    Assembled directly in torch with the native subpath tensor layout (no
    native kernel: the update is gather+FMA on the sampled table values).
    Scattering contributions are POWER-ONLY ensemble estimates. No coherent
    phase or cross-polarized Jones amplitude is available from the current
    Kirchhoff table, so the Complex3 carrier is explicitly cleared instead of
    manufacturing a zero-phase field. ``throughput_real`` remains a sampling
    proxy and must never be consumed as a field amplitude.
    """

    sampled = sample_scatter_directions(material_id, wi_local, uniforms, runtimes)
    wo_local = sampled["wo_local"]
    pdf_forward = sampled["pdf_forward"]
    wo_world = local_to_world(wo_local, frame_t1, frame_t2, normal)
    f_te, f_tm = eval_bsdf_rows(material_id, wi_local, wo_local, runtimes)
    incident_power = (p_te + p_tm).clamp_min(1.0e-20)
    f_weighted = (p_te * f_te + p_tm * f_tm) / incident_power
    cos_o = wo_local[:, 2].clamp_min(0.0)
    power_scale = (
        f_weighted
        * cos_o
        / (pdf_forward.clamp_min(1.0e-12) * p_scatter.clamp_min(1.0e-4))
    )
    valid = choose_scatter & (pdf_forward > 0.0) & (cos_o > 0.0) & state["valid"]
    amplitude_scale = torch.where(
        valid, power_scale.sqrt(), torch.zeros_like(power_scale)
    )

    epsilon = scale_aware_epsilon(hit["p"], scene_diagonal=scene_diagonal)
    origin = hit["p"] + wo_world * epsilon[:, None]
    scattered = dict(state)
    scattered["origin"] = origin
    scattered["direction"] = wo_world
    scattered["field_real"] = torch.zeros_like(state["field_real"])
    scattered["field_imag"] = torch.zeros_like(state["field_imag"])
    throughput_amp = torch.sqrt(
        state["throughput_real"] ** 2 + state["throughput_imag"] ** 2
    )
    scattered["throughput_real"] = throughput_amp * amplitude_scale
    scattered["throughput_imag"] = torch.zeros_like(throughput_amp)
    scattered["pdf_forward"] = state["pdf_forward"] * p_scatter * pdf_forward
    scattered["pdf_reverse"] = state["pdf_reverse"] * p_scatter * sampled["pdf_reverse"]
    scattered["depth"] = state["depth"] + 1
    scattered["component_mask"] = state["component_mask"] | MASK_SCATTERING
    scattered["event_type"] = torch.full_like(
        state["event_type"], SCATTERING_EVENT_TYPE
    )
    scattered["primitive_id"] = hit["global_prim_id"]
    scattered["path_length"] = state["path_length"] + hit["t"].clamp_min(0.0)
    scattered["valid"] = valid
    return scattered


def scattering_map_matrix(
    scene: Any,
    raydn: Any,
    tx_pos: torch.Tensor,
    tx_power: torch.Tensor,
    rx_pos: torch.Tensor,
    *,
    samples: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int]]:
    """(tx, rx) matrix of the MC basic Kirchhoff scattering path gain.

    Estimator (derivation shared with the BDPT NEE builder in the module
    docstring): the scattering path gain at a receiver point x_r is the
    wall-area integral

        g(x_r) = Int_A f_unpol(wi, wo) cos_i cos_o
                 * (lambda/(4*pi))^2 / (r1^2 * r2^2) dA

    estimated with ``samples`` area-weighted points on the rough faces
    (uniform-per-area density, weight A_total / samples). The matrix holds
    the PATH GAIN at each receiver point times the transmitter power,
    mirroring the LoS / transmission matrix conventions exactly.

    v1 simplifications (documented, deliberately truthful rather than
    approximate): the incident segment tx->point must be UNOBSTRUCTED (a
    binary visibility test; no through-wall incident power), the outgoing
    segment point->receiver likewise; MC basic is unpolarized
    (supports_polarization is False), so the unpolarized mean kernel
    0.5*(f_te + f_tm) is used - the average over incident polarizations.
    Scattering is single-bounce: tx -> rough point -> receiver only.
    """

    matrix = torch.zeros((int(tx_pos.shape[0]), int(rx_pos.shape[0])), device=device)
    stats = {
        "sample_count": 0,
        "rough_face_count": 0,
        "tx_visible_samples": 0,
        "deposited_rows": 0,
    }
    runtimes = rough_material_runtimes(scene.compile())
    if not runtimes:
        return matrix, stats

    handle = raydn.require_handle()
    bundle = face_material_field_bundle(scene, device=device)
    face_material_id = bundle["material_id"].to(torch.int64)
    face_scatter_model = bundle["scatter_model_id"].index_select(0, face_material_id)
    records = raydn.edge_records()
    vertices = records.vertices
    faces = records.faces.to(torch.int64)
    v0 = vertices.index_select(0, faces[:, 0])
    v1 = vertices.index_select(0, faces[:, 1])
    v2 = vertices.index_select(0, faces[:, 2])
    areas = 0.5 * torch.linalg.cross(v1 - v0, v2 - v0).norm(dim=-1)
    rough_faces = torch.nonzero(face_scatter_model == 1, as_tuple=False).flatten()
    stats["rough_face_count"] = int(rough_faces.numel())
    if int(rough_faces.numel()) == 0:
        return matrix, stats
    rough_areas = areas.index_select(0, rough_faces)
    total_area = float(rough_areas.sum())
    if not total_area > 0.0:
        return matrix, stats

    # Area-weighted point sampling: face by area (multinomial), then uniform
    # barycentric inside the face. Density is uniform per unit area, so each
    # sample carries weight A_total / N.
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    count = int(samples)
    chosen = rough_faces.index_select(
        0, torch.multinomial(rough_areas, count, replacement=True, generator=generator)
    )
    uv = torch.rand((count, 2), device=device, generator=generator)
    su = uv[:, 0].sqrt()
    b0 = (1.0 - su)[:, None]
    b1 = (su * (1.0 - uv[:, 1]))[:, None]
    b2 = (su * uv[:, 1])[:, None]
    points = (
        b0 * v0.index_select(0, chosen)
        + b1 * v1.index_select(0, chosen)
        + b2 * v2.index_select(0, chosen)
    )
    # edge_records face normals are area-weighted; the frames and cosine
    # factors below need unit mean-plane normals.
    normals = records.face_normals.index_select(0, chosen)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    point_material = face_material_id.index_select(0, chosen).to(torch.int32)
    axis_rad = bundle["rough_axis_rad"].index_select(0, point_material.to(torch.int64))
    stats["sample_count"] = count

    diagonal = scene_diagonal_m(scene)
    epsilon = scale_aware_epsilon(points, scene_diagonal=diagonal)
    rx_count = int(rx_pos.shape[0])
    wavelength = LIGHT_SPEED_M_PER_S / float(scene.frequency)
    amplitude_sq = (wavelength / (4.0 * math.pi)) ** 2
    area_weight = total_area / float(count)
    cell_chunk = max(1, 4_194_304 // max(count, 1))

    for tx_index in range(int(tx_pos.shape[0])):
        to_tx = tx_pos[tx_index][None, :] - points
        r1 = to_tx.norm(dim=-1).clamp_min(1.0e-6)
        wi_world = to_tx / r1[:, None]
        # Roughness applies to whichever side is illuminated in v1: flip the
        # mean-plane normal toward the transmitter.
        side = (normals * wi_world).sum(dim=-1)
        normal_flipped = torch.where((side < 0.0)[:, None], -normals, normals)
        cos_i = (wi_world * normal_flipped).sum(dim=-1)
        frame_t1, frame_t2 = local_frames(normal_flipped, axis_rad)
        wi_local = world_to_local(wi_world, frame_t1, frame_t2, normal_flipped)
        # Unobstructed tx->point requirement (v1): binary visibility on the
        # segment shortened off the surface by the scale-aware epsilon.
        candidates = cos_i > 1.0e-6
        visible_tx = geometry_bridge.raydn_visibility_forward(
            handle,
            tx_pos[tx_index][None, :].expand(count, 3).contiguous(),
            (points + normal_flipped * epsilon[:, None]).contiguous(),
            candidates.contiguous(),
        )[0]
        active = candidates & visible_tx
        stats["tx_visible_samples"] += int(active.sum())
        if not bool(active.any()):
            continue
        incident = torch.where(active, cos_i / (r1 * r1), torch.zeros_like(cos_i))
        for start in range(0, rx_count, cell_chunk):
            end = min(start + cell_chunk, rx_count)
            cells = rx_pos[start:end]
            block = int(cells.shape[0])
            delta = cells[None, :, :] - points[:, None, :]
            r2 = delta.norm(dim=-1).clamp_min(1.0e-6)
            wo_world = delta / r2[..., None]
            cos_o = (wo_world * normal_flipped[:, None, :]).sum(dim=-1)
            pair_active = active[:, None] & (cos_o > 0.0)
            if not bool(pair_active.any()):
                continue
            wo_flat = wo_world.reshape(count * block, 3)
            wo_local = torch.stack(
                (
                    (wo_flat * frame_t1.repeat_interleave(block, dim=0)).sum(dim=-1),
                    (wo_flat * frame_t2.repeat_interleave(block, dim=0)).sum(dim=-1),
                    (wo_flat * normal_flipped.repeat_interleave(block, dim=0)).sum(
                        dim=-1
                    ),
                ),
                dim=-1,
            )
            f_te, f_tm = eval_bsdf_rows(
                point_material.repeat_interleave(block),
                wi_local.repeat_interleave(block, dim=0).contiguous(),
                wo_local.contiguous(),
                runtimes,
            )
            f_unpol = 0.5 * (f_te + f_tm)
            visible_rx = geometry_bridge.raydn_visibility_forward(
                handle,
                (points[:, None, :] + wo_world * epsilon[:, None, None])
                .reshape(count * block, 3)
                .contiguous(),
                cells[None, :, :]
                .expand(count, block, 3)
                .reshape(count * block, 3)
                .contiguous(),
                pair_active.reshape(count * block).contiguous(),
            )[0]
            deposit = (
                area_weight
                * amplitude_sq
                * incident[:, None].expand(count, block).reshape(-1)
                * f_unpol
                * cos_o.reshape(-1).clamp_min(0.0)
                / (r2.reshape(-1) ** 2)
            )
            deposit = torch.where(
                pair_active.reshape(-1) & visible_rx,
                deposit,
                torch.zeros_like(deposit),
            )
            stats["deposited_rows"] += int((deposit > 0.0).sum())
            matrix[tx_index, start:end] += deposit.reshape(count, block).sum(dim=0)
    return matrix * tx_power[:, None], stats
