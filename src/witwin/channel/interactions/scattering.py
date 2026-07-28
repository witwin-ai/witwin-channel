"""Rough-surface scattering: discovery, geometry and enumerated orchestration.

One concept, one file. This module is the single owner of enumerated
scattering: the AD radiometric scalars, the single-bounce ensemble and
realization row builders, the ADR-021 D1 scatter-chain discovery, and the chain
append path. It replaces the four former modules
``propagation/enumerated/scattering.py``, ``scattering_chain.py``,
``scattering_chain_append.py`` and ``scattering_scalars.py``; that split was a
file-size artifact, not an ownership boundary. Native evaluation still belongs
to the kernel facades in ``witwin.channel.kernels`` and the typed row contracts
still belong to ``witwin.channel.propagation.rows``.

``scattering_chain`` restated ``_R2_ALPHA``, ``_MAX_SAMPLES_PER_FACE``,
``_MIN_COS``, ``_VISIBILITY_CHUNK``, ``_r2_barycentric`` and ``_offset_eps``
byte-identically from the single-bounce module. With one file there is one
definition of each, and every caller keeps the value and the code it already
executed.

Each origin docstring is preserved verbatim as a comment above the section it
describes.
"""

from __future__ import annotations

import math
from dataclasses import (
    dataclass,
    fields as _dataclass_fields,
    replace,
)
from typing import (
    Any,
    TYPE_CHECKING,
    TypedDict,
)

import torch

from witwin.core import PhaseScreen
from witwin.core import SurfaceRoughness  # noqa: F401 - legacy reachable global

from witwin.channel import scattering as kirchhoff_tables
from witwin.channel.constants import C0
from witwin.channel.field_state import (
    receiver_polarizations,
    transmitter_polarizations,
)
from witwin.channel.interactions.reflection import (
    ReflectionEpcQuery,
    iter_reflection_multibounce_epc_requests,
    iter_reflection_order1_epc_requests,
    prepare_reflection_multibounce_plan,
    prepare_reflection_order1_plan,
    query_reflection_epc,
)
from witwin.channel.interactions.transmission import (
    _EVENT_PROBABILITY_FLOOR,
    _splitmix64,
    event_selection_seed,
    scale_aware_epsilon,
    scene_diagonal_m,
)
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.kernels import materials as material_kernels
from witwin.channel.kernels import scattering as scattering_kernels
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel.kernels.scattering import (
    scattering_event_probabilities,
    scattering_table_eval_ad,
    scattering_table_sample,
)
from witwin.channel.materials import (
    face_material_field_bundle,
    face_material_tensors,
    face_material_thickness,
)
from witwin.channel.propagation.enumerated.contracts import TopologyConfig
from witwin.channel.propagation.geometry.endpoints import (
    receiver_positions_and_layout,
    transmitter_tensors,
)
from witwin.channel.propagation.geometry.reevaluate import _cached_coplanar_face_groups
from witwin.channel.propagation.rows import (
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathTopology,
)
from witwin.channel.propagation.topology.export import EvaluatedPathSidecars
from witwin.channel.scattering import KirchhoffTable  # noqa: F401
from witwin.channel.scene.endpoints import require_compiled
from witwin.channel.scene.resources import (
    RoughMaterialRuntime,
    realization_phase_screens,
)
from witwin.channel.tensor_math import normalize_vec3

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene

__all__ = [
    "ChainSamples",
    "KMAX_AD_DEPTH",
    "MASK_SCATTERING",
    "SCATTERING_COMPONENT_ID",
    "SCATTERING_EVENT_TYPE",
    "RoughMaterialRuntime",
    "ScatterChainDiscovery",
    "append_chain_scattering_paths",
    "append_scattering_evaluated_paths",
    "build_chain_samples",
    "discover_scatter_chains",
    "eval_bsdf_rows",
    "local_frames",
    "solid_angle_to_area_jacobian",
    "local_to_world",
    "rough_material_runtimes",
    "sample_scatter_directions",
    "scatter_carried_incident_power",
    "scatter_direction_uniforms",
    "scattered_subpath_state",
    "scattering_map_matrix",
    "scattering_nee_connection_samples",
    "te_tm_incident_power",
    "three_way_rough_probabilities",
    "world_to_local",
]


# -------------------------------------------------------------------------
# AD radiometric scalars (was scattering_scalars.py)
# -------------------------------------------------------------------------
#
# AD-scalar construction for deterministic scattering rows (ADR-014/015).
#
# Frequency-dependent radiometric scalars used by the ensemble and realization
# scattering paths. In AD mode these become Torch scalars so frequency gradients
# flow through the ensemble radiometric ``coef`` and the realization ``k0`` /
# amplitude scale; in the primal path the callers keep their plain Python-float
# scalars, so this module carries only the differentiable branch.


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


# -------------------------------------------------------------------------
# Single-bounce scattering (was enumerated/scattering.py)
# -------------------------------------------------------------------------
#
# Deterministic rough-surface scattering (plan 05 wave 3, contract section 6).
#
# Appends single-bounce ``component_id=6`` scattering rows to canonical typed
# ``EvaluatedPaths`` contracts. Two mutually exclusive per-surface modes:
#
# - ``ensemble`` (production): Kirchhoff ensemble BSDF patch quadrature.
#   Incoherent POWER rows, one row per visible patch sample.
# - ``realization_coherent`` (reference): phase-screen Kirchhoff patch integral
#   for surfaces carrying a ``PhaseScreen`` assignment. One coherent complex
#   row per (tx, rx, structure); it REPLACES both the delta specular and the
#   ensemble lobe for that surface (contract 6.7.3 - the two models are never
#   summed for one surface).
#
# Normalization (ensemble). The repo's ``path_gain`` is a received power for
# unit-gain antennas (LoS ``P_t * (lambda / (4*pi*d))^2`` with the matching
# complex amplitude under ``core.field_state.PHASE_CONVENTION``). With the
# Kirchhoff power BSDF ``f`` (per steradian, hemispherically normalized to
# ``R_diff``) and aperture ``A_e = lambda^2/(4*pi)``, a patch of area ``A`` yields
# the plan section 9 patch-quadrature power ``P_r = P_t * f * cos_theta_i *
# cos_theta_o * A * lambda^2 / ((4*pi)^2 * r1^2 * r2^2)`` (``gamma = 4*pi*f``).
# Cross-check (tested): the specular-delta limit collapses the patch sum over an
# infinite plane to the image-source ``P_t * R * (lambda/(4*pi*(r1+r2)))^2``.
#
# Polarization (v1): the tx polarization is projected onto the incident
# transverse plane, decomposed in the local s/p basis (``s = normalize(n x d)``,
# ``p = s x d``, contract section 2), the co-pol table channels are weighted by
# the squared projections and the receive side applies the receiver's outgoing
# s/p projections. Cross-pol arises only from this frame rotation.
#
# Realization mode phase bookkeeping: ``patch_phase_integral`` computes
# ``Int exp(-j*(q.x + q_n*h)) dA`` with ``q = k_s - k_i``. The physical patch
# factor ``e^{-j k0 (r1c + r2c)} * Int exp(+j*(q.delta + q_n*h)) dA`` (first-order
# expansion about centroid ``c``) is obtained losslessly by SWAPPING wave vectors
# (``k_i <-> k_s`` flips ``q`` and the ``exp(+j*q_n*h)`` height sign); the leftover
# absolute-position phase is removed with ``exp(-j*q.c)``. The prefactor
# ``j*k0*F/(4*pi)``, ``F = |q|^2/(k0*q_n)``, makes the smooth large-plate limit
# collapse to the exact image-source reflection by stationary phase (tested).
#
# Scattering rows accumulate in the POWER domain (plan 7.3): ensemble rows carry
# ``path_field = sqrt(path_gain)`` with zero phase (metadata flag
# ``scattering_paths_incoherent``); realization rows keep their physical complex
# field in the row but still fold into totals as power.


# Documented caps (module docstring / config docstrings).
_MAX_SAMPLES_PER_FACE = 4096
_MAX_REALIZATION_PATCH_GRID = 64  # per-face subdivision grid cap (m x m x m^2 tris)
_VISIBILITY_CHUNK = 1 << 20
_PAIR_SAMPLE_CHUNK = 1 << 20
_MIN_COS = 1.0e-4
# R2 low-discrepancy sequence (plastic constant) for deterministic,
# refinement-stable barycentric patch samples.
_R2_ALPHA = (0.7548776662466927, 0.5698402909980532)


def _unit(v: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return normalize_vec3(v, eps=eps)


def _stable_tangent(n: torch.Tensor) -> torch.Tensor:
    """Deterministic unit tangent per normal (smallest |component| axis)."""

    axis = torch.zeros_like(n)
    pick = n.abs().argmin(dim=-1)
    axis.scatter_(-1, pick.unsqueeze(-1), 1.0)
    t = axis - (axis * n).sum(dim=-1, keepdim=True) * n
    return _unit(t)


def _sp_basis(
    n: torch.Tensor, d: torch.Tensor, backup_axis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local ``s = normalize(n x d)``, ``p = s x d`` with a deterministic
    backup axis at normal incidence (contract section 2)."""

    s = torch.cross(n, d, dim=-1)
    degenerate = torch.linalg.vector_norm(s, dim=-1, keepdim=True) < 1.0e-6
    s = torch.where(degenerate, backup_axis, _unit(s))
    p = torch.cross(s, d, dim=-1)
    return s, p


def _offset_eps(points: torch.Tensor, scene_diagonal: torch.Tensor) -> torch.Tensor:
    """Scale-aware hit offset (contract section 4)."""

    return torch.maximum(
        torch.linalg.vector_norm(points, dim=-1) * 1.0e-6, scene_diagonal * 1.0e-6
    ).clamp_min(1.0e-6)


def _visible(
    rayd: object, start: torch.Tensor, end: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Chunked segment visibility; returns (mask, launch_count)."""

    count = int(start.shape[0])
    if count == 0:
        return torch.empty((0,), device=start.device, dtype=torch.bool), 0
    handle = rayd.require_resource()
    masks = []
    launches = 0
    for lo in range(0, count, _VISIBILITY_CHUNK):
        hi = min(lo + _VISIBILITY_CHUNK, count)
        masks.append(
            geometry_kernels.rayd_visibility_forward(
                handle, start[lo:hi].contiguous(), end[lo:hi].contiguous(), None
            )[0]
        )
        launches += 1
    return torch.cat(masks), launches


def _r2_barycentric(
    counts: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic per-face R2 sample coordinates.

    Returns ``(face_of_sample, a, b)`` where ``(a, b)`` are uniform triangle
    barycentric coordinates (unit-square R2 points folded across the
    diagonal, an area-preserving map).
    """

    total = int(counts.sum())
    face_of_sample = torch.repeat_interleave(
        torch.arange(counts.shape[0], device=device, dtype=torch.int64), counts
    )
    offsets = torch.cumsum(counts, dim=0) - counts
    rank = (
        torch.arange(total, device=device, dtype=torch.int64)
        - offsets[face_of_sample]
    ).to(torch.float64)
    a = torch.frac(0.5 + (rank + 1.0) * _R2_ALPHA[0])
    b = torch.frac(0.5 + (rank + 1.0) * _R2_ALPHA[1])
    fold = (a + b) > 1.0
    a = torch.where(fold, 1.0 - a, a).to(torch.float32)
    b = torch.where(fold, 1.0 - b, b).to(torch.float32)
    return face_of_sample, a, b


def _keep_strongest_per_pair(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    gain: torch.Tensor,
    *,
    num_rx: int,
    cap: int,
) -> torch.Tensor:
    """Row indices of the strongest <= cap samples per (tx, rx) pair."""

    pair = tx_id.to(torch.int64) * int(num_rx) + rx_id.to(torch.int64)
    # Lexsort (pair asc, gain desc) via two stable sorts.
    order = torch.argsort(-gain.to(torch.float64), stable=True)
    order = order[torch.argsort(pair[order], stable=True)]
    sorted_pair = pair[order]
    row = torch.arange(order.numel(), device=order.device, dtype=torch.int64)
    first = torch.ones_like(sorted_pair, dtype=torch.bool)
    first[1:] = sorted_pair[1:] != sorted_pair[:-1]
    starts = torch.where(first, row, torch.zeros_like(row)).cummax(dim=0).values
    return order[(row - starts) < int(cap)]


class _RowCollector:
    """Accumulates flat scattering rows before the topology concat."""

    def __init__(self, device: torch.device):
        self.device = device
        self.fields: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "tx_id",
                "rx_id",
                "primitive_id",
                "material_id",
                "position",
                "normal",
                "path_length_m",
                "path_gain",
                "path_field",
                "coefficient",
                "direction",
            )
        }

    def add(self, **tensors: torch.Tensor) -> None:
        for name, value in tensors.items():
            self.fields[name].append(value)

    def cat(self) -> dict[str, torch.Tensor] | None:
        if not self.fields["path_gain"]:
            return None
        return {name: torch.cat(values) for name, values in self.fields.items()}


def _ensemble_rows(
    scene: Scene,
    compiled: object,
    config: TopologyConfig,
    collector: _RowCollector,
    *,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    ensemble_faces: torch.Tensor,
    scene_diagonal: torch.Tensor,
    info: dict[str, Any],
    ad_mode: str = "none",
) -> None:
    device = tx_positions.device
    ad_enabled = ad_mode != "none"
    records = compiled.rayd.edge_records()
    faces = records.faces.to(torch.int64)[ensemble_faces]
    tri = records.vertices[faces]  # [F, 3, 3]
    normals = _unit(records.face_normals[ensemble_faces])
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * torch.linalg.vector_norm(torch.cross(e1, e2, dim=-1), dim=-1)
    face_material = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    )[ensemble_faces]

    density = float(getattr(config, "scattering_samples_per_m2", 8.0))
    counts = (
        torch.ceil(areas * density).to(torch.int64).clamp(1, _MAX_SAMPLES_PER_FACE)
    )
    face_of_sample, a, b = _r2_barycentric(counts, device)
    points = (
        tri[face_of_sample, 0]
        + a[:, None] * e1[face_of_sample]
        + b[:, None] * e2[face_of_sample]
    )
    weights = (areas / counts.to(torch.float32))[face_of_sample]  # [S] patch areas
    sample_normal = normals[face_of_sample]
    sample_face = ensemble_faces[face_of_sample]
    sample_material = face_material[face_of_sample]
    sample_count = int(points.shape[0])
    info["ensemble_sample_count"] = sample_count
    info["ensemble_face_count"] = int(ensemble_faces.numel())

    resources = compiled.kirchhoff_resources  # raises kirchhoff_domain_exceeded
    stack = resources.stack
    materials = compiled.materials
    axis_rad = materials.rough_axis_rad.to(device=device, dtype=torch.float32)
    sample_material_i32 = sample_material.to(torch.int32).contiguous()

    frequency = float(scene.frequency)
    wavelength = C0 / frequency
    # Repo path_gain units (module docstring): P_t * f * cos_i * cos_o * A
    # * lambda^2 / ((4*pi)^2 * r1^2 * r2^2).
    power_scale = wavelength**2 / (4.0 * math.pi) ** 2
    # AD mode (ADR-014): the radiometric scale ``coef`` becomes a Torch scalar
    # so frequency gradients flow through the ensemble rows (their only
    # frequency dependence). Ensemble rows are zero-phase power rows, so nothing
    # else here is differentiable w.r.t. frequency.
    coef_scale_t = ensemble_coef_scale(scene, device, ad_enabled=ad_enabled)

    eps = _offset_eps(points, scene_diagonal)
    threshold = float(getattr(config, "scattering_power_threshold", 0.0))
    num_rx = int(rx_positions.shape[0])

    for tx_index in range(int(tx_positions.shape[0])):
        tx = tx_positions[tx_index]
        to_tx = tx.reshape(1, 3) - points
        r1 = torch.linalg.vector_norm(to_tx, dim=-1).clamp_min(1.0e-6)
        wi_w = to_tx / r1[:, None]
        # Flip the mean-plane normal toward the transmitter side: v1 treats
        # the illuminated side as the rough front surface.
        side = torch.sign((wi_w * sample_normal).sum(-1))
        side = torch.where(side == 0.0, torch.ones_like(side), side)
        n_o = sample_normal * side[:, None]
        cos_i = (wi_w * n_o).sum(-1)
        tx_active = cos_i > _MIN_COS
        offset_points = points + n_o * eps[:, None]
        vis_tx, launches = _visible(
            compiled.rayd,
            tx.reshape(1, 3).expand(sample_count, 3).contiguous(),
            offset_points,
        )
        info["visibility_launch_count"] += launches
        tx_active &= vis_tx
        if not bool(tx_active.any()):
            continue

        # Roughness principal frame and incident s/p projections (fixed per
        # sample for this transmitter).
        backup_axis = _stable_tangent(n_o)
        t1 = backup_axis
        angle = axis_rad[sample_material]
        t2 = torch.cross(n_o, t1, dim=-1)
        t1r = t1 * torch.cos(angle)[:, None] + t2 * torch.sin(angle)[:, None]
        t2r = torch.cross(n_o, t1r, dim=-1)
        d_i = -wi_w  # incident propagation direction
        s_i, p_i = _sp_basis(n_o, d_i, backup_axis)
        pol_t = tx_pol[tx_index].reshape(1, 3)
        pol_t_perp = pol_t - (pol_t * d_i).sum(-1, keepdim=True) * d_i
        a_te2 = (pol_t_perp * s_i).sum(-1).square()
        a_tm2 = (pol_t_perp * p_i).sum(-1).square()
        wi_local = torch.stack(
            ((wi_w * t1r).sum(-1), (wi_w * t2r).sum(-1), cos_i), dim=-1
        )

        rx_chunk = max(1, _PAIR_SAMPLE_CHUNK // max(sample_count, 1))
        for rx_start in range(0, num_rx, rx_chunk):
            rx_end = min(rx_start + rx_chunk, num_rx)
            rx_block = rx_positions[rx_start:rx_end]  # [Rc, 3]
            to_rx = rx_block[:, None, :] - points[None, :, :]  # [Rc, S, 3]
            r2 = torch.linalg.vector_norm(to_rx, dim=-1).clamp_min(1.0e-6)
            wo_w = to_rx / r2[..., None]
            cos_o = (wo_w * n_o[None]).sum(-1)
            candidate = tx_active[None, :] & (cos_o > _MIN_COS)
            rows = torch.nonzero(candidate, as_tuple=False)
            if int(rows.shape[0]) == 0:
                continue
            rc, sc = rows[:, 0], rows[:, 1]
            vis_rx, launches = _visible(
                compiled.rayd,
                rx_block[rc].contiguous(),
                offset_points[sc].contiguous(),
            )
            info["visibility_launch_count"] += launches
            row_valid = vis_rx[vis_rx].contiguous()
            rc, sc = rc[vis_rx], sc[vis_rx]
            if int(sc.shape[0]) == 0:
                continue

            # Native Kirchhoff ensemble row physics (ADR-010 op 1). The
            # candidate grid (to_rx/r2/wo/cos_o) stays Torch per the ADR;
            # its surviving rows are gathered here (bitwise the values the
            # previous Torch physics consumed) and the kernel owns wo_local,
            # the stacked-table lookup, the outgoing s/p basis + receiver
            # projections, f_eff, gain, keep, amplitude and length. The row
            # selection/concat below stays Torch (structural).
            wo_row = wo_w[rc, sc].contiguous()
            ensemble_args = (
                row_valid,
                wo_row,
                r2[rc, sc].contiguous(),
                cos_o[rc, sc].contiguous(),
                n_o,
                t1r,
                t2r,
                wi_local,
                cos_i,
                r1,
                a_te2,
                a_tm2,
                weights,
                sample_material_i32,
                backup_axis,
                rx_pol[rx_start:rx_end].contiguous(),
                rc.contiguous(),
                sc.contiguous(),
                stack.f_te_flat,
                stack.f_tm_flat,
                stack.table_offset,
                stack.table_dims,
                stack.material_slot,
            )
            if ad_enabled:
                evaluated = scattering_kernels.scattering_ensemble_eval_ad(
                    *ensemble_args,
                    coef=float(tx_power[tx_index]) * coef_scale_t,
                    threshold=max(threshold, 0.0),
                )
            else:
                evaluated = scattering_kernels.scattering_ensemble_eval(
                    *ensemble_args,
                    coef=float(tx_power[tx_index]) * power_scale,
                    threshold=max(threshold, 0.0),
                )
            keep = evaluated["keep"]
            if not bool(keep.any()):
                continue
            rc, sc = rc[keep], sc[keep]
            gain = evaluated["gain"][keep]
            length = evaluated["length"][keep]
            wo_row = wo_row[keep]
            amplitude = evaluated["amplitude"][keep]
            zero = torch.zeros_like(amplitude)
            collector.add(
                tx_id=torch.full_like(sc, tx_index, dtype=torch.int64).to(torch.int32),
                rx_id=(rx_start + rc).to(torch.int32),
                primitive_id=sample_face[sc].to(torch.int32),
                material_id=sample_material[sc].to(torch.int32),
                position=points[sc],
                normal=n_o[sc],
                path_length_m=length,
                path_gain=gain,
                # Incoherent power rows: zero-phase sqrt(power) magnitude
                # (metadata flag scattering_paths_incoherent).
                path_field=torch.complex(amplitude, zero),
                coefficient=torch.complex(
                    amplitude / max(float(tx_power[tx_index]), 1.0e-30) ** 0.5, zero
                ),
                direction=wo_row,
            )


def _subdivide_face(
    tri: torch.Tensor, uv: torch.Tensor, m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform barycentric m^2 subdivision of one triangle (+ matching UV)."""

    device = tri.device
    i, j = torch.meshgrid(
        torch.arange(m + 1, device=device),
        torch.arange(m + 1, device=device),
        indexing="ij",
    )
    up_mask = (i + j) <= (m - 1)
    dn_mask = (i + j) <= (m - 2)
    iu, ju = i[up_mask].float(), j[up_mask].float()
    idn, jdn = i[dn_mask].float(), j[dn_mask].float()
    corners = []
    for ii, jj in (
        (iu, ju),
        (iu + 1, ju),
        (iu, ju + 1),
    ):
        corners.append(torch.stack((ii, jj), dim=-1))
    up = torch.stack(corners, dim=1)  # [Tu, 3, 2] integer barycentric
    corners = []
    for ii, jj in (
        (idn + 1, jdn),
        (idn + 1, jdn + 1),
        (idn, jdn + 1),
    ):
        corners.append(torch.stack((ii, jj), dim=-1))
    down = torch.stack(corners, dim=1)
    grid = torch.cat((up, down), dim=0) / float(m)  # [T, 3, 2] (a, b)
    a = grid[..., 0:1]
    b = grid[..., 1:2]
    verts = tri[0] + a * (tri[1] - tri[0]) + b * (tri[2] - tri[0])
    uvs = uv[0] + a * (uv[1] - uv[0]) + b * (uv[2] - uv[0])
    return verts, uvs


def _realization_rows(
    scene: Scene,
    compiled: object,
    config: TopologyConfig,
    collector: _RowCollector,
    *,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    screens: dict[int, PhaseScreen],
    scene_diagonal: torch.Tensor,
    info: dict[str, Any],
    ad_mode: str = "none",
) -> None:
    device = tx_positions.device
    ad_enabled = ad_mode != "none"
    records = compiled.rayd.edge_records()
    materials = compiled.materials
    layer_offset = materials.layer_offset.to(device=device, dtype=torch.int32)
    layer_count = materials.layer_count.to(device=device, dtype=torch.int32)
    layer_thickness = materials.layer_thickness_m.to(device=device, dtype=torch.float32)
    layer_eps = materials.layer_eps_r.to(device=device, dtype=torch.float32)
    layer_sigma = materials.layer_sigma_e.to(device=device, dtype=torch.float32)
    layer_mu = materials.layer_mu_r.to(device=device, dtype=torch.float32)

    frequency = float(scene.frequency)
    k0 = 2.0 * math.pi * frequency / C0
    wavelength = C0 / frequency
    # AD mode (ADR-014): k0 and the outer amplitude scale become Torch scalars
    # so frequency gradients flow through the coherent phase, the Kirchhoff
    # prefactor and the radiometric normalization.
    if ad_enabled:
        frequency_t, k0_t, amplitude_scale_t = realization_scalars(scene, device)
    resources = compiled.phase_screen_resources.structures
    density = float(getattr(config, "scattering_samples_per_m2", 8.0))

    for structure_index in sorted(screens):
        resource = resources[structure_index]
        runtime = resource.runtime
        if resource.face_count == 0:
            continue
        first_face = resource.first_face
        face_stop = resource.face_range[1]
        faces = records.faces[first_face:face_stop].to(torch.int64)
        tri = records.vertices[faces]  # [F, 3, 3]
        material_index = resource.material_index
        areas = resource.face_areas_m2
        uv_tris = resource.uv_tris

        # Per-face subdivision: patch edges must resolve both the requested
        # density and the Fresnel-zone linearization of the carrier phase
        # (patch size <= 0.5*sqrt(lambda*r_min/2) keeps the neglected
        # quadratic phase under ~0.1 rad).
        all_tris: list[torch.Tensor] = []
        all_uvs: list[torch.Tensor] = []
        endpoint_r = torch.cat(
            (
                torch.linalg.vector_norm(
                    tri.reshape(-1, 3)[None] - tx_positions[:, None], dim=-1
                ).reshape(-1),
                torch.linalg.vector_norm(
                    tri.reshape(-1, 3)[None] - rx_positions[:, None], dim=-1
                ).reshape(-1),
            )
        )
        r_min = float(endpoint_r.min().clamp_min(1.0e-3))
        s_max = 0.5 * math.sqrt(wavelength * r_min / 2.0)
        for local in range(resource.face_count):
            edge = float(
                torch.linalg.vector_norm(
                    tri[local] - tri[local].roll(1, dims=0), dim=-1
                ).max()
            )
            m_density = math.ceil(math.sqrt(max(1.0, density * float(areas[local]))))
            m = max(m_density, math.ceil(edge / max(s_max, 1.0e-6)))
            m = min(max(m, 1), _MAX_REALIZATION_PATCH_GRID)
            verts, uvs = _subdivide_face(
                tri[local], uv_tris[local], m
            )
            all_tris.append(verts)
            all_uvs.append(uvs)
        patch_tris = torch.cat(all_tris)  # [P, 3, 3]
        patch_uvs = torch.cat(all_uvs)  # [P, 3, 2]
        patch_normal = _unit(
            torch.cross(
                patch_tris[:, 1] - patch_tris[:, 0],
                patch_tris[:, 2] - patch_tris[:, 0],
                dim=-1,
            )
        )
        centroids = patch_tris.mean(dim=1)
        patch_count = int(centroids.shape[0])
        info["realization_patch_count"] += patch_count
        eps = _offset_eps(centroids, scene_diagonal)

        for tx_index in range(int(tx_positions.shape[0])):
            tx = tx_positions[tx_index]
            to_tx = tx.reshape(1, 3) - centroids
            r1 = torch.linalg.vector_norm(to_tx, dim=-1).clamp_min(1.0e-6)
            wi_w = to_tx / r1[:, None]
            # v1 treats the illuminated side as the rough front surface (same
            # convention as the ensemble path): the mean-plane normal flips
            # toward the transmitter. When that flips the winding normal, the
            # phase-screen heights are effectively mirrored (-h), which is a
            # statistically identical, reproducible realization.
            side = torch.sign((wi_w * patch_normal).sum(-1))
            side = torch.where(side == 0.0, torch.ones_like(side), side)
            n_o = patch_normal * side[:, None]
            cos_i = (wi_w * n_o).sum(-1)
            front = cos_i > _MIN_COS
            offset_points = centroids + n_o * eps[:, None]
            vis_tx, launches = _visible(
                compiled.rayd,
                tx.reshape(1, 3).expand(patch_count, 3).contiguous(),
                offset_points,
            )
            info["visibility_launch_count"] += launches
            front &= vis_tx
            if not bool(front.any()):
                continue
            for rx_index in range(int(rx_positions.shape[0])):
                rx = rx_positions[rx_index]
                to_rx = rx.reshape(1, 3) - centroids
                r2 = torch.linalg.vector_norm(to_rx, dim=-1).clamp_min(1.0e-6)
                wo_w = to_rx / r2[:, None]
                cos_o = (wo_w * n_o).sum(-1)
                active = front & (cos_o > _MIN_COS)
                rows = torch.nonzero(active, as_tuple=False).reshape(-1)
                if int(rows.numel()) == 0:
                    continue
                vis_rx, launches = _visible(
                    compiled.rayd,
                    rx.reshape(1, 3).expand(int(rows.numel()), 3).contiguous(),
                    offset_points[rows].contiguous(),
                )
                info["visibility_launch_count"] += launches
                row_valid = vis_rx[vis_rx].contiguous()
                rows = rows[vis_rx]
                if int(rows.numel()) == 0:
                    continue

                # Smooth-stack Jones at the mean plane per patch.
                stack_args = (
                    cos_i[rows].contiguous(),
                    torch.full(
                        (int(rows.numel()),),
                        material_index,
                        device=device,
                        dtype=torch.int32,
                    ),
                    layer_offset,
                    layer_count,
                    layer_thickness,
                    layer_eps,
                    layer_sigma,
                    layer_mu,
                )
                if ad_enabled:
                    # AD mode (ADR-015 Part B): the differentiable stack keeps
                    # the shared CSR layer eps_r / sigma_e / thickness and the
                    # carrier frequency on the graph, so the realization
                    # r_te/r_tm carry material and frequency gradients. Frequency
                    # threads as the Torch scalar the AD branch already builds
                    # (mirrors the transmission event section of interactions/transmission.py).
                    stack = material_kernels.em_layer_stack_ad(
                        *stack_args,
                        frequency=frequency_t,
                    )
                else:
                    stack = material_kernels.em_layer_stack_eval(
                        *stack_args,
                        frequency_hz=frequency,
                    )
                r_te = torch.complex(stack["r_te_real"], stack["r_te_imag"])
                r_tm = torch.complex(stack["r_tm_real"], stack["r_tm_imag"])

                d_i = -wi_w[rows]
                d_o = wo_w[rows]
                n_rows = n_o[rows]

                # Native phase-screen patch integral (ADR-010 op 2): the
                # kernel owns the jones/prefactor/carrier assembly and the
                # Duffy-mapped 16x16 Gauss-Legendre quadrature over the
                # phase-screen heights, evaluating the swapped-wave-vector
                # integrand exp(-j*(q'.x + q_n'*h)) with q' = -q (the
                # physical +j integrand of the module docstring derivation)
                # and the fixed-order deterministic total reduction.
                patch_args = (
                    row_valid,
                    patch_tris,
                    patch_uvs,
                    rows.contiguous(),
                    d_i.contiguous(),
                    d_o.contiguous(),
                    n_rows.contiguous(),
                    r_te.contiguous(),
                    r_tm.contiguous(),
                    tx_pol[tx_index].contiguous(),
                    rx_pol[rx_index].contiguous(),
                    r1[rows].contiguous(),
                    r2[rows].contiguous(),
                    centroids[rows].contiguous(),
                    runtime.heights_m,
                )
                if ad_enabled:
                    evaluated = scattering_kernels.scattering_patch_integral_eval_ad(
                        *patch_args,
                        k0=k0_t,
                    )
                else:
                    evaluated = scattering_kernels.scattering_patch_integral_eval(
                        *patch_args,
                        k0=k0,
                    )
                total = evaluated["total"]
                # Same VALUE as the previous per-patch loop counter: one
                # integral per selected row, counted host-side.
                info["realization_patch_integrals"] += int(rows.numel())

                if ad_enabled:
                    amplitude_scale = (
                        math.sqrt(max(float(tx_power[tx_index]), 1.0e-30))
                        * amplitude_scale_t
                    )
                else:
                    amplitude_scale = math.sqrt(
                        max(float(tx_power[tx_index]), 1.0e-30)
                    ) * wavelength / (4.0 * math.pi)
                path_field = (amplitude_scale * total).reshape(1)
                gain = path_field.abs().square().to(torch.float32)
                patch_areas = 0.5 * torch.linalg.vector_norm(
                    torch.cross(
                        patch_tris[rows][:, 1] - patch_tris[rows][:, 0],
                        patch_tris[rows][:, 2] - patch_tris[rows][:, 0],
                        dim=-1,
                    ),
                    dim=-1,
                )
                area_total = patch_areas.sum().clamp_min(1.0e-20)
                mean_length = (
                    ((r1[rows] + r2[rows]) * patch_areas).sum() / area_total
                ).reshape(1)
                mean_position = (
                    (centroids[rows] * patch_areas[:, None]).sum(0) / area_total
                ).reshape(1, 3)
                mean_normal = _unit(
                    (n_o[rows] * patch_areas[:, None]).sum(0)
                ).reshape(1, 3)
                mean_direction = _unit(rx.reshape(1, 3) - mean_position)
                collector.add(
                    tx_id=torch.tensor([tx_index], device=device, dtype=torch.int32),
                    rx_id=torch.tensor([rx_index], device=device, dtype=torch.int32),
                    primitive_id=torch.tensor(
                        [first_face], device=device, dtype=torch.int32
                    ),
                    material_id=torch.tensor(
                        [material_index], device=device, dtype=torch.int32
                    ),
                    position=mean_position,
                    normal=mean_normal,
                    path_length_m=mean_length.to(torch.float32),
                    path_gain=gain,
                    path_field=path_field.to(torch.complex64),
                    coefficient=(
                        path_field
                        / math.sqrt(max(float(tx_power[tx_index]), 1.0e-30))
                    ).to(torch.complex64),
                    direction=mean_direction,
                )


class _ExtendedScatteringRows(TypedDict):
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    path_gain: torch.Tensor
    path_field: torch.Tensor
    field_xyz: torch.Tensor
    coefficient: torch.Tensor
    field_direction: torch.Tensor
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_type: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor


def _extended_scattering_rows(
    source: EvaluatedPaths,
    rows: dict[str, torch.Tensor],
) -> _ExtendedScatteringRows:
    """Single 22-tensor concatenation owner for typed path contracts."""

    topology = source.topology
    geometry = source.geometry
    fields = source.fields

    device = topology.valid.device
    count = int(rows["path_gain"].shape[0])
    width = int(topology.primitive_sequence.shape[1])
    primitive_sequence_base = topology.primitive_sequence
    material_sequence_base = topology.material_sequence
    interaction_type_base = topology.interaction_type
    interaction_positions_base = geometry.interaction_positions
    interaction_normals_base = geometry.interaction_normals
    if width < 1:
        # A scattering-only request has no specular block from which to infer
        # sequence width. Give any existing zero-depth rows one inactive slot
        # before appending the single SCATTERING interaction.
        existing = int(topology.valid.numel())
        primitive_sequence_base = torch.full(
            (existing, 1), -1, device=device, dtype=torch.int32
        )
        material_sequence_base = torch.full(
            (existing, 1), -1, device=device, dtype=torch.int32
        )
        interaction_type_base = torch.zeros(
            (existing, 1), device=device, dtype=torch.int32
        )
        interaction_positions_base = torch.zeros(
            (existing, 1, 3), device=device, dtype=torch.float32
        )
        interaction_normals_base = torch.zeros(
            (existing, 1, 3), device=device, dtype=torch.float32
        )
        width = 1

    depth = torch.ones((count,), device=device, dtype=torch.int32)
    component = torch.full((count,), 6, device=device, dtype=torch.int32)
    minus_one = torch.full((count, width), -1, device=device, dtype=torch.int32)
    primitive_sequence = minus_one.clone()
    primitive_sequence[:, 0] = rows["primitive_id"]
    material_sequence = minus_one.clone()
    material_sequence[:, 0] = rows["material_id"]
    positions = torch.zeros((count, width, 3), device=device, dtype=torch.float32)
    positions[:, 0] = rows["position"]
    normals = torch.zeros_like(positions)
    normals[:, 0] = rows["normal"]
    interaction_type = torch.zeros((count, width), device=device, dtype=torch.int32)
    interaction_type[:, 0] = 8  # InteractionType.SCATTERING

    field_xyz = torch.zeros((count, 3), device=device, dtype=torch.complex64)

    def cat(existing: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        return torch.cat((existing, new.to(existing.dtype))).contiguous()

    return _ExtendedScatteringRows(
        valid=cat(
            topology.valid, torch.ones((count,), device=device, dtype=torch.bool)
        ),
        tx_id=cat(topology.tx_id, rows["tx_id"]),
        rx_id=cat(topology.rx_id, rows["rx_id"]),
        depth=cat(topology.depth, depth),
        component_id=cat(topology.component_id, component),
        primitive_id=cat(topology.primitive_id, rows["primitive_id"]),
        edge_id=cat(
            topology.edge_id,
            torch.full((count,), -1, device=device, dtype=torch.int32),
        ),
        path_length_m=cat(geometry.path_length_m, rows["path_length_m"]),
        delay_s=cat(geometry.delay_s, rows["path_length_m"] / C0),
        path_gain=cat(fields.path_gain, rows["path_gain"]),
        path_field=cat(fields.path_field, rows["path_field"]),
        field_xyz=cat(fields.field_xyz, field_xyz),
        coefficient=cat(fields.coefficient, rows["coefficient"]),
        field_direction=cat(geometry.field_direction, rows["direction"]),
        interaction_position=cat(geometry.interaction_position, rows["position"]),
        interaction_normal=cat(geometry.interaction_normal, rows["normal"]),
        material_id=cat(topology.material_id, rows["material_id"]),
        primitive_sequence=cat(primitive_sequence_base, primitive_sequence),
        material_sequence=cat(material_sequence_base, material_sequence),
        interaction_type=cat(interaction_type_base, interaction_type),
        interaction_positions=cat(interaction_positions_base, positions),
        interaction_normals=cat(interaction_normals_base, normals),
    )


def _extend_evaluated_paths(
    evaluated: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
    rows: dict[str, torch.Tensor],
    *,
    launch_count_delta: int,
    candidate_count_delta: int,
    guardrail_count_delta: int,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    extended = _extended_scattering_rows(evaluated, rows)
    topology = PathTopology(
        valid=extended["valid"],
        tx_id=extended["tx_id"],
        rx_id=extended["rx_id"],
        depth=extended["depth"],
        component_id=extended["component_id"],
        primitive_id=extended["primitive_id"],
        edge_id=extended["edge_id"],
        material_id=extended["material_id"],
        primitive_sequence=extended["primitive_sequence"],
        material_sequence=extended["material_sequence"],
        interaction_type=extended["interaction_type"],
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=extended["path_length_m"],
        delay_s=extended["delay_s"],
        field_direction=extended["field_direction"],
        interaction_position=extended["interaction_position"],
        interaction_normal=extended["interaction_normal"],
        interaction_positions=extended["interaction_positions"],
        interaction_normals=extended["interaction_normals"],
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=extended["path_gain"],
        path_field=extended["path_field"],
        field_xyz=extended["field_xyz"],
        coefficient=extended["coefficient"],
    )
    evaluated = EvaluatedPaths(
        topology=topology,
        geometry=geometry,
        fields=fields,
    )
    sidecars = replace(
        sidecars,
        execution=replace(
            sidecars.execution,
            launch_count=sidecars.execution.launch_count + launch_count_delta,
            candidate_count=(
                sidecars.execution.candidate_count + candidate_count_delta
            ),
            guardrail_count=sidecars.execution.guardrail_count + guardrail_count_delta,
        ),
    )
    return evaluated, sidecars


def _scattering_info() -> dict[str, Any]:
    return {
        "scattering_paths_incoherent": True,
        "accumulation": "power_domain",
        "ensemble_face_count": 0,
        "ensemble_sample_count": 0,
        "realization_structure_count": 0,
        "realization_patch_count": 0,
        "realization_patch_integrals": 0,
        "visibility_launch_count": 0,
        "path_count": 0,
        "capped_path_count": 0,
        # ADR-021 D1 enumerated scatter-chain diagnostics (0 when default-OFF).
        "chain_sample_count": 0,
        "chain_row_count": 0,
        "chain_kept_count": 0,
    }


def _collect_scattering_rows(
    scene: Scene,
    config: TopologyConfig,
    *,
    device: torch.device,
    info: dict[str, Any],
    ad_mode: str = "none",
    endpoint_tensors: object | None = None,
) -> tuple[dict[str, torch.Tensor] | None, int, int, int]:
    compiled = require_compiled(scene)
    screens = realization_phase_screens(compiled.materials, compiled.assignments)
    face_material = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    )
    scatter_face = (
        compiled.materials.scatter_model_id.to(device=device)[face_material] == 1
    )
    face_structure = compiled.geometry.face_structure_id.to(
        device=device, dtype=torch.int64
    )
    realization_face = torch.zeros_like(scatter_face)
    for structure_index in screens:
        structure_id = int(compiled.assignments.structure_id[structure_index])
        realization_face |= face_structure == structure_id
    ensemble_faces = torch.nonzero(
        scatter_face & ~realization_face, as_tuple=False
    ).reshape(-1)
    info["realization_structure_count"] = len(screens)
    if int(ensemble_faces.numel()) == 0 and not screens:
        return None, 0, 0, 0
    if not compiled.rayd.available:
        raise RuntimeError(
            "deterministic scattering requires RayD native scene capability"
        )

    if endpoint_tensors is None:
        tx_positions, tx_power = transmitter_tensors(scene, device=device)
        rx_positions, _layout = receiver_positions_and_layout(scene, device=device)
        tx_pol = transmitter_polarizations(scene, device=device)
        rx_pol = receiver_polarizations(scene, device=device)
    else:
        tx_positions = endpoint_tensors.tx_positions
        tx_power = endpoint_tensors.tx_power
        rx_positions = endpoint_tensors.rx_positions
        tx_pol = endpoint_tensors.tx_polarizations
        rx_pol = endpoint_tensors.rx_polarizations
    if tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return None, 0, 0, 0
    records = compiled.rayd.edge_records()
    vertices = records.vertices
    scene_diagonal = (vertices.max(dim=0).values - vertices.min(dim=0).values).norm()

    launch_before = info["visibility_launch_count"]
    collector = _RowCollector(device)
    if int(ensemble_faces.numel()) > 0:
        _ensemble_rows(
            scene,
            compiled,
            config,
            collector,
            tx_positions=tx_positions,
            tx_power=tx_power,
            rx_positions=rx_positions,
            tx_pol=tx_pol,
            rx_pol=rx_pol,
            ensemble_faces=ensemble_faces,
            scene_diagonal=scene_diagonal,
            info=info,
            ad_mode=ad_mode,
        )
    if screens:
        _realization_rows(
            scene,
            compiled,
            config,
            collector,
            tx_positions=tx_positions,
            tx_power=tx_power,
            rx_positions=rx_positions,
            tx_pol=tx_pol,
            rx_pol=rx_pol,
            screens=screens,
            scene_diagonal=scene_diagonal,
            info=info,
            ad_mode=ad_mode,
        )

    rows = collector.cat()
    if rows is None:
        return None, 0, 0, 0

    candidate_count = int(rows["path_gain"].shape[0])
    cap = int(getattr(config, "scattering_max_paths_per_pair", 4096))
    keep = _keep_strongest_per_pair(
        rows["tx_id"],
        rows["rx_id"],
        rows["path_gain"],
        num_rx=int(rx_positions.shape[0]),
        cap=cap,
    )
    dropped = candidate_count - int(keep.numel())
    if dropped > 0:
        # The cap is a hard truthfulness boundary: dropped samples vanish from
        # BOTH the exported paths and the accumulated power (reported here,
        # never silently redistributed onto the kept rows).
        total = float(rows["path_gain"].sum())
        kept_power = float(rows["path_gain"][keep].sum())
        info["capped_path_count"] = dropped
        info["capped_power_fraction"] = (
            0.0 if total <= 0.0 else max(0.0, 1.0 - kept_power / total)
        )
        rows = {name: value[keep] for name, value in rows.items()}
    info["path_count"] = int(rows["path_gain"].shape[0])

    return (
        rows,
        info["visibility_launch_count"] - launch_before,
        candidate_count,
        dropped,
    )


def append_scattering_evaluated_paths(
    scene: Scene,
    config: TopologyConfig,
    evaluated: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
    *,
    endpoint_tensors: object | None = None,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars, dict[str, Any]]:
    """Append scattering rows to canonical typed path contracts."""

    info = _scattering_info()
    components = set(config.components)
    if "scattering" not in components or int(config.max_depth) < 1:
        return evaluated, sidecars, info
    if not scene.structures:
        return evaluated, sidecars, info

    ad_mode = str(getattr(config, "ad_mode", "none"))
    collector_kwargs = {
        "device": evaluated.device,
        "info": info,
        "ad_mode": ad_mode,
    }
    if endpoint_tensors is not None:
        collector_kwargs["endpoint_tensors"] = endpoint_tensors
    rows, launch_delta, candidate_delta, guardrail_delta = _collect_scattering_rows(
        scene, config, **collector_kwargs
    )
    if rows is not None:
        evaluated, sidecars = _extend_evaluated_paths(
            evaluated,
            sidecars,
            rows,
            launch_count_delta=launch_delta,
            candidate_count_delta=candidate_delta,
            guardrail_count_delta=guardrail_delta,
        )
    # ADR-021 D1 enumerated scatter-chain rows. DEFAULT-OFF: the branch is only
    # entered when scattering_chain_max_depth >= 1, so the single-bounce pipeline
    # above stays byte-identical when the new config is absent/0. The chain append
    # path is defined further down this module; the former lazy import existed
    # only to break a module-load cycle between the two files.
    if int(getattr(config, "scattering_chain_max_depth", 0)) >= 1:
        evaluated, sidecars = append_chain_scattering_paths(
            scene,
            config,
            evaluated,
            sidecars,
            info,
            ad_mode=ad_mode,
        )
    return evaluated, sidecars, info


# -------------------------------------------------------------------------
# Scatter-chain discovery (was scattering_chain.py)
# -------------------------------------------------------------------------
#
# ADR-021 D1 enumerated scatter-chain discovery (Deterministic + Path).
#
# Discovers the enumerated scatter-chain path class
#
#     TX --C1 (reflections, depth d1 >= 0)--> v_s --C2 (depth d2 >= 0)--> RX
#     1 <= d1 + d2 <= scattering_chain_max_depth
#
# by running the EXISTING RayD image-method reflection enumeration twice against a
# dedicated chain-sample set as virtual endpoints (``tx -> {samples}`` for C1 and
# ``rx -> {samples}`` for C2, reciprocal), then joining the two legs on the sample
# index. No geometry is recomputed in Python/Torch: every hit point, normal, and
# face id comes from the native RayD EPC (``query_reflection_epc`` /
# ``rayd_reflection_epc_paths_forward``) exactly as the deterministic reflection
# topology owner uses it (``interactions/reflection.py``). The only
# Python work here is the sanctioned structural boundary work (plan 10a section 2):
# join on the sample index, keep-strongest budgeting, stable row ordering, padding
# the per-leg bounce blocks to the native ``kMaxAdDepth = 8`` capacity, and packing
# the derived per-row lengths / spreading / incident-outgoing directions the frozen
# :class:`ScatterChainDiscovery` contract requires.
#
# The produced :class:`ScatterChainDiscovery` is the read-only typed contract the
# native Op A / Op B chain facades (ADR-021 D2, owner ``kernels/scattering.py``)
# consume; this module owns discovery only. Directions (``d_i``/``d_o``), lengths
# (``L1``/``L2``), spreading (``sp1``/``sp2``), and cosines are derived from the
# RayD-owned hit positions as structural packing, matching the reference oracles
# ``tests/reference/chain_ensemble.py`` / ``chain_realization.py``.


# Native on-stack ReflectionChain capacity (field_transport_ad_common.cuh
# kMaxAdDepth). Each specular leg is padded to this width independently.
KMAX_AD_DEPTH = 8


# ---------------------------------------------------------------------------
# Frozen typed contract (plan 10a section 2).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScatterChainDiscovery:
    """C1/C2 join output consumed read-only by the native chain facades.

    ``R`` = joined chain rows (budgeted keep-strongest-per-pair +
    ``scattering_chain_max_rows`` per (tx, rx)). Rows are frozen in tx-major
    order, so a consumer may pass a narrow view of its existing per-tx device
    selection mask as the explicit RayD row-valid contract without allocating
    a replacement mask. Every tensor shares one CUDA device and is
    C-contiguous; ``Dmax = KMAX_AD_DEPTH = 8``. See plan 10a section 2 for the
    normative field-by-field contract.
    """

    # Row identity.
    tx_id: torch.Tensor  # [R]     int32
    rx_id: torch.Tensor  # [R]     int32
    sample_index: torch.Tensor  # [R]     int64
    d1: torch.Tensor  # [R]     int32
    d2: torch.Tensor  # [R]     int32

    # C1 leg (TX -> v_s), padded to Dmax.
    c1_positions: torch.Tensor  # [R, Dmax, 3] f32
    c1_normals: torch.Tensor  # [R, Dmax, 3] f32
    c1_primitive: torch.Tensor  # [R, Dmax]    int32
    c1_material: torch.Tensor  # [R, Dmax]    int32
    L1: torch.Tensor  # [R]     f32
    d_i: torch.Tensor  # [R, 3]  f32

    # C2 leg (v_s -> RX), padded to Dmax (indexed from the vertex outward).
    c2_positions: torch.Tensor  # [R, Dmax, 3] f32
    c2_normals: torch.Tensor  # [R, Dmax, 3] f32
    c2_primitive: torch.Tensor  # [R, Dmax]    int32
    c2_material: torch.Tensor  # [R, Dmax]    int32
    L2: torch.Tensor  # [R]     f32
    d_o: torch.Tensor  # [R, 3]  f32

    # Vertex data.
    v_pos: torch.Tensor  # [R, 3]  f32
    v_normal: torch.Tensor  # [R, 3]  f32
    v_material: torch.Tensor  # [R]     int32
    weight: torch.Tensor  # [R]     f32   per-vertex patch area A_patch (op-1 weights)
    cos_i: torch.Tensor  # [R]     f32
    cos_o: torch.Tensor  # [R]     f32
    patch_row: torch.Tensor  # [R]     int64  (Op B only; -1 for ensemble)

    @property
    def row_count(self) -> int:
        return int(self.tx_id.shape[0])

    @property
    def device(self) -> torch.device:
        return self.tx_id.device

    def validate(self) -> None:
        """Assert the frozen shape / dtype / device / contiguity contract."""

        r = int(self.tx_id.shape[0])
        d = KMAX_AD_DEPTH
        device = self.tx_id.device
        expected: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            "tx_id": ((r,), torch.int32),
            "rx_id": ((r,), torch.int32),
            "sample_index": ((r,), torch.int64),
            "d1": ((r,), torch.int32),
            "d2": ((r,), torch.int32),
            "c1_positions": ((r, d, 3), torch.float32),
            "c1_normals": ((r, d, 3), torch.float32),
            "c1_primitive": ((r, d), torch.int32),
            "c1_material": ((r, d), torch.int32),
            "L1": ((r,), torch.float32),
            "d_i": ((r, 3), torch.float32),
            "c2_positions": ((r, d, 3), torch.float32),
            "c2_normals": ((r, d, 3), torch.float32),
            "c2_primitive": ((r, d), torch.int32),
            "c2_material": ((r, d), torch.int32),
            "L2": ((r,), torch.float32),
            "d_o": ((r, 3), torch.float32),
            "v_pos": ((r, 3), torch.float32),
            "v_normal": ((r, 3), torch.float32),
            "v_material": ((r,), torch.int32),
            "weight": ((r,), torch.float32),
            "cos_i": ((r,), torch.float32),
            "cos_o": ((r,), torch.float32),
            "patch_row": ((r,), torch.int64),
        }
        for spec in _dataclass_fields(self):
            value = getattr(self, spec.name)
            shape, dtype = expected[spec.name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"ScatterChainDiscovery.{spec.name} must be a tensor")
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"ScatterChainDiscovery.{spec.name} has shape {tuple(value.shape)}, "
                    f"expected {shape}"
                )
            if value.dtype != dtype:
                raise TypeError(
                    f"ScatterChainDiscovery.{spec.name} has dtype {value.dtype}, "
                    f"expected {dtype}"
                )
            if value.device != device:
                raise ValueError(
                    f"ScatterChainDiscovery.{spec.name} is on {value.device}, "
                    f"expected {device}"
                )
            if not value.is_contiguous():
                raise ValueError(
                    f"ScatterChainDiscovery.{spec.name} must be C-contiguous"
                )
        # Structural row invariants (plan 10a section 2).
        if r > 0:
            if bool(((self.d1 < 0) | (self.d1 > d)).any()):
                raise ValueError("d1 out of [0, Dmax]")
            if bool(((self.d2 < 0) | (self.d2 > d)).any()):
                raise ValueError("d2 out of [0, Dmax]")
            if bool(((self.d1 + self.d2) < 1).any()):
                raise ValueError("chain rows must have d1 + d2 >= 1")


@dataclass(frozen=True)
class ChainSamples:
    """Chain-sample vertex set drawn on the ensemble scatter faces."""

    position: torch.Tensor  # [S, 3] f32
    normal: torch.Tensor  # [S, 3] f32
    material_id: torch.Tensor  # [S]    int32
    face_id: torch.Tensor  # [S]    int32
    weight: torch.Tensor  # [S]    f32   per-sample patch area (A_patch)


# ---------------------------------------------------------------------------
# Pure structural helpers (device-agnostic; unit-tested without a scene).
# ---------------------------------------------------------------------------


def _normalize(vec: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return geometry_kernels.deterministic_normalize_vec3(vec.contiguous(), eps=eps)


def _pad_bounce_block(
    values: torch.Tensor, depth: int, dmax: int, fill: float | int
) -> torch.Tensor:
    """Right-pad the bounce axis (axis 1) of a ``[M, depth, ...]`` block to Dmax."""

    if depth == dmax:
        return values.contiguous()
    if depth > dmax:
        raise ValueError(f"leg depth {depth} exceeds Dmax {dmax}")
    tail_shape = (values.shape[0], dmax - depth, *tuple(values.shape[2:]))
    tail = torch.full(
        tail_shape, fill, device=values.device, dtype=values.dtype
    )
    return torch.cat((values, tail), dim=1).contiguous()


def _equi_join_indices(
    a_key: torch.Tensor, b_key: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized equi-join: return ``(ai, bi)`` with ``a_key[ai] == b_key[bi]``.

    Produces every match combination (Cartesian per shared key), fully on device
    (no host sync per element). Row order is grouped by ascending dense key then
    by the input order within each side (stable), so downstream stable sorts make
    the final order fully deterministic.
    """

    device = a_key.device
    if a_key.numel() == 0 or b_key.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int64)
        return empty, empty
    keys = torch.unique(torch.cat((a_key, b_key)))
    k = int(keys.numel())
    a_id = torch.searchsorted(keys, a_key)
    b_id = torch.searchsorted(keys, b_key)
    a_count = torch.bincount(a_id, minlength=k)
    b_count = torch.bincount(b_id, minlength=k)
    pair_count = a_count * b_count
    total = int(pair_count.sum())
    if total == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int64)
        return empty, empty
    a_order = torch.argsort(a_id, stable=True)
    b_order = torch.argsort(b_id, stable=True)
    a_off = torch.cumsum(a_count, 0) - a_count
    b_off = torch.cumsum(b_count, 0) - b_count
    pair_off = torch.cumsum(pair_count, 0) - pair_count
    key_of_pair = torch.repeat_interleave(
        torch.arange(k, device=device, dtype=torch.int64), pair_count
    )
    local = (
        torch.arange(total, device=device, dtype=torch.int64)
        - pair_off[key_of_pair]
    )
    b_count_pair = b_count[key_of_pair]
    ai_local = local // b_count_pair
    bi_local = local % b_count_pair
    ai = a_order[a_off[key_of_pair] + ai_local]
    bi = b_order[b_off[key_of_pair] + bi_local]
    return ai, bi


def _stable_chain_order(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    sample_index: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
) -> torch.Tensor:
    """Deterministic row order: stable sort on (tx, rx, sample, d1, d2).

    Implemented as a lexicographic sort via a single composite key so the order
    is reproducible run-to-run (plan 10a section 2).
    """

    n = int(tx_id.shape[0])
    if n == 0:
        return torch.empty((0,), device=tx_id.device, dtype=torch.int64)
    d1l = d1.to(torch.int64)
    d2l = d2.to(torch.int64)
    txl = tx_id.to(torch.int64)
    rxl = rx_id.to(torch.int64)
    sidl = sample_index.to(torch.int64)
    # Mixed-radix composite key, least-significant first. Each field's radix is
    # (max + 1) so the packing is collision-free.
    def _radix(t: torch.Tensor) -> int:
        return int(t.max().item()) + 1 if t.numel() else 1

    r_d2 = _radix(d2l)
    r_d1 = _radix(d1l)
    r_sid = _radix(sidl)
    r_rx = _radix(rxl)
    key = d2l
    stride = r_d2
    key = key + d1l * stride
    stride *= r_d1
    key = key + sidl * stride
    stride *= r_sid
    key = key + rxl * stride
    stride *= r_rx
    key = key + txl * stride
    return torch.argsort(key, stable=True)


def _budget_chain_rows(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    strength: torch.Tensor,
    *,
    num_rx: int,
    cap: int,
) -> torch.Tensor:
    """Row indices of the strongest ``<= cap`` chain rows per (tx, rx) pair.

    Mirrors the single-bounce ``_keep_strongest_per_pair`` policy above,
    extended with the per-(tx, rx)
    ``scattering_chain_max_rows`` cap. ``strength`` is the geometric proxy
    ``1 / (L1^2 L2^2)`` (the dominant Op A gain factor); the final physical keep
    gate still runs in the native op.
    """

    pair = tx_id.to(torch.int64) * int(num_rx) + rx_id.to(torch.int64)
    order = torch.argsort(-strength.to(torch.float64), stable=True)
    order = order[torch.argsort(pair[order], stable=True)]
    sorted_pair = pair[order]
    row = torch.arange(order.numel(), device=order.device, dtype=torch.int64)
    first = torch.ones_like(sorted_pair, dtype=torch.bool)
    first[1:] = sorted_pair[1:] != sorted_pair[:-1]
    starts = torch.where(first, row, torch.zeros_like(row)).cummax(dim=0).values
    return order[(row - starts) < int(cap)]


# ---------------------------------------------------------------------------
# Chain-sample vertex set.
# ---------------------------------------------------------------------------


def build_chain_samples(
    compiled: object,
    config: "TopologyConfig",
    ensemble_faces: torch.Tensor,
    *,
    device: torch.device,
) -> ChainSamples | None:
    """Draw the chain-sample vertex set on the ensemble scatter faces.

    Uses the same R2 low-discrepancy barycentric scheme as the single-bounce
    ensemble sampler at the documented lower ``scattering_chain_samples_per_m2``
    density (plan 10a section 2 / ADR-021 D1). Returns ``None`` when no ensemble
    scatter face carries a sample.
    """

    if int(ensemble_faces.numel()) == 0:
        return None
    records = compiled.rayd.edge_records()
    faces = records.faces.to(torch.int64)[ensemble_faces]
    tri = records.vertices[faces]  # [F, 3, 3]
    normals = _normalize(records.face_normals[ensemble_faces])
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * torch.linalg.vector_norm(torch.cross(e1, e2, dim=-1), dim=-1)
    face_material = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    )[ensemble_faces]

    density = float(getattr(config, "scattering_chain_samples_per_m2", 2.0))
    counts = (
        torch.ceil(areas * density).to(torch.int64).clamp(1, _MAX_SAMPLES_PER_FACE)
    )
    face_of_sample, a, b = _r2_barycentric(counts, device)
    points = (
        tri[face_of_sample, 0]
        + a[:, None] * e1[face_of_sample]
        + b[:, None] * e2[face_of_sample]
    )
    if int(points.shape[0]) == 0:
        return None
    # Per-sample patch area A_patch = face_area / samples_on_face (op-1 weights
    # convention; the single-bounce `_ensemble_rows` above).
    weight = (areas / counts.to(torch.float32))[face_of_sample]
    return ChainSamples(
        position=points.contiguous(),
        normal=normals[face_of_sample].contiguous(),
        material_id=face_material[face_of_sample].to(torch.int32).contiguous(),
        face_id=ensemble_faces[face_of_sample].to(torch.int32).contiguous(),
        weight=weight.contiguous(),
    )


# ---------------------------------------------------------------------------
# Reflection scene geometry shared by both legs (mirrors reflection.py setup).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ReflectionGeometry:
    tri_a: torch.Tensor
    normals: torch.Tensor
    groups: dict[str, torch.Tensor]
    face_eps_r: torch.Tensor
    face_sigma_e: torch.Tensor
    face_mu_r: torch.Tensor
    face_gain: torch.Tensor
    face_material_id: torch.Tensor
    face_group_id: torch.Tensor


def _reflection_geometry(compiled: object, device: torch.device) -> _ReflectionGeometry:
    records = compiled.rayd.edge_records()
    vertices = records.vertices
    faces = records.faces.contiguous()
    normals = geometry_kernels.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    tri_a = topology_kernels.deterministic_face_anchor_points(
        vertices.contiguous(), faces
    )
    face_eps_r, face_sigma_e, face_mu_r, face_gain, _valid = face_material_tensors(
        compiled, device=device
    )
    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    face_group_source = compiled.geometry.face_surface_id.to(
        device=device, dtype=torch.long
    ).contiguous()
    groups = _cached_coplanar_face_groups(
        compiled.rayd, tri_a, normals, face_group_source
    )
    return _ReflectionGeometry(
        tri_a=tri_a,
        normals=normals,
        groups=groups,
        face_eps_r=face_eps_r,
        face_sigma_e=face_sigma_e,
        face_mu_r=face_mu_r,
        face_gain=face_gain,
        face_material_id=face_material_id,
        face_group_id=groups["face_group_id"],
    )


def _polyline_length(
    source: torch.Tensor,
    hits: torch.Tensor,
    endpoint: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    """Unfolded chain length source -> hits[0..depth-1] -> endpoint (structural).

    Sum of the RayD-owned segment lengths; equals the image-source length for a
    specular chain to f32. ``hits`` is ``[M, depth, 3]`` (may be depth 0).
    """

    if depth == 0:
        return torch.linalg.vector_norm(endpoint - source, dim=-1)
    seg = torch.linalg.vector_norm(hits[:, 0] - source, dim=-1)
    for bounce in range(1, depth):
        seg = seg + torch.linalg.vector_norm(hits[:, bounce] - hits[:, bounce - 1], dim=-1)
    seg = seg + torch.linalg.vector_norm(endpoint - hits[:, depth - 1], dim=-1)
    return seg


def _visibility(
    rayd: object, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    count = int(start.shape[0])
    if count == 0:
        return torch.empty((0,), device=start.device, dtype=torch.bool)
    handle = rayd.require_resource()
    masks = []
    for lo in range(0, count, _VISIBILITY_CHUNK):
        hi = min(lo + _VISIBILITY_CHUNK, count)
        masks.append(
            geometry_kernels.rayd_visibility_forward(
                handle, start[lo:hi].contiguous(), end[lo:hi].contiguous(), None
            )[0]
        )
    return torch.cat(masks)


def _empty_leg(device: torch.device) -> dict[str, torch.Tensor]:
    dmax = KMAX_AD_DEPTH
    return {
        "sample_index": torch.empty((0,), device=device, dtype=torch.int64),
        "depth": torch.empty((0,), device=device, dtype=torch.int32),
        "positions": torch.empty((0, dmax, 3), device=device, dtype=torch.float32),
        "normals": torch.empty((0, dmax, 3), device=device, dtype=torch.float32),
        "primitive": torch.empty((0, dmax), device=device, dtype=torch.int32),
        "material": torch.empty((0, dmax), device=device, dtype=torch.int32),
        "length": torch.empty((0,), device=device, dtype=torch.float32),
        "endpoint_dir": torch.empty((0, 3), device=device, dtype=torch.float32),
    }


def _leg_batch(
    *,
    sample_index: torch.Tensor,
    depth: int,
    hits: torch.Tensor,
    normals: torch.Tensor,
    primitive: torch.Tensor,
    material: torch.Tensor,
    source: torch.Tensor,
    endpoint: torch.Tensor,
    reverse: bool,
) -> dict[str, torch.Tensor]:
    """Pack one same-depth EPC batch into a padded leg record.

    ``hits``/``normals`` are ``[M, depth, 3]`` ordered source-first;
    ``primitive``/``material`` are ``[M, depth]``. For C2 (``reverse=True``) the
    bounce axis is flipped so the leg is indexed from the vertex outward toward
    RX, and ``endpoint_dir`` is the outgoing direction leaving the vertex; for C1
    (``reverse=False``) ``endpoint_dir`` is the incident direction arriving at
    the vertex. ``endpoint`` is the vertex position.
    """

    device = sample_index.device
    dmax = KMAX_AD_DEPTH
    m = int(sample_index.shape[0])
    length = _polyline_length(source, hits, endpoint, depth)
    if depth == 0:
        pos = torch.zeros((m, dmax, 3), device=device, dtype=torch.float32)
        nrm = torch.zeros((m, dmax, 3), device=device, dtype=torch.float32)
        prim = torch.full((m, dmax), -1, device=device, dtype=torch.int32)
        mat = torch.full((m, dmax), -1, device=device, dtype=torch.int32)
        # Empty leg: incident/outgoing direction is the direct source<->vertex ray.
        direction = (
            _normalize(endpoint - source) if not reverse else _normalize(source - endpoint)
        )
        return {
            "sample_index": sample_index.to(torch.int64),
            "depth": torch.zeros((m,), device=device, dtype=torch.int32),
            "positions": pos,
            "normals": nrm,
            "primitive": prim,
            "material": mat,
            "length": length.to(torch.float32),
            "endpoint_dir": direction.to(torch.float32),
        }

    # The hit nearest the vertex is the last source-first bounce.
    near_vertex_hit = hits[:, depth - 1]
    if reverse:
        # v_s -> RX order: flip the valid bounce block, outgoing from the vertex.
        hits = torch.flip(hits, dims=(1,))
        normals = torch.flip(normals, dims=(1,))
        primitive = torch.flip(primitive, dims=(1,))
        material = torch.flip(material, dims=(1,))
        direction = _normalize(near_vertex_hit - endpoint)
    else:
        # TX -> v_s order preserved; incident direction arrives at the vertex.
        direction = _normalize(endpoint - near_vertex_hit)

    return {
        "sample_index": sample_index.to(torch.int64),
        "depth": torch.full((m,), int(depth), device=device, dtype=torch.int32),
        "positions": _pad_bounce_block(hits.to(torch.float32), depth, dmax, 0.0),
        "normals": _pad_bounce_block(normals.to(torch.float32), depth, dmax, 0.0),
        "primitive": _pad_bounce_block(primitive.to(torch.int32), depth, dmax, -1),
        "material": _pad_bounce_block(material.to(torch.int32), depth, dmax, -1),
        "length": length.to(torch.float32),
        "endpoint_dir": direction.to(torch.float32),
    }


def _gather_leg(
    *,
    compiled: object,
    geom: _ReflectionGeometry,
    source: torch.Tensor,
    samples: ChainSamples,
    scene_diagonal: torch.Tensor,
    max_leg_depth: int,
    reverse: bool,
) -> dict[str, torch.Tensor]:
    """Enumerate specular reflection chains ``source -> {samples}`` at every leg
    depth ``0..max_leg_depth`` and pack them into a concatenated padded leg table.

    All hit geometry is produced by the native RayD EPC; this only selects,
    reverses (for C2), pads, and derives lengths/directions structurally.
    """

    device = samples.position.device
    rayd = compiled.rayd
    sample_pos = samples.position
    sample_normal = samples.normal
    s_count = int(sample_pos.shape[0])
    batches: list[dict[str, torch.Tensor]] = []
    source_row = source.reshape(1, 3)
    tx_power_ref = torch.ones((1,), device=device, dtype=torch.float32)

    # Depth 0: direct source -> vertex visibility (no reflection).
    if max_leg_depth >= 0:
        eps = _offset_eps(sample_pos, scene_diagonal)
        to_src = source_row - sample_pos
        side = torch.sign((to_src * sample_normal).sum(-1))
        side = torch.where(side == 0.0, torch.ones_like(side), side)
        n_o = sample_normal * side[:, None]
        offset_points = sample_pos + n_o * eps[:, None]
        visible = _visibility(
            rayd,
            source_row.expand(s_count, 3).contiguous(),
            offset_points,
        )
        idx = torch.nonzero(visible, as_tuple=False).reshape(-1)
        if int(idx.numel()) > 0:
            batches.append(
                _leg_batch(
                    sample_index=idx,
                    depth=0,
                    hits=torch.zeros((int(idx.numel()), 0, 3), device=device, dtype=torch.float32),
                    normals=torch.zeros((int(idx.numel()), 0, 3), device=device, dtype=torch.float32),
                    primitive=torch.zeros((int(idx.numel()), 0), device=device, dtype=torch.int32),
                    material=torch.zeros((int(idx.numel()), 0), device=device, dtype=torch.int32),
                    source=source,
                    endpoint=sample_pos[idx],
                    reverse=reverse,
                )
            )

    # Depth 1: order-1 EPC against the samples as virtual receivers.
    if max_leg_depth >= 1:
        _gather_leg_order1(
            compiled=compiled,
            geom=geom,
            source=source,
            samples=samples,
            tx_power_ref=tx_power_ref,
            reverse=reverse,
            batches=batches,
        )

    # Depth 2..max_leg_depth: multibounce EPC.
    if max_leg_depth >= 2:
        _gather_leg_multibounce(
            compiled=compiled,
            geom=geom,
            source=source,
            samples=samples,
            tx_power_ref=tx_power_ref,
            max_leg_depth=max_leg_depth,
            reverse=reverse,
            batches=batches,
        )

    if not batches:
        return _empty_leg(device)
    return {name: torch.cat([b[name] for b in batches]).contiguous() for name in _empty_leg(device)}


def _gather_leg_order1(
    *,
    compiled: object,
    geom: _ReflectionGeometry,
    source: torch.Tensor,
    samples: ChainSamples,
    tx_power_ref: torch.Tensor,
    reverse: bool,
    batches: list[dict[str, torch.Tensor]],
) -> None:
    groups = geom.groups
    group_count = int(groups["group_count"])
    if group_count <= 0:
        return
    plan = prepare_reflection_order1_plan(
        group_count=group_count,
        representative_faces=groups["representative_faces"].contiguous(),
        face_group_id=geom.face_group_id,
    )
    source_positions = source.reshape(1, 3)
    for request in iter_reflection_order1_epc_requests(
        plan,
        tx_positions=source_positions,
        rx_positions=samples.position,
        tri_a=geom.tri_a,
        normals=geom.normals,
        trace_group_chains=lambda tx, *, face_group_id, max_depth: _trace_group_chains(
            compiled.rayd, tx, face_group_id=face_group_id, max_depth=max_depth
        ),
    ):
        epc_inputs = request.epc_inputs
        epc = query_reflection_epc(
            ReflectionEpcQuery(
                rayd=compiled.rayd,
                source=epc_inputs["tx_batch"],
                receiver=epc_inputs["rx_batch"],
                active=None,
                expected_prim_ids=epc_inputs["sequence_batch"],
                direct_plane_points=epc_inputs["direct_plane_points"],
                direct_plane_normals=epc_inputs["direct_plane_normals"],
                surface_group_id=groups["surface_group_id"],
                surface_group_size=groups["surface_group_size"],
                surface_group_members=groups["surface_group_members"],
                max_bounces=1,
                visibility_ignore_mode=1,
            )
        )
        selected = topology_kernels.deterministic_reflection_order1_compact(
            visible=epc.visible,
            epc_faces=epc.resolved_prim_ids,
            epc_hits=epc.hit_positions,
            epc_normals=epc.normals,
            sequence_batch=epc_inputs["sequence_batch"],
            rx_indices=epc_inputs["rx_indices"],
            tx=source,
            rx_positions=samples.position,
            tx_power=tx_power_ref,
            tx_index=0,
            face_eps_r=geom.face_eps_r,
            face_sigma_e=geom.face_sigma_e,
            face_mu_r=geom.face_mu_r,
            face_gain=geom.face_gain,
            face_material_id=geom.face_material_id,
            grouped_export=True,
        )
        m = int(selected["selected_faces"].shape[0])
        if m == 0:
            continue
        sample_index = selected["selected_rx_id"].to(torch.int64)
        batches.append(
            _leg_batch(
                sample_index=sample_index,
                depth=1,
                hits=selected["selected_points"].reshape(m, 1, 3),
                normals=selected["selected_normals"].reshape(m, 1, 3),
                primitive=selected["selected_faces"].reshape(m, 1),
                material=selected["material_id"].reshape(m, 1),
                source=source,
                endpoint=samples.position[sample_index],
                reverse=reverse,
            )
        )


def _gather_leg_multibounce(
    *,
    compiled: object,
    geom: _ReflectionGeometry,
    source: torch.Tensor,
    samples: ChainSamples,
    tx_power_ref: torch.Tensor,
    max_leg_depth: int,
    reverse: bool,
    batches: list[dict[str, torch.Tensor]],
) -> None:
    groups = geom.groups
    group_count = int(groups["group_count"])
    if group_count <= 0:
        return
    plan = prepare_reflection_multibounce_plan(
        group_count=group_count,
        representative_faces=groups["representative_faces"].contiguous(),
        face_group_id=geom.face_group_id,
        min_depth=2,
        max_depth=int(max_leg_depth),
    )
    source_positions = source.reshape(1, 3)
    for request in iter_reflection_multibounce_epc_requests(
        plan,
        tx_positions=source_positions,
        rx_positions=samples.position,
        sequence_reference=tx_power_ref,
        tri_a=geom.tri_a,
        normals=geom.normals,
        trace_group_chains=lambda tx, *, face_group_id, max_depth: _trace_group_chains(
            compiled.rayd, tx, face_group_id=face_group_id, max_depth=max_depth
        ),
        record_candidate_count=lambda candidate_count: None,
    ):
        depth = int(request.depth)
        epc_inputs = request.epc_inputs
        epc = query_reflection_epc(
            ReflectionEpcQuery(
                rayd=compiled.rayd,
                source=epc_inputs["tx_batch"],
                receiver=epc_inputs["rx_batch"],
                active=None,
                expected_prim_ids=epc_inputs["sequence_batch"],
                direct_plane_points=epc_inputs["direct_plane_points"],
                direct_plane_normals=epc_inputs["direct_plane_normals"],
                surface_group_id=groups["surface_group_id"],
                surface_group_size=groups["surface_group_size"],
                surface_group_members=groups["surface_group_members"],
                max_bounces=depth,
                visibility_ignore_mode=1,
            )
        )
        selected = topology_kernels.deterministic_reflection_sequence_compact(
            visible=epc.visible,
            epc_sequences=epc.resolved_prim_ids,
            epc_hits=epc.hit_positions,
            epc_normals=epc.normals,
            rx_indices=epc_inputs["rx_indices"],
            tx=source,
            rx_positions=samples.position,
            tx_power=tx_power_ref,
            tx_index=0,
            face_eps_r=geom.face_eps_r,
            face_sigma_e=geom.face_sigma_e,
            face_mu_r=geom.face_mu_r,
            face_gain=geom.face_gain,
            face_material_id=geom.face_material_id,
            max_count=-1,
        )
        m = int(selected["selected_sequences"].shape[0])
        if m == 0:
            continue
        sample_index = selected["selected_rx_id"].to(torch.int64)
        batches.append(
            _leg_batch(
                sample_index=sample_index,
                depth=depth,
                hits=selected["selected_hits"],
                normals=selected["selected_normals"],
                primitive=selected["selected_sequences"],
                material=selected["material_sequence"],
                source=source,
                endpoint=samples.position[sample_index],
                reverse=reverse,
            )
        )


def _trace_group_chains(
    rayd: object,
    tx: torch.Tensor,
    *,
    face_group_id: torch.Tensor,
    max_depth: int,
    ray_count: int = 262_144,
) -> torch.Tensor:
    """Trace specular chains from a source and map hits to plane-group ids.

    Mirrors ``interactions/reflection._discovered_group_chains`` (used only by
    the non-exhaustive discovery branch of the shared iterators).
    """

    from witwin.channel.kernels.topology import (
        mc_sample_directions,
    )

    device = face_group_id.device
    ray_o = tx.reshape(1, 3).expand(ray_count, 3).contiguous()
    ray_d = mc_sample_directions(ray_count, tx.reshape(1, 3))
    ray_tmax = torch.empty((0,), device=device, dtype=torch.float32)
    out = geometry_kernels.rayd_trace_reflections_forward(
        rayd.require_resource(), ray_o, ray_d, ray_tmax, None, int(max_depth)
    )
    prim_chain = out[2].to(dtype=torch.long).reshape(ray_count, int(max_depth))
    chains = torch.full_like(prim_chain, -1)
    hit = prim_chain >= 0
    chains[hit] = face_group_id[prim_chain[hit]]
    return chains


# ---------------------------------------------------------------------------
# Public discovery entry.
# ---------------------------------------------------------------------------


def discover_scatter_chains(
    compiled: object,
    config: "TopologyConfig",
    *,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    samples: ChainSamples,
    scene_diagonal: torch.Tensor,
) -> ScatterChainDiscovery | None:
    """Enumerate and join C1/C2 specular chains around each chain vertex.

    Runs the RayD reflection EPC for ``tx -> {samples}`` (C1) and
    ``rx -> {samples}`` (C2), joins the legs on the sample index (excluding the
    ``d1 = d2 = 0`` single-bounce collapse), enforces ``d1 + d2 <=
    scattering_chain_max_depth`` with each leg ``<= KMAX_AD_DEPTH``, budgets by
    keep-strongest per (tx, rx) up to ``scattering_chain_max_rows``, and returns
    the frozen :class:`ScatterChainDiscovery` in deterministic row order. Returns
    ``None`` when no chain row survives.
    """

    device = samples.position.device
    max_chain_depth = int(getattr(config, "scattering_chain_max_depth", 0))
    if max_chain_depth < 1:
        return None
    max_leg_depth = min(max_chain_depth, KMAX_AD_DEPTH)
    max_rows = int(getattr(config, "scattering_chain_max_rows", 256))
    num_tx = int(tx_positions.shape[0])
    num_rx = int(rx_positions.shape[0])
    if num_tx == 0 or num_rx == 0 or int(samples.position.shape[0]) == 0:
        return None

    geom = _reflection_geometry(compiled, device)
    c1_tables, c2_tables = _enumerate_leg_tables(
        compiled=compiled,
        geom=geom,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        samples=samples,
        scene_diagonal=scene_diagonal,
        max_leg_depth=max_leg_depth,
    )
    rows = _join_leg_tables(
        c1_tables=c1_tables,
        c2_tables=c2_tables,
        num_tx=num_tx,
        num_rx=num_rx,
        max_chain_depth=max_chain_depth,
        samples=samples,
        device=device,
    )
    if not rows:
        return None
    return _budget_and_assemble(rows, num_rx=num_rx, max_rows=max_rows)


def _enumerate_leg_tables(
    *,
    compiled: object,
    geom: "_ReflectionGeometry",
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    samples: ChainSamples,
    scene_diagonal: torch.Tensor,
    max_leg_depth: int,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """Trace the per-source specular legs tagged by endpoint id (tx C1, rx C2)."""

    c1_tables = [
        _gather_leg(
            compiled=compiled,
            geom=geom,
            source=tx_positions[i],
            samples=samples,
            scene_diagonal=scene_diagonal,
            max_leg_depth=max_leg_depth,
            reverse=False,
        )
        for i in range(int(tx_positions.shape[0]))
    ]
    c2_tables = [
        _gather_leg(
            compiled=compiled,
            geom=geom,
            source=rx_positions[j],
            samples=samples,
            scene_diagonal=scene_diagonal,
            max_leg_depth=max_leg_depth,
            reverse=True,
        )
        for j in range(int(rx_positions.shape[0]))
    ]
    return c1_tables, c2_tables


def _join_leg_tables(
    *,
    c1_tables: list[dict[str, torch.Tensor]],
    c2_tables: list[dict[str, torch.Tensor]],
    num_tx: int,
    num_rx: int,
    max_chain_depth: int,
    samples: ChainSamples,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Join C1/C2 legs per (tx, rx) on the sample index into chain-row blocks.

    Excludes the ``d1 = d2 = 0`` single-bounce collapse and enforces the total
    ``d1 + d2 <= scattering_chain_max_depth`` gate before packing each survivor.
    """

    rows: list[dict[str, torch.Tensor]] = []
    for i in range(num_tx):
        c1 = c1_tables[i]
        if int(c1["sample_index"].numel()) == 0:
            continue
        for j in range(num_rx):
            c2 = c2_tables[j]
            if int(c2["sample_index"].numel()) == 0:
                continue
            ai, bi = _equi_join_indices(c1["sample_index"], c2["sample_index"])
            if int(ai.numel()) == 0:
                continue
            d1 = c1["depth"][ai]
            d2 = c2["depth"][bi]
            total = d1.to(torch.int64) + d2.to(torch.int64)
            keep = (total >= 1) & (total <= max_chain_depth)
            if not bool(keep.any()):
                continue
            ai, bi = ai[keep], bi[keep]
            rows.append(
                _join_pair_rows(
                    tx_index=i,
                    rx_index=j,
                    c1=c1,
                    c2=c2,
                    ai=ai,
                    bi=bi,
                    samples=samples,
                    device=device,
                )
            )
    return rows


def _budget_and_assemble(
    rows: list[dict[str, torch.Tensor]],
    *,
    num_rx: int,
    max_rows: int,
) -> ScatterChainDiscovery:
    """Merge chain rows, budget per (tx, rx), sort deterministically, and freeze."""

    merged = {name: torch.cat([r[name] for r in rows]) for name in rows[0]}

    strength = 1.0 / (
        merged["L1"].to(torch.float64).square() * merged["L2"].to(torch.float64).square()
    ).clamp_min(1.0e-30)
    keep = _budget_chain_rows(
        merged["tx_id"], merged["rx_id"], strength, num_rx=num_rx, cap=max_rows
    )
    merged = {name: value[keep] for name, value in merged.items()}

    order = _stable_chain_order(
        merged["tx_id"], merged["rx_id"], merged["sample_index"], merged["d1"], merged["d2"]
    )
    merged = {name: value[order].contiguous() for name, value in merged.items()}

    discovery = _assemble_discovery(merged)
    discovery.validate()
    return discovery


def _join_pair_rows(
    *,
    tx_index: int,
    rx_index: int,
    c1: dict[str, torch.Tensor],
    c2: dict[str, torch.Tensor],
    ai: torch.Tensor,
    bi: torch.Tensor,
    samples: ChainSamples,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    n = int(ai.numel())
    sample_index = c1["sample_index"][ai]
    d_i = c1["endpoint_dir"][ai]
    d_o = c2["endpoint_dir"][bi]
    v_normal = samples.normal[sample_index]
    l1 = c1["length"][ai]
    l2 = c2["length"][bi]
    return {
        "tx_id": torch.full((n,), tx_index, device=device, dtype=torch.int32),
        "rx_id": torch.full((n,), rx_index, device=device, dtype=torch.int32),
        "sample_index": sample_index.to(torch.int64),
        "d1": c1["depth"][ai].to(torch.int32),
        "d2": c2["depth"][bi].to(torch.int32),
        "c1_positions": c1["positions"][ai],
        "c1_normals": c1["normals"][ai],
        "c1_primitive": c1["primitive"][ai],
        "c1_material": c1["material"][ai],
        "L1": l1.to(torch.float32),
        "d_i": d_i,
        "c2_positions": c2["positions"][bi],
        "c2_normals": c2["normals"][bi],
        "c2_primitive": c2["primitive"][bi],
        "c2_material": c2["material"][bi],
        "L2": l2.to(torch.float32),
        "d_o": d_o,
        "v_pos": samples.position[sample_index],
        "v_normal": v_normal,
        "v_material": samples.material_id[sample_index].to(torch.int32),
        # Per-vertex patch area A_patch (op-1 weights convention); the native Op A
        # radiometric assembly uses weights + 1/(L1^2 L2^2).
        "weight": samples.weight[sample_index].to(torch.float32),
        "cos_i": (d_i * v_normal).sum(-1).abs().to(torch.float32),
        "cos_o": (d_o * v_normal).sum(-1).abs().to(torch.float32),
        # patch_row is Op B only; ensemble (Op A) rows carry -1 (plan 10a section 2).
        "patch_row": torch.full((n,), -1, device=device, dtype=torch.int64),
    }


def _assemble_discovery(merged: dict[str, torch.Tensor]) -> ScatterChainDiscovery:
    return ScatterChainDiscovery(
        tx_id=merged["tx_id"],
        rx_id=merged["rx_id"],
        sample_index=merged["sample_index"],
        d1=merged["d1"],
        d2=merged["d2"],
        c1_positions=merged["c1_positions"],
        c1_normals=merged["c1_normals"],
        c1_primitive=merged["c1_primitive"],
        c1_material=merged["c1_material"],
        L1=merged["L1"],
        d_i=merged["d_i"],
        c2_positions=merged["c2_positions"],
        c2_normals=merged["c2_normals"],
        c2_primitive=merged["c2_primitive"],
        c2_material=merged["c2_material"],
        L2=merged["L2"],
        d_o=merged["d_o"],
        v_pos=merged["v_pos"],
        v_normal=merged["v_normal"],
        v_material=merged["v_material"],
        weight=merged["weight"],
        cos_i=merged["cos_i"],
        cos_o=merged["cos_o"],
        patch_row=merged["patch_row"],
    )


# -------------------------------------------------------------------------
# Scatter-chain append (was scattering_chain_append.py)
# -------------------------------------------------------------------------
#
# ADR-021 D1 enumerated scatter-chain append path (Deterministic + Path).
#
# Owns everything reachable only when ``scattering_chain_max_depth >= 1``: the
# chain-sample building, discovery invocation, native chain-ensemble facade
# dispatch, and multi-slot chain row assembly onto canonical typed path
# contracts. ``append_scattering_evaluated_paths`` above calls the single public
# entry ``append_chain_scattering_paths`` in this section.
#
# Discovery is the preceding section; this one consumes its
# ``ScatterChainDiscovery`` contract and appends the joined, budgeted rows. The vertex-frame geometry mirrors the single-bounce ensemble
# convention and reuses ``_unit``/``_stable_tangent`` plus the scene-owned
# phase-screen assignment resolver so the two paths never diverge.


# ADR-021 D5 has never been reachable from this append path. Before the kernel
# facades were lifted into ``witwin.channel.kernels``, the Op A ``_ad`` probe
# below read the single-bounce ``scattering/kernels/autograd.py`` namespace,
# which never carried a chain symbol, so ``ad_mode != "none"`` always refused
# here and the AD dispatch a few lines further down was unreachable. The
# facades are one module now, so that refusal is stated instead of implied.
# Turning it on is an ADR-021 D5 numerical decision with its own acceptance
# evidence, not a layout move.
_ADR021_D5_CHAIN_AD_WIRED = False


def _ensemble_scatter_faces(
    compiled: object, screens: dict[int, PhaseScreen], *, device: torch.device
) -> torch.Tensor:
    """Ensemble (non-realization) rough scatter faces (single-bounce parity)."""

    face_material = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    )
    scatter_face = (
        compiled.materials.scatter_model_id.to(device=device)[face_material] == 1
    )
    face_structure = compiled.geometry.face_structure_id.to(
        device=device, dtype=torch.int64
    )
    realization_face = torch.zeros_like(scatter_face)
    for structure_index in screens:
        structure_id = int(compiled.assignments.structure_id[structure_index])
        realization_face |= face_structure == structure_id
    return torch.nonzero(scatter_face & ~realization_face, as_tuple=False).reshape(-1)


def _chain_bounce_material(
    face_param: torch.Tensor, primitive: torch.Tensor, pad: float
) -> torch.Tensor:
    """Gather a per-face Fresnel parameter onto a padded ``[R, Dmax]`` leg block.

    Padded slots (``primitive < 0``) take the identity ``pad`` value so the
    native chain transport treats them as no-ops (ADR-021 section 1 padding).
    """

    valid = primitive >= 0
    idx = primitive.clamp_min(0).to(torch.int64)
    gathered = face_param.index_select(0, idx.reshape(-1)).reshape(primitive.shape)
    return torch.where(valid, gathered, torch.full_like(gathered, pad)).contiguous()


def _chain_vertex_frame(
    discovery: ScatterChainDiscovery, rough_axis_rad: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Vertex roughness frame + incident local coords (single-bounce parity).

    Mirrors ``_ensemble_rows``: the mean-plane normal is flipped toward the
    incident side, the principal roughness axes ``t1r``/``t2r`` are rotated by the
    material azimuth, and the incident direction is expressed in the local table
    frame (``wi_local``).
    """

    d_i = discovery.d_i
    v_material = discovery.v_material.to(torch.int64)
    side = torch.sign((-d_i * discovery.v_normal).sum(-1))
    side = torch.where(side == 0.0, torch.ones_like(side), side)
    n_o = _unit(discovery.v_normal * side[:, None])
    backup_axis = _stable_tangent(n_o)
    angle = rough_axis_rad.to(device=n_o.device, dtype=torch.float32)[v_material]
    t1 = backup_axis
    t2 = torch.cross(n_o, t1, dim=-1)
    t1r = t1 * torch.cos(angle)[:, None] + t2 * torch.sin(angle)[:, None]
    t2r = torch.cross(n_o, t1r, dim=-1)
    wi_hat = -d_i
    cos_i = (wi_hat * n_o).sum(-1)
    wi_local = torch.stack(
        ((wi_hat * t1r).sum(-1), (wi_hat * t2r).sum(-1), cos_i), dim=-1
    )
    return {
        "n_o": n_o.contiguous(),
        "t1r": t1r.contiguous(),
        "t2r": t2r.contiguous(),
        "backup_axis": backup_axis.contiguous(),
        "cos_i": cos_i.contiguous(),
        "wi_local": wi_local.contiguous(),
    }


def _chain_ensemble_evaluate(
    scene: Scene,
    compiled: object,
    config: TopologyConfig,
    discovery: ScatterChainDiscovery,
    *,
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    device: torch.device,
    ad_mode: str,
) -> dict[str, torch.Tensor]:
    """Dispatch the native ADR-021 Op A chain-ensemble kernel per transmitter.

    Builds the AS-BUILT native Op A argument set from the discovery contract plus
    the resident Kirchhoff tables and per-face material tensors, then dispatches
    the native ``scattering_chain_ensemble_eval`` (or its ``_ad`` companion) once
    per transmitter so the AD-live radiometric ``coef`` (which carries the per-tx
    power) crosses the ABI as a scalar exactly as the single-bounce op-1 path.
    The endpoint positions ``source``/``vertex``/``target`` feed the C1/C2
    transport and ``weights`` (per-vertex ``A_patch``) follows the op-1
    convention (with the ``1/(L1^2 L2^2)`` spreading applied in-kernel).
    """

    op_a = getattr(scattering_kernels, "scattering_chain_ensemble_eval", None)
    op_a_ad = (
        getattr(scattering_kernels, "scattering_chain_ensemble_eval_ad", None)
        if _ADR021_D5_CHAIN_AD_WIRED
        else None
    )
    if op_a is None or (ad_mode != "none" and op_a_ad is None):
        raise RuntimeError(
            "ADR-021 D1 requires the native Op A chain facade "
            "'scattering_chain_ensemble_eval' (and its '_ad' companion under "
            "ad_mode != 'none'); it is owned by the ADR-021 D2 native chain "
            "kernels (witwin.channel.kernels.scattering) and is not yet wired "
            "into this append path. Enable scattering_chain_max_depth only "
            "once the D5 wave has landed."
        )

    ad_enabled = ad_mode != "none"
    face_eps, face_sigma, face_mu, face_gain, _valid = face_material_tensors(
        compiled, device=device
    )
    face_thickness = face_material_thickness(compiled, device=device)

    def leg_material(primitive: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "eps_r": _chain_bounce_material(face_eps, primitive, 1.0),
            "sigma_e": _chain_bounce_material(face_sigma, primitive, 0.0),
            "mu_r": _chain_bounce_material(face_mu, primitive, 1.0),
            "gain": _chain_bounce_material(face_gain, primitive, 1.0),
            "thickness": _chain_bounce_material(face_thickness, primitive, 0.0),
        }

    c1 = leg_material(discovery.c1_primitive)
    c2 = leg_material(discovery.c2_primitive)
    frame = _chain_vertex_frame(discovery, compiled.materials.rough_axis_rad)

    stack = compiled.kirchhoff_resources.stack
    material_id = discovery.v_material.to(torch.int32).contiguous()
    rows = discovery.row_count

    frequency = float(scene.frequency)
    wavelength = C0 / frequency
    power_scale = wavelength**2 / (4.0 * math.pi) ** 2
    coef_scale_t = ensemble_coef_scale(scene, device, ad_enabled=ad_enabled)
    threshold = max(float(getattr(config, "scattering_power_threshold", 0.0)), 0.0)
    tx_id = discovery.tx_id.to(torch.int64)

    gain = torch.zeros((rows,), device=device, dtype=torch.float32)
    amplitude = torch.zeros((rows,), device=device, dtype=torch.float32)
    length = torch.zeros((rows,), device=device, dtype=torch.float32)
    keep = torch.zeros((rows,), device=device, dtype=torch.bool)

    # Discovery freezes rows in tx-major order. Reuse each existing device
    # selection mask as the RayD row-valid contract: narrowing the contiguous
    # tx block is a view (no allocation or kernel), while manufacturing an
    # all-true tensor here would violate the caller-owned validity contract.
    row_offset = 0
    for tx_index in torch.unique(tx_id).tolist():
        mask = tx_id == tx_index
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        row_count = int(idx.numel())
        if row_count == 0:
            continue
        row_valid = mask.narrow(0, row_offset, row_count)
        row_offset += row_count
        row_tx = discovery.tx_id[idx].to(torch.int64)
        row_rx = discovery.rx_id[idx].to(torch.int64)
        args = (
            row_valid,
            tx_pol.index_select(0, row_tx).contiguous(),
            rx_pol.index_select(0, row_rx).contiguous(),
            tx_positions.index_select(0, row_tx).contiguous(),
            discovery.v_pos[idx].contiguous(),
            rx_positions.index_select(0, row_rx).contiguous(),
            discovery.c1_positions[idx].contiguous(),
            discovery.c1_normals[idx].contiguous(),
            c1["eps_r"][idx],
            c1["sigma_e"][idx],
            c1["mu_r"][idx],
            c1["gain"][idx],
            c1["thickness"][idx],
            discovery.d1[idx].contiguous(),
            discovery.c2_positions[idx].contiguous(),
            discovery.c2_normals[idx].contiguous(),
            c2["eps_r"][idx],
            c2["sigma_e"][idx],
            c2["mu_r"][idx],
            c2["gain"][idx],
            c2["thickness"][idx],
            discovery.d2[idx].contiguous(),
            frame["n_o"][idx],
            frame["t1r"][idx],
            frame["t2r"][idx],
            frame["backup_axis"][idx],
            frame["wi_local"][idx],
            discovery.cos_i[idx].contiguous(),
            discovery.cos_o[idx].contiguous(),
            discovery.d_i[idx].contiguous(),
            discovery.d_o[idx].contiguous(),
            discovery.L1[idx].contiguous(),
            discovery.L2[idx].contiguous(),
            discovery.weight[idx].contiguous(),
            material_id[idx],
            stack.f_te_flat,
            stack.f_tm_flat,
            stack.table_offset,
            stack.table_dims,
            stack.material_slot,
        )
        if ad_enabled:
            evaluated = op_a_ad(
                *args,
                coef=float(tx_power[tx_index]) * coef_scale_t,
                threshold=threshold,
                frequency=frequency,
            )
        else:
            evaluated = op_a(
                *args,
                coef=float(tx_power[tx_index]) * power_scale,
                threshold=threshold,
                frequency_hz=frequency,
            )
        gain[idx] = evaluated["gain"].to(torch.float32)
        amplitude[idx] = evaluated["amplitude"].to(torch.float32)
        length[idx] = evaluated["length"].to(torch.float32)
        keep[idx] = evaluated["keep"]

    if row_offset != rows:
        raise RuntimeError("scatter-chain tx-major row coverage contract was violated")

    return {
        "gain": gain,
        "amplitude": amplitude,
        "length": length,
        "keep": keep,
        "n_o": frame["n_o"],
    }


def _chain_topology_slots(
    discovery: ScatterChainDiscovery,
    vertex_face: torch.Tensor,
    vertex_normal: torch.Tensor,
    width: int,
) -> dict[str, torch.Tensor]:
    """Build the multi-slot interaction sequence of every chain row.

    Slot layout per row (ADR-021 D1): ``[REFLECTION]*d1 + [SCATTERING] +
    [REFLECTION]*d2``; C1 fills the leading ``d1`` slots, the diffuse vertex sits
    at slot ``d1``, and C2 fills slots ``d1 + 1 .. d1 + d2``. Padded slots carry
    ``primitive/material = -1`` and ``interaction_type = 0`` (inactive).
    """

    device = discovery.device
    rows = discovery.row_count
    dmax = KMAX_AD_DEPTH
    d1 = discovery.d1.to(torch.int64)
    d2 = discovery.d2.to(torch.int64)
    prim = torch.full((rows, width), -1, device=device, dtype=torch.int32)
    mat = torch.full((rows, width), -1, device=device, dtype=torch.int32)
    itype = torch.zeros((rows, width), device=device, dtype=torch.int32)
    pos = torch.zeros((rows, width, 3), device=device, dtype=torch.float32)
    nrm = torch.zeros((rows, width, 3), device=device, dtype=torch.float32)
    ones = torch.ones((rows,), device=device, dtype=torch.int32)

    span = min(dmax, width)
    # C1 leg: leading slots 0..d1-1.
    for k in range(span):
        active = k < d1
        col = torch.where(active, discovery.c1_primitive[:, k], prim[:, k])
        prim[:, k] = col
        mat[:, k] = torch.where(active, discovery.c1_material[:, k], mat[:, k])
        itype[:, k] = torch.where(active, ones, itype[:, k])
        pos[:, k] = torch.where(
            active[:, None], discovery.c1_positions[:, k], pos[:, k]
        )
        nrm[:, k] = torch.where(
            active[:, None], discovery.c1_normals[:, k], nrm[:, k]
        )

    # Diffuse vertex at slot d1 (SCATTERING=8).
    row_idx = torch.arange(rows, device=device, dtype=torch.int64)
    prim[row_idx, d1] = vertex_face.to(torch.int32)
    mat[row_idx, d1] = discovery.v_material.to(torch.int32)
    itype[row_idx, d1] = 8
    pos[row_idx, d1] = discovery.v_pos
    nrm[row_idx, d1] = vertex_normal

    # C2 leg: slots d1+1 .. d1+d2.
    for k in range(span):
        active = k < d2
        rows_k = torch.nonzero(active, as_tuple=False).reshape(-1)
        if int(rows_k.numel()) == 0:
            continue
        slot = d1[rows_k] + 1 + k
        prim[rows_k, slot] = discovery.c2_primitive[rows_k, k]
        mat[rows_k, slot] = discovery.c2_material[rows_k, k]
        itype[rows_k, slot] = 1
        pos[rows_k, slot] = discovery.c2_positions[rows_k, k]
        nrm[rows_k, slot] = discovery.c2_normals[rows_k, k]

    return {
        "primitive_sequence": prim,
        "material_sequence": mat,
        "interaction_type": itype,
        "interaction_positions": pos,
        "interaction_normals": nrm,
    }


def _extend_evaluated_paths_chain(
    evaluated: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
    discovery: ScatterChainDiscovery,
    physics: dict[str, torch.Tensor],
    vertex_face: torch.Tensor,
    *,
    tx_power: torch.Tensor,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Append component_id=6 multi-slot scatter-chain rows to typed contracts."""

    topology = evaluated.topology
    geometry = evaluated.geometry
    fields = evaluated.fields
    device = topology.valid.device
    keep = physics["keep"]
    idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
    count = int(idx.numel())
    if count == 0:
        return evaluated, sidecars

    discovery = _index_discovery(discovery, idx)
    vertex_face = vertex_face[idx]
    vertex_normal = physics["n_o"][idx]
    gain = physics["gain"][idx]
    amplitude = physics["amplitude"][idx]
    length = physics["length"][idx]

    depth = (discovery.d1 + 1 + discovery.d2).to(torch.int32)
    chain_width = int(depth.max().item())
    base_width = int(topology.primitive_sequence.shape[1])
    width = max(base_width, chain_width, 1)

    slots = _chain_topology_slots(discovery, vertex_face, vertex_normal, width)
    zero = torch.zeros_like(amplitude)
    tx_power_row = tx_power.index_select(0, discovery.tx_id.to(torch.int64)).clamp_min(
        1.0e-30
    )
    coefficient = torch.complex(amplitude / tx_power_row.sqrt(), zero)
    field_xyz = torch.zeros((count, 3), device=device, dtype=torch.complex64)

    def cat(existing: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        return torch.cat((existing, new.to(existing.dtype))).contiguous()

    def pad_seq(seq: torch.Tensor, fill: int) -> torch.Tensor:
        if int(seq.shape[1]) >= width:
            return seq
        tail = torch.full(
            (seq.shape[0], width - int(seq.shape[1])),
            fill,
            device=device,
            dtype=seq.dtype,
        )
        return torch.cat((seq, tail), dim=1)

    def pad_vec(seq: torch.Tensor) -> torch.Tensor:
        if int(seq.shape[1]) >= width:
            return seq
        tail = torch.zeros(
            (seq.shape[0], width - int(seq.shape[1]), 3),
            device=device,
            dtype=seq.dtype,
        )
        return torch.cat((seq, tail), dim=1)

    new_topology = PathTopology(
        valid=cat(topology.valid, torch.ones((count,), device=device, dtype=torch.bool)),
        tx_id=cat(topology.tx_id, discovery.tx_id),
        rx_id=cat(topology.rx_id, discovery.rx_id),
        depth=cat(topology.depth, depth),
        component_id=cat(
            topology.component_id,
            torch.full((count,), 6, device=device, dtype=torch.int32),
        ),
        primitive_id=cat(topology.primitive_id, vertex_face),
        edge_id=cat(
            topology.edge_id,
            torch.full((count,), -1, device=device, dtype=torch.int32),
        ),
        material_id=cat(topology.material_id, discovery.v_material),
        primitive_sequence=torch.cat(
            (pad_seq(topology.primitive_sequence, -1), slots["primitive_sequence"])
        ).contiguous(),
        material_sequence=torch.cat(
            (pad_seq(topology.material_sequence, -1), slots["material_sequence"])
        ).contiguous(),
        interaction_type=torch.cat(
            (pad_seq(topology.interaction_type, 0), slots["interaction_type"])
        ).contiguous(),
    )
    new_geometry = PathGeometry(
        row_identity=new_topology.row_identity,
        path_length_m=cat(geometry.path_length_m, length),
        delay_s=cat(geometry.delay_s, length / C0),
        field_direction=cat(geometry.field_direction, discovery.d_o),
        interaction_position=cat(geometry.interaction_position, discovery.v_pos),
        interaction_normal=cat(geometry.interaction_normal, vertex_normal),
        interaction_positions=torch.cat(
            (pad_vec(geometry.interaction_positions), slots["interaction_positions"])
        ).contiguous(),
        interaction_normals=torch.cat(
            (pad_vec(geometry.interaction_normals), slots["interaction_normals"])
        ).contiguous(),
    )
    new_fields = PathFields(
        row_identity=new_topology.row_identity,
        path_gain=cat(fields.path_gain, gain),
        path_field=cat(fields.path_field, torch.complex(amplitude, zero)),
        field_xyz=cat(fields.field_xyz, field_xyz),
        coefficient=cat(fields.coefficient, coefficient),
    )
    return (
        EvaluatedPaths(
            topology=new_topology, geometry=new_geometry, fields=new_fields
        ),
        sidecars,
    )


def _index_discovery(
    discovery: ScatterChainDiscovery, idx: torch.Tensor
) -> ScatterChainDiscovery:
    return ScatterChainDiscovery(
        **{
            spec: getattr(discovery, spec)[idx].contiguous()
            for spec in (
                "tx_id",
                "rx_id",
                "sample_index",
                "d1",
                "d2",
                "c1_positions",
                "c1_normals",
                "c1_primitive",
                "c1_material",
                "L1",
                "d_i",
                "c2_positions",
                "c2_normals",
                "c2_primitive",
                "c2_material",
                "L2",
                "d_o",
                "v_pos",
                "v_normal",
                "v_material",
                "weight",
                "cos_i",
                "cos_o",
                "patch_row",
            )
        }
    )


def append_chain_scattering_paths(
    scene: Scene,
    config: TopologyConfig,
    evaluated: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
    info: dict[str, Any],
    *,
    ad_mode: str,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """ADR-021 D1: discover and append enumerated scatter-chain rows.

    Default-OFF: ``scattering_chain_max_depth < 1`` returns the inputs unchanged
    (this branch is never entered, so the pipeline is byte-identical).
    """

    device = evaluated.device
    compiled = require_compiled(scene)
    if not compiled.rayd.available:
        raise RuntimeError(
            "deterministic scatter-chain discovery requires RayD native capability"
        )
    screens = realization_phase_screens(compiled.materials, compiled.assignments)
    ensemble_faces = _ensemble_scatter_faces(compiled, screens, device=device)
    samples = build_chain_samples(compiled, config, ensemble_faces, device=device)
    info["chain_sample_count"] = 0 if samples is None else int(samples.position.shape[0])
    if samples is None:
        return evaluated, sidecars

    tx_positions, tx_power = transmitter_tensors(scene, device=device)
    rx_positions, _layout = receiver_positions_and_layout(scene, device=device)
    if tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return evaluated, sidecars
    tx_pol = transmitter_polarizations(scene, device=device)
    rx_pol = receiver_polarizations(scene, device=device)
    records = compiled.rayd.edge_records()
    vertices = records.vertices
    scene_diagonal = (vertices.max(dim=0).values - vertices.min(dim=0).values).norm()

    discovery = discover_scatter_chains(
        compiled,
        config,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        samples=samples,
        scene_diagonal=scene_diagonal,
    )
    info["chain_row_count"] = 0 if discovery is None else discovery.row_count
    if discovery is None:
        return evaluated, sidecars

    physics = _chain_ensemble_evaluate(
        scene,
        compiled,
        config,
        discovery,
        tx_pol=tx_pol,
        rx_pol=rx_pol,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tx_power=tx_power,
        device=device,
        ad_mode=ad_mode,
    )
    vertex_face = samples.face_id[discovery.sample_index].contiguous()
    evaluated, sidecars = _extend_evaluated_paths_chain(
        evaluated,
        sidecars,
        discovery,
        physics,
        vertex_face,
        tx_power=tx_power,
    )
    info["chain_kept_count"] = int(physics["keep"].sum().item())
    return evaluated, sidecars


# -------------------------------------------------------------------------
# Shared Monte Carlo Kirchhoff scattering events (was
# montecarlo/events/scattering.py)
# -------------------------------------------------------------------------
#
# Shared Kirchhoff rough-surface scattering helpers for the MC solvers.
#
# Extends the wave-2 two-way {reflect, transmit} event machinery
# (:mod:`witwin.channel.interactions.transmission`) to the three-way
# {reflect, scatter, transmit} selection of plan section 7.1 at hits on rough
# faces (``scatter_model_id == 1``), and hosts the pure-torch pieces both MC
# solvers share: local roughness frames, TE/TM incident power decomposition,
# per-material Kirchhoff table sampling/eval, and the BDPT scattering NEE
# connection-row builder.
#
# Measure and normalization conventions (documented once, used everywhere):
#
# - Event probabilities are proportional to the native energy
#   budgets ``(R_coh, R_diff, T_bar)`` at the hit incidence with the same
#   minimum-probability floor pattern as the wave-2 two-way split; the
#   selected branch's POWER is divided by its probability (fields by
#   ``sqrt(p)``), keeping the estimator unbiased. Smooth faces
#   (``scatter_model_id != 1``) keep the wave-2 two-way logic bit-identically:
#   their scatter probability is exactly zero and their transmit probability
#   still comes from the native stack budgets, so the single selection
#   uniform partitions identically.
# - Scattering contributions are POWER-ONLY (ensemble-average Kirchhoff
#   BSDF): the scattered field carries phase 0 as a placeholder that only
#   ever enters ``|field|^2``; no random phase is assigned (plan section 7.3).
# - Specular reflection and delta transmission stay discrete events;
#   Kirchhoff scattering is a continuous solid-angle density. The scattered
#   subpath stores ``pdf_forward *= pdf(wo|wi)`` and ``pdf_reverse *=
#   pdf(wi|wo)`` from the SAME reciprocal table with swapped arguments
#   (contract section 5).
# - Depth rule (BDPT ``max_scattering_order``, ADR-021 D4):
#     * order 1 (default): scattering is single-bounce. A scattered subpath
#       connects to the receivers (NEE) and terminates; reflection/transmission
#       never follow a scattering event.
#     * order > 1: the terminal rule is lifted. A scattered subpath emits its
#       NEE row and then CONTINUES in its lobe-sampled direction (its power
#       already divided by ``p_scatter * pdf(wo)`` in
#       :func:`scattered_subpath_state`), and may reflect/transmit/scatter again
#       up to ``max_scattering_order`` scatter events, emitting an NEE row at
#       every scatter vertex. Because a scatter event clears the Complex3 Jones
#       carrier, a subsequent scatter vertex reads its incident power from the
#       scalar throughput (:func:`scatter_carried_incident_power`), split
#       unpolarized across the local TE/TM channels. Scattering stays power-only
#       and excluded from the ADR-019 coherent combine in both modes.
#
# NEE contribution derivation (the BDPT connection convention is
# ``contribution = source_power * |field|^2 * (lambda/(4*pi*L))^2 / N`` with
# no 1/r stored in the field, i.e. an isotropic-source picture where the
# power flux density at unfolded distance ``r1`` is
# ``source_power * |F|^2 / (4*pi*r1^2)``):
#
#     g_scatter = Int_wall dA |F|^2 cos_i f_K(wi,wo) cos_o
#                 * (lambda/(4*pi))^2 / (r1^2 * r2^2)
#
# Monte Carlo over the tx launch directions (uniform sphere, pdf 1/(4*pi))
# converts solid angle to wall area over the UNFOLDED prefix distance,
# ``dA = r1^2 dOmega / cos_i`` (specular prefixes preserve the image-source
# solid-angle measure), so per hitting ray both ``r1^2`` and ``cos_i``
# cancel:
#
#     row = source_power * (P_te*f_te + P_tm*f_tm) * cos_o
#           * (lambda/(4*pi))^2 * 4*pi / (r2^2 * p_scatter * N)
#
# where ``P_te/P_tm`` are the incident Jones powers in the local s/p basis
# (the same basis the table's co-pol kernels use) and the division by
# ``p_scatter`` accounts for NEE being evaluated only on scatter-selected
# rows. Cross-polarization is not tracked in v1 (contract section 6): the
# receiver is treated as polarization-matched to the scattered power, which
# keeps BDPT consistent with the unpolarized MC basic map.
#
# MIS: the NEE connection is the ONLY strategy that produces a
# tx->...->scatter->rx path in v1 (the directional continuation terminates
# without a sensor-intersection strategy, and a point/cell-center sensor has
# zero probability under the continuous direction density), so the balance
# heuristic over {directional sample, connection} degenerates to weight 1
# for the connection rows. The directional density converted to the
# connection measure is still recorded in the row ``pdf`` for diagnostics
# and future multi-bounce MIS.
LIGHT_SPEED_M_PER_S = 299_792_458.0


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


def rough_material_runtimes(compiled: Any) -> dict[int, RoughMaterialRuntime]:
    """Kirchhoff runtimes per material index (``scatter_model_id == 1``).

    Reads the CSR layers and roughness fields from the compiled
    MaterialStore; the tables come from the compiled-scene lazy cache and
    raise ``kirchhoff_domain_exceeded`` for out-of-domain roughness.
    """

    return compiled.rough_material_runtimes


def scatter_direction_uniforms(
    count: int, *, seed: int, tx_index: int, depth: int, device: torch.device
) -> torch.Tensor:
    """Reproducible ``[count, 2]`` uniforms for Kirchhoff direction sampling.

    Derived like the wave-2 event-selection seeds but salted into a distinct
    stream, so the selection uniforms of
    :func:`witwin.channel.interactions.transmission.event_uniforms`
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


def scatter_carried_incident_power(
    throughput_real: torch.Tensor, throughput_imag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpolarized incident ``(P_te, P_tm)`` of an already-scattered subpath.

    A scatter event clears the Complex3 Jones carrier (scattering is power-only
    in v1) and moves the transported power into the scalar throughput, so a
    FURTHER scatter vertex on the same multi-order subpath (ADR-021 D4) reads
    its incident power from the throughput rather than the zeroed field. The
    carried power is unpolarized, so it is split evenly across the local TE/TM
    channels; both halves feed the same ``P_te*f_te + P_tm*f_tm`` NEE kernel and
    the continuation's polarization-weighted BSDF as the single-bounce field
    decomposition (:func:`te_tm_incident_power`) does at the first scatter."""

    power = throughput_real**2 + throughput_imag**2
    half = 0.5 * power
    return half, half


def _grouped_rows(
    valid: torch.Tensor,
    material_id: torch.Tensor,
    runtimes: dict[int, RoughMaterialRuntime],
) -> list[tuple[RoughMaterialRuntime, torch.Tensor, torch.Tensor]]:
    groups = []
    for index, runtime in runtimes.items():
        material_valid = valid & (material_id == index)
        rows = torch.nonzero(material_valid, as_tuple=False).flatten()
        if int(rows.numel()):
            groups.append(
                (runtime, rows, material_valid.index_select(0, rows).contiguous())
            )
    return groups


def sample_scatter_directions(
    valid: torch.Tensor,
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
    for runtime, rows, valid_rows in _grouped_rows(valid, material_id, runtimes):
        wi_rows = wi_local.index_select(0, rows).contiguous()
        sampled = scattering_table_sample(
            valid_rows,
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
    valid: torch.Tensor,
    material_id: torch.Tensor,
    wi_local: torch.Tensor,
    wo_local: torch.Tensor,
    runtimes: dict[int, RoughMaterialRuntime],
    *,
    ad: bool = False,
    ledger: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched ``(f_te, f_tm)`` lookups grouped by rough material.

    Under ``ad`` the native forward is dispatched behind
    :func:`scattering_table_eval_ad` so the resident table values keep their
    gradient (ADR-015 op 1); the plain (``ad`` off) path is bitwise unchanged.
    """

    count = int(material_id.shape[0])
    device = wi_local.device
    f_te = torch.zeros((count,), device=device, dtype=torch.float32)
    f_tm = torch.zeros((count,), device=device, dtype=torch.float32)
    for runtime, rows, valid_rows in _grouped_rows(valid, material_id, runtimes):
        wi_rows = wi_local.index_select(0, rows).contiguous()
        wo_rows = wo_local.index_select(0, rows).contiguous()
        if ad:
            if ledger is not None:
                ledger.add(
                    runtime.table.f_te, runtime.table.f_tm, wi_rows, wo_rows
                )
            te_rows, tm_rows = scattering_table_eval_ad(
                valid_rows,
                wi_rows,
                wo_rows,
                runtime.table.f_te,
                runtime.table.f_tm,
            )
        else:
            te_rows, tm_rows = kirchhoff_tables.eval_bsdf(
                runtime.table, valid_rows, wi_rows, wo_rows
            )
        f_te[rows] = te_rows
        f_tm[rows] = tm_rows
    return f_te, f_tm


def scattering_nee_connection_samples(
    rayd: Any,
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
    frequency_hz: float | torch.Tensor,
    samples: int,
    scene_diagonal: float,
    ad: bool = False,
    ledger: object | None = None,
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
    # ADR-015 Part A: under ad the radiometric lambda factor is built from the
    # live frequency tensor so its gradient flows through amplitude^2; the primal
    # path reads the detached host scalar and is bitwise unchanged (mirrors
    # scattering_map_matrix for MC-basic).
    wavelength = LIGHT_SPEED_M_PER_S / (frequency_hz if ad else float(frequency_hz))
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
    rx_id_rows = sensor["rx_id"][None, :].expand(vertex_count, sensor_count)
    bsdf_valid = (
        flat(front)
        & flat(expand_rows(tx_id >= 0))
        & (flat(rx_id_rows) >= 0)
        & flat(sensor["valid"][None, :].expand(vertex_count, sensor_count))
    )
    f_te, f_tm = eval_bsdf_rows(
        bsdf_valid,
        material_rows,
        wi_rows,
        wo_local.contiguous(),
        runtimes,
        ad=ad,
        ledger=ledger,
    )

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
    visible = geometry_kernels.rayd_visibility_forward(
        rayd.require_resource(),
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
            valid.index_select(0, rows).contiguous(),
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
    ad: bool = False,
    ledger: object | None = None,
) -> dict[str, torch.Tensor]:
    """Continued light subpath after a Kirchhoff scattering event.

    Assembled directly in torch with the native subpath tensor layout (no
    native kernel: the update is gather+FMA on the sampled table values).
    Scattering contributions are POWER-ONLY ensemble estimates. No coherent
    phase or cross-polarized Jones amplitude is available from the current
    Kirchhoff table, so the Complex3 carrier is explicitly cleared instead of
    manufacturing a zero-phase field.

    Post-scatter carrier semantics (ADR-021 D4): the scattered subpath's
    ``throughput_real`` is seeded from the FIELD-BASED incident power at this
    vertex (``sqrt(p_te + p_tm)``, which excludes ``source_power`` - the
    connection convention multiplies ``source_power`` separately), times the
    unbiased continuation amplitude ``sqrt(f_weighted * cos_o / (pdf * p))``.
    From this vertex on, ``|throughput|^2`` IS the authoritative unpolarized
    power weight of the subpath: subsequent specular events scale it by
    ``sqrt(gain * R_eff)`` at the actual incidence angle, which is the exact
    unpolarized power transport. The PRE-scatter throughput remains the
    sampling proxy of contract section 5 and is deliberately not consumed
    here.
    """

    scatter_valid = choose_scatter & state["valid"]
    sampled = sample_scatter_directions(
        scatter_valid, material_id, wi_local, uniforms, runtimes
    )
    wo_local = sampled["wo_local"]
    pdf_forward = sampled["pdf_forward"]
    wo_world = local_to_world(wo_local, frame_t1, frame_t2, normal)
    f_te, f_tm = eval_bsdf_rows(
        scatter_valid,
        material_id,
        wi_local,
        wo_local,
        runtimes,
        ad=ad,
        ledger=ledger,
    )
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
    # Seed the post-scatter carrier from the field-based incident power at
    # this vertex, NOT from the pre-scatter proxy throughput: the proxy
    # includes sqrt(tx_power) (double-counted by the connection convention's
    # separate source_power factor) and approximates the polarized specular
    # chain, while (p_te + p_tm) is exact for the first scatter and is the
    # carried unpolarized power for later scatters on the same subpath.
    incident_amp = incident_power.sqrt()
    scattered["throughput_real"] = incident_amp * amplitude_scale
    scattered["throughput_imag"] = torch.zeros_like(incident_amp)
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
    rayd: Any,
    tx_pos: torch.Tensor,
    tx_power: torch.Tensor,
    rx_pos: torch.Tensor,
    *,
    samples: int,
    seed: int,
    device: torch.device,
    ad: bool = False,
    ledger: object | None = None,
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
    runtimes = rough_material_runtimes(require_compiled(scene))
    if not runtimes:
        return matrix, stats

    handle = rayd.require_resource()
    bundle = face_material_field_bundle(scene, device=device)
    face_material_id = bundle["material_id"].to(torch.int64)
    face_scatter_model = bundle["scatter_model_id"].index_select(0, face_material_id)
    records = rayd.edge_records()
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
    # Under ad the carrier stays a live tensor so lambda / amplitude^2 carry the
    # frequency gradient; the plain path reads the detached host scalar and is
    # bitwise unchanged.
    frequency = scene.frequency if ad else float(scene.frequency)
    wavelength = LIGHT_SPEED_M_PER_S / frequency
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
        visible_tx = geometry_kernels.rayd_visibility_forward(
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
                pair_active.reshape(count * block).contiguous(),
                point_material.repeat_interleave(block),
                wi_local.repeat_interleave(block, dim=0).contiguous(),
                wo_local.contiguous(),
                runtimes,
                ad=ad,
                ledger=ledger,
            )
            f_unpol = 0.5 * (f_te + f_tm)
            visible_rx = geometry_kernels.rayd_visibility_forward(
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
