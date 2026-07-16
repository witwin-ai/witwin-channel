"""Deterministic rough-surface scattering (plan 05 wave 3, contract section 6).

Appends single-bounce ``component_id=6`` scattering rows through one canonical
tensor-concatenation owner shared by the legacy ``TopologyBatch`` wrapper and
the typed ``EvaluatedPaths`` bridge. Two mutually exclusive per-surface modes:

- ``ensemble`` (production): Kirchhoff ensemble BSDF patch quadrature.
  Incoherent POWER rows, one row per visible patch sample.
- ``realization_coherent`` (reference): phase-screen Kirchhoff patch integral
  for surfaces carrying a ``PhaseScreen`` assignment. One coherent complex
  row per (tx, rx, structure); it REPLACES both the delta specular and the
  ensemble lobe for that surface (contract 6.7.3 - the two models are never
  summed for one surface).

Normalization derivation (ensemble). The repo's deterministic ``path_gain``
is a received power for unit-gain antennas: LoS carries
``path_gain = P_t * (lambda / (4*pi*d))^2`` and ``path_field`` carries the
matching complex amplitude ``sqrt(P_t) * lambda/(4*pi) * e^{-j k d}/d``
(``core.field_state.PHASE_CONVENTION``). Radiometrically, a patch of area
``A`` receives the flux density ``P_t/(4*pi*r1^2) * cos_theta_i``, re-emits
the radiance ``f * E_i`` (``f`` is the Kirchhoff power BSDF per steradian,
hemispherically normalized to ``R_diff``), and the receiver collects through
the effective aperture ``A_e = lambda^2/(4*pi)``:

    P_r = P_t * f * cos_theta_i * cos_theta_o * A * lambda^2
          / ((4*pi)^2 * r1^2 * r2^2)

This is the plan section 9 patch-quadrature formula with ``gamma = 4*pi*f``
and ``A_e = lambda^2/(4*pi)`` substituted, expressed in the repo's
``path_gain`` units. Cross-check (tested): in the specular-delta limit the
patch sum over an infinite plane collapses to the image-source result
``P_t * R * (lambda/(4*pi*(r1+r2)))^2``, so scattering + C_r-attenuated
specular reproduces the smooth-wall reflection power.

Polarization (v1, documented): the tx polarization is projected onto the
transverse plane of the incident propagation direction and decomposed in the
local s/p basis (``s = normalize(n x d)``, ``p = s x d``, contract section
2); the co-pol table channels are weighted by the squared projections and
the receive side applies the outgoing s/p projections of the receiver
polarization. Cross-pol arises only from this frame rotation (contract
section 6).

Realization mode phase bookkeeping: ``patch_phase_integral`` computes
``Int exp(-j*(q.x + q_n*h)) dA`` with ``q = k_s - k_i`` built from
propagation wave vectors. The physical patch factor of the point-source
Kirchhoff integral is ``e^{-j k0 (r1c + r2c)} * Int exp(+j*(q.delta +
q_n*h)) dA`` (first-order expansion of ``k0*(r1(x) + r2(x))`` around the
patch centroid ``c``, ``delta = x - c``). The ``+j`` integral is obtained
losslessly by calling ``patch_phase_integral`` with SWAPPED wave vectors
(``k_i <-> k_s`` flips ``q``), which also feeds heights with the physical
``exp(+j*q_n*h)`` sign; the leftover absolute-position phase is removed with
``exp(-j*q.c)``. The per-patch prefactor ``j*k0*F/(4*pi)`` with the
Kirchhoff geometry factor ``F = |q|^2/(k0*q_n)`` makes the smooth
(``h = 0``) large-plate limit collapse to the exact image-source reflection
``r * e^{-j k0 (r1+r2)}/(r1+r2)`` by stationary phase (tested).

Scattering rows accumulate in the POWER domain (plan 7.3): ensemble rows
carry ``path_field = sqrt(path_gain)`` with zero phase (metadata flag
``scattering_paths_incoherent``); realization rows keep their physical
complex field in the row but still fold into totals as power.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypedDict

import torch
from witwin.channel_native.core.tensor_math import normalize_vec3

from witwin.channel_native.core.field_state import (
    receiver_polarizations,
    transmitter_polarizations,
)
from witwin.channel_native.materials.kernels import functional as material_kernels
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.propagation.geometry.endpoints import (
    receiver_positions_and_layout,
    transmitter_tensors,
)
from witwin.channel_native.propagation.enumerated.contracts import (
    TopologyBatch,
    TopologyConfig,
)
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths
from witwin.channel_native.propagation.models.fields import PathFields
from witwin.channel_native.propagation.models.geometry import PathGeometry
from witwin.channel_native.propagation.models.topology import PathTopology
from witwin.channel_native.propagation.topology.export import EvaluatedPathSidecars
from witwin.channel_native.materials.models import PhaseScreen
from witwin.channel_native.physics.oracle import C0
from witwin.channel_native.scattering import eval_bsdf, patch_phase_integral
from witwin.channel_native.scattering.tables import MAX_RMS_SLOPE

if TYPE_CHECKING:
    from witwin.channel_native.core.scene import Scene

__all__ = ["append_scattering_paths"]

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
    raydn: object, start: torch.Tensor, end: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Chunked segment visibility; returns (mask, launch_count)."""

    count = int(start.shape[0])
    if count == 0:
        return torch.empty((0,), device=start.device, dtype=torch.bool), 0
    handle = raydn.require_handle()
    masks = []
    launches = 0
    for lo in range(0, count, _VISIBILITY_CHUNK):
        hi = min(lo + _VISIBILITY_CHUNK, count)
        masks.append(
            geometry_bridge.raydn_visibility_forward(
                handle, start[lo:hi].contiguous(), end[lo:hi].contiguous(), None
            )[0]
        )
        launches += 1
    return torch.cat(masks), launches


def _realization_structures(compiled: object) -> dict[int, PhaseScreen]:
    """Structures whose scattering is the coherent phase-screen realization.

    Enforces contract 6.7.3 exclusivity: a ``PhaseScreen`` with
    ``mode='ensemble_bsdf'`` on a Kirchhoff-rough material is refused because
    it would define a second ensemble source for the same surface; the
    material roughness already is the ensemble model.
    """

    screens: dict[int, PhaseScreen] = {}
    structure_material = compiled.assignments.structure_material_id.to(torch.int64)
    scatter_model = compiled.materials.scatter_model_id
    for index, screen in compiled.assignments.structure_phase_screens.items():
        material_index = int(structure_material[index])
        rough = int(scatter_model[material_index]) == 1
        if screen.mode == "realization_coherent":
            screens[index] = screen
            continue
        # mode == "ensemble_bsdf"
        if rough:
            raise RuntimeError(
                "scattering_mode_conflict: structure "
                f"{index} combines a PhaseScreen(mode='ensemble_bsdf') with a "
                "Kirchhoff-rough material; realization and ensemble models "
                "must never be summed for one surface (contract 6.7.3). Use "
                "mode='realization_coherent' to replace the ensemble lobe, "
                "or drop the phase screen to keep the material ensemble table"
            )
        raise RuntimeError(
            "PhaseScreen mode 'ensemble_bsdf' requires a Kirchhoff-rough "
            "material to define the ensemble statistics; assign a rough "
            "PhysicalSurface or use mode='realization_coherent'"
        )
    return screens


def _guard_phase_screen_geometry(
    runtime: object, uv_scale_m: float, structure_index: int
) -> None:
    """Applicability guard (contract section 6): the phase screen never moves
    geometry, so its RMS metric slope must stay in the tangent-plane domain.
    Out-of-domain surfaces raise instead of silently degrading."""

    heights = runtime.heights_m
    rows, cols = heights.shape
    if uv_scale_m <= 0.0 or not math.isfinite(uv_scale_m):
        raise RuntimeError(
            "phase_screen_geometry_limit_exceeded: structure "
            f"{structure_index} has a degenerate UV-to-world scale"
        )
    slope_u = (heights[:, 1:] - heights[:, :-1]) * (cols / uv_scale_m)
    slope_v = (heights[1:, :] - heights[:-1, :]) * (rows / uv_scale_m)
    rms_slope = math.sqrt(
        float(slope_u.square().mean()) + float(slope_v.square().mean())
    )
    if rms_slope > MAX_RMS_SLOPE:
        raise RuntimeError(
            "phase_screen_geometry_limit_exceeded: structure "
            f"{structure_index} phase screen RMS slope {rms_slope:.3g} exceeds "
            f"the tangent-plane limit {MAX_RMS_SLOPE:g}; heights of this "
            "magnitude change occlusion/silhouettes and cannot be represented "
            "as a pure phase screen"
        )


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
) -> None:
    device = tx_positions.device
    records = compiled.raydn.edge_records()
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

    tables = compiled.kirchhoff_tables  # raises kirchhoff_domain_exceeded
    materials = compiled.materials
    axis_rad = materials.rough_axis_rad.to(device=device, dtype=torch.float32)

    frequency = float(scene.frequency)
    wavelength = C0 / frequency
    # Repo path_gain units (module docstring): P_t * f * cos_i * cos_o * A
    # * lambda^2 / ((4*pi)^2 * r1^2 * r2^2).
    power_scale = wavelength**2 / (4.0 * math.pi) ** 2

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
            compiled.raydn,
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
                compiled.raydn,
                rx_block[rc].contiguous(),
                offset_points[sc].contiguous(),
            )
            info["visibility_launch_count"] += launches
            rc, sc = rc[vis_rx], sc[vis_rx]
            if int(sc.shape[0]) == 0:
                continue

            wo_row = wo_w[rc, sc]
            cos_o_row = cos_o[rc, sc]
            r2_row = r2[rc, sc]
            wo_local = torch.stack(
                (
                    (wo_row * t1r[sc]).sum(-1),
                    (wo_row * t2r[sc]).sum(-1),
                    cos_o_row,
                ),
                dim=-1,
            )
            f_te = torch.zeros_like(cos_o_row)
            f_tm = torch.zeros_like(cos_o_row)
            for material_index, table in tables.items():
                mask = sample_material[sc] == material_index
                if not bool(mask.any()):
                    continue
                te, tm = eval_bsdf(
                    table, wi_local[sc][mask].contiguous(), wo_local[mask].contiguous()
                )
                f_te[mask] = te
                f_tm[mask] = tm

            # Receive-side co-pol s/p projections of the receiver polarization.
            s_o, p_o = _sp_basis(n_o[sc], wo_row, backup_axis[sc])
            pol_r = rx_pol[rx_start + rc]
            pol_r_perp = pol_r - (pol_r * wo_row).sum(-1, keepdim=True) * wo_row
            g_te2 = (pol_r_perp * s_o).sum(-1).square()
            g_tm2 = (pol_r_perp * p_o).sum(-1).square()
            f_eff = f_te * a_te2[sc] * g_te2 + f_tm * a_tm2[sc] * g_tm2

            gain = (
                float(tx_power[tx_index])
                * power_scale
                * f_eff
                * cos_i[sc]
                * cos_o_row
                * weights[sc]
                / (r1[sc].square() * r2_row.square())
            )
            keep = gain > max(threshold, 0.0)
            if not bool(keep.any()):
                continue
            rc, sc, gain, r2_row = rc[keep], sc[keep], gain[keep], r2_row[keep]
            wo_row = wo_row[keep]
            length = r1[sc] + r2_row
            amplitude = gain.clamp_min(0.0).sqrt()
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
) -> None:
    device = tx_positions.device
    records = compiled.raydn.edge_records()
    face_structure = compiled.geometry.face_structure_id.to(
        device=device, dtype=torch.int64
    )
    face_material = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    )
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
    runtimes = compiled.phase_screen_runtimes
    density = float(getattr(config, "scattering_samples_per_m2", 8.0))

    for structure_index in sorted(screens):
        structure = scene.structures[structure_index]
        if structure.uv is None or structure.face_uv is None:
            raise RuntimeError(
                "realization_coherent phase screen requires structure UV "
                f"(structure {structure_index} has none); contract section 6"
            )
        runtime = runtimes[structure_index]
        global_faces = torch.nonzero(
            face_structure == structure_index, as_tuple=False
        ).reshape(-1)
        if int(global_faces.numel()) == 0:
            continue
        first_face = int(global_faces.min())
        faces = records.faces.to(torch.int64)[global_faces]
        tri = records.vertices[faces]  # [F, 3, 3]
        uv_vertices = structure.uv.to(device=device, dtype=torch.float32)
        face_uv = structure.face_uv.to(device=device, dtype=torch.int64)
        material_index = int(face_material[first_face])

        # UV -> world metric scale for the geometry guard: compare face area
        # with its UV-space area (isotropic estimate, documented).
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        areas = 0.5 * torch.linalg.vector_norm(torch.cross(e1, e2, dim=-1), dim=-1)
        uv_tris = uv_vertices[face_uv[(global_faces - first_face)]]
        uv_e1 = uv_tris[:, 1] - uv_tris[:, 0]
        uv_e2 = uv_tris[:, 2] - uv_tris[:, 0]
        uv_areas = 0.5 * (
            uv_e1[:, 0] * uv_e2[:, 1] - uv_e1[:, 1] * uv_e2[:, 0]
        ).abs()
        uv_scale = float(
            torch.sqrt(areas.sum() / uv_areas.sum().clamp_min(1.0e-20))
        )
        _guard_phase_screen_geometry(runtime, uv_scale, structure_index)

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
        for local in range(int(global_faces.numel())):
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
                compiled.raydn,
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
                    compiled.raydn,
                    rx.reshape(1, 3).expand(int(rows.numel()), 3).contiguous(),
                    offset_points[rows].contiguous(),
                )
                info["visibility_launch_count"] += launches
                rows = rows[vis_rx]
                if int(rows.numel()) == 0:
                    continue

                # Smooth-stack Jones at the mean plane per patch.
                stack = material_kernels.em_layer_stack_eval(
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
                    frequency_hz=frequency,
                )
                r_te = torch.complex(stack["r_te_real"], stack["r_te_imag"])
                r_tm = torch.complex(stack["r_tm_real"], stack["r_tm_imag"])

                d_i = -wi_w[rows]
                d_o = wo_w[rows]
                n_rows = n_o[rows]
                backup_axis = _stable_tangent(n_rows)
                s_i, p_i = _sp_basis(n_rows, d_i, backup_axis)
                s_o, p_o = _sp_basis(n_rows, d_o, backup_axis)
                pol_t = tx_pol[tx_index].reshape(1, 3)
                pol_t_perp = pol_t - (pol_t * d_i).sum(-1, keepdim=True) * d_i
                pol_r = rx_pol[rx_index].reshape(1, 3)
                pol_r_perp = pol_r - (pol_r * d_o).sum(-1, keepdim=True) * d_o
                a_te = (pol_t_perp * s_i).sum(-1)
                a_tm = (pol_t_perp * p_i).sum(-1)
                g_te = (pol_r_perp * s_o).sum(-1)
                g_tm = (pol_r_perp * p_o).sum(-1)
                jones = r_te * (a_te * g_te) + r_tm * (a_tm * g_tm)

                k_i_vec = d_i * k0
                k_s_vec = d_o * k0
                q = k_s_vec - k_i_vec
                q_norm = torch.linalg.vector_norm(q, dim=-1)
                q_n = (q * n_rows).sum(-1)
                prefactor = (
                    1j * k0 * (q_norm.square() / (k0 * q_n.clamp_min(1.0e-9)))
                    / (4.0 * math.pi)
                )
                carrier = torch.polar(
                    torch.ones_like(q_n),
                    -(k0 * (r1[rows] + r2[rows]) + (q * centroids[rows]).sum(-1)),
                ).to(torch.complex64)

                total = torch.zeros((), device=device, dtype=torch.complex64)
                row_list = rows.tolist()
                for slot, patch_index in enumerate(row_list):
                    # Swapped wave vectors: patch_phase_integral integrates
                    # exp(-j*(q'.x + q_n'*h)) with q' = -q, i.e. the physical
                    # +j integrand of the module docstring derivation.
                    integral = patch_phase_integral(
                        runtime,
                        patch_tris[patch_index],
                        patch_uvs[patch_index],
                        k_s_vec[slot],
                        k_i_vec[slot],
                        frequency,
                    )
                    total = total + (
                        prefactor[slot]
                        * jones[slot]
                        * carrier[slot]
                        / (r1[patch_index] * r2[patch_index])
                    ) * integral
                    info["realization_patch_integrals"] += 1

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
    source: TopologyBatch | EvaluatedPaths,
    rows: dict[str, torch.Tensor],
) -> _ExtendedScatteringRows:
    """Single 22-tensor cat owner shared by the legacy and typed entrypoints."""

    if isinstance(source, EvaluatedPaths):
        topology = source.topology
        geometry = source.geometry
        fields = source.fields
    else:
        topology = source
        geometry = source
        fields = source

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


def _extend_topology(
    topology: TopologyBatch,
    rows: dict[str, torch.Tensor],
    *,
    launch_count_delta: int,
    candidate_count_delta: int,
    guardrail_count_delta: int,
) -> TopologyBatch:
    return replace(
        topology,
        **_extended_scattering_rows(topology, rows),
        launch_count=topology.launch_count + launch_count_delta,
        candidate_count=topology.candidate_count + candidate_count_delta,
        guardrail_count=topology.guardrail_count + guardrail_count_delta,
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
    }


def _collect_scattering_rows(
    scene: Scene,
    config: TopologyConfig,
    *,
    device: torch.device,
    info: dict[str, Any],
) -> tuple[dict[str, torch.Tensor] | None, int, int, int]:
    compiled = scene.compile()
    screens = _realization_structures(compiled)
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
    for index in screens:
        realization_face |= face_structure == index
    ensemble_faces = torch.nonzero(
        scatter_face & ~realization_face, as_tuple=False
    ).reshape(-1)
    info["realization_structure_count"] = len(screens)
    if int(ensemble_faces.numel()) == 0 and not screens:
        return None, 0, 0, 0
    if not compiled.raydn.available:
        raise RuntimeError(
            "deterministic scattering requires RayDN native scene capability"
        )

    tx_positions, tx_power = transmitter_tensors(scene, device=device)
    rx_positions, _layout = receiver_positions_and_layout(scene, device=device)
    if tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return None, 0, 0, 0
    tx_pol = transmitter_polarizations(scene, device=device)
    rx_pol = receiver_polarizations(scene, device=device)
    records = compiled.raydn.edge_records()
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


def append_scattering_paths(
    scene: Scene, config: TopologyConfig, topology: TopologyBatch
) -> tuple[TopologyBatch, dict[str, Any]]:
    """Append component_id=6 scattering rows and return (topology, info).

    No-ops (empty info) when scattering was not requested, ``max_depth < 1``
    or the scene carries no rough/phase-screen surfaces. Scattering rows are
    appended after the canonical specular selection, so ``max_paths`` does
    not apply to them; their budget is ``scattering_max_paths_per_pair``.
    """

    info = _scattering_info()
    components = set(config.components)
    if "scattering" not in components or int(config.max_depth) < 1:
        return topology, info
    if not scene.structures:
        return topology, info

    rows, launch_delta, candidate_delta, guardrail_delta = _collect_scattering_rows(
        scene,
        config,
        device=topology.valid.device,
        info=info,
    )
    if rows is None:
        return topology, info

    extended = _extend_topology(
        topology,
        rows,
        launch_count_delta=launch_delta,
        candidate_count_delta=candidate_delta,
        guardrail_count_delta=guardrail_delta,
    )
    return extended, info


def append_scattering_evaluated_paths(
    scene: Scene,
    config: TopologyConfig,
    evaluated: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars, dict[str, Any]]:
    """Append scattering rows to canonical typed path contracts."""

    info = _scattering_info()
    components = set(config.components)
    if "scattering" not in components or int(config.max_depth) < 1:
        return evaluated, sidecars, info
    if not scene.structures:
        return evaluated, sidecars, info

    rows, launch_delta, candidate_delta, guardrail_delta = _collect_scattering_rows(
        scene,
        config,
        device=evaluated.device,
        info=info,
    )
    if rows is None:
        return evaluated, sidecars, info

    evaluated, sidecars = _extend_evaluated_paths(
        evaluated,
        sidecars,
        rows,
        launch_count_delta=launch_delta,
        candidate_count_delta=candidate_delta,
        guardrail_count_delta=guardrail_delta,
    )
    return evaluated, sidecars, info
