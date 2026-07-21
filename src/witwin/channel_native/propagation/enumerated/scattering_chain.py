"""ADR-021 D1 enumerated scatter-chain discovery (Deterministic + Path).

Discovers the enumerated scatter-chain path class

    TX --C1 (reflections, depth d1 >= 0)--> v_s --C2 (depth d2 >= 0)--> RX
    1 <= d1 + d2 <= scattering_chain_max_depth

by running the EXISTING RayD image-method reflection enumeration twice against a
dedicated chain-sample set as virtual endpoints (``tx -> {samples}`` for C1 and
``rx -> {samples}`` for C2, reciprocal), then joining the two legs on the sample
index. No geometry is recomputed in Python/Torch: every hit point, normal, and
face id comes from the native RayD EPC (``query_reflection_epc`` /
``rayd_reflection_epc_paths_forward``) exactly as the deterministic reflection
topology owner uses it (``propagation/enumerated/reflection.py``). The only
Python work here is the sanctioned structural boundary work (plan 10a section 2):
join on the sample index, keep-strongest budgeting, stable row ordering, padding
the per-leg bounce blocks to the native ``kMaxAdDepth = 8`` capacity, and packing
the derived per-row lengths / spreading / incident-outgoing directions the frozen
:class:`ScatterChainDiscovery` contract requires.

The produced :class:`ScatterChainDiscovery` is the read-only typed contract the
native Op A / Op B chain facades (ADR-021 D2, owner ``scattering/kernels/``)
consume; this module owns discovery only. Directions (``d_i``/``d_o``), lengths
(``L1``/``L2``), spreading (``sp1``/``sp2``), and cosines are derived from the
RayD-owned hit positions as structural packing, matching the reference oracles
``tests/reference/chain_ensemble.py`` / ``chain_realization.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as _dataclass_fields
from typing import TYPE_CHECKING

import torch

from witwin.channel_native.materials.encoding import face_material_tensors
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.propagation.geometry.kernels import (
    primitives as geometry_primitives,
)
from witwin.channel_native.propagation.geometry.reevaluate import (
    _cached_coplanar_face_groups,
)
from witwin.channel_native.propagation.geometry.reflection import (
    ReflectionEpcQuery,
    query_reflection_epc,
)
from witwin.channel_native.propagation.topology.discovery.reflection import (
    iter_reflection_multibounce_epc_requests,
    iter_reflection_order1_epc_requests,
    prepare_reflection_multibounce_plan,
    prepare_reflection_order1_plan,
)
from witwin.channel_native.propagation.topology.kernels import (
    compaction as topology_compaction,
)
from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from witwin.channel_native.propagation.enumerated.contracts import TopologyConfig

__all__ = [
    "KMAX_AD_DEPTH",
    "ScatterChainDiscovery",
    "ChainSamples",
    "build_chain_samples",
    "discover_scatter_chains",
]

# Native on-stack ReflectionChain capacity (field_transport_ad_common.cuh
# kMaxAdDepth). Each specular leg is padded to this width independently.
KMAX_AD_DEPTH = 8

# R2 low-discrepancy sequence (plastic constant); identical scheme to the
# single-bounce sampler (propagation/enumerated/scattering.py) so chain vertices
# are deterministic and refinement-stable.
_R2_ALPHA = (0.7548776662466927, 0.5698402909980532)
_MAX_SAMPLES_PER_FACE = 4096
_MIN_COS = 1.0e-4
_VISIBILITY_CHUNK = 1 << 20


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
    return geometry_primitives.deterministic_normalize_vec3(vec.contiguous(), eps=eps)


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

    Mirrors the single-bounce ``_keep_strongest_per_pair`` policy
    (propagation/enumerated/scattering.py) extended with the per-(tx, rx)
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


def _r2_barycentric(
    counts: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic per-face R2 barycentric samples (single-bounce parity)."""

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
    # convention; propagation/enumerated/scattering.py `_ensemble_rows`).
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
    normals = geometry_primitives.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    tri_a = topology_construction.deterministic_face_anchor_points(
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
            geometry_bridge.rayd_visibility_forward(
                handle, start[lo:hi].contiguous(), end[lo:hi].contiguous(), None
            )[0]
        )
    return torch.cat(masks)


def _offset_eps(points: torch.Tensor, scene_diagonal: torch.Tensor) -> torch.Tensor:
    return torch.maximum(
        torch.linalg.vector_norm(points, dim=-1) * 1.0e-6, scene_diagonal * 1.0e-6
    ).clamp_min(1.0e-6)


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
        selected = topology_compaction.deterministic_reflection_order1_compact(
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
        selected = topology_compaction.deterministic_reflection_sequence_compact(
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

    Mirrors ``propagation/enumerated/reflection._discovered_group_chains`` (used
    only by the non-exhaustive discovery branch of the shared iterators).
    """

    from witwin.channel_native.propagation.topology.kernels.sampling import (
        mc_sample_directions,
    )

    device = face_group_id.device
    ray_o = tx.reshape(1, 3).expand(ray_count, 3).contiguous()
    ray_d = mc_sample_directions(ray_count, tx.reshape(1, 3))
    ray_tmax = torch.empty((0,), device=device, dtype=torch.float32)
    out = geometry_bridge.rayd_trace_reflections_forward(
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
