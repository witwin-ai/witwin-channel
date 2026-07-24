"""Scene-policy filtering and cross-structure merging for diffraction edges.

The native edge-geometry kernel selects every interior wedge and every
boundary half-plane edge regardless of the scene's edge policy, and exports
one boundary record per structure even when two structures share the same
geometric edge. This module refines the exported geometry tuple so that:

- the scene's ``EdgePolicy`` (``vertical_only`` filter, ``boundary_edge_policy``)
  is actually enforced for path generation (audit DF-4), and
- boundary edges shared between two structures are merged into a single wedge
  record with the correct exterior angle instead of two duplicate half-plane
  records that double count the diffracted field (audit D-6).
"""

from __future__ import annotations

import math

import torch

from .edge_policy import DEFAULT_EDGE_POLICY, EdgePolicy

_ENDPOINT_QUANTIZATION = 1.0e-4
_NORMAL_COS_TOL = 1.0 - 1.0e-5
_TWO_PI = 2.0 * math.pi


def resolve_scene_edge_policy(scene: object) -> EdgePolicy:
    policy = getattr(scene, "metadata", {}).get("sionna_import_edge_policy")
    return policy if isinstance(policy, EdgePolicy) else DEFAULT_EDGE_POLICY


def _lexicographic_min_first(p0: torch.Tensor, p1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    le = (
        (p0[:, 0] < p1[:, 0])
        | ((p0[:, 0] == p1[:, 0]) & (p0[:, 1] < p1[:, 1]))
        | ((p0[:, 0] == p1[:, 0]) & (p0[:, 1] == p1[:, 1]) & (p0[:, 2] <= p1[:, 2]))
    ).unsqueeze(1)
    return torch.where(le, p0, p1), torch.where(le, p1, p0)


def _duplicate_boundary_pairs(
    records: object,
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group boundary edges by quantized endpoints.

    Returns (first_of_pair, second_of_pair, extra_duplicates); the extras are
    third-or-later records of a group and are simply deselected.
    """

    device = candidate.device
    empty = torch.empty((0,), device=device, dtype=torch.long)
    index = candidate.nonzero(as_tuple=False).squeeze(1)
    if int(index.numel()) < 2:
        return empty, empty, empty
    v0 = records.edge_v0[index].to(dtype=torch.long)
    v1 = records.edge_v1[index].to(dtype=torch.long)
    p0 = torch.round(records.vertices[v0] / _ENDPOINT_QUANTIZATION).to(dtype=torch.long)
    p1 = torch.round(records.vertices[v1] / _ENDPOINT_QUANTIZATION).to(dtype=torch.long)
    first, second = _lexicographic_min_first(p0, p1)
    key = torch.cat([first, second], dim=1)
    _, inverse, counts = torch.unique(key, dim=0, return_inverse=True, return_counts=True)
    order = torch.argsort(inverse, stable=True)
    sorted_inverse = inverse[order]
    is_first = torch.ones_like(sorted_inverse, dtype=torch.bool)
    is_first[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
    is_second = torch.zeros_like(is_first)
    is_second[1:] = is_first[:-1] & (sorted_inverse[1:] == sorted_inverse[:-1])
    in_pair = counts[sorted_inverse] >= 2
    pair_first = index[order[is_first & in_pair]]
    pair_second = index[order[is_second]]
    extras = index[order[~is_first & ~is_second]]
    return pair_first, pair_second, extras


def refine_edge_geometry(
    rayd: object,
    geometry: tuple[torch.Tensor, ...],
    *,
    policy: EdgePolicy | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return the geometry tuple with policy filtering and shared-edge merges."""

    (selected, edge_pos, edge_dir, lengths, line_min, line_max, n0, n1, face0, face1, exterior_angle) = geometry
    if int(selected.numel()) == 0:
        return geometry
    records = rayd.edge_records()
    if policy is None:
        policy = rayd.runtime_cache.get("edge_policy")
    if not isinstance(policy, EdgePolicy):
        policy = DEFAULT_EDGE_POLICY

    selected = selected.clone()
    n1 = n1.clone()
    face1 = face1.clone()
    exterior_angle = exterior_angle.clone()

    boundary = records.face1 < 0
    pair_first, pair_second, extras = _duplicate_boundary_pairs(records, selected & boundary)
    if int(pair_first.numel()) > 0:
        normal_a = n0[pair_first]
        normal_b = n0[pair_second]
        normal_dot = (normal_a * normal_b).sum(dim=1).clamp(-1.0, 1.0)
        interior_angle = torch.arccos((-normal_dot).clamp(-1.0, 1.0))
        merged_exterior = _TWO_PI - interior_angle
        coplanar = normal_dot.abs() >= _NORMAL_COS_TOL
        keep = (merged_exterior > math.pi * (1.0 + 1.0e-6)) & ~coplanar
        selected[pair_first] = keep
        selected[pair_second] = False
        n1[pair_first] = normal_b
        face1[pair_first] = face0[pair_second]
        exterior_angle[pair_first] = merged_exterior
        boundary = boundary.clone()
        boundary[pair_first] = False
    if int(extras.numel()) > 0:
        selected[extras] = False

    if policy.boundary_edge_policy != "half_plane":
        selected &= ~boundary
    if policy.vertical_only:
        v0 = records.edge_v0.to(dtype=torch.long)
        v1 = records.edge_v1.to(dtype=torch.long)
        delta = records.vertices[v1] - records.vertices[v0]
        length = delta.norm(dim=1).clamp_min(1.0e-6)
        selected &= (delta[:, 2].abs() / length) > float(policy.vertical_ratio)

    return (selected, edge_pos, edge_dir, lengths, line_min, line_max, n0, n1, face0, face1, exterior_angle)


__all__ = ["refine_edge_geometry", "resolve_scene_edge_policy"]
