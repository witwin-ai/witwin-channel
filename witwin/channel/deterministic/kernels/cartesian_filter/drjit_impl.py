"""
Pure-DrJit reference implementation of the fused Cartesian filter.

Interface matches cartesian_filter/cartesian_filter.h.

The C++ kernel does all validity checks in one pass with a single
atomic-compaction output. This DrJit path chains ``dr.compress()``
calls as the reference baseline.
"""

from __future__ import annotations

import drjit as dr

from witwin.channel.deterministic import types as wt


def cartesian_filter_bruteforce(
    prev_edge_idx,
    prev_edge_history: list,
    prev_power,
    n_prev: int,
    n_edges: int,
    min_power: float = 1e-20,
):
    """
    Fused Cartesian filter: distinct-edge + power threshold + compaction.

    Creates all ``(n_prev x n_edges)`` candidate pairs, filters them by:
      1. Power threshold (``prev_power > min_power``)
      2. Distinct edge (``edge_idx != prev_edge_idx`` and not in history)

    Returns compacted ``(prev_idx, edge_idx)`` arrays of valid pairs.

    Parameters
    ----------
    prev_edge_idx : wt.UInt32
        Current edge index per previous state ``[n_prev]``.
    prev_edge_history : list[wt.Int32]
        Edge history arrays ``[n_prev]`` per slot. Unused slots should be -1.
    prev_power : wt.Float
        Incident power per previous state ``[n_prev]``.
    n_prev : int
        Number of previous states.
    n_edges : int
        Number of candidate edges.
    min_power : float
        Power threshold.

    Returns
    -------
    out_prev_idx : wt.UInt32
        Previous-state indices of valid pairs.
    out_edge_idx : wt.UInt32
        Edge indices of valid pairs.
    """
    if n_prev <= 0 or n_edges <= 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)

    n_pairs = n_prev * n_edges
    pair_idx = dr.arange(wt.UInt32, n_pairs)
    prev_idx = pair_idx // n_edges
    edge_idx = pair_idx % n_edges

    # Check 1: power threshold
    power = dr.gather(wt.Float, prev_power, prev_idx)
    valid = power > min_power

    # Check 2: distinct from current edge
    cur_edge = dr.gather(wt.UInt32, prev_edge_idx, prev_idx)
    valid = valid & (edge_idx != cur_edge)

    # Check 3: distinct from edge history
    for hist in prev_edge_history:
        h = dr.gather(wt.Int32, hist, prev_idx)
        valid = valid & ((h < 0) | (wt.UInt32(h) != edge_idx))

    # Compact
    keep = dr.compress(valid)
    if dr.width(keep) == 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)

    return (
        dr.gather(wt.UInt32, prev_idx, keep),
        dr.gather(wt.UInt32, edge_idx, keep),
    )


def compact_index_pairs(lhs_idx, rhs_idx, active_mask):
    if dr.width(lhs_idx) == 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)
    keep = dr.compress(active_mask)
    if dr.width(keep) == 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)
    return (
        dr.gather(wt.UInt32, wt.UInt32(lhs_idx), keep),
        dr.gather(wt.UInt32, wt.UInt32(rhs_idx), keep),
    )
