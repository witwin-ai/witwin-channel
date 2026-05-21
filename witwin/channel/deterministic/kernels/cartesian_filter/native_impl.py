"""
Native C++/CUDA implementation of the fused Cartesian filter.

ONE kernel call replaces the multi-round dr.compress() + dr.width() loop.
Python only packs inputs and reads back the compacted count.
"""

from __future__ import annotations

import drjit as dr

from witwin.channel.deterministic import types as wt
from witwin.channel._native.deterministic import NativeExtension


def cartesian_filter_bruteforce(
    prev_edge_idx,
    prev_edge_history: list,
    prev_power,
    n_prev: int,
    n_edges: int,
    min_power: float = 1e-20,
):
    """
    Native CUDA fused Cartesian filter.

    Same signature as ``drjit_impl.cartesian_filter_bruteforce``.
    """
    ext = NativeExtension.load()
    if n_prev <= 0 or n_edges <= 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)

    history_size = len(prev_edge_history)
    prev_edge_idx_i = wt.Int32(prev_edge_idx)
    hist_flat = dr.concat(prev_edge_history) if history_size > 0 else dr.zeros(wt.Int32, 0)
    out_prev, out_edge = ext.cartesian_filter_bruteforce_arrays(
        prev_edge_idx_i,
        hist_flat,
        history_size,
        prev_power,
        n_prev,
        n_edges,
        min_power,
    )
    return (
        wt.UInt32(out_prev),
        wt.UInt32(out_edge),
    )


def deduplicate_cartesian_pairs(prev_idx, edge_idx, n_edges: int):
    """
    Sort and deduplicate `(prev_idx, edge_idx)` pairs on the GPU.

    Parameters
    ----------
    prev_idx : wt.UInt32
        Previous-state indices.
    edge_idx : wt.UInt32
        Edge indices.
    n_edges : int
        Edge count used to build the canonical lexicographic order.
    """
    ext = NativeExtension.load()
    count = dr.width(prev_idx)
    if count == 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)

    out_prev, out_edge = ext.deduplicate_cartesian_pairs_arrays(
        wt.Int32(prev_idx),
        wt.Int32(edge_idx),
        n_edges,
    )
    return (
        wt.UInt32(out_prev),
        wt.UInt32(out_edge),
    )


def compact_index_pairs(lhs_idx, rhs_idx, active_mask):
    """
    Compact already-built pair index arrays with an active mask.

    Parameters
    ----------
    lhs_idx : wt.UInt32 | wt.Int32
        Left-hand pair indices.
    rhs_idx : wt.UInt32 | wt.Int32
        Right-hand pair indices.
    active_mask : wt.Bool | wt.Int32
        Mask selecting the pairs to keep.
    """
    ext = NativeExtension.load()
    count = dr.width(lhs_idx)
    if count == 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)

    active_i = wt.Int32(dr.select(active_mask != 0, wt.Int32(1), wt.Int32(0)))
    out_lhs, out_rhs = ext.compact_index_pairs_arrays(
        wt.Int32(lhs_idx),
        wt.Int32(rhs_idx),
        active_i,
    )
    return wt.UInt32(out_lhs), wt.UInt32(out_rhs)
