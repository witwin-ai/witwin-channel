"""
Pure-DrJit reference implementation of the UTD accumulation kernels.

Interface matches native/utd/utd_accumulate.h -- all SoA DrJit arrays.
This module is the authoritative reference for numerical correctness;
the C++ kernel must produce bit-identical* results.

Functions
---------
utd_accumulate_forward
    Fused field evaluation + scatter-reduce for all (state, rx) pairs.
    Returns scalar totals (direct/multi), vector totals, and optional
    per-edge breakdowns.
utd_accumulate_backward
    Reverse-mode VJP (delegated to DrJit AD on the forward graph).
"""

from __future__ import annotations

import math

import drjit as dr

import witwin as wt
from witwin.channel.trace.diffraction.field import _edge_state_field_to_targets
from witwin.channel.kernels.trace.packed_state import gather_field_evaluation_state_fields
from witwin.channel.trace.diffraction.geometry import _edge_owner_structure_idx, _segment_visibility_mask
from witwin.channel.utils.drjit_ops import ArrayInit, eval_complex
from witwin.channel.utils.polarization import (
    scalarize_tangential_jones,
    tangential_jones,
    vector_eval,
    vector_zero,
)
from witwin.channel.trace.diffraction.constants import (
    OWNERSHIP_DIRECT_DIFFRACTION,
    OWNERSHIP_MIXED_DIFFRACTION,
    _ownership_code_from_depths,
)
from witwin.channel.config import coerce_diffraction_execution


# ---------------------------------------------------------------------------
# Forward accumulation
# ---------------------------------------------------------------------------

_UTD_DRJIT_PAIR_CHUNK_BUDGET = 1 << 22


def _utd_pair_chunk_shape(n_states: int, n_rx: int) -> tuple[int, int]:
    if n_states <= 0 or n_rx <= 0:
        return 0, 0
    pair_budget = max(1, int(_UTD_DRJIT_PAIR_CHUNK_BUDGET))
    state_chunk_size = max(1, min(n_states, int(math.sqrt(pair_budget))))
    rx_chunk_size = max(1, min(n_rx, pair_budget // state_chunk_size))
    return state_chunk_size, rx_chunk_size

def utd_accumulate_forward(
    state_arrays: dict,
    rx_pos,
    k: float,
    n_edges: int,
    return_per_edge: bool,
    *,
    scene=None,
    wavelength: float | None = None,
    material_detail=None,
    rx_polarization=None,
    receiver_axis: str = "z",
    execution=None,
):
    """
    Accumulate UTD diffraction field from edge states to receivers.

    This is the DrJit reference implementation of the fused mega-kernel
    ``native/utd/utd_accumulate.cu::utd_accumulate_forward_kernel``.

    Parameters
    ----------
    state_arrays : dict
        SoA state dictionary produced by ``_make_state_arrays``.
    rx_pos : wt.Point3f
        Receiver positions ``[n_rx]``.
    k : float
        Wave number.
    n_edges : int
        Total edge count (needed for *return_per_edge*).
    return_per_edge : bool
        If True, return per-edge scalar contributions.
    scene : optional
        Ray-query scene for visibility checks (None = skip visibility).
    wavelength : float, optional
        Wavelength in metres (for material Fresnel).
    material_detail : optional
        Material configuration.
    rx_polarization : tuple or None
        Receiver polarization direction.  ``(1.0, 0.0, 0.0)`` when None.
    receiver_axis : str
        Axis for tangential-Jones scalarization (``"z"``, ``"x"``, or ``"y"``).
    execution : optional
        Diffraction execution config.

    Returns
    -------
    direct_total : wt.Complex2f
        Scalar field for direct-diffraction paths, per receiver ``[n_rx]``.
    multi_total : wt.Complex2f
        Scalar field for mixed (reflection+diffraction) paths, per receiver ``[n_rx]``.
    direct_vector_total : dict
        Complex3 vector ``{x, y, z}`` for direct diffraction, per receiver ``[n_rx]``.
    multi_vector_total : dict
        Complex3 vector ``{x, y, z}`` for mixed paths, per receiver ``[n_rx]``.
    per_edge_list : list
        Per-edge ``(real, imag)`` scalar tuples when *return_per_edge* is True,
        otherwise an empty list.
    """
    execution = coerce_diffraction_execution(execution)
    n_states = state_arrays["n_states"]
    n_rx = dr.width(rx_pos.x)

    if n_states == 0 or n_rx == 0:
        zero_field = ArrayInit.complex_zero(n_rx)
        zero_vector = vector_zero(n_rx)
        per_edge_list = []
        if return_per_edge:
            per_edge_list = [(zero_field.real, zero_field.imag) for _ in range(n_edges)]
        return zero_field, zero_field, zero_vector, zero_vector, per_edge_list

    direct_total = ArrayInit.complex_zero(n_rx)
    multi_total = ArrayInit.complex_zero(n_rx)
    direct_vector_total = vector_zero(n_rx)
    multi_vector_total = vector_zero(n_rx)
    active_rx_polarization = (1.0, 0.0, 0.0) if rx_polarization is None else rx_polarization

    per_edge_list = []
    per_edge_vector = None
    if return_per_edge:
        per_edge_vector = vector_zero(n_edges * n_rx)

    state_chunk_size, base_rx_chunk_size = _utd_pair_chunk_shape(n_states, n_rx)

    for state_start in range(0, n_states, state_chunk_size):
        chunk_n_states = min(state_chunk_size, n_states - state_start)
        rx_chunk_size = max(1, min(n_rx, base_rx_chunk_size))
        rx_chunk_size = max(1, min(rx_chunk_size, _UTD_DRJIT_PAIR_CHUNK_BUDGET // chunk_n_states))

        for rx_start in range(0, n_rx, rx_chunk_size):
            chunk_n_rx = min(rx_chunk_size, n_rx - rx_start)
            n_pairs = chunk_n_states * chunk_n_rx
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            state_idx = pair_idx // chunk_n_rx + wt.UInt32(state_start)
            rx_idx = pair_idx % chunk_n_rx + wt.UInt32(rx_start)

            # Optional visibility filtering via scene ray-test
            if scene is not None:
                state_edge_pos = dr.gather(wt.Point3f, state_arrays["edge_pos"], state_idx)
                state_adjacent_face0 = dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx)
                state_adjacent_face1 = dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx)
                owner_structure_idx = _edge_owner_structure_idx(
                    scene,
                    state_adjacent_face0,
                    state_adjacent_face1,
                )
                batch_rx_all = wt.Point3f(
                    dr.gather(wt.Float, rx_pos.x, rx_idx),
                    dr.gather(wt.Float, rx_pos.y, rx_idx),
                    dr.gather(wt.Float, rx_pos.z, rx_idx),
                )
                pair_keep_idx = dr.compress(
                    _segment_visibility_mask(
                        state_edge_pos,
                        batch_rx_all,
                        scene,
                        ignore_prim_idx=(state_adjacent_face0, state_adjacent_face1),
                        ignore_structure_idx=owner_structure_idx,
                    )
                )
                if dr.width(pair_keep_idx) == 0:
                    continue
                state_idx = pair_keep_idx // chunk_n_rx + wt.UInt32(state_start)
                rx_idx = pair_keep_idx % chunk_n_rx + wt.UInt32(rx_start)
            else:
                batch_rx_all = None
                pair_keep_idx = None

            batch_states = gather_field_evaluation_state_fields(
                state_arrays,
                state_idx,
                include_stored_operators=wavelength is None,
            )
            batch_rx = (
                dr.gather(wt.Point3f, batch_rx_all, pair_keep_idx)
                if scene is not None
                else wt.Point3f(
                    dr.gather(wt.Float, rx_pos.x, rx_idx),
                    dr.gather(wt.Float, rx_pos.y, rx_idx),
                    dr.gather(wt.Float, rx_pos.z, rx_idx),
                )
            )
            _, pair_vector = _edge_state_field_to_targets(
                batch_states,
                batch_rx,
                k,
                return_vector=True,
                wavelength=wavelength,
                material_detail=material_detail,
                scene=scene,
                smooth_exterior_shadow=True,
            )

            ownership_code = _ownership_code_from_depths(
                dr.gather(wt.UInt32, state_arrays["prefix_reflection_depth"], state_idx),
                dr.gather(wt.UInt32, state_arrays["intermediate_reflection_depth"], state_idx),
                dr.gather(wt.UInt32, state_arrays["suffix_reflection_depth"], state_idx),
            )
            direct_mask = ownership_code == wt.UInt32(OWNERSHIP_DIRECT_DIFFRACTION)
            multi_mask = ownership_code == wt.UInt32(OWNERSHIP_MIXED_DIFFRACTION)

            for axis in ("x", "y", "z"):
                dr.scatter_reduce(
                    dr.ReduceOp.Add, direct_vector_total[axis].real,
                    dr.select(direct_mask, pair_vector[axis].real, wt.Float(0.0)), rx_idx, direct_mask
                )
                dr.scatter_reduce(
                    dr.ReduceOp.Add, direct_vector_total[axis].imag,
                    dr.select(direct_mask, pair_vector[axis].imag, wt.Float(0.0)), rx_idx, direct_mask
                )
                dr.scatter_reduce(
                    dr.ReduceOp.Add, multi_vector_total[axis].real,
                    dr.select(multi_mask, pair_vector[axis].real, wt.Float(0.0)), rx_idx, multi_mask
                )
                dr.scatter_reduce(
                    dr.ReduceOp.Add, multi_vector_total[axis].imag,
                    dr.select(multi_mask, pair_vector[axis].imag, wt.Float(0.0)), rx_idx, multi_mask
                )

            if return_per_edge:
                edge_idx = dr.gather(wt.UInt32, state_arrays["edge_idx"], state_idx)
                flat_idx = edge_idx * n_rx + rx_idx
                for axis in ("x", "y", "z"):
                    dr.scatter_reduce(
                        dr.ReduceOp.Add, per_edge_vector[axis].real,
                        dr.select(direct_mask, pair_vector[axis].real, wt.Float(0.0)), flat_idx, direct_mask
                    )
                    dr.scatter_reduce(
                        dr.ReduceOp.Add, per_edge_vector[axis].imag,
                        dr.select(direct_mask, pair_vector[axis].imag, wt.Float(0.0)), flat_idx, direct_mask
                    )

    if return_per_edge:
        for edge_idx_scalar in range(n_edges):
            gather_idx = dr.arange(wt.UInt32, n_rx) + wt.UInt32(edge_idx_scalar * n_rx)
            edge_vector = {
                axis: eval_complex(dr.gather(wt.Complex2f, per_edge_vector[axis], gather_idx))
                for axis in ("x", "y", "z")
            }
            edge_scalar = eval_complex(
                scalarize_tangential_jones(
                    tangential_jones(edge_vector, axis=receiver_axis),
                    active_rx_polarization,
                    axis=receiver_axis,
                )
            )
            per_edge_list.append((edge_scalar.real, edge_scalar.imag))

    direct_vector_total = vector_eval(direct_vector_total)
    multi_vector_total = vector_eval(multi_vector_total)
    direct_total = eval_complex(
        scalarize_tangential_jones(
            tangential_jones(direct_vector_total, axis=receiver_axis),
            active_rx_polarization,
            axis=receiver_axis,
        )
    )
    multi_total = eval_complex(
        scalarize_tangential_jones(
            tangential_jones(multi_vector_total, axis=receiver_axis),
            active_rx_polarization,
            axis=receiver_axis,
        )
    )
    return direct_total, multi_total, direct_vector_total, multi_vector_total, per_edge_list


# ---------------------------------------------------------------------------
# Backward (VJP) -- delegates to DrJit AD
# ---------------------------------------------------------------------------

def utd_accumulate_backward(
    state_arrays: dict,
    rx_pos,
    k: float,
    n_edges: int,
    return_per_edge: bool,
    grad_direct_total,
    grad_multi_total,
    grad_direct_vector,
    grad_multi_vector,
    *,
    scene=None,
    wavelength: float | None = None,
    material_detail=None,
    rx_polarization=None,
    receiver_axis: str = "z",
    execution=None,
):
    """
    VJP of ``utd_accumulate_forward``.

    In the DrJit backend this is handled automatically by DrJit's AD graph
    when the forward call is made inside ``dr.enable_grad()`` scope.
    The C++ native path uses a hand-written backward kernel instead.

    This function is a placeholder that documents the interface; callers
    should use DrJit AD directly for the reference path.
    """
    raise NotImplementedError(
        "DrJit backward is implicit via AD graph. "
        "Use dr.backward() on the forward output instead."
    )
