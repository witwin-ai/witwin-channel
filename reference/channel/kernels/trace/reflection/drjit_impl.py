"""
Pure-DrJit reference implementation of the reflection accumulation kernel.

Interface matches native/reflection/reflection_accumulate.h.
This module is the authoritative reference for numerical correctness;
the C++ kernel must produce bit-identical* results.

Functions
---------
reflection_accumulate_forward
    Fused EPC + point-source field + scatter-reduce for all
    (path, rx) pairs, per bounce depth.
"""

from __future__ import annotations

import drjit as dr

import witwin as wt
from witwin.channel.utils.polarization import vector_eval, vector_scale, vector_select, vector_zero
from witwin.channel.utils.drjit_ops import Gather
from witwin.channel.trace.diffraction.geometry import _point_source_field
from witwin.channel.trace.reflection.epc import epc_reflection_chain_to_target

_REFLECTION_EXACT_PAIR_CHUNK_BUDGET = 1 << 22


def _reflection_exact_chunk_size(n_paths: int, n_rx: int) -> int:
    if n_paths <= 0 or n_rx <= 0:
        return 0
    return max(1, min(int(n_paths), _REFLECTION_EXACT_PAIR_CHUNK_BUDGET // int(n_rx)))


def reflection_accumulate_forward(
    *,
    rx_pos,
    scene,
    wavelength: float,
    k: float,
    source_paths_per_bounce: list,
    reflection_detail,
    tx_polarization=(1.0, 0.0, 0.0),
):
    """
    Accumulate reflected vector field from reflection paths to receivers.

    This is the DrJit reference implementation of the fused mega-kernel
    ``native/reflection/reflection_accumulate.cu``.

    The function iterates over bounce depths, and for each depth computes
    the Cartesian product of (path, receiver) pairs in chunks.  Each pair
        runs reflection-chain EPC to get the Jones polarization transport
        vector, multiplies by the point-source field from the image source,
        and scatter-reduces the result into the per-receiver accumulator.

    Parameters
    ----------
    rx_pos : wt.Point3f
        Receiver positions ``[n_rx]``.
    scene : object
        Ray-query scene for visibility / BVH queries during EPC.
    wavelength : float
        Wavelength in metres.
    k : float
        Wave number.
    source_paths_per_bounce : list[dict]
        Per-bounce-depth path data from ``_trace_reflection_paths()``.
        Each dict has: ``image_source``, ``chain_depth``, ``n_paths``,
        and per-slot ``path_plane_point_{slot}``, ``path_plane_normal_{slot}``,
        ``path_prim_idx_{slot}``.
    reflection_detail : object
        Material/configuration detail (coerced internally by EPC).
    tx_polarization : tuple
        TX polarization direction.

    Returns
    -------
    polarization_per_bounce : list[dict]
        Per-bounce vector field ``{x: Complex, y: Complex, z: Complex}``
        accumulated to ``[n_rx]``.
    """
    n_rx = dr.width(rx_pos.x)
    zero_vector = vector_zero(n_rx)
    polarization_per_bounce = []

    for paths in source_paths_per_bounce:
        chain_depth = 0 if paths is None else int(paths.get("chain_depth", 0))
        n_paths = 0 if paths is None else int(paths.get("n_paths", 0))
        if n_paths <= 0 or chain_depth <= 0:
            polarization_per_bounce.append(zero_vector)
            continue

        total_vector = vector_zero(n_rx)
        path_chunk_size = _reflection_exact_chunk_size(n_paths, n_rx)

        for path_start in range(0, n_paths, path_chunk_size):
            chunk_n_paths = min(path_chunk_size, n_paths - path_start)
            n_pairs = chunk_n_paths * n_rx
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            path_idx = pair_idx // n_rx + wt.UInt32(path_start)
            rx_idx = pair_idx % n_rx

            image_source = Gather.point3(paths["image_source"], path_idx)
            target_pos = wt.Point3f(
                dr.gather(wt.Float, rx_pos.x, rx_idx),
                dr.gather(wt.Float, rx_pos.y, rx_idx),
                dr.gather(wt.Float, rx_pos.z, rx_idx),
            )

            valid, chain_vector = epc_reflection_chain_to_target(
                paths=paths,
                path_idx=path_idx,
                target_pos=target_pos,
                scene=scene,
                target_adjacent_faces=(),
                reflection_detail=reflection_detail,
                wavelength=wavelength,
                tx_polarization=tx_polarization,
            )

            unit_field = _point_source_field(image_source, wt.Complex2f(1.0, 0.0), target_pos, wavelength, k)
            polarization_field = vector_scale(chain_vector, unit_field)
            polarization_field = vector_select(valid, polarization_field, vector_zero(n_pairs))

            for axis in ("x", "y", "z"):
                dr.scatter_reduce(
                    dr.ReduceOp.Add,
                    total_vector[axis].real,
                    polarization_field[axis].real,
                    rx_idx,
                    valid,
                )
                dr.scatter_reduce(
                    dr.ReduceOp.Add,
                    total_vector[axis].imag,
                    polarization_field[axis].imag,
                    rx_idx,
                    valid,
                )

        dr.eval(
            total_vector["x"].real,
            total_vector["x"].imag,
            total_vector["y"].real,
            total_vector["y"].imag,
            total_vector["z"].real,
            total_vector["z"].imag,
        )
        polarization_per_bounce.append(vector_eval(total_vector))

    return polarization_per_bounce
