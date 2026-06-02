"""
Native C++/CUDA implementation of the reflection accumulation kernel.

Python does ONLY:
  1. Build (path, rx) pair indices
  2. Visibility via DrJit BVH queries during EPC (unavoidable)
  3. Pack slot data + material into contiguous arrays
  4. ONE call to C++ kernel (chain EPC + Fresnel + Jones + field + atomicAdd)
  5. Wrap output

NO dr.scatter_reduce or per-axis loops on the Python side.
"""

from __future__ import annotations

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension
from witwin.channel.kernels.monitors.common.receiver_tiles import resolve_receiver_tiles
from witwin.channel.kernels.monitors.common.receiver_tiles.native_impl import receiver_index_for_tile_slot
from witwin.channel.utils.drjit_ops import Gather
from witwin.channel.utils.polarization import vector_eval, vector_scale, vector_select, vector_zero
from witwin.channel.trace.diffraction.constants import _cartesian_chunk_size
from witwin.channel.trace.materials import reflection_material_omega

from witwin.channel.kernels.monitors.common.reflection_family_tiles import (
    build_reflection_family_tile_plan,
)


def _array_has_grad(value) -> bool:
    try:
        return bool(dr.grad_enabled(value))
    except TypeError:
        return False


def _point_has_grad(value) -> bool:
    return any(_array_has_grad(getattr(value, axis)) for axis in ("x", "y", "z"))


def _reflection_paths_require_ad(*, paths, rx_pos) -> bool:
    if _point_has_grad(rx_pos):
        return True
    if paths is None:
        return False
    if "image_source" in paths and _point_has_grad(paths["image_source"]):
        return True
    chain_depth = int(paths.get("chain_depth", 0))
    for slot in range(chain_depth):
        if _point_has_grad(paths[f"path_plane_point_{slot}"]):
            return True
        if _point_has_grad(paths[f"path_plane_normal_{slot}"]):
            return True
    return False


def _pack_reflection_chunk_arrays(
    *,
    paths,
    chunk_path_idx,
    chain_depth: int,
    default_eta_r: float,
    default_sigma: float,
    default_gain: float,
):
    chunk_n = dr.width(chunk_path_idx)
    image_src = Gather.point3(paths["image_source"], chunk_path_idx)
    slot_arrays = []
    for slot in range(chain_depth):
        plane_point = Gather.point3(paths[f"path_plane_point_{slot}"], chunk_path_idx)
        plane_normal = Gather.vector3(paths[f"path_plane_normal_{slot}"], chunk_path_idx)
        slot_arrays.append(
            (
                plane_point,
                plane_normal,
                dr.full(wt.Float, default_eta_r, chunk_n),
                dr.full(wt.Float, default_sigma, chunk_n),
                dr.full(wt.Float, default_gain, chunk_n),
            )
        )

    return (
        image_src,
        dr.concat([slot[0].x for slot in slot_arrays]) if slot_arrays else dr.zeros(wt.Float, 0),
        dr.concat([slot[0].y for slot in slot_arrays]) if slot_arrays else dr.zeros(wt.Float, 0),
        dr.concat([slot[0].z for slot in slot_arrays]) if slot_arrays else dr.zeros(wt.Float, 0),
        dr.concat([slot[1].x for slot in slot_arrays]) if slot_arrays else dr.zeros(wt.Float, 0),
        dr.concat([slot[1].y for slot in slot_arrays]) if slot_arrays else dr.zeros(wt.Float, 0),
        dr.concat([slot[1].z for slot in slot_arrays]) if slot_arrays else dr.zeros(wt.Float, 0),
        dr.concat([slot[2] for slot in slot_arrays]) if slot_arrays else dr.zeros(wt.Float, 0),
        dr.concat([slot[3] for slot in slot_arrays]) if slot_arrays else dr.zeros(wt.Float, 0),
        dr.concat([slot[4] for slot in slot_arrays]) if slot_arrays else dr.zeros(wt.Float, 0),
    )


def _empty_reflection_outputs(n_rx: int):
    out_xr = dr.zeros(wt.Float, n_rx)
    out_xi = dr.zeros(wt.Float, n_rx)
    out_yr = dr.zeros(wt.Float, n_rx)
    out_yi = dr.zeros(wt.Float, n_rx)
    out_zr = dr.zeros(wt.Float, n_rx)
    out_zi = dr.zeros(wt.Float, n_rx)
    dr.eval(out_xr, out_xi, out_yr, out_yi, out_zr, out_zi)
    return out_xr, out_xi, out_yr, out_yi, out_zr, out_zi


def reflection_prefix_pair_filter(
    *,
    has_reflected_support,
    source_pos,
    edge_pos,
    edge_dir,
    n0,
    nn,
    chain_vector,
    wavelength: float,
    field_power_threshold: float = 1.0e-20,
):
    """Fuse the pair-space support and field-power filters for prefix states."""
    ext = _extension()
    n_pairs = dr.width(source_pos.x)
    if n_pairs == 0:
        empty = dr.zeros(wt.Bool, 0)
        return empty, empty

    support_i, keep_i = ext.reflection_prefix_filter_arrays(
        wt.Int32(dr.select(has_reflected_support, wt.Int32(1), wt.Int32(0))),
        source_pos.x,
        source_pos.y,
        source_pos.z,
        edge_pos.x,
        edge_pos.y,
        edge_pos.z,
        edge_dir.x,
        edge_dir.y,
        edge_dir.z,
        n0.x,
        n0.y,
        n0.z,
        nn.x,
        nn.y,
        nn.z,
        chain_vector["x"].real,
        chain_vector["x"].imag,
        chain_vector["y"].real,
        chain_vector["y"].imag,
        chain_vector["z"].real,
        chain_vector["z"].imag,
        float(wavelength),
        float(field_power_threshold),
    )
    return support_i != 0, keep_i != 0


def _accumulate_reflection_chunk_arrays(
    *,
    path_idx,
    rx_idx,
    valid_mask,
    image_source,
    slot_plane_point,
    slot_plane_normal,
    slot_eta_r,
    slot_sigma,
    slot_gain,
    rx_pos,
    tx_polarization,
    n_pairs: int,
    n_paths: int,
    chain_depth: int,
    k: float,
    omega: float,
):
    if n_pairs <= 0 or n_paths <= 0 or chain_depth <= 0:
        return _empty_reflection_outputs(dr.width(rx_pos.x))

    ext = _extension()
    return ext.reflection_accumulate(
        wt.Int32(path_idx),
        wt.Int32(rx_idx),
        wt.Int32(valid_mask),
        image_source.x,
        image_source.y,
        image_source.z,
        slot_plane_point.x,
        slot_plane_point.y,
        slot_plane_point.z,
        slot_plane_normal.x,
        slot_plane_normal.y,
        slot_plane_normal.z,
        slot_eta_r,
        slot_sigma,
        slot_gain,
        rx_pos.x,
        rx_pos.y,
        rx_pos.z,
        float(tx_polarization[0]),
        float(tx_polarization[1]),
        float(tx_polarization[2]),
        int(n_pairs),
        int(n_paths),
        int(chain_depth),
        float(k),
        float(omega),
    )


def _accumulate_reflection_paths_tiled(
    *,
    paths,
    rx_pos,
    scene,
    wavelength: float,
    k: float,
    reflection_detail,
    tx_polarization,
    receiver_tiles,
    default_eta_r: float,
    default_sigma: float,
    default_gain: float,
):
    from witwin.channel.trace.reflection.epc import (
        epc_reflection_chain_to_target,
    )
    from witwin.channel.trace.diffraction.geometry import _point_source_field

    chain_depth = int(paths.get("chain_depth", 0))
    n_paths = int(paths.get("n_paths", 0))
    n_rx = dr.width(rx_pos.x)
    if chain_depth <= 0 or n_paths <= 0 or receiver_tiles is None or int(receiver_tiles.n_tiles) <= 1:
        return None
    requires_ad = _reflection_paths_require_ad(paths=paths, rx_pos=rx_pos)
    if requires_ad:
        # The tiled rollout is a primal-only fast path. Keep AD-sensitive
        # workloads on the untiled native custom op so JVP/VJP stay in C++.
        return None

    tile_plan = build_reflection_family_tile_plan(
        paths=paths,
        scene=scene,
        receiver_tiles=receiver_tiles,
    )
    if tile_plan is None or tile_plan.tile_task_count <= 0:
        return None
    if int(tile_plan.estimated_pair_count) >= int(n_paths * n_rx):
        return None

    omega = float(reflection_material_omega(wavelength)[0])
    out_xr, out_xi, out_yr, out_yi, out_zr, out_zi = _empty_reflection_outputs(n_rx)

    for tile_idx in range(int(tile_plan.n_tiles)):
        tile_keep_idx = dr.compress(tile_plan.tile_task_tile_idx == wt.UInt32(tile_idx))
        local_n_paths = dr.width(tile_keep_idx)
        local_n_rx = int(receiver_tiles.tile_extent_0[tile_idx] * receiver_tiles.tile_extent_1[tile_idx])
        if local_n_paths <= 0 or local_n_rx <= 0:
            continue

        tile_family_idx = dr.gather(wt.UInt32, tile_plan.tile_task_family_idx, tile_keep_idx)
        chunk_size = _cartesian_chunk_size(local_n_paths, local_n_rx)

        for path_start in range(0, local_n_paths, chunk_size):
            chunk_n = min(chunk_size, local_n_paths - path_start)
            chunk_family_idx = dr.gather(
                wt.UInt32,
                tile_family_idx,
                dr.arange(wt.UInt32, chunk_n) + wt.UInt32(path_start),
            )
            n_pairs = chunk_n * local_n_rx
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            local_path_idx = pair_idx // local_n_rx
            tile_rx_slot_idx = pair_idx % local_n_rx
            rx_idx = receiver_index_for_tile_slot(
                receiver_tiles,
                wt.UInt32(tile_idx),
                tile_rx_slot_idx,
            )
            path_idx = dr.gather(wt.UInt32, chunk_family_idx, local_path_idx)

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
            if not bool(dr.any(valid)):
                continue

            (
                image_src,
                ppx,
                ppy,
                ppz,
                pnx,
                pny,
                pnz,
                s_eta,
                s_sig,
                s_gn,
            ) = _pack_reflection_chunk_arrays(
                paths=paths,
                chunk_path_idx=chunk_family_idx,
                chain_depth=chain_depth,
                default_eta_r=default_eta_r,
                default_sigma=default_sigma,
                default_gain=default_gain,
            )
            valid_int = wt.Int32(dr.select(valid, wt.UInt32(1), wt.UInt32(0)))
            chunk_xr, chunk_xi, chunk_yr, chunk_yi, chunk_zr, chunk_zi = _accumulate_reflection_chunk_arrays(
                path_idx=local_path_idx,
                rx_idx=rx_idx,
                valid_mask=valid_int,
                image_source=image_src,
                slot_plane_point=wt.Point3f(ppx, ppy, ppz),
                slot_plane_normal=wt.Vector3f(pnx, pny, pnz),
                slot_eta_r=s_eta,
                slot_sigma=s_sig,
                slot_gain=s_gn,
                rx_pos=rx_pos,
                tx_polarization=tx_polarization,
                n_pairs=n_pairs,
                n_paths=chunk_n,
                chain_depth=chain_depth,
                k=k,
                omega=omega,
            )
            out_xr = out_xr + chunk_xr
            out_xi = out_xi + chunk_xi
            out_yr = out_yr + chunk_yr
            out_yi = out_yi + chunk_yi
            out_zr = out_zr + chunk_zr
            out_zi = out_zi + chunk_zi

    return vector_eval(
        {
            "x": wt.Complex2f(out_xr, out_xi),
            "y": wt.Complex2f(out_yr, out_yi),
            "z": wt.Complex2f(out_zr, out_zi),
        }
    )


def reflection_accumulate_forward(
    *,
    rx_pos,
    scene,
    wavelength: float,
    k: float,
    source_paths_per_bounce: list,
    reflection_detail,
    tx_polarization=(1.0, 0.0, 0.0),
    receiver_tiles=None,
):
    """
    Native CUDA path for reflection accumulation.

    Same signature as ``drjit_impl.reflection_accumulate_forward``.

    When a receiver-tile descriptor is available and the workload is non-AD,
    the implementation runs reflection-family EPC tile-locally to avoid
    expanding ``path x all_receivers`` on dense monitor grids.
    """
    from witwin.channel.trace.reflection.epc import (
        epc_reflection_chain_to_target,
    )
    from witwin.channel.trace.materials import coerce_reflection_trace_detail
    receiver_tiles = resolve_receiver_tiles(
        grid=None,
        receiver_positions=rx_pos,
        receiver_tiles=receiver_tiles,
    )
    omega = float(reflection_material_omega(wavelength)[0])
    detail = coerce_reflection_trace_detail(reflection_detail)
    override_material = detail.reflection_material or {}
    default_eta_r = float(override_material.get("relative_permittivity", 5.0))
    default_sigma = float(override_material.get("conductivity", 0.0))
    default_gain = float(override_material.get("gain", detail.reflection_gain))
    n_rx = dr.width(rx_pos.x)
    polarization_per_bounce = []
    requires_ad = any(
        _reflection_paths_require_ad(paths=paths, rx_pos=rx_pos)
        for paths in source_paths_per_bounce
        if paths is not None
    )
    if requires_ad:
        from witwin.channel.kernels.trace.reflection import drjit_impl

        return drjit_impl.reflection_accumulate_forward(
            rx_pos=rx_pos,
            scene=scene,
            wavelength=wavelength,
            k=k,
            source_paths_per_bounce=source_paths_per_bounce,
            reflection_detail=reflection_detail,
            tx_polarization=tx_polarization,
        )

    for paths in source_paths_per_bounce:
        chain_depth = 0 if paths is None else int(paths.get("chain_depth", 0))
        n_paths = 0 if paths is None else int(paths.get("n_paths", 0))
        if n_paths <= 0 or chain_depth <= 0:
            polarization_per_bounce.append(vector_zero(n_rx))
            continue

        tiled_vector = _accumulate_reflection_paths_tiled(
            paths=paths,
            rx_pos=rx_pos,
            scene=scene,
            wavelength=wavelength,
            k=k,
            reflection_detail=reflection_detail,
            tx_polarization=tx_polarization,
            receiver_tiles=receiver_tiles,
            default_eta_r=default_eta_r,
            default_sigma=default_sigma,
            default_gain=default_gain,
        )
        if tiled_vector is not None:
            polarization_per_bounce.append(tiled_vector)
            continue

        out_xr, out_xi, out_yr, out_yi, out_zr, out_zi = _empty_reflection_outputs(n_rx)

        chunk_size = _cartesian_chunk_size(n_paths, n_rx)

        for path_start in range(0, n_paths, chunk_size):
            chunk_n = min(chunk_size, n_paths - path_start)
            n_pairs = chunk_n * n_rx
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            local_path_idx = pair_idx // n_rx
            path_idx = local_path_idx + wt.UInt32(path_start)
            rx_idx = pair_idx % n_rx
            chunk_path_idx = dr.arange(wt.UInt32, chunk_n) + wt.UInt32(path_start)

            # --- Visibility (DrJit BVH 鈥?unavoidable) ---
            target_pos = wt.Point3f(
                dr.gather(wt.Float, rx_pos.x, rx_idx),
                dr.gather(wt.Float, rx_pos.y, rx_idx),
                dr.gather(wt.Float, rx_pos.z, rx_idx),
            )
            valid, _ = epc_reflection_chain_to_target(
                paths=paths,
                path_idx=path_idx,
                target_pos=target_pos,
                scene=scene,
                target_adjacent_faces=(),
                reflection_detail=reflection_detail,
                wavelength=wavelength,
                tx_polarization=tx_polarization,
            )
            valid_int = wt.Int32(dr.select(valid, wt.UInt32(1), wt.UInt32(0)))

            (
                image_src,
                ppx,
                ppy,
                ppz,
                pnx,
                pny,
                pnz,
                s_eta,
                s_sig,
                s_gn,
            ) = _pack_reflection_chunk_arrays(
                paths=paths,
                chunk_path_idx=chunk_path_idx,
                chain_depth=chain_depth,
                default_eta_r=default_eta_r,
                default_sigma=default_sigma,
                default_gain=default_gain,
            )

            chunk_xr, chunk_xi, chunk_yr, chunk_yi, chunk_zr, chunk_zi = _accumulate_reflection_chunk_arrays(
                path_idx=local_path_idx,
                rx_idx=rx_idx,
                valid_mask=valid_int,
                image_source=image_src,
                slot_plane_point=wt.Point3f(ppx, ppy, ppz),
                slot_plane_normal=wt.Vector3f(pnx, pny, pnz),
                slot_eta_r=s_eta,
                slot_sigma=s_sig,
                slot_gain=s_gn,
                rx_pos=rx_pos,
                tx_polarization=tx_polarization,
                n_pairs=n_pairs,
                n_paths=chunk_n,
                chain_depth=chain_depth,
                k=k,
                omega=omega,
            )
            out_xr = out_xr + chunk_xr
            out_xi = out_xi + chunk_xi
            out_yr = out_yr + chunk_yr
            out_yi = out_yi + chunk_yi
            out_zr = out_zr + chunk_zr
            out_zi = out_zi + chunk_zi

        # Wrap output
        polarization_per_bounce.append(vector_eval({
            "x": wt.Complex2f(out_xr, out_xi),
            "y": wt.Complex2f(out_yr, out_yi),
            "z": wt.Complex2f(out_zr, out_zi),
        }))

    return polarization_per_bounce

