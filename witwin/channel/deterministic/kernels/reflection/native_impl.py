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

from dataclasses import replace

import drjit as dr

from witwin.channel.deterministic import types as wt
from witwin.channel._native.deterministic import NativeExtension
from witwin.channel.core.runtime import Rx, Tx, Wave
from witwin.channel.core.numerics.arrays import gather_point3, gather_vector3, scalar as scalar_value
from witwin.channel.core.physics.polarization import vector_eval, vector_scale, vector_zero
from witwin.channel.deterministic.diffraction.state import Geo
from witwin.channel.core.runtime import material_angular_frequency, resolve_surface_material
from witwin.channel.deterministic.reflection.boundary import nearest_surface_boundary_edge
from witwin.channel.deterministic.reflection.detail import coerce_trace_detail


def _pack_reflection_chunk_arrays(
    *,
    paths,
    scene,
    chunk_path_idx,
    chain_depth: int,
    default_gain: float,
):
    chunk_n = dr.width(chunk_path_idx)
    image_src = gather_point3(paths.image_source, chunk_path_idx)
    slot_arrays = []
    for slot in range(chain_depth):
        plane_point = gather_point3(paths.plane_point(slot), chunk_path_idx)
        plane_normal = gather_vector3(paths.plane_normal(slot), chunk_path_idx)
        prim_idx = dr.gather(wt.Int32, paths.prim_idx(slot), chunk_path_idx)
        material_inputs = resolve_surface_material(
            scene=scene,
            prim_idx=prim_idx,
            default_gain=default_gain,
            valid_mask=prim_idx >= 0,
        )
        slot_arrays.append(
            (
                plane_point,
                plane_normal,
                material_inputs["eta_r"],
                material_inputs["mu_r"],
                material_inputs["sigma"],
                material_inputs["gain"],
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
        dr.concat([slot[5] for slot in slot_arrays]) if slot_arrays else dr.zeros(wt.Float, 0),
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
    ext = NativeExtension.load()
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


def reflection_prefix_compact_representatives(
    *,
    bounce_count,
    discovery_count,
    representative_ray_index,
    global_prim_ids,
    image_sources,
    canonical_prim_table,
    ray_count: int,
    max_bounces: int,
    depth: int,
    image_source_tolerance: float,
):
    ext = NativeExtension.load()
    return ext.reflection_prefix_compact_representatives(
        wt.Int32(bounce_count),
        wt.Int32(discovery_count),
        wt.Int32(representative_ray_index),
        wt.Int32(global_prim_ids),
        wt.Float(image_sources.x),
        wt.Float(image_sources.y),
        wt.Float(image_sources.z),
        wt.Int32(canonical_prim_table),
        int(ray_count),
        int(max_bounces),
        int(depth),
        float(image_source_tolerance),
    )


def _accumulate_reflection_chunk_arrays(
    *,
    path_idx,
    rx_idx,
    valid_mask,
    image_source,
    slot_plane_point,
    slot_plane_normal,
    slot_eta_r,
    slot_mu_r,
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

    ext = NativeExtension.load()
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
        slot_mu_r,
        slot_sigma,
        slot_gain,
        rx_pos.x,
        rx_pos.y,
        rx_pos.z,
        scalar_value(tx_polarization[0]),
        scalar_value(tx_polarization[1]),
        scalar_value(tx_polarization[2]),
        int(n_pairs),
        int(n_paths),
        int(chain_depth),
        float(k),
        float(omega),
    )


def _accumulate_reflection_f_weight_chunk_arrays(
    *,
    path_idx,
    rx_idx,
    valid_mask,
    image_source,
    slot_plane_point,
    slot_plane_normal,
    slot_eta_r,
    slot_mu_r,
    slot_sigma,
    slot_gain,
    transition_support_valid,
    transition_primary_side,
    transition_edge_distance,
    transition_edge_v0,
    transition_edge_v1,
    adjacent_valid,
    adjacent_plane_point,
    adjacent_plane_normal,
    adjacent_eta_r,
    adjacent_mu_r,
    adjacent_sigma,
    adjacent_gain,
    rx_pos,
    tx_pos,
    tx_polarization,
    n_pairs: int,
    n_paths: int,
    chain_depth: int,
    k: float,
    omega: float,
):
    if n_pairs <= 0 or n_paths <= 0 or chain_depth <= 0:
        return _empty_reflection_outputs(dr.width(rx_pos.x))

    ext = NativeExtension.load()
    return ext.reflection_accumulate_f_weight(
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
        slot_mu_r,
        slot_sigma,
        slot_gain,
        wt.Int32(transition_support_valid),
        wt.Int32(transition_primary_side),
        transition_edge_distance,
        transition_edge_v0.x,
        transition_edge_v0.y,
        transition_edge_v0.z,
        transition_edge_v1.x,
        transition_edge_v1.y,
        transition_edge_v1.z,
        wt.Int32(adjacent_valid),
        adjacent_plane_point.x,
        adjacent_plane_point.y,
        adjacent_plane_point.z,
        adjacent_plane_normal.x,
        adjacent_plane_normal.y,
        adjacent_plane_normal.z,
        adjacent_eta_r,
        adjacent_mu_r,
        adjacent_sigma,
        adjacent_gain,
        rx_pos.x,
        rx_pos.y,
        rx_pos.z,
        scalar_value(tx_pos.x),
        scalar_value(tx_pos.y),
        scalar_value(tx_pos.z),
        scalar_value(tx_polarization[0]),
        scalar_value(tx_polarization[1]),
        scalar_value(tx_polarization[2]),
        int(n_pairs),
        int(n_paths),
        int(chain_depth),
        float(k),
        float(omega),
    )


def reflection_accumulate_forward(
    *,
    rx: Rx,
    tx: Tx,
    scene,
    wave: Wave,
    source_paths_per_bounce: list,
    reflection_detail,
):
    """
    Native CUDA path for reflection accumulation.

    Uses the runtime-bundle reflection accumulation signature.

    The implementation uses chunked Cartesian replay for each bounce depth.
    """
    from witwin.channel.deterministic.reflection.epc import chain_to_target as epc_reflection_chain_to_target
    rx_pos = rx.positions
    wavelength = wave.wavelength_scalar
    k = wave.k_scalar
    tx_polarization = tx.polarization_tuple
    detail = coerce_trace_detail(reflection_detail)
    if (
        detail.reflection_secondary_visibility_mode != "hard"
        and detail.reflection_transition_mode == "f_weight_native"
    ):
        raise RuntimeError(
            "reflection_secondary_visibility_mode='f_weight' is not available with "
            "reflection_transition_mode='f_weight_native' until the native CUDA "
            "secondary-visibility descriptor and JVP path are implemented."
        )
    if detail.reflection_secondary_visibility_mode != "hard":
        return _reflection_accumulate_f_weight_reference(
            rx=rx,
            tx=tx,
            scene=scene,
            wave=wave,
            source_paths_per_bounce=source_paths_per_bounce,
            reflection_detail=detail,
        )
    if detail.reflection_transition_mode == "f_weight_reference":
        return _reflection_accumulate_f_weight_reference(
            rx=rx,
            tx=tx,
            scene=scene,
            wave=wave,
            source_paths_per_bounce=source_paths_per_bounce,
            reflection_detail=detail,
        )
    if detail.reflection_transition_mode == "f_weight_native":
        return _reflection_accumulate_f_weight_native_cuda(
            rx=rx,
            tx=tx,
            scene=scene,
            wave=wave,
            source_paths_per_bounce=source_paths_per_bounce,
            reflection_detail=detail,
        )
    default_gain = float(detail.reflection_gain)
    omega = float(material_angular_frequency(wavelength)[0])
    n_rx = dr.width(rx_pos.x)
    polarization_per_bounce = []

    for paths in source_paths_per_bounce:
        chain_depth = 0 if paths is None else int(paths.chain_depth)
        n_paths = 0 if paths is None else int(paths.n_paths)
        if n_paths <= 0 or chain_depth <= 0:
            polarization_per_bounce.append(vector_zero(n_rx))
            continue

        out_xr, out_xi, out_yr, out_yi, out_zr, out_zi = _empty_reflection_outputs(n_rx)

        chunk_size = Geo.cart_chunk(n_paths, n_rx)

        for path_start in range(0, n_paths, chunk_size):
            chunk_n = min(chunk_size, n_paths - path_start)
            n_pairs = chunk_n * n_rx
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            local_path_idx = pair_idx // n_rx
            path_idx = local_path_idx + wt.UInt32(path_start)
            rx_idx = pair_idx % n_rx
            chunk_path_idx = dr.arange(wt.UInt32, chunk_n) + wt.UInt32(path_start)

            # Visibility uses DrJit BVH queries.
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
                reflection_detail=detail,
                wave=wave,
                tx=tx,
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
                s_mu,
                s_sig,
                s_gn,
            ) = _pack_reflection_chunk_arrays(
                paths=paths,
                scene=scene,
                chunk_path_idx=chunk_path_idx,
                chain_depth=chain_depth,
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
                slot_mu_r=s_mu,
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


def _adjacent_face_for_support(support, prim_idx):
    face0 = support.adjacent_face0
    face1 = support.adjacent_face1
    return dr.select(
        face0 == prim_idx,
        face1,
        dr.select(face1 == prim_idx, face0, wt.Int32(-1)),
    )


def _select_point(mask, yes, no):
    return wt.Point3f(
        dr.select(mask, yes.x, no.x),
        dr.select(mask, yes.y, no.y),
        dr.select(mask, yes.z, no.z),
    )


def _select_vector(mask, yes, no):
    return wt.Vector3f(
        dr.select(mask, yes.x, no.x),
        dr.select(mask, yes.y, no.y),
        dr.select(mask, yes.z, no.z),
    )


def _pack_f_weight_transition_chunk_arrays(
    *,
    scene,
    wave: Wave,
    paths,
    path_idx,
    reflection_detail,
    geometry,
    chain_depth: int,
    default_gain: float,
):
    from witwin.channel.deterministic.reflection.epc import surface_contains

    width = dr.width(geometry["tx_pos"].x)
    tri_data = scene._triangle_runtime()
    tri_surface_data = None
    primary_sides = geometry.get("primary_sides")
    zero_point = dr.zeros(wt.Point3f, width)
    zero_vector = dr.zeros(wt.Vector3f, width)

    support_valid_i = []
    primary_side_i = []
    edge_distance = []
    edge_v0 = []
    edge_v1 = []
    adjacent_valid_i = []
    adjacent_points = []
    adjacent_normals = []
    adjacent_eta = []
    adjacent_mu = []
    adjacent_sigma = []
    adjacent_gain = []
    support_any = dr.full(wt.Bool, False, width)

    for slot in range(chain_depth):
        prim_idx = geometry["prim_indices"][slot]
        hit_p = geometry["hit_points"][slot]
        geom_n = geometry["normals"][slot]
        valid_prim = prim_idx >= wt.Int32(0)
        if primary_sides is None:
            if tri_surface_data is None:
                tri_surface_data = {
                    "group_size": tri_data["surface_group_size"],
                    "group_members": tri_data["surface_group_members"],
                    "max_group_size": int(tri_data["surface_max_group_size"]),
                }
            primary_side = surface_contains(
                hit_p,
                prim_idx,
                tri_data,
                tri_surface_data,
                valid_prim,
                scene=scene,
                plane_normal=geom_n,
            )
        else:
            primary_side = primary_sides[slot]
        support = nearest_surface_boundary_edge(
            scene=scene,
            prim_idx=prim_idx,
            hit_p=hit_p,
            mode=reflection_detail.reflection_transition_mode,
            wavelength=wave.wavelength_scalar,
            boundary_radius_wavelengths=reflection_detail.reflection_f_weight_boundary_radius_wavelengths,
            max_edges_per_slot=reflection_detail.reflection_f_weight_max_edges_per_slot,
        )
        support_any = support_any | support.valid

        primary_n0 = dr.abs(dr.dot(support.n0, geom_n))
        primary_nn = dr.abs(dr.dot(support.n_face_n, geom_n))
        n0_is_primary = primary_n0 >= primary_nn
        adjacent_normal = _select_vector(n0_is_primary, support.n_face_n, support.n0)
        adjacent_prim_idx = _adjacent_face_for_support(support, prim_idx)
        adjacent_ok = support.valid & (~primary_side) & (adjacent_prim_idx >= wt.Int32(0))
        material_inputs = resolve_surface_material(
            scene=scene,
            prim_idx=adjacent_prim_idx,
            default_gain=default_gain,
            valid_mask=adjacent_ok,
        )

        support_valid_i.append(wt.Int32(dr.select(support.valid, wt.Int32(1), wt.Int32(0))))
        primary_side_i.append(wt.Int32(dr.select(primary_side, wt.Int32(1), wt.Int32(0))))
        edge_distance.append(dr.select(support.valid, support.distance, wt.Float(0.0)))
        edge_v0.append(_select_point(support.valid, support.edge_v0, zero_point))
        edge_v1.append(_select_point(support.valid, support.edge_v1, zero_point))
        adjacent_valid_i.append(wt.Int32(dr.select(adjacent_ok, wt.Int32(1), wt.Int32(0))))
        adjacent_points.append(_select_point(adjacent_ok, support.edge_pos, zero_point))
        adjacent_normals.append(_select_vector(adjacent_ok, adjacent_normal, zero_vector))
        adjacent_eta.append(material_inputs["eta_r"])
        adjacent_mu.append(material_inputs["mu_r"])
        adjacent_sigma.append(material_inputs["sigma"])
        adjacent_gain.append(material_inputs["gain"])

    return {
        "support_any": support_any,
        "support_valid": dr.concat(support_valid_i) if support_valid_i else dr.zeros(wt.Int32, 0),
        "primary_side": dr.concat(primary_side_i) if primary_side_i else dr.zeros(wt.Int32, 0),
        "edge_distance": dr.concat(edge_distance) if edge_distance else dr.zeros(wt.Float, 0),
        "edge_v0": (
            wt.Point3f(
                dr.concat([point.x for point in edge_v0]),
                dr.concat([point.y for point in edge_v0]),
                dr.concat([point.z for point in edge_v0]),
            )
            if edge_v0
            else dr.zeros(wt.Point3f, 0)
        ),
        "edge_v1": (
            wt.Point3f(
                dr.concat([point.x for point in edge_v1]),
                dr.concat([point.y for point in edge_v1]),
                dr.concat([point.z for point in edge_v1]),
            )
            if edge_v1
            else dr.zeros(wt.Point3f, 0)
        ),
        "adjacent_valid": dr.concat(adjacent_valid_i) if adjacent_valid_i else dr.zeros(wt.Int32, 0),
        "adjacent_plane_point": (
            wt.Point3f(
                dr.concat([point.x for point in adjacent_points]),
                dr.concat([point.y for point in adjacent_points]),
                dr.concat([point.z for point in adjacent_points]),
            )
            if adjacent_points
            else dr.zeros(wt.Point3f, 0)
        ),
        "adjacent_plane_normal": (
            wt.Vector3f(
                dr.concat([normal.x for normal in adjacent_normals]),
                dr.concat([normal.y for normal in adjacent_normals]),
                dr.concat([normal.z for normal in adjacent_normals]),
            )
            if adjacent_normals
            else dr.zeros(wt.Vector3f, 0)
        ),
        "adjacent_eta_r": dr.concat(adjacent_eta) if adjacent_eta else dr.zeros(wt.Float, 0),
        "adjacent_mu_r": dr.concat(adjacent_mu) if adjacent_mu else dr.zeros(wt.Float, 0),
        "adjacent_sigma": dr.concat(adjacent_sigma) if adjacent_sigma else dr.zeros(wt.Float, 0),
        "adjacent_gain": dr.concat(adjacent_gain) if adjacent_gain else dr.zeros(wt.Float, 0),
    }


def _scatter_f_weight_reference_pairs(
    *,
    vector_coherent,
    paths,
    path_idx,
    rx_idx,
    target_pos,
    scene,
    wave: Wave,
    tx: Tx,
    reflection_detail,
):
    from witwin.channel.deterministic.reflection.epc import chain_to_target as epc_reflection_chain_to_target

    valid, chain_vector, _ = epc_reflection_chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=(),
        reflection_detail=reflection_detail,
        wave=wave,
        tx=tx,
        return_endpoints=True,
    )
    keep_idx = dr.compress(valid)
    if dr.width(keep_idx) == 0:
        return vector_coherent

    rx_idx_keep = dr.gather(type(rx_idx), rx_idx, keep_idx)
    target_pos_keep = gather_point3(target_pos, keep_idx)
    image_source = gather_point3(paths.image_source, path_idx)
    image_source_keep = gather_point3(image_source, keep_idx)
    field_vector = {
        axis: dr.gather(wt.Complex2f, chain_vector[axis], keep_idx)
        for axis in ("x", "y", "z")
    }
    unit_field = Geo.source_field(
        image_source_keep,
        wt.Complex2f(1.0, 0.0),
        target_pos_keep,
        wave,
    )
    field_vector = vector_scale(field_vector, unit_field)
    for axis in ("x", "y", "z"):
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            vector_coherent[axis].real,
            field_vector[axis].real,
            rx_idx_keep,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            vector_coherent[axis].imag,
            field_vector[axis].imag,
            rx_idx_keep,
        )
    return vector_coherent


def _reflection_accumulate_f_weight_native_cuda(
    *,
    rx: Rx,
    tx: Tx,
    scene,
    wave: Wave,
    source_paths_per_bounce: list,
    reflection_detail,
):
    from witwin.channel.deterministic.reflection.epc import chain_to_target as epc_reflection_chain_to_target

    detail = coerce_trace_detail(reflection_detail)
    hard_detail = replace(detail, reflection_transition_mode="hard")
    default_gain = float(detail.reflection_gain)
    omega = float(material_angular_frequency(wave.wavelength_scalar)[0])
    rx_pos = rx.positions
    n_rx = dr.width(rx_pos.x)
    polarization_per_bounce = []

    for paths in source_paths_per_bounce:
        chain_depth = 0 if paths is None else int(paths.chain_depth)
        n_paths = 0 if paths is None else int(paths.n_paths)
        if n_paths <= 0 or chain_depth <= 0:
            polarization_per_bounce.append(vector_zero(n_rx))
            continue

        out_xr, out_xi, out_yr, out_yi, out_zr, out_zi = _empty_reflection_outputs(n_rx)
        chunk_size = Geo.cart_chunk(n_paths, n_rx)
        for path_start in range(0, n_paths, chunk_size):
            chunk_n = min(chunk_size, n_paths - path_start)
            n_pairs = chunk_n * n_rx
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            local_path_idx = pair_idx // n_rx
            path_idx = local_path_idx + wt.UInt32(path_start)
            rx_idx = pair_idx % n_rx
            chunk_path_idx = dr.arange(wt.UInt32, chunk_n) + wt.UInt32(path_start)
            target_pos = wt.Point3f(
                dr.gather(wt.Float, rx_pos.x, rx_idx),
                dr.gather(wt.Float, rx_pos.y, rx_idx),
                dr.gather(wt.Float, rx_pos.z, rx_idx),
            )

            hard_valid, _, geometry = epc_reflection_chain_to_target(
                paths=paths,
                path_idx=path_idx,
                target_pos=target_pos,
                scene=scene,
                target_adjacent_faces=(),
                reflection_detail=hard_detail,
                wave=wave,
                tx=tx,
                return_geometry=True,
                prefer_rayd_epc=False,
            )
            transition = _pack_f_weight_transition_chunk_arrays(
                scene=scene,
                wave=wave,
                paths=paths,
                path_idx=path_idx,
                reflection_detail=detail,
                geometry=geometry,
                chain_depth=chain_depth,
                default_gain=default_gain,
            )
            valid_for_kernel = hard_valid | transition["support_any"]

            (
                image_src,
                ppx,
                ppy,
                ppz,
                pnx,
                pny,
                pnz,
                s_eta,
                s_mu,
                s_sig,
                s_gn,
            ) = _pack_reflection_chunk_arrays(
                paths=paths,
                scene=scene,
                chunk_path_idx=chunk_path_idx,
                chain_depth=chain_depth,
                default_gain=default_gain,
            )
            chunk_xr, chunk_xi, chunk_yr, chunk_yi, chunk_zr, chunk_zi = (
                _accumulate_reflection_f_weight_chunk_arrays(
                    path_idx=local_path_idx,
                    rx_idx=rx_idx,
                    valid_mask=wt.Int32(dr.select(valid_for_kernel, wt.UInt32(1), wt.UInt32(0))),
                    image_source=image_src,
                    slot_plane_point=wt.Point3f(ppx, ppy, ppz),
                    slot_plane_normal=wt.Vector3f(pnx, pny, pnz),
                    slot_eta_r=s_eta,
                    slot_mu_r=s_mu,
                    slot_sigma=s_sig,
                    slot_gain=s_gn,
                    transition_support_valid=transition["support_valid"],
                    transition_primary_side=transition["primary_side"],
                    transition_edge_distance=transition["edge_distance"],
                    transition_edge_v0=transition["edge_v0"],
                    transition_edge_v1=transition["edge_v1"],
                    adjacent_valid=transition["adjacent_valid"],
                    adjacent_plane_point=transition["adjacent_plane_point"],
                    adjacent_plane_normal=transition["adjacent_plane_normal"],
                    adjacent_eta_r=transition["adjacent_eta_r"],
                    adjacent_mu_r=transition["adjacent_mu_r"],
                    adjacent_sigma=transition["adjacent_sigma"],
                    adjacent_gain=transition["adjacent_gain"],
                    rx_pos=rx_pos,
                    tx_pos=tx.position,
                    tx_polarization=tx.polarization_tuple,
                    n_pairs=n_pairs,
                    n_paths=chunk_n,
                    chain_depth=chain_depth,
                    k=wave.k_scalar,
                    omega=omega,
                )
            )
            out_xr = out_xr + chunk_xr
            out_xi = out_xi + chunk_xi
            out_yr = out_yr + chunk_yr
            out_yi = out_yi + chunk_yi
            out_zr = out_zr + chunk_zr
            out_zi = out_zi + chunk_zi

        polarization_per_bounce.append(vector_eval({
            "x": wt.Complex2f(out_xr, out_xi),
            "y": wt.Complex2f(out_yr, out_yi),
            "z": wt.Complex2f(out_zr, out_zi),
        }))

    return polarization_per_bounce


def _reflection_accumulate_f_weight_reference(
    *,
    rx: Rx,
    tx: Tx,
    scene,
    wave: Wave,
    source_paths_per_bounce: list,
    reflection_detail,
):
    rx_pos = rx.positions
    n_rx = dr.width(rx_pos.x)
    polarization_per_bounce = []
    for paths in source_paths_per_bounce:
        chain_depth = 0 if paths is None else int(paths.chain_depth)
        n_paths = 0 if paths is None else int(paths.n_paths)
        if n_paths <= 0 or chain_depth <= 0:
            polarization_per_bounce.append(vector_zero(n_rx))
            continue

        vector_coherent = vector_zero(n_rx)
        chunk_size = Geo.cart_chunk(n_paths, n_rx)
        for path_start in range(0, n_paths, chunk_size):
            chunk_n = min(chunk_size, n_paths - path_start)
            n_pairs = chunk_n * n_rx
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            local_path_idx = pair_idx // n_rx
            path_idx = local_path_idx + wt.UInt32(path_start)
            rx_idx = pair_idx % n_rx
            target_pos = wt.Point3f(
                dr.gather(wt.Float, rx_pos.x, rx_idx),
                dr.gather(wt.Float, rx_pos.y, rx_idx),
                dr.gather(wt.Float, rx_pos.z, rx_idx),
            )
            vector_coherent = _scatter_f_weight_reference_pairs(
                vector_coherent=vector_coherent,
                paths=paths,
                path_idx=path_idx,
                rx_idx=rx_idx,
                target_pos=target_pos,
                scene=scene,
                wave=wave,
                tx=tx,
                reflection_detail=reflection_detail,
            )

        polarization_per_bounce.append(vector_eval(vector_coherent))

    return polarization_per_bounce
