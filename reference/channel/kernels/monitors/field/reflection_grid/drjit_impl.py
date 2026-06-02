"""Dr.Jit reference implementation of reflection grid accumulation."""

import math

import drjit as dr
import witwin as wt

from witwin.channel.utils.constants import EPS
from witwin.channel.utils.geometry import surface_contains_point
from witwin.channel.utils.plane_axes import point_on_axis_aligned_plane


@dr.syntax
def _run_dda_symbolic(
    cur_x,
    cur_y,
    t,
    t_max_x,
    t_max_y,
    loop_active,
    max_steps,
    blocker_dist,
    step_x,
    step_y,
    dt_x,
    dt_y,
    prev_refl_p,
    prev_refl_n,
    prev_tx,
    prev_weight,
    prev_pol_x,
    prev_pol_y,
    prev_pol_z,
    params,
):
    step_count = wt.UInt32(0)
    while dr.hint(
        loop_active & (step_count < max_steps),
        mode="symbolic",
        max_iterations=max_steps,
        label="reflection_dda_symbolic",
        exclude=[params],
    ):
        (
            x_min,
            x_max,
            y_min,
            y_max,
            _cell_size_x,
            _cell_size_y,
            nx,
            x_coords_dr,
            y_coords_dr,
            rx_z,
            wavelength,
            k,
            validate_paths,
            has_mesh_data,
            tri_v0,
            tri_v1,
            tri_v2,
            tri_surface_data,
            grid,
            result_real,
            result_imag,
            result_count,
            result_pol_real_x,
            result_pol_imag_x,
            result_pol_real_y,
            result_pol_imag_y,
            result_pol_real_z,
            result_pol_imag_z,
            bounce_idx,
            prev_prim_idx,
        ) = params
        in_bounds = (
            loop_active
            & (cur_x >= x_min)
            & (cur_x < x_max)
            & (cur_y >= y_min)
            & (cur_y < y_max)
            & (t < blocker_dist)
        )

        cell_idx = grid.pos_to_idx(cur_x, cur_y)
        cell_ix = wt.UInt32(cell_idx % nx)
        cell_iy = wt.UInt32(cell_idx // nx)
        cell_x = dr.gather(wt.Float, x_coords_dr, cell_ix)
        cell_y = dr.gather(wt.Float, y_coords_dr, cell_iy)

        d_to_plane = dr.dot(prev_tx - prev_refl_p, prev_refl_n)
        mirror = prev_tx - 2 * d_to_plane * prev_refl_n

        cell_pos = wt.Point3f(cell_x, cell_y, wt.Float(rx_z))
        d_mirror = dr.norm(cell_pos - mirror)

        valid_path = in_bounds
        if validate_paths and has_mesh_data:
            dir_to_rx = cell_pos - mirror
            denom = dr.dot(dir_to_rx, prev_refl_n)
            t_intersect = dr.dot(prev_refl_p - mirror, prev_refl_n) / (denom + EPS)
            int_p = mirror + t_intersect * dir_to_rx
            t_valid = (t_intersect > 0) & (t_intersect < 1)
            denom_valid = dr.abs(denom) > EPS
            valid_path = (
                in_bounds
                & surface_contains_point(
                    int_p,
                    wt.Int32(prev_prim_idx),
                    tri_v0,
                    tri_v1,
                    tri_v2,
                    tri_surface_data,
                )
                & t_valid
                & denom_valid
            )

        fspl = wt.Float(wavelength) / (4.0 * math.pi * dr.maximum(d_mirror, wt.Float(0.01)))
        phase = -wt.Float(k) * d_mirror
        phase_factor = wt.Complex2f(dr.cos(phase), dr.sin(phase))
        field_contrib = prev_weight * fspl * phase_factor
        pol_x = prev_pol_x * fspl * phase_factor
        pol_y = prev_pol_y * fspl * phase_factor
        pol_z = prev_pol_z * fspl * phase_factor

        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_real[bounce_idx],
            dr.select(valid_path, field_contrib.real, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_imag[bounce_idx],
            dr.select(valid_path, field_contrib.imag, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_count[bounce_idx],
            dr.select(valid_path, wt.Float(1.0), wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_real_x[bounce_idx],
            dr.select(valid_path, pol_x.real, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_imag_x[bounce_idx],
            dr.select(valid_path, pol_x.imag, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_real_y[bounce_idx],
            dr.select(valid_path, pol_y.real, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_imag_y[bounce_idx],
            dr.select(valid_path, pol_y.imag, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_real_z[bounce_idx],
            dr.select(valid_path, pol_z.real, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_imag_z[bounce_idx],
            dr.select(valid_path, pol_z.imag, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )

        move_x = t_max_x < t_max_y
        t = dr.select(move_x, t_max_x, t_max_y)
        cur_x = dr.select(move_x, cur_x + step_x, cur_x)
        cur_y = dr.select(~move_x, cur_y + step_y, cur_y)
        t_max_x = dr.select(move_x, t_max_x + dt_x, t_max_x)
        t_max_y = dr.select(~move_x, t_max_y + dt_y, t_max_y)
        loop_active = in_bounds
        step_count += 1
    return cur_x, cur_y, t, t_max_x, t_max_y, step_count, loop_active


def run_dda_traversal(
    *,
    grid,
    ray_origin,
    ray_dir,
    active,
    blocker_dist,
    prev_refl_p,
    prev_refl_n,
    prev_tx,
    prev_weight,
    prev_polarization,
    prev_prim_idx,
    x_min,
    x_max,
    y_min,
    y_max,
    cell_size_x,
    cell_size_y,
    nx,
    max_steps,
    x_coords_dr,
    y_coords_dr,
    rx_z,
    wavelength,
    k,
    validate_paths,
    has_mesh_data,
    tri_v0,
    tri_v1,
    tri_v2,
    tri_surface_data,
    result_real,
    result_imag,
    result_count,
    result_pol_real_x,
    result_pol_imag_x,
    result_pol_real_y,
    result_pol_imag_y,
    result_pol_real_z,
    result_pol_imag_z,
    bounce_idx,
):
    """Run one bounce worth of z-plane DDA accumulation."""
    dr.eval(blocker_dist)

    t = dr.zeros(wt.Float, dr.width(ray_origin.x))
    dt_x = dr.abs(wt.Float(cell_size_x) / dr.maximum(dr.abs(ray_dir.x), wt.Float(EPS)))
    dt_y = dr.abs(wt.Float(cell_size_y) / dr.maximum(dr.abs(ray_dir.y), wt.Float(EPS)))

    cur_x = ray_origin.x
    cur_y = ray_origin.y

    step_x = dr.select(ray_dir.x > 0, wt.Float(cell_size_x), wt.Float(-cell_size_x))
    step_y = dr.select(ray_dir.y > 0, wt.Float(cell_size_y), wt.Float(-cell_size_y))

    next_x = dr.select(
        ray_dir.x > 0,
        (dr.floor((cur_x - x_min) / cell_size_x) + 1) * cell_size_x + x_min,
        dr.floor((cur_x - x_min) / cell_size_x) * cell_size_x + x_min,
    )
    next_y = dr.select(
        ray_dir.y > 0,
        (dr.floor((cur_y - y_min) / cell_size_y) + 1) * cell_size_y + y_min,
        dr.floor((cur_y - y_min) / cell_size_y) * cell_size_y + y_min,
    )

    t_max_x = dr.abs((next_x - cur_x) / dr.maximum(dr.abs(ray_dir.x), wt.Float(EPS)))
    t_max_y = dr.abs((next_y - cur_y) / dr.maximum(dr.abs(ray_dir.y), wt.Float(EPS)))

    loop_active = (
        active
        & (cur_x >= x_min)
        & (cur_x < x_max)
        & (cur_y >= y_min)
        & (cur_y < y_max)
        & (t < blocker_dist)
    )

    params_symbolic = (
        wt.Float(x_min),
        wt.Float(x_max),
        wt.Float(y_min),
        wt.Float(y_max),
        wt.Float(cell_size_x),
        wt.Float(cell_size_y),
        nx,
        x_coords_dr,
        y_coords_dr,
        rx_z,
        wavelength,
        k,
        validate_paths,
        has_mesh_data,
        tri_v0,
        tri_v1,
        tri_v2,
        tri_surface_data,
        grid,
        result_real,
        result_imag,
        result_count,
        result_pol_real_x,
        result_pol_imag_x,
        result_pol_real_y,
        result_pol_imag_y,
        result_pol_real_z,
        result_pol_imag_z,
        bounce_idx,
        prev_prim_idx,
    )
    _run_dda_symbolic(
        cur_x,
        cur_y,
        t,
        t_max_x,
        t_max_y,
        loop_active,
        max_steps,
        blocker_dist,
        step_x,
        step_y,
        dt_x,
        dt_y,
        prev_refl_p,
        prev_refl_n,
        prev_tx,
        prev_weight,
        prev_polarization["x"],
        prev_polarization["y"],
        prev_polarization["z"],
        params_symbolic,
    )


_SCATTER_CHUNK_RAY_THRESHOLD = 1_000_000
_SCATTER_CHUNK_SIZE = 262_144


def extract_plane_components(value, axis: str):
    if axis == "x":
        return value.y, value.z, value.x
    if axis == "y":
        return value.x, value.z, value.y
    return value.x, value.y, value.z


def prepare_plane_intersections(
    *,
    grid,
    ray_origin,
    ray_dir,
    active,
    blocker_dist,
    plane_position,
):
    tangential_0, tangential_1, normal_origin = extract_plane_components(ray_origin, grid.axis)
    dir_tangential_0, dir_tangential_1, dir_normal = extract_plane_components(ray_dir, grid.axis)
    bound_0, bound_1 = grid.bounds

    distance_to_plane = wt.Float(plane_position) - normal_origin
    parallel = active & (dr.abs(dir_normal) <= wt.Float(EPS))
    points_away = active & ~parallel & (distance_to_plane * dir_normal < 0)
    candidate = active & ~parallel & ~points_away
    safe_dir_normal = dr.select(candidate, dir_normal, wt.Float(1.0))
    t_plane = distance_to_plane / safe_dir_normal
    hit_tangential_0 = tangential_0 + t_plane * dir_tangential_0
    hit_tangential_1 = tangential_1 + t_plane * dir_tangential_1
    before_blocker = t_plane < blocker_dist
    in_bounds = (
        (hit_tangential_0 >= bound_0[0])
        & (hit_tangential_0 < bound_0[1])
        & (hit_tangential_1 >= bound_1[0])
        & (hit_tangential_1 < bound_1[1])
    )
    valid = candidate & (t_plane >= 0) & before_blocker & in_bounds

    return {
        "candidate": candidate,
        "t_plane": t_plane,
        "hit_tangential_0": hit_tangential_0,
        "hit_tangential_1": hit_tangential_1,
        "parallel": parallel,
        "points_away": points_away,
        "valid": valid,
    }


def intersect_and_scatter(
    *,
    grid,
    plane_position,
    intersections,
    prev_refl_p,
    prev_refl_n,
    prev_tx,
    prev_weight,
    prev_polarization,
    prev_prim_idx,
    nx,
    x_coords_dr,
    y_coords_dr,
    wavelength,
    k,
    validate_paths,
    has_mesh_data,
    tri_v0,
    tri_v1,
    tri_v2,
    tri_surface_data,
    result_real,
    result_imag,
    result_count,
    result_pol_real_x,
    result_pol_imag_x,
    result_pol_real_y,
    result_pol_imag_y,
    result_pol_real_z,
    result_pol_imag_z,
    bounce_idx,
):
    valid = intersections["valid"]
    if not dr.any(valid):
        return {
            "compressed_candidate_rays": 0,
            "scatter_chunks_used": 0,
            "chunked_scatter": False,
            "scatter_chunk_size": 0,
        }

    candidate_idx = dr.compress(valid)
    candidate_count = int(dr.width(candidate_idx))
    if candidate_count == 0:
        return {
            "compressed_candidate_rays": 0,
            "scatter_chunks_used": 0,
            "chunked_scatter": False,
            "scatter_chunk_size": 0,
        }

    chunk_size = candidate_count
    if candidate_count > _SCATTER_CHUNK_RAY_THRESHOLD:
        chunk_size = min(_SCATTER_CHUNK_SIZE, candidate_count)
    chunks_used = 0

    for chunk_start in range(0, candidate_count, chunk_size):
        chunk_count = min(chunk_size, candidate_count - chunk_start)
        if chunk_start == 0 and chunk_count == candidate_count:
            chunk_idx = candidate_idx
        else:
            selector = dr.arange(wt.UInt32, chunk_count) + wt.UInt32(chunk_start)
            chunk_idx = dr.gather(wt.UInt32, candidate_idx, selector)
        chunks_used += 1

        hit_tangential_0 = dr.gather(wt.Float, intersections["hit_tangential_0"], chunk_idx)
        hit_tangential_1 = dr.gather(wt.Float, intersections["hit_tangential_1"], chunk_idx)
        chunk_prev_refl_p = dr.gather(wt.Point3f, prev_refl_p, chunk_idx)
        chunk_prev_refl_n = dr.gather(wt.Vector3f, prev_refl_n, chunk_idx)
        chunk_prev_tx = dr.gather(wt.Point3f, prev_tx, chunk_idx)
        chunk_prev_weight = dr.gather(wt.Complex2f, prev_weight, chunk_idx)
        chunk_prev_prim_idx = dr.gather(wt.UInt32, prev_prim_idx, chunk_idx)
        chunk_prev_polarization = {
            "x": dr.gather(wt.Complex2f, prev_polarization["x"], chunk_idx),
            "y": dr.gather(wt.Complex2f, prev_polarization["y"], chunk_idx),
            "z": dr.gather(wt.Complex2f, prev_polarization["z"], chunk_idx),
        }

        cell_idx = grid.pos_to_idx(hit_tangential_0, hit_tangential_1)
        cell_ix = wt.UInt32(cell_idx % nx)
        cell_iy = wt.UInt32(cell_idx // nx)
        cell_tangential_0 = dr.gather(wt.Float, x_coords_dr, cell_ix)
        cell_tangential_1 = dr.gather(wt.Float, y_coords_dr, cell_iy)

        d_to_plane = dr.dot(chunk_prev_tx - chunk_prev_refl_p, chunk_prev_refl_n)
        mirror = chunk_prev_tx - 2 * d_to_plane * chunk_prev_refl_n
        cell_pos = point_on_axis_aligned_plane(
            axis=grid.axis,
            position=plane_position,
            tangential_0=cell_tangential_0,
            tangential_1=cell_tangential_1,
        )
        d_mirror = dr.norm(cell_pos - mirror)

        valid_path = dr.full(wt.Bool, True, dr.width(chunk_idx))
        if validate_paths and has_mesh_data:
            dir_to_rx = cell_pos - mirror
            denom = dr.dot(dir_to_rx, chunk_prev_refl_n)
            t_intersect = dr.dot(chunk_prev_refl_p - mirror, chunk_prev_refl_n) / (denom + EPS)
            int_p = mirror + t_intersect * dir_to_rx
            t_valid = (t_intersect > 0) & (t_intersect < 1)
            denom_valid = dr.abs(denom) > EPS
            valid_path = (
                surface_contains_point(
                    int_p,
                    wt.Int32(chunk_prev_prim_idx),
                    tri_v0,
                    tri_v1,
                    tri_v2,
                    tri_surface_data,
                )
                & t_valid
                & denom_valid
            )

        fspl = wt.Float(wavelength) / (4.0 * math.pi * dr.maximum(d_mirror, wt.Float(0.01)))
        phase = -wt.Float(k) * d_mirror
        phase_factor = wt.Complex2f(dr.cos(phase), dr.sin(phase))
        field_contrib = chunk_prev_weight * fspl * phase_factor
        pol_x = chunk_prev_polarization["x"] * fspl * phase_factor
        pol_y = chunk_prev_polarization["y"] * fspl * phase_factor
        pol_z = chunk_prev_polarization["z"] * fspl * phase_factor

        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_real[bounce_idx],
            dr.select(valid_path, field_contrib.real, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_imag[bounce_idx],
            dr.select(valid_path, field_contrib.imag, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_count[bounce_idx],
            dr.select(valid_path, wt.Float(1.0), wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_real_x[bounce_idx],
            dr.select(valid_path, pol_x.real, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_imag_x[bounce_idx],
            dr.select(valid_path, pol_x.imag, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_real_y[bounce_idx],
            dr.select(valid_path, pol_y.real, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_imag_y[bounce_idx],
            dr.select(valid_path, pol_y.imag, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_real_z[bounce_idx],
            dr.select(valid_path, pol_z.real, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            result_pol_imag_z[bounce_idx],
            dr.select(valid_path, pol_z.imag, wt.Float(0.0)),
            cell_idx,
            valid_path,
        )

    return {
        "compressed_candidate_rays": candidate_count,
        "scatter_chunks_used": chunks_used,
        "chunked_scatter": chunks_used > 1,
        "scatter_chunk_size": chunk_size,
    }


__all__ = [
    "extract_plane_components",
    "intersect_and_scatter",
    "prepare_plane_intersections",
    "run_dda_traversal",
]
