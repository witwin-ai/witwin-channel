"""Dr.Jit reference implementation of reflected suffix grid accumulation."""

from __future__ import annotations

import math

import drjit as dr
import witwin as wt

from witwin.channel.config import coerce_diffraction_execution
from witwin.channel.trace.diffraction.constants import _cartesian_chunk_size
from witwin.channel.utils.constants import EPS
from witwin.channel.utils.drjit_ops import ArrayInit, complex_abs_sqr
from witwin.channel.utils.plane_axes import point_on_axis_aligned_plane
from witwin.channel.utils.polarization import vector_zero


def suffix_dda_inputs_require_native_ad(*values) -> bool:
    for value in values:
        if value is None:
            continue
        if dr.grad_enabled(value):
            return True
    return False


def _dda_cell_contribute_and_scatter(
    in_bounds,
    cur_coord_0,
    cur_coord_1,
    n_coord_0,
    coord_0_dr,
    coord_1_dr,
    wavelength,
    k,
    grid,
    result_real,
    result_imag,
    result_vector_real_x,
    result_vector_imag_x,
    result_vector_real_y,
    result_vector_imag_y,
    result_vector_real_z,
    result_vector_imag_z,
    seg_origin,
    seg_field_real,
    seg_field_imag,
    seg_vector_real_x,
    seg_vector_imag_x,
    seg_vector_real_y,
    seg_vector_imag_y,
    seg_vector_real_z,
    seg_vector_imag_z,
):
    cell_idx = grid.pos_to_idx(cur_coord_0, cur_coord_1)
    cell_i0 = wt.UInt32(cell_idx % n_coord_0)
    cell_i1 = wt.UInt32(cell_idx // n_coord_0)
    cell_coord_0 = dr.gather(wt.Float, coord_0_dr, cell_i0)
    cell_coord_1 = dr.gather(wt.Float, coord_1_dr, cell_i1)

    cell_pos = point_on_axis_aligned_plane(
        axis=grid.axis,
        position=grid.position,
        tangential_0=cell_coord_0,
        tangential_1=cell_coord_1,
    )
    d_seg = dr.norm(cell_pos - seg_origin) + EPS

    fspl = wt.Float(wavelength / (4.0 * math.pi)) / dr.maximum(d_seg, wt.Float(0.01))
    phase = -wt.Float(k) * d_seg
    cos_p = dr.cos(phase)
    sin_p = dr.sin(phase)

    real_contrib = fspl * (seg_field_real * cos_p - seg_field_imag * sin_p)
    imag_contrib = fspl * (seg_field_real * sin_p + seg_field_imag * cos_p)
    vector_real_x = fspl * (seg_vector_real_x * cos_p - seg_vector_imag_x * sin_p)
    vector_imag_x = fspl * (seg_vector_real_x * sin_p + seg_vector_imag_x * cos_p)
    vector_real_y = fspl * (seg_vector_real_y * cos_p - seg_vector_imag_y * sin_p)
    vector_imag_y = fspl * (seg_vector_real_y * sin_p + seg_vector_imag_y * cos_p)
    vector_real_z = fspl * (seg_vector_real_z * cos_p - seg_vector_imag_z * sin_p)
    vector_imag_z = fspl * (seg_vector_real_z * sin_p + seg_vector_imag_z * cos_p)

    zero = wt.Float(0.0)
    dr.scatter_reduce(dr.ReduceOp.Add, result_real, dr.select(in_bounds, real_contrib, zero), cell_idx, in_bounds)
    dr.scatter_reduce(dr.ReduceOp.Add, result_imag, dr.select(in_bounds, imag_contrib, zero), cell_idx, in_bounds)
    dr.scatter_reduce(dr.ReduceOp.Add, result_vector_real_x, dr.select(in_bounds, vector_real_x, zero), cell_idx, in_bounds)
    dr.scatter_reduce(dr.ReduceOp.Add, result_vector_imag_x, dr.select(in_bounds, vector_imag_x, zero), cell_idx, in_bounds)
    dr.scatter_reduce(dr.ReduceOp.Add, result_vector_real_y, dr.select(in_bounds, vector_real_y, zero), cell_idx, in_bounds)
    dr.scatter_reduce(dr.ReduceOp.Add, result_vector_imag_y, dr.select(in_bounds, vector_imag_y, zero), cell_idx, in_bounds)
    dr.scatter_reduce(dr.ReduceOp.Add, result_vector_real_z, dr.select(in_bounds, vector_real_z, zero), cell_idx, in_bounds)
    dr.scatter_reduce(dr.ReduceOp.Add, result_vector_imag_z, dr.select(in_bounds, vector_imag_z, zero), cell_idx, in_bounds)


@dr.syntax
def _run_suffix_dda_symbolic(
    cur_coord_0,
    cur_coord_1,
    t,
    t_max_0,
    t_max_1,
    loop_active,
    max_steps,
    seg_origin,
    seg_field_real,
    seg_field_imag,
    seg_vector_real_x,
    seg_vector_imag_x,
    seg_vector_real_y,
    seg_vector_imag_y,
    seg_vector_real_z,
    seg_vector_imag_z,
    params,
):
    step_count = wt.UInt32(0)
    while dr.hint(
        loop_active & (step_count < max_steps),
        mode="symbolic",
        max_iterations=max_steps,
        label="diffraction_suffix_dda_symbolic",
        exclude=[params],
    ):
        (
            coord_0_min,
            coord_0_max,
            coord_1_min,
            coord_1_max,
            _cell_size_0,
            _cell_size_1,
            n_coord_0,
            coord_0_dr,
            coord_1_dr,
            wavelength,
            k,
            grid,
            result_real,
            result_imag,
            result_vector_real_x,
            result_vector_imag_x,
            result_vector_real_y,
            result_vector_imag_y,
            result_vector_real_z,
            result_vector_imag_z,
            blocker_dist,
            step_0,
            step_1,
            dt_0,
            dt_1,
        ) = params
        in_bounds = (
            loop_active
            & (cur_coord_0 >= coord_0_min)
            & (cur_coord_0 < coord_0_max)
            & (cur_coord_1 >= coord_1_min)
            & (cur_coord_1 < coord_1_max)
            & (t < blocker_dist)
        )

        _dda_cell_contribute_and_scatter(
            in_bounds, cur_coord_0, cur_coord_1, n_coord_0,
            coord_0_dr, coord_1_dr, wavelength, k, grid,
            result_real, result_imag,
            result_vector_real_x, result_vector_imag_x,
            result_vector_real_y, result_vector_imag_y,
            result_vector_real_z, result_vector_imag_z,
            seg_origin,
            seg_field_real, seg_field_imag,
            seg_vector_real_x, seg_vector_imag_x,
            seg_vector_real_y, seg_vector_imag_y,
            seg_vector_real_z, seg_vector_imag_z,
        )

        move_0 = t_max_0 < t_max_1
        t = dr.select(move_0, t_max_0, t_max_1)
        cur_coord_0 = dr.select(move_0, cur_coord_0 + step_0, cur_coord_0)
        cur_coord_1 = dr.select(~move_0, cur_coord_1 + step_1, cur_coord_1)
        t_max_0 = dr.select(move_0, t_max_0 + dt_0, t_max_0)
        t_max_1 = dr.select(~move_0, t_max_1 + dt_1, t_max_1)
        loop_active = in_bounds
        step_count += 1
    return cur_coord_0, cur_coord_1, t, t_max_0, t_max_1, step_count, loop_active


def accumulate_reflected_segment_fields_chunk(
    grid,
    grid_data,
    seg_origin,
    seg_dir,
    blocker_dist,
    seg_field,
    seg_vector,
    wavelength,
    k,
    active,
    execution,
):
    execution = coerce_diffraction_execution(execution)
    n_rx = grid.n_cells
    if dr.width(seg_dir.x) == 0:
        return ArrayInit.complex_zero(n_rx), vector_zero(n_rx)

    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    cell_size_0, cell_size_1 = grid.cell_size
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)
    axis_0, axis_1 = grid.tangential_axes

    coord_0_dr = grid_data["x_coords"]
    coord_1_dr = grid_data["y_coords"]

    result_real = dr.zeros(wt.Float, n_rx)
    result_imag = dr.zeros(wt.Float, n_rx)
    result_vector_real_x = dr.zeros(wt.Float, n_rx)
    result_vector_imag_x = dr.zeros(wt.Float, n_rx)
    result_vector_real_y = dr.zeros(wt.Float, n_rx)
    result_vector_imag_y = dr.zeros(wt.Float, n_rx)
    result_vector_real_z = dr.zeros(wt.Float, n_rx)
    result_vector_imag_z = dr.zeros(wt.Float, n_rx)

    t = dr.zeros(wt.Float, dr.width(seg_dir.x))
    seg_dir_0 = getattr(seg_dir, axis_0)
    seg_dir_1 = getattr(seg_dir, axis_1)
    cur_coord_0 = getattr(seg_origin, axis_0)
    cur_coord_1 = getattr(seg_origin, axis_1)
    dt_0 = dr.abs(wt.Float(cell_size_0) / dr.maximum(dr.abs(seg_dir_0), wt.Float(EPS)))
    dt_1 = dr.abs(wt.Float(cell_size_1) / dr.maximum(dr.abs(seg_dir_1), wt.Float(EPS)))

    step_0 = dr.select(seg_dir_0 > 0, wt.Float(cell_size_0), wt.Float(-cell_size_0))
    step_1 = dr.select(seg_dir_1 > 0, wt.Float(cell_size_1), wt.Float(-cell_size_1))

    next_0 = dr.select(
        seg_dir_0 > 0,
        (dr.floor((cur_coord_0 - coord_0_min) / cell_size_0) + 1) * cell_size_0 + coord_0_min,
        dr.floor((cur_coord_0 - coord_0_min) / cell_size_0) * cell_size_0 + coord_0_min,
    )
    next_1 = dr.select(
        seg_dir_1 > 0,
        (dr.floor((cur_coord_1 - coord_1_min) / cell_size_1) + 1) * cell_size_1 + coord_1_min,
        dr.floor((cur_coord_1 - coord_1_min) / cell_size_1) * cell_size_1 + coord_1_min,
    )

    t_max_0 = dr.abs((next_0 - cur_coord_0) / dr.maximum(dr.abs(seg_dir_0), wt.Float(EPS)))
    t_max_1 = dr.abs((next_1 - cur_coord_1) / dr.maximum(dr.abs(seg_dir_1), wt.Float(EPS)))
    loop_active = (
        active
        & (cur_coord_0 >= coord_0_min)
        & (cur_coord_0 < coord_0_max)
        & (cur_coord_1 >= coord_1_min)
        & (cur_coord_1 < coord_1_max)
    )

    params_symbolic = (
        wt.Float(coord_0_min), wt.Float(coord_0_max), wt.Float(coord_1_min), wt.Float(coord_1_max),
        wt.Float(cell_size_0), wt.Float(cell_size_1), n_coord_0,
        coord_0_dr, coord_1_dr, wavelength, k,
        grid, result_real, result_imag,
        result_vector_real_x, result_vector_imag_x,
        result_vector_real_y, result_vector_imag_y,
        result_vector_real_z, result_vector_imag_z,
        blocker_dist, step_0, step_1, dt_0, dt_1,
    )
    if execution.suffix_dda != "symbolic":
        raise RuntimeError("Dr.Jit suffix backend only supports suffix_dda='symbolic'.")
    if suffix_dda_inputs_require_native_ad(
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vector["x"],
        seg_vector["y"],
        seg_vector["z"],
    ):
        raise RuntimeError(
            "Dr.Jit suffix backend only supports suffix_dda='symbolic' with non-differentiable suffix inputs. "
            "Use suffix_backend='native' for AD-sensitive suffix workloads."
        )
    _run_suffix_dda_symbolic(
        cur_coord_0,
        cur_coord_1,
        t,
        t_max_0,
        t_max_1,
        loop_active,
        max_steps,
        seg_origin,
        seg_field.real,
        seg_field.imag,
        seg_vector["x"].real,
        seg_vector["x"].imag,
        seg_vector["y"].real,
        seg_vector["y"].imag,
        seg_vector["z"].real,
        seg_vector["z"].imag,
        params_symbolic,
    )
    dr.eval(
        result_real, result_imag,
        result_vector_real_x, result_vector_imag_x,
        result_vector_real_y, result_vector_imag_y,
        result_vector_real_z, result_vector_imag_z,
    )
    return (
        wt.Complex2f(result_real, result_imag),
        {
            "x": wt.Complex2f(result_vector_real_x, result_vector_imag_x),
            "y": wt.Complex2f(result_vector_real_y, result_vector_imag_y),
            "z": wt.Complex2f(result_vector_real_z, result_vector_imag_z),
        },
    )


def accumulate_reflected_segment_fields_batched(
    grid,
    grid_data,
    seg_origin,
    seg_dir,
    blocker_dist,
    seg_field,
    seg_vector,
    state_idx,
    n_states,
    wavelength,
    k,
    active,
    execution,
):
    n_rx = grid.n_cells
    if n_states <= 0:
        return ArrayInit.complex_zero(n_rx), vector_zero(n_rx)

    total_field = ArrayInit.complex_zero(n_rx)
    total_vector = vector_zero(n_rx)
    state_chunk_size = _cartesian_chunk_size(n_states, n_rx)

    for state_start in range(0, n_states, state_chunk_size):
        chunk_n_states = min(state_chunk_size, n_states - state_start)
        chunk_mask = active & (state_idx >= wt.UInt32(state_start)) & (
            state_idx < wt.UInt32(state_start + chunk_n_states)
        )
        ray_keep_idx = dr.compress(chunk_mask)
        if dr.width(ray_keep_idx) == 0:
            continue

        chunk_field, chunk_vector = accumulate_reflected_segment_fields_chunk(
            grid=grid,
            grid_data=grid_data,
            seg_origin=dr.gather(wt.Point3f, seg_origin, ray_keep_idx),
            seg_dir=dr.gather(wt.Vector3f, seg_dir, ray_keep_idx),
            blocker_dist=dr.gather(wt.Float, blocker_dist, ray_keep_idx),
            seg_field=wt.Complex2f(
                dr.gather(wt.Float, seg_field.real, ray_keep_idx),
                dr.gather(wt.Float, seg_field.imag, ray_keep_idx),
            ),
            seg_vector={
                "x": dr.gather(wt.Complex2f, seg_vector["x"], ray_keep_idx),
                "y": dr.gather(wt.Complex2f, seg_vector["y"], ray_keep_idx),
                "z": dr.gather(wt.Complex2f, seg_vector["z"], ray_keep_idx),
            },
            wavelength=wavelength,
            k=k,
            active=dr.full(wt.Bool, True, dr.width(ray_keep_idx)),
            execution=execution,
        )
        total_field = total_field + chunk_field
        total_vector = {
            "x": total_vector["x"] + chunk_vector["x"],
            "y": total_vector["y"] + chunk_vector["y"],
            "z": total_vector["z"] + chunk_vector["z"],
        }

    return total_field, total_vector


__all__ = [
    "accumulate_reflected_segment_fields_batched",
    "accumulate_reflected_segment_fields_chunk",
    "suffix_dda_inputs_require_native_ad",
]
