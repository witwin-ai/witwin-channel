"""Native CUDA reflected suffix grid accumulation with Dr.Jit CustomOp AD."""

from __future__ import annotations

import drjit as dr

from witwin.channel.deterministic import types as wt
from witwin.channel._native.deterministic import NativeExtension
from witwin.channel.deterministic.config import coerce_diffraction_execution
from witwin.channel.deterministic.diffraction.state import Geo
from witwin.channel.core.numerics.arrays import complex_zero, eval_complex
from witwin.channel.core.physics.polarization import vector_eval, vector_zero


def _axis_to_int(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[str(axis)]


_SUFFIX_REQUIRED = (
    "suffix_grid_forward_raw",
    "suffix_grid_jvp_raw",
    "suffix_grid_backward_raw",
)


def _require_native_suffix_kernel():
    return NativeExtension.require_functions(_SUFFIX_REQUIRED, context="Native suffix grid backend raw launchers")


def _zero_complex(width: int):
    return wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))


def _zero_vector(width: int):
    return {
        "x": _zero_complex(width),
        "y": _zero_complex(width),
        "z": _zero_complex(width),
    }


def _zero_point(width: int):
    return wt.Point3f(
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
    )


def _complex_from_grad(grad_value, width: int):
    if grad_value is None:
        return _zero_complex(width)
    return wt.Complex2f(grad_value.real, grad_value.imag)


def _point_from_grad(grad_value, width: int):
    if grad_value is None:
        return _zero_point(width)
    return wt.Point3f(grad_value.x, grad_value.y, grad_value.z)


def _bool_mask_to_int(active):
    return wt.Int32(dr.select(active, wt.Int32(1), wt.Int32(0)))


def _identity_task_segment_idx(n_rays: int):
    return dr.arange(wt.Int32, n_rays)


def suffix_dda_inputs_require_native_ad(*values) -> bool:
    for value in values:
        if value is None:
            continue
        if dr.grad_enabled(value):
            return True
    return False


def _array_has_grad(value) -> bool:
    try:
        return bool(dr.grad_enabled(value))
    except TypeError:
        return False


def _point_has_grad(value) -> bool:
    return any(_array_has_grad(getattr(value, axis)) for axis in ("x", "y", "z"))


def _complex_has_grad(value) -> bool:
    return _array_has_grad(value.real) or _array_has_grad(value.imag)


def _vector_complex_has_grad(value) -> bool:
    return any(_complex_has_grad(value[axis]) for axis in ("x", "y", "z"))


def _launch_suffix_grid_forward(*, grid, grid_data, seg_origin, seg_dir, blocker_dist, seg_field, seg_vec_x, seg_vec_y, seg_vec_z, active, wavelength, k):
    ext = _require_native_suffix_kernel()
    n_rays = dr.width(seg_origin.x)
    active_i = dr.detach(_bool_mask_to_int(active))
    task_segment_idx = _identity_task_segment_idx(n_rays)
    coord_0_dr = grid_data["x_coords"]
    coord_1_dr = grid_data["y_coords"]
    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    cell_size_0, cell_size_1 = grid.cell_size
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)
    dr.eval(
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vec_x,
        seg_vec_y,
        seg_vec_z,
    )
    outputs = ext.suffix_grid_forward_raw(
        _axis_to_int(grid.axis),
        float(grid.position),
        float(coord_0_min),
        float(coord_0_max),
        float(coord_1_min),
        float(coord_1_max),
        float(cell_size_0),
        float(cell_size_1),
        int(n_coord_0),
        int(n_coord_1),
        int(max_steps),
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin.x,
        seg_origin.y,
        seg_origin.z,
        seg_dir.x,
        seg_dir.y,
        seg_dir.z,
        blocker_dist,
        seg_field.real,
        seg_field.imag,
        seg_vec_x.real,
        seg_vec_x.imag,
        seg_vec_y.real,
        seg_vec_y.imag,
        seg_vec_z.real,
        seg_vec_z.imag,
        float(wavelength),
        float(k),
        int(n_rays),
    )
    return outputs


def _launch_suffix_grid_forward_resume_batched(
    *,
    grid,
    grid_data,
    task_segment_idx,
    seg_origin,
    seg_dir,
    blocker_dist,
    seg_field,
    seg_vec_x,
    seg_vec_y,
    seg_vec_z,
    trace_coord_0,
    trace_coord_1,
    trace_t,
    trace_t_max_0,
    trace_t_max_1,
    tile_i0,
    tile_i1,
    tile_extent_0,
    tile_extent_1,
    active,
    wavelength,
    k,
):
    ext = NativeExtension.require_functions(
        ("suffix_grid_forward_resume_batched_raw",),
        context="Native suffix tile replay",
    )
    n_rays = dr.width(task_segment_idx)
    active_i = dr.detach(_bool_mask_to_int(active))
    coord_0_dr = grid_data["x_coords"]
    coord_1_dr = grid_data["y_coords"]
    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    cell_size_0, cell_size_1 = grid.cell_size
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)
    dr.eval(
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vec_x,
        seg_vec_y,
        seg_vec_z,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
    )
    outputs = ext.suffix_grid_forward_resume_batched_raw(
        _axis_to_int(grid.axis),
        float(grid.position),
        float(coord_0_min),
        float(coord_0_max),
        float(coord_1_min),
        float(coord_1_max),
        float(cell_size_0),
        float(cell_size_1),
        int(n_coord_0),
        int(n_coord_1),
        int(max_steps),
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin.x,
        seg_origin.y,
        seg_origin.z,
        seg_dir.x,
        seg_dir.y,
        seg_dir.z,
        blocker_dist,
        seg_field.real,
        seg_field.imag,
        seg_vec_x.real,
        seg_vec_x.imag,
        seg_vec_y.real,
        seg_vec_y.imag,
        seg_vec_z.real,
        seg_vec_z.imag,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
        float(wavelength),
        float(k),
        int(n_rays),
    )
    return outputs


def _launch_suffix_grid_jvp_resume_batched(
    *,
    grid,
    grid_data,
    task_segment_idx,
    seg_origin,
    seg_dir,
    blocker_dist,
    seg_field,
    seg_vec_x,
    seg_vec_y,
    seg_vec_z,
    trace_coord_0,
    trace_coord_1,
    trace_t,
    trace_t_max_0,
    trace_t_max_1,
    tile_i0,
    tile_i1,
    tile_extent_0,
    tile_extent_1,
    t_seg_origin,
    t_seg_field,
    t_seg_vec_x,
    t_seg_vec_y,
    t_seg_vec_z,
    active,
    wavelength,
    k,
):
    ext = NativeExtension.require_functions(
        ("suffix_grid_jvp_resume_batched_raw",),
        context="Native suffix tile replay AD",
    )
    n_rays = dr.width(task_segment_idx)
    active_i = dr.detach(_bool_mask_to_int(active))
    coord_0_dr = grid_data["x_coords"]
    coord_1_dr = grid_data["y_coords"]
    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    cell_size_0, cell_size_1 = grid.cell_size
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)
    dr.eval(
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vec_x,
        seg_vec_y,
        seg_vec_z,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
        t_seg_origin,
        t_seg_field,
        t_seg_vec_x,
        t_seg_vec_y,
        t_seg_vec_z,
    )
    outputs = ext.suffix_grid_jvp_resume_batched_raw(
        _axis_to_int(grid.axis),
        float(grid.position),
        float(coord_0_min),
        float(coord_0_max),
        float(coord_1_min),
        float(coord_1_max),
        float(cell_size_0),
        float(cell_size_1),
        int(n_coord_0),
        int(n_coord_1),
        int(max_steps),
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin.x,
        seg_origin.y,
        seg_origin.z,
        seg_dir.x,
        seg_dir.y,
        seg_dir.z,
        blocker_dist,
        seg_field.real,
        seg_field.imag,
        seg_vec_x.real,
        seg_vec_x.imag,
        seg_vec_y.real,
        seg_vec_y.imag,
        seg_vec_z.real,
        seg_vec_z.imag,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
        t_seg_origin.x,
        t_seg_origin.y,
        t_seg_origin.z,
        t_seg_field.real,
        t_seg_field.imag,
        t_seg_vec_x.real,
        t_seg_vec_x.imag,
        t_seg_vec_y.real,
        t_seg_vec_y.imag,
        t_seg_vec_z.real,
        t_seg_vec_z.imag,
        float(wavelength),
        float(k),
        int(n_rays),
    )
    return outputs


def _launch_suffix_grid_jvp(*, grid, grid_data, seg_origin, seg_dir, blocker_dist, seg_field, seg_vec_x, seg_vec_y, seg_vec_z, t_seg_origin, t_seg_field, t_seg_vec_x, t_seg_vec_y, t_seg_vec_z, active, wavelength, k):
    ext = _require_native_suffix_kernel()
    n_rays = dr.width(seg_origin.x)
    active_i = dr.detach(_bool_mask_to_int(active))
    task_segment_idx = _identity_task_segment_idx(n_rays)
    coord_0_dr = grid_data["x_coords"]
    coord_1_dr = grid_data["y_coords"]
    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    cell_size_0, cell_size_1 = grid.cell_size
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)
    dr.eval(
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vec_x,
        seg_vec_y,
        seg_vec_z,
        t_seg_origin,
        t_seg_field,
        t_seg_vec_x,
        t_seg_vec_y,
        t_seg_vec_z,
    )
    outputs = ext.suffix_grid_jvp_raw(
        _axis_to_int(grid.axis),
        float(grid.position),
        float(coord_0_min),
        float(coord_0_max),
        float(coord_1_min),
        float(coord_1_max),
        float(cell_size_0),
        float(cell_size_1),
        int(n_coord_0),
        int(n_coord_1),
        int(max_steps),
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin.x,
        seg_origin.y,
        seg_origin.z,
        seg_dir.x,
        seg_dir.y,
        seg_dir.z,
        blocker_dist,
        seg_field.real,
        seg_field.imag,
        seg_vec_x.real,
        seg_vec_x.imag,
        seg_vec_y.real,
        seg_vec_y.imag,
        seg_vec_z.real,
        seg_vec_z.imag,
        t_seg_origin.x,
        t_seg_origin.y,
        t_seg_origin.z,
        t_seg_field.real,
        t_seg_field.imag,
        t_seg_vec_x.real,
        t_seg_vec_x.imag,
        t_seg_vec_y.real,
        t_seg_vec_y.imag,
        t_seg_vec_z.real,
        t_seg_vec_z.imag,
        float(wavelength),
        float(k),
        int(n_rays),
    )
    return outputs


def _launch_suffix_grid_backward_resume_batched(
    *,
    grid,
    grid_data,
    task_segment_idx,
    seg_origin,
    seg_dir,
    blocker_dist,
    seg_field,
    seg_vec_x,
    seg_vec_y,
    seg_vec_z,
    trace_coord_0,
    trace_coord_1,
    trace_t,
    trace_t_max_0,
    trace_t_max_1,
    tile_i0,
    tile_i1,
    tile_extent_0,
    tile_extent_1,
    active,
    wavelength,
    k,
    grad_outputs,
):
    ext = NativeExtension.require_functions(
        ("suffix_grid_backward_resume_batched_raw",),
        context="Native suffix tile replay AD",
    )
    n_rays = dr.width(task_segment_idx)
    active_i = dr.detach(_bool_mask_to_int(active))
    coord_0_dr = grid_data["x_coords"]
    coord_1_dr = grid_data["y_coords"]
    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    cell_size_0, cell_size_1 = grid.cell_size
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)
    dr.eval(
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vec_x,
        seg_vec_y,
        seg_vec_z,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
        grad_outputs[0],
        grad_outputs[1],
        grad_outputs[2],
        grad_outputs[3],
        grad_outputs[4],
        grad_outputs[5],
        grad_outputs[6],
        grad_outputs[7],
    )
    grads = ext.suffix_grid_backward_resume_batched_raw(
        _axis_to_int(grid.axis),
        float(grid.position),
        float(coord_0_min),
        float(coord_0_max),
        float(coord_1_min),
        float(coord_1_max),
        float(cell_size_0),
        float(cell_size_1),
        int(n_coord_0),
        int(n_coord_1),
        int(max_steps),
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin.x,
        seg_origin.y,
        seg_origin.z,
        seg_dir.x,
        seg_dir.y,
        seg_dir.z,
        blocker_dist,
        seg_field.real,
        seg_field.imag,
        seg_vec_x.real,
        seg_vec_x.imag,
        seg_vec_y.real,
        seg_vec_y.imag,
        seg_vec_z.real,
        seg_vec_z.imag,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
        float(wavelength),
        float(k),
        int(n_rays),
        grad_outputs[0],
        grad_outputs[1],
        grad_outputs[2],
        grad_outputs[3],
        grad_outputs[4],
        grad_outputs[5],
        grad_outputs[6],
        grad_outputs[7],
    )
    dr.eval(*grads)
    grad_seg_origin = wt.Point3f(grads[0], grads[1], grads[2])
    grad_seg_field = wt.Complex2f(grads[3], grads[4])
    grad_seg_vec_x = wt.Complex2f(grads[5], grads[6])
    grad_seg_vec_y = wt.Complex2f(grads[7], grads[8])
    grad_seg_vec_z = wt.Complex2f(grads[9], grads[10])
    return grad_seg_origin, grad_seg_field, grad_seg_vec_x, grad_seg_vec_y, grad_seg_vec_z


def _launch_suffix_grid_backward(*, grid, grid_data, seg_origin, seg_dir, blocker_dist, seg_field, seg_vec_x, seg_vec_y, seg_vec_z, active, wavelength, k, grad_outputs):
    ext = _require_native_suffix_kernel()
    n_rays = dr.width(seg_origin.x)
    active_i = dr.detach(_bool_mask_to_int(active))
    task_segment_idx = _identity_task_segment_idx(n_rays)
    coord_0_dr = grid_data["x_coords"]
    coord_1_dr = grid_data["y_coords"]
    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    cell_size_0, cell_size_1 = grid.cell_size
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)
    dr.eval(
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vec_x,
        seg_vec_y,
        seg_vec_z,
        grad_outputs[0],
        grad_outputs[1],
        grad_outputs[2],
        grad_outputs[3],
        grad_outputs[4],
        grad_outputs[5],
        grad_outputs[6],
        grad_outputs[7],
    )
    grads = ext.suffix_grid_backward_raw(
        _axis_to_int(grid.axis),
        float(grid.position),
        float(coord_0_min),
        float(coord_0_max),
        float(coord_1_min),
        float(coord_1_max),
        float(cell_size_0),
        float(cell_size_1),
        int(n_coord_0),
        int(n_coord_1),
        int(max_steps),
        coord_0_dr,
        coord_1_dr,
        active_i,
        task_segment_idx,
        seg_origin.x,
        seg_origin.y,
        seg_origin.z,
        seg_dir.x,
        seg_dir.y,
        seg_dir.z,
        blocker_dist,
        seg_field.real,
        seg_field.imag,
        seg_vec_x.real,
        seg_vec_x.imag,
        seg_vec_y.real,
        seg_vec_y.imag,
        seg_vec_z.real,
        seg_vec_z.imag,
        float(wavelength),
        float(k),
        int(n_rays),
        grad_outputs[0],
        grad_outputs[1],
        grad_outputs[2],
        grad_outputs[3],
        grad_outputs[4],
        grad_outputs[5],
        grad_outputs[6],
        grad_outputs[7],
    )
    dr.eval(*grads)
    grad_seg_origin = wt.Point3f(grads[0], grads[1], grads[2])
    grad_seg_field = wt.Complex2f(grads[3], grads[4])
    grad_seg_vec_x = wt.Complex2f(grads[5], grads[6])
    grad_seg_vec_y = wt.Complex2f(grads[7], grads[8])
    grad_seg_vec_z = wt.Complex2f(grads[9], grads[10])
    return grad_seg_origin, grad_seg_field, grad_seg_vec_x, grad_seg_vec_y, grad_seg_vec_z


def _coerce_grid_data(*, grid, grid_data):
    if grid_data is not None:
        return grid_data
    if grid is None:
        raise ValueError("suffix grid accumulation requires either grid_data or grid.")
    return grid.get_coordinates()


def _accumulate_reflected_segment_fields_chunk_primal(
    *,
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
):
    n_rx = int(grid.n_cells)
    if dr.width(seg_dir.x) == 0:
        return complex_zero(n_rx), vector_zero(n_rx)

    outputs = _launch_suffix_grid_forward(
        grid=grid,
        grid_data=grid_data,
        seg_origin=seg_origin,
        seg_dir=seg_dir,
        blocker_dist=blocker_dist,
        seg_field=seg_field,
        seg_vec_x=seg_vector["x"],
        seg_vec_y=seg_vector["y"],
        seg_vec_z=seg_vector["z"],
        active=active,
        wavelength=wavelength,
        k=k,
    )
    field = wt.Complex2f(outputs[0], outputs[1])
    vector = {
        "x": wt.Complex2f(outputs[2], outputs[3]),
        "y": wt.Complex2f(outputs[4], outputs[5]),
        "z": wt.Complex2f(outputs[6], outputs[7]),
    }
    return field, vector


def _accumulate_reflected_segment_fields_chunk_primal_resume(
    *,
    grid,
    grid_data,
    task_segment_idx,
    seg_origin,
    seg_dir,
    blocker_dist,
    seg_field,
    seg_vector,
    trace_coord_0,
    trace_coord_1,
    trace_t,
    trace_t_max_0,
    trace_t_max_1,
    tile_i0,
    tile_i1,
    tile_extent_0,
    tile_extent_1,
    wavelength,
    k,
    active,
):
    n_rx = int(grid.n_cells)
    if dr.width(seg_dir.x) == 0:
        return complex_zero(n_rx), vector_zero(n_rx)

    outputs = _launch_suffix_grid_forward_resume_batched(
        grid=grid,
        grid_data=grid_data,
        task_segment_idx=task_segment_idx,
        seg_origin=seg_origin,
        seg_dir=seg_dir,
        blocker_dist=blocker_dist,
        seg_field=seg_field,
        seg_vec_x=seg_vector["x"],
        seg_vec_y=seg_vector["y"],
        seg_vec_z=seg_vector["z"],
        trace_coord_0=trace_coord_0,
        trace_coord_1=trace_coord_1,
        trace_t=trace_t,
        trace_t_max_0=trace_t_max_0,
        trace_t_max_1=trace_t_max_1,
        tile_i0=tile_i0,
        tile_i1=tile_i1,
        tile_extent_0=tile_extent_0,
        tile_extent_1=tile_extent_1,
        active=active,
        wavelength=wavelength,
        k=k,
    )
    field = wt.Complex2f(outputs[0], outputs[1])
    vector = {
        "x": wt.Complex2f(outputs[2], outputs[3]),
        "y": wt.Complex2f(outputs[4], outputs[5]),
        "z": wt.Complex2f(outputs[6], outputs[7]),
    }
    return field, vector


def _accumulate_reflected_segment_fields_chunk_resume(
    *,
    grid,
    grid_data,
    task_segment_idx,
    seg_origin,
    seg_dir,
    blocker_dist,
    seg_field,
    seg_vector,
    trace_coord_0,
    trace_coord_1,
    trace_t,
    trace_t_max_0,
    trace_t_max_1,
    tile_i0,
    tile_i1,
    tile_extent_0,
    tile_extent_1,
    wavelength,
    k,
    active,
):
    n_rx = int(grid.n_cells)
    if dr.width(seg_dir.x) == 0:
        return complex_zero(n_rx), vector_zero(n_rx)
    if not suffix_dda_inputs_require_native_ad(
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vector["x"],
        seg_vector["y"],
        seg_vector["z"],
    ):
        return _accumulate_reflected_segment_fields_chunk_primal_resume(
            grid=grid,
            grid_data=grid_data,
            task_segment_idx=task_segment_idx,
            seg_origin=seg_origin,
            seg_dir=seg_dir,
            blocker_dist=blocker_dist,
            seg_field=seg_field,
            seg_vector=seg_vector,
            trace_coord_0=trace_coord_0,
            trace_coord_1=trace_coord_1,
            trace_t=trace_t,
            trace_t_max_0=trace_t_max_0,
            trace_t_max_1=trace_t_max_1,
            tile_i0=tile_i0,
            tile_i1=tile_i1,
            tile_extent_0=tile_extent_0,
            tile_extent_1=tile_extent_1,
            wavelength=wavelength,
            k=k,
            active=active,
        )
    outputs = dr.custom(
        _SuffixGridResumeOp,
        task_segment_idx,
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vector["x"],
        seg_vector["y"],
        seg_vector["z"],
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
        grid=grid,
        grid_data=grid_data,
        active=active,
        wavelength=float(wavelength),
        k=float(k),
    )
    field = wt.Complex2f(outputs[0], outputs[1])
    vector = {
        "x": wt.Complex2f(outputs[2], outputs[3]),
        "y": wt.Complex2f(outputs[4], outputs[5]),
        "z": wt.Complex2f(outputs[6], outputs[7]),
    }
    return field, vector


class _SuffixGridOp(dr.CustomOp):
    def eval(
        self,
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vec_x,
        seg_vec_y,
        seg_vec_z,
        *,
        grid,
        grid_data,
        active,
        wavelength,
        k,
    ):
        self.grid = grid
        self.grid_data = grid_data
        self.active = active
        self.wavelength = float(wavelength)
        self.k = float(k)
        self.seg_origin = seg_origin
        self.seg_dir = seg_dir
        self.blocker_dist = blocker_dist
        self.seg_field = seg_field
        self.seg_vec_x = seg_vec_x
        self.seg_vec_y = seg_vec_y
        self.seg_vec_z = seg_vec_z
        return _launch_suffix_grid_forward(
            grid=grid,
            grid_data=grid_data,
            seg_origin=seg_origin,
            seg_dir=seg_dir,
            blocker_dist=blocker_dist,
            seg_field=seg_field,
            seg_vec_x=seg_vec_x,
            seg_vec_y=seg_vec_y,
            seg_vec_z=seg_vec_z,
            active=active,
            wavelength=self.wavelength,
            k=self.k,
        )

    def forward(self):
        width = dr.width(self.seg_origin.x)
        t_seg_origin = _point_from_grad(self.grad_in("seg_origin"), width)
        t_seg_field = _complex_from_grad(self.grad_in("seg_field"), width)
        t_seg_vec_x = _complex_from_grad(self.grad_in("seg_vec_x"), width)
        t_seg_vec_y = _complex_from_grad(self.grad_in("seg_vec_y"), width)
        t_seg_vec_z = _complex_from_grad(self.grad_in("seg_vec_z"), width)
        self.set_grad_out(
            _launch_suffix_grid_jvp(
                grid=self.grid,
                grid_data=self.grid_data,
                seg_origin=self.seg_origin,
                seg_dir=self.seg_dir,
                blocker_dist=self.blocker_dist,
                seg_field=self.seg_field,
                seg_vec_x=self.seg_vec_x,
                seg_vec_y=self.seg_vec_y,
                seg_vec_z=self.seg_vec_z,
                t_seg_origin=t_seg_origin,
                t_seg_field=t_seg_field,
                t_seg_vec_x=t_seg_vec_x,
                t_seg_vec_y=t_seg_vec_y,
                t_seg_vec_z=t_seg_vec_z,
                active=self.active,
                wavelength=self.wavelength,
                k=self.k,
            )
        )

    def backward(self):
        width = dr.width(self.seg_origin.x)
        grad_outputs = self.grad_out()
        grad_seg_origin, grad_seg_field, grad_seg_vec_x, grad_seg_vec_y, grad_seg_vec_z = _launch_suffix_grid_backward(
            grid=self.grid,
            grid_data=self.grid_data,
            seg_origin=self.seg_origin,
            seg_dir=self.seg_dir,
            blocker_dist=self.blocker_dist,
            seg_field=self.seg_field,
            seg_vec_x=self.seg_vec_x,
            seg_vec_y=self.seg_vec_y,
            seg_vec_z=self.seg_vec_z,
            active=self.active,
            wavelength=self.wavelength,
            k=self.k,
            grad_outputs=grad_outputs,
        )
        self.set_grad_in("seg_origin", grad_seg_origin)
        self.set_grad_in(
            "seg_dir",
            wt.Vector3f(
                dr.zeros(wt.Float, width),
                dr.zeros(wt.Float, width),
                dr.zeros(wt.Float, width),
            ),
        )
        self.set_grad_in("blocker_dist", dr.zeros(wt.Float, width))
        self.set_grad_in("seg_field", grad_seg_field)
        self.set_grad_in("seg_vec_x", grad_seg_vec_x)
        self.set_grad_in("seg_vec_y", grad_seg_vec_y)
        self.set_grad_in("seg_vec_z", grad_seg_vec_z)


class _SuffixGridResumeOp(dr.CustomOp):
    def eval(
        self,
        task_segment_idx,
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vec_x,
        seg_vec_y,
        seg_vec_z,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
        *,
        grid,
        grid_data,
        active,
        wavelength,
        k,
    ):
        self.grid = grid
        self.grid_data = grid_data
        self.active = active
        self.wavelength = float(wavelength)
        self.k = float(k)
        self.task_segment_idx = task_segment_idx
        self.seg_origin = seg_origin
        self.seg_dir = seg_dir
        self.blocker_dist = blocker_dist
        self.seg_field = seg_field
        self.seg_vec_x = seg_vec_x
        self.seg_vec_y = seg_vec_y
        self.seg_vec_z = seg_vec_z
        self.trace_coord_0 = trace_coord_0
        self.trace_coord_1 = trace_coord_1
        self.trace_t = trace_t
        self.trace_t_max_0 = trace_t_max_0
        self.trace_t_max_1 = trace_t_max_1
        self.tile_i0 = tile_i0
        self.tile_i1 = tile_i1
        self.tile_extent_0 = tile_extent_0
        self.tile_extent_1 = tile_extent_1
        return _launch_suffix_grid_forward_resume_batched(
            grid=grid,
            grid_data=grid_data,
            task_segment_idx=task_segment_idx,
            seg_origin=seg_origin,
            seg_dir=seg_dir,
            blocker_dist=blocker_dist,
            seg_field=seg_field,
            seg_vec_x=seg_vec_x,
            seg_vec_y=seg_vec_y,
            seg_vec_z=seg_vec_z,
            trace_coord_0=trace_coord_0,
            trace_coord_1=trace_coord_1,
            trace_t=trace_t,
            trace_t_max_0=trace_t_max_0,
            trace_t_max_1=trace_t_max_1,
            tile_i0=tile_i0,
            tile_i1=tile_i1,
            tile_extent_0=tile_extent_0,
            tile_extent_1=tile_extent_1,
            active=active,
            wavelength=self.wavelength,
            k=self.k,
        )

    def forward(self):
        width = dr.width(self.seg_origin.x)
        t_seg_origin = _point_from_grad(self.grad_in("seg_origin"), width)
        t_seg_field = _complex_from_grad(self.grad_in("seg_field"), width)
        t_seg_vec_x = _complex_from_grad(self.grad_in("seg_vec_x"), width)
        t_seg_vec_y = _complex_from_grad(self.grad_in("seg_vec_y"), width)
        t_seg_vec_z = _complex_from_grad(self.grad_in("seg_vec_z"), width)
        self.set_grad_out(
            _launch_suffix_grid_jvp_resume_batched(
                grid=self.grid,
                grid_data=self.grid_data,
                task_segment_idx=self.task_segment_idx,
                seg_origin=self.seg_origin,
                seg_dir=self.seg_dir,
                blocker_dist=self.blocker_dist,
                seg_field=self.seg_field,
                seg_vec_x=self.seg_vec_x,
                seg_vec_y=self.seg_vec_y,
                seg_vec_z=self.seg_vec_z,
                trace_coord_0=self.trace_coord_0,
                trace_coord_1=self.trace_coord_1,
                trace_t=self.trace_t,
                trace_t_max_0=self.trace_t_max_0,
                trace_t_max_1=self.trace_t_max_1,
                tile_i0=self.tile_i0,
                tile_i1=self.tile_i1,
                tile_extent_0=self.tile_extent_0,
                tile_extent_1=self.tile_extent_1,
                t_seg_origin=t_seg_origin,
                t_seg_field=t_seg_field,
                t_seg_vec_x=t_seg_vec_x,
                t_seg_vec_y=t_seg_vec_y,
                t_seg_vec_z=t_seg_vec_z,
                active=self.active,
                wavelength=self.wavelength,
                k=self.k,
            )
        )

    def backward(self):
        width = dr.width(self.seg_origin.x)
        grad_outputs = self.grad_out()
        grad_seg_origin, grad_seg_field, grad_seg_vec_x, grad_seg_vec_y, grad_seg_vec_z = _launch_suffix_grid_backward_resume_batched(
            grid=self.grid,
            grid_data=self.grid_data,
            task_segment_idx=self.task_segment_idx,
            seg_origin=self.seg_origin,
            seg_dir=self.seg_dir,
            blocker_dist=self.blocker_dist,
            seg_field=self.seg_field,
            seg_vec_x=self.seg_vec_x,
            seg_vec_y=self.seg_vec_y,
            seg_vec_z=self.seg_vec_z,
            trace_coord_0=self.trace_coord_0,
            trace_coord_1=self.trace_coord_1,
            trace_t=self.trace_t,
            trace_t_max_0=self.trace_t_max_0,
            trace_t_max_1=self.trace_t_max_1,
            tile_i0=self.tile_i0,
            tile_i1=self.tile_i1,
            tile_extent_0=self.tile_extent_0,
            tile_extent_1=self.tile_extent_1,
            active=self.active,
            wavelength=self.wavelength,
            k=self.k,
            grad_outputs=grad_outputs,
        )
        self.set_grad_in("seg_origin", grad_seg_origin)
        self.set_grad_in(
            "seg_dir",
            wt.Vector3f(
                dr.zeros(wt.Float, width),
                dr.zeros(wt.Float, width),
                dr.zeros(wt.Float, width),
            ),
        )
        self.set_grad_in("blocker_dist", dr.zeros(wt.Float, width))
        self.set_grad_in("seg_field", grad_seg_field)
        self.set_grad_in("seg_vec_x", grad_seg_vec_x)
        self.set_grad_in("seg_vec_y", grad_seg_vec_y)
        self.set_grad_in("seg_vec_z", grad_seg_vec_z)


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
    if execution.suffix_backend != "native":
        raise RuntimeError("suffix_grid.native_impl requires execution.suffix_backend='native'.")
    grid_data = _coerce_grid_data(grid=grid, grid_data=grid_data)
    n_rx = grid.n_cells
    if dr.width(seg_dir.x) == 0:
        return complex_zero(n_rx), vector_zero(n_rx)
    if not suffix_dda_inputs_require_native_ad(
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vector["x"],
        seg_vector["y"],
        seg_vector["z"],
    ):
        return _accumulate_reflected_segment_fields_chunk_primal(
            grid=grid,
            grid_data=grid_data,
            seg_origin=seg_origin,
            seg_dir=seg_dir,
            blocker_dist=blocker_dist,
            seg_field=seg_field,
            seg_vector=seg_vector,
            wavelength=wavelength,
            k=k,
            active=active,
        )
    outputs = dr.custom(
        _SuffixGridOp,
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vector["x"],
        seg_vector["y"],
        seg_vector["z"],
        grid=grid,
        grid_data=grid_data,
        active=active,
        wavelength=float(wavelength),
        k=float(k),
    )
    field = wt.Complex2f(outputs[0], outputs[1])
    vector = {
        "x": wt.Complex2f(outputs[2], outputs[3]),
        "y": wt.Complex2f(outputs[4], outputs[5]),
        "z": wt.Complex2f(outputs[6], outputs[7]),
    }
    return field, vector


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
    execution = coerce_diffraction_execution(execution)
    if execution.suffix_backend != "native":
        raise RuntimeError("suffix_grid.native_impl requires execution.suffix_backend='native'.")
    grid_data = _coerce_grid_data(grid=grid, grid_data=grid_data)
    n_rx = grid.n_cells
    if n_states <= 0:
        return complex_zero(n_rx), vector_zero(n_rx)

    total_buffers = {
        "field_re": dr.zeros(wt.Float, n_rx),
        "field_im": dr.zeros(wt.Float, n_rx),
        "vec_x_re": dr.zeros(wt.Float, n_rx),
        "vec_x_im": dr.zeros(wt.Float, n_rx),
        "vec_y_re": dr.zeros(wt.Float, n_rx),
        "vec_y_im": dr.zeros(wt.Float, n_rx),
        "vec_z_re": dr.zeros(wt.Float, n_rx),
        "vec_z_im": dr.zeros(wt.Float, n_rx),
    }
    state_chunk_size = Geo.cart_chunk(n_states, n_rx)

    for state_start in range(0, n_states, state_chunk_size):
        chunk_n_states = min(state_chunk_size, n_states - state_start)
        chunk_mask = active & (state_idx >= wt.UInt32(state_start)) & (
            state_idx < wt.UInt32(state_start + chunk_n_states)
        )
        ray_keep_idx = dr.compress(chunk_mask)
        if dr.width(ray_keep_idx) == 0:
            continue
        chunk_seg_origin = dr.gather(wt.Point3f, seg_origin, ray_keep_idx)
        chunk_seg_dir = dr.gather(wt.Vector3f, seg_dir, ray_keep_idx)
        chunk_blocker_dist = dr.gather(wt.Float, blocker_dist, ray_keep_idx)
        chunk_seg_field = wt.Complex2f(
            dr.gather(wt.Float, seg_field.real, ray_keep_idx),
            dr.gather(wt.Float, seg_field.imag, ray_keep_idx),
        )
        chunk_seg_vector = {
            "x": dr.gather(wt.Complex2f, seg_vector["x"], ray_keep_idx),
            "y": dr.gather(wt.Complex2f, seg_vector["y"], ray_keep_idx),
            "z": dr.gather(wt.Complex2f, seg_vector["z"], ray_keep_idx),
        }
        chunk_state_idx = dr.gather(wt.UInt32, state_idx, ray_keep_idx)
        chunk_active = dr.full(wt.Bool, True, dr.width(ray_keep_idx))

        chunk_field, chunk_vector = accumulate_reflected_segment_fields_chunk(
            grid=grid,
            grid_data=grid_data,
            seg_origin=chunk_seg_origin,
            seg_dir=chunk_seg_dir,
            blocker_dist=chunk_blocker_dist,
            seg_field=chunk_seg_field,
            seg_vector=chunk_seg_vector,
            wavelength=wavelength,
            k=k,
            active=chunk_active,
            execution=execution,
        )
        total_buffers["field_re"] = total_buffers["field_re"] + chunk_field.real
        total_buffers["field_im"] = total_buffers["field_im"] + chunk_field.imag
        total_buffers["vec_x_re"] = total_buffers["vec_x_re"] + chunk_vector["x"].real
        total_buffers["vec_x_im"] = total_buffers["vec_x_im"] + chunk_vector["x"].imag
        total_buffers["vec_y_re"] = total_buffers["vec_y_re"] + chunk_vector["y"].real
        total_buffers["vec_y_im"] = total_buffers["vec_y_im"] + chunk_vector["y"].imag
        total_buffers["vec_z_re"] = total_buffers["vec_z_re"] + chunk_vector["z"].real
        total_buffers["vec_z_im"] = total_buffers["vec_z_im"] + chunk_vector["z"].imag

    total_field = wt.Complex2f(total_buffers["field_re"], total_buffers["field_im"])
    total_vector = {
        "x": wt.Complex2f(total_buffers["vec_x_re"], total_buffers["vec_x_im"]),
        "y": wt.Complex2f(total_buffers["vec_y_re"], total_buffers["vec_y_im"]),
        "z": wt.Complex2f(total_buffers["vec_z_re"], total_buffers["vec_z_im"]),
    }
    return total_field, total_vector


__all__ = [
    "accumulate_reflected_segment_fields_batched",
    "accumulate_reflected_segment_fields_chunk",
]

