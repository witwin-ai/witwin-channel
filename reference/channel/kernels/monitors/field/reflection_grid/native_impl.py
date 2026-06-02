"""Native CUDA reflection grid accumulation with Dr.Jit CustomOp AD."""

from __future__ import annotations

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension
from witwin.channel.kernels.monitors.common.receiver_tiles import resolve_receiver_tiles


def _axis_to_int(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[str(axis)]


def _require_native_reflection_grid_kernel():
    ext = _extension()
    required = (
        "reflection_grid_forward_arrays",
        "reflection_grid_jvp_arrays",
        "reflection_grid_backward_arrays",
    )
    missing = [name for name in required if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(
            "Native reflection grid backend requires array launchers "
            + ", ".join(missing)
            + ". Rebuild the witwin.channel native extension."
        )
    return ext


def _zero_complex(width: int):
    return wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))


def _zero_point(width: int):
    return wt.Point3f(
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
    )


def _zero_vector(width: int):
    return wt.Vector3f(
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


def _vector_from_grad(grad_value, width: int):
    if grad_value is None:
        return _zero_vector(width)
    return wt.Vector3f(grad_value.x, grad_value.y, grad_value.z)


def _bool_mask_to_int(active):
    return wt.Int32(dr.select(active, wt.Int32(1), wt.Int32(0)))


def _tri_surface_fields(tri_data):
    if tri_data is None:
        return (None, None, None)
    surface = tri_data.get("surface_group_size")
    members = tri_data.get("surface_group_members")
    if surface is None or members is None:
        return (None, None, None)
    return (surface, members, int(tri_data.get("surface_max_group_size", 0)))


def _empty_float():
    return dr.zeros(wt.Float, 0)


def _empty_int():
    return dr.zeros(wt.Int32, 0)


def _tri_kernel_arrays(tri_data):
    if tri_data is None:
        empty = _empty_float()
        empty_point = wt.Point3f(empty, empty, empty)
        empty_int = _empty_int()
        return empty_point, empty_point, empty_point, empty_int, empty_int, 0
    surface = tri_data.get("surface_group_size")
    members = tri_data.get("surface_group_members")
    return (
        tri_data["v0"],
        tri_data["v1"],
        tri_data["v2"],
        _empty_int() if surface is None else dr.detach(wt.Int32(surface)),
        _empty_int() if members is None else dr.detach(wt.Int32(members)),
        int(tri_data.get("surface_max_group_size", 0)),
    )


def _launch_reflection_grid_forward(
    *,
    grid,
    plane_position,
    grid_data,
    ray_origin,
    ray_dir,
    active,
    blocker_dist,
    prev_refl_p,
    prev_refl_n,
    prev_tx,
    prev_weight,
    prev_pol_x,
    prev_pol_y,
    prev_pol_z,
    prev_prim_idx,
    wavelength,
    k,
    validate_paths,
    tri_data,
):
    ext = _require_native_reflection_grid_kernel()
    n_rays = dr.width(ray_origin.x)
    active_i = dr.detach(_bool_mask_to_int(active))
    prim_idx_i = dr.detach(wt.Int32(prev_prim_idx))
    coord_0_dr = grid_data["x_coords"]
    coord_1_dr = grid_data["y_coords"]
    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    cell_size_0, cell_size_1 = grid.cell_size
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)
    tri_v0, tri_v1, tri_v2, tri_group_size, tri_group_members, max_group_size = _tri_kernel_arrays(tri_data)
    dr.eval(
        coord_0_dr,
        coord_1_dr,
        active_i,
        prim_idx_i,
        ray_origin,
        ray_dir,
        blocker_dist,
        prev_refl_p,
        prev_refl_n,
        prev_tx,
        prev_weight,
        prev_pol_x,
        prev_pol_y,
        prev_pol_z,
    )
    outputs = ext.reflection_grid_forward_arrays(
        _axis_to_int(grid.axis),
        float(plane_position),
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
        ray_origin.x,
        ray_origin.y,
        ray_origin.z,
        ray_dir.x,
        ray_dir.y,
        ray_dir.z,
        blocker_dist,
        prev_refl_p.x,
        prev_refl_p.y,
        prev_refl_p.z,
        prev_refl_n.x,
        prev_refl_n.y,
        prev_refl_n.z,
        prev_tx.x,
        prev_tx.y,
        prev_tx.z,
        prev_weight.real,
        prev_weight.imag,
        prev_pol_x.real,
        prev_pol_x.imag,
        prev_pol_y.real,
        prev_pol_y.imag,
        prev_pol_z.real,
        prev_pol_z.imag,
        prim_idx_i,
        int(bool(validate_paths)),
        int(tri_data is not None),
        tri_v0.x,
        tri_v0.y,
        tri_v0.z,
        tri_v1.x,
        tri_v1.y,
        tri_v1.z,
        tri_v2.x,
        tri_v2.y,
        tri_v2.z,
        tri_group_size,
        tri_group_members,
        int(max_group_size),
        float(wavelength),
        float(k),
        int(n_rays),
    )
    return outputs


def _launch_reflection_grid_jvp(
    *,
    grid,
    plane_position,
    grid_data,
    ray_origin,
    ray_dir,
    active,
    blocker_dist,
    prev_refl_p,
    prev_refl_n,
    prev_tx,
    prev_weight,
    prev_pol_x,
    prev_pol_y,
    prev_pol_z,
    prev_prim_idx,
    t_prev_refl_p,
    t_prev_refl_n,
    t_prev_tx,
    t_prev_weight,
    t_prev_pol_x,
    t_prev_pol_y,
    t_prev_pol_z,
    wavelength,
    k,
    validate_paths,
    tri_data,
):
    ext = _require_native_reflection_grid_kernel()
    n_rays = dr.width(ray_origin.x)
    active_i = dr.detach(_bool_mask_to_int(active))
    prim_idx_i = dr.detach(wt.Int32(prev_prim_idx))
    coord_0_dr = grid_data["x_coords"]
    coord_1_dr = grid_data["y_coords"]
    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    cell_size_0, cell_size_1 = grid.cell_size
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)
    tri_v0, tri_v1, tri_v2, tri_group_size, tri_group_members, max_group_size = _tri_kernel_arrays(tri_data)
    dr.eval(
        coord_0_dr,
        coord_1_dr,
        active_i,
        prim_idx_i,
        ray_origin,
        ray_dir,
        blocker_dist,
        prev_refl_p,
        prev_refl_n,
        prev_tx,
        prev_weight,
        prev_pol_x,
        prev_pol_y,
        prev_pol_z,
        t_prev_refl_p,
        t_prev_refl_n,
        t_prev_tx,
        t_prev_weight,
        t_prev_pol_x,
        t_prev_pol_y,
        t_prev_pol_z,
    )
    outputs = ext.reflection_grid_jvp_arrays(
        _axis_to_int(grid.axis),
        float(plane_position),
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
        ray_origin.x,
        ray_origin.y,
        ray_origin.z,
        ray_dir.x,
        ray_dir.y,
        ray_dir.z,
        blocker_dist,
        prev_refl_p.x,
        prev_refl_p.y,
        prev_refl_p.z,
        prev_refl_n.x,
        prev_refl_n.y,
        prev_refl_n.z,
        prev_tx.x,
        prev_tx.y,
        prev_tx.z,
        prev_weight.real,
        prev_weight.imag,
        prev_pol_x.real,
        prev_pol_x.imag,
        prev_pol_y.real,
        prev_pol_y.imag,
        prev_pol_z.real,
        prev_pol_z.imag,
        prim_idx_i,
        int(bool(validate_paths)),
        int(tri_data is not None),
        tri_v0.x,
        tri_v0.y,
        tri_v0.z,
        tri_v1.x,
        tri_v1.y,
        tri_v1.z,
        tri_v2.x,
        tri_v2.y,
        tri_v2.z,
        tri_group_size,
        tri_group_members,
        int(max_group_size),
        t_prev_refl_p.x,
        t_prev_refl_p.y,
        t_prev_refl_p.z,
        t_prev_refl_n.x,
        t_prev_refl_n.y,
        t_prev_refl_n.z,
        t_prev_tx.x,
        t_prev_tx.y,
        t_prev_tx.z,
        t_prev_weight.real,
        t_prev_weight.imag,
        t_prev_pol_x.real,
        t_prev_pol_x.imag,
        t_prev_pol_y.real,
        t_prev_pol_y.imag,
        t_prev_pol_z.real,
        t_prev_pol_z.imag,
        float(wavelength),
        float(k),
        int(n_rays),
    )
    return outputs


def _launch_reflection_grid_backward(
    *,
    grid,
    plane_position,
    grid_data,
    ray_origin,
    ray_dir,
    active,
    blocker_dist,
    prev_refl_p,
    prev_refl_n,
    prev_tx,
    prev_weight,
    prev_pol_x,
    prev_pol_y,
    prev_pol_z,
    prev_prim_idx,
    wavelength,
    k,
    validate_paths,
    tri_data,
    grad_outputs,
):
    ext = _require_native_reflection_grid_kernel()
    n_rays = dr.width(ray_origin.x)
    active_i = dr.detach(_bool_mask_to_int(active))
    prim_idx_i = dr.detach(wt.Int32(prev_prim_idx))
    coord_0_dr = grid_data["x_coords"]
    coord_1_dr = grid_data["y_coords"]
    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    cell_size_0, cell_size_1 = grid.cell_size
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)
    tri_v0, tri_v1, tri_v2, tri_group_size, tri_group_members, max_group_size = _tri_kernel_arrays(tri_data)
    dr.eval(
        coord_0_dr,
        coord_1_dr,
        active_i,
        prim_idx_i,
        ray_origin,
        ray_dir,
        blocker_dist,
        prev_refl_p,
        prev_refl_n,
        prev_tx,
        prev_weight,
        prev_pol_x,
        prev_pol_y,
        prev_pol_z,
        grad_outputs[0],
        grad_outputs[1],
        grad_outputs[3],
        grad_outputs[4],
        grad_outputs[5],
        grad_outputs[6],
        grad_outputs[7],
        grad_outputs[8],
    )
    grads = ext.reflection_grid_backward_arrays(
        _axis_to_int(grid.axis),
        float(plane_position),
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
        ray_origin.x,
        ray_origin.y,
        ray_origin.z,
        ray_dir.x,
        ray_dir.y,
        ray_dir.z,
        blocker_dist,
        prev_refl_p.x,
        prev_refl_p.y,
        prev_refl_p.z,
        prev_refl_n.x,
        prev_refl_n.y,
        prev_refl_n.z,
        prev_tx.x,
        prev_tx.y,
        prev_tx.z,
        prev_weight.real,
        prev_weight.imag,
        prev_pol_x.real,
        prev_pol_x.imag,
        prev_pol_y.real,
        prev_pol_y.imag,
        prev_pol_z.real,
        prev_pol_z.imag,
        prim_idx_i,
        int(bool(validate_paths)),
        int(tri_data is not None),
        tri_v0.x,
        tri_v0.y,
        tri_v0.z,
        tri_v1.x,
        tri_v1.y,
        tri_v1.z,
        tri_v2.x,
        tri_v2.y,
        tri_v2.z,
        tri_group_size,
        tri_group_members,
        int(max_group_size),
        float(wavelength),
        float(k),
        int(n_rays),
        grad_outputs[0],
        grad_outputs[1],
        grad_outputs[3],
        grad_outputs[4],
        grad_outputs[5],
        grad_outputs[6],
        grad_outputs[7],
        grad_outputs[8],
    )
    dr.eval(*grads)
    grad_prev_refl_p = wt.Point3f(grads[0], grads[1], grads[2])
    grad_prev_refl_n = wt.Vector3f(grads[3], grads[4], grads[5])
    grad_prev_tx = wt.Point3f(grads[6], grads[7], grads[8])
    grad_prev_weight = wt.Complex2f(grads[9], grads[10])
    grad_prev_pol_x = wt.Complex2f(grads[11], grads[12])
    grad_prev_pol_y = wt.Complex2f(grads[13], grads[14])
    grad_prev_pol_z = wt.Complex2f(grads[15], grads[16])
    return (
        grad_prev_refl_p,
        grad_prev_refl_n,
        grad_prev_tx,
        grad_prev_weight,
        grad_prev_pol_x,
        grad_prev_pol_y,
        grad_prev_pol_z,
    )


class _ReflectionGridOp(dr.CustomOp):
    def eval(
        self,
        ray_origin,
        ray_dir,
        blocker_dist,
        prev_refl_p,
        prev_refl_n,
        prev_tx,
        prev_weight,
        prev_pol_x,
        prev_pol_y,
        prev_pol_z,
        *,
        grid,
        plane_position,
        grid_data,
        active,
        prev_prim_idx,
        wavelength,
        k,
        validate_paths,
        tri_data,
    ):
        self.grid = grid
        self.plane_position = float(plane_position)
        self.grid_data = grid_data
        self.active = active
        self.prev_prim_idx = prev_prim_idx
        self.wavelength = float(wavelength)
        self.k = float(k)
        self.validate_paths = bool(validate_paths)
        self.tri_data = tri_data
        self.ray_origin = ray_origin
        self.ray_dir = ray_dir
        self.blocker_dist = blocker_dist
        self.prev_refl_p = prev_refl_p
        self.prev_refl_n = prev_refl_n
        self.prev_tx = prev_tx
        self.prev_weight = prev_weight
        self.prev_pol_x = prev_pol_x
        self.prev_pol_y = prev_pol_y
        self.prev_pol_z = prev_pol_z
        return _launch_reflection_grid_forward(
            grid=grid,
            plane_position=self.plane_position,
            grid_data=grid_data,
            ray_origin=ray_origin,
            ray_dir=ray_dir,
            active=active,
            blocker_dist=blocker_dist,
            prev_refl_p=prev_refl_p,
            prev_refl_n=prev_refl_n,
            prev_tx=prev_tx,
            prev_weight=prev_weight,
            prev_pol_x=prev_pol_x,
            prev_pol_y=prev_pol_y,
            prev_pol_z=prev_pol_z,
            prev_prim_idx=prev_prim_idx,
            wavelength=self.wavelength,
            k=self.k,
            validate_paths=self.validate_paths,
            tri_data=tri_data,
        )

    def forward(self):
        width = dr.width(self.ray_origin.x)
        t_prev_refl_p = _point_from_grad(self.grad_in("prev_refl_p"), width)
        t_prev_refl_n = _vector_from_grad(self.grad_in("prev_refl_n"), width)
        t_prev_tx = _point_from_grad(self.grad_in("prev_tx"), width)
        t_prev_weight = _complex_from_grad(self.grad_in("prev_weight"), width)
        t_prev_pol_x = _complex_from_grad(self.grad_in("prev_pol_x"), width)
        t_prev_pol_y = _complex_from_grad(self.grad_in("prev_pol_y"), width)
        t_prev_pol_z = _complex_from_grad(self.grad_in("prev_pol_z"), width)
        outputs = _launch_reflection_grid_jvp(
            grid=self.grid,
            plane_position=self.plane_position,
            grid_data=self.grid_data,
            ray_origin=self.ray_origin,
            ray_dir=self.ray_dir,
            active=self.active,
            blocker_dist=self.blocker_dist,
            prev_refl_p=self.prev_refl_p,
            prev_refl_n=self.prev_refl_n,
            prev_tx=self.prev_tx,
            prev_weight=self.prev_weight,
            prev_pol_x=self.prev_pol_x,
            prev_pol_y=self.prev_pol_y,
            prev_pol_z=self.prev_pol_z,
            prev_prim_idx=self.prev_prim_idx,
            t_prev_refl_p=t_prev_refl_p,
            t_prev_refl_n=t_prev_refl_n,
            t_prev_tx=t_prev_tx,
            t_prev_weight=t_prev_weight,
            t_prev_pol_x=t_prev_pol_x,
            t_prev_pol_y=t_prev_pol_y,
            t_prev_pol_z=t_prev_pol_z,
            wavelength=self.wavelength,
            k=self.k,
            validate_paths=self.validate_paths,
            tri_data=self.tri_data,
        )
        zero_count = dr.zeros(wt.Float, self.grid.n_cells)
        self.set_grad_out((
            outputs[0],
            outputs[1],
            zero_count,
            outputs[2],
            outputs[3],
            outputs[4],
            outputs[5],
            outputs[6],
            outputs[7],
        ))

    def backward(self):
        width = dr.width(self.ray_origin.x)
        grads = _launch_reflection_grid_backward(
            grid=self.grid,
            plane_position=self.plane_position,
            grid_data=self.grid_data,
            ray_origin=self.ray_origin,
            ray_dir=self.ray_dir,
            active=self.active,
            blocker_dist=self.blocker_dist,
            prev_refl_p=self.prev_refl_p,
            prev_refl_n=self.prev_refl_n,
            prev_tx=self.prev_tx,
            prev_weight=self.prev_weight,
            prev_pol_x=self.prev_pol_x,
            prev_pol_y=self.prev_pol_y,
            prev_pol_z=self.prev_pol_z,
            prev_prim_idx=self.prev_prim_idx,
            wavelength=self.wavelength,
            k=self.k,
            validate_paths=self.validate_paths,
            tri_data=self.tri_data,
            grad_outputs=self.grad_out(),
        )
        self.set_grad_in("ray_origin", _zero_point(width))
        self.set_grad_in("ray_dir", _zero_vector(width))
        self.set_grad_in("blocker_dist", dr.zeros(wt.Float, width))
        self.set_grad_in("prev_refl_p", grads[0])
        self.set_grad_in("prev_refl_n", grads[1])
        self.set_grad_in("prev_tx", grads[2])
        self.set_grad_in("prev_weight", grads[3])
        self.set_grad_in("prev_pol_x", grads[4])
        self.set_grad_in("prev_pol_y", grads[5])
        self.set_grad_in("prev_pol_z", grads[6])


def accumulate_reflection_grid(
    *,
    grid,
    plane_position,
    grid_data,
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
    wavelength,
    k,
    validate_paths,
    tri_data,
    receiver_tiles=None,
):
    receiver_tiles = resolve_receiver_tiles(
        grid=grid,
        plane_position=plane_position,
        grid_data=grid_data,
        receiver_tiles=receiver_tiles,
    )
    if receiver_tiles is not None:
        plane_position = receiver_tiles.plane_position
        grid_data = {
            "x_coords": receiver_tiles.x_coords,
            "y_coords": receiver_tiles.y_coords,
        }
    n_rx = grid.n_cells
    if dr.width(ray_origin.x) == 0:
        return (
            dr.zeros(wt.Float, n_rx),
            dr.zeros(wt.Float, n_rx),
            dr.zeros(wt.Float, n_rx),
            dr.zeros(wt.Float, n_rx),
            dr.zeros(wt.Float, n_rx),
            dr.zeros(wt.Float, n_rx),
            dr.zeros(wt.Float, n_rx),
            dr.zeros(wt.Float, n_rx),
            dr.zeros(wt.Float, n_rx),
        )
    outputs = dr.custom(
        _ReflectionGridOp,
        ray_origin,
        ray_dir,
        blocker_dist,
        prev_refl_p,
        prev_refl_n,
        prev_tx,
        prev_weight,
        prev_polarization["x"],
        prev_polarization["y"],
        prev_polarization["z"],
        grid=grid,
        plane_position=float(plane_position),
        grid_data=grid_data,
        active=active,
        prev_prim_idx=prev_prim_idx,
        wavelength=float(wavelength),
        k=float(k),
        validate_paths=bool(validate_paths),
        tri_data=tri_data,
    )
    return outputs


__all__ = ["accumulate_reflection_grid"]
