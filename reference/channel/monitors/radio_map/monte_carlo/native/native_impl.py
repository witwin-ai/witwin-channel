from __future__ import annotations

import drjit as dr
import witwin as wt

from witwin.channel._native import _extension, native_extension_available


def _require_monte_carlo_native_kernel():
    ext = _extension()
    required = (
        "monte_carlo_sparse_coeff_jvp_into",
        "monte_carlo_sparse_coeff_vjp_into",
    )
    missing = [name for name in required if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(
            "Monte Carlo native AD kernel requires "
            + ", ".join(missing)
            + ". Rebuild the witwin.channel native extension."
        )
    return ext


def native_monte_carlo_ad_available() -> bool:
    if not native_extension_available():
        return False
    ext = _extension()
    return all(
        hasattr(ext, name)
        for name in (
            "monte_carlo_sparse_coeff_jvp_into",
            "monte_carlo_sparse_coeff_vjp_into",
        )
    )


def _eval_sparse_coeff_buffers(buffers) -> None:
    dr.eval(
        buffers.cell_idx,
        buffers.tx_coeff_x,
        buffers.tx_coeff_y,
        buffers.tx_coeff_z,
        buffers.vertex_indices,
        buffers.vertex_coeff_x,
        buffers.vertex_coeff_y,
        buffers.vertex_coeff_z,
        buffers.material_indices,
        buffers.material_coeff_eps,
        buffers.material_coeff_sigma,
    )


def _nonempty_float(value):
    return dr.zeros(wt.Float, 1) if int(dr.width(value)) <= 0 else value


def _nonempty_int(value):
    return dr.zeros(wt.Int32, 1) if int(dr.width(value)) <= 0 else value


def launch_sparse_coeff_jvp_into(
    *,
    buffers,
    tx_tangent,
    vertex_tangent,
    material_tangent,
    out_size: int,
):
    n_samples = int(dr.width(buffers.cell_idx))
    if n_samples <= 0:
        return dr.zeros(wt.Float, int(out_size))
    ext = _require_monte_carlo_native_kernel()
    _eval_sparse_coeff_buffers(buffers)
    tx_tangent_x = _nonempty_float(dr.repeat(tx_tangent.x, n_samples))
    tx_tangent_y = _nonempty_float(dr.repeat(tx_tangent.y, n_samples))
    tx_tangent_z = _nonempty_float(dr.repeat(tx_tangent.z, n_samples))
    vertex_indices = _nonempty_int(buffers.vertex_indices)
    vertex_coeff_x = _nonempty_float(buffers.vertex_coeff_x)
    vertex_coeff_y = _nonempty_float(buffers.vertex_coeff_y)
    vertex_coeff_z = _nonempty_float(buffers.vertex_coeff_z)
    material_indices = _nonempty_int(buffers.material_indices)
    material_coeff_eps = _nonempty_float(buffers.material_coeff_eps)
    material_coeff_sigma = _nonempty_float(buffers.material_coeff_sigma)
    vertex_tangent_x = _nonempty_float(vertex_tangent.x)
    vertex_tangent_y = _nonempty_float(vertex_tangent.y)
    vertex_tangent_z = _nonempty_float(vertex_tangent.z)
    material_tangent_eps = _nonempty_float(material_tangent["eps_r"])
    material_tangent_sigma = _nonempty_float(material_tangent["sigma_e"])
    dr.eval(
        tx_tangent_x,
        tx_tangent_y,
        tx_tangent_z,
        vertex_indices,
        vertex_coeff_x,
        vertex_coeff_y,
        vertex_coeff_z,
        material_indices,
        material_coeff_eps,
        material_coeff_sigma,
        vertex_tangent_x,
        vertex_tangent_y,
        vertex_tangent_z,
        material_tangent_eps,
        material_tangent_sigma,
    )
    return wt.Float(ext.monte_carlo_sparse_coeff_jvp_into(
        buffers.cell_idx,
        buffers.tx_coeff_x,
        buffers.tx_coeff_y,
        buffers.tx_coeff_z,
        vertex_indices,
        vertex_coeff_x,
        vertex_coeff_y,
        vertex_coeff_z,
        int(buffers.vertex_slot_count),
        material_indices,
        material_coeff_eps,
        material_coeff_sigma,
        int(buffers.material_slot_count),
        tx_tangent_x,
        tx_tangent_y,
        tx_tangent_z,
        vertex_tangent_x,
        vertex_tangent_y,
        vertex_tangent_z,
        material_tangent_eps,
        material_tangent_sigma,
        n_samples,
        int(out_size),
    ))


def launch_sparse_coeff_vjp_into(
    *,
    buffers,
    upstream_component,
    n_vertices: int,
    n_materials: int,
):
    if int(dr.width(buffers.cell_idx)) <= 0:
        return (
            dr.zeros(wt.Float, 0),
            dr.zeros(wt.Float, 0),
            dr.zeros(wt.Float, 0),
            dr.zeros(wt.Float, int(n_vertices)),
            dr.zeros(wt.Float, int(n_vertices)),
            dr.zeros(wt.Float, int(n_vertices)),
            dr.zeros(wt.Float, int(n_materials)),
            dr.zeros(wt.Float, int(n_materials)),
        )
    ext = _require_monte_carlo_native_kernel()
    _eval_sparse_coeff_buffers(buffers)
    vertex_indices = _nonempty_int(buffers.vertex_indices)
    vertex_coeff_x = _nonempty_float(buffers.vertex_coeff_x)
    vertex_coeff_y = _nonempty_float(buffers.vertex_coeff_y)
    vertex_coeff_z = _nonempty_float(buffers.vertex_coeff_z)
    material_indices = _nonempty_int(buffers.material_indices)
    material_coeff_eps = _nonempty_float(buffers.material_coeff_eps)
    material_coeff_sigma = _nonempty_float(buffers.material_coeff_sigma)
    upstream_component = _nonempty_float(upstream_component)
    dr.eval(
        upstream_component,
        vertex_indices,
        vertex_coeff_x,
        vertex_coeff_y,
        vertex_coeff_z,
        material_indices,
        material_coeff_eps,
        material_coeff_sigma,
    )
    return ext.monte_carlo_sparse_coeff_vjp_into(
        buffers.cell_idx,
        buffers.tx_coeff_x,
        buffers.tx_coeff_y,
        buffers.tx_coeff_z,
        vertex_indices,
        vertex_coeff_x,
        vertex_coeff_y,
        vertex_coeff_z,
        int(buffers.vertex_slot_count),
        material_indices,
        material_coeff_eps,
        material_coeff_sigma,
        int(buffers.material_slot_count),
        upstream_component,
        int(dr.width(buffers.cell_idx)),
        int(n_vertices),
        int(n_materials),
    )


__all__ = [
    "launch_sparse_coeff_jvp_into",
    "launch_sparse_coeff_vjp_into",
    "native_monte_carlo_ad_available",
]
