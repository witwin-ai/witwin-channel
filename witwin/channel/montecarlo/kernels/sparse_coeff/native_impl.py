from __future__ import annotations

import drjit as dr
from witwin.channel.montecarlo import types as wt
from witwin.channel._native.montecarlo import NativeExtension
_REQUIRED = (
    "monte_carlo_sparse_coeff_jvp_into",
    "monte_carlo_sparse_coeff_vjp_into",
)


class SparseCoeffKernel:
    """Static namespace for sparse coefficient JVP and VJP CUDA kernels."""

    @staticmethod
    def require():
        return NativeExtension.require_functions(_REQUIRED, context="Monte Carlo native AD kernel")

    @staticmethod
    def available() -> bool:
        return NativeExtension.has_functions(_REQUIRED)

    @staticmethod
    def eval_buffers(buffers) -> None:
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

    @staticmethod
    def nonempty_float(value):
        return dr.zeros(wt.Float, 1) if int(dr.width(value)) <= 0 else value

    @staticmethod
    def nonempty_int(value):
        return dr.zeros(wt.Int32, 1) if int(dr.width(value)) <= 0 else value

    @staticmethod
    def launch_jvp_into(
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
        ext = SparseCoeffKernel.require()
        SparseCoeffKernel.eval_buffers(buffers)
        tx_tangent_x = SparseCoeffKernel.nonempty_float(dr.repeat(tx_tangent.x, n_samples))
        tx_tangent_y = SparseCoeffKernel.nonempty_float(dr.repeat(tx_tangent.y, n_samples))
        tx_tangent_z = SparseCoeffKernel.nonempty_float(dr.repeat(tx_tangent.z, n_samples))
        vertex_indices = SparseCoeffKernel.nonempty_int(buffers.vertex_indices)
        vertex_coeff_x = SparseCoeffKernel.nonempty_float(buffers.vertex_coeff_x)
        vertex_coeff_y = SparseCoeffKernel.nonempty_float(buffers.vertex_coeff_y)
        vertex_coeff_z = SparseCoeffKernel.nonempty_float(buffers.vertex_coeff_z)
        material_indices = SparseCoeffKernel.nonempty_int(buffers.material_indices)
        material_coeff_eps = SparseCoeffKernel.nonempty_float(buffers.material_coeff_eps)
        material_coeff_sigma = SparseCoeffKernel.nonempty_float(buffers.material_coeff_sigma)
        vertex_tangent_x = SparseCoeffKernel.nonempty_float(vertex_tangent.x)
        vertex_tangent_y = SparseCoeffKernel.nonempty_float(vertex_tangent.y)
        vertex_tangent_z = SparseCoeffKernel.nonempty_float(vertex_tangent.z)
        material_tangent_eps = SparseCoeffKernel.nonempty_float(material_tangent["eps_r"])
        material_tangent_sigma = SparseCoeffKernel.nonempty_float(material_tangent["sigma_e"])
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
        return wt.Float(
            ext.monte_carlo_sparse_coeff_jvp_into(
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
            )
        )

    @staticmethod
    def launch_vjp_into(
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
        ext = SparseCoeffKernel.require()
        SparseCoeffKernel.eval_buffers(buffers)
        vertex_indices = SparseCoeffKernel.nonempty_int(buffers.vertex_indices)
        vertex_coeff_x = SparseCoeffKernel.nonempty_float(buffers.vertex_coeff_x)
        vertex_coeff_y = SparseCoeffKernel.nonempty_float(buffers.vertex_coeff_y)
        vertex_coeff_z = SparseCoeffKernel.nonempty_float(buffers.vertex_coeff_z)
        material_indices = SparseCoeffKernel.nonempty_int(buffers.material_indices)
        material_coeff_eps = SparseCoeffKernel.nonempty_float(buffers.material_coeff_eps)
        material_coeff_sigma = SparseCoeffKernel.nonempty_float(buffers.material_coeff_sigma)
        upstream_component = SparseCoeffKernel.nonempty_float(upstream_component)
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
    "SparseCoeffKernel",
]
