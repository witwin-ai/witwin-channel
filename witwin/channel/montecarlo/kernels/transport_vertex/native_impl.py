from __future__ import annotations

import drjit as dr
from witwin.channel.montecarlo import types as wt
from witwin.channel._native.montecarlo import NativeExtension
_REQUIRED = (
    "monte_carlo_transport_vertex_jvp_into",
    "monte_carlo_transport_vertex_vjp_into",
)


class TransportVertexKernel:
    """Native transport-vertex JVP/VJP kernel wrapper."""

    @staticmethod
    def require():
        return NativeExtension.require_functions(_REQUIRED, context="Monte Carlo transport-vertex kernel")

    @staticmethod
    def available() -> bool:
        return NativeExtension.has_functions(_REQUIRED)

    @staticmethod
    def eval_buffers(buffers) -> None:
        dr.eval(
            buffers.coord_0,
            buffers.coord_1,
            buffers.power,
            buffers.active_mask,
            buffers.vertex_indices,
            buffers.coord_0_coeff_x,
            buffers.coord_0_coeff_y,
            buffers.coord_0_coeff_z,
            buffers.coord_1_coeff_x,
            buffers.coord_1_coeff_y,
            buffers.coord_1_coeff_z,
        )

    @staticmethod
    def launch_jvp_into(*, buffers, vertex_tangent, out_size: int, bounds, cell_size, grid_shape):
        n_samples = int(dr.width(buffers.coord_0))
        if n_samples <= 0:
            return dr.zeros(wt.Float, int(out_size))
        ext = TransportVertexKernel.require()
        TransportVertexKernel.eval_buffers(buffers)
        dr.eval(vertex_tangent.x, vertex_tangent.y, vertex_tangent.z)
        return wt.Float(
            ext.monte_carlo_transport_vertex_jvp_into(
                buffers.coord_0,
                buffers.coord_1,
                buffers.power,
                buffers.active_mask,
                buffers.vertex_indices,
                buffers.coord_0_coeff_x,
                buffers.coord_0_coeff_y,
                buffers.coord_0_coeff_z,
                buffers.coord_1_coeff_x,
                buffers.coord_1_coeff_y,
                buffers.coord_1_coeff_z,
                int(buffers.vertex_slot_count),
                vertex_tangent.x,
                vertex_tangent.y,
                vertex_tangent.z,
                n_samples,
                int(out_size),
                float(bounds[0][0]),
                float(bounds[1][0]),
                float(cell_size[0]),
                float(cell_size[1]),
                int(grid_shape[0]),
                int(grid_shape[1]),
            )
        )

    @staticmethod
    def launch_vjp_into(*, buffers, upstream_component, n_vertices: int, bounds, cell_size, grid_shape):
        n_samples = int(dr.width(buffers.coord_0))
        zero = dr.zeros(wt.Float, int(n_vertices))
        if n_samples <= 0:
            return wt.Point3f(zero, zero, zero)
        ext = TransportVertexKernel.require()
        TransportVertexKernel.eval_buffers(buffers)
        dr.eval(upstream_component)
        grad_x, grad_y, grad_z = ext.monte_carlo_transport_vertex_vjp_into(
            buffers.coord_0,
            buffers.coord_1,
            buffers.power,
            buffers.active_mask,
            buffers.vertex_indices,
            buffers.coord_0_coeff_x,
            buffers.coord_0_coeff_y,
            buffers.coord_0_coeff_z,
            buffers.coord_1_coeff_x,
            buffers.coord_1_coeff_y,
            buffers.coord_1_coeff_z,
            int(buffers.vertex_slot_count),
            upstream_component,
            n_samples,
            int(n_vertices),
            float(bounds[0][0]),
            float(bounds[1][0]),
            float(cell_size[0]),
            float(cell_size[1]),
            int(grid_shape[0]),
            int(grid_shape[1]),
        )
        return wt.Point3f(wt.Float(grad_x), wt.Float(grad_y), wt.Float(grad_z))


__all__ = [
    "TransportVertexKernel",
]
