from __future__ import annotations

import drjit as dr
from witwin.channel.montecarlo import types as wt
from witwin.channel._native.montecarlo import NativeExtension
class _TransportGridOp(dr.CustomOp):
    def eval(
        self,
        coord_0,
        coord_1,
        power,
        active_mask,
        *,
        n_samples: int,
        bounds,
        cell_size,
        grid_shape,
    ):
        self.coord_0 = coord_0
        self.coord_1 = coord_1
        self.power = power
        self.active_mask = active_mask
        self.n_samples = int(n_samples)
        self.bounds = bounds
        self.cell_size = cell_size
        self.grid_shape = grid_shape
        return TransportGridKernel._forward(
            coord_0=coord_0,
            coord_1=coord_1,
            power=power,
            active_mask=active_mask,
            n_samples=self.n_samples,
            bounds=bounds,
            cell_size=cell_size,
            grid_shape=grid_shape,
        )

    def forward(self):
        coord_width = int(dr.width(self.coord_0))
        zero_coord = dr.zeros(wt.Float, coord_width)
        t_coord_0 = zero_coord if self.grad_in("coord_0") is None else wt.Float(self.grad_in("coord_0"))
        t_coord_1 = zero_coord if self.grad_in("coord_1") is None else wt.Float(self.grad_in("coord_1"))
        t_power = zero_coord if self.grad_in("power") is None else wt.Float(self.grad_in("power"))
        self.set_grad_out(
            TransportGridKernel._jvp(
                coord_0=self.coord_0,
                coord_1=self.coord_1,
                power=self.power,
                active_mask=self.active_mask,
                t_coord_0=t_coord_0,
                t_coord_1=t_coord_1,
                t_power=t_power,
                n_samples=self.n_samples,
                bounds=self.bounds,
                cell_size=self.cell_size,
                grid_shape=self.grid_shape,
            )
        )

    def backward(self):
        upstream = self.grad_out()
        if upstream is None:
            return
        grad_coord_0, grad_coord_1, grad_power = TransportGridKernel._backward(
            coord_0=self.coord_0,
            coord_1=self.coord_1,
            power=self.power,
            active_mask=self.active_mask,
            upstream_grid=wt.Float(upstream),
            n_samples=self.n_samples,
            bounds=self.bounds,
            cell_size=self.cell_size,
            grid_shape=self.grid_shape,
        )
        self.set_grad_in("coord_0", grad_coord_0)
        self.set_grad_in("coord_1", grad_coord_1)
        self.set_grad_in("power", grad_power)

    def name(self):
        return "MonteCarloTransportGrid"


_REQUIRED = (
    "monte_carlo_transport_grid_forward_raw",
    "monte_carlo_transport_grid_jvp_raw",
    "monte_carlo_transport_grid_backward_raw",
)


class TransportGridKernel:
    """Static namespace for differentiable tent-grid transport accumulation."""

    @staticmethod
    def require():
        return NativeExtension.require_functions(_REQUIRED, context="Monte Carlo transport-grid native kernel")

    @staticmethod
    def available() -> bool:
        return NativeExtension.has_functions(_REQUIRED)

    @staticmethod
    def _forward(*, coord_0, coord_1, power, active_mask, n_samples: int, bounds, cell_size, grid_shape):
        ext = TransportGridKernel.require()
        return wt.Float(
            ext.monte_carlo_transport_grid_forward_raw(
                wt.Float(coord_0),
                wt.Float(coord_1),
                wt.Float(power),
                wt.Int32(active_mask),
                int(n_samples),
                float(bounds[0][0]),
                float(bounds[1][0]),
                float(cell_size[0]),
                float(cell_size[1]),
                int(grid_shape[0]),
                int(grid_shape[1]),
            )
        )

    @staticmethod
    def _jvp(
        *,
        coord_0,
        coord_1,
        power,
        active_mask,
        t_coord_0,
        t_coord_1,
        t_power,
        n_samples: int,
        bounds,
        cell_size,
        grid_shape,
    ):
        ext = TransportGridKernel.require()
        return wt.Float(
            ext.monte_carlo_transport_grid_jvp_raw(
                wt.Float(coord_0),
                wt.Float(coord_1),
                wt.Float(power),
                wt.Int32(active_mask),
                wt.Float(t_coord_0),
                wt.Float(t_coord_1),
                wt.Float(t_power),
                int(n_samples),
                float(bounds[0][0]),
                float(bounds[1][0]),
                float(cell_size[0]),
                float(cell_size[1]),
                int(grid_shape[0]),
                int(grid_shape[1]),
            )
        )

    @staticmethod
    def _backward(*, coord_0, coord_1, power, active_mask, upstream_grid, n_samples: int, bounds, cell_size, grid_shape):
        ext = TransportGridKernel.require()
        grad_coord_0, grad_coord_1, grad_power = ext.monte_carlo_transport_grid_backward_raw(
            wt.Float(coord_0),
            wt.Float(coord_1),
            wt.Float(power),
            wt.Int32(active_mask),
            wt.Float(upstream_grid),
            int(n_samples),
            float(bounds[0][0]),
            float(bounds[1][0]),
            float(cell_size[0]),
            float(cell_size[1]),
            int(grid_shape[0]),
            int(grid_shape[1]),
        )
        return wt.Float(grad_coord_0), wt.Float(grad_coord_1), wt.Float(grad_power)

    @staticmethod
    def tent_splat(
        *,
        coord_0,
        coord_1,
        power,
        active,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        cell_size: tuple[float, float],
        grid_shape: tuple[int, int],
    ):
        n_cells = int(grid_shape[0]) * int(grid_shape[1])
        n_samples = int(dr.width(power))
        if n_samples <= 0:
            return dr.zeros(wt.Float, n_cells)
        active_mask = wt.Int32(dr.select(active, wt.Int32(1), wt.Int32(0)))
        dr.eval(coord_0, coord_1, power, active_mask)
        return wt.Float(
            dr.custom(
                _TransportGridOp,
                coord_0,
                coord_1,
                power,
                active_mask,
                n_samples=n_samples,
                bounds=bounds,
                cell_size=cell_size,
                grid_shape=grid_shape,
            )
        )


__all__ = [
    "TransportGridKernel",
]
