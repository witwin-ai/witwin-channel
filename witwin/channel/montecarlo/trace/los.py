"""Per-cell line-of-sight tracing for Monte Carlo radio maps."""

from __future__ import annotations

from dataclasses import dataclass

import drjit as dr

from witwin.channel.core.scene import Scene
from witwin.channel.core.numerics import arrays
from witwin.channel.core.physics import polarization
from witwin.channel.core.numerics.constants import EPS
from witwin.channel.montecarlo import types as wt

from .. import grid_ops
from ..config import ResolvedTraceConfig
from witwin.channel.core.grid import Grid
from ..sampler import Sampler
from .ad_support import MC_TX_TRANSPORT_FD_STEP, SceneQuery, SparseCoeffBuffers


@dataclass(slots=True)
class LosTape:
    ray_dir: object
    cell_idx: object
    transport_ray_dir: object
    transport_blocker_prim_idx: object

    @staticmethod
    def empty() -> "LosTape":
        return LosTape(
            ray_dir=arrays.empty_vector3(),
            cell_idx=dr.zeros(wt.UInt32, 0),
            transport_ray_dir=arrays.empty_vector3(),
            transport_blocker_prim_idx=dr.zeros(wt.Int32, 0),
        )

    @classmethod
    def from_payload(cls, payload: dict) -> "LosTape":
        return cls(
            ray_dir=payload["ray_dir"],
            cell_idx=payload["cell_idx"],
            transport_ray_dir=payload["transport_ray_dir"],
            transport_blocker_prim_idx=payload["transport_blocker_prim_idx"],
        )


@dataclass(slots=True)
class LoSTraceResult:
    """Per-cell LoS output used by integrators and AD replay."""

    power: object
    visible: object
    path_count: object
    tape: LosTape | None


class LoS:
    """Static namespace for cell-centered LoS evaluation."""

    @staticmethod
    def power_to_targets(*, tx_pos, target_pos, config: ResolvedTraceConfig):
        ray_dir = target_pos - tx_pos
        distance = dr.norm(ray_dir)
        ray_hat = ray_dir / (distance + wt.Float(EPS))
        field = grid_ops.point_source(
            tx_pos,
            wt.Complex2f(1.0, 0.0),
            target_pos,
            config.wavelength,
            config.k,
        )
        field_vector = polarization.vector_scale(Sampler.source_field(ray_hat), field)
        return polarization.vector_power(field_vector), ray_hat

    @staticmethod
    def trace(
        *,
        scene: Scene,
        grid: Grid,
        tx_pos,
        config: ResolvedTraceConfig,
        collect_ad_tapes: bool = False,
    ) -> LoSTraceResult:
        n_cells = int(grid.n_cells)
        cell_idx = dr.arange(wt.UInt32, n_cells)
        power, ray_hat = LoS.power_to_targets(
            tx_pos=tx_pos,
            target_pos=grid.cell_centers,
            config=config,
        )
        visible = scene.segment_visible(tx_pos, grid.cell_centers)
        power = dr.select(visible, power, wt.Float(0.0))
        path_count = wt.UInt32(dr.count(visible))
        tape = None
        if collect_ad_tapes:
            visible_lane = dr.compress(visible)
            visible_count = int(dr.width(visible_lane))
            if visible_count > 0:
                visible_cell_idx = dr.gather(wt.UInt32, cell_idx, visible_lane)
                visible_ray_hat = dr.gather(wt.Vector3f, ray_hat, visible_lane)
                tape = LosTape(
                    ray_dir=visible_ray_hat,
                    cell_idx=visible_cell_idx,
                    transport_ray_dir=visible_ray_hat,
                    transport_blocker_prim_idx=dr.full(wt.Int32, -1, visible_count),
                )
            else:
                tape = LosTape.empty()
        return LoSTraceResult(
            power=power,
            visible=visible,
            path_count=path_count,
            tape=tape,
        )


class LosAD:
    """Namespace for LOS sparse coefficient extraction."""

    @staticmethod
    def sparse_coeffs(
        *,
        tape: LosTape,
        tx_pos,
        grid: Grid,
        config: ResolvedTraceConfig,
        solid_angle_per_ray: float,
        cell_area: float,
    ):
        width = int(dr.width(tape.cell_idx))
        if width <= 0:
            return SparseCoeffBuffers.empty()

        local_tx = SceneQuery.tx_lanes(tx_pos, width)
        dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
        target_pos = dr.gather(wt.Point3f, grid.cell_centers, tape.cell_idx)
        ray_dir = target_pos - local_tx
        distance = dr.norm(ray_dir)
        ray_hat = ray_dir / (distance + wt.Float(1.0e-6))
        field = grid_ops.point_source(
            local_tx,
            wt.Complex2f(1.0, 0.0),
            target_pos,
            config.wavelength,
            config.k,
        )
        del solid_angle_per_ray, cell_area
        field_vector = polarization.vector_scale(Sampler.source_field(ray_hat), field)
        contribution = polarization.vector_power(field_vector)
        dr.backward(dr.sum(contribution))
        return SparseCoeffBuffers(
            cell_idx=tape.cell_idx,
            tx_coeff_x=dr.grad(local_tx.x),
            tx_coeff_y=dr.grad(local_tx.y),
            tx_coeff_z=dr.grad(local_tx.z),
            vertex_indices=dr.zeros(wt.Int32, 0),
            vertex_coeff_x=dr.zeros(wt.Float, 0),
            vertex_coeff_y=dr.zeros(wt.Float, 0),
            vertex_coeff_z=dr.zeros(wt.Float, 0),
            vertex_slot_count=0,
            material_indices=dr.zeros(wt.Int32, 0),
            material_coeff_eps=dr.zeros(wt.Float, 0),
            material_coeff_sigma=dr.zeros(wt.Float, 0),
            material_slot_count=0,
        )

    @staticmethod
    def transport_maps(
        *,
        tape: LosTape,
        scene: Scene,
        tx_pos,
        grid: Grid,
        config: ResolvedTraceConfig,
        solid_angle_per_ray: float,
        cell_area: float,
        transport_step: float = MC_TX_TRANSPORT_FD_STEP,
    ):
        width = int(dr.width(tape.cell_idx))
        zero_map = dr.zeros(wt.Float, int(grid.n_cells))
        if width <= 0:
            return {"x": zero_map, "y": zero_map, "z": zero_map}
        local_tx = SceneQuery.tx_lanes(tx_pos, width)
        target_pos = dr.gather(wt.Point3f, grid.cell_centers, tape.cell_idx)
        step_scalar = float(transport_step)
        step = wt.Float(step_scalar)
        del scene, solid_angle_per_ray, cell_area

        def replay(tx_lanes):
            ray_dir = target_pos - tx_lanes
            distance = dr.norm(ray_dir)
            ray_hat = ray_dir / (distance + wt.Float(1.0e-6))
            field = grid_ops.point_source(
                tx_lanes,
                wt.Complex2f(1.0, 0.0),
                target_pos,
                config.wavelength,
                config.k,
            )
            field_vector = polarization.vector_scale(Sampler.source_field(ray_hat), field)
            return dr.detach(polarization.vector_power(field_vector))

        def scatter_cell_power(power):
            out = dr.zeros(wt.Float, int(grid.n_cells))
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                out,
                power,
                tape.cell_idx,
                dr.full(wt.Bool, True, width),
            )
            return out

        def map_for_shift(dx: float, dy: float, dz: float):
            shifted_tx = wt.Point3f(local_tx.x + wt.Float(dx), local_tx.y + wt.Float(dy), local_tx.z + wt.Float(dz))
            return scatter_cell_power(replay(shifted_tx))

        return {
            "x": (map_for_shift(step_scalar, 0.0, 0.0) - map_for_shift(-step_scalar, 0.0, 0.0)) / (wt.Float(2.0) * step),
            "y": (map_for_shift(0.0, step_scalar, 0.0) - map_for_shift(0.0, -step_scalar, 0.0)) / (wt.Float(2.0) * step),
            "z": (map_for_shift(0.0, 0.0, step_scalar) - map_for_shift(0.0, 0.0, -step_scalar)) / (wt.Float(2.0) * step),
        }


__all__ = ["LoS", "LosAD", "LoSTraceResult", "LosTape"]
