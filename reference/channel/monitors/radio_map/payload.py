from __future__ import annotations

import drjit as dr
import witwin as wt

from .monitor import RadioMapMonitor
from ...utils import scalar


class RadioMapPayload(dict):
    def __init__(
        self,
        *,
        monitor: RadioMapMonitor,
        grid,
        weighted_diagnostics,
        metadata,
        path_gain,
        rss,
        sinr,
        tx_pos,
        noise_power: float,
        sample_payload_positions,
        timing,
    ) -> None:
        n_rx = int(grid.n_cells)
        super().__init__(
            {
                "name": monitor.name,
                "kind": monitor.kind,
                "metric": monitor.metric,
                "combine_mode": monitor.combine_mode,
                "receiver_model": monitor.receiver_model,
                "grid_shape": grid.grid_shape,
                "cell_size": grid.cell_size,
                "surface": grid.surface_descriptor(),
                "coords": {
                    "grid_x": grid.grid_x,
                    "grid_y": grid.grid_y,
                    "x": grid.x_coords,
                    "y": grid.y_coords,
                    "axis_x": grid.tangential_axes[0],
                    "axis_y": grid.tangential_axes[1],
                    "cell_centers": grid.cell_centers,
                    "sample_positions": tuple(sample_payload_positions),
                },
                "metrics": {
                    "path_gain": path_gain,
                    "rss": rss,
                    "sinr": sinr,
                    "tx_association": dr.zeros(wt.Int32, n_rx),
                },
                "diagnostics": {
                    "coherent": weighted_diagnostics["coherent"],
                    "incoherent": weighted_diagnostics["incoherent"],
                    "coherent_power": weighted_diagnostics["coherent_power"],
                },
                "metadata": metadata,
                "tx_pos": (scalar(tx_pos.x), scalar(tx_pos.y), scalar(tx_pos.z)),
                "tx_power": float(monitor.tx_power),
                "noise_power": float(noise_power),
            }
        )
        if timing is not None:
            self["timing"] = timing


__all__ = ["RadioMapPayload"]
