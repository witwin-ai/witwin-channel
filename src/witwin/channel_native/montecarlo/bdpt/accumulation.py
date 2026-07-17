"""Result accumulation and path-sample assembly helpers for the BDPT solver.

Pure tensor-shaping helpers the pipeline finalize/export stages call: turning
accumulated component matrices into per-cell maps and packing exported
connection-sample dictionaries into :class:`BDPTPathSamples`.
"""

from __future__ import annotations

import torch

from witwin.channel_native.montecarlo.bdpt.kernels.maps import (
    bdpt_los_component_maps_from_matrix,
)

from .result import BDPTPathSamples


def _component_maps_from_matrices(
    component_matrices: dict[str, torch.Tensor],
    *,
    rows: int,
    cols: int,
) -> dict[str, torch.Tensor]:
    return {
        name: bdpt_los_component_maps_from_matrix(component, rows=rows, cols=cols)
        for name, component in component_matrices.items()
    }


def _path_samples_from_connection_export(
    exported: dict[str, torch.Tensor],
) -> BDPTPathSamples:
    return BDPTPathSamples(
        topology=exported["topology"],
        contribution=exported["contribution"],
        pdf=exported["pdf"],
        mis_weight=exported["mis_weight"],
        component_id=exported["component_id"],
        valid=exported["valid"],
        tx_id=exported["tx_id"],
        rx_id=exported["rx_id"],
        grid_linear_id=exported["grid_linear_id"],
        light_depth=exported["light_depth"],
        sensor_depth=exported["sensor_depth"],
        path_length_m=exported["path_length_m"],
    )


def _empty_path_samples(reference: torch.Tensor) -> BDPTPathSamples:
    device = reference.device
    empty_float = torch.empty((0,), device=device, dtype=torch.float32)
    empty_int = torch.empty((0,), device=device, dtype=torch.int32)
    return BDPTPathSamples(
        topology=torch.empty((0, 4), device=device, dtype=torch.int32),
        contribution=empty_float,
        pdf=empty_float.clone(),
        mis_weight=empty_float.clone(),
        component_id=empty_int,
        valid=torch.empty((0,), device=device, dtype=torch.bool),
        tx_id=empty_int.clone(),
        rx_id=empty_int.clone(),
        grid_linear_id=empty_int.clone(),
        light_depth=empty_int.clone(),
        sensor_depth=empty_int.clone(),
        path_length_m=empty_float.clone(),
    )
