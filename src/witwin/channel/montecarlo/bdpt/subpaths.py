from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.montecarlo.bdpt.kernels.paths import (
    bdpt_empty_subpath_state,
)


@dataclass(frozen=True, slots=True)
class SubpathState:
    origin: torch.Tensor
    direction: torch.Tensor
    throughput_real: torch.Tensor
    throughput_imag: torch.Tensor
    pdf_forward: torch.Tensor
    pdf_reverse: torch.Tensor
    depth: torch.Tensor
    component_mask: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    grid_linear_id: torch.Tensor
    valid: torch.Tensor
    path_length: torch.Tensor
    field_real: torch.Tensor
    field_imag: torch.Tensor
    source_power: torch.Tensor
    event_type: torch.Tensor


def empty_subpath_state(reference: torch.Tensor) -> SubpathState:
    state = bdpt_empty_subpath_state(reference)
    return SubpathState(
        origin=state["origin"],
        direction=state["direction"],
        throughput_real=state["throughput_real"],
        throughput_imag=state["throughput_imag"],
        pdf_forward=state["pdf_forward"],
        pdf_reverse=state["pdf_reverse"],
        depth=state["depth"],
        component_mask=state["component_mask"],
        primitive_id=state["primitive_id"],
        edge_id=state["edge_id"],
        tx_id=state["tx_id"],
        rx_id=state["rx_id"],
        grid_linear_id=state["grid_linear_id"],
        valid=state["valid"],
        path_length=state["path_length"],
        field_real=state["field_real"],
        field_imag=state["field_imag"],
        source_power=state["source_power"],
        event_type=state["event_type"],
    )
