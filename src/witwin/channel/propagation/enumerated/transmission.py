"""Enumerated straight-segment transmission topology discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel.scene import endpoints as scene_endpoints
from witwin.channel.propagation.geometry.kernels import (
    rayd_segment_penetration_ad,
    rayd_segment_penetration_forward,
)
from witwin.channel.propagation.penetration import (
    SegmentPenetrationPolicy,
)
from witwin.channel.propagation.topology.concatenate import _empty_path_block
from witwin.channel.propagation.topology.export import _ensure_topology_fields
from witwin.channel.propagation.topology.kernels import (
    enumerated_transmission_topology_pack,
)
from witwin.channel.runtime import (
    CapacityExecutionCounts,
    CapacityFailureState,
    CudaProfileRange,
    profiled_cuda_range,
)

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene


def _pair_major_endpoints(
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structurally expand endpoints as ``tx * rx_count + rx`` rows."""

    tx_count = int(tx_positions.shape[0])
    rx_count = int(rx_positions.shape[0])
    device = tx_positions.device
    tx_index = torch.arange(
        tx_count, device=device, dtype=torch.int64
    ).repeat_interleave(rx_count)
    rx_index = torch.arange(rx_count, device=device, dtype=torch.int64).repeat(tx_count)
    return (
        tx_positions.index_select(0, tx_index).contiguous(),
        rx_positions.index_select(0, rx_index).contiguous(),
    )


@profiled_cuda_range(CudaProfileRange.ENUMERATED_PENETRATION_DISCOVERY)
def _transmission_topology(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    max_depth: int,
    ad_mode: str,
    failure_state: CapacityFailureState | None,
) -> tuple[dict[str, torch.Tensor], int, CapacityExecutionCounts | None]:
    """Trace and pack one pair-major RayD penetration batch.

    Device-selected cardinality remains in the returned execution sidecar.
    Structural no-op scenes do not allocate a failure state or launch native
    work; every non-empty request shares the engine-owned failure state between
    RayD penetration and the component-5 topology pack.
    """

    device = tx_positions.device
    tx_count = int(tx_positions.shape[0])
    rx_count = int(rx_positions.shape[0])
    if not scene.structures or tx_count == 0 or rx_count == 0 or max_depth < 1:
        return _ensure_topology_fields(_empty_path_block(device)), 0, None
    if not isinstance(failure_state, CapacityFailureState):
        raise TypeError("non-empty transmission discovery requires failure_state")

    rayd = compiled.rayd
    if not rayd.available:
        raise RuntimeError(
            "deterministic transmission requires RayD native scene capability"
        )

    if ad_mode == "none":
        tx_geometry = tx_positions
        rx_geometry = rx_positions
        vertices = None
    else:
        vertices = scene_endpoints.scene_vertex_table(scene, compiled)
        tx_geometry = scene_endpoints.transmitter_positions_ad(
            scene, tx_positions, device=device
        )
        rx_geometry = scene_endpoints.receiver_positions_ad(
            scene, rx_positions, device=device
        )
    origins, targets = _pair_major_endpoints(tx_geometry, rx_geometry)

    penetration_args = {
        "input_active_any": True,
        "hit_capacity": int(max_depth),
        "policy": SegmentPenetrationPolicy.EnumeratedFullDistance,
        "scene_diagonal": compiled.enumerated_penetration_scene_diagonal_m,
        "failure_state": failure_state,
    }
    if ad_mode == "none":
        penetration = rayd_segment_penetration_forward(
            rayd,
            origins,
            targets,
            None,
            **penetration_args,
        )
    else:
        assert vertices is not None
        penetration = rayd_segment_penetration_ad(
            rayd,
            vertices,
            origins,
            targets,
            None,
            **penetration_args,
        )

    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    geometry_mode_id = compiled.materials.geometry_mode_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    topology = enumerated_transmission_topology_pack(
        penetration,
        face_material_id,
        geometry_mode_id,
        tx_count=tx_count,
        rx_count=rx_count,
    )
    capacity_block = topology.as_block()
    # Transitional ADR-027 activation: preserve the live compact row shape
    # until the ADR-029 atomic selector/gather switch. This device-selected
    # structural compaction is the remaining Phase D synchronization blocker;
    # no geometry or RF quantity is recomputed here.
    selected = torch.nonzero(topology.valid, as_tuple=False).reshape(-1)
    block = {
        name: value.index_select(0, selected).contiguous()
        for name, value in capacity_block.items()
    }
    return _ensure_topology_fields(block), 1, topology.execution
