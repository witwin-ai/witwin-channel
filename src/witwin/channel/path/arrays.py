from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

from witwin.channel.scene.endpoints import (
    orientation_matrix,
    pattern_field_response,
    steering_vector,
)
from witwin.channel.scene.endpoints import (
    ReceiverGrid,
    ReceiverPoint,
    SolverScene as Scene,
    Transmitter,
)

from .result import PathResult


Receiver = ReceiverPoint | ReceiverGrid


def _unit_vector(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    sin_theta = torch.sin(theta)
    return torch.stack(
        (
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            torch.cos(theta),
        ),
        dim=-1,
    )


def _local_direction(direction: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
    rotation = orientation_matrix(orientation).to(device=direction.device)
    return direction @ rotation


def _flatten_receivers(receivers: Sequence[Receiver]) -> list[Receiver]:
    flattened: list[Receiver] = []
    for receiver in receivers:
        count = (
            receiver.shape[0] * receiver.shape[1]
            if isinstance(receiver, ReceiverGrid)
            else 1
        )
        flattened.extend([receiver] * count)
    return flattened


def _synthetic_endpoint_factor(
    endpoints: Sequence[object],
    directions: torch.Tensor,
    *,
    num_ant: int,
    frequency_hz: float,
    conjugate_pattern: bool,
) -> torch.Tensor:
    """Batch steering/pattern weights over endpoints sharing the same object.

    ``directions`` has shape ``(endpoint, *batch, 3)`` where ``directions[i]``
    is the far-field direction for ``endpoints[i]``. ``steering_vector`` and
    ``AntennaPattern.field_response`` are per-direction operations (the only
    reductions are the fixed length-3 vec3 contractions inside each element),
    so evaluating a group of endpoints in one batched launch is numerically
    identical to evaluating them one at a time. Endpoints that reference the
    same object (for example every element of one ``ReceiverGrid``) share
    array/orientation/pattern and are batched together, collapsing a per-
    endpoint Python loop into one native launch per distinct endpoint object.
    Returns ``(endpoint, *batch, num_ant)``.
    """

    groups: dict[int, list[int]] = {}
    for index, endpoint in enumerate(endpoints):
        groups.setdefault(id(endpoint), []).append(index)

    def _weights(endpoint: object, sub: torch.Tensor) -> torch.Tensor:
        steering = steering_vector(
            endpoint.array,
            sub,
            frequency_hz=frequency_hz,
            orientation=endpoint.orientation,
        )
        pattern = pattern_field_response(
            endpoint.pattern,
            _local_direction(sub, endpoint.orientation),
        )
        if conjugate_pattern:
            pattern = pattern.conj()
        return steering * pattern.unsqueeze(-1)

    if len(groups) == 1:
        # Single shared endpoint object: evaluate every direction in one launch
        # with no scatter (the common single-grid case).
        return _weights(endpoints[0], directions)

    out = torch.empty(
        (len(endpoints), *directions.shape[1:-1], num_ant),
        dtype=torch.complex64,
        device=directions.device,
    )
    for indices in groups.values():
        idx = torch.tensor(indices, device=directions.device, dtype=torch.long)
        out.index_copy_(
            0, idx, _weights(endpoints[indices[0]], directions.index_select(0, idx))
        )
    return out


def _stack_endpoint_weights(
    endpoints: Sequence[object], *, attribute: str, device: torch.device
) -> torch.Tensor | None:
    values = [getattr(endpoint, attribute) for endpoint in endpoints]
    if not values or all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{attribute} must be configured on every endpoint or none")
    return torch.stack(values).to(device=device, dtype=torch.complex64)


def _validate_endpoint_weight_coverage(
    endpoints: Sequence[object], *, attribute: str
) -> None:
    values = [getattr(endpoint, attribute) for endpoint in endpoints]
    if any(value is None for value in values) and any(
        value is not None for value in values
    ):
        raise ValueError(f"{attribute} must be configured on every endpoint or none")


def validate_synthetic_array_scene(scene: Scene) -> None:
    """Reject unavailable synthetic layouts before native scene allocation."""

    flat_receivers = _flatten_receivers(scene.receivers)
    endpoints = [*scene.transmitters, *flat_receivers]
    if any(not endpoint.synthetic_array for endpoint in endpoints):
        raise ValueError("explicit arrays require per-element topology tracing")
    tx_antennas = {transmitter.array.num_antennas for transmitter in scene.transmitters}
    rx_antennas = {receiver.array.num_antennas for receiver in flat_receivers}
    if len(tx_antennas) > 1 or len(rx_antennas) > 1:
        raise ValueError("all endpoints on each side must use the same antenna count")
    _validate_endpoint_weight_coverage(scene.transmitters, attribute="precoding")
    _validate_endpoint_weight_coverage(flat_receivers, attribute="combining")


def pack_synthetic_arrays(
    result: PathResult,
    *,
    frequency_hz: float,
    transmitters: Sequence[Transmitter],
    receivers: Sequence[Receiver],
) -> PathResult:
    """Expand centre-reference paths using far-field array phase weighting.

    Synthetic arrays share one geometric path set across their elements. An
    explicit array must instead trace the element positions and is rejected
    here so that a far-field approximation is never reported as explicit.
    """

    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    flat_receivers = _flatten_receivers(receivers)
    if len(transmitters) != result.num_tx or len(flat_receivers) != result.num_rx:
        raise ValueError("endpoint counts must match the centre-reference result")
    if result.num_tx_ant != 1 or result.num_rx_ant != 1:
        raise ValueError("synthetic packing requires a single-antenna centre result")
    endpoints = [*transmitters, *flat_receivers]
    if any(not endpoint.synthetic_array for endpoint in endpoints):
        raise ValueError("explicit arrays require per-element topology tracing")
    tx_antennas = {transmitter.array.num_antennas for transmitter in transmitters}
    rx_antennas = {receiver.array.num_antennas for receiver in flat_receivers}
    if len(tx_antennas) > 1 or len(rx_antennas) > 1:
        raise ValueError("all endpoints on each side must use the same antenna count")
    num_tx_ant = next(iter(tx_antennas), 1)
    num_rx_ant = next(iter(rx_antennas), 1)
    tx_weights = _stack_endpoint_weights(
        transmitters, attribute="precoding", device=result.a.device
    )
    rx_weights = _stack_endpoint_weights(
        flat_receivers, attribute="combining", device=result.a.device
    )
    if result.num_tx == 0 or result.num_rx == 0:
        metadata = dict(result.metadata)
        metadata.update(
            {
                "array_semantics": "synthetic_far_field_phase_weighting",
                "array_phase_convention": "exp(+j*k*element_position_dot_endpoint_direction)",
                "num_rx_ant": num_rx_ant,
                "num_tx_ant": num_tx_ant,
            }
        )
        return replace(
            result,
            metadata=metadata,
            tx_weights=tx_weights,
            rx_weights=rx_weights,
        )

    theta_t = result.theta_t[:, 0, :, 0]
    phi_t = result.phi_t[:, 0, :, 0]
    theta_r = result.theta_r[:, 0, :, 0]
    phi_r = result.phi_r[:, 0, :, 0]
    departure = _unit_vector(theta_t, phi_t)
    arrival_source_direction = _unit_vector(theta_r, phi_r)

    tx_factors: list[torch.Tensor] = []
    for tx_id, transmitter in enumerate(transmitters):
        direction = departure[:, tx_id]
        steering = steering_vector(
            transmitter.array,
            direction,
            frequency_hz=frequency_hz,
            orientation=transmitter.orientation,
        )
        pattern = pattern_field_response(
            transmitter.pattern,
            _local_direction(direction, transmitter.orientation),
        )
        tx_factors.append(steering * pattern.unsqueeze(-1))
    # (rx, tx, path, tx_ant) -> (rx, tx, tx_ant, path)
    tx_factor = torch.stack(tx_factors, dim=1).permute(0, 1, 3, 2)

    # (rx, tx, path, rx_ant) -> (rx, rx_ant, tx, path)
    rx_factor = _synthetic_endpoint_factor(
        flat_receivers,
        arrival_source_direction,
        num_ant=num_rx_ant,
        frequency_hz=frequency_hz,
        conjugate_pattern=True,
    ).permute(0, 3, 1, 2)
    factor = rx_factor[:, :, :, None, :] * tx_factor[:, None, :, :, :]

    def expand_path(value: torch.Tensor) -> torch.Tensor:
        base = value[:, 0, :, 0]
        return base[:, None, :, None].expand(
            result.num_rx,
            num_rx_ant,
            result.num_tx,
            num_tx_ant,
            *base.shape[2:],
        ).contiguous()

    a = expand_path(result.a) * factor.unsqueeze(-1)
    field_xyz = expand_path(result.field_xyz) * factor.unsqueeze(-1)
    metadata = dict(result.metadata)
    metadata.update(
        {
            "array_semantics": "synthetic_far_field_phase_weighting",
            "array_phase_convention": "exp(+j*k*element_position_dot_endpoint_direction)",
            "num_rx_ant": num_rx_ant,
            "num_tx_ant": num_tx_ant,
        }
    )
    return replace(
        result,
        a=a.to(torch.complex64),
        tau=expand_path(result.tau),
        theta_t=expand_path(result.theta_t),
        phi_t=expand_path(result.phi_t),
        theta_r=expand_path(result.theta_r),
        phi_r=expand_path(result.phi_r),
        valid=expand_path(result.valid),
        interaction_type=expand_path(result.interaction_type),
        primitive_id=expand_path(result.primitive_id),
        material_id=expand_path(result.material_id),
        position=expand_path(result.position),
        normal=expand_path(result.normal),
        num_paths=expand_path(result.num_paths),
        metadata=metadata,
        field_xyz=field_xyz.to(torch.complex64),
        field_direction=expand_path(result.field_direction),
        tx_weights=tx_weights,
        rx_weights=rx_weights,
    )


def explicit_array_scene(scene: Scene) -> tuple[Scene, int, int]:
    """Expand point-endpoint arrays into independently traced scene endpoints."""

    if any(isinstance(receiver, ReceiverGrid) for receiver in scene.receivers):
        raise ValueError("explicit arrays currently require point receivers")
    tx_counts = {tx.array.num_antennas for tx in scene.transmitters}
    rx_counts = {rx.array.num_antennas for rx in scene.receivers}
    if len(tx_counts) > 1 or len(rx_counts) > 1:
        raise ValueError("all explicit endpoints on each side must share antenna count")
    _validate_endpoint_weight_coverage(scene.transmitters, attribute="precoding")
    _validate_endpoint_weight_coverage(scene.receivers, attribute="combining")
    num_tx_ant = next(iter(tx_counts), 1)
    num_rx_ant = next(iter(rx_counts), 1)
    expanded_tx: list[Transmitter] = []
    for transmitter in scene.transmitters:
        positions = transmitter.array.world_positions(
            transmitter.position, transmitter.orientation
        )
        for position in positions:
            expanded_tx.append(
                replace(
                    transmitter,
                    position_override=position,
                    single_element=True,
                )
            )
    expanded_rx: list[ReceiverPoint] = []
    for receiver in scene.receivers:
        assert isinstance(receiver, ReceiverPoint)
        positions = receiver.array.world_positions(receiver.position, receiver.orientation)
        for position in positions:
            expanded_rx.append(
                replace(
                    receiver,
                    position_override=position,
                    single_element=True,
                )
            )
    return (
        replace(
            scene,
            transmitters=tuple(expanded_tx),
            receivers=tuple(expanded_rx),
        ),
        num_rx_ant,
        num_tx_ant,
    )


def pack_explicit_arrays(
    result: PathResult,
    *,
    scene: Scene,
    num_rx_ant: int,
    num_tx_ant: int,
) -> PathResult:
    """Pack independently traced element endpoints into antenna dimensions."""

    num_rx = len(scene.receivers)
    num_tx = len(scene.transmitters)
    if result.num_rx != num_rx * num_rx_ant or result.num_tx != num_tx * num_tx_ant:
        raise ValueError("expanded explicit result does not match endpoint array layout")

    def reshape(value: torch.Tensor) -> torch.Tensor:
        base = value[:, 0, :, 0]
        return base.reshape(num_rx, num_rx_ant, num_tx, num_tx_ant, *base.shape[2:])

    a = reshape(result.a)
    field_xyz = reshape(result.field_xyz)
    theta_t = reshape(result.theta_t)
    phi_t = reshape(result.phi_t)
    theta_r = reshape(result.theta_r)
    phi_r = reshape(result.phi_r)
    pattern_factor = torch.ones_like(theta_t, dtype=torch.complex64)
    departure = _unit_vector(theta_t, phi_t)
    arrival = _unit_vector(theta_r, phi_r)
    for tx_id, transmitter in enumerate(scene.transmitters):
        pattern_factor[:, :, tx_id] *= pattern_field_response(
            transmitter.pattern,
            _local_direction(departure[:, :, tx_id], transmitter.orientation),
        )
    for rx_id, receiver in enumerate(scene.receivers):
        pattern_factor[rx_id] *= pattern_field_response(
            receiver.pattern,
            _local_direction(arrival[rx_id], receiver.orientation),
        ).conj()
    a = a * pattern_factor.unsqueeze(-1)
    field_xyz = field_xyz * pattern_factor.unsqueeze(-1)
    metadata = dict(result.metadata)
    metadata.update(
        {
            "array_semantics": "explicit_per_element_topology",
            "num_rx_ant": num_rx_ant,
            "num_tx_ant": num_tx_ant,
        }
    )
    tx_weights = _stack_endpoint_weights(
        scene.transmitters, attribute="precoding", device=result.a.device
    )
    rx_weights = _stack_endpoint_weights(
        scene.receivers, attribute="combining", device=result.a.device
    )
    return replace(
        result,
        a=a.contiguous(),
        tau=reshape(result.tau),
        theta_t=theta_t,
        phi_t=phi_t,
        theta_r=theta_r,
        phi_r=phi_r,
        valid=reshape(result.valid),
        interaction_type=reshape(result.interaction_type),
        primitive_id=reshape(result.primitive_id),
        material_id=reshape(result.material_id),
        position=reshape(result.position),
        normal=reshape(result.normal),
        num_paths=reshape(result.num_paths),
        metadata=metadata,
        field_xyz=field_xyz.contiguous(),
        field_direction=reshape(result.field_direction),
        tx_weights=tx_weights,
        rx_weights=rx_weights,
    )


__all__ = [
    "explicit_array_scene",
    "pack_explicit_arrays",
    "pack_synthetic_arrays",
    "validate_synthetic_array_scene",
]
