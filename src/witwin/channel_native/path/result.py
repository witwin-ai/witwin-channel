from __future__ import annotations

import enum
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from .schema import RaggedPathSoA

_LIGHT_SPEED_M_PER_S = 299_792_458.0

if TYPE_CHECKING:
    from witwin.channel_native.core.path_topology import TopologyBatch
    from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths


def _validate_tensor_predicate(predicate: torch.Tensor, message: str) -> None:
    if predicate.device.type == "cuda":
        torch._assert_async(predicate, message)
    elif not bool(predicate):
        raise ValueError(message)


class InteractionType(enum.IntFlag):
    NONE = 0
    REFLECTION = 1
    DIFFRACTION = 2
    TRANSMISSION = 4
    SCATTERING = 8


def _masked_min(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] == 0:
        return torch.zeros(
            (*values.shape[:-1], 1), device=values.device, dtype=values.dtype
        )
    minima = (
        torch.where(valid, values, torch.full_like(values, float("inf")))
        .min(dim=-1, keepdim=True)
        .values
    )
    return torch.where(torch.isfinite(minima), minima, torch.zeros_like(minima))


@dataclass(frozen=True, slots=True)
class PathResult:
    """Padded public path result with explicit antenna and time dimensions."""

    a: torch.Tensor
    tau: torch.Tensor
    theta_t: torch.Tensor
    phi_t: torch.Tensor
    theta_r: torch.Tensor
    phi_r: torch.Tensor
    valid: torch.Tensor
    interaction_type: torch.Tensor
    primitive_id: torch.Tensor
    material_id: torch.Tensor
    position: torch.Tensor
    normal: torch.Tensor
    num_paths: torch.Tensor
    metadata: dict[str, Any]
    field_xyz: torch.Tensor | None = None
    field_direction: torch.Tensor | None = None
    tx_weights: torch.Tensor | None = None
    rx_weights: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.a.ndim != 6 or self.a.dtype != torch.complex64:
            raise ValueError(
                "a must be complex64 with shape (rx, rx_ant, tx, tx_ant, path, time)"
            )
        path_shape = self.a.shape[:-1]
        for name in ("tau", "theta_t", "phi_t", "theta_r", "phi_r", "valid"):
            if getattr(self, name).shape != path_shape:
                raise ValueError(f"{name} must match the path shape")
        depth_shape = (*path_shape, self.max_depth)
        for name in ("interaction_type", "primitive_id", "material_id"):
            if getattr(self, name).shape != depth_shape:
                raise ValueError(f"{name} must match the depth shape")
        vector_shape = (*depth_shape, 3)
        for name in ("position", "normal"):
            if getattr(self, name).shape != vector_shape:
                raise ValueError(f"{name} must match the vector depth shape")
        if self.num_paths.shape != self.path_count_shape:
            raise ValueError("num_paths must match the endpoint-pair shape")
        if self.field_xyz is None:
            object.__setattr__(
                self,
                "field_xyz",
                torch.zeros((*path_shape, 3), device=self.a.device, dtype=torch.complex64),
            )
        if self.field_direction is None:
            object.__setattr__(
                self,
                "field_direction",
                torch.zeros((*path_shape, 3), device=self.a.device, dtype=torch.float32),
            )
        if self.field_xyz.shape != (*path_shape, 3):
            raise ValueError("field_xyz must match the path shape with a complex3 tail")
        if self.field_direction.shape != (*path_shape, 3):
            raise ValueError("field_direction must match the path shape with a vec3 tail")
        if self.field_xyz.dtype != torch.complex64:
            raise ValueError("field_xyz must use complex64")
        if self.field_direction.dtype != torch.float32:
            raise ValueError("field_direction must use float32")
        if self.tx_weights is not None:
            if self.tx_weights.shape != (self.num_tx, self.num_tx_ant):
                raise ValueError("tx_weights must match the transmitter antenna shape")
            if self.tx_weights.device != self.a.device:
                raise ValueError("tx_weights must share the PathResult device")
        if self.rx_weights is not None:
            if self.rx_weights.shape != (self.num_rx, self.num_rx_ant):
                raise ValueError("rx_weights must match the receiver antenna shape")
            if self.rx_weights.device != self.a.device:
                raise ValueError("rx_weights must share the PathResult device")
        if self.valid.dtype != torch.bool:
            raise ValueError("valid must use bool")
        if any(
            getattr(self, name).dtype != torch.float32
            for name in (
                "tau",
                "theta_t",
                "phi_t",
                "theta_r",
                "phi_r",
                "position",
                "normal",
            )
        ):
            raise ValueError("path geometry tensors must use float32")
        if any(
            getattr(self, name).dtype != torch.int32
            for name in ("interaction_type", "primitive_id", "material_id", "num_paths")
        ):
            raise ValueError("interaction and path-count tensors must use int32")
        tensors = (
            self.a,
            self.tau,
            self.theta_t,
            self.phi_t,
            self.theta_r,
            self.phi_r,
            self.valid,
            self.interaction_type,
            self.primitive_id,
            self.material_id,
            self.position,
            self.normal,
            self.num_paths,
            self.field_xyz,
            self.field_direction,
        )
        if any(tensor.device != self.a.device for tensor in tensors):
            raise ValueError("all PathResult tensors must share one device")
        _validate_tensor_predicate(
            (self.num_paths == self.valid.sum(dim=-1, dtype=torch.int32)).all(),
            "num_paths must equal the valid path count for each pair",
        )

    @property
    def num_rx(self) -> int:
        return int(self.a.shape[0])

    @property
    def num_rx_ant(self) -> int:
        return int(self.a.shape[1])

    @property
    def num_tx(self) -> int:
        return int(self.a.shape[2])

    @property
    def num_tx_ant(self) -> int:
        return int(self.a.shape[3])

    @property
    def max_num_paths(self) -> int:
        return int(self.a.shape[4])

    @property
    def num_time_steps(self) -> int:
        return int(self.a.shape[5])

    @property
    def max_depth(self) -> int:
        return int(self.interaction_type.shape[-1])

    @property
    def path_shape(self) -> tuple[int, int, int, int, int]:
        return tuple(int(value) for value in self.a.shape[:-1])

    @property
    def path_count_shape(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.a.shape[:4])

    @property
    def rx_id(self) -> torch.Tensor:
        ids = torch.arange(self.num_rx, device=self.a.device, dtype=torch.int32)
        return ids.reshape(-1, 1, 1, 1, 1).expand(self.path_shape).contiguous()

    @property
    def tx_id(self) -> torch.Tensor:
        ids = torch.arange(self.num_tx, device=self.a.device, dtype=torch.int32)
        return ids.reshape(1, 1, -1, 1, 1).expand(self.path_shape).contiguous()

    @property
    def path_length_m(self) -> torch.Tensor:
        length = self.tau * _LIGHT_SPEED_M_PER_S
        return torch.where(self.valid, length, torch.full_like(length, -1.0))

    @property
    def types(self) -> torch.Tensor:
        return self.interaction_type

    @property
    def vertices(self) -> torch.Tensor:
        return self.position

    @property
    def normals(self) -> torch.Tensor:
        return self.normal

    @property
    def objects(self) -> torch.Tensor:
        return self.primitive_id

    @classmethod
    def from_ragged(
        cls,
        ragged: RaggedPathSoA,
        *,
        max_paths_per_pair: int | None = None,
        minimum_path_width: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> "PathResult":
        counts = ragged.pair_offsets[1:] - ragged.pair_offsets[:-1]
        if max_paths_per_pair is None:
            # A padded tensor has a data-dependent dimension. This is the one
            # required CUDA-to-host scalar synchronization in dynamic packing;
            # all validation remains asynchronous on CUDA.
            max_paths_per_pair = int(counts.max().item()) if counts.numel() else 0
        max_paths_per_pair = max(int(max_paths_per_pair), int(minimum_path_width))
        if max_paths_per_pair < 0:
            raise ValueError("max_paths_per_pair must be non-negative")
        _validate_tensor_predicate(
            (counts <= max_paths_per_pair).all(),
            "ragged input exceeds max_paths_per_pair",
        )
        base_shape = (
            ragged.num_rx,
            ragged.num_rx_ant,
            ragged.num_tx,
            ragged.num_tx_ant,
        )
        path_shape = (*base_shape, int(max_paths_per_pair))
        device = ragged.delay_s.device
        starts = ragged.pair_offsets[:-1]
        ranks = torch.arange(
            ragged.path_count, device=device, dtype=torch.int64
        ) - torch.repeat_interleave(starts, counts)
        pair_index = torch.repeat_interleave(
            torch.arange(ragged.pair_count, device=device, dtype=torch.int64), counts
        )
        linear = pair_index * int(max_paths_per_pair) + ranks

        def padded_scalar(source: torch.Tensor, fill: float | int) -> torch.Tensor:
            out = torch.full(
                (ragged.pair_count * int(max_paths_per_pair),),
                fill,
                device=device,
                dtype=source.dtype,
            )
            out[linear] = source
            return out.reshape(path_shape)

        def padded_tail(
            source: torch.Tensor, tail: tuple[int, ...], fill: float | int
        ) -> torch.Tensor:
            out = torch.full(
                (ragged.pair_count * int(max_paths_per_pair), *tail),
                fill,
                device=device,
                dtype=source.dtype,
            )
            out[linear] = source
            return out.reshape(*path_shape, *tail)

        valid = torch.zeros(
            (ragged.pair_count * int(max_paths_per_pair),),
            device=device,
            dtype=torch.bool,
        )
        valid[linear] = True
        result_metadata = dict(metadata or {})
        result_metadata.update(
            {
                "schema": "PathResult",
                "schema_version": 1,
                "max_paths_scope": "per_pair",
                "stable_order": "input_order_within_receiver_antenna_transmitter_antenna_pair",
            }
        )
        return cls(
            a=padded_tail(ragged.field, (ragged.num_time_steps,), 0.0),
            tau=padded_scalar(ragged.delay_s, -1.0),
            theta_t=padded_scalar(ragged.theta_t, 0.0),
            phi_t=padded_scalar(ragged.phi_t, 0.0),
            theta_r=padded_scalar(ragged.theta_r, 0.0),
            phi_r=padded_scalar(ragged.phi_r, 0.0),
            valid=valid.reshape(path_shape),
            interaction_type=padded_tail(
                ragged.interaction_type, (ragged.max_depth,), int(InteractionType.NONE)
            ),
            primitive_id=padded_tail(ragged.primitive_id, (ragged.max_depth,), -1),
            material_id=padded_tail(ragged.material_id, (ragged.max_depth,), -1),
            position=padded_tail(ragged.position, (ragged.max_depth, 3), 0.0),
            normal=padded_tail(ragged.normal, (ragged.max_depth, 3), 0.0),
            num_paths=counts.to(dtype=torch.int32).reshape(base_shape),
            metadata=result_metadata,
        )

    def cir(
        self, *, normalize_delays: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tau = self.tau
        if normalize_delays:
            tau = tau - _masked_min(tau, self.valid)
        tau = torch.where(self.valid, tau, torch.full_like(tau, -1.0))
        return torch.where(
            self.valid.unsqueeze(-1), self.a, torch.zeros_like(self.a)
        ), tau

    def cfr(
        self, frequencies: torch.Tensor, *, normalize_delays: bool = True
    ) -> torch.Tensor:
        if frequencies.ndim != 1:
            raise ValueError("frequencies must have shape (frequency,)")
        tau = self.tau
        if normalize_delays:
            tau = tau - _masked_min(tau, self.valid)
        tau = torch.where(self.valid, tau, torch.zeros_like(tau))
        coeff = torch.where(self.valid.unsqueeze(-1), self.a, torch.zeros_like(self.a))
        frequency = frequencies.to(device=self.a.device, dtype=torch.float32)
        phase = torch.exp(-2.0j * math.pi * tau.unsqueeze(-1) * frequency)
        return (coeff.unsqueeze(-1) * phase.unsqueeze(-2)).sum(dim=-3)

    def taps(
        self, bandwidth: float, num_taps: int, *, normalize_delays: bool = True
    ) -> torch.Tensor:
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be > 0")
        if num_taps <= 0:
            raise ValueError("num_taps must be > 0")
        tau = self.tau
        if normalize_delays:
            tau = tau - _masked_min(tau, self.valid)
        tap_index = torch.round(tau * float(bandwidth)).to(dtype=torch.int64)
        output = torch.zeros(
            (*self.path_count_shape, self.num_time_steps, int(num_taps)),
            device=self.a.device,
            dtype=torch.complex64,
        )
        if self.max_num_paths == 0:
            return output
        base_count = self.num_rx * self.num_rx_ant * self.num_tx * self.num_tx_ant
        coeff = self.a.reshape(base_count, self.max_num_paths, self.num_time_steps)
        tap_index = tap_index.reshape(base_count, self.max_num_paths)
        keep = (
            self.valid.reshape(base_count, self.max_num_paths)
            & (tap_index >= 0)
            & (tap_index < num_taps)
        )
        base = (
            torch.arange(base_count, device=self.a.device)
            .view(-1, 1, 1)
            .expand_as(coeff.real)
        )
        time = (
            torch.arange(self.num_time_steps, device=self.a.device)
            .view(1, 1, -1)
            .expand_as(coeff.real)
        )
        tap = tap_index.unsqueeze(-1).expand_as(coeff.real)
        keep = keep.unsqueeze(-1).expand_as(coeff.real)
        output.reshape(base_count, self.num_time_steps, num_taps).index_put_(
            (base[keep], time[keep], tap[keep]), coeff[keep], accumulate=True
        )
        return output

    def beamform(
        self,
        *,
        tx_weights: torch.Tensor | None = None,
        rx_weights: torch.Tensor | None = None,
    ) -> "BeamformedPathResult":
        """Return a signal view using complex per-endpoint antenna weights.

        The raw antenna channel in this result is unchanged.  When weights are
        omitted, the precoding/combining weights captured from the solved
        scene are used.
        """

        tx = self.tx_weights if tx_weights is None else tx_weights
        rx = self.rx_weights if rx_weights is None else rx_weights
        if tx is None or rx is None:
            raise ValueError(
                "beamform requires tx_weights and rx_weights, either explicitly "
                "or on every solved endpoint"
            )
        tx = tx.to(device=self.a.device, dtype=torch.complex64)
        rx = rx.to(device=self.a.device, dtype=torch.complex64)
        if tx.ndim == 1 and self.num_tx == 1:
            tx = tx.unsqueeze(0)
        if rx.ndim == 1 and self.num_rx == 1:
            rx = rx.unsqueeze(0)
        if tx.shape != (self.num_tx, self.num_tx_ant):
            raise ValueError(
                f"tx_weights must have shape {(self.num_tx, self.num_tx_ant)}"
            )
        if rx.shape != (self.num_rx, self.num_rx_ant):
            raise ValueError(
                f"rx_weights must have shape {(self.num_rx, self.num_rx_ant)}"
            )
        return BeamformedPathResult(source=self, tx_weights=tx, rx_weights=rx)

    def filter_by_type(self, *interaction_types: int) -> "PathResult":
        if not interaction_types:
            return self
        requested = {int(value) for value in interaction_types}
        nonempty = self.interaction_type != int(InteractionType.NONE)
        keep = self.valid & (~nonempty.any(dim=-1) if 0 in requested else False)
        for value in requested - {0}:
            keep = keep | (self.valid & (self.interaction_type == value).any(dim=-1))
        pair_count = self.num_rx * self.num_rx_ant * self.num_tx * self.num_tx_ant
        pair = (
            torch.arange(pair_count, device=self.a.device)
            .view(-1, 1)
            .expand(-1, self.max_num_paths)
        )
        pair = pair.reshape(-1)[keep.reshape(-1)]
        rx_id = pair // (self.num_rx_ant * self.num_tx * self.num_tx_ant)
        remainder = pair % (self.num_rx_ant * self.num_tx * self.num_tx_ant)
        rx_ant_id = remainder // (self.num_tx * self.num_tx_ant)
        remainder = remainder % (self.num_tx * self.num_tx_ant)
        tx_id = remainder // self.num_tx_ant
        tx_ant_id = remainder % self.num_tx_ant
        ragged = RaggedPathSoA.from_flat(
            num_rx=self.num_rx,
            num_rx_ant=self.num_rx_ant,
            num_tx=self.num_tx,
            num_tx_ant=self.num_tx_ant,
            rx_id=rx_id,
            rx_ant_id=rx_ant_id,
            tx_id=tx_id,
            tx_ant_id=tx_ant_id,
            field=self.a[keep],
            delay_s=self.tau[keep],
            theta_t=self.theta_t[keep],
            phi_t=self.phi_t[keep],
            theta_r=self.theta_r[keep],
            phi_r=self.phi_r[keep],
            interaction_type=self.interaction_type[keep],
            primitive_id=self.primitive_id[keep],
            material_id=self.material_id[keep],
            position=self.position[keep],
            normal=self.normal[keep],
        )
        metadata = dict(self.metadata)
        metadata["filtered_interaction_types"] = sorted(requested)
        filtered = PathResult.from_ragged(
            ragged,
            minimum_path_width=1,
            metadata=metadata,
        )
        return replace(
            filtered,
            tx_weights=self.tx_weights,
            rx_weights=self.rx_weights,
        )


@dataclass(frozen=True, slots=True)
class BeamformedPathResult:
    """Beamformed signal views backed by an immutable raw :class:`PathResult`."""

    source: PathResult
    tx_weights: torch.Tensor
    rx_weights: torch.Tensor

    def cir(
        self, *, normalize_delays: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.source
        factor = (
            self.rx_weights.conj().reshape(source.num_rx, source.num_rx_ant, 1, 1, 1, 1)
            * self.tx_weights.reshape(1, 1, source.num_tx, source.num_tx_ant, 1, 1)
        )
        coefficient = torch.where(
            source.valid.unsqueeze(-1), source.a * factor, torch.zeros_like(source.a)
        )
        coefficient = coefficient.permute(0, 2, 1, 3, 4, 5).reshape(
            source.num_rx,
            source.num_tx,
            source.num_rx_ant * source.num_tx_ant * source.max_num_paths,
            source.num_time_steps,
        )
        valid = source.valid.permute(0, 2, 1, 3, 4).reshape(
            source.num_rx, source.num_tx, -1
        )
        tau = source.tau.permute(0, 2, 1, 3, 4).reshape(
            source.num_rx, source.num_tx, -1
        )
        if normalize_delays:
            tau = tau - _masked_min(tau, valid)
        tau = torch.where(valid, tau, torch.full_like(tau, -1.0))
        return coefficient, tau

    def cfr(
        self, frequencies: torch.Tensor, *, normalize_delays: bool = True
    ) -> torch.Tensor:
        if frequencies.ndim != 1:
            raise ValueError("frequencies must have shape (frequency,)")
        coefficient, tau = self.cir(normalize_delays=normalize_delays)
        valid = tau >= 0.0
        safe_tau = torch.where(valid, tau, torch.zeros_like(tau))
        frequency = frequencies.to(device=coefficient.device, dtype=torch.float32)
        phase = torch.exp(-2.0j * math.pi * safe_tau.unsqueeze(-1) * frequency)
        return (coefficient.unsqueeze(-1) * phase.unsqueeze(-2)).sum(dim=-3)

    def taps(
        self, bandwidth: float, num_taps: int, *, normalize_delays: bool = True
    ) -> torch.Tensor:
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be > 0")
        if num_taps <= 0:
            raise ValueError("num_taps must be > 0")
        coefficient, tau = self.cir(normalize_delays=normalize_delays)
        tap_index = torch.round(tau * float(bandwidth)).to(dtype=torch.int64)
        output = torch.zeros(
            (
                self.source.num_rx,
                self.source.num_tx,
                self.source.num_time_steps,
                int(num_taps),
            ),
            device=coefficient.device,
            dtype=torch.complex64,
        )
        if coefficient.shape[2] == 0:
            return output
        base_count = self.source.num_rx * self.source.num_tx
        coefficient = coefficient.reshape(
            base_count, coefficient.shape[2], self.source.num_time_steps
        )
        tap_index = tap_index.reshape(base_count, tap_index.shape[2])
        keep = (tap_index >= 0) & (tap_index < num_taps)
        base = (
            torch.arange(base_count, device=coefficient.device)
            .view(-1, 1, 1)
            .expand_as(coefficient.real)
        )
        time = (
            torch.arange(self.source.num_time_steps, device=coefficient.device)
            .view(1, 1, -1)
            .expand_as(coefficient.real)
        )
        tap = tap_index.unsqueeze(-1).expand_as(coefficient.real)
        keep = keep.unsqueeze(-1).expand_as(coefficient.real)
        output.reshape(base_count, self.source.num_time_steps, num_taps).index_put_(
            (base[keep], time[keep], tap[keep]), coefficient[keep], accumulate=True
        )
        return output


def endpoint_angles(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(direction, dim=-1)
    safe = torch.clamp(norm, min=torch.finfo(direction.dtype).tiny)
    theta = torch.acos(torch.clamp(direction[..., 2] / safe, -1.0, 1.0))
    phi = torch.atan2(direction[..., 1], direction[..., 0])
    return theta.to(torch.float32), phi.to(torch.float32)


def from_evaluated_paths(
    paths: "EvaluatedPaths",
    *,
    num_rx: int,
    num_tx: int,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> PathResult:
    """Pack evaluated propagation rows without losing event sequences."""

    topology = paths.topology
    geometry = paths.geometry
    fields = paths.fields
    count = paths.row_count
    tx_id = topology.tx_id.to(dtype=torch.int32)
    rx_id = topology.rx_id.to(dtype=torch.int32)
    tx_for_path = tx_positions[tx_id.to(dtype=torch.int64)]
    rx_for_path = rx_positions[rx_id.to(dtype=torch.int64)]
    direct = rx_for_path - tx_for_path
    width = int(topology.interaction_type.shape[1])
    if width:
        first = geometry.interaction_positions[:, 0]
        last_index = torch.clamp(topology.depth.to(dtype=torch.int64) - 1, min=0)
        row = torch.arange(count, device=geometry.delay_s.device)
        last = geometry.interaction_positions[row, last_index]
        has_interaction = topology.depth > 0
        departure = torch.where(has_interaction[:, None], first - tx_for_path, direct)
        arrival = torch.where(has_interaction[:, None], rx_for_path - last, direct)
    else:
        departure = direct
        arrival = direct
    theta_t, phi_t = endpoint_angles(departure)
    theta_r, phi_r = endpoint_angles(-arrival)

    object_sequence = topology.primitive_sequence.clone()
    if width:
        diffraction = (topology.component_id == 2) & (topology.depth > 0)
        object_sequence[diffraction, 0] = topology.edge_id[diffraction]
    interaction_normals = geometry.interaction_normals
    diffraction = topology.interaction_type == int(InteractionType.DIFFRACTION)
    interaction_normals = torch.where(
        diffraction.unsqueeze(-1) & ~torch.isfinite(interaction_normals),
        torch.zeros_like(interaction_normals),
        interaction_normals,
    )
    ragged = RaggedPathSoA.from_flat(
        num_rx=num_rx,
        num_rx_ant=1,
        num_tx=num_tx,
        num_tx_ant=1,
        rx_id=rx_id,
        tx_id=tx_id,
        field=fields.coefficient.unsqueeze(-1),
        delay_s=geometry.delay_s,
        theta_t=theta_t,
        phi_t=phi_t,
        theta_r=theta_r,
        phi_r=phi_r,
        interaction_type=topology.interaction_type,
        primitive_id=object_sequence,
        material_id=topology.material_sequence,
        position=geometry.interaction_positions,
        normal=interaction_normals,
    )
    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "coefficient_semantics": "unit_excitation_dimensionless_receiver_projection",
            "coupled_coefficient_semantics": "unified_complex3_jones",
            "interaction_geometry": "canonical_topology",
        }
    )
    result = PathResult.from_ragged(ragged, metadata=result_metadata)
    field_xyz = torch.zeros(
        (*result.path_shape, 3),
        device=result.a.device,
        dtype=torch.complex64,
    )
    field_direction = torch.zeros(
        (*result.path_shape, 3),
        device=result.a.device,
        dtype=torch.float32,
    )
    field_xyz[result.valid] = fields.field_xyz
    field_direction[result.valid] = geometry.field_direction
    return replace(
        result,
        field_xyz=field_xyz,
        field_direction=field_direction,
    )


def from_topology_result(
    paths: "TopologyBatch",
    *,
    num_rx: int,
    num_tx: int,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> PathResult:
    """Pack a legacy mixed topology table through the split row contracts."""

    from witwin.channel_native.propagation.models.adapters import (
        evaluated_paths_from_topology_batch,
    )

    evaluated, _ = evaluated_paths_from_topology_batch(paths)
    return from_evaluated_paths(
        evaluated,
        num_rx=num_rx,
        num_tx=num_tx,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        metadata=metadata,
    )


__all__ = [
    "InteractionType",
    "PathResult",
    "endpoint_angles",
    "from_evaluated_paths",
    "from_topology_result",
]
