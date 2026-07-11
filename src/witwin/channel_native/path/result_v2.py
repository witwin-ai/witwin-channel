from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .schema import RaggedPathSoA

if TYPE_CHECKING:
    from witwin.channel_native.core.path_topology import TopologyBatch


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
class PathResultV2:
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
        )
        if any(tensor.device != self.a.device for tensor in tensors):
            raise ValueError("all PathResultV2 tensors must share one device")
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
    ) -> "PathResultV2":
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
                "schema": "PathResultV2",
                "schema_version": 2,
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

    def filter_by_type(self, *interaction_types: int) -> "PathResultV2":
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
        return PathResultV2.from_ragged(
            ragged,
            minimum_path_width=1,
            metadata=metadata,
        )


def endpoint_angles(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(direction, dim=-1)
    safe = torch.clamp(norm, min=torch.finfo(direction.dtype).tiny)
    theta = torch.acos(torch.clamp(direction[..., 2] / safe, -1.0, 1.0))
    phi = torch.atan2(direction[..., 1], direction[..., 0])
    return theta.to(torch.float32), phi.to(torch.float32)


def from_legacy_result(
    result: object,
    *,
    num_rx: int,
    num_tx: int,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    max_paths_per_pair: int | None = None,
) -> PathResultV2:
    """Explicitly adapt the legacy flat scalar-power result to V2."""

    count = int(result.delay_s.shape[0])
    device = result.delay_s.device
    tx_id = result.tx_id.to(device=device, dtype=torch.int32)
    rx_id = result.rx_id.to(device=device, dtype=torch.int32)
    direction = (
        rx_positions[rx_id.to(torch.int64)] - tx_positions[tx_id.to(torch.int64)]
    )
    theta_t, phi_t = endpoint_angles(direction)
    theta_r, phi_r = endpoint_angles(-direction)
    scattering = result.component_id != 0
    unknown_angle = torch.full_like(theta_t, float("nan"))
    theta_t = torch.where(scattering, unknown_angle, theta_t)
    phi_t = torch.where(scattering, unknown_angle, phi_t)
    theta_r = torch.where(scattering, unknown_angle, theta_r)
    phi_r = torch.where(scattering, unknown_angle, phi_r)
    depth = max(int(result.metadata.get("effective_max_depth", 0)), 0)
    interaction_type = torch.zeros((count, depth), device=device, dtype=torch.int32)
    primitive_id = torch.full((count, depth), -1, device=device, dtype=torch.int32)
    material_id = torch.full((count, depth), -1, device=device, dtype=torch.int32)
    if depth:
        reflection = result.component_id == 1
        diffraction = result.component_id == 2
        interaction_type[reflection, 0] = int(InteractionType.REFLECTION)
        interaction_type[diffraction, 0] = int(InteractionType.DIFFRACTION)
        primitive_id[:, 0] = result.primitive_id.to(dtype=torch.int32)
    position = torch.full(
        (count, depth, 3), float("nan"), device=device, dtype=torch.float32
    )
    normal = torch.full_like(position, float("nan"))
    field = torch.complex(
        torch.sqrt(torch.clamp(result.path_gain.to(torch.float32), min=0.0)),
        torch.zeros((count,), device=device, dtype=torch.float32),
    ).unsqueeze(-1)
    ragged = RaggedPathSoA.from_flat(
        num_rx=num_rx,
        num_rx_ant=1,
        num_tx=num_tx,
        num_tx_ant=1,
        rx_id=rx_id,
        tx_id=tx_id,
        field=field,
        delay_s=result.delay_s,
        theta_t=theta_t,
        phi_t=phi_t,
        theta_r=theta_r,
        phi_r=phi_r,
        interaction_type=interaction_type,
        primitive_id=primitive_id,
        material_id=material_id,
        position=position,
        normal=normal,
        max_paths_per_pair=max_paths_per_pair,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "adapter": "legacy_flat_scalar_power_to_PathResultV2",
            "coefficient_semantics": "magnitude_only_zero_phase",
            "phase_convention": "adapter_zero_phase_not_physical",
            "angle_semantics": "los_exact_scattering_unavailable",
            "interaction_geometry": "unavailable_in_legacy_source",
        }
    )
    return PathResultV2.from_ragged(
        ragged,
        max_paths_per_pair=max_paths_per_pair,
        metadata=metadata,
    )


def from_topology_result(
    paths: "TopologyBatch",
    *,
    num_rx: int,
    num_tx: int,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> PathResultV2:
    """Pack the shared canonical topology without losing event sequences."""

    count = int(paths.delay_s.numel())
    tx_id = paths.tx_id.to(dtype=torch.int32)
    rx_id = paths.rx_id.to(dtype=torch.int32)
    tx_for_path = tx_positions[tx_id.to(dtype=torch.int64)]
    rx_for_path = rx_positions[rx_id.to(dtype=torch.int64)]
    direct = rx_for_path - tx_for_path
    width = int(paths.interaction_type.shape[1])
    if width:
        first = paths.interaction_positions[:, 0]
        last_index = torch.clamp(paths.depth.to(dtype=torch.int64) - 1, min=0)
        row = torch.arange(count, device=paths.delay_s.device)
        last = paths.interaction_positions[row, last_index]
        has_interaction = paths.depth > 0
        departure = torch.where(has_interaction[:, None], first - tx_for_path, direct)
        arrival = torch.where(has_interaction[:, None], rx_for_path - last, direct)
    else:
        departure = direct
        arrival = direct
    theta_t, phi_t = endpoint_angles(departure)
    theta_r, phi_r = endpoint_angles(-arrival)

    object_sequence = paths.primitive_sequence.clone()
    if width:
        diffraction = (paths.component_id == 2) & (paths.depth > 0)
        object_sequence[diffraction, 0] = paths.edge_id[diffraction]
    ragged = RaggedPathSoA.from_flat(
        num_rx=num_rx,
        num_rx_ant=1,
        num_tx=num_tx,
        num_tx_ant=1,
        rx_id=rx_id,
        tx_id=tx_id,
        field=paths.path_field.unsqueeze(-1),
        delay_s=paths.delay_s,
        theta_t=theta_t,
        phi_t=phi_t,
        theta_r=theta_r,
        phi_r=phi_r,
        interaction_type=paths.interaction_type,
        primitive_id=object_sequence,
        material_id=paths.material_sequence,
        position=paths.interaction_positions,
        normal=paths.interaction_normals,
    )
    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "adapter": "canonical_topology_to_PathResultV2",
            "coefficient_semantics": "native_scalar_complex_field",
            "coupled_coefficient_semantics": "nan_until_unified_phase_3_transport",
            "interaction_geometry": "canonical_topology",
        }
    )
    return PathResultV2.from_ragged(ragged, metadata=result_metadata)


__all__ = [
    "InteractionType",
    "PathResultV2",
    "endpoint_angles",
    "from_legacy_result",
    "from_topology_result",
]
