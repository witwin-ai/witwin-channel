"""Explicit path solver API for path export and diagnostics.

``docs/dev/path/README.md`` holds the full ownership contract; a module has no
directory to hold a README. The sections below follow the former package
layout: configuration, ragged storage, padded results, metadata, antenna array
packing, the shared solve pipeline, and the public entry point.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeAlias

import torch

from witwin.channel.components import (
    AD_MODES as _VALID_AD_MODES,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    DEPTH_CAPPED_COMPONENTS,
    apply_exported_path_counts,
    component_availability_status,
    component_max_depth,
    validate_coupled_candidate_limit,
    validate_coupled_gate,
    validate_isb_boundary_taper,
    validate_max_depth,
    validate_scatter_chain,
    validated_components,
)
from witwin.core import Scene, SceneSnapshot
from witwin.channel.propagation.enumerated.engine import (
    evaluate_enumerated_paths,
)
from witwin.channel.propagation.enumerated.scattering import (
    append_scattering_evaluated_paths,
)
from witwin.channel.constants import UNIT_EXCITATION_PHASE_CONVENTION
from witwin.channel.runtime import SolveCapacityTransaction, make_metadata
from witwin.channel import build_info
from witwin.channel.propagation.enumerated.capacity import (
    sanitize_enumerated_capacity_transaction,
)
from witwin.channel.propagation.consumer.replay import compact_evaluated_paths
from witwin.channel.scene.compiler import (
    compile as compile_scene,
    receiver_positions as _shared_receiver_positions,
    transmitter_positions as _shared_transmitter_positions,
)
from witwin.channel.scene.endpoints import (
    ReceiverGrid,
    ReceiverPoint,
    SolverScene,
    Transmitter,
    bind_solver_scene,
    orientation_matrix,
    pattern_field_response,
    steering_vector,
)

if TYPE_CHECKING:
    from witwin.channel.propagation.rows import EvaluatedPaths


# --- Configuration --------------------------------------------------------

# Public component set. transmission exports specular wall-penetration paths
# (wave 2); scattering exports single-bounce incoherent Kirchhoff patch paths
# (wave 3). transmission depth is capped like reflection (chains count wall
# penetrations); scattering is single-bounce in v1.
# Default component set is unchanged: the new components are strictly opt-in.
_VALID_MAX_PATHS_SCOPES = frozenset({"per_pair"})


@dataclass(frozen=True, slots=True)
class Config:
    max_depth: int = 1
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = (
        _DEFAULT_COMPONENTS
    )
    max_paths: int | None = None
    max_paths_scope: str = "per_pair"
    ad_mode: str = "none"
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000
    # Rough-surface scattering quadrature (wave 3): fixed per-area sample
    # density (per-face cap of 4096 samples), a per-pair strongest-paths cap,
    # and an absolute path_gain floor for exported patch paths.
    scattering_samples_per_m2: float = 8.0
    scattering_max_paths_per_pair: int = 4096
    scattering_power_threshold: float = 0.0
    # Enumerated scatter-chain path class (ADR-021 D1). DEFAULT-OFF opt-in:
    # scattering_chain_max_depth = 0 disables chain discovery so exported paths
    # are byte-identical to today. When >= 1 it caps d1 + d2, the combined
    # specular reflection depth of the two legs around the single diffuse vertex
    # (TX --C1(d1)--> v_s --C2(d2)--> RX). Each leg is bounded by the native
    # kMaxAdDepth = 8, so the public cap is 2 * 8 = 16. Chain vertices are drawn
    # at a documented lower density (scattering_chain_samples_per_m2) and only
    # the strongest scattering_chain_max_rows joined rows per (tx, rx) survive.
    # Requires the 'scattering' component.
    scattering_chain_max_depth: int = 0
    scattering_chain_samples_per_m2: float = 2.0
    scattering_chain_max_rows: int = 256
    # ISB boundary taper (ADR-017). DEFAULT-OFF visual-continuity heuristic: the
    # hard LoS occlusion gate becomes a C1 membership taper tau(c / (width * w_F))
    # and the compensating order-1 diffraction odd step spreads over the same
    # congruent window. OFF (the default) is bit-identical to the hard gate and
    # the unchanged diffraction window for every existing caller (enforced by a
    # bitwise regression test); the switch must never default ON. The width
    # scales the Fresnel penumbra w_F of the grazed silhouette edge; the
    # projection-validated optimum is 0.5 (artifacts/isb-taper/report.json).
    isb_boundary_taper: bool = False
    isb_boundary_taper_width: float = 0.5
    def __post_init__(self) -> None:
        validate_max_depth(self.max_depth)
        validate_isb_boundary_taper(self.isb_boundary_taper_width)
        if self.scattering_samples_per_m2 <= 0.0:
            raise ValueError("scattering_samples_per_m2 must be positive")
        if self.scattering_max_paths_per_pair <= 0:
            raise ValueError("scattering_max_paths_per_pair must be positive")
        if self.scattering_power_threshold < 0.0:
            raise ValueError("scattering_power_threshold must be non-negative")
        components = validated_components(
            self.components, error_message="components must be a subset of {valid}"
        )
        validate_scatter_chain(
            max_depth=self.scattering_chain_max_depth,
            samples_per_m2=self.scattering_chain_samples_per_m2,
            max_rows=self.scattering_chain_max_rows,
            components=components,
        )
        if self.max_depth > 5 and components & DEPTH_CAPPED_COMPONENTS:
            raise RuntimeError("path reflection/transmission support max_depth <= 5")
        validate_coupled_gate(
            coupled_paths=self.coupled_paths,
            max_depth=self.max_depth,
            components=components,
        )
        if self.max_paths is not None and self.max_paths <= 0:
            raise ValueError("max_paths must be positive when set")
        if self.max_paths_scope not in _VALID_MAX_PATHS_SCOPES:
            raise ValueError("path max_paths_scope must be 'per_pair'")
        validate_coupled_candidate_limit(self.coupled_candidate_limit)
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError(
                f"path ad_mode must be one of {sorted(_VALID_AD_MODES)}"
            )
        object.__setattr__(self, "components", components)


# --- Ragged per-link storage ----------------------------------------------


def _validate_tensor_predicate(predicate: torch.Tensor, message: str) -> None:
    """Validate without synchronizing CUDA tensors back to the host."""

    if predicate.device.type == "cuda":
        torch._assert_async(predicate, message)
    elif not bool(predicate):
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class RaggedPathSoA:
    """Stable per-link ragged path storage used before public padding."""

    num_rx: int
    num_rx_ant: int
    num_tx: int
    num_tx_ant: int
    num_time_steps: int
    pair_offsets: torch.Tensor
    rx_id: torch.Tensor
    rx_ant_id: torch.Tensor
    tx_id: torch.Tensor
    tx_ant_id: torch.Tensor
    field: torch.Tensor
    delay_s: torch.Tensor
    theta_t: torch.Tensor
    phi_t: torch.Tensor
    theta_r: torch.Tensor
    phi_r: torch.Tensor
    interaction_type: torch.Tensor
    primitive_id: torch.Tensor
    material_id: torch.Tensor
    position: torch.Tensor
    normal: torch.Tensor

    @property
    def path_count(self) -> int:
        return int(self.delay_s.shape[0])

    @property
    def pair_count(self) -> int:
        return self.num_rx * self.num_rx_ant * self.num_tx * self.num_tx_ant

    @property
    def max_depth(self) -> int:
        return int(self.interaction_type.shape[1])

    @classmethod
    def from_flat(
        cls,
        *,
        num_rx: int,
        num_rx_ant: int,
        num_tx: int,
        num_tx_ant: int,
        rx_id: torch.Tensor,
        tx_id: torch.Tensor,
        field: torch.Tensor,
        delay_s: torch.Tensor,
        theta_t: torch.Tensor,
        phi_t: torch.Tensor,
        theta_r: torch.Tensor,
        phi_r: torch.Tensor,
        interaction_type: torch.Tensor,
        primitive_id: torch.Tensor,
        material_id: torch.Tensor,
        position: torch.Tensor,
        normal: torch.Tensor,
        rx_ant_id: torch.Tensor | None = None,
        tx_ant_id: torch.Tensor | None = None,
        max_paths_per_pair: int | None = None,
    ) -> "RaggedPathSoA":
        count = int(delay_s.shape[0])
        device = delay_s.device
        if field.ndim == 1:
            field = field.unsqueeze(-1)
        if field.ndim != 2 or field.shape[0] != count:
            raise ValueError("field must have shape (path, time)")
        if field.dtype != torch.complex64:
            raise ValueError("field must use complex64")
        if max_paths_per_pair is not None and max_paths_per_pair <= 0:
            raise ValueError("max_paths_per_pair must be positive when set")
        rx_ant_id = (
            torch.zeros((count,), device=device, dtype=torch.int32)
            if rx_ant_id is None
            else rx_ant_id.to(device=device, dtype=torch.int32)
        )
        tx_ant_id = (
            torch.zeros((count,), device=device, dtype=torch.int32)
            if tx_ant_id is None
            else tx_ant_id.to(device=device, dtype=torch.int32)
        )
        rx_id = rx_id.to(device=device, dtype=torch.int32)
        tx_id = tx_id.to(device=device, dtype=torch.int32)
        for name, value in {
            "rx_id": rx_id,
            "rx_ant_id": rx_ant_id,
            "tx_id": tx_id,
            "tx_ant_id": tx_ant_id,
            "delay_s": delay_s,
            "theta_t": theta_t,
            "phi_t": phi_t,
            "theta_r": theta_r,
            "phi_r": phi_r,
        }.items():
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (path,)")
        depth = int(interaction_type.shape[1]) if interaction_type.ndim == 2 else -1
        if depth < 0:
            raise ValueError("interaction_type must have shape (path, depth)")
        for name, value in {
            "interaction_type": interaction_type,
            "primitive_id": primitive_id,
            "material_id": material_id,
        }.items():
            if value.shape != (count, depth):
                raise ValueError(f"{name} must have shape (path, depth)")
        for name, value in {"position": position, "normal": normal}.items():
            if value.shape != (count, depth, 3):
                raise ValueError(f"{name} must have shape (path, depth, 3)")

        pair_count = int(num_rx) * int(num_rx_ant) * int(num_tx) * int(num_tx_ant)
        pair_index = (
            (rx_id.to(torch.int64) * int(num_rx_ant) + rx_ant_id) * int(num_tx) + tx_id
        ) * int(num_tx_ant) + tx_ant_id
        if count and pair_count <= 0:
            raise ValueError("non-empty paths require non-empty endpoint dimensions")
        endpoint_ranges = (
            (rx_id, int(num_rx), "rx_id"),
            (rx_ant_id, int(num_rx_ant), "rx_ant_id"),
            (tx_id, int(num_tx), "tx_id"),
            (tx_ant_id, int(num_tx_ant), "tx_ant_id"),
        )
        for endpoint_id, dimension, name in endpoint_ranges:
            _validate_tensor_predicate(
                ((endpoint_id >= 0) & (endpoint_id < dimension)).all(),
                f"{name} is outside the declared endpoint dimension",
            )
        order = torch.argsort(pair_index, stable=True)
        pair_index = pair_index[order]
        counts = torch.bincount(pair_index, minlength=pair_count)
        starts = torch.cumsum(counts, dim=0) - counts
        ranks = torch.arange(
            count, device=device, dtype=torch.int64
        ) - torch.repeat_interleave(starts, counts)
        keep = (
            torch.ones((count,), device=device, dtype=torch.bool)
            if max_paths_per_pair is None
            else ranks < int(max_paths_per_pair)
        )
        order = order[keep]
        pair_index = pair_index[keep]
        counts = torch.bincount(pair_index, minlength=pair_count)
        pair_offsets = torch.cat(
            (
                torch.zeros((1,), device=device, dtype=torch.int64),
                torch.cumsum(counts, dim=0),
            )
        )

        def select(
            value: torch.Tensor, *, dtype: torch.dtype | None = None
        ) -> torch.Tensor:
            selected = value.to(device=device, dtype=dtype or value.dtype)[order]
            return selected.contiguous()

        return cls(
            num_rx=int(num_rx),
            num_rx_ant=int(num_rx_ant),
            num_tx=int(num_tx),
            num_tx_ant=int(num_tx_ant),
            num_time_steps=int(field.shape[1]),
            pair_offsets=pair_offsets.contiguous(),
            rx_id=select(rx_id, dtype=torch.int32),
            rx_ant_id=select(rx_ant_id, dtype=torch.int32),
            tx_id=select(tx_id, dtype=torch.int32),
            tx_ant_id=select(tx_ant_id, dtype=torch.int32),
            field=select(field, dtype=torch.complex64),
            delay_s=select(delay_s, dtype=torch.float32),
            theta_t=select(theta_t, dtype=torch.float32),
            phi_t=select(phi_t, dtype=torch.float32),
            theta_r=select(theta_r, dtype=torch.float32),
            phi_r=select(phi_r, dtype=torch.float32),
            interaction_type=select(interaction_type, dtype=torch.int32),
            primitive_id=select(primitive_id, dtype=torch.int32),
            material_id=select(material_id, dtype=torch.int32),
            position=select(position, dtype=torch.float32),
            normal=select(normal, dtype=torch.float32),
        )


# --- Padded public result and signal views --------------------------------

_LIGHT_SPEED_M_PER_S = 299_792_458.0


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
        if self.field_xyz.shape != (*path_shape, 3):  # type: ignore[union-attr]
            raise ValueError("field_xyz must match the path shape with a complex3 tail")
        if self.field_direction.shape != (*path_shape, 3):  # type: ignore[union-attr]
            raise ValueError("field_direction must match the path shape with a vec3 tail")
        if self.field_xyz.dtype != torch.complex64:  # type: ignore[union-attr]
            raise ValueError("field_xyz must use complex64")
        if self.field_direction.dtype != torch.float32:  # type: ignore[union-attr]
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
        if any(tensor.device != self.a.device for tensor in tensors):  # type: ignore[union-attr]
            raise ValueError("all PathResult tensors must share one device")
        # Device-selected cardinality is already reflected in the compact native
        # result. Construction is metadata-only and must not recompute it.

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
        return tuple(int(value) for value in self.a.shape[:-1])  # type: ignore[return-value]

    @property
    def path_count_shape(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.a.shape[:4])  # type: ignore[return-value]

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


# --- Solver metadata ------------------------------------------------------


def _metadata(
    *,
    config: Config,
    path_count: int,
    reflection_available: bool,
    diffraction_available: bool,
    path_native_available: bool,
    transmission_path_count: int = 0,
    scattering_path_count: int = 0,
    ad_companion_launches: int = 0,
    ad_tape_bytes: int = 0,
    forward_time_ms: float = 0.0,
    peak_memory_bytes: int = 0,
    scattering_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Plan 07 AD-4: the real registered-companion accounting. vjp retains
    # tape and schedules its companions on the user's later backward; jvp
    # runs its dual companions inside this forward and retains no tape.
    kernel = make_metadata(
        primitive="path_solver",
        forward_launch_count=1 if path_count else 0,
        backward_launch_count=(
            ad_companion_launches if config.ad_mode == "vjp" else 0
        ),
        jvp_launch_count=(
            ad_companion_launches if config.ad_mode == "jvp" else 0
        ),
        tape_bytes=ad_tape_bytes if config.ad_mode == "vjp" else 0,
        accumulation_strategy="none",
        scheduling_strategy="native_cuda",
        rayd_native=reflection_available or diffraction_available,
        ad_status=config.ad_mode,
        forward_time_ms=forward_time_ms,
        peak_memory_bytes=peak_memory_bytes,
    )
    capability = {
        "path_native": path_native_available,
        "rayd_native": reflection_available or diffraction_available,
        "reflection": reflection_available,
        "diffraction": diffraction_available,
    }
    component_depths = component_max_depth(
        config.components, chain_depth=config.max_depth, single_bounce_depth=1
    )
    if config.coupled_paths:
        # The coupled 1R1D/1D1R family reaches depth 2 whenever it is requested,
        # including when plain diffraction is not in the component set.
        component_depths["diffraction"] = 2
    effective_max_depth = max(
        component_depths["los"],
        component_depths["reflection"],
        component_depths["diffraction"],
    )
    components = component_availability_status(
        config.components,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        reflection_error="reflection paths require RayD native capability",
        diffraction_error="diffraction paths require RayD native capability",
        depth_available=config.max_depth >= 1,
        reflection_depth_error="reflection paths require max_depth >= 1",
        diffraction_depth_error="diffraction paths require max_depth >= 1",
    )
    apply_exported_path_counts(
        components,
        config.components,
        transmission_path_count=transmission_path_count,
        scattering_path_count=scattering_path_count,
    )
    metadata = {
        "solver": "path",
        "device": "cuda",
        "path_count": path_count,
        "effective_max_depth": effective_max_depth,
        "component_max_depth": component_depths,
        "max_paths_per_pair": config.max_paths,
        "native_capabilities": capability,
        "components": components,
        "kernel": kernel,
        "field_abi": "complex3_v1",
        "phase_convention": dict(UNIT_EXCITATION_PHASE_CONVENTION),
        "coefficient_semantics": "unit_excitation_dimensionless_receiver_projection",
        "coupled_paths": {
            "requested": config.coupled_paths,
            "geometry": "native_1r1d_reciprocal"
            if config.coupled_paths
            else "not_requested",
            "coefficient": "unified_complex3_jones"
            if config.coupled_paths
            else "not_requested",
        },
    }
    if "transmission" in config.components:
        # Endpoint-connection thin_sheet contract (plan 05 section 4).
        metadata["transmission"] = {
            "thin_sheet_straight_path_approximation": True,
            "group_delay": "geometric",
        }
    if scattering_info is not None:
        # Incoherent Kirchhoff patch quadrature (plan 05 wave 3); the flag
        # documents that per-path phases are NOT physical for ensemble rows.
        metadata["scattering"] = dict(scattering_info)
    return metadata


# --- Antenna array packing ------------------------------------------------

Receiver: TypeAlias = ReceiverPoint | ReceiverGrid


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
            endpoint.array,  # type: ignore[attr-defined]
            sub,
            frequency_hz=frequency_hz,
            orientation=endpoint.orientation,  # type: ignore[attr-defined]
        )
        pattern = pattern_field_response(
            endpoint.pattern,  # type: ignore[attr-defined]
            _local_direction(sub, endpoint.orientation),  # type: ignore[attr-defined]
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


def validate_synthetic_array_scene(scene: SolverScene) -> None:
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


def explicit_array_scene(scene: SolverScene) -> tuple[SolverScene, int, int]:
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
    scene: SolverScene,
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


# --- Shared solve pipeline ------------------------------------------------

_COMPONENT_ID = {
    "los": 0,
    "reflection": 1,
    "diffraction": 2,
    "reflection_diffraction": 3,
    "diffraction_reflection": 4,
    # transmission exports specular wall-penetration paths since wave 2;
    # scattering exports incoherent Kirchhoff patch paths since wave 3.
    "transmission": 5,
    "scattering": 6,
}


@dataclass(frozen=True, slots=True)
class _DeferredPathResult:
    """Internal result plus the one terminal check deferred past array packing."""

    result: PathResult
    capacity_transaction: SolveCapacityTransaction | None


def _stable_endpoint_id_lookups(
    scene: Scene, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    source_ids = tuple(int(view.source.antenna_id) for view in scene.transmitters)
    sink_ids: list[int] = []
    for view in scene.receivers:
        sample_count = (
            int(view.shape[0]) * int(view.shape[1])
            if hasattr(view, "shape")
            else 1
        )
        sink_ids.extend((int(view.source.antenna_id),) * sample_count)
    return (
        torch.tensor(source_ids, device=device, dtype=torch.int64),
        torch.tensor(sink_ids, device=device, dtype=torch.int64),
    )


def _transmitter_tensors(scene: Scene) -> tuple[torch.Tensor, torch.Tensor]:
    return _shared_transmitter_positions(scene, device=torch.device("cuda"))  # type: ignore[no-any-return]


def _receiver_positions(scene: Scene, *, reference: torch.Tensor) -> torch.Tensor:
    return _shared_receiver_positions(
        scene, device=reference.device, reference=reference
    )


def _validate_runtime(config: Config) -> tuple[bool, bool, bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel.path solver requires CUDA")
    if config.isb_boundary_taper and config.ad_mode != "none":
        # ISB boundary taper (ADR-017) gate 3: the C1 clearance-factor AD
        # companion is a documented follow-up; reject taper + AD loudly rather
        # than returning a silently incomplete gradient.
        raise RuntimeError(
            "isb_boundary_taper does not support ad_mode != 'none' yet "
            "(ADR-017 gate 3 C1 clearance companion is a follow-up)"
        )
    info = build_info()
    reflection_available = bool(info["uses_rayd_native"])
    diffraction_available = bool(info["uses_rayd_native"])
    path_native_available = bool(info.get("uses_path_native", False))
    if not path_native_available:
        raise RuntimeError(
            "path solver requires _channel path native CUDA kernels"
        )
    if "reflection" in config.components and not reflection_available:
        raise RuntimeError("reflection paths require RayD native capability")
    if "diffraction" in config.components and not diffraction_available:
        raise RuntimeError("diffraction paths require RayD native capability")
    if config.max_depth < 1 and (
        "reflection" in config.components or "diffraction" in config.components
    ):
        raise RuntimeError("requested scattering paths require max_depth >= 1")
    return reflection_available, diffraction_available, path_native_available


def _pipeline_solve_base(
    scene: Scene,
    config: Config,
    *,
    validate_runtime: Callable[[Config], tuple[bool, bool, bool]] = _validate_runtime,
    evaluate_enumerated_paths: Callable[..., Any] = evaluate_enumerated_paths,
    append_scattering_evaluated_paths: Callable[..., Any] = (
        append_scattering_evaluated_paths
    ),
    metadata: Callable[..., dict[str, Any]] = _metadata,
    transmitter_tensors: Callable[..., tuple[torch.Tensor, torch.Tensor]] = (
        _transmitter_tensors
    ),
    receiver_positions: Callable[..., torch.Tensor] = _receiver_positions,
    pack_evaluated_paths: Callable[..., PathResult] = from_evaluated_paths,
) -> _DeferredPathResult:
    reflection_available, diffraction_available, path_native_available = (
        validate_runtime(config)
    )
    # Solve-level wall time and CUDA high-water-mark delta for the kernel
    # metadata (plan 07 AD-4). AD instrumentation only: the syncs would break
    # host/device overlap for a caller looping over ad_mode="none" solves, so
    # none-mode reports zeros and takes no sync (zero-overhead primal contract).
    ad_instrumented = config.ad_mode != "none"
    solve_start = 0.0
    peak_before = 0
    if ad_instrumented:
        torch.cuda.synchronize()
        solve_start = perf_counter()
        peak_before = torch.cuda.max_memory_allocated()
    evaluated, sidecars = evaluate_enumerated_paths(
        scene, config, defer_capacity_terminal=True
    )
    scattering_info = None
    appended_scattering = "scattering" in config.components
    if appended_scattering:
        evaluated, sidecars, scattering_info = append_scattering_evaluated_paths(
            scene,
            config,
            evaluated,
            sidecars,
        )
    evaluated, sidecars = sanitize_enumerated_capacity_transaction(evaluated, sidecars)
    if appended_scattering:
        # The legacy incoherent scattering owner appends independently
        # compacted rows after the canonical coherent block. Re-establish the
        # public Path pair-major order only for that legacy solver route.
        source_ids, sink_ids = _stable_endpoint_id_lookups(
            scene, device=evaluated.device
        )
        evaluated = compact_evaluated_paths(
            evaluated,
            source_stable_ids=source_ids,
            sink_stable_ids=sink_ids,
        ).evaluated
    path_count = evaluated.row_count
    if ad_instrumented:
        torch.cuda.synchronize()
        forward_time_ms = (perf_counter() - solve_start) * 1.0e3
        peak_memory_bytes = max(0, torch.cuda.max_memory_allocated() - peak_before)
    else:
        forward_time_ms = 0.0
        peak_memory_bytes = 0
    result_metadata = metadata(
        config=config,
        path_count=path_count,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        path_native_available=path_native_available,
        transmission_path_count=int(
            (evaluated.topology.component_id == _COMPONENT_ID["transmission"])
            .sum()
            .item()
        ),
        scattering_path_count=int(
            (evaluated.topology.component_id == _COMPONENT_ID["scattering"])
            .sum()
            .item()
        ),
        ad_companion_launches=sidecars.execution.ad_companion_launches,
        ad_tape_bytes=sidecars.execution.ad_tape_bytes,
        forward_time_ms=forward_time_ms,
        peak_memory_bytes=peak_memory_bytes,
        scattering_info=scattering_info,
    )
    tx_positions, _tx_power = transmitter_tensors(scene)
    rx_positions = receiver_positions(scene, reference=tx_positions)
    return _DeferredPathResult(
        result=pack_evaluated_paths(
            evaluated,
            num_rx=int(rx_positions.shape[0]),
            num_tx=int(tx_positions.shape[0]),
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            metadata=result_metadata,
        ),
        capacity_transaction=sidecars.capacity_transaction,
    )


def _pipeline_solve(
    scene: Scene,
    config: Config,
    *,
    solve_base: Callable[[Scene, Config], _DeferredPathResult] = _pipeline_solve_base,
) -> PathResult:
    """Solve canonical paths and pack synthetic or explicit antenna arrays."""

    endpoints = [*scene.transmitters, *scene.receivers]
    if any(not endpoint.synthetic_array for endpoint in endpoints):
        expanded_scene, num_rx_ant, num_tx_ant = explicit_array_scene(scene)
        deferred = solve_base(expanded_scene, config)
        result = pack_explicit_arrays(
            deferred.result,
            scene=scene,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
        )
        if deferred.capacity_transaction is not None:
            deferred.capacity_transaction.terminal_check()
        return result
    validate_synthetic_array_scene(scene)
    deferred = solve_base(scene, config)
    result = pack_synthetic_arrays(
        deferred.result,
        frequency_hz=scene.frequency,
        transmitters=scene.transmitters,
        receivers=scene.receivers,
    )
    if deferred.capacity_transaction is not None:
        deferred.capacity_transaction.terminal_check()
    return result


# --- Public entry point ---------------------------------------------------


def _solve_base(scene: SolverScene, config: Config) -> _DeferredPathResult:
    """Delegate one centre-endpoint solve through the shared pipeline."""

    return _pipeline_solve_base(
        scene,
        config,
        validate_runtime=_validate_runtime,
        evaluate_enumerated_paths=evaluate_enumerated_paths,
        append_scattering_evaluated_paths=append_scattering_evaluated_paths,
        metadata=_metadata,
        transmitter_tensors=_transmitter_tensors,
        receiver_positions=_receiver_positions,
        pack_evaluated_paths=from_evaluated_paths,
    )


def solve(  # type: ignore[no-untyped-def]
    scene: Scene | SceneSnapshot,
    config: Config,
    *,
    reference_frequency_hz,
) -> PathResult:
    """Solve canonical paths and pack synthetic or explicit antenna arrays."""

    compiled = compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )
    return _pipeline_solve(
        bind_solver_scene(compiled), config, solve_base=_solve_base
    )


__all__ = [
    "Config",
    "InteractionType",
    "PathResult",
    "RaggedPathSoA",
    "solve",
]
