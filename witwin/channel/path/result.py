from __future__ import annotations

import enum
import math
from dataclasses import dataclass, fields
from typing import Mapping

import torch

from witwin.channel.core.numerics.tensors import (
    BoolTensor,
    FloatTensor,
    IntTensor,
    to_torch_view,
    to_bool_tensor,
    to_complex_array,
    to_float_tensor,
    to_int_tensor,
    to_mapping_proxy,
    to_vector_tensor,
)


class InteractionType(enum.IntFlag):
    """Interaction type codes stored per depth slot in ``Result.types``."""

    NONE = 0
    REFLECTION = 1
    DIFFRACTION = 2
    TRANSMISSION = 4
    SCATTERING = 8


def _masked_min(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] == 0:
        return torch.zeros((*values.shape[:-1], 1), device=values.device, dtype=values.dtype)
    inf = torch.full_like(values, float("inf"))
    minima = torch.where(valid, values, inf).min(dim=-1, keepdim=True).values
    return torch.where(torch.isfinite(minima), minima, torch.zeros_like(minima))


@dataclass(frozen=True)
class PathResult:
    """Per-path structured output for discrete transmitter-receiver links."""

    name: str
    num_tx: int
    num_rx: int
    num_tx_ant: int
    num_rx_ant: int
    max_num_paths: int
    num_time_steps: int
    max_depth: int
    tx_pos: tuple[float, float, float]
    tx_positions: FloatTensor
    rx_positions: FloatTensor
    frequency: float
    wavelength: float
    a: object
    tau: FloatTensor
    theta_t: FloatTensor
    phi_t: FloatTensor
    theta_r: FloatTensor
    phi_r: FloatTensor
    valid: BoolTensor
    types: IntTensor
    num_paths: IntTensor
    vertices: FloatTensor | None
    normals: FloatTensor | None
    objects: IntTensor | None
    metadata: Mapping[str, object]

    @property
    def path_shape(self) -> tuple[int, int, int, int, int]:
        return (self.num_rx, self.num_rx_ant, self.num_tx, self.num_tx_ant, self.max_num_paths)

    @property
    def coeff_shape(self) -> tuple[int, int, int, int, int, int]:
        return (*self.path_shape, self.num_time_steps)

    @property
    def path_count_shape(self) -> tuple[int, int, int, int]:
        return (self.num_rx, self.num_rx_ant, self.num_tx, self.num_tx_ant)

    @property
    def depth_shape(self) -> tuple[int, int, int, int, int, int]:
        return (*self.path_shape, self.max_depth)

    def coeff_tensor(self) -> torch.Tensor:
        return to_torch_view(self.a, dtype=torch.complex64).reshape(self.coeff_shape)

    @classmethod
    def _from_payload(cls, payload: Mapping[str, object]) -> "PathResult":
        num_rx = int(payload["num_rx"])
        num_tx = int(payload.get("num_tx", 1))
        num_rx_ant = int(payload.get("num_rx_ant", 1))
        num_tx_ant = int(payload.get("num_tx_ant", 1))
        num_time_steps = int(payload.get("num_time_steps", 1))
        max_num_paths = int(payload["max_num_paths"])
        max_depth = int(payload["max_depth"])
        path_shape = (num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths)
        coeff_shape = (*path_shape, num_time_steps)
        path_count_shape = (num_rx, num_rx_ant, num_tx, num_tx_ant)
        depth_shape = (*path_shape, max_depth)
        tx_positions_payload = payload.get("tx_positions")
        if tx_positions_payload is None:
            tx_positions_payload = [payload["tx_pos"]]

        def _opt(key, fn):
            value = payload.get(key)
            return None if value is None else fn(value)

        return cls(
            name=str(payload["name"]),
            num_tx=num_tx,
            num_rx=num_rx,
            num_tx_ant=num_tx_ant,
            num_rx_ant=num_rx_ant,
            max_num_paths=max_num_paths,
            num_time_steps=num_time_steps,
            max_depth=max_depth,
            tx_pos=tuple(float(v) for v in payload["tx_pos"]),
            tx_positions=to_vector_tensor(tx_positions_payload, component_shape=(num_tx,)),
            rx_positions=to_vector_tensor(payload["rx_positions"], component_shape=(num_rx,)),
            frequency=float(payload["frequency"]),
            wavelength=float(payload["wavelength"]),
            a=to_complex_array(payload["a"], shape=coeff_shape),
            tau=to_float_tensor(payload["tau"], shape=path_shape),
            theta_t=to_float_tensor(payload["theta_t"], shape=path_shape),
            phi_t=to_float_tensor(payload["phi_t"], shape=path_shape),
            theta_r=to_float_tensor(payload["theta_r"], shape=path_shape),
            phi_r=to_float_tensor(payload["phi_r"], shape=path_shape),
            valid=to_bool_tensor(payload["valid"], shape=path_shape),
            types=to_int_tensor(payload["types"], shape=depth_shape),
            num_paths=to_int_tensor(payload["num_paths"], shape=path_count_shape),
            vertices=_opt("vertices", lambda v: to_vector_tensor(v, component_shape=depth_shape)),
            normals=_opt("normals", lambda v: to_vector_tensor(v, component_shape=depth_shape)),
            objects=_opt("objects", lambda v: to_int_tensor(v, shape=depth_shape)),
            metadata=to_mapping_proxy(payload.get("metadata")),
        )

    def cir(self, *, normalize_delays: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        tau = to_torch_view(self.tau, dtype=torch.float32)
        valid = to_torch_view(self.valid, dtype=torch.bool)
        coeff = self.coeff_tensor()
        if normalize_delays:
            tau = tau - _masked_min(tau, valid)
        tau = torch.where(valid, tau, torch.full_like(tau, -1.0))
        coeff_valid = valid.unsqueeze(-1)
        return torch.where(coeff_valid, coeff, torch.zeros_like(coeff)), tau

    def cfr(self, frequencies: torch.Tensor, *, normalize_delays: bool = True) -> torch.Tensor:
        freq_tensor = to_torch_view(frequencies)
        coeff = self.coeff_tensor()
        valid = to_torch_view(self.valid, dtype=torch.bool)
        tau = to_torch_view(self.tau, dtype=torch.float32)
        freq = freq_tensor.to(
            device=coeff.device,
            dtype=freq_tensor.dtype if torch.is_complex(freq_tensor) else torch.float32,
        )
        if normalize_delays:
            tau = tau - _masked_min(tau, valid)
        tau = torch.where(valid, tau, torch.zeros_like(tau))
        coeff = torch.where(valid.unsqueeze(-1), coeff, torch.zeros_like(coeff))
        freq_shape = (1,) * (tau.ndim + 1) + (-1,)
        phase = -2.0j * math.pi * tau.unsqueeze(-1).unsqueeze(-1) * freq.reshape(freq_shape)
        return (coeff.unsqueeze(-1) * torch.exp(phase)).sum(dim=-3)

    def taps(self, bandwidth: float, num_taps: int, *, normalize_delays: bool = True) -> torch.Tensor:
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be > 0.")
        if int(num_taps) <= 0:
            raise ValueError("num_taps must be > 0.")
        coeff = self.coeff_tensor()
        tau = to_torch_view(self.tau, dtype=torch.float32)
        valid = to_torch_view(self.valid, dtype=torch.bool)
        if normalize_delays:
            tau = tau - _masked_min(tau, valid)
        tap_idx = torch.round(tau * float(bandwidth)).to(dtype=torch.int64)
        taps = torch.zeros((*self.path_count_shape, self.num_time_steps, int(num_taps)), device=coeff.device, dtype=coeff.dtype)
        if self.max_num_paths == 0:
            return taps
        keep = valid & (tap_idx >= 0) & (tap_idx < int(num_taps))
        if bool(keep.any()):
            flat_base_count = self.num_rx * self.num_rx_ant * self.num_tx * self.num_tx_ant
            coeff_flat = coeff.reshape(flat_base_count, self.max_num_paths, self.num_time_steps)
            tap_idx_flat = tap_idx.reshape(flat_base_count, self.max_num_paths)
            keep_flat = keep.reshape(flat_base_count, self.max_num_paths)
            base_idx = torch.arange(flat_base_count, device=coeff.device, dtype=torch.int64).view(-1, 1, 1)
            base_idx = base_idx.expand(-1, self.max_num_paths, self.num_time_steps)
            time_idx = torch.arange(self.num_time_steps, device=coeff.device, dtype=torch.int64).view(1, 1, -1)
            time_idx = time_idx.expand(flat_base_count, self.max_num_paths, -1)
            tap_idx_expanded = tap_idx_flat.unsqueeze(-1).expand(-1, -1, self.num_time_steps)
            keep_expanded = keep_flat.unsqueeze(-1).expand(-1, -1, self.num_time_steps)
            taps_flat = taps.reshape(flat_base_count, self.num_time_steps, int(num_taps))
            taps_flat.index_put_(
                (base_idx[keep_expanded], time_idx[keep_expanded], tap_idx_expanded[keep_expanded]),
                coeff_flat[keep_expanded],
                accumulate=True,
            )
        return taps

    def filter_by_type(self, *interaction_types: int) -> "PathResult":
        if len(interaction_types) == 0:
            return self
        coeff = self.coeff_tensor()
        tau = to_torch_view(self.tau, dtype=torch.float32)
        theta_t = to_torch_view(self.theta_t, dtype=torch.float32)
        phi_t = to_torch_view(self.phi_t, dtype=torch.float32)
        theta_r = to_torch_view(self.theta_r, dtype=torch.float32)
        phi_r = to_torch_view(self.phi_r, dtype=torch.float32)
        valid = to_torch_view(self.valid, dtype=torch.bool)
        types = to_torch_view(self.types, dtype=torch.int32)
        vertices = None if self.vertices is None else to_torch_view(self.vertices, dtype=torch.float32)
        normals = None if self.normals is None else to_torch_view(self.normals, dtype=torch.float32)
        objects = None if self.objects is None else to_torch_view(self.objects, dtype=torch.int32)

        type_values = {int(v) for v in interaction_types}
        non_empty = types != 0
        keep = torch.zeros_like(valid)
        if 0 in type_values:
            keep = keep | (valid & ~non_empty.any(dim=-1))
        non_zero_types = [v for v in type_values if v != 0]
        if non_zero_types:
            wanted = torch.zeros_like(valid)
            for value in non_zero_types:
                wanted = wanted | (types == int(value)).any(dim=-1)
            keep = keep | (valid & wanted)
        valid = valid & keep
        num_paths = valid.to(dtype=torch.int32).sum(dim=-1)
        compact_max_num_paths = max(1, int(num_paths.max().item()) if num_paths.numel() > 0 else 1)
        stable_order = torch.argsort(
            torch.where(
                valid,
                torch.zeros_like(valid, dtype=torch.int64),
                torch.ones_like(valid, dtype=torch.int64),
            ),
            dim=-1,
            stable=True,
        )

        def _gather(tensor: torch.Tensor) -> torch.Tensor:
            idx = stable_order
            while idx.ndim < tensor.ndim:
                idx = idx.unsqueeze(-1)
            idx = idx.expand(*stable_order.shape, *tensor.shape[stable_order.ndim:])
            path_dim = len(self.path_shape) - 1
            return torch.gather(tensor, path_dim, idx).narrow(path_dim, 0, compact_max_num_paths)

        compact_valid = _gather(valid)

        def _compact(tensor: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
            gathered = _gather(tensor)
            mask = compact_valid
            while mask.ndim < gathered.ndim:
                mask = mask.unsqueeze(-1)
            return torch.where(mask, gathered, torch.full_like(gathered, fill))

        return PathResult._from_payload({
            "name": self.name,
            "num_tx": self.num_tx,
            "num_rx": self.num_rx,
            "num_tx_ant": self.num_tx_ant,
            "num_rx_ant": self.num_rx_ant,
            "num_time_steps": self.num_time_steps,
            "max_num_paths": compact_max_num_paths,
            "max_depth": self.max_depth,
            "tx_pos": self.tx_pos,
            "tx_positions": self.tx_positions,
            "rx_positions": self.rx_positions,
            "frequency": self.frequency,
            "wavelength": self.wavelength,
            "a": _compact(coeff),
            "tau": _compact(tau, -1.0),
            "theta_t": _compact(theta_t),
            "phi_t": _compact(phi_t),
            "theta_r": _compact(theta_r),
            "phi_r": _compact(phi_r),
            "valid": compact_valid,
            "types": _compact(types),
            "num_paths": num_paths,
            "vertices": None if vertices is None else _compact(vertices),
            "normals": None if normals is None else _compact(normals),
            "objects": None if objects is None else _compact(objects, -1),
            "metadata": dict(self.metadata),
        })

    def to_dict(self) -> dict[str, object]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["metadata"] = dict(self.metadata)
        return data


__all__ = ["InteractionType", "PathResult"]
