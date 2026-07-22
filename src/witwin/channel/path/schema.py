from __future__ import annotations

from dataclasses import dataclass

import torch


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


__all__ = ["RaggedPathSoA"]
