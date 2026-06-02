from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from ...utils.tensor_conversion import (
    FloatTensor,
    IntTensor,
    to_complex_array,
    to_float_tensor,
    to_int_tensor,
    to_mapping_proxy,
    to_vector_tensor,
)
from ...utils.torch_bridge import drjit_to_torch_view

@dataclass(frozen=True)
class RadioMapCoordinates:
    """Coordinate payload for a traced radio-map monitor."""

    grid_x: FloatTensor
    grid_y: FloatTensor
    x: FloatTensor
    y: FloatTensor
    cell_centers: FloatTensor
    sample_positions_payload: tuple[FloatTensor, ...]
    axis_x: str = "u"
    axis_y: str = "v"

    @property
    def tangential_axes(self) -> tuple[str, str]:
        return (self.axis_x, self.axis_y)

    def sample_positions(self, sample_index: int | None = None):
        if sample_index is None:
            return self.sample_positions_payload
        return self.sample_positions_payload[int(sample_index)]


def _radio_map_complex_components(value: Mapping[str, object], *, shape: tuple[int, int]):
    return MappingProxyType(
        {
            str(key): to_complex_array(component, shape=shape)
            for key, component in dict(value).items()
        }
    )


def _radio_map_float_components(value: Mapping[str, object], *, shape: tuple[int, int]):
    return MappingProxyType(
        {
            str(key): to_float_tensor(component, shape=shape)
            for key, component in dict(value).items()
        }
    )


@dataclass(frozen=True)
class RadioMapResult:
    """Structured result payload for a traced radio-map monitor."""

    name: str
    kind: str
    metric: str
    combine_mode: str
    receiver_model: str
    grid_shape: tuple[int, int]
    cell_size: tuple[float, float]
    surface: Mapping[str, object]
    coords: RadioMapCoordinates
    path_gain: FloatTensor
    rss: FloatTensor
    sinr: FloatTensor
    tx_association_map: IntTensor
    coherent: Mapping[str, object]
    incoherent: Mapping[str, FloatTensor]
    coherent_power: Mapping[str, FloatTensor]
    metadata: Mapping[str, object]
    tx_pos: tuple[float, float, float]
    tx_power: float
    noise_power: float
    timing: Mapping[str, object] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RadioMapResult":
        grid_shape = tuple(int(value) for value in payload["grid_shape"])
        tensor_shape = (int(grid_shape[1]), int(grid_shape[0]))
        coords_payload = payload["coords"]
        metrics_payload = payload["metrics"]
        diagnostics_payload = payload.get("diagnostics", {})
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            metric=str(payload["metric"]),
            combine_mode=str(payload.get("combine_mode", payload.get("metadata", {}).get("combine_mode", "incoherent"))),
            receiver_model=str(
                payload.get(
                    "receiver_model",
                    payload.get("metadata", {}).get("receiver_model", "projected_polarized"),
                )
            ),
            grid_shape=grid_shape,
            cell_size=tuple(float(value) for value in payload["cell_size"]),
            surface=to_mapping_proxy(payload.get("surface")),
            coords=RadioMapCoordinates(
                grid_x=to_float_tensor(coords_payload["grid_x"], shape=tensor_shape),
                grid_y=to_float_tensor(coords_payload["grid_y"], shape=tensor_shape),
                x=to_float_tensor(coords_payload["x"], shape=(int(grid_shape[0]),)),
                y=to_float_tensor(coords_payload["y"], shape=(int(grid_shape[1]),)),
                cell_centers=to_vector_tensor(
                    coords_payload["cell_centers"],
                    component_shape=tensor_shape,
                ),
                sample_positions_payload=tuple(
                    to_vector_tensor(sample_positions, component_shape=tensor_shape)
                    for sample_positions in tuple(coords_payload.get("sample_positions", ()))
                ),
                axis_x=str(coords_payload.get("axis_x", "u")),
                axis_y=str(coords_payload.get("axis_y", "v")),
            ),
            path_gain=to_float_tensor(metrics_payload["path_gain"], shape=tensor_shape),
            rss=to_float_tensor(metrics_payload["rss"], shape=tensor_shape),
            sinr=to_float_tensor(metrics_payload["sinr"], shape=tensor_shape),
            tx_association_map=to_int_tensor(metrics_payload["tx_association"], shape=tensor_shape),
            coherent=_radio_map_complex_components(
                diagnostics_payload.get("coherent", {}),
                shape=tensor_shape,
            ),
            incoherent=_radio_map_float_components(
                diagnostics_payload.get("incoherent", {}),
                shape=tensor_shape,
            ),
            coherent_power=_radio_map_float_components(
                diagnostics_payload.get("coherent_power", {}),
                shape=tensor_shape,
            ),
            metadata=to_mapping_proxy(payload.get("metadata")),
            tx_pos=tuple(float(value) for value in payload["tx_pos"]),
            tx_power=float(payload["tx_power"]),
            noise_power=float(payload["noise_power"]),
            timing=None if payload.get("timing") is None else to_mapping_proxy(payload["timing"]),
        )

    @property
    def values(self):
        return self.metric_value(self.metric)

    @property
    def primary(self) -> "RadioMapResult":
        return self

    def metric_value(self, name: str) -> FloatTensor | IntTensor:
        return {
            "path_gain": self.path_gain,
            "rss": self.rss,
            "sinr": self.sinr,
            "tx_association": self.tx_association_map,
        }[str(name)]

    @property
    def tensor_shape(self) -> tuple[int, int]:
        return (int(self.grid_shape[1]), int(self.grid_shape[0]))

    @property
    def tangential_axes(self) -> tuple[str, str]:
        return self.coords.tangential_axes

    def sample_positions(self, sample_index: int | None = None):
        return self.coords.sample_positions(sample_index=sample_index)

    def metric_tensor(self, name: str | None = None) -> torch.Tensor:
        resolved_name = self.metric if name is None else str(name)
        value = self.metric_value(resolved_name)
        dtype = torch.int32 if resolved_name == "tx_association" else torch.float32
        return drjit_to_torch_view(value, dtype=dtype)

    def tx_association(self, *, as_names: bool = False):
        if not as_names:
            return self.tx_association_map
        labels = tuple(self.metadata.get("aggregate_tx_labels", ()))
        association = np.asarray(self.tx_association_map, dtype=np.int64)
        if len(labels) == 0:
            return association
        named = np.empty(association.shape, dtype=object)
        for tx_index, label in enumerate(labels):
            named[association == tx_index] = str(label)
        return named

    def _resolve_tx_association_index(self, selector: int | str) -> int:
        if isinstance(selector, str):
            labels = tuple(str(label) for label in self.metadata.get("aggregate_tx_labels", ()))
            if len(labels) == 0:
                raise ValueError(
                    "Radio-map result does not carry aggregate_tx_labels metadata for named association sampling."
                )
            try:
                return labels.index(str(selector))
            except ValueError as exc:
                raise ValueError(
                    f"Unknown transmitter association label '{selector}'."
                ) from exc
        resolved = int(selector)
        if resolved < 0:
            raise ValueError("tx_association must be >= 0.")
        return resolved

    def sample_metric_positions(
        self,
        count: int,
        *,
        metric: str | None = None,
        min_value: float | None = None,
        tx_association: int | str | None = None,
        replacement: bool = True,
        jitter: bool = True,
        seed: int | None = None,
        return_cell_indices: bool = False,
    ):
        resolved_count = int(count)
        if resolved_count <= 0:
            raise ValueError("count must be > 0.")

        weights = self.metric_tensor(name=metric).detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        eligible_mask = torch.isfinite(weights)
        if min_value is not None:
            eligible_mask = eligible_mask & (weights >= float(min_value))
        if tx_association is not None:
            association_index = self._resolve_tx_association_index(tx_association)
            association = self.metric_tensor(name="tx_association").detach().to(
                dtype=torch.int64,
                device="cpu",
            ).reshape(-1)
            eligible_mask = eligible_mask & (association == association_index)

        eligible = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
        if int(eligible.numel()) <= 0:
            raise ValueError("No radio-map cells satisfy the requested sampling filter.")
        if not replacement and resolved_count > int(eligible.numel()):
            raise ValueError("count exceeds the number of eligible radio-map cells when replacement=False.")

        selected_weights = weights.index_select(0, eligible).clamp_min(0.0)
        if not torch.any(selected_weights > 0.0):
            selected_weights = torch.ones_like(selected_weights)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))

        selected_slots = torch.multinomial(
            selected_weights,
            resolved_count,
            replacement=replacement,
            generator=generator,
        )
        flat_indices_cpu = eligible.index_select(0, selected_slots)

        cell_centers = drjit_to_torch_view(self.coords.cell_centers, dtype=torch.float32).reshape(-1, 3)
        flat_indices = flat_indices_cpu.to(device=cell_centers.device, dtype=torch.int64)
        positions = cell_centers.index_select(0, flat_indices)
        if jitter:
            offsets = torch.rand(
                (resolved_count, 2),
                generator=generator,
                dtype=torch.float32,
            ) - 0.5
            offsets[:, 0] *= float(self.cell_size[0])
            offsets[:, 1] *= float(self.cell_size[1])
            offsets = offsets.to(device=cell_centers.device)
            basis_u = torch.tensor(
                self.surface.get("basis_u", (1.0, 0.0, 0.0)),
                dtype=torch.float32,
                device=cell_centers.device,
            ).reshape(1, 3)
            basis_v = torch.tensor(
                self.surface.get("basis_v", (0.0, 1.0, 0.0)),
                dtype=torch.float32,
                device=cell_centers.device,
            ).reshape(1, 3)
            positions = positions + offsets[:, 0:1] * basis_u + offsets[:, 1:2] * basis_v

        if not return_cell_indices:
            return positions

        nx = int(self.grid_shape[0])
        flat_indices_long = flat_indices_cpu.to(dtype=torch.int64)
        cell_indices = torch.stack(
            (
                torch.div(flat_indices_long, nx, rounding_mode="floor"),
                torch.remainder(flat_indices_long, nx),
            ),
            dim=-1,
        )
        return positions, cell_indices

    def with_metric_overrides(
        self,
        *,
        sinr=None,
        tx_association=None,
        metadata: Mapping[str, object] | None = None,
    ) -> "RadioMapResult":
        resolved_metadata = self.metadata if metadata is None else to_mapping_proxy(metadata)
        return replace(
            self,
            sinr=self.sinr if sinr is None else to_float_tensor(sinr, shape=self.tensor_shape),
            tx_association_map=(
                self.tx_association_map
                if tx_association is None
                else to_int_tensor(tx_association, shape=self.tensor_shape)
            ),
            metadata=resolved_metadata,
        )



__all__ = [
    "RadioMapCoordinates",
    "RadioMapResult",
]
