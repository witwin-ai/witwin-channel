"""Shared radiomap result containers for standalone channel solvers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import drjit as dr
import torch

from witwin.channel.core.numerics.tensors import (
    FloatTensor,
    to_torch_view,
    to_float_tensor,
    to_int_tensor,
    to_mapping_proxy,
    to_vector_tensor,
)


@dataclass(frozen=True)
class RadioMapCoordinates:
    """Coordinate payload shared by deterministic and Monte Carlo radiomaps."""

    grid_x: FloatTensor
    grid_y: FloatTensor
    x: FloatTensor
    y: FloatTensor
    cell_centers: FloatTensor
    sample_positions: tuple[FloatTensor, ...]
    axis_x: str = "u"
    axis_y: str = "v"


@dataclass(frozen=True)
class RadioMapFieldPayload:
    """Coherent field payload for deterministic radiomap results."""

    vector_coherent: Mapping[str, object]


@dataclass(frozen=True)
class RadioMapPowerPayload:
    """Power-domain diagnostic payload for Monte Carlo radiomap results."""

    incoherent: Mapping[str, FloatTensor]
    coherent_power: Mapping[str, FloatTensor]
    coherent: Mapping[str, object] | None = None


@dataclass(frozen=True)
class RadioMapResult:
    """Common public payload returned by standalone radiomap solvers."""

    name: str
    kind: str
    metric: str
    solver: str
    grid_shape: tuple[int, int]
    cell_size: tuple[float, float]
    surface: Mapping[str, object]
    coords: RadioMapCoordinates
    path_gain: FloatTensor
    rss: FloatTensor
    sinr: FloatTensor
    best_tx_index: object | None
    tx_pos: tuple[float, float, float]
    tx_power: float
    noise_power: float
    metadata: Mapping[str, object]
    field: RadioMapFieldPayload | None = None
    power: RadioMapPowerPayload | None = None
    components: Mapping[str, FloatTensor] | None = None
    combine_mode: str | None = None
    receiver_model: str | None = None
    tx_association_map: object | None = None
    coherent: Mapping[str, object] | None = None
    incoherent: Mapping[str, FloatTensor] | None = None
    coherent_power: Mapping[str, FloatTensor] | None = None
    timing: Mapping[str, object] | None = None

    def cell_association(self, metric: str = "rss"):
        """Return the strongest transmitter index per cell for ``metric``."""
        if not hasattr(self, metric):
            raise ValueError(f"Unknown radio map metric {metric!r}.")
        values = to_torch_view(getattr(self, metric), dtype=torch.float32)
        if values.ndim == 2:
            best = torch.zeros_like(values, dtype=torch.int32)
        elif values.ndim == 3:
            best = torch.argmax(values, dim=0).to(dtype=torch.int32)
        else:
            raise ValueError(f"Radio map metric {metric!r} must be 2D or 3D, got shape {tuple(values.shape)}.")
        return to_int_tensor(best, shape=tuple(int(v) for v in best.shape))

    def squeeze_tx(self, index: int = 0) -> "RadioMapResult":
        """Return a single-TX view with the leading transmitter axis removed."""
        tx_index = int(index)
        tensor_shape = (int(self.grid_shape[1]), int(self.grid_shape[0]))

        def _squeeze_float(value):
            tensor = to_torch_view(value, dtype=torch.float32)
            if tensor.ndim == 3:
                tensor = tensor[tx_index]
            return to_float_tensor(tensor, shape=tensor_shape)

        def _squeeze_mapping(mapping):
            if mapping is None:
                return None
            return to_mapping_proxy({str(name): _squeeze_float(value) for name, value in dict(mapping).items()})

        squeezed_incoherent = _squeeze_mapping(self.incoherent)
        squeezed_coherent_power = _squeeze_mapping(self.coherent_power)
        single_tx_association = to_int_tensor(
            torch.zeros(tensor_shape, dtype=torch.int32),
            shape=tensor_shape,
        )
        squeezed_power = None if self.power is None else replace(
            self.power,
            incoherent=squeezed_incoherent,
            coherent_power=squeezed_coherent_power,
        )

        return replace(
            self,
            path_gain=_squeeze_float(self.path_gain),
            rss=_squeeze_float(self.rss),
            sinr=_squeeze_float(self.sinr),
            best_tx_index=single_tx_association,
            tx_association_map=single_tx_association,
            components=_squeeze_mapping(self.components),
            incoherent=squeezed_incoherent,
            coherent_power=squeezed_coherent_power,
            power=squeezed_power,
        )


def _stack_metric(results: tuple[RadioMapResult, ...], attr: str) -> FloatTensor:
    tensors = []
    for result in results:
        tensor = getattr(result, attr)
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) == 2:
            tensor = dr.reshape(FloatTensor, tensor, (1, *shape))
        tensors.append(tensor)
    return dr.concat(tensors)


def stack_radiomap_results(results, *, noise_power: float) -> RadioMapResult:
    """Stack single-transmitter radiomap results into the public multi-TX contract."""
    resolved = tuple(results)
    if not resolved:
        raise ValueError("At least one radiomap result is required.")
    first = resolved[0]
    for result in resolved[1:]:
        if result.grid_shape != first.grid_shape:
            raise ValueError("Cannot stack radiomap results with different grid shapes.")
        if result.cell_size != first.cell_size:
            raise ValueError("Cannot stack radiomap results with different cell sizes.")

    tensor_shape = (len(resolved), int(first.grid_shape[1]), int(first.grid_shape[0]))
    path_gain = _stack_metric(resolved, "path_gain")
    rss = _stack_metric(resolved, "rss")
    total_rss = dr.sum(rss, axis=0)
    denominator = float(noise_power) + total_rss - rss
    sinr = dr.select(
        denominator > 0.0,
        rss / denominator,
        dr.full(FloatTensor, float("inf"), tensor_shape),
    )
    best = torch.argmax(to_torch_view(rss, dtype=torch.float32), dim=0).to(dtype=torch.int32)

    metadata = dict(first.metadata)
    metadata["transmitter_count"] = int(len(resolved))
    metadata["multi_tx_sinr"] = "rss_i / (noise + sum_j!=i rss_j)"

    return replace(
        first,
        path_gain=to_float_tensor(path_gain, shape=tensor_shape),
        rss=to_float_tensor(rss, shape=tensor_shape),
        sinr=to_float_tensor(sinr, shape=tensor_shape),
        best_tx_index=to_int_tensor(best, shape=(int(first.grid_shape[1]), int(first.grid_shape[0]))),
        tx_association_map=to_int_tensor(best, shape=(int(first.grid_shape[1]), int(first.grid_shape[0]))),
        noise_power=float(noise_power),
        metadata=to_mapping_proxy(metadata),
    )


def coordinates_from_grid(grid, *, sample_positions: tuple[object, ...] | None = None) -> RadioMapCoordinates:
    """Convert a resolved solver grid into the shared coordinate payload."""
    grid_shape = (int(grid.grid_shape[0]), int(grid.grid_shape[1]))
    tensor_shape = (grid_shape[1], grid_shape[0])
    positions = tuple(sample_positions or ())
    return RadioMapCoordinates(
        grid_x=to_float_tensor(grid.grid_x, shape=tensor_shape),
        grid_y=to_float_tensor(grid.grid_y, shape=tensor_shape),
        x=to_float_tensor(grid.x_coords, shape=(grid_shape[0],)),
        y=to_float_tensor(grid.y_coords, shape=(grid_shape[1],)),
        cell_centers=to_vector_tensor(grid.cell_centers, component_shape=tensor_shape),
        sample_positions=tuple(
            to_vector_tensor(position, component_shape=tensor_shape)
            for position in positions
        ),
        axis_x=str(grid.tangential_axes[0]),
        axis_y=str(grid.tangential_axes[1]),
    )


__all__ = [
    "RadioMapCoordinates",
    "RadioMapFieldPayload",
    "RadioMapPowerPayload",
    "RadioMapResult",
    "coordinates_from_grid",
    "stack_radiomap_results",
    "to_mapping_proxy",
]
