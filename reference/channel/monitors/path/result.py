from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import drjit as dr
import numpy as np
import torch
import witwin as wt

from ...types import InteractionType
from ...utils.tensor_conversion import (
    BoolTensor,
    FloatTensor,
    IntTensor,
    to_bool_tensor,
    to_complex_array,
    to_float_tensor,
    to_int_tensor,
    to_mapping_proxy,
    to_vector_tensor,
)
from ...utils.torch_bridge import drjit_to_torch_view

_PATH_RESULT_REPLAY_CHUNK_SIZE = 4096


def _torch_tensor(value, *, dtype=None, device=None, detach: bool = False) -> torch.Tensor:
    wants_complex = dtype is not None and torch.is_complex(torch.empty((), dtype=dtype))
    if isinstance(value, torch.Tensor):
        tensor = value.detach() if detach else value
    else:
        try:
            x, y, z = value.x, value.y, value.z
        except Exception:
            if wants_complex:
                try:
                    real, imag = value.real, value.imag
                except Exception:
                    tensor = drjit_to_torch_view(value, detach=detach, device=device)
                else:
                    real_t = drjit_to_torch_view(
                        real,
                        detach=detach,
                        dtype=torch.float32,
                        device=device,
                    )
                    imag_t = drjit_to_torch_view(
                        imag,
                        detach=detach,
                        dtype=torch.float32,
                        device=device,
                    )
                    tensor = torch.complex(real_t, imag_t)
            else:
                tensor = drjit_to_torch_view(value, detach=detach, device=device)
        else:
            tensor = torch.stack(
                [
                    drjit_to_torch_view(x, detach=detach, dtype=torch.float32, device=device),
                    drjit_to_torch_view(y, detach=detach, dtype=torch.float32, device=device),
                    drjit_to_torch_view(z, detach=detach, dtype=torch.float32, device=device),
                ],
                dim=-1,
            )
    if device is not None or dtype is not None:
        tensor = tensor.to(
            device=device if device is not None else tensor.device,
            dtype=dtype if dtype is not None else tensor.dtype,
        )
    return tensor


def _masked_min(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.shape[1] == 0:
        return torch.zeros((values.shape[0], 1), device=values.device, dtype=values.dtype)
    inf = torch.full_like(values, float("inf"))
    masked = torch.where(valid, values, inf)
    minima = masked.min(dim=1, keepdim=True).values
    return torch.where(torch.isfinite(minima), minima, torch.zeros_like(minima))


def _group_ranks_and_counts(
    sorted_group_ids: torch.Tensor,
    *,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_num_groups = max(0, int(num_groups))
    if sorted_group_ids.numel() == 0:
        return (
            torch.zeros((0,), device=sorted_group_ids.device, dtype=torch.int64),
            torch.zeros((resolved_num_groups,), device=sorted_group_ids.device, dtype=torch.int64),
        )
    counts = torch.bincount(sorted_group_ids, minlength=resolved_num_groups)
    start_idx = torch.cumsum(counts, dim=0) - counts
    ranks = (
        torch.arange(sorted_group_ids.shape[0], device=sorted_group_ids.device, dtype=torch.int64)
        - start_idx.index_select(0, sorted_group_ids)
    )
    return ranks, counts


def _group_ranks(sorted_group_ids: torch.Tensor, *, num_groups: int) -> torch.Tensor:
    return _group_ranks_and_counts(sorted_group_ids, num_groups=num_groups)[0]


def _raw_path_payload_kind(raw: Mapping[str, object]) -> str:
    return str(raw.get("payload_kind", ""))


def _chunk_slices(length: int, chunk_size: int):
    resolved_chunk_size = max(1, int(chunk_size))
    for start in range(0, int(length), resolved_chunk_size):
        yield slice(start, min(int(length), start + resolved_chunk_size))


def _torch_path_indices(indices: torch.Tensor):
    if indices.numel() == 0:
        return dr.zeros(wt.UInt32, 0)
    return wt.UInt32(indices.detach().cpu().numpy().astype(np.uint32, copy=False))


def _gather_raw_path_value(value, indices):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_gather_raw_path_value(item, indices) for item in value)
    if isinstance(value, list):
        return [_gather_raw_path_value(item, indices) for item in value]
    return dr.gather(type(value), value, indices)


def _take_dense_raw_path_collection(raw: Mapping[str, object], path_indices) -> Mapping[str, object]:
    taken = dict(raw)
    for key in ("rx_index", "a", "tau", "theta_t", "phi_t", "theta_r", "phi_r"):
        if key in raw:
            taken[key] = _gather_raw_path_value(raw[key], path_indices)
    for key in ("type_slots", "vertex_slots", "normal_slots", "object_slots"):
        value = raw.get(key)
        if value is None:
            taken[key] = None
        else:
            taken[key] = tuple(_gather_raw_path_value(slot, path_indices) for slot in value)
    taken["metadata"] = dict(raw.get("metadata", {}))
    return taken


def _summarize_raw_path_collection(
    raw: Mapping[str, object],
    *,
    device,
) -> dict[str, object]:
    rx_index = _torch_tensor(raw["rx_index"], dtype=torch.int64, device=device, detach=True).reshape(-1)
    payload_kind = _raw_path_payload_kind(raw)
    type_slots = tuple(raw.get("type_slots") or ())
    summary = {
        "raw": raw,
        "payload_kind": payload_kind,
        "rx_index": rx_index,
        "a": _torch_tensor(raw["a"], dtype=torch.complex64, device=device, detach=False).reshape(-1),
        "tau": _torch_tensor(raw["tau"], dtype=torch.float32, device=device, detach=False).reshape(-1),
        "count": int(rx_index.shape[0]),
        "depth_hint": (
            int(raw.get("max_depth_hint", 1))
            if payload_kind == "reflection_path_refs_v1"
            else (1 if payload_kind == "diffraction_state_refs_v1" else max(1, len(type_slots)))
        ),
    }
    if (
        payload_kind not in {"diffraction_state_refs_v1", "reflection_path_refs_v1"}
        and all(key in raw for key in ("theta_t", "phi_t", "theta_r", "phi_r"))
    ):
        summary["direct_no_geometry_kind"] = "materialized_dense"
        summary["theta_t"] = _torch_tensor(
            raw["theta_t"],
            dtype=torch.float32,
            device=device,
            detach=False,
        ).reshape(-1)
        summary["phi_t"] = _torch_tensor(
            raw["phi_t"],
            dtype=torch.float32,
            device=device,
            detach=False,
        ).reshape(-1)
        summary["theta_r"] = _torch_tensor(
            raw["theta_r"],
            dtype=torch.float32,
            device=device,
            detach=False,
        ).reshape(-1)
        summary["phi_r"] = _torch_tensor(
            raw["phi_r"],
            dtype=torch.float32,
            device=device,
            detach=False,
        ).reshape(-1)
        if len(type_slots) == 0:
            summary["types_torch"] = torch.zeros((int(rx_index.shape[0]), 1), device=device, dtype=torch.int32)
        else:
            summary["types_torch"] = torch.stack(
                [
                    _torch_tensor(slot, dtype=torch.int32, device=device, detach=True).reshape(-1)
                    for slot in type_slots
                ],
                dim=1,
            )
    elif (
        payload_kind == "reflection_path_refs_v1"
        and raw.get("theta_t") is not None
        and raw.get("phi_t") is not None
        and raw.get("theta_r") is not None
        and raw.get("phi_r") is not None
        and raw.get("path_depth") is not None
    ):
        summary["direct_no_geometry_kind"] = "reflection_cached"
        summary["theta_t"] = _torch_tensor(
            raw["theta_t"],
            dtype=torch.float32,
            device=device,
            detach=False,
        ).reshape(-1)
        summary["phi_t"] = _torch_tensor(
            raw["phi_t"],
            dtype=torch.float32,
            device=device,
            detach=False,
        ).reshape(-1)
        summary["theta_r"] = _torch_tensor(
            raw["theta_r"],
            dtype=torch.float32,
            device=device,
            detach=False,
        ).reshape(-1)
        summary["phi_r"] = _torch_tensor(
            raw["phi_r"],
            dtype=torch.float32,
            device=device,
            detach=False,
        ).reshape(-1)
        summary["path_depth_torch"] = _torch_tensor(
            raw["path_depth"],
            dtype=torch.int32,
            device=device,
            detach=True,
        ).reshape(-1)
    return summary


def _summary_has_direct_no_geometry_tensors(summary: Mapping[str, object]) -> bool:
    return summary.get("direct_no_geometry_kind") is not None


def _build_direct_no_geometry_chunk(
    summary: Mapping[str, object],
    selected_paths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(selected_paths.shape[0])
    a = summary["a"].index_select(0, selected_paths)
    tau = summary["tau"].index_select(0, selected_paths)
    theta_t = summary["theta_t"].index_select(0, selected_paths)
    phi_t = summary["phi_t"].index_select(0, selected_paths)
    theta_r = summary["theta_r"].index_select(0, selected_paths)
    phi_r = summary["phi_r"].index_select(0, selected_paths)
    packed_chunk = torch.cat(
        [
            torch.view_as_real(a).reshape(count, 2),
            tau.reshape(count, 1),
            theta_t.reshape(count, 1),
            phi_t.reshape(count, 1),
            theta_r.reshape(count, 1),
            phi_r.reshape(count, 1),
        ],
        dim=1,
    )
    direct_kind = str(summary["direct_no_geometry_kind"])
    if direct_kind == "materialized_dense":
        types = summary["types_torch"].index_select(0, selected_paths)
        return packed_chunk, types
    if direct_kind == "reflection_cached":
        path_depth = summary["path_depth_torch"].index_select(0, selected_paths)
        depth = max(1, int(path_depth.max().item()) if count > 0 else 1)
        depth_index = torch.arange(depth, device=path_depth.device, dtype=path_depth.dtype)
        reflection_code = torch.full(
            (count, depth),
            int(InteractionType.REFLECTION),
            device=path_depth.device,
            dtype=torch.int32,
        )
        none_code = torch.zeros((count, depth), device=path_depth.device, dtype=torch.int32)
        types = torch.where(path_depth.unsqueeze(1) > depth_index.unsqueeze(0), reflection_code, none_code)
        return packed_chunk, types
    raise ValueError(f"Unsupported direct no-geometry summary kind: {direct_kind}")


def _resolve_selected_path_indices(
    *,
    cat_rx_index: torch.Tensor,
    cat_a: torch.Tensor,
    cat_tau: torch.Tensor,
    num_rx: int,
    max_num_paths: int | None,
    resolved_max_num_paths: int,
) -> torch.Tensor:
    total_paths = int(cat_a.shape[0])
    device = cat_a.device
    if total_paths <= 0 or resolved_max_num_paths <= 0:
        return torch.zeros((0,), device=device, dtype=torch.int64)
    if max_num_paths is None:
        tau_order = torch.argsort(cat_tau, stable=True)
        return tau_order.index_select(
            0,
            torch.argsort(cat_rx_index.index_select(0, tau_order), stable=True),
        )

    strength_order = torch.argsort(torch.abs(cat_a), descending=True, stable=True)
    grouped_by_strength = strength_order.index_select(
        0,
        torch.argsort(cat_rx_index.index_select(0, strength_order), stable=True),
    )
    grouped_rx = cat_rx_index.index_select(0, grouped_by_strength).to(dtype=torch.int64)
    strength_rank = _group_ranks(grouped_rx, num_groups=num_rx)
    selected_idx = grouped_by_strength[strength_rank < resolved_max_num_paths]
    if int(selected_idx.numel()) <= 0:
        return selected_idx
    tau_order = torch.argsort(cat_tau.index_select(0, selected_idx), stable=True)
    selected_idx = selected_idx.index_select(0, tau_order)
    return selected_idx.index_select(
        0,
        torch.argsort(cat_rx_index.index_select(0, selected_idx), stable=True),
    )


def _build_active_summary_selections(
    summaries: list[Mapping[str, object]],
    *,
    selected_idx: torch.Tensor,
    flat_slots: torch.Tensor,
    device,
) -> list[tuple[Mapping[str, object], torch.Tensor, torch.Tensor]]:
    if len(summaries) == 0 or int(selected_idx.numel()) == 0:
        return []
    active_summaries = [
        summary
        for summary in summaries
        if int(summary["count"]) > 0
    ]
    if len(active_summaries) == 0:
        return []
    if len(active_summaries) == 1:
        summary = active_summaries[0]
        return [
            (
                summary,
                selected_idx - int(summary["offset"]),
                flat_slots,
            )
        ]

    summary_offsets = torch.tensor(
        [int(summary["offset"]) for summary in active_summaries],
        device=device,
        dtype=torch.int64,
    )
    selected_summary = torch.bucketize(
        selected_idx,
        summary_offsets[1:],
        right=True,
    )
    local_selected = selected_idx - summary_offsets.index_select(0, selected_summary)
    summary_order = torch.argsort(selected_summary, stable=True)
    grouped_summary = selected_summary.index_select(0, summary_order)
    grouped_local_selected = local_selected.index_select(0, summary_order)
    grouped_flat_slots = flat_slots.index_select(0, summary_order)
    grouped_counts = torch.bincount(grouped_summary, minlength=len(active_summaries))
    grouped_offsets = torch.cumsum(grouped_counts, dim=0) - grouped_counts

    active_selections = []
    for summary_idx, summary in enumerate(active_summaries):
        count = int(grouped_counts[summary_idx].item())
        if count <= 0:
            continue
        start = int(grouped_offsets[summary_idx].item())
        end = start + count
        active_selections.append(
            (
                summary,
                grouped_local_selected[start:end],
                grouped_flat_slots[start:end],
            )
        )
    return active_selections


def _iter_selected_path_chunks(
    local_selected: torch.Tensor,
    flat_slots: torch.Tensor,
):
    total = int(local_selected.shape[0])
    for chunk_slice in _chunk_slices(total, _PATH_RESULT_REPLAY_CHUNK_SIZE):
        yield local_selected[chunk_slice], flat_slots[chunk_slice]


def _selected_diffraction_ref_max_depth(
    raw: Mapping[str, object],
    path_indices: torch.Tensor,
    *,
    device,
) -> int:
    if path_indices.numel() == 0:
        return 1
    state_arrays = raw["state_arrays"]
    max_depth = 1
    for chunk_slice in _chunk_slices(int(path_indices.numel()), _PATH_RESULT_REPLAY_CHUNK_SIZE):
        selected_path_idx = _torch_path_indices(path_indices[chunk_slice])
        selected_state_idx = dr.gather(wt.UInt32, raw["state_idx"], selected_path_idx)
        prefix_depth = _torch_tensor(
            dr.gather(wt.UInt32, state_arrays["prefix_reflection_depth"], selected_state_idx),
            dtype=torch.int32,
            device=device,
            detach=True,
        ).reshape(-1)
        intermediate_depth = _torch_tensor(
            dr.gather(wt.UInt32, state_arrays["intermediate_reflection_depth"], selected_state_idx),
            dtype=torch.int32,
            device=device,
            detach=True,
        ).reshape(-1)
        suffix_depth = _torch_tensor(
            dr.gather(wt.UInt32, state_arrays["suffix_reflection_depth"], selected_state_idx),
            dtype=torch.int32,
            device=device,
            detach=True,
        ).reshape(-1)
        order = _torch_tensor(
            dr.gather(wt.UInt32, state_arrays["order"], selected_state_idx),
            dtype=torch.int32,
            device=device,
            detach=True,
        ).reshape(-1)
        if order.numel() == 0:
            continue
        total_depth = prefix_depth + intermediate_depth + suffix_depth + order
        max_depth = max(max_depth, int(total_depth.max().item()))
    return max_depth


def _selected_reflection_ref_max_depth(
    raw: Mapping[str, object],
    path_indices: torch.Tensor,
    *,
    device,
) -> int:
    if path_indices.numel() == 0:
        return 1
    max_depth_hint = raw.get("max_depth_hint")
    if max_depth_hint is not None:
        return max(1, int(max_depth_hint))
    path_depth = raw.get("path_depth")
    if path_depth is not None:
        selected_path_idx = _torch_path_indices(path_indices)
        selected_depth = _torch_tensor(
            dr.gather(wt.UInt32, path_depth, selected_path_idx),
            dtype=torch.int32,
            device=device,
            detach=True,
        ).reshape(-1)
        if selected_depth.numel() == 0:
            return 1
        return max(1, int(selected_depth.max().item()))
    from ...trace.materials import coerce_reflection_trace_detail

    detail = coerce_reflection_trace_detail(raw["reflection_detail"])
    depth_lookup = torch.tensor(
        [
            int(paths.get("chain_depth", 0)) if paths is not None else 0
            for paths in detail.source_paths_per_bounce
        ],
        device=device,
        dtype=torch.int32,
    )
    if depth_lookup.numel() == 0:
        return 1
    selected_path_idx = _torch_path_indices(path_indices)
    selected_group_idx = _torch_tensor(
        dr.gather(wt.UInt32, raw["path_group_index"], selected_path_idx),
        dtype=torch.int64,
        device=device,
        detach=True,
    ).reshape(-1)
    if selected_group_idx.numel() == 0:
        return 1
    selected_depth = depth_lookup.index_select(0, selected_group_idx)
    return max(1, int(selected_depth.max().item()))


def _selected_summary_max_depth(
    summary: Mapping[str, object],
    path_indices: torch.Tensor,
    *,
    device,
) -> int:
    if int(path_indices.numel()) == 0:
        return 1
    cached_path_depth = summary.get("path_depth_torch")
    if cached_path_depth is not None:
        selected_depth = cached_path_depth.index_select(0, path_indices)
        if selected_depth.numel() == 0:
            return 1
        return max(1, int(selected_depth.max().item()))
    if str(summary["payload_kind"]) == "diffraction_state_refs_v1":
        return _selected_diffraction_ref_max_depth(summary["raw"], path_indices, device=device)
    if str(summary["payload_kind"]) == "reflection_path_refs_v1":
        return _selected_reflection_ref_max_depth(summary["raw"], path_indices, device=device)
    return int(summary["depth_hint"])


def _materialize_raw_path_collection(
    raw: Mapping[str, object],
    *,
    return_geometry: bool,
    path_indices: torch.Tensor | None = None,
) -> Mapping[str, object]:
    payload_kind = str(raw.get("payload_kind", ""))
    if payload_kind == "diffraction_state_refs_v1":
        from .collectors import _materialize_diffraction_state_path_refs

        return _materialize_diffraction_state_path_refs(
            raw,
            return_geometry=return_geometry,
            path_indices=None if path_indices is None else _torch_path_indices(path_indices),
        )
    if payload_kind == "reflection_path_refs_v1":
        from .collectors import _materialize_reflection_path_refs

        return _materialize_reflection_path_refs(
            raw,
            return_geometry=return_geometry,
            path_indices=None if path_indices is None else _torch_path_indices(path_indices),
        )
    if path_indices is not None:
        raw = _take_dense_raw_path_collection(raw, _torch_path_indices(path_indices))
    return raw


def _normalize_raw_path_collection(
    raw: Mapping[str, object],
    *,
    device,
    return_geometry: bool,
    path_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    raw = _materialize_raw_path_collection(
        raw,
        return_geometry=return_geometry,
        path_indices=path_indices,
    )
    rx_index = _torch_tensor(raw["rx_index"], dtype=torch.int64, device=device, detach=True).reshape(-1)
    count = int(rx_index.shape[0])
    type_slots = tuple(raw.get("type_slots") or ())
    if len(type_slots) == 0:
        depth = 1
        types = torch.zeros((count, 1), device=device, dtype=torch.int32)
    else:
        depth = len(type_slots)
        types = torch.stack(
            [
                _torch_tensor(slot, dtype=torch.int32, device=device, detach=True).reshape(-1)
                for slot in type_slots
            ],
            dim=1,
        )

    normalized = {
        "rx_index": rx_index,
        "a": _torch_tensor(raw["a"], dtype=torch.complex64, device=device, detach=False).reshape(-1),
        "tau": _torch_tensor(raw["tau"], dtype=torch.float32, device=device, detach=False).reshape(-1),
        "theta_t": _torch_tensor(raw["theta_t"], dtype=torch.float32, device=device, detach=False).reshape(-1),
        "phi_t": _torch_tensor(raw["phi_t"], dtype=torch.float32, device=device, detach=False).reshape(-1),
        "theta_r": _torch_tensor(raw["theta_r"], dtype=torch.float32, device=device, detach=False).reshape(-1),
        "phi_r": _torch_tensor(raw["phi_r"], dtype=torch.float32, device=device, detach=False).reshape(-1),
        "types": types,
        "vertices": None,
        "normals": None,
        "objects": None,
        "metadata": dict(raw.get("metadata", {})),
    }
    if not return_geometry:
        return normalized

    vertex_slots = raw.get("vertex_slots")
    normal_slots = raw.get("normal_slots")
    object_slots = raw.get("object_slots")
    if vertex_slots is None or normal_slots is None or object_slots is None:
        normalized["vertices"] = torch.zeros((count, depth, 3), device=device, dtype=torch.float32)
        normalized["normals"] = torch.zeros((count, depth, 3), device=device, dtype=torch.float32)
        normalized["objects"] = torch.full((count, depth), -1, device=device, dtype=torch.int32)
        return normalized

    normalized["vertices"] = torch.stack(
        [
            _torch_tensor(slot, dtype=torch.float32, device=device, detach=True).reshape(count, 3)
            for slot in vertex_slots
        ],
        dim=1,
    )
    normalized["normals"] = torch.stack(
        [
            _torch_tensor(slot, dtype=torch.float32, device=device, detach=True).reshape(count, 3)
            for slot in normal_slots
        ],
        dim=1,
    )
    normalized["objects"] = torch.stack(
        [
            _torch_tensor(slot, dtype=torch.int32, device=device, detach=True).reshape(-1)
            for slot in object_slots
        ],
        dim=1,
    )
    return normalized


@dataclass(frozen=True)
class PathResult:
    """Per-path structured output for a set of discrete receivers."""

    name: str
    num_rx: int
    max_num_paths: int
    max_depth: int
    tx_pos: tuple[float, float, float]
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
    def path_shape(self) -> tuple[int, int]:
        return (self.num_rx, self.max_num_paths)

    @property
    def depth_shape(self) -> tuple[int, int, int]:
        return (self.num_rx, self.max_num_paths, self.max_depth)

    def coeff_tensor(self) -> torch.Tensor:
        return drjit_to_torch_view(self.a, dtype=torch.complex64).reshape(self.path_shape)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PathResult":
        num_rx = int(payload["num_rx"])
        max_num_paths = int(payload["max_num_paths"])
        max_depth = int(payload["max_depth"])
        path_shape = (num_rx, max_num_paths)
        depth_shape = (num_rx, max_num_paths, max_depth)
        return cls(
            name=str(payload["name"]),
            num_rx=num_rx,
            max_num_paths=max_num_paths,
            max_depth=max_depth,
            tx_pos=tuple(float(value) for value in payload["tx_pos"]),
            rx_positions=to_vector_tensor(payload["rx_positions"], component_shape=(num_rx,)),
            frequency=float(payload["frequency"]),
            wavelength=float(payload["wavelength"]),
            a=to_complex_array(payload["a"], shape=path_shape),
            tau=to_float_tensor(payload["tau"], shape=path_shape),
            theta_t=to_float_tensor(payload["theta_t"], shape=path_shape),
            phi_t=to_float_tensor(payload["phi_t"], shape=path_shape),
            theta_r=to_float_tensor(payload["theta_r"], shape=path_shape),
            phi_r=to_float_tensor(payload["phi_r"], shape=path_shape),
            valid=to_bool_tensor(payload["valid"], shape=path_shape),
            types=to_int_tensor(payload["types"], shape=depth_shape),
            num_paths=to_int_tensor(payload["num_paths"], shape=(num_rx,)),
            vertices=(
                None
                if payload.get("vertices") is None
                else to_vector_tensor(payload["vertices"], component_shape=depth_shape)
            ),
            normals=(
                None
                if payload.get("normals") is None
                else to_vector_tensor(payload["normals"], component_shape=depth_shape)
            ),
            objects=(
                None
                if payload.get("objects") is None
                else to_int_tensor(payload["objects"], shape=depth_shape)
            ),
            metadata=to_mapping_proxy(payload.get("metadata")),
        )

    @classmethod
    def from_raw_collections(
        cls,
        *,
        name: str,
        num_rx: int,
        max_num_paths: int | None,
        tx_pos: tuple[float, float, float],
        rx_positions,
        frequency: float,
        wavelength: float,
        raw_collections: list[Mapping[str, object]],
        return_geometry: bool,
        metadata: Mapping[str, object] | None = None,
    ) -> "PathResult":
        rx_positions_t = drjit_to_torch_view(rx_positions, dtype=torch.float32)
        device = rx_positions_t.device
        summaries = []
        running_offset = 0
        for raw in raw_collections:
            if raw is None:
                continue
            summary = _summarize_raw_path_collection(raw, device=device)
            summary["offset"] = int(running_offset)
            running_offset += int(summary["count"])
            summaries.append(summary)

        if len(summaries) == 0:
            cat_rx_index = torch.empty((0,), device=device, dtype=torch.int64)
            cat_a = torch.empty((0,), device=device, dtype=torch.complex64)
            cat_tau = torch.empty((0,), device=device, dtype=torch.float32)
        else:
            cat_rx_index = torch.cat([summary["rx_index"] for summary in summaries], dim=0)
            cat_a = torch.cat([summary["a"] for summary in summaries], dim=0)
            cat_tau = torch.cat([summary["tau"] for summary in summaries], dim=0)

        total_paths = int(cat_a.shape[0])
        if total_paths > 0:
            per_rx_counts = torch.bincount(
                cat_rx_index.to(dtype=torch.int64),
                minlength=max(0, int(num_rx)),
            ).to(dtype=torch.int32)
        else:
            per_rx_counts = torch.zeros((num_rx,), device=device, dtype=torch.int32)

        if max_num_paths is None:
            resolved_max_num_paths = max(1, int(per_rx_counts.max().item()) if num_rx > 0 else 1)
        else:
            resolved_max_num_paths = int(max_num_paths)

        max_depth = 1
        kept_counts = torch.zeros((num_rx,), device=device, dtype=torch.int32)
        active_selections: list[tuple[Mapping[str, object], torch.Tensor, torch.Tensor]] = []
        flat_slots = torch.zeros((0,), device=device, dtype=torch.int64)
        if total_paths > 0 and resolved_max_num_paths > 0:
            selected_idx = _resolve_selected_path_indices(
                cat_rx_index=cat_rx_index,
                cat_a=cat_a,
                cat_tau=cat_tau,
                num_rx=num_rx,
                max_num_paths=max_num_paths,
                resolved_max_num_paths=resolved_max_num_paths,
            )

            if int(selected_idx.numel()) > 0:
                selected_rx = cat_rx_index.index_select(0, selected_idx).to(dtype=torch.int64)
                slot_rank, selected_counts = _group_ranks_and_counts(
                    selected_rx,
                    num_groups=num_rx,
                )
                flat_slots = selected_rx * resolved_max_num_paths + slot_rank
                kept_counts = selected_counts.to(dtype=torch.int32)

                active_selections = _build_active_summary_selections(
                    summaries,
                    selected_idx=selected_idx,
                    flat_slots=flat_slots,
                    device=device,
                )
                for summary, selected_paths, _ in active_selections:
                    max_depth = max(
                        max_depth,
                        _selected_summary_max_depth(summary, selected_paths, device=device),
                    )

        flat_count = num_rx * resolved_max_num_paths
        packed_paths = torch.zeros((flat_count, 7), device=device, dtype=torch.float32)
        if flat_count > 0:
            packed_paths[:, 2] = -1.0
        valid = torch.zeros((num_rx, resolved_max_num_paths), device=device, dtype=torch.bool)
        types = torch.zeros((num_rx, resolved_max_num_paths, max_depth), device=device, dtype=torch.int32)
        vertices = None
        normals = None
        objects = None
        if return_geometry:
            vertices = torch.zeros((num_rx, resolved_max_num_paths, max_depth, 3), device=device, dtype=torch.float32)
            normals = torch.zeros((num_rx, resolved_max_num_paths, max_depth, 3), device=device, dtype=torch.float32)
            objects = torch.full((num_rx, resolved_max_num_paths, max_depth), -1, device=device, dtype=torch.int32)

        if int(flat_slots.numel()) > 0:
            valid.view(-1).index_fill_(0, flat_slots, True)

        if flat_count > 0 and len(active_selections) > 0:
            packed_paths_flat = packed_paths
            types_flat = types.view(flat_count, max_depth)
            vertices_flat = None if vertices is None else vertices.view(flat_count, max_depth, 3)
            normals_flat = None if normals is None else normals.view(flat_count, max_depth, 3)
            objects_flat = None if objects is None else objects.view(flat_count, max_depth)

            for summary, selected_paths, selected_slots in active_selections:
                for selected_chunk, flat_slot_chunk in _iter_selected_path_chunks(
                    selected_paths,
                    selected_slots,
                ):
                    count = int(selected_chunk.shape[0])
                    if count <= 0:
                        continue
                    if not return_geometry and _summary_has_direct_no_geometry_tensors(summary):
                        packed_chunk, types_chunk = _build_direct_no_geometry_chunk(
                            summary,
                            selected_chunk,
                        )
                        packed_paths_flat.index_copy_(0, flat_slot_chunk, packed_chunk)
                        normalized_depth = int(types_chunk.shape[1])
                        if normalized_depth > 0:
                            types_flat[:, :normalized_depth].index_copy_(
                                0,
                                flat_slot_chunk,
                                types_chunk,
                            )
                        continue

                    normalized = _normalize_raw_path_collection(
                        summary["raw"],
                        device=device,
                        return_geometry=return_geometry,
                        path_indices=selected_chunk,
                    )

                    packed_chunk = torch.cat(
                        [
                            torch.view_as_real(normalized["a"]).reshape(count, 2),
                            normalized["tau"].reshape(count, 1),
                            normalized["theta_t"].reshape(count, 1),
                            normalized["phi_t"].reshape(count, 1),
                            normalized["theta_r"].reshape(count, 1),
                            normalized["phi_r"].reshape(count, 1),
                        ],
                        dim=1,
                    )
                    packed_paths_flat.index_copy_(0, flat_slot_chunk, packed_chunk)

                    normalized_depth = int(normalized["types"].shape[1])
                    if normalized_depth > 0:
                        types_flat[:, :normalized_depth].index_copy_(
                            0,
                            flat_slot_chunk,
                            normalized["types"],
                        )

                    if (
                        return_geometry
                        and vertices_flat is not None
                        and normals_flat is not None
                        and objects_flat is not None
                    ):
                        normalized_geometry_depth = int(normalized["vertices"].shape[1])
                        if normalized_geometry_depth > 0:
                            vertices_flat[:, :normalized_geometry_depth, :].index_copy_(
                                0,
                                flat_slot_chunk,
                                normalized["vertices"],
                            )
                            normals_flat[:, :normalized_geometry_depth, :].index_copy_(
                                0,
                                flat_slot_chunk,
                                normalized["normals"],
                            )
                            objects_flat[:, :normalized_geometry_depth].index_copy_(
                                0,
                                flat_slot_chunk,
                                normalized["objects"],
                            )

        packed_paths = packed_paths.view(num_rx, resolved_max_num_paths, 7)
        a = torch.complex(packed_paths[..., 0], packed_paths[..., 1])
        tau = packed_paths[..., 2]
        theta_t = packed_paths[..., 3]
        phi_t = packed_paths[..., 4]
        theta_r = packed_paths[..., 5]
        phi_r = packed_paths[..., 6]

        return cls.from_payload(
            {
                "name": name,
                "num_rx": int(num_rx),
                "max_num_paths": int(resolved_max_num_paths),
                "max_depth": int(max_depth),
                "tx_pos": tx_pos,
                "rx_positions": rx_positions,
                "frequency": float(frequency),
                "wavelength": float(wavelength),
                "a": a,
                "tau": tau,
                "theta_t": theta_t,
                "phi_t": phi_t,
                "theta_r": theta_r,
                "phi_r": phi_r,
                "valid": valid,
                "types": types,
                "num_paths": kept_counts,
                "vertices": vertices,
                "normals": normals,
                "objects": objects,
                "metadata": dict(metadata or {}),
            }
        )

    def cir(self, *, normalize_delays: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        tau = drjit_to_torch_view(self.tau, dtype=torch.float32)
        valid = drjit_to_torch_view(self.valid, dtype=torch.bool)
        coeff = self.coeff_tensor()
        if normalize_delays:
            tau = tau - _masked_min(tau, valid)
        tau = torch.where(valid, tau, torch.full_like(tau, -1.0))
        return torch.where(valid, coeff, torch.zeros_like(coeff)), tau

    def cfr(self, frequencies: torch.Tensor, *, normalize_delays: bool = True) -> torch.Tensor:
        freq_tensor = drjit_to_torch_view(frequencies)
        coeff = self.coeff_tensor()
        valid = drjit_to_torch_view(self.valid, dtype=torch.bool)
        tau = drjit_to_torch_view(self.tau, dtype=torch.float32)
        freq = freq_tensor.to(
            device=coeff.device,
            dtype=freq_tensor.dtype if torch.is_complex(freq_tensor) else torch.float32,
        )
        if normalize_delays:
            tau = tau - _masked_min(tau, valid)
        tau = torch.where(valid, tau, torch.zeros_like(tau))
        coeff = torch.where(valid, coeff, torch.zeros_like(coeff))
        phase = -2.0j * math.pi * tau.unsqueeze(-1) * freq.reshape(1, 1, -1)
        return (coeff.unsqueeze(-1) * torch.exp(phase)).sum(dim=1)

    def taps(
        self,
        bandwidth: float,
        num_taps: int,
        *,
        normalize_delays: bool = True,
    ) -> torch.Tensor:
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be > 0.")
        if int(num_taps) <= 0:
            raise ValueError("num_taps must be > 0.")
        coeff = self.coeff_tensor()
        tau = drjit_to_torch_view(self.tau, dtype=torch.float32)
        valid = drjit_to_torch_view(self.valid, dtype=torch.bool)
        if normalize_delays:
            tau = tau - _masked_min(tau, valid)
        tap_idx = torch.round(tau * float(bandwidth)).to(dtype=torch.int64)
        taps = torch.zeros(
            (self.num_rx, int(num_taps)),
            device=coeff.device,
            dtype=coeff.dtype,
        )
        if self.max_num_paths == 0:
            return taps
        rx_idx = torch.arange(self.num_rx, device=coeff.device, dtype=torch.int64).unsqueeze(1)
        keep = valid & (tap_idx >= 0) & (tap_idx < int(num_taps))
        if keep.any():
            taps.index_put_(
                (rx_idx.expand_as(tap_idx)[keep], tap_idx[keep]),
                coeff[keep],
                accumulate=True,
            )
        return taps

    def filter_by_type(self, *interaction_types: int) -> "PathResult":
        if len(interaction_types) == 0:
            return self
        coeff = self.coeff_tensor()
        tau = drjit_to_torch_view(self.tau, dtype=torch.float32)
        theta_t = drjit_to_torch_view(self.theta_t, dtype=torch.float32)
        phi_t = drjit_to_torch_view(self.phi_t, dtype=torch.float32)
        theta_r = drjit_to_torch_view(self.theta_r, dtype=torch.float32)
        phi_r = drjit_to_torch_view(self.phi_r, dtype=torch.float32)
        valid = drjit_to_torch_view(self.valid, dtype=torch.bool)
        types = drjit_to_torch_view(self.types, dtype=torch.int32)
        vertices = None if self.vertices is None else drjit_to_torch_view(self.vertices, dtype=torch.float32)
        normals = None if self.normals is None else drjit_to_torch_view(self.normals, dtype=torch.float32)
        objects = None if self.objects is None else drjit_to_torch_view(self.objects, dtype=torch.int32)

        type_values = {int(value) for value in interaction_types}
        non_empty = types != 0
        keep = torch.zeros_like(valid)
        if 0 in type_values:
            keep = keep | (valid & ~non_empty.any(dim=-1))
        non_zero_types = [value for value in type_values if value != 0]
        if non_zero_types:
            wanted = torch.zeros_like(valid)
            for value in non_zero_types:
                wanted = wanted | (types == int(value)).any(dim=-1)
            keep = keep | (valid & wanted)
        valid = valid & keep
        num_paths = valid.to(dtype=torch.int32).sum(dim=1)
        compact_max_num_paths = max(1, int(num_paths.max().item()) if num_paths.numel() > 0 else 1)
        stable_order = torch.argsort(
            torch.where(
                valid,
                torch.zeros_like(valid, dtype=torch.int64),
                torch.ones_like(valid, dtype=torch.int64),
            ),
            dim=1,
            stable=True,
        )

        def _gather_2d(tensor: torch.Tensor) -> torch.Tensor:
            return torch.gather(tensor, 1, stable_order)[:, :compact_max_num_paths]

        def _gather_3d(tensor: torch.Tensor) -> torch.Tensor:
            gather_idx = stable_order.unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
            return torch.gather(tensor, 1, gather_idx)[:, :compact_max_num_paths]

        def _gather_4d(tensor: torch.Tensor) -> torch.Tensor:
            gather_idx = stable_order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, tensor.shape[-2], tensor.shape[-1])
            return torch.gather(tensor, 1, gather_idx)[:, :compact_max_num_paths]

        compact_valid = _gather_2d(valid)
        compact_a = torch.where(
            compact_valid,
            _gather_2d(coeff),
            torch.zeros((self.num_rx, compact_max_num_paths), device=coeff.device, dtype=coeff.dtype),
        )
        compact_tau = torch.where(
            compact_valid,
            _gather_2d(tau),
            torch.full((self.num_rx, compact_max_num_paths), -1.0, device=tau.device, dtype=tau.dtype),
        )
        compact_theta_t = torch.where(
            compact_valid,
            _gather_2d(theta_t),
            torch.zeros((self.num_rx, compact_max_num_paths), device=theta_t.device, dtype=theta_t.dtype),
        )
        compact_phi_t = torch.where(
            compact_valid,
            _gather_2d(phi_t),
            torch.zeros((self.num_rx, compact_max_num_paths), device=phi_t.device, dtype=phi_t.dtype),
        )
        compact_theta_r = torch.where(
            compact_valid,
            _gather_2d(theta_r),
            torch.zeros((self.num_rx, compact_max_num_paths), device=theta_r.device, dtype=theta_r.dtype),
        )
        compact_phi_r = torch.where(
            compact_valid,
            _gather_2d(phi_r),
            torch.zeros((self.num_rx, compact_max_num_paths), device=phi_r.device, dtype=phi_r.dtype),
        )
        compact_types = torch.where(
            compact_valid.unsqueeze(-1),
            _gather_3d(types),
            torch.zeros((self.num_rx, compact_max_num_paths, self.max_depth), device=types.device, dtype=types.dtype),
        )
        compact_vertices = None if vertices is None else torch.where(
            compact_valid.unsqueeze(-1).unsqueeze(-1),
            _gather_4d(vertices),
            torch.zeros(
                (self.num_rx, compact_max_num_paths, self.max_depth, 3),
                device=vertices.device,
                dtype=vertices.dtype,
            ),
        )
        compact_normals = None if normals is None else torch.where(
            compact_valid.unsqueeze(-1).unsqueeze(-1),
            _gather_4d(normals),
            torch.zeros(
                (self.num_rx, compact_max_num_paths, self.max_depth, 3),
                device=normals.device,
                dtype=normals.dtype,
            ),
        )
        compact_objects = None if objects is None else torch.where(
            compact_valid.unsqueeze(-1),
            _gather_3d(objects),
            torch.full(
                (self.num_rx, compact_max_num_paths, self.max_depth),
                -1,
                device=objects.device,
                dtype=objects.dtype,
            ),
        )
        return PathResult.from_payload(
            {
                "name": self.name,
                "num_rx": self.num_rx,
                "max_num_paths": compact_max_num_paths,
                "max_depth": self.max_depth,
                "tx_pos": self.tx_pos,
                "rx_positions": self.rx_positions,
                "frequency": self.frequency,
                "wavelength": self.wavelength,
                "a": compact_a,
                "tau": compact_tau,
                "theta_t": compact_theta_t,
                "phi_t": compact_phi_t,
                "theta_r": compact_theta_r,
                "phi_r": compact_phi_r,
                "valid": compact_valid,
                "types": compact_types,
                "num_paths": num_paths,
                "vertices": compact_vertices,
                "normals": compact_normals,
                "objects": compact_objects,
                "metadata": dict(self.metadata),
            }
        )

    @property
    def primary(self) -> "PathResult":
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "num_rx": self.num_rx,
            "max_num_paths": self.max_num_paths,
            "max_depth": self.max_depth,
            "tx_pos": self.tx_pos,
            "rx_positions": self.rx_positions,
            "frequency": self.frequency,
            "wavelength": self.wavelength,
            "a": self.a,
            "tau": self.tau,
            "theta_t": self.theta_t,
            "phi_t": self.phi_t,
            "theta_r": self.theta_r,
            "phi_r": self.phi_r,
            "valid": self.valid,
            "types": self.types,
            "num_paths": self.num_paths,
            "vertices": self.vertices,
            "normals": self.normals,
            "objects": self.objects,
            "metadata": dict(self.metadata),
        }

__all__ = [
    "PathResult",
    "_PATH_RESULT_REPLAY_CHUNK_SIZE",
]
