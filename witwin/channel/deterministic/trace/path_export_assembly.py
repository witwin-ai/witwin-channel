"""Path-result assembly: summarize raw collections and pack the Result payload.

The collectors in ``path_export`` produce raw path dicts in three payload
flavours (``materialized_path_payload_v1``, ``diffraction_state_refs_v1``,
``reflection_path_refs_v1``). This module turns those raw dicts into the
flat per-link payload that the path Result needs: torch tensor conversion,
ranking and selection of paths per (rx, tx) link, chunked replay of selected
indices, and the final packed tensors.
"""

from __future__ import annotations

from typing import Mapping

import drjit as dr
import numpy as np
import torch

from witwin.channel.core.numerics.tensors import to_torch_view
from witwin.channel.deterministic import types as wt
from witwin.channel.deterministic.reflection.detail import (
    coerce_trace_detail as coerce_reflection_trace_detail,
)
from witwin.channel.deterministic.types import InteractionType

from .path_export import (
    _DIFFRACTION_STATE_REFS_PAYLOAD,
    _PATH_RESULT_REPLAY_CHUNK_SIZE,
    _REFLECTION_PATH_REFS_PAYLOAD,
    _diffraction_state_ref_type_slots,
    _materialize_diffraction_state_path_refs,
    _materialize_reflection_path_refs,
)


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
                    tensor = to_torch_view(value, detach=detach, device=device)
                else:
                    real_t = to_torch_view(
                        real,
                        detach=detach,
                        dtype=torch.float32,
                        device=device,
                    )
                    imag_t = to_torch_view(
                        imag,
                        detach=detach,
                        dtype=torch.float32,
                        device=device,
                    )
                    tensor = torch.complex(real_t, imag_t)
            else:
                tensor = to_torch_view(value, detach=detach, device=device)
        else:
            tensor = torch.stack(
                [
                    to_torch_view(x, detach=detach, dtype=torch.float32, device=device),
                    to_torch_view(y, detach=detach, dtype=torch.float32, device=device),
                    to_torch_view(z, detach=detach, dtype=torch.float32, device=device),
                ],
                dim=-1,
            )
    if device is not None or dtype is not None:
        tensor = tensor.to(
            device=device if device is not None else tensor.device,
            dtype=dtype if dtype is not None else tensor.dtype,
        )
    return tensor


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


def _drjit_value_grad_enabled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (tuple, list)):
        return any(_drjit_value_grad_enabled(item) for item in value)
    if dr.is_array_v(type(value)) or dr.is_tensor_v(type(value)):
        try:
            if bool(dr.grad_enabled(value)):
                return True
        except TypeError:
            pass
        components = ("real", "imag") if dr.is_complex_v(type(value)) else ("x", "y", "z")
        for component in components:
            try:
                component_value = getattr(value, component)
            except Exception:
                continue
            if component_value is value:
                continue
            if _drjit_value_grad_enabled(component_value):
                return True
        return False
    for component in ("real", "imag", "x", "y", "z"):
        try:
            component_value = getattr(value, component)
        except Exception:
            continue
        if component_value is value:
            continue
        if _drjit_value_grad_enabled(component_value):
            return True
    try:
        return bool(dr.grad_enabled(value))
    except TypeError:
        return False


def _summary_has_differentiable_values(summary: Mapping[str, object]) -> bool:
    raw = summary["raw"]
    return any(
        _drjit_value_grad_enabled(raw.get(key))
        for key in ("a", "tau", "theta_t", "phi_t", "theta_r", "phi_r")
    )


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
    for key in ("rx_index", "tx_index", "a", "tau", "theta_t", "phi_t", "theta_r", "phi_r"):
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
    default_tx_index = dr.zeros(wt.UInt32, int(rx_index.shape[0]))
    tx_index = _torch_tensor(
        raw.get("tx_index", default_tx_index),
        dtype=torch.int64,
        device=device,
        detach=True,
    ).reshape(-1)
    payload_kind = _raw_path_payload_kind(raw)
    type_slots = tuple(raw.get("type_slots") or ())
    summary = {
        "raw": raw,
        "payload_kind": payload_kind,
        "rx_index": rx_index,
        "tx_index": tx_index,
        "a": _torch_tensor(raw["a"], dtype=torch.complex64, device=device, detach=False).reshape(-1),
        "tau": _torch_tensor(raw["tau"], dtype=torch.float32, device=device, detach=False).reshape(-1),
        "count": int(rx_index.shape[0]),
        "depth_hint": (
            int(raw.get("max_depth_hint", 1))
            if payload_kind == _REFLECTION_PATH_REFS_PAYLOAD
            else (1 if payload_kind == _DIFFRACTION_STATE_REFS_PAYLOAD else max(1, len(type_slots)))
        ),
    }
    if (
        payload_kind not in {_DIFFRACTION_STATE_REFS_PAYLOAD, _REFLECTION_PATH_REFS_PAYLOAD}
        and all(key in raw for key in ("theta_t", "phi_t", "theta_r", "phi_r"))
    ):
        summary["direct_no_geometry_kind"] = "materialized_dense"
        summary["theta_t"] = _torch_tensor(raw["theta_t"], dtype=torch.float32, device=device, detach=False).reshape(-1)
        summary["phi_t"] = _torch_tensor(raw["phi_t"], dtype=torch.float32, device=device, detach=False).reshape(-1)
        summary["theta_r"] = _torch_tensor(raw["theta_r"], dtype=torch.float32, device=device, detach=False).reshape(-1)
        summary["phi_r"] = _torch_tensor(raw["phi_r"], dtype=torch.float32, device=device, detach=False).reshape(-1)
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
        payload_kind == _REFLECTION_PATH_REFS_PAYLOAD
        and raw.get("theta_t") is not None
        and raw.get("phi_t") is not None
        and raw.get("theta_r") is not None
        and raw.get("phi_r") is not None
        and raw.get("path_depth") is not None
    ):
        summary["direct_no_geometry_kind"] = "reflection_cached"
        summary["theta_t"] = _torch_tensor(raw["theta_t"], dtype=torch.float32, device=device, detach=False).reshape(-1)
        summary["phi_t"] = _torch_tensor(raw["phi_t"], dtype=torch.float32, device=device, detach=False).reshape(-1)
        summary["theta_r"] = _torch_tensor(raw["theta_r"], dtype=torch.float32, device=device, detach=False).reshape(-1)
        summary["phi_r"] = _torch_tensor(raw["phi_r"], dtype=torch.float32, device=device, detach=False).reshape(-1)
        summary["path_depth_torch"] = _torch_tensor(
            raw["path_depth"],
            dtype=torch.int32,
            device=device,
            detach=True,
        ).reshape(-1)
    elif (
        payload_kind == _DIFFRACTION_STATE_REFS_PAYLOAD
        and raw.get("theta_t") is not None
        and raw.get("phi_t") is not None
        and raw.get("theta_r") is not None
        and raw.get("phi_r") is not None
        and raw.get("path_depth") is not None
    ):
        summary["direct_no_geometry_kind"] = "diffraction_cached"
        summary["theta_t"] = _torch_tensor(raw["theta_t"], dtype=torch.float32, device=device, detach=False).reshape(-1)
        summary["phi_t"] = _torch_tensor(raw["phi_t"], dtype=torch.float32, device=device, detach=False).reshape(-1)
        summary["theta_r"] = _torch_tensor(raw["theta_r"], dtype=torch.float32, device=device, detach=False).reshape(-1)
        summary["phi_r"] = _torch_tensor(raw["phi_r"], dtype=torch.float32, device=device, detach=False).reshape(-1)
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
    if direct_kind == "diffraction_cached":
        type_slots = _diffraction_state_ref_type_slots(
            summary["raw"],
            path_indices=_torch_path_indices(selected_paths),
        )
        if len(type_slots) == 0:
            types = torch.zeros((count, 1), device=selected_paths.device, dtype=torch.int32)
        else:
            types = torch.stack(
                [
                    _torch_tensor(slot, dtype=torch.int32, device=selected_paths.device, detach=True).reshape(-1)
                    for slot in type_slots
                ],
                dim=1,
            )
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
    cat_group_index: torch.Tensor,
    cat_a: torch.Tensor,
    cat_tau: torch.Tensor,
    num_groups: int,
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
            torch.argsort(cat_group_index.index_select(0, tau_order), stable=True),
        )

    strength_order = torch.argsort(torch.abs(cat_a), descending=True, stable=True)
    grouped_by_strength = strength_order.index_select(
        0,
        torch.argsort(cat_group_index.index_select(0, strength_order), stable=True),
    )
    grouped_ids = cat_group_index.index_select(0, grouped_by_strength).to(dtype=torch.int64)
    strength_rank = _group_ranks(grouped_ids, num_groups=num_groups)
    selected_idx = grouped_by_strength[strength_rank < resolved_max_num_paths]
    if int(selected_idx.numel()) <= 0:
        return selected_idx
    tau_order = torch.argsort(cat_tau.index_select(0, selected_idx), stable=True)
    selected_idx = selected_idx.index_select(0, tau_order)
    return selected_idx.index_select(
        0,
        torch.argsort(cat_group_index.index_select(0, selected_idx), stable=True),
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
    active_summaries = [summary for summary in summaries if int(summary["count"]) > 0]
    if len(active_summaries) == 0:
        return []
    if len(active_summaries) == 1:
        summary = active_summaries[0]
        return [(summary, selected_idx - int(summary["offset"]), flat_slots)]

    summary_offsets = torch.tensor(
        [int(summary["offset"]) for summary in active_summaries],
        device=device,
        dtype=torch.int64,
    )
    selected_summary = torch.bucketize(selected_idx, summary_offsets[1:], right=True)
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


def _iter_selected_path_chunks(local_selected: torch.Tensor, flat_slots: torch.Tensor):
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
            dtype=torch.int32, device=device, detach=True,
        ).reshape(-1)
        intermediate_depth = _torch_tensor(
            dr.gather(wt.UInt32, state_arrays["intermediate_reflection_depth"], selected_state_idx),
            dtype=torch.int32, device=device, detach=True,
        ).reshape(-1)
        suffix_depth = _torch_tensor(
            dr.gather(wt.UInt32, state_arrays["suffix_reflection_depth"], selected_state_idx),
            dtype=torch.int32, device=device, detach=True,
        ).reshape(-1)
        order = _torch_tensor(
            dr.gather(wt.UInt32, state_arrays["order"], selected_state_idx),
            dtype=torch.int32, device=device, detach=True,
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
            dtype=torch.int32, device=device, detach=True,
        ).reshape(-1)
        if selected_depth.numel() == 0:
            return 1
        return max(1, int(selected_depth.max().item()))

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
        dtype=torch.int64, device=device, detach=True,
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
    if str(summary["payload_kind"]) == _DIFFRACTION_STATE_REFS_PAYLOAD:
        return _selected_diffraction_ref_max_depth(summary["raw"], path_indices, device=device)
    if str(summary["payload_kind"]) == _REFLECTION_PATH_REFS_PAYLOAD:
        return _selected_reflection_ref_max_depth(summary["raw"], path_indices, device=device)
    return int(summary["depth_hint"])


def _materialize_raw_path_collection(
    raw: Mapping[str, object],
    *,
    return_geometry: bool,
    path_indices: torch.Tensor | None = None,
) -> Mapping[str, object]:
    payload_kind = str(raw.get("payload_kind", ""))
    if payload_kind == _DIFFRACTION_STATE_REFS_PAYLOAD:
        return _materialize_diffraction_state_path_refs(
            raw,
            return_geometry=return_geometry,
            path_indices=None if path_indices is None else _torch_path_indices(path_indices),
        )
    if payload_kind == _REFLECTION_PATH_REFS_PAYLOAD:
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


def _selected_path_values_drjit(
    active_selections: list[tuple[Mapping[str, object], torch.Tensor, torch.Tensor]],
    *,
    flat_count: int,
) -> tuple[object, object, object, object, object, object]:
    real = dr.zeros(wt.Float, flat_count)
    imag = dr.zeros(wt.Float, flat_count)
    tau = dr.full(wt.Float, -1.0, flat_count)
    theta_t = dr.zeros(wt.Float, flat_count)
    phi_t = dr.zeros(wt.Float, flat_count)
    theta_r = dr.zeros(wt.Float, flat_count)
    phi_r = dr.zeros(wt.Float, flat_count)

    for summary, selected_paths, selected_slots in active_selections:
        for selected_chunk, flat_slot_chunk in _iter_selected_path_chunks(
            selected_paths,
            selected_slots,
        ):
            if int(selected_chunk.shape[0]) <= 0:
                continue
            raw = _materialize_raw_path_collection(
                summary["raw"],
                return_geometry=False,
                path_indices=selected_chunk,
            )
            flat_slot_idx = _torch_path_indices(flat_slot_chunk)
            dr.scatter(real, raw["a"].real, flat_slot_idx)
            dr.scatter(imag, raw["a"].imag, flat_slot_idx)
            dr.scatter(tau, raw["tau"], flat_slot_idx)
            dr.scatter(theta_t, raw["theta_t"], flat_slot_idx)
            dr.scatter(phi_t, raw["phi_t"], flat_slot_idx)
            dr.scatter(theta_r, raw["theta_r"], flat_slot_idx)
            dr.scatter(phi_r, raw["phi_r"], flat_slot_idx)

    return wt.Complex2f(real, imag), tau, theta_t, phi_t, theta_r, phi_r


def assemble_result_payload(
    *,
    name: str,
    num_tx: int,
    num_rx: int,
    max_num_paths: int | None,
    tx_pos: tuple[float, float, float],
    tx_positions,
    rx_positions,
    frequency: float,
    wavelength: float,
    raw_collections: list[Mapping[str, object]],
    return_geometry: bool,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Pack heterogeneous raw path collections into a Result-ready payload dict."""
    rx_positions_t = to_torch_view(rx_positions, dtype=torch.float32)
    device = rx_positions_t.device
    num_links = int(num_rx) * int(num_tx)
    summaries = []
    running_offset = 0
    for raw in raw_collections:
        if raw is None:
            continue
        summary = _summarize_raw_path_collection(raw, device=device)
        summary["offset"] = int(running_offset)
        running_offset += int(summary["count"])
        summaries.append(summary)
    preserve_value_ad = any(_summary_has_differentiable_values(summary) for summary in summaries)

    if len(summaries) == 0:
        cat_rx_index = torch.empty((0,), device=device, dtype=torch.int64)
        cat_tx_index = torch.empty((0,), device=device, dtype=torch.int64)
        cat_a = torch.empty((0,), device=device, dtype=torch.complex64)
        cat_tau = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        cat_rx_index = torch.cat([summary["rx_index"] for summary in summaries], dim=0)
        cat_tx_index = torch.cat([summary["tx_index"] for summary in summaries], dim=0)
        cat_a = torch.cat([summary["a"] for summary in summaries], dim=0)
        cat_tau = torch.cat([summary["tau"] for summary in summaries], dim=0)
    cat_link_index = cat_rx_index.to(dtype=torch.int64) * int(num_tx) + cat_tx_index.to(dtype=torch.int64)

    total_paths = int(cat_a.shape[0])
    if total_paths > 0:
        per_link_counts = torch.bincount(
            cat_link_index,
            minlength=max(0, int(num_links)),
        ).to(dtype=torch.int32)
    else:
        per_link_counts = torch.zeros((num_links,), device=device, dtype=torch.int32)

    if max_num_paths is None:
        resolved_max_num_paths = max(1, int(per_link_counts.max().item()) if num_links > 0 else 1)
    else:
        resolved_max_num_paths = int(max_num_paths)

    max_depth = 1
    kept_counts = torch.zeros((num_links,), device=device, dtype=torch.int32)
    active_selections: list[tuple[Mapping[str, object], torch.Tensor, torch.Tensor]] = []
    flat_slots = torch.zeros((0,), device=device, dtype=torch.int64)
    if total_paths > 0 and resolved_max_num_paths > 0:
        selected_idx = _resolve_selected_path_indices(
            cat_group_index=cat_link_index,
            cat_a=cat_a,
            cat_tau=cat_tau,
            num_groups=num_links,
            max_num_paths=max_num_paths,
            resolved_max_num_paths=resolved_max_num_paths,
        )

        if int(selected_idx.numel()) > 0:
            selected_link = cat_link_index.index_select(0, selected_idx).to(dtype=torch.int64)
            slot_rank, selected_counts = _group_ranks_and_counts(
                selected_link,
                num_groups=num_links,
            )
            flat_slots = selected_link * resolved_max_num_paths + slot_rank
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

    flat_count = num_links * resolved_max_num_paths
    packed_paths = torch.zeros((flat_count, 7), device=device, dtype=torch.float32)
    if flat_count > 0:
        packed_paths[:, 2] = -1.0
    valid = torch.zeros((num_rx, num_tx, resolved_max_num_paths), device=device, dtype=torch.bool)
    types = torch.zeros((num_rx, num_tx, resolved_max_num_paths, max_depth), device=device, dtype=torch.int32)
    vertices = None
    normals = None
    objects = None
    if return_geometry:
        vertices = torch.zeros((num_rx, num_tx, resolved_max_num_paths, max_depth, 3), device=device, dtype=torch.float32)
        normals = torch.zeros((num_rx, num_tx, resolved_max_num_paths, max_depth, 3), device=device, dtype=torch.float32)
        objects = torch.full((num_rx, num_tx, resolved_max_num_paths, max_depth), -1, device=device, dtype=torch.int32)

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

    packed_paths = packed_paths.view(num_rx, num_tx, resolved_max_num_paths, 7)
    a = torch.complex(packed_paths[..., 0], packed_paths[..., 1])
    tau = packed_paths[..., 2]
    theta_t = packed_paths[..., 3]
    phi_t = packed_paths[..., 4]
    theta_r = packed_paths[..., 5]
    phi_r = packed_paths[..., 6]
    if preserve_value_ad:
        a, tau, theta_t, phi_t, theta_r, phi_r = _selected_path_values_drjit(
            active_selections,
            flat_count=flat_count,
        )

    return {
        "name": name,
        "num_tx": int(num_tx),
        "num_rx": int(num_rx),
        "max_num_paths": int(resolved_max_num_paths),
        "max_depth": int(max_depth),
        "tx_pos": tx_pos,
        "tx_positions": tx_positions,
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
        "num_paths": kept_counts.view(num_rx, num_tx),
        "vertices": vertices,
        "normals": normals,
        "objects": objects,
        "metadata": dict(metadata or {}),
    }


__all__ = ["assemble_result_payload"]
