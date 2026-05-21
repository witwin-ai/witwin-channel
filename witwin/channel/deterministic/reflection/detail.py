"""Reflection path detail payloads and coercion helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from witwin.channel.deterministic import types as wt

TRACE_DETAIL_KIND = "reflection_trace_detail"
REFLECTION_TRANSITION_MODES = {"hard", "f_weight_reference", "f_weight_native"}
REFLECTION_SECONDARY_VISIBILITY_MODES = {"hard", "f_weight"}
DEFAULT_REFLECTION_TRANSITION_MODE = "hard"
DEFAULT_REFLECTION_SECONDARY_VISIBILITY_MODE = "hard"
DEFAULT_REFLECTION_F_WEIGHT_BOUNDARY_RADIUS_WAVELENGTHS = 2.0
DEFAULT_REFLECTION_F_WEIGHT_MAX_EDGES_PER_SLOT = 1
DetailPayload: TypeAlias = Mapping[str, object]


@dataclass
class TraceDetail:
    reflection_model: str
    reflection_model_source: str
    reflection_gain: float
    source_paths_per_bounce: tuple["SourcePathSet | None", ...]
    reflection_transition_mode: str = DEFAULT_REFLECTION_TRANSITION_MODE
    reflection_f_weight_boundary_radius_wavelengths: float = DEFAULT_REFLECTION_F_WEIGHT_BOUNDARY_RADIUS_WAVELENGTHS
    reflection_f_weight_max_edges_per_slot: int = DEFAULT_REFLECTION_F_WEIGHT_MAX_EDGES_PER_SLOT
    reflection_secondary_visibility_mode: str = DEFAULT_REFLECTION_SECONDARY_VISIBILITY_MODE


@dataclass(frozen=True)
class SourcePathSet:
    image_source: wt.Point3f
    discovery_count: wt.UInt32
    chain_depth: int
    n_paths: int
    path_prim_idx: tuple[wt.Int32, ...]
    path_plane_point: tuple[wt.Point3f, ...]
    path_plane_normal: tuple[wt.Vector3f, ...]
    path_hit_point: tuple[wt.Point3f, ...]

    @classmethod
    def from_payload(
        cls,
        payload: "SourcePathSet | Mapping[str, object]",
    ) -> "SourcePathSet":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping):
            raise TypeError("Reflection source-path payloads must be mapping-like.")
        chain_depth = int(payload.get("chain_depth", 0))
        return cls(
            image_source=payload["image_source"],
            discovery_count=payload["discovery_count"],
            chain_depth=chain_depth,
            n_paths=int(payload.get("n_paths", 0)),
            path_prim_idx=tuple(payload[f"path_prim_idx_{slot}"] for slot in range(chain_depth)),
            path_plane_point=tuple(payload[f"path_plane_point_{slot}"] for slot in range(chain_depth)),
            path_plane_normal=tuple(payload[f"path_plane_normal_{slot}"] for slot in range(chain_depth)),
            path_hit_point=tuple(payload[f"path_hit_point_{slot}"] for slot in range(chain_depth)),
        )

    def prim_idx(self, slot: int) -> wt.Int32:
        return self.path_prim_idx[int(slot)]

    def plane_point(self, slot: int) -> wt.Point3f:
        return self.path_plane_point[int(slot)]

    def plane_normal(self, slot: int) -> wt.Vector3f:
        return self.path_plane_normal[int(slot)]

    def hit_point(self, slot: int) -> wt.Point3f:
        return self.path_hit_point[int(slot)]


@dataclass(frozen=True)
class MaterialContext:
    reflection_gain: float


def build_trace_detail(
    *,
    reflection_model: str,
    reflection_model_source: str,
    reflection_gain: float,
    source_paths_per_bounce: Sequence[SourcePathSet | Mapping[str, object] | None] = (),
    reflection_transition_mode: str = DEFAULT_REFLECTION_TRANSITION_MODE,
    reflection_f_weight_boundary_radius_wavelengths: float = DEFAULT_REFLECTION_F_WEIGHT_BOUNDARY_RADIUS_WAVELENGTHS,
    reflection_f_weight_max_edges_per_slot: int = DEFAULT_REFLECTION_F_WEIGHT_MAX_EDGES_PER_SLOT,
    reflection_secondary_visibility_mode: str = DEFAULT_REFLECTION_SECONDARY_VISIBILITY_MODE,
) -> TraceDetail:
    transition_mode = str(reflection_transition_mode)
    if transition_mode not in REFLECTION_TRANSITION_MODES:
        raise ValueError(
            "reflection_transition_mode must be one of "
            f"{sorted(REFLECTION_TRANSITION_MODES)}; got {transition_mode!r}."
        )
    secondary_visibility_mode = str(reflection_secondary_visibility_mode)
    if secondary_visibility_mode not in REFLECTION_SECONDARY_VISIBILITY_MODES:
        raise ValueError(
            "reflection_secondary_visibility_mode must be one of "
            f"{sorted(REFLECTION_SECONDARY_VISIBILITY_MODES)}; got {secondary_visibility_mode!r}."
        )
    boundary_radius = float(reflection_f_weight_boundary_radius_wavelengths)
    if boundary_radius <= 0.0:
        raise ValueError("reflection_f_weight_boundary_radius_wavelengths must be > 0.")
    max_edges = int(reflection_f_weight_max_edges_per_slot)
    if max_edges <= 0:
        raise ValueError("reflection_f_weight_max_edges_per_slot must be > 0.")
    return TraceDetail(
        reflection_model=str(reflection_model),
        reflection_model_source=str(reflection_model_source),
        reflection_gain=float(reflection_gain),
        source_paths_per_bounce=tuple(
            None if paths is None else SourcePathSet.from_payload(paths)
            for paths in source_paths_per_bounce
        ),
        reflection_transition_mode=transition_mode,
        reflection_f_weight_boundary_radius_wavelengths=boundary_radius,
        reflection_f_weight_max_edges_per_slot=max_edges,
        reflection_secondary_visibility_mode=secondary_visibility_mode,
    )


def coerce_trace_detail(
    reflection_detail: TraceDetail | DetailPayload,
) -> TraceDetail:
    if isinstance(reflection_detail, TraceDetail):
        return reflection_detail
    if not isinstance(reflection_detail, Mapping):
        raise TypeError("reflection_detail must be a payload returned by compute_field().")
    if reflection_detail["detail_kind"] != TRACE_DETAIL_KIND:
        raise TypeError(
            "reflection_detail must carry detail_kind='reflection_trace_detail'; "
            "pass the detail payload returned by compute_field()."
        )
    source_paths = reflection_detail["source_paths_per_bounce"]
    if isinstance(source_paths, (str, bytes)) or not isinstance(source_paths, Sequence):
        raise TypeError("reflection_detail['source_paths_per_bounce'] must be a sequence.")
    return TraceDetail(
        reflection_model=str(reflection_detail["reflection_model"]),
        reflection_model_source=str(reflection_detail["reflection_model_source"]),
        reflection_gain=float(reflection_detail["reflection_gain"]),
        source_paths_per_bounce=tuple(
            None if paths is None else SourcePathSet.from_payload(paths)
            for paths in source_paths
        ),
        reflection_transition_mode=str(
            reflection_detail.get("reflection_transition_mode", DEFAULT_REFLECTION_TRANSITION_MODE)
        ),
        reflection_f_weight_boundary_radius_wavelengths=float(
            reflection_detail.get(
                "reflection_f_weight_boundary_radius_wavelengths",
                DEFAULT_REFLECTION_F_WEIGHT_BOUNDARY_RADIUS_WAVELENGTHS,
            )
        ),
        reflection_f_weight_max_edges_per_slot=int(
            reflection_detail.get(
                "reflection_f_weight_max_edges_per_slot",
                DEFAULT_REFLECTION_F_WEIGHT_MAX_EDGES_PER_SLOT,
            )
        ),
        reflection_secondary_visibility_mode=str(
            reflection_detail.get(
                "reflection_secondary_visibility_mode",
                DEFAULT_REFLECTION_SECONDARY_VISIBILITY_MODE,
            )
        ),
    )


def coerce_material_context(
    reflection_detail: TraceDetail | DetailPayload | None,
    *,
    default_gain: float,
) -> MaterialContext:
    if reflection_detail is None:
        return MaterialContext(
            reflection_gain=float(default_gain),
        )
    detail = coerce_trace_detail(reflection_detail)
    return MaterialContext(
        reflection_gain=detail.reflection_gain,
    )
