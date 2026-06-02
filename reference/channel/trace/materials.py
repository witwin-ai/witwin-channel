"""Shared material resolution helpers for reflection and diffraction tracing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Iterator, TypeAlias

import drjit as dr
import witwin as wt

from ..utils.material import scalar_fresnel_reflection
from ..scene.runtime_queries import get_triangle_material_data
from ..utils.constants import SPEED_OF_LIGHT

REFLECTION_TRACE_DETAIL_KIND = "reflection_trace_detail"
ReflectionMaterial: TypeAlias = dict[str, float] | None
ReflectionDetailPayload: TypeAlias = Mapping[str, object]


@dataclass
class ReflectionTraceDetail(Mapping[str, object]):
    reflection_model: str
    reflection_model_source: str
    reflection_gain: float
    reflection_material: ReflectionMaterial
    use_scene_materials: bool
    source_paths_per_bounce: tuple["ReflectionSourcePathSet | None", ...]
    extra_payload: Mapping[str, object] = field(default_factory=dict)

    @property
    def detail_kind(self) -> str:
        return REFLECTION_TRACE_DETAIL_KIND

    def __getitem__(self, key: str) -> object:
        if key == "detail_kind":
            return self.detail_kind
        if key == "reflection_model":
            return self.reflection_model
        if key == "reflection_model_source":
            return self.reflection_model_source
        if key == "reflection_gain":
            return self.reflection_gain
        if key == "reflection_material":
            return self.reflection_material
        if key == "use_scene_materials":
            return self.use_scene_materials
        if key == "source_paths_per_bounce":
            return self.source_paths_per_bounce
        if key in self.extra_payload:
            return self.extra_payload[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield "detail_kind"
        yield "reflection_model"
        yield "reflection_model_source"
        yield "reflection_gain"
        yield "reflection_material"
        yield "use_scene_materials"
        yield "source_paths_per_bounce"
        for key in self.extra_payload:
            if key not in {
                "detail_kind",
                "reflection_model",
                "reflection_model_source",
                "reflection_gain",
                "reflection_material",
                "use_scene_materials",
                "source_paths_per_bounce",
            }:
                yield str(key)

    def __len__(self) -> int:
        reserved_count = 7
        extra_keys = {
            str(key)
            for key in self.extra_payload
            if key not in {
                "detail_kind",
                "reflection_model",
                "reflection_model_source",
                "reflection_gain",
                "reflection_material",
                "use_scene_materials",
                "source_paths_per_bounce",
            }
        }
        return reserved_count + len(extra_keys)

    def to_dict(self) -> dict[str, object]:
        detail = {
            "detail_kind": self.detail_kind,
            "reflection_model": self.reflection_model,
            "reflection_model_source": self.reflection_model_source,
            "reflection_gain": self.reflection_gain,
            "reflection_material": self.reflection_material,
            "use_scene_materials": self.use_scene_materials,
            "source_paths_per_bounce": self.source_paths_per_bounce,
        }
        detail.update(dict(self.extra_payload))
        return detail

    def __setitem__(self, key: str, value: object) -> None:
        key_name = str(key)
        if key_name in {
            "detail_kind",
            "reflection_model",
            "reflection_model_source",
            "reflection_gain",
            "reflection_material",
            "use_scene_materials",
            "source_paths_per_bounce",
        }:
            raise KeyError(f"Cannot assign reserved reflection-detail key '{key_name}'.")
        self.extra_payload[key_name] = value


@dataclass(frozen=True)
class ReflectionSourcePathSet(Mapping[str, object]):
    image_source: wt.Point3f
    discovery_count: wt.UInt32
    chain_depth: int
    n_paths: int
    path_prim_idx: tuple[wt.Int32, ...]
    path_plane_point: tuple[wt.Point3f, ...]
    path_plane_normal: tuple[wt.Vector3f, ...]
    path_hit_point: tuple[wt.Point3f, ...]

    @classmethod
    def from_payload(cls, payload: ReflectionSourcePathSet | Mapping[str, object]) -> "ReflectionSourcePathSet":
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

    def __getitem__(self, key: str) -> object:
        if key == "image_source":
            return self.image_source
        if key == "discovery_count":
            return self.discovery_count
        if key == "chain_depth":
            return self.chain_depth
        if key == "n_paths":
            return self.n_paths
        for prefix, values in (
            ("path_prim_idx_", self.path_prim_idx),
            ("path_plane_point_", self.path_plane_point),
            ("path_plane_normal_", self.path_plane_normal),
            ("path_hit_point_", self.path_hit_point),
        ):
            if key.startswith(prefix):
                slot = int(key[len(prefix):])
                return values[slot]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield "image_source"
        yield "discovery_count"
        yield "chain_depth"
        yield "n_paths"
        for slot in range(self.chain_depth):
            yield f"path_prim_idx_{slot}"
            yield f"path_plane_point_{slot}"
            yield f"path_plane_normal_{slot}"
            yield f"path_hit_point_{slot}"

    def __len__(self) -> int:
        return 4 + 4 * int(self.chain_depth)


@dataclass(frozen=True)
class ReflectionMaterialContext:
    reflection_material: ReflectionMaterial
    reflection_gain: float
    use_scene_materials: bool


def reflection_material_omega(wavelength) -> wt.Float:
    return wt.Float(2.0 * math.pi * SPEED_OF_LIGHT / wavelength)


def normalized_override_material(
    reflection_material: Mapping[str, object] | None,
    reflection_coef: float,
    eta_r: float,
    sigma: float,
) -> ReflectionMaterial:
    if reflection_material is None:
        return None
    if not isinstance(reflection_material, Mapping):
        raise TypeError(
            "reflection_material must be a mapping with relative_permittivity/conductivity/gain keys."
        )
    return {
        "relative_permittivity": float(reflection_material.get("relative_permittivity", eta_r)),
        "conductivity": float(reflection_material.get("conductivity", sigma)),
        "gain": float(reflection_material.get("gain", reflection_coef)),
    }


def build_reflection_trace_detail(
    *,
    reflection_model: str,
    reflection_model_source: str,
    reflection_gain: float,
    reflection_material: Mapping[str, object] | None,
    use_scene_materials: bool,
    source_paths_per_bounce: Sequence[ReflectionSourcePathSet | Mapping[str, object] | None] = (),
    **payload,
    ) -> ReflectionTraceDetail:
    detail_payload: dict[str, object] = dict(payload)
    if "reflection_sampling" not in detail_payload:
        if isinstance(detail_payload.get("dda_stats"), Mapping):
            detail_payload["reflection_sampling"] = dict(detail_payload["dda_stats"])
        elif isinstance(detail_payload.get("discovery_sampling"), Mapping):
            detail_payload["reflection_sampling"] = dict(detail_payload["discovery_sampling"])
    return ReflectionTraceDetail(
        reflection_model=str(reflection_model),
        reflection_model_source=str(reflection_model_source),
        reflection_gain=float(reflection_gain),
        reflection_material=normalized_override_material(
            reflection_material,
            reflection_coef=reflection_gain,
            eta_r=5.0,
            sigma=0.0,
        ),
        use_scene_materials=bool(use_scene_materials),
        source_paths_per_bounce=tuple(
            None if paths is None else ReflectionSourcePathSet.from_payload(paths)
            for paths in source_paths_per_bounce
        ),
        extra_payload=detail_payload,
    )


def coerce_reflection_trace_detail(
    reflection_detail: ReflectionTraceDetail | ReflectionDetailPayload,
) -> ReflectionTraceDetail:
    if isinstance(reflection_detail, ReflectionTraceDetail):
        return reflection_detail
    if not isinstance(reflection_detail, Mapping):
        raise TypeError(
            "reflection_detail must be a payload returned by compute_reflection_field()."
        )
    detail_kind = reflection_detail["detail_kind"]
    if detail_kind != REFLECTION_TRACE_DETAIL_KIND:
        raise TypeError(
            "reflection_detail must carry detail_kind='reflection_trace_detail'; "
            "pass the detail payload returned by compute_reflection_field()."
        )
    source_paths = reflection_detail["source_paths_per_bounce"]
    if isinstance(source_paths, (str, bytes)) or not isinstance(source_paths, Sequence):
        raise TypeError("reflection_detail['source_paths_per_bounce'] must be a sequence.")
    extra_payload = {
        str(key): value
        for key, value in dict(reflection_detail).items()
        if key not in {
            "detail_kind",
            "reflection_model",
            "reflection_model_source",
            "reflection_gain",
            "reflection_material",
            "use_scene_materials",
            "source_paths_per_bounce",
        }
    }
    return ReflectionTraceDetail(
        reflection_model=str(reflection_detail["reflection_model"]),
        reflection_model_source=str(reflection_detail["reflection_model_source"]),
        reflection_gain=float(reflection_detail["reflection_gain"]),
        reflection_material=normalized_override_material(
            reflection_detail["reflection_material"],
            reflection_coef=float(reflection_detail["reflection_gain"]),
            eta_r=5.0,
            sigma=0.0,
        ),
        use_scene_materials=bool(reflection_detail["use_scene_materials"]),
        source_paths_per_bounce=tuple(
            None if paths is None else ReflectionSourcePathSet.from_payload(paths)
            for paths in source_paths
        ),
        extra_payload=extra_payload,
    )


def coerce_reflection_material_context(
    reflection_detail: ReflectionTraceDetail | ReflectionDetailPayload | None,
    *,
    default_gain: float,
) -> ReflectionMaterialContext:
    if reflection_detail is None:
        return ReflectionMaterialContext(
            reflection_material=None,
            reflection_gain=float(default_gain),
            use_scene_materials=False,
        )
    detail = coerce_reflection_trace_detail(reflection_detail)
    return ReflectionMaterialContext(
        reflection_material=detail.reflection_material,
        reflection_gain=detail.reflection_gain,
        use_scene_materials=detail.use_scene_materials,
    )


def scene_uses_material_table(scene, *, use_scene_materials: bool = True) -> bool:
    if not use_scene_materials:
        return False
    if scene is None or scene.tri_data_gpu is None:
        return False
    return True


def material_source_label(scene, override_material, *, use_scene_materials: bool = True) -> str:
    if override_material is not None:
        return "override"
    if scene_uses_material_table(scene, use_scene_materials=use_scene_materials):
        return "scene"
    return "default"


def reflection_model_label(scene, override_material, *, use_scene_materials: bool = True) -> str:
    del scene, override_material, use_scene_materials
    return "materialized"


def _mask_width(mask_or_indices) -> int:
    return int(dr.width(mask_or_indices))


def _default_surface_material_inputs(
    *,
    width: int,
    valid_mask,
    reflection_coef: float,
    default_eta_r: float,
    default_sigma: float,
):
    return {
        "eta_r": dr.full(wt.Float, float(default_eta_r), width),
        "sigma": dr.full(wt.Float, float(default_sigma), width),
        "gain": dr.full(wt.Float, float(reflection_coef), width),
        "use_fresnel": valid_mask,
        "structure_idx": dr.full(wt.Int32, -1, width),
        "valid": valid_mask,
    }


def resolve_surface_material(
    *,
    scene,
    prim_idx,
    override_material,
    reflection_coef: float,
    default_eta_r: float,
    default_sigma: float,
    valid_mask=None,
    use_scene_materials: bool = True,
):
    prim_idx_i32 = wt.Int32(prim_idx)
    width = _mask_width(prim_idx_i32)
    if valid_mask is None:
        valid_mask = dr.full(wt.Bool, True, width)

    if override_material is not None:
        return {
            "eta_r": dr.full(
                wt.Float,
                float(override_material["relative_permittivity"]),
                width,
            ),
            "sigma": dr.full(
                wt.Float,
                float(override_material["conductivity"]),
                width,
            ),
            "gain": dr.full(
                wt.Float,
                float(override_material["gain"]),
                width,
            ),
            "use_fresnel": valid_mask,
            "structure_idx": dr.full(wt.Int32, -1, width),
            "valid": valid_mask,
        }

    if not scene_uses_material_table(scene, use_scene_materials=use_scene_materials):
        return _default_surface_material_inputs(
            width=width,
            valid_mask=valid_mask,
            reflection_coef=reflection_coef,
            default_eta_r=default_eta_r,
            default_sigma=default_sigma,
        )

    triangle_material = get_triangle_material_data(scene, prim_idx_i32, valid_mask=valid_mask)
    use_scene_values = triangle_material["valid"] & triangle_material["specified"]
    return {
        "eta_r": dr.select(
            use_scene_values,
            triangle_material["eps_r"],
            wt.Float(default_eta_r),
        ),
        "sigma": dr.select(
            use_scene_values,
            triangle_material["sigma_e"],
            wt.Float(default_sigma),
        ),
        "gain": dr.full(wt.Float, float(reflection_coef), width),
        "use_fresnel": triangle_material["valid"],
        "structure_idx": triangle_material["structure_idx"],
        "valid": triangle_material["valid"],
    }


def bounce_reflection_weight(
    *,
    incident_dir,
    normal,
    wavelength,
    reflection_coef: float,
    material_inputs,
    tx_polarization=(1.0, 0.0, 0.0),
):
    del tx_polarization
    incident_hat = incident_dir / (dr.norm(incident_dir) + 1e-12)
    normal_hat = normal / (dr.norm(normal) + 1e-12)
    cos_theta = dr.clip(dr.abs(dr.dot(incident_hat, normal_hat)), wt.Float(1e-6), wt.Float(1.0))
    fresnel_weight = scalar_fresnel_reflection(
        cos_theta=cos_theta,
        eta_r=material_inputs["eta_r"],
        sigma=material_inputs["sigma"],
        omega=reflection_material_omega(wavelength),
        gain=material_inputs["gain"],
    )
    return dr.select(material_inputs["valid"], fresnel_weight, wt.Complex2f(-reflection_coef, 0.0))
