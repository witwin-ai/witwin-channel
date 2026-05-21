"""DrJit array helpers: scalar extraction, init, concat, broadcast, gather, eval/sync."""

from __future__ import annotations

from collections.abc import Mapping

import drjit as dr
import numpy as np
from witwin.channel import types as wt

from .constants import EPS


def scalar(value) -> float:
    """Extract a Python float from a scalar or DrJit array."""
    if hasattr(value, "__len__") and dr.width(value) > 0:
        return float(np.array(value).flat[0])
    return float(value)


def mask_count(mask) -> int:
    return int(scalar(dr.sum(dr.select(mask, wt.UInt32(1), wt.UInt32(0)))))


def empty_complex() -> wt.Complex2f:
    return dr.zeros(wt.Complex2f, 0)


def empty_point3() -> wt.Point3f:
    return dr.zeros(wt.Point3f, 0)


def empty_vector3() -> wt.Vector3f:
    return dr.zeros(wt.Vector3f, 0)


def empty_vector3u() -> wt.Vector3u:
    return dr.zeros(wt.Vector3u, 0)


def zeros_point3(width: int) -> wt.Point3f:
    return dr.zeros(wt.Point3f, width)


def zeros_vector3(width: int) -> wt.Vector3f:
    return dr.zeros(wt.Vector3f, width)


def complex_zero(width: int) -> wt.Complex2f:
    return dr.zeros(wt.Complex2f, width)


def gather_point3(source, index):
    return dr.gather(type(source), source, index)


def gather_vector3(source, index):
    return dr.gather(type(source), source, index)


def concat_arrays(array_type, values):
    non_empty = [v for v in values if dr.width(v) > 0]
    return dr.concat(non_empty) if non_empty else dr.zeros(array_type, 0)


def _concat_components(values, ctor, axes):
    non_empty = [v for v in values if dr.width(getattr(v, axes[0])) > 0]
    if not non_empty:
        return dr.zeros(ctor, 0)
    return ctor(*(dr.concat([getattr(v, a) for v in non_empty]) for a in axes))


def concat_complex(values) -> wt.Complex2f:
    return _concat_components(values, wt.Complex2f, ("real", "imag"))


def concat_points(values) -> wt.Point3f:
    return _concat_components(values, wt.Point3f, ("x", "y", "z"))


def concat_vectors(values) -> wt.Vector3f:
    return _concat_components(values, wt.Vector3f, ("x", "y", "z"))


def concat_ints(values):
    return concat_arrays(wt.Int32, values)


def concat_uints(values):
    return concat_arrays(wt.UInt32, values)


def concat_floats(values):
    return concat_arrays(wt.Float, values)


def _broadcast(value, width):
    return value if dr.width(value) == width else dr.repeat(value, width)


broadcast_float = _broadcast
broadcast_int = _broadcast
broadcast_complex = _broadcast
broadcast_point = _broadcast
broadcast_vector = _broadcast


def broadcast_vector_dict(values: Mapping[str, object], width: int) -> dict[str, object]:
    """Broadcast every vector/Jones dictionary component to ``width`` lanes."""
    return {str(key): broadcast_complex(value, width) for key, value in dict(values).items()}


def _repeat_typed(value, width, *, caster=None):
    typed = value if caster is None else caster(value)
    current_width = int(dr.width(typed))
    if current_width == int(width):
        return typed
    if current_width == 1:
        return dr.repeat(typed, int(width))
    raise ValueError(f"Expected scalar or width {width} input, got width {current_width}.")


def repeat_float(value, width):
    if isinstance(value, bool):
        return _repeat_typed(value, width, caster=wt.Bool)
    if isinstance(value, int):
        return _repeat_typed(float(value), width, caster=wt.Float)
    if isinstance(value, float):
        return _repeat_typed(value, width, caster=wt.Float)
    return _repeat_typed(value, width)


def repeat_int(value, width):
    return _repeat_typed(value, width, caster=wt.Int32 if isinstance(value, int) else None)


def repeat_complex(value, width):
    if isinstance(value, complex):
        return _repeat_typed(wt.Complex2f(value.real, value.imag), width)
    return _repeat_typed(value, width)


def complex_abs_sqr(field):
    return field.real * field.real + field.imag * field.imag


def eval_complex(field):
    dr.eval(field.real, field.imag)
    return field


def safe_normalize(vec, eps: float = EPS):
    """Eps-floored normalization; returns zero on degenerate input.

    Differs from :func:`witwin.channel.core.physics.polarization.safe_normalize_with_fallback`,
    which substitutes a normalized fallback vector instead of zero.
    """
    norm = dr.norm(vec)
    return dr.select(norm > eps, vec / (norm + eps), vec * 0.0)


_TIMING_SYNC_ENABLED: bool = False


def set_timing(enabled: bool) -> None:
    global _TIMING_SYNC_ENABLED
    _TIMING_SYNC_ENABLED = bool(enabled)


def timing_enabled() -> bool:
    return _TIMING_SYNC_ENABLED


_COMPONENT_GROUPS = (("x", "y", "z"), ("u", "v"), ("m00", "m01", "m10", "m11"))


def collect_eval_targets(value, targets: list[object]) -> None:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    if isinstance(value, Mapping):
        for item in value.values():
            collect_eval_targets(item, targets)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            collect_eval_targets(item, targets)
        return

    # DrJit's __getattr__ raises RuntimeError for unknown components, which
    # the three-arg getattr form does not catch.
    def _attr(target, name):
        try:
            return getattr(target, name)
        except Exception:
            return None

    for axes in _COMPONENT_GROUPS:
        components = [_attr(value, a) for a in axes]
        if all(c is not None for c in components):
            for c in components:
                collect_eval_targets(c, targets)
            return

    real, imag = _attr(value, "real"), _attr(value, "imag")
    if real is not None and imag is not None and real is not value:
        collect_eval_targets(real, targets)
        collect_eval_targets(imag, targets)
        return

    if dr.is_array_v(type(value)):
        targets.append(value)


def eval_nested(*values):
    targets: list[object] = []
    for value in values:
        collect_eval_targets(value, targets)
    if targets:
        dr.eval(*targets)
    return values[0] if len(values) == 1 else values


def sync_thread(*, force: bool = False) -> None:
    if force or _TIMING_SYNC_ENABLED:
        dr.sync_thread()


def eval_and_sync(*values, force_sync: bool = False):
    eval_nested(*values)
    sync_thread(force=force_sync)
    return values[0] if len(values) == 1 else values


def barrier(*values):
    if _TIMING_SYNC_ENABLED:
        return eval_and_sync(*values, force_sync=True)
    return values[0] if len(values) == 1 else values


__all__ = [
    "barrier",
    "broadcast_complex",
    "broadcast_float",
    "broadcast_int",
    "broadcast_point",
    "broadcast_vector",
    "broadcast_vector_dict",
    "collect_eval_targets",
    "complex_abs_sqr",
    "complex_zero",
    "concat_arrays",
    "concat_complex",
    "concat_floats",
    "concat_ints",
    "concat_points",
    "concat_uints",
    "concat_vectors",
    "empty_complex",
    "empty_point3",
    "empty_vector3",
    "empty_vector3u",
    "eval_and_sync",
    "eval_complex",
    "eval_nested",
    "gather_point3",
    "gather_vector3",
    "mask_count",
    "safe_normalize",
    "scalar",
    "set_timing",
    "sync_thread",
    "timing_enabled",
    "zeros_point3",
    "zeros_vector3",
]
