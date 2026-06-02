from __future__ import annotations

from collections.abc import Mapping
import os

import numpy as np

import drjit as dr
import witwin as wt

from .constants import EPS


# Zero/empty array constructors.
class ArrayInit:
    """Zero/empty array constructors."""

    @staticmethod
    def empty_complex() -> wt.Complex2f:
        return wt.Complex2f(dr.zeros(wt.Float, 0), dr.zeros(wt.Float, 0))

    @staticmethod
    def empty_point3():
        return wt.Point3f(dr.zeros(wt.Float, 0), dr.zeros(wt.Float, 0), dr.zeros(wt.Float, 0))

    @staticmethod
    def empty_vector3():
        return wt.Vector3f(dr.zeros(wt.Float, 0), dr.zeros(wt.Float, 0), dr.zeros(wt.Float, 0))

    @staticmethod
    def empty_vector3u():
        return wt.Vector3u(dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0))

    @staticmethod
    def zeros_point3(width: int):
        return wt.Point3f(
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
        )

    @staticmethod
    def zeros_vector3(width: int):
        return wt.Vector3f(
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
        )

    @staticmethod
    def complex_zero(width):
        return wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))


# Concatenate non-empty DrJit arrays.
class Concat:
    """Concatenate non-empty DrJit arrays."""

    @staticmethod
    def complex(values) -> wt.Complex2f:
        non_empty = [value for value in values if dr.width(value.real) > 0]
        if len(non_empty) == 0:
            return ArrayInit.empty_complex()
        return wt.Complex2f(
            dr.concat([value.real for value in non_empty]),
            dr.concat([value.imag for value in non_empty]),
        )

    @staticmethod
    def points(values):
        non_empty = [value for value in values if dr.width(value.x) > 0]
        if len(non_empty) == 0:
            return ArrayInit.empty_point3()
        return wt.Point3f(
            dr.concat([value.x for value in non_empty]),
            dr.concat([value.y for value in non_empty]),
            dr.concat([value.z for value in non_empty]),
        )

    @staticmethod
    def vectors(values):
        non_empty = [value for value in values if dr.width(value.x) > 0]
        if len(non_empty) == 0:
            return ArrayInit.empty_vector3()
        return wt.Vector3f(
            dr.concat([value.x for value in non_empty]),
            dr.concat([value.y for value in non_empty]),
            dr.concat([value.z for value in non_empty]),
        )

    @staticmethod
    def ints(values):
        non_empty = [value for value in values if dr.width(value) > 0]
        if len(non_empty) == 0:
            return dr.zeros(wt.Int32, 0)
        return dr.concat(non_empty)

    @staticmethod
    def uints(values):
        non_empty = [value for value in values if dr.width(value) > 0]
        if len(non_empty) == 0:
            return dr.zeros(wt.UInt32, 0)
        return dr.concat(non_empty)

    @staticmethod
    def floats(values):
        non_empty = [value for value in values if dr.width(value) > 0]
        if len(non_empty) == 0:
            return dr.zeros(wt.Float, 0)
        return dr.concat(non_empty)

    @staticmethod
    def arrays(array_type, values):
        """Concatenate non-empty DrJit arrays of the same type."""
        non_empty = [value for value in values if dr.width(value) > 0]
        if len(non_empty) == 0:
            return dr.zeros(array_type, 0)
        return dr.concat(non_empty)


# Broadcast/repeat values to target width.
class Broadcast:
    """Broadcast/repeat values to target width."""

    @staticmethod
    def point(pt, width):
        if dr.width(pt.x) == width:
            return pt
        return wt.Point3f(
            Broadcast.scalar(pt.x, width),
            Broadcast.scalar(pt.y, width),
            Broadcast.scalar(pt.z, width),
        )

    @staticmethod
    def vector(vec, width):
        if dr.width(vec.x) == width:
            return vec
        return wt.Vector3f(
            Broadcast.scalar(vec.x, width),
            Broadcast.scalar(vec.y, width),
            Broadcast.scalar(vec.z, width),
        )

    @staticmethod
    def scalar(value, width):
        return value if dr.width(value) == width else dr.repeat(value, width)

    @staticmethod
    def complex(value, width):
        if dr.width(value.real) == width:
            return value
        return wt.Complex2f(dr.repeat(value.real, width), dr.repeat(value.imag, width))

    @staticmethod
    def int_val(value, width):
        return value if dr.width(value) == width else dr.repeat(value, width)

    @staticmethod
    def vector_dict(vec, width):
        def _repeat_component(component):
            return component if dr.width(component.real) == width else wt.Complex2f(
                dr.repeat(component.real, width),
                dr.repeat(component.imag, width),
            )

        return {
            "x": _repeat_component(vec["x"]),
            "y": _repeat_component(vec["y"]),
            "z": _repeat_component(vec["z"]),
        }


def broadcast_point(point, width):
    typed_point = wt.Point3f(point.x, point.y, point.z)
    return Broadcast.point(typed_point, width)


def broadcast_vector(vector, width):
    typed_vector = wt.Vector3f(vector.x, vector.y, vector.z)
    return Broadcast.vector(typed_vector, width)


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


def _collect_eval_targets(value, targets: list[object]) -> None:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _collect_eval_targets(item, targets)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_eval_targets(item, targets)
        return

    def _attr_or_none(target, attr_name):
        try:
            return getattr(target, attr_name)
        except Exception:
            return None

    xyz = [_attr_or_none(value, attr_name) for attr_name in ("x", "y", "z")]
    if all(component is not None for component in xyz):
        for component in xyz:
            _collect_eval_targets(component, targets)
        return

    uv = [_attr_or_none(value, attr_name) for attr_name in ("u", "v")]
    if all(component is not None for component in uv):
        for component in uv:
            _collect_eval_targets(component, targets)
        return

    matrix = [_attr_or_none(value, attr_name) for attr_name in ("m00", "m01", "m10", "m11")]
    if all(component is not None for component in matrix):
        for component in matrix:
            _collect_eval_targets(component, targets)
        return

    real = _attr_or_none(value, "real")
    imag = _attr_or_none(value, "imag")
    if real is not None and imag is not None:
        if real is not value:
            _collect_eval_targets(real, targets)
            _collect_eval_targets(imag, targets)
            return

    try:
        dr.width(value)
    except Exception:
        return
    targets.append(value)


# Evaluation and timing synchronization.
class EvalSync:
    """Evaluation and timing synchronization."""

    @staticmethod
    def timing_enabled() -> bool:
        value = os.environ.get("WITWIN_BENCHMARK_SYNC_TIMING", "")
        return value.lower() not in {"", "0", "false", "no", "off"}

    @staticmethod
    def nested(*values):
        targets: list[object] = []
        for value in values:
            _collect_eval_targets(value, targets)
        if targets:
            dr.eval(*targets)
        if len(values) == 1:
            return values[0]
        return values

    @staticmethod
    def and_sync(*values, force_sync: bool = False):
        EvalSync.nested(*values)
        EvalSync.sync(force=force_sync)
        if len(values) == 1:
            return values[0]
        return values

    @staticmethod
    def sync(*, force: bool = False) -> None:
        if not force and not EvalSync.timing_enabled():
            return
        if hasattr(dr, "sync_thread"):
            dr.sync_thread()

    @staticmethod
    def barrier(*values):
        if EvalSync.timing_enabled():
            return EvalSync.and_sync(*values, force_sync=True)
        if len(values) == 1:
            return values[0]
        return values


# Gather from structured arrays.
class Gather:
    """Gather from structured arrays."""

    @staticmethod
    def point3(source, index):
        component_type = type(source.x)
        return type(source)(
            dr.gather(component_type, source.x, index),
            dr.gather(component_type, source.y, index),
            dr.gather(component_type, source.z, index),
        )

    @staticmethod
    def vector3(source, index):
        component_type = type(source.x)
        return type(source)(
            dr.gather(component_type, source.x, index),
            dr.gather(component_type, source.y, index),
            dr.gather(component_type, source.z, index),
        )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def array_scalar(value) -> float:
    """Extract a scalar from a DrJit or numpy array."""
    if hasattr(value, "__len__") and dr.width(value) > 0:
        return float(np.array(value).flat[0])
    return float(value)


def mask_count(mask) -> int:
    """Count True elements in a DrJit Bool array."""
    return int(array_scalar(dr.sum(dr.select(mask, wt.UInt32(1), wt.UInt32(0)))))


def complex_abs_sqr(field):
    return field.real * field.real + field.imag * field.imag


def eval_complex(field):
    dr.eval(field.real, field.imag)
    return field


def safe_normalize(vec, eps: float = EPS):
    norm = dr.norm(vec)
    normalized = vec / (norm + eps)
    return dr.select(norm > eps, normalized, vec * 0.0)


__all__ = [
    "ArrayInit",
    "Broadcast",
    "Concat",
    "EvalSync",
    "Gather",
    "array_scalar",
    "complex_abs_sqr",
    "eval_complex",
    "mask_count",
    "safe_normalize",
]
