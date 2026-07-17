"""Native handle normalization independent of scene and propagation owners."""

from __future__ import annotations


def _raydn_scene_handle_id(handle: object) -> int:
    if isinstance(handle, int):
        return handle
    value = getattr(handle, "handle", None)
    if isinstance(value, int):
        return value
    handle_fn = getattr(handle, "handle", None)
    if callable(handle_fn):
        value = handle_fn()
        if isinstance(value, int):
            return value
    raise TypeError("RayDN scene handle must be an int or expose handle() -> int")


__all__ = ["_raydn_scene_handle_id"]
