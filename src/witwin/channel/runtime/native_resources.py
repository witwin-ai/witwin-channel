"""Typed native resource normalization at Python dispatch boundaries."""

from __future__ import annotations


def _rayd_scene_resource(value: object) -> object:
    """Return a typed RayD scene resource; integer handles are forbidden."""

    if isinstance(value, int):
        raise TypeError("RayD scene operations require a typed scene resource")
    require = getattr(value, "require_resource", None)
    if callable(require):
        resource = require()
        if isinstance(resource, int):
            raise TypeError("RayD scene operations require a typed scene resource")
        return resource
    return value


__all__ = ["_rayd_scene_resource"]
