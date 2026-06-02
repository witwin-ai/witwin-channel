"""Backend adapters for the wedge runtime."""

from .rayd_scene import RayDSceneAdapter, is_rayd_scene_like

__all__ = [
    "RayDSceneAdapter",
    "is_rayd_scene_like",
]
