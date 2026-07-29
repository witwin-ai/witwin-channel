# Copyright Xingyu Chen.
# Core-world to Channel-runtime compilation boundary.

"""Core-world to Channel-runtime compilation boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from witwin.core import Scene, SceneSnapshot

if TYPE_CHECKING:
    from .compiler import CompiledScene


def compile(scene_or_snapshot: Scene | SceneSnapshot, *, reference_frequency_hz) -> "CompiledScene":
    from .compiler import compile as compile_scene

    return compile_scene(
        scene_or_snapshot,
        reference_frequency_hz=reference_frequency_hz,
    )


def clear_compile_cache() -> None:
    from .compiler import clear_compile_cache as clear

    clear()


def __getattr__(name: str):
    if name == "CompiledScene":
        from .compiler import CompiledScene

        return CompiledScene
    raise AttributeError(name)

__all__ = [
    "CompiledScene",
    "clear_compile_cache",
    "compile",
]