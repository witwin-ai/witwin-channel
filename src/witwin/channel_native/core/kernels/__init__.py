"""Native kernel loading and facade helpers."""

from importlib import import_module
from typing import TYPE_CHECKING

from .extension import build_info

if TYPE_CHECKING:
    from . import ops

__all__ = ["build_info", "ops"]


def __getattr__(name: str):
    if name != "ops":
        raise AttributeError(name)

    ops = import_module(f"{__name__}.ops")
    globals()[name] = ops
    return ops
