"""Modular diffraction solver package.

The package namespace exposes the public API lazily and still supports
selected internal helper imports used by tests and validation code. This
avoids importing the full diffraction stack when only low-level modules
such as ``constants`` are needed during kernel initialization.
"""

from __future__ import annotations

from importlib import import_module


_SEARCH_MODULES = (
    "witwin.channel.trace.diffraction.constants",
    "witwin.channel.trace.diffraction.geometry",
    "witwin.channel.trace.diffraction.field",
    "witwin.channel.trace.diffraction.builders",
    "witwin.channel.trace.diffraction.state",
    "witwin.channel.trace.diffraction.material_ops",
    "witwin.channel.trace.diffraction.operator",
    "witwin.channel.trace.diffraction.utd",
    "witwin.channel.trace.diffraction.api",
)

__all__ = [
    "compute_diffraction_field",
    "compute_diffraction_order_breakdown",
    "preload_diffraction_edges",
]


def __getattr__(name: str):
    for module_name in _SEARCH_MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
