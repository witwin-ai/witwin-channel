"""Trace-layer package exports."""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "Tracer": "witwin.channel.trace.tracer",
    "compute_los_field": "witwin.channel.trace.los",
    "los_blocked": "witwin.channel.trace.los",
    "compute_reflection_field": "witwin.channel.trace.reflection",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
