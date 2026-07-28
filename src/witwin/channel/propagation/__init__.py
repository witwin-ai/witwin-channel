"""Internal propagation namespace with lazy owner exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "EvaluatedPaths",
    "PathFields",
    "PathGeometry",
    "PathTopology",
    "evaluate_enumerated_paths",
]


def __getattr__(name: str):
    if name == "evaluate_enumerated_paths":
        value = import_module(
            "witwin.channel.propagation.enumerated.engine"
        ).evaluate_enumerated_paths
    elif name in {"EvaluatedPaths", "PathFields", "PathGeometry", "PathTopology"}:
        value = getattr(import_module("witwin.channel.propagation.rows"), name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
