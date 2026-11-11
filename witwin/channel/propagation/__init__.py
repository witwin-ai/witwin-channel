# Copyright Xingyu Chen.
# Internal propagation namespace with lazy owner exports.

"""Internal propagation namespace with lazy owner exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from witwin.channel.propagation.enumerated import evaluate_enumerated_paths
    from witwin.channel.propagation.rows import (
        EvaluatedPaths,
        PathFields,
        PathGeometry,
        PathTopology,
    )

__all__ = [
    "EvaluatedPaths",
    "PathFields",
    "PathGeometry",
    "PathTopology",
    "evaluate_enumerated_paths",
]


def __getattr__(name: str) -> object:
    if name == "evaluate_enumerated_paths":
        value = import_module(
            "witwin.channel.propagation.enumerated"
        ).evaluate_enumerated_paths
    elif name in {"EvaluatedPaths", "PathFields", "PathGeometry", "PathTopology"}:
        value = getattr(import_module("witwin.channel.propagation.rows"), name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
