# Copyright Xingyu Chen.
# Top-level Witwin namespace package.

"""Top-level Witwin namespace package."""

from __future__ import annotations

from importlib import import_module
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_LAZY_EXPORTS = {
    "channel": "witwin.channel",
    "core": "witwin.core",
}

__all__ = [
    "channel",
    "core",
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    globals()[name] = module
    return module