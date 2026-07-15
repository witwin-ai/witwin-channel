"""Validated access to Channel Native extension symbols."""

from __future__ import annotations

import sys

from .extension import _load_native_extension


class NativeSymbolError(RuntimeError):
    """A required Channel Native extension symbol is unavailable."""


def native_extension() -> object:
    """Return the process-cached, ABI-validated native extension."""

    return _load_native_extension()


def _required_symbol(extension: object, name: str) -> object:
    if extension is None or not hasattr(extension, name):
        raise NativeSymbolError(f"_channel_native.{name} CUDA kernel is required")
    return getattr(extension, name)


_native_symbols = sys.modules[__name__]


def _required_native_op(name: str):
    return _native_symbols._required_symbol(native_extension(), name)


def required_symbol(name: str) -> object:
    """Return a required symbol or raise :class:`NativeSymbolError`."""

    return _required_symbol(native_extension(), name)


def optional_symbol(name: str) -> object | None:
    """Return an optional symbol, or ``None`` when that symbol is absent."""

    extension = native_extension()
    if extension is None:
        return None
    return getattr(extension, name, None)


def has_symbol(name: str) -> bool:
    """Report whether the validated extension exposes ``name``."""

    extension = native_extension()
    return extension is not None and hasattr(extension, name)


__all__ = [
    "NativeSymbolError",
    "has_symbol",
    "native_extension",
    "optional_symbol",
    "required_symbol",
]
