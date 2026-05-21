"""Deterministic radiomap solver package."""

from __future__ import annotations

from witwin.channel._native.deterministic import NativeExtension
from .config import Config, Tuning
from .field import FieldResult, FieldSpec, solve_field
from witwin.channel.core.results import RadioMapResult
from .solver import solve

native_extension_available = NativeExtension.native_extension_available

__all__ = [
    "Config",
    "FieldResult",
    "FieldSpec",
    "NativeExtension",
    "RadioMapResult",
    "Tuning",
    "native_extension_available",
    "solve",
    "solve_field",
]
