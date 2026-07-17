"""Compatibility imports for the centralized runtime extension loader."""

from witwin.channel_native.runtime.extension import build_info
from witwin.channel_native.runtime.symbols import native_extension

__all__ = ["build_info", "native_extension"]
