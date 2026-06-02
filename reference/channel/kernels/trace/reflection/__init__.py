"""Reflection accumulation kernel entrypoints."""

from witwin.channel._native import extension_available

if extension_available():
    from .native_impl import reflection_accumulate_forward
else:
    from .drjit_impl import reflection_accumulate_forward
