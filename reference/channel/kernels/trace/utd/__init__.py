"""UTD diffraction kernel entrypoints."""

from witwin.channel._native import extension_available

__all__ = [
    "utd_accumulate_forward",
    "utd_accumulate_backward",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if extension_available():
        from . import native_impl as _impl
    else:
        from . import drjit_impl as _impl
    return getattr(_impl, name)
