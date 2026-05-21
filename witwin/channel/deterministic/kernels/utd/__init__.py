"""UTD diffraction kernel entrypoints."""

from witwin.channel._native.deterministic import NativeExtension

__all__ = [
    "utd_accumulate_forward",
    "utd_pair_vectors",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not NativeExtension.extension_available():
        raise RuntimeError(
            "UTD diffraction requires the deterministic radiomap native extension."
        )
    from . import native_impl as _impl
    return getattr(_impl, name)
