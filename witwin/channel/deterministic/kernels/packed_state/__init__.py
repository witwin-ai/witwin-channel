"""Packed state buffer operations for gather, concat, and subset."""

__all__ = [
    "build_diffraction_path_slots",
    "gather_state_arrays",
    "gather_field_evaluation_state_fields",
    "concat_state_arrays",
    "subset_state_arrays",
    "gather_inserted_reflection_state_fields",
]


def _use_native():
    try:
        from witwin.channel._native.deterministic import NativeExtension

        return NativeExtension.extension_available()
    except ImportError:
        return False


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _use_native():
        from . import native_impl as _impl
    else:
        from . import drjit_impl as _impl
    return getattr(_impl, name)

