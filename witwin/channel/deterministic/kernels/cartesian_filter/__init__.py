"""Fused Cartesian filter for higher-order diffraction state generation."""

def _use_native():
    try:
        from witwin.channel._native.deterministic import NativeExtension

        return NativeExtension.extension_available()
    except ImportError:
        return False

if _use_native():
    from .native_impl import cartesian_filter_bruteforce, compact_index_pairs
else:
    from .drjit_impl import cartesian_filter_bruteforce, compact_index_pairs

