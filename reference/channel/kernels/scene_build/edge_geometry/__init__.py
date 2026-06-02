"""Batch edge geometry computation."""

def _use_native():
    try:
        from witwin.channel._native import extension_available

        return extension_available()
    except ImportError:
        return False

if _use_native():
    from .native_impl import batch_edge_geometry
else:
    from .drjit_impl import batch_edge_geometry

