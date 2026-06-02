"""Batch surface coplanarity check."""

def _use_native():
    try:
        from witwin.channel._native import extension_available

        return extension_available()
    except ImportError:
        return False

if _use_native():
    from .native_impl import batch_coplanarity_check
else:
    from .drjit_impl import batch_coplanarity_check

