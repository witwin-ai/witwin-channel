"""Compatibility wrapper for reflection grid accumulation helpers."""

from ...kernels.monitors.field.reflection_grid import drjit_impl as _drjit_impl

_SCATTER_CHUNK_RAY_THRESHOLD = _drjit_impl._SCATTER_CHUNK_RAY_THRESHOLD
_SCATTER_CHUNK_SIZE = _drjit_impl._SCATTER_CHUNK_SIZE
_DEFAULT_SCATTER_CHUNK_RAY_THRESHOLD = _SCATTER_CHUNK_RAY_THRESHOLD
_DEFAULT_SCATTER_CHUNK_SIZE = _SCATTER_CHUNK_SIZE


def _sync_chunked_scatter_config():
    _drjit_impl._SCATTER_CHUNK_RAY_THRESHOLD = _SCATTER_CHUNK_RAY_THRESHOLD
    _drjit_impl._SCATTER_CHUNK_SIZE = _SCATTER_CHUNK_SIZE


def chunked_scatter_override_active() -> bool:
    return (
        _SCATTER_CHUNK_RAY_THRESHOLD != _DEFAULT_SCATTER_CHUNK_RAY_THRESHOLD
        or _SCATTER_CHUNK_SIZE != _DEFAULT_SCATTER_CHUNK_SIZE
    )


def extract_plane_components(*args, **kwargs):
    _sync_chunked_scatter_config()
    return _drjit_impl.extract_plane_components(*args, **kwargs)


def prepare_plane_intersections(*args, **kwargs):
    _sync_chunked_scatter_config()
    return _drjit_impl.prepare_plane_intersections(*args, **kwargs)


def run_dda_traversal(*args, **kwargs):
    _sync_chunked_scatter_config()
    return _drjit_impl.run_dda_traversal(*args, **kwargs)


def intersect_and_scatter(*args, **kwargs):
    _sync_chunked_scatter_config()
    return _drjit_impl.intersect_and_scatter(*args, **kwargs)


__all__ = [
    "_SCATTER_CHUNK_RAY_THRESHOLD",
    "_SCATTER_CHUNK_SIZE",
    "_DEFAULT_SCATTER_CHUNK_RAY_THRESHOLD",
    "_DEFAULT_SCATTER_CHUNK_SIZE",
    "chunked_scatter_override_active",
    "extract_plane_components",
    "intersect_and_scatter",
    "prepare_plane_intersections",
    "run_dda_traversal",
]
