from __future__ import annotations


class RayDNScene:
    """Opaque wrapper for a native RayDN scene/cache handle."""

    def __init__(self, handle: object | None = None) -> None:
        self._handle = handle

    @property
    def handle(self) -> object | None:
        return self._handle
