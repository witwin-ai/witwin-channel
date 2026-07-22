"""Structural propagation configuration contracts."""

from __future__ import annotations

from typing import Protocol


class TopologyConfig(Protocol):
    max_depth: int
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str]
    max_paths: int | None
    max_paths_scope: str


TopologyConfig.__module__ = "witwin.channel.propagation.models.contracts"
