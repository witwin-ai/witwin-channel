from __future__ import annotations

from dataclasses import dataclass


_BOUNDARY_EDGE_POLICIES = {"exclude", "half_plane"}
_EDGE_SELECTION_MODES = {"vertical_only", "all_edges"}


@dataclass(slots=True)
class EdgePolicy:
    vertical_ratio: float = 0.7
    edge_selection_mode: str = "all_edges"
    edge_diffraction: bool | None = True
    boundary_edge_policy: str | None = None

    def __post_init__(self) -> None:
        mode = str(self.edge_selection_mode)
        if mode not in _EDGE_SELECTION_MODES:
            raise ValueError(f"edge_selection_mode must be one of {sorted(_EDGE_SELECTION_MODES)}")
        requested = None if self.edge_diffraction is None else bool(self.edge_diffraction)
        boundary = (
            "half_plane" if requested is not False else "exclude"
        ) if self.boundary_edge_policy is None else str(self.boundary_edge_policy)
        if boundary not in _BOUNDARY_EDGE_POLICIES:
            raise ValueError(f"boundary_edge_policy must be one of {sorted(_BOUNDARY_EDGE_POLICIES)}")
        resolved = boundary == "half_plane"
        if requested is not None and requested != resolved:
            raise ValueError(
                f"edge_diffraction={requested!r} conflicts with boundary_edge_policy={boundary!r}"
            )
        self.vertical_ratio = float(self.vertical_ratio)
        self.edge_selection_mode = mode
        self.edge_diffraction = resolved
        self.boundary_edge_policy = boundary

    @property
    def vertical_only(self) -> bool:
        return self.edge_selection_mode == "vertical_only"

    @property
    def cache_key(self) -> tuple[float, str, str]:
        return (self.vertical_ratio, self.edge_selection_mode, self.boundary_edge_policy or "half_plane")


DEFAULT_EDGE_POLICY = EdgePolicy()


__all__ = ["DEFAULT_EDGE_POLICY", "EdgePolicy"]
