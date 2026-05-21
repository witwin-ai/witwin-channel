from __future__ import annotations

from dataclasses import dataclass


_BOUNDARY_EDGE_POLICIES = {"exclude", "half_plane"}
_EDGE_SELECTION_MODES = {"vertical_only", "all_edges"}


@dataclass(slots=True)
class EdgePolicy:
    """Solver-owned diffraction-edge selection policy."""

    vertical_ratio: float = 0.7
    edge_selection_mode: str = "all_edges"
    edge_diffraction: bool | None = None
    boundary_edge_policy: str | None = None

    def __post_init__(self) -> None:
        edge_selection_mode = str(self.edge_selection_mode)
        if edge_selection_mode not in _EDGE_SELECTION_MODES:
            raise ValueError(
                f"edge_selection_mode must be one of {sorted(_EDGE_SELECTION_MODES)}; "
                f"got {edge_selection_mode!r}."
            )
        requested_edge_diffraction = (
            None if self.edge_diffraction is None else bool(self.edge_diffraction)
        )
        resolved_boundary = (
            "half_plane" if requested_edge_diffraction is not False else "exclude"
        ) if self.boundary_edge_policy is None else str(self.boundary_edge_policy)
        if resolved_boundary not in _BOUNDARY_EDGE_POLICIES:
            raise ValueError(
                f"boundary_edge_policy must be one of {sorted(_BOUNDARY_EDGE_POLICIES)}; "
                f"got {resolved_boundary!r}."
            )
        resolved_edge_diffraction = resolved_boundary == "half_plane"
        if (
            requested_edge_diffraction is not None
            and requested_edge_diffraction != resolved_edge_diffraction
        ):
            raise ValueError(
                f"edge_diffraction={requested_edge_diffraction!r} conflicts with "
                f"boundary_edge_policy={resolved_boundary!r}."
            )
        self.vertical_ratio = float(self.vertical_ratio)
        self.edge_selection_mode = edge_selection_mode
        self.boundary_edge_policy = resolved_boundary
        self.edge_diffraction = resolved_edge_diffraction

    @property
    def vertical_only(self) -> bool:
        return self.edge_selection_mode == "vertical_only"

    @property
    def cache_key(self) -> tuple[float, str, str]:
        return (
            self.vertical_ratio,
            self.edge_selection_mode,
            self.boundary_edge_policy or "half_plane",
        )


DEFAULT_EDGE_POLICY = EdgePolicy()


def coerce_edge_policy(edge_policy: EdgePolicy | None = None, **overrides) -> EdgePolicy:
    if edge_policy is not None and overrides:
        raise TypeError("edge_policy cannot be combined with edge policy overrides.")
    if edge_policy is not None:
        if not isinstance(edge_policy, EdgePolicy):
            raise TypeError("edge_policy must be an EdgePolicy instance.")
        return edge_policy
    if not overrides:
        return DEFAULT_EDGE_POLICY
    return EdgePolicy(**overrides)


__all__ = ["DEFAULT_EDGE_POLICY", "EdgePolicy", "coerce_edge_policy"]
