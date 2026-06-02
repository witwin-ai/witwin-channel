"""Finite-wedge-only validation helpers."""

from __future__ import annotations


def _require_bounds(line_min, line_max, *, context: str, min_name: str, max_name: str):
    if line_min is None or line_max is None:
        raise RuntimeError(
            f"{context} requires finite-wedge {min_name} and {max_name}."
        )
    return line_min, line_max


def require_edge_data_line_bounds(edge_data, *, context: str):
    line_min = None if edge_data is None else edge_data.get("line_min")
    line_max = None if edge_data is None else edge_data.get("line_max")
    return _require_bounds(
        line_min,
        line_max,
        context=context,
        min_name="line_min",
        max_name="line_max",
    )


def require_edge_state_line_bounds(edge_state, *, context: str):
    edge_line_min = None if edge_state is None else edge_state.get("edge_line_min")
    edge_line_max = None if edge_state is None else edge_state.get("edge_line_max")
    return _require_bounds(
        edge_line_min,
        edge_line_max,
        context=context,
        min_name="edge_line_min",
        max_name="edge_line_max",
    )
