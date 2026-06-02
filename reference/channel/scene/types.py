"""Typed structures for RFDT geometry and tracing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence, Tuple

import witwin as wt


class Edge2D(NamedTuple):
    p0: wt.Vector2f
    p1: wt.Vector2f
    normal: wt.Vector2f
    name: str


class Corner2D(NamedTuple):
    position: wt.Vector2f
    face0_point: wt.Vector2f
    face_n_point: wt.Vector2f
    name: str
    edge_info: Optional["DiffractionPoint"]


@dataclass
class VerticalEdge:
    vertex_indices: Tuple[int, int]
    p0: wt.Point3f
    p1: wt.Point3f
    adjacent_faces: Sequence[int]
    is_boundary: bool
    edge_vector: wt.Vector3f
    length: wt.Float
    global_index: int = -1
    normal_2d: Optional[wt.Vector2f] = None
    wedge_n: Optional[wt.Float] = None
    face_normals_3d: Optional[List[wt.Vector3f]] = None


@dataclass(frozen=True)
class DiffractionPoint:
    position: wt.Point3f
    edge_vector: wt.Vector3f
    length: wt.Float
    wedge_n: wt.Float
    face_normals_3d: Sequence[wt.Vector3f]
    adjacent_faces: Sequence[int] = ()
    vertex_indices: Tuple[int, int] = (-1, -1)
    global_index: int = -1
    line_min: Optional[wt.Float] = None
    line_max: Optional[wt.Float] = None


__all__ = [
    "Edge2D",
    "Corner2D",
    "VerticalEdge",
    "DiffractionPoint",
]

