# Copyright Xingyu Chen.
# Imports Mitsuba/Sionna XML, PLY, OBJ and glTF meshes as witwin.core structures.

"""Import Mitsuba/Sionna XML, PLY, OBJ and glTF scenes as ``witwin.core`` structures."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from os import PathLike
from pathlib import Path
import struct
from typing import Any
import xml.etree.ElementTree as ElementTree
import zlib

import numpy as np
import torch

from witwin.core import AntennaState, MaterialSpec, Mesh, PhysicalMaterial, Scene, Structure

from witwin.channel.presets import itu_material

__all__ = [
    "ImportedScene",
    "MeshData",
    "StructureReport",
    "build_structures",
    "load_mesh",
    "load_mitsuba",
    "read_gltf",
    "read_obj",
    "read_ply",
    "weld_vertices",
]

StrPath = str | PathLike[str]
MaterialMap = Mapping[str | None, MaterialSpec]
DEFAULT_WELD_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class MeshData:
    """One triangle mesh read from a file, in world coordinates (metres)."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    material: str | None = None

    def __post_init__(self) -> None:
        vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"mesh {self.name!r}: vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"mesh {self.name!r}: faces must have shape (F, 3)")
        if not np.isfinite(vertices).all():
            raise ValueError(f"mesh {self.name!r}: vertices must be finite")
        if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
            raise ValueError(f"mesh {self.name!r}: face index out of range")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    def transformed(self, matrix: np.ndarray) -> MeshData:
        """Return this mesh with a 4x4 affine ``matrix`` applied to its vertices."""

        linear, offset = matrix[:3, :3], matrix[:3, 3]
        faces = self.faces if np.linalg.det(linear) >= 0.0 else self.faces[:, ::-1]
        return MeshData(self.name, self.vertices @ linear.T + offset, faces, self.material)


@dataclass(frozen=True)
class StructureReport:
    """What import did to the meshes that became one structure."""

    name: str
    material: str | None
    source_meshes: tuple[str, ...]
    input_vertices: int
    output_vertices: int
    welded_vertices: int
    removed_faces: int
    faces: int


@dataclass(frozen=True)
class ImportedScene:
    """Core structures and materials produced by an importer, plus an import report."""

    structures: tuple[Structure, ...]
    materials: Mapping[str | None, MaterialSpec]
    report: tuple[StructureReport, ...]
    warnings: tuple[str, ...] = ()

    @property
    def vertex_count(self) -> int:
        return sum(entry.output_vertices for entry in self.report)

    @property
    def face_count(self) -> int:
        return sum(entry.faces for entry in self.report)

    @property
    def welded_vertex_count(self) -> int:
        return sum(entry.welded_vertices for entry in self.report)

    def to_scene(
        self, *, endpoints: Iterable[AntennaState] = (), metadata: Mapping[str, Any] | None = None,
    ) -> Scene:
        """Return a ``witwin.core.Scene`` holding these structures and ``endpoints``."""

        return Scene(structures=self.structures, endpoints=tuple(endpoints), metadata=metadata)

    def summary(self) -> str:
        """Return a short human-readable import summary."""

        return (
            f"{len(self.structures)} structures, {self.vertex_count} vertices, "
            f"{self.face_count} faces; welded {self.welded_vertex_count} duplicate vertices, "
            f"removed {sum(entry.removed_faces for entry in self.report)} degenerate faces"
        )


# --------------------------------------------------------------------------- welding


def weld_vertices(
    vertices: np.ndarray, faces: np.ndarray, *, tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Merge vertices closer than ``tolerance`` on a grid and drop degenerate faces.

    Returns ``(vertices, faces, welded_vertex_count, removed_face_count)``. Two
    vertices weld when they round to the same ``tolerance``-sized grid cell; a
    face that loses a distinct corner or has zero area is removed, and vertices
    no face references are dropped.
    """

    if not (math.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError("weld tolerance must be finite and positive")
    keys = np.round(vertices / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    welded = int(vertices.shape[0] - first.shape[0])
    merged_faces = inverse[faces]
    merged_vertices = vertices[first]
    keep = _non_degenerate(merged_vertices, merged_faces)
    removed = int(faces.shape[0] - int(keep.sum()))
    kept_faces = merged_faces[keep]
    used, compact = np.unique(kept_faces.reshape(-1), return_inverse=True)
    return merged_vertices[used], compact.reshape(-1, 3), welded, removed


def _non_degenerate(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    distinct = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    corners = vertices[faces]
    area2 = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    return distinct & (area2 > 0.0)


def _concat(meshes: Sequence[MeshData]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.cumsum([0, *(mesh.vertices.shape[0] for mesh in meshes)])
    vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
    faces = np.concatenate([mesh.faces + offsets[i] for i, mesh in enumerate(meshes)], axis=0)
    return vertices, faces


def _weld_tolerance(meshes: Sequence[MeshData], weld_tolerance: float | None) -> float:
    if weld_tolerance is not None:
        return float(weld_tolerance)
    points = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return max(DEFAULT_WELD_TOLERANCE * diagonal, 1.0e-9)


# --------------------------------------------------------------------------- assembly


def _resolve_material(materials: MaterialSpec | MaterialMap, key: str | None) -> MaterialSpec:
    if isinstance(materials, MaterialSpec):
        return materials
    if key in materials:
        return materials[key]
    if None in materials:
        return materials[None]
    raise ValueError(
        f"no material for mesh material {key!r}; pass it in materials= "
        f"(known keys: {sorted(str(name) for name in materials)})"
    )


def _structure(
    name: str, meshes: Sequence[MeshData], material: MaterialSpec, *, weld: bool, tolerance: float,
    material_key: str | None,
) -> tuple[Structure | None, StructureReport]:
    vertices, faces = _concat(meshes)
    input_vertices = int(vertices.shape[0])
    if weld:
        vertices, faces, welded, removed = weld_vertices(vertices, faces, tolerance=tolerance)
    else:
        keep = _non_degenerate(vertices, faces)
        welded, removed = 0, int(faces.shape[0] - int(keep.sum()))
        faces = faces[keep]
    report = StructureReport(
        name=name, material=material_key, source_meshes=tuple(mesh.name for mesh in meshes),
        input_vertices=input_vertices, output_vertices=int(vertices.shape[0]),
        welded_vertices=welded, removed_faces=removed, faces=int(faces.shape[0]),
    )
    if faces.shape[0] == 0:
        return None, report
    geometry = Mesh(
        torch.as_tensor(vertices, dtype=torch.float32),
        torch.as_tensor(faces, dtype=torch.int64),
        recenter=False, fill_mode="surface", topology_diagnostics=False,
    )
    return Structure(geometry, material, name=name), report


def _groups(meshes: Sequence[MeshData], merge: bool) -> list[tuple[str, list[MeshData]]]:
    if not merge:
        return [(mesh.name, [mesh]) for mesh in meshes]
    grouped: dict[str | None, list[MeshData]] = {}
    for mesh in meshes:
        grouped.setdefault(mesh.material, []).append(mesh)
    return [
        (group[0].name if len(group) == 1 else f"merged:{key}", group)
        for key, group in grouped.items()
    ]


def build_structures(
    meshes: Sequence[MeshData], materials: MaterialSpec | MaterialMap, *, merge: bool = True,
    weld: bool = True, weld_tolerance: float | None = None, warnings: Sequence[str] = (),
) -> ImportedScene:
    """Turn meshes into Core ``Structure`` values, one per mesh or per material.

    ``materials`` is one material for every mesh, or a mapping from the mesh
    material key (``None`` is the default entry). With ``merge=True`` meshes
    that share a material key become one structure. With ``weld=True``
    duplicate vertices within ``weld_tolerance`` metres (default: 1e-6 of the
    scene bounding-box diagonal) are merged so that adjacent triangles share
    edges, which diffraction edge extraction relies on.
    """

    meshes = [mesh for mesh in meshes if mesh.faces.shape[0] > 0]
    if not meshes:
        raise ValueError("no triangles to import")
    tolerance = _weld_tolerance(meshes, weld_tolerance)
    structures: list[Structure] = []
    reports: list[StructureReport] = []
    used: dict[str | None, MaterialSpec] = {}
    notes = list(warnings)
    for name, group in _groups(meshes, merge):
        key = group[0].material
        material = _resolve_material(materials, key)
        used[key] = material
        structure, report = _structure(
            name, group, material, weld=weld, tolerance=tolerance, material_key=key
        )
        reports.append(report)
        if structure is None:
            notes.append(f"{name}: every face was degenerate; structure skipped")
        else:
            structures.append(structure)
    return ImportedScene(tuple(structures), used, tuple(reports), tuple(notes))


# --------------------------------------------------------------------------- PLY

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


@dataclass
class _PlyElement:
    name: str
    count: int
    properties: list[tuple[str, str, str | None]] = field(default_factory=list)

    def has_lists(self) -> bool:
        return any(count_type is not None for _, _, count_type in self.properties)


def _ply_type(token: str, path: Path) -> str:
    if token not in _PLY_TYPES:
        raise ValueError(f"{path}: unsupported PLY property type {token!r}")
    return _PLY_TYPES[token]


def _ply_header(handle: Any, path: Path) -> tuple[str, list[_PlyElement]]:
    if handle.readline().strip() != b"ply":
        raise ValueError(f"{path}: not a PLY file")
    fmt = ""
    elements: list[_PlyElement] = []
    while True:
        line = handle.readline()
        if not line:
            raise ValueError(f"{path}: PLY header has no end_header")
        tokens = line.decode("ascii", errors="replace").split()
        if not tokens or tokens[0] in {"comment", "obj_info"}:
            continue
        if tokens[0] == "end_header":
            return fmt, elements
        if tokens[0] == "format":
            fmt = tokens[1]
        elif tokens[0] == "element":
            elements.append(_PlyElement(tokens[1], int(tokens[2])))
        elif tokens[0] == "property" and tokens[1] == "list":
            elements[-1].properties.append(
                (tokens[4], _ply_type(tokens[3], path), _ply_type(tokens[2], path))
            )
        elif tokens[0] == "property":
            elements[-1].properties.append((tokens[2], _ply_type(tokens[1], path), None))


def _ply_vertices(columns: Mapping[str, np.ndarray], path: Path) -> np.ndarray:
    if not {"x", "y", "z"} <= set(columns):
        raise ValueError(f"{path}: PLY vertex element lacks x/y/z")
    return np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)


def _fan(polygons: Iterable[Sequence[int]]) -> np.ndarray:
    triangles = [
        (polygon[0], polygon[i], polygon[i + 1])
        for polygon in polygons
        for i in range(1, len(polygon) - 1)
    ]
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def _face_index_property(element: _PlyElement, path: Path) -> str:
    for name, _, count_type in element.properties:
        if count_type is not None and name in {"vertex_indices", "vertex_index"}:
            return name
    raise ValueError(f"{path}: PLY face element lacks vertex_indices")


def _binary_fixed_lists(
    data: bytes, offset: int, element: _PlyElement, endian: str,
) -> tuple[dict[str, np.ndarray], int] | None:
    """Read a list element whose every list has the first row's length, else ``None``."""

    dtype_fields: list[tuple[str, str] | tuple[str, str, tuple[int]]] = []
    cursor = offset
    for name, value_type, count_type in element.properties:
        if count_type is None:
            dtype_fields.append((name, endian + value_type))
            cursor += np.dtype(value_type).itemsize
            continue
        length = int(np.frombuffer(data, endian + count_type, 1, cursor)[0])
        dtype_fields.append((f"{name}__count", endian + count_type))
        dtype_fields.append((name, endian + value_type, (length,)))
        cursor += np.dtype(count_type).itemsize + length * np.dtype(value_type).itemsize
    dtype = np.dtype(dtype_fields)  # type: ignore[arg-type]
    if offset + dtype.itemsize * element.count > len(data):
        return None
    rows = np.frombuffer(data, dtype, element.count, offset)
    for name, _, count_type in element.properties:
        if count_type is not None and not np.all(rows[f"{name}__count"] == rows[name].shape[1]):
            return None
    return {name: rows[name] for name, _, _ in element.properties}, offset + rows.nbytes


def _binary_ragged_lists(
    data: bytes, offset: int, element: _PlyElement, endian: str,
) -> tuple[dict[str, list[Any]], int]:
    columns: dict[str, list[Any]] = {name: [] for name, _, _ in element.properties}
    cursor = offset
    for _ in range(element.count):
        for name, value_type, count_type in element.properties:
            if count_type is None:
                columns[name].append(np.frombuffer(data, endian + value_type, 1, cursor)[0])
                cursor += np.dtype(value_type).itemsize
                continue
            length = int(np.frombuffer(data, endian + count_type, 1, cursor)[0])
            cursor += np.dtype(count_type).itemsize
            columns[name].append(np.frombuffer(data, endian + value_type, length, cursor))
            cursor += length * np.dtype(value_type).itemsize
    return columns, cursor


def _binary_element(
    data: bytes, offset: int, element: _PlyElement, endian: str,
) -> tuple[Mapping[str, Any], int]:
    if not element.has_lists():
        dtype = np.dtype([(name, endian + kind) for name, kind, _ in element.properties])
        rows = np.frombuffer(data, dtype, element.count, offset)
        return {name: rows[name] for name, _, _ in element.properties}, offset + rows.nbytes
    if element.count == 0:
        return {name: [] for name, _, _ in element.properties}, offset
    fixed = _binary_fixed_lists(data, offset, element, endian)
    if fixed is not None:
        return fixed
    return _binary_ragged_lists(data, offset, element, endian)


def _ascii_element(
    tokens: list[str], cursor: int, element: _PlyElement,
) -> tuple[dict[str, Any], int]:
    columns: dict[str, list[Any]] = {name: [] for name, _, _ in element.properties}
    for _ in range(element.count):
        for name, value_type, count_type in element.properties:
            if count_type is None:
                columns[name].append(float(tokens[cursor]))
                cursor += 1
                continue
            length = int(tokens[cursor])
            values = [float(token) for token in tokens[cursor + 1 : cursor + 1 + length]]
            columns[name].append(np.asarray(values, dtype=value_type))
            cursor += 1 + length
    resolved = {
        name: np.asarray(columns[name]) if count_type is None else columns[name]
        for name, _, count_type in element.properties
    }
    return resolved, cursor


def read_ply(path: StrPath, *, name: str | None = None, material: str | None = None) -> MeshData:
    """Read an ASCII or binary (little/big endian) PLY triangle or polygon mesh."""

    ply_path = Path(path).expanduser()
    with ply_path.open("rb") as handle:
        fmt, elements = _ply_header(handle, ply_path)
        body = handle.read()
    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"{ply_path}: unsupported PLY format {fmt!r}")
    columns: dict[str, Mapping[str, Any]] = {}
    if fmt == "ascii":
        tokens = body.decode("ascii", errors="replace").split()
        cursor = 0
        for element in elements:
            columns[element.name], cursor = _ascii_element(tokens, cursor, element)
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        offset = 0
        for element in elements:
            columns[element.name], offset = _binary_element(body, offset, element, endian)
    if "vertex" not in columns or "face" not in columns:
        raise ValueError(f"{ply_path}: PLY file needs vertex and face elements")
    face_element = next(element for element in elements if element.name == "face")
    polygons = columns["face"][_face_index_property(face_element, ply_path)]
    if isinstance(polygons, np.ndarray) and polygons.ndim == 2 and polygons.shape[1] == 3:
        faces = polygons.astype(np.int64)
    else:
        faces = _fan([np.asarray(polygon, dtype=np.int64).tolist() for polygon in polygons])
    vertices = _ply_vertices(columns["vertex"], ply_path)
    return MeshData(name or ply_path.stem, vertices, faces, material)


# --------------------------------------------------------------------------- OBJ


def _obj_index(token: str, vertex_count: int, where: str) -> int:
    raw = int(token.split("/")[0])
    index = raw - 1 if raw > 0 else vertex_count + raw
    if not 0 <= index < vertex_count:
        raise ValueError(f"{where}: OBJ face index {raw} out of range")
    return index


def read_obj(path: StrPath) -> tuple[MeshData, ...]:
    """Read a Wavefront OBJ file, returning one mesh per ``usemtl`` material.

    Polygons are fan-triangulated; texture and normal indices are ignored.
    """

    obj_path = Path(path).expanduser()
    vertices: list[tuple[float, float, float]] = []
    polygons: dict[str | None, list[list[int]]] = {}
    current: str | None = None
    with obj_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "v":
                vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            elif tokens[0] == "usemtl":
                current = " ".join(tokens[1:]) or None
            elif tokens[0] == "f":
                where = f"{obj_path}:{line_no}"
                polygon = [_obj_index(token, len(vertices), where) for token in tokens[1:]]
                polygons.setdefault(current, []).append(polygon)
    if not vertices or not polygons:
        raise ValueError(f"{obj_path}: OBJ file has no vertices or faces")
    points = np.asarray(vertices, dtype=np.float64)
    return tuple(
        _referenced(obj_path.stem if key is None else f"{obj_path.stem}:{key}", points,
                    _fan(group), key)
        for key, group in polygons.items()
    )


def _referenced(name: str, points: np.ndarray, faces: np.ndarray, key: str | None) -> MeshData:
    """Keep only the vertices ``faces`` references, so a split file shares no vertex rows."""

    used, compact = np.unique(faces.reshape(-1), return_inverse=True)
    return MeshData(name, points[used], compact.reshape(-1, 3), key)


# --------------------------------------------------------------------------- glTF

_GLTF_COMPONENTS = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
_GLTF_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_Y_UP_TO_Z_UP = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def _read_glb(data: bytes, path: Path) -> tuple[dict[str, Any], bytes | None]:
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path}: not a glTF 2.0 binary file")
    offset, document, binary = 12, None, None
    while offset < min(length, len(data)):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + chunk_length]
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942 and binary is None:
            binary = bytes(chunk)
        offset += 8 + chunk_length
    if document is None:
        raise ValueError(f"{path}: glTF binary has no JSON chunk")
    return document, binary


def _gltf_buffer(
    document: dict[str, Any], index: int, glb_binary: bytes | None, base: Path,
) -> bytes:
    buffer = document["buffers"][index]
    uri = buffer.get("uri")
    if uri is None:
        if glb_binary is None:
            raise ValueError("glTF buffer without uri needs a GLB binary chunk")
        return glb_binary
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    if "://" in uri or Path(uri).is_absolute():
        raise ValueError(f"glTF buffer uri {uri!r} must be a relative file path or data URI")
    return (base / str(uri)).read_bytes()


def _gltf_accessor(
    document: dict[str, Any], index: int, buffers: Callable[[int], bytes],
) -> np.ndarray:
    accessor = document["accessors"][index]
    if "sparse" in accessor or "bufferView" not in accessor:
        raise ValueError("sparse or buffer-less glTF accessors are not supported")
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype("<" + _GLTF_COMPONENTS[accessor["componentType"]])
    width = _GLTF_WIDTH[accessor["type"]]
    count = int(accessor["count"])
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", 0)) or dtype.itemsize * width
    data = buffers(int(view["buffer"]))
    if count == 0:
        return np.zeros((0, width), dtype=dtype)
    if start + stride * (count - 1) + dtype.itemsize * width > len(data):
        raise ValueError("glTF accessor reads past the end of its buffer")
    raw = np.lib.stride_tricks.as_strided(
        np.frombuffer(data, np.uint8, offset=start), shape=(count, dtype.itemsize * width),
        strides=(stride, 1), writeable=False,
    )
    return np.ascontiguousarray(raw).view(dtype).reshape(count, width)


def _quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _gltf_node_matrix(node: Mapping[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    matrix = np.eye(4)
    matrix[:3, :3] = _quat_matrix(*node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    matrix[:3, :3] = matrix[:3, :3] @ np.diag(node.get("scale", (1.0, 1.0, 1.0)))
    matrix[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return matrix


def _primitive_faces(mode: int, indices: np.ndarray) -> np.ndarray:
    if mode == 4:
        return indices[: indices.shape[0] - indices.shape[0] % 3].reshape(-1, 3)
    if mode == 5:
        faces = [
            (indices[i], indices[i + 1], indices[i + 2])
            if i % 2 == 0
            else (indices[i + 1], indices[i], indices[i + 2])
            for i in range(indices.shape[0] - 2)
        ]
        return np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return _fan([indices.tolist()])


def _gltf_material_name(document: dict[str, Any], primitive: Mapping[str, Any]) -> str | None:
    index = primitive.get("material")
    if index is None:
        return None
    material = document.get("materials", [])[int(index)]
    return str(material.get("name") or f"material_{index}")


@dataclass
class _GltfReader:
    document: dict[str, Any]
    buffers: Callable[[int], bytes]
    stem: str
    meshes: list[MeshData] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def primitive(self, label: str, primitive: Mapping[str, Any], matrix: np.ndarray) -> None:
        mode = int(primitive.get("mode", 4))
        if mode not in {4, 5, 6}:
            self.warnings.append(f"{label}: skipped primitive with non-triangle mode {mode}")
            return
        if "POSITION" not in primitive.get("attributes", {}):
            raise ValueError(f"{label}: glTF primitive has no POSITION attribute")
        positions = _gltf_accessor(self.document, primitive["attributes"]["POSITION"], self.buffers)
        if positions.dtype != np.float32:
            raise ValueError(f"{label}: quantized glTF positions are not supported")
        if "indices" in primitive:
            indices = _gltf_accessor(self.document, primitive["indices"], self.buffers)
            indices = indices.reshape(-1).astype(np.int64)
        else:
            indices = np.arange(positions.shape[0], dtype=np.int64)
        faces = _primitive_faces(mode, indices)
        material = _gltf_material_name(self.document, primitive)
        mesh = MeshData(label, positions.astype(np.float64), faces, material)
        self.meshes.append(mesh.transformed(matrix))

    def node(self, index: int, parent: np.ndarray, depth: int) -> None:
        if depth > 256:
            raise ValueError("glTF node hierarchy is cyclic or deeper than 256 levels")
        node = self.document["nodes"][index]
        matrix = parent @ _gltf_node_matrix(node)
        if "mesh" in node:
            mesh = self.document["meshes"][node["mesh"]]
            base = node.get("name") or mesh.get("name") or f"{self.stem}:node{index}"
            for number, primitive in enumerate(mesh.get("primitives", [])):
                self.primitive(f"{base}:{number}", primitive, matrix)
        for child in node.get("children", []):
            self.node(int(child), matrix, depth + 1)


def _gltf_roots(document: dict[str, Any]) -> list[int]:
    scenes = document.get("scenes")
    if scenes:
        return [int(node) for node in scenes[int(document.get("scene", 0))].get("nodes", [])]
    children = {child for node in document.get("nodes", []) for child in node.get("children", [])}
    return [i for i in range(len(document.get("nodes", []))) if i not in children]


def read_gltf(path: StrPath, *, up_axis: str = "z") -> tuple[tuple[MeshData, ...], tuple[str, ...]]:
    """Read triangle meshes from a glTF 2.0 ``.glb`` or ``.gltf`` file with node transforms.

    glTF is +Y-up; ``up_axis="z"`` (default) rotates the scene so +Y becomes
    +Z, matching Channel scenes, and ``up_axis="y"`` keeps file coordinates.
    Each primitive's material key is its glTF material name. Returns
    ``(meshes, warnings)``.
    """

    gltf_path = Path(path).expanduser()
    if up_axis not in {"y", "z"}:
        raise ValueError("up_axis must be 'y' or 'z'")
    data = gltf_path.read_bytes()
    if data[:4] == b"glTF":
        document, binary = _read_glb(data, gltf_path)
    else:
        document, binary = json.loads(data.decode("utf-8")), None
    required = document.get("extensionsRequired", [])
    if required:
        raise ValueError(f"{gltf_path}: required glTF extensions are not supported: {required}")
    cache: dict[int, bytes] = {}

    def buffers(index: int) -> bytes:
        if index not in cache:
            cache[index] = _gltf_buffer(document, index, binary, gltf_path.parent)
        return cache[index]

    reader = _GltfReader(document, buffers, gltf_path.stem)
    root = _Y_UP_TO_Z_UP if up_axis == "z" else np.eye(4)
    for index in _gltf_roots(document):
        reader.node(index, root, 0)
    if not reader.meshes:
        raise ValueError(f"{gltf_path}: glTF file has no triangle meshes")
    return tuple(reader.meshes), tuple(reader.warnings)


# --------------------------------------------------------------------------- single files


def load_mesh(
    path: StrPath, materials: MaterialSpec | MaterialMap, *, merge: bool = True, weld: bool = True,
    weld_tolerance: float | None = None, up_axis: str = "z",
) -> ImportedScene:
    """Import one ``.ply``, ``.obj``, ``.glb`` or ``.gltf`` file as Core structures.

    ``materials`` is one material for the whole file or a mapping from the
    file's material names (OBJ ``usemtl``, glTF material name; ``None`` is the
    default entry). ``up_axis`` applies to glTF only.
    """

    mesh_path = Path(path).expanduser()
    suffix = mesh_path.suffix.lower()
    warnings: tuple[str, ...] = ()
    if suffix == ".ply":
        meshes: tuple[MeshData, ...] = (read_ply(mesh_path),)
    elif suffix == ".obj":
        meshes = read_obj(mesh_path)
    elif suffix in {".glb", ".gltf"}:
        meshes, warnings = read_gltf(mesh_path, up_axis=up_axis)
    else:
        raise ValueError(f"{mesh_path}: unsupported mesh format {suffix!r}")
    return build_structures(
        meshes, materials, merge=merge, weld=weld, weld_tolerance=weld_tolerance, warnings=warnings
    )


# --------------------------------------------------------------------------- Mitsuba serialized


def _read_serialized(path: Path, shape_index: int) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, len(data) - 4)[0]
    if not 0 <= shape_index < count:
        raise ValueError(f"{path}: shape_index {shape_index} outside 0..{count - 1}")
    if struct.unpack_from("<H", data, 2)[0] >= 4:
        offset = struct.unpack_from("<Q", data, len(data) - 4 - 8 * (count - shape_index))[0]
    else:
        offset = struct.unpack_from("<I", data, len(data) - 4 - 4 * (count - shape_index))[0]
    header, version = struct.unpack_from("<HH", data, offset)
    if header != 0x041C:
        raise ValueError(f"{path}: not a Mitsuba serialized mesh")
    stream = zlib.decompressobj().decompress(data[offset + 4 :])
    flags = struct.unpack_from("<I", stream, 0)[0]
    cursor = stream.index(b"\0", 4) + 1 if version >= 4 else 4
    vertex_count, face_count = struct.unpack_from("<QQ", stream, cursor)
    cursor += 16
    real = "<f8" if flags & 0x2000 else "<f4"
    size = np.dtype(real).itemsize
    vertices = np.frombuffer(stream, real, vertex_count * 3, cursor).reshape(-1, 3)
    cursor += vertex_count * 3 * size
    for flag, width in ((0x0001, 3), (0x0002, 2), (0x0008, 3)):
        if flags & flag:
            cursor += vertex_count * width * size
    index = "<u8" if vertex_count > 0xFFFFFFFF else "<u4"
    faces = np.frombuffer(stream, index, face_count * 3, cursor).reshape(-1, 3)
    return vertices.astype(np.float64), faces.astype(np.int64)


# --------------------------------------------------------------------------- Mitsuba XML

_RECTANGLE = (
    np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]),
    np.array([[0, 1, 2], [0, 2, 3]]),
)
_CUBE_VERTICES = np.array(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
)
_CUBE_FACES = np.array(
    [
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ]
)


def _floats(text: str) -> list[float]:
    return [float(token) for token in text.replace(",", " ").split()]


def _vector(element: ElementTree.Element, default: float) -> np.ndarray:
    if "value" in element.attrib:
        values = _floats(element.attrib["value"])
        return np.asarray(values * 3 if len(values) == 1 else values, dtype=np.float64)
    return np.asarray([float(element.attrib.get(axis, default)) for axis in "xyz"])


def _rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return c * np.eye(3) + s * cross + (1.0 - c) * np.outer(axis, axis)


def _lookat(element: ElementTree.Element) -> np.ndarray:
    origin = np.asarray(_floats(element.attrib["origin"]))
    target = np.asarray(_floats(element.attrib["target"]))
    up = np.asarray(_floats(element.attrib.get("up", "0 1 0")))
    forward = (target - origin) / np.linalg.norm(target - origin)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = left, np.cross(forward, left), forward
    matrix[:3, 3] = origin
    return matrix


def _transform_step(element: ElementTree.Element) -> np.ndarray:
    step = np.eye(4)
    tag = element.tag
    if tag == "translate":
        step[:3, 3] = _vector(element, 0.0)
    elif tag == "scale":
        step[:3, :3] = np.diag(_vector(element, 1.0))
    elif tag == "rotate":
        step[:3, :3] = _rotation(_vector(element, 0.0), float(element.attrib["angle"]))
    elif tag == "matrix":
        values = _floats(element.attrib["value"])
        if len(values) == 9:
            step[:3, :3] = np.asarray(values).reshape(3, 3)
        elif len(values) == 16:
            step = np.asarray(values).reshape(4, 4)
        else:
            raise ValueError("Mitsuba <matrix> needs 9 or 16 values")
    elif tag == "lookat":
        step = _lookat(element)
    else:
        raise ValueError(f"unsupported Mitsuba transform operation <{tag}>")
    return step


def _transform(element: ElementTree.Element | None) -> np.ndarray:
    matrix = np.eye(4)
    if element is None:
        return matrix
    for step in element:
        matrix = _transform_step(step) @ matrix
    return matrix


def _properties(element: ElementTree.Element) -> dict[str, str]:
    return {
        child.attrib["name"]: child.attrib.get("value", "")
        for child in element
        if child.tag in {"string", "float", "integer", "boolean", "rgb"} and "name" in child.attrib
    }


def _substitute(element: ElementTree.Element, defaults: Mapping[str, str]) -> None:
    for node in element.iter():
        for key, value in node.attrib.items():
            if "$" in value:
                for name, replacement in defaults.items():
                    value = value.replace(f"${name}", replacement)
                node.attrib[key] = value


@dataclass
class _MitsubaContext:
    base: Path
    frequency_hz: float | tuple[float, float]
    overrides: Mapping[str, MaterialSpec]
    dispersive: bool
    bsdfs: dict[str, ElementTree.Element] = field(default_factory=dict)
    materials: dict[str | None, MaterialSpec] = field(default_factory=dict)

    def material(self, key: str | None) -> None:
        if key in self.materials:
            return
        if key is not None and key in self.overrides:
            self.materials[key] = self.overrides[key]
            return
        if key is None:
            raise ValueError("a Mitsuba shape has no bsdf; pass default_material=")
        if key not in self.bsdfs:
            raise ValueError(f"Mitsuba shape references unknown bsdf {key!r}")
        self.materials[key] = _radio_material(key, self.bsdfs[key], self)


def _material_kwargs(props: Mapping[str, str]) -> dict[str, float]:
    kwargs = {"thickness_m": float(props.get("thickness", 0.1))}
    for source, target in (
        ("scattering_coefficient", "scattering_coefficient"),
        ("xpd_coefficient", "xpd_coefficient"),
    ):
        if source in props:
            kwargs[target] = float(props[source])
    return kwargs


def _legacy_itu_name(key: str) -> str | None:
    lowered = key.lower()
    for prefix in ("mat-itu_", "itu_"):
        if lowered.startswith(prefix):
            return lowered[len(prefix) :]
    return None


def _radio_material(key: str, element: ElementTree.Element, ctx: _MitsubaContext) -> MaterialSpec:
    while element.attrib.get("type") == "twosided" and element.find("bsdf") is not None:
        element = element.find("bsdf")  # type: ignore[assignment]
    kind = element.attrib.get("type", "")
    props = _properties(element)
    if kind == "itu-radio-material":
        return itu_material(
            props["type"], ctx.frequency_hz, dispersive=ctx.dispersive, material_name=key,
            **_material_kwargs(props),
        )
    if kind == "radio-material":
        return PhysicalMaterial(
            eps_r=float(props.get("relative_permittivity", 1.0)),
            sigma_e=float(props.get("conductivity", 0.0)), name=key, **_material_kwargs(props),
        )
    legacy = _legacy_itu_name(key)
    if legacy is not None:
        return itu_material(
            legacy, ctx.frequency_hz, dispersive=ctx.dispersive, material_name=key,
            **_material_kwargs(props),
        )
    raise ValueError(
        f"Mitsuba bsdf {key!r} of type {kind!r} is not a radio material; map it with materials="
    )


def _shape_bsdf(shape: ElementTree.Element, ctx: _MitsubaContext, label: str) -> str | None:
    for child in shape:
        if child.tag == "ref" and child.attrib.get("name", "bsdf") == "bsdf":
            return child.attrib["id"]
        if child.tag == "bsdf":
            key = child.attrib.get("id", f"{label}:bsdf")
            ctx.bsdfs[key] = child
            return key
    return None


def _shape_geometry(
    shape: ElementTree.Element, ctx: _MitsubaContext,
) -> tuple[np.ndarray, np.ndarray]:
    kind = shape.attrib.get("type", "")
    props = _properties(shape)
    if kind == "rectangle":
        return _RECTANGLE
    if kind == "cube":
        return _CUBE_VERTICES, _CUBE_FACES
    if kind in {"ply", "obj", "serialized"} and "filename" not in props:
        raise ValueError(f"Mitsuba {kind} shape has no filename")
    if kind == "ply":
        mesh = read_ply(ctx.base / props["filename"])
        return mesh.vertices, mesh.faces
    if kind == "obj":
        meshes = read_obj(ctx.base / props["filename"])
        return _concat(meshes)
    if kind == "serialized":
        return _read_serialized(ctx.base / props["filename"], int(props.get("shape_index", 0)))
    raise ValueError(f"unsupported Mitsuba shape type {kind!r}")


def _shape_mesh(shape: ElementTree.Element, ctx: _MitsubaContext, index: int) -> MeshData:
    label = shape.attrib.get("id", f"shape{index}")
    vertices, faces = _shape_geometry(shape, ctx)
    if _properties(shape).get("flip_normals", "false").lower() == "true":
        faces = faces[:, ::-1]
    material = _shape_bsdf(shape, ctx, label)
    mesh = MeshData(label, vertices, faces, material)
    return mesh.transformed(_transform(shape.find("transform[@name='to_world']")))


def _mitsuba_root(path: Path) -> ElementTree.Element:
    root = ElementTree.parse(path).getroot()
    if root.tag != "scene":
        raise ValueError(f"{path}: Mitsuba root element must be <scene>")
    if root.find("include") is not None:
        raise ValueError(f"{path}: Mitsuba <include> is not supported; inline the file")
    defaults = {
        child.attrib["name"]: child.attrib["value"] for child in root if child.tag == "default"
    }
    _substitute(root, defaults)
    return root


def load_mitsuba(
    path: StrPath, *, frequency_hz: float | tuple[float, float],
    materials: Mapping[str, MaterialSpec] | None = None,
    default_material: MaterialSpec | None = None, merge_shapes: bool = True, weld: bool = True,
    weld_tolerance: float | None = None, dispersive: bool = True,
) -> ImportedScene:
    """Import a Mitsuba 3 / Sionna RT XML scene as Core structures.

    Supported shapes are ``ply``, ``obj``, ``serialized``, ``rectangle`` and
    ``cube`` with ``to_world`` transforms (``translate``, ``rotate``,
    ``scale``, ``matrix``, ``lookat``); other shape types raise. A bsdf of type
    ``itu-radio-material`` becomes :func:`witwin.channel.presets.itu_material`
    at ``frequency_hz``, ``radio-material`` becomes a ``PhysicalMaterial`` from
    ``relative_permittivity``/``conductivity``/``thickness``, and a bsdf id
    ``itu_<name>`` or ``mat-itu_<name>`` maps to that ITU material.
    ``materials`` overrides any bsdf id; ``default_material`` covers shapes
    without a bsdf. ``merge_shapes`` merges shapes that share a bsdf into one
    structure, and ``weld`` merges duplicate vertices (see
    :func:`build_structures`).
    """

    xml_path = Path(path).expanduser()
    root = _mitsuba_root(xml_path)
    ctx = _MitsubaContext(xml_path.parent, frequency_hz, dict(materials or {}), dispersive)
    for child in root:
        if child.tag == "bsdf" and "id" in child.attrib:
            ctx.bsdfs[child.attrib["id"]] = child
    unsupported = sorted({child.tag for child in root if child.tag in {"shapegroup", "instance"}})
    if unsupported:
        raise ValueError(f"{xml_path}: Mitsuba {unsupported} elements are not supported")
    shapes = [child for child in root if child.tag == "shape"]
    meshes = [_shape_mesh(shape, ctx, index) for index, shape in enumerate(shapes)]
    if not meshes:
        raise ValueError(f"{xml_path}: Mitsuba scene has no shapes")
    if default_material is not None:
        ctx.materials[None] = default_material
    for mesh in meshes:
        ctx.material(mesh.material)
    return build_structures(
        meshes, ctx.materials, merge=merge_shapes, weld=weld, weld_tolerance=weld_tolerance
    )
