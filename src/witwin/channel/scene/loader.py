from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from witwin.channel.core.edge_policy import EdgePolicy
from witwin.channel.materials.models import Dielectric, ITUMaterial
from witwin.channel.scene.models import Structure


_REPO_ROOT = Path(__file__).resolve().parents[5]
_PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}

_ITU_MATERIALS_PROPERTIES = {
    "concrete": (((1.0, 100.0), (5.24, 0.0, 0.0462, 0.7822)),),
    "brick": (((1.0, 40.0), (3.91, 0.0, 0.0238, 0.16)),),
    "plasterboard": (((1.0, 100.0), (2.73, 0.0, 0.0085, 0.9395)),),
    "wood": (((0.001, 100.0), (1.99, 0.0, 0.0047, 1.0718)),),
    "glass": (
        ((0.1, 100.0), (6.31, 0.0, 0.0036, 1.3394)),
        ((220.0, 450.0), (5.79, 0.0, 0.0004, 1.658)),
    ),
    "ceiling_board": (
        ((1.0, 100.0), (1.48, 0.0, 0.0011, 1.0750)),
        ((220.0, 450.0), (1.52, 0.0, 0.0029, 1.029)),
    ),
    "chipboard": (((1.0, 100.0), (2.58, 0.0, 0.0217, 0.7800)),),
    "plywood": (((1.0, 40.0), (2.71, 0.0, 0.33, 0.0)),),
    "marble": (((1.0, 60.0), (7.074, 0.0, 0.0055, 0.9262)),),
    "floorboard": (((50.0, 100.0), (3.66, 0.0, 0.0044, 1.3515)),),
    "metal": (((1.0, 100.0), (1.0, 0.0, 1.0e7, 0.0)),),
    "very_dry_ground": (((1.0, 10.0), (3.0, 0.0, 0.00015, 2.52)),),
    "medium_dry_ground": (((1.0, 10.0), (15.0, -0.1, 0.035, 1.63)),),
    "wet_ground": (((1.0, 10.0), (30.0, -0.4, 0.15, 1.30)),),
}

_MUNICH_MERGED_GROUP_ORDER = ("wood", "marble", "metal", "brick")


@dataclass(frozen=True, slots=True)
class _NativeMesh:
    name: str
    material_name: str | None
    material_ref: str | None
    thickness_m: float
    scattering_coefficient: float
    xpd_coefficient: float
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None = None
    face_uv: np.ndarray | None = None
    source_count: int = 1


def _sanitize(name: str | None, default: str) -> str:
    text = (
        re.sub(r"[^0-9A-Za-z_-]+", "-", "" if name is None else str(name)).strip("-_")
        or default
    )
    return f"{default}-{text}" if text[0].isdigit() else text


def _unique(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    index = 2
    while f"{base}-{index}" in used:
        index += 1
    value = f"{base}-{index}"
    used.add(value)
    return value


def _xml_child_value(
    node: ET.Element, tag: str, name: str, default: str | None = None
) -> str | None:
    child = node.find(f"{tag}[@name='{name}']")
    if child is None:
        return default
    return child.attrib.get("value", default)


def _xml_child_bool(node: ET.Element, name: str, default: bool = False) -> bool:
    value = _xml_child_value(node, "boolean", name)
    if value is None:
        return default
    return value.lower() in {"true", "1", "yes"}


def itu_material_parameters(
    material_name: str, frequency_hz: float
) -> tuple[float, float]:
    """Evaluate an ITU-R P.2040 power-law material in SI units."""

    if material_name not in _ITU_MATERIALS_PROPERTIES:
        raise ValueError(f"ITU radio material is not recognized: {material_name!r}")
    f_ghz = float(frequency_hz) / 1.0e9
    for (f_min, f_max), (a, b, c, d) in _ITU_MATERIALS_PROPERTIES[material_name]:
        if f_min <= f_ghz <= f_max:
            return a * (f_ghz**b), c * (f_ghz**d)
    raise ValueError(
        f"ITU material {material_name!r} is not defined for {frequency_hz} Hz"
    )


def _native_material(
    material_name: str | None,
    frequency_hz: float,
    *,
    thickness_m: float = 0.1,
    scattering_coefficient: float = 0.0,
    xpd_coefficient: float = 0.0,
) -> Dielectric | ITUMaterial:
    if material_name is None:
        return Dielectric(eps_r=1.0, thickness_m=thickness_m)
    if material_name not in _ITU_MATERIALS_PROPERTIES:
        raise ValueError(f"ITU radio material is not recognized: {material_name!r}")
    # Validate the requested frequency immediately, but retain the dispersive
    # model so Scene.with_frequency() re-evaluates instead of freezing values.
    itu_material_parameters(material_name, frequency_hz)
    return ITUMaterial(
        name=material_name,
        thickness_m=thickness_m,
        scattering_coefficient=scattering_coefficient,
        xpd_coefficient=xpd_coefficient,
    )


def _read_ply_header(handle) -> tuple[list[str], int, list[tuple[str, str]], int]:
    lines: list[str] = []
    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError("unexpected end of PLY header")
        line = raw.decode("ascii").strip()
        lines.append(line)
        if line == "end_header":
            break

    if (
        len(lines) < 3
        or lines[0] != "ply"
        or lines[1] != "format binary_little_endian 1.0"
    ):
        raise ValueError("only binary_little_endian PLY meshes are supported")

    vertex_count = 0
    face_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            elif current_element == "face":
                face_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] not in _PLY_SCALAR_DTYPES:
                raise ValueError(f"vertex PLY property is not accepted: {line}")
            vertex_properties.append((parts[2], parts[1]))
        elif parts[0] == "property" and current_element == "face":
            if parts != ["property", "list", "uchar", "int", "vertex_indices"]:
                raise ValueError(f"face PLY property is not accepted: {line}")

    if vertex_count <= 0 or face_count < 0:
        raise ValueError("invalid PLY element counts")
    if not {"x", "y", "z"}.issubset({name for name, _ in vertex_properties}):
        raise ValueError("PLY vertices must include x, y, and z properties")
    return lines, vertex_count, vertex_properties, face_count


def _load_ply_mesh(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with path.open("rb") as handle:
        _lines, vertex_count, vertex_properties, face_count = _read_ply_header(handle)
        vertex_dtype = np.dtype(
            [
                (name, _PLY_SCALAR_DTYPES[prop_type])
                for name, prop_type in vertex_properties
            ]
        )
        vertex_data = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
        if int(vertex_data.shape[0]) != vertex_count:
            raise ValueError(f"truncated PLY vertex data: {path}")
        vertices = np.column_stack(
            (vertex_data["x"], vertex_data["y"], vertex_data["z"])
        ).astype(np.float32, copy=False)
        property_names = {name for name, _ in vertex_properties}
        uv_pairs = (("u", "v"), ("s", "t"), ("texture_u", "texture_v"))
        partial = [pair for pair in uv_pairs if bool(set(pair) & property_names) and not set(pair) <= property_names]
        if partial:
            raise ValueError(f"PLY UV aliases must be provided in pairs: {partial[0]}")
        uv_values = [
            np.column_stack((vertex_data[u_name], vertex_data[v_name])).astype(
                np.float32, copy=False
            )
            for u_name, v_name in uv_pairs
            if {u_name, v_name} <= property_names
        ]
        if any(not np.allclose(uv_values[0], value) for value in uv_values[1:]):
            raise ValueError(f"PLY defines conflicting per-vertex UV aliases: {path}")
        uv = np.ascontiguousarray(uv_values[0]) if uv_values else None

        face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
        face_data = np.fromfile(handle, dtype=face_dtype, count=face_count)
        if int(face_data.shape[0]) != face_count:
            raise ValueError(f"truncated PLY face data: {path}")
        if not np.all(face_data["count"] == 3):
            raise ValueError(f"only triangular PLY faces are supported: {path}")
        faces = np.array(face_data["indices"], dtype=np.int32, copy=True, order="C")

    faces = faces.copy(order="C")
    face_uv = faces.copy() if uv is not None else None
    return np.ascontiguousarray(vertices), faces, uv, face_uv


def _xml_numbers(value: str | None, *, context: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{context} is missing a value")
    try:
        numbers = np.asarray(
            [float(item) for item in re.split(r"[\s,]+", value.strip()) if item],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise ValueError(f"{context} contains a non-numeric value") from exc
    if not np.isfinite(numbers).all():
        raise ValueError(f"{context} must be finite")
    return numbers


def _xml_vector3(
    node: ET.Element,
    *,
    defaults: tuple[float, float, float],
    context: str,
    allow_scalar: bool = False,
) -> np.ndarray:
    if "value" in node.attrib:
        values = _xml_numbers(node.attrib["value"], context=context)
        if values.size == 1 and allow_scalar:
            values = np.repeat(values, 3)
        if values.size != 3:
            raise ValueError(f"{context} must contain one or three values")
        return values
    values = np.asarray(
        [float(node.attrib.get(axis, default)) for axis, default in zip("xyz", defaults)],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{context} must be finite")
    return values


def _transform_operation(node: ET.Element) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    if node.tag == "translate":
        matrix[:3, 3] = _xml_vector3(
            node, defaults=(0.0, 0.0, 0.0), context="translate"
        )
    elif node.tag == "scale":
        scale = _xml_vector3(
            node,
            defaults=(1.0, 1.0, 1.0),
            context="scale",
            allow_scalar=True,
        )
        matrix[:3, :3] = np.diag(scale)
    elif node.tag == "rotate":
        axis = _xml_vector3(node, defaults=(0.0, 0.0, 0.0), context="rotate axis")
        norm = float(np.linalg.norm(axis))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("rotate axis must be non-zero")
        if "angle" not in node.attrib:
            raise ValueError("rotate is missing angle")
        angle = float(node.attrib["angle"])
        if not math.isfinite(angle):
            raise ValueError("rotate angle must be finite")
        x, y, z = axis / norm
        radians = math.radians(angle)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        one_minus = 1.0 - cosine
        matrix[:3, :3] = np.asarray(
            [
                [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
                [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
                [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
            ],
            dtype=np.float64,
        )
    elif node.tag == "matrix":
        values = _xml_numbers(node.attrib.get("value"), context="matrix")
        if values.size == 16:
            matrix = values.reshape(4, 4)
        elif values.size == 9:
            matrix[:3, :3] = values.reshape(3, 3)
        else:
            raise ValueError("matrix must contain 9 or 16 row-major values")
    else:
        raise ValueError(f"to_world operation is not supported: {node.tag!r}")
    return matrix


def _shape_transform(shape: ET.Element) -> np.ndarray:
    transforms = [child for child in shape if child.tag == "transform"]
    if any(child.attrib.get("name") != "to_world" for child in transforms):
        raise ValueError("shape transforms must be named 'to_world'")
    if len(transforms) > 1:
        raise ValueError("shape has more than one to_world transform")
    if not transforms:
        return np.eye(4, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    # Mitsuba XML composes incremental transforms by left multiplication.
    for operation in transforms[0]:
        matrix = _transform_operation(operation) @ matrix
    if not np.isfinite(matrix).all():
        raise ValueError("to_world transform must be finite")
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
        raise ValueError("to_world matrix must be affine")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not math.isfinite(determinant) or np.linalg.matrix_rank(matrix[:3, :3]) < 3:
        raise ValueError("to_world transform is singular")
    return matrix


def _apply_mesh_transform(mesh: _NativeMesh, matrix: np.ndarray) -> _NativeMesh:
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix[:3, :3]) < 3:
        raise ValueError("combined to_world transform is singular or non-finite")
    homogeneous = np.concatenate(
        (mesh.vertices.astype(np.float64), np.ones((mesh.vertices.shape[0], 1))), axis=1
    )
    transformed = (matrix @ homogeneous.T).T
    if not np.isfinite(transformed).all() or not np.allclose(transformed[:, 3], 1.0):
        raise ValueError("to_world produced invalid vertices")
    faces = mesh.faces
    face_uv = mesh.face_uv
    # A reflection changes handedness. Reverse winding so derived geometric
    # normals preserve the source mesh's outward-orientation convention.
    if float(np.linalg.det(matrix[:3, :3])) < 0.0:
        faces = faces[:, [0, 2, 1]]
        if face_uv is not None:
            face_uv = face_uv[:, [0, 2, 1]]
    return replace(
        mesh,
        vertices=np.ascontiguousarray(transformed[:, :3].astype(np.float32)),
        faces=np.ascontiguousarray(faces),
        face_uv=np.ascontiguousarray(face_uv) if face_uv is not None else None,
    )


def _merge_meshes(meshes: list[_NativeMesh], name: str) -> _NativeMesh:
    if not meshes:
        raise ValueError("cannot merge an empty mesh group")
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    uv: list[np.ndarray] = []
    face_uv: list[np.ndarray] = []
    vertex_offset = 0
    uv_offset = 0
    has_uv = [mesh.uv is not None for mesh in meshes]
    if any(has_uv) and not all(has_uv):
        raise ValueError("cannot merge meshes with mixed UV availability")
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + vertex_offset)
        vertex_offset += int(mesh.vertices.shape[0])
        if mesh.uv is not None:
            if mesh.face_uv is None:
                raise ValueError("mesh UV requires face_uv indices")
            uv.append(mesh.uv)
            face_uv.append(mesh.face_uv + uv_offset)
            uv_offset += int(mesh.uv.shape[0])
    return _NativeMesh(
        name=name,
        material_name=meshes[0].material_name,
        material_ref=meshes[0].material_ref,
        thickness_m=meshes[0].thickness_m,
        scattering_coefficient=meshes[0].scattering_coefficient,
        xpd_coefficient=meshes[0].xpd_coefficient,
        vertices=np.ascontiguousarray(np.concatenate(vertices, axis=0)),
        faces=np.ascontiguousarray(np.concatenate(faces, axis=0)),
        uv=np.ascontiguousarray(np.concatenate(uv, axis=0)) if uv else None,
        face_uv=np.ascontiguousarray(np.concatenate(face_uv, axis=0)) if uv else None,
        source_count=sum(mesh.source_count for mesh in meshes),
    )


def _material_defs(root: ET.Element) -> dict[str, dict[str, object]]:
    materials: dict[str, dict[str, object]] = {}
    for bsdf in root.findall("./bsdf") + root.findall(".//shape/bsdf"):
        mat_id = bsdf.attrib.get("id")
        if mat_id is None:
            continue
        material_type = _xml_child_value(bsdf, "string", "type", mat_id)
        thickness = float(_xml_child_value(bsdf, "float", "thickness", "0.1") or 0.1)
        materials[mat_id] = {
            "type": material_type,
            "thickness": thickness,
            "scattering_coefficient": float(
                _xml_child_value(bsdf, "float", "scattering_coefficient", "0.0") or 0.0
            ),
            "xpd_coefficient": float(
                _xml_child_value(bsdf, "float", "xpd_coefficient", "0.0") or 0.0
            ),
        }
    return materials


def _shape_ref_material(shape: ET.Element) -> str | None:
    ref = shape.find("ref[@name='bsdf']")
    if ref is None:
        ref = shape.find("ref")
    if ref is None:
        return None
    return ref.attrib.get("id")


def _ordered_merged_materials(
    groups: dict[str | None, list[_NativeMesh]],
) -> list[str | None]:
    ordered: list[str | None] = []
    for material_name in _MUNICH_MERGED_GROUP_ORDER:
        ordered.extend(
            key
            for key, meshes in groups.items()
            if meshes[0].material_name == material_name
        )
    for material_ref, meshes in groups.items():
        if material_ref not in ordered and meshes[0].material_name != "concrete":
            ordered.append(material_ref)
    ordered.extend(
        key
        for key, meshes in groups.items()
        if key not in ordered and meshes[0].material_name == "concrete"
    )
    return ordered


def _ply_shape_mesh(
    shape: ET.Element,
    *,
    scene_path: Path,
    material_defs: dict[str, dict[str, object]],
    transform: np.ndarray,
    name_prefix: str | None,
) -> _NativeMesh:
    filename = _xml_child_value(shape, "string", "filename")
    if filename is None:
        raise ValueError("PLY shape is missing a filename")
    mesh_path = (scene_path.parent / filename).resolve()
    shape_id = shape.attrib.get("id", mesh_path.stem)
    object_key = shape_id[5:] if shape_id.startswith("mesh-") else shape_id
    if name_prefix:
        object_key = f"{name_prefix}-{object_key}"
    material_id = _shape_ref_material(shape)
    material_name = str(
        material_defs.get(material_id, {}).get("type", material_id or "")
    )
    if material_id is None:
        material_name = None
    material_info = material_defs.get(material_id or "", {})
    vertices, faces, uv, face_uv = _load_ply_mesh(mesh_path)
    mesh = _NativeMesh(
        name=object_key,
        material_name=material_name,
        material_ref=material_id,
        thickness_m=float(material_info.get("thickness", 0.1) or 0.1),
        scattering_coefficient=float(
            material_info.get("scattering_coefficient", 0.0) or 0.0
        ),
        xpd_coefficient=float(material_info.get("xpd_coefficient", 0.0) or 0.0),
        vertices=vertices,
        faces=faces,
        uv=uv,
        face_uv=face_uv,
    )
    return _apply_mesh_transform(mesh, transform @ _shape_transform(shape))


def _expanded_shapes(
    root: ET.Element,
    *,
    scene_path: Path,
    material_defs: dict[str, dict[str, object]],
) -> list[tuple[ET.Element, _NativeMesh]]:
    groups: dict[str, ET.Element] = {}
    for shape in root.findall("shape"):
        if shape.attrib.get("type") != "shapegroup":
            continue
        group_id = shape.attrib.get("id")
        if not group_id:
            raise ValueError("top-level shapegroup is missing id")
        if group_id in groups:
            raise ValueError(f"duplicate shapegroup id: {group_id!r}")
        groups[group_id] = shape

    def validate_group(group_id: str, stack: tuple[str, ...]) -> None:
        if group_id in stack:
            cycle = " -> ".join((*stack, group_id))
            raise ValueError(f"shapegroup instance cycle: {cycle}")
        group = groups[group_id]
        _shape_transform(group)
        for child in group:
            if child.tag == "transform":
                continue
            if child.tag != "shape":
                raise ValueError(f"shapegroup child is not supported: {child.tag!r}")
            shape_type = child.attrib.get("type")
            if shape_type == "ply":
                _shape_transform(child)
                continue
            if shape_type != "instance":
                raise ValueError(
                    f"native Mitsuba loader does not support shape type {shape_type!r}"
                )
            refs = [node for node in child if node.tag == "ref"]
            if len(refs) != 1 or not refs[0].attrib.get("id"):
                raise ValueError("instance must contain exactly one shapegroup ref")
            target = refs[0].attrib["id"]
            if target not in groups:
                raise ValueError(f"instance references missing shapegroup: {target!r}")
            _shape_transform(child)
            validate_group(target, (*stack, group_id))

    for group_id in groups:
        validate_group(group_id, ())

    def expand(
        shape: ET.Element,
        *,
        parent_transform: np.ndarray,
        stack: tuple[str, ...],
        name_prefix: str | None,
    ) -> list[tuple[ET.Element, _NativeMesh]]:
        shape_type = shape.attrib.get("type")
        if shape_type == "ply":
            return [
                (
                    shape,
                    _ply_shape_mesh(
                        shape,
                        scene_path=scene_path,
                        material_defs=material_defs,
                        transform=parent_transform,
                        name_prefix=name_prefix,
                    ),
                )
            ]
        if shape_type != "instance":
            raise ValueError(f"native Mitsuba loader does not support shape type {shape_type!r}")
        refs = [child for child in shape if child.tag == "ref"]
        if len(refs) != 1 or not refs[0].attrib.get("id"):
            raise ValueError("instance must contain exactly one shapegroup ref")
        group_id = refs[0].attrib["id"]
        if group_id not in groups:
            raise ValueError(f"instance references missing shapegroup: {group_id!r}")
        if group_id in stack:
            cycle = " -> ".join((*stack, group_id))
            raise ValueError(f"shapegroup instance cycle: {cycle}")
        instance_id = shape.attrib.get("id", group_id)
        prefix = f"{name_prefix}-{instance_id}" if name_prefix else instance_id
        group = groups[group_id]
        transform = (
            parent_transform @ _shape_transform(shape) @ _shape_transform(group)
        )
        meshes: list[tuple[ET.Element, _NativeMesh]] = []
        for child in group:
            if child.tag == "shape":
                meshes.extend(
                    expand(
                        child,
                        parent_transform=transform,
                        stack=(*stack, group_id),
                        name_prefix=prefix,
                    )
                )
            elif child.tag != "transform":
                raise ValueError(
                    f"shapegroup child is not supported: {child.tag!r}"
                )
        return meshes

    expanded: list[tuple[ET.Element, _NativeMesh]] = []
    identity = np.eye(4, dtype=np.float64)
    for shape in root.findall("shape"):
        if shape.attrib.get("type") == "shapegroup":
            continue
        expanded.extend(
            expand(
                shape,
                parent_transform=identity,
                stack=(),
                name_prefix=None,
            )
        )
    return expanded


def _load_mitsuba_native_ply(
    scene_path: Path,
    *,
    scene_cls,
    merge_shapes: bool,
    merge_shapes_exclude_regex: str | None,
    frequency: float,
    metadata: dict[str, object] | None,
    vertical_ratio: float,
    edge_selection_mode: str,
    edge_diffraction: bool | None,
    boundary_edge_policy: str | None,
    source_root: str | Path | None,
):
    root = ET.parse(scene_path).getroot()
    material_defs = _material_defs(root)
    exclude_regex = (
        re.compile(merge_shapes_exclude_regex) if merge_shapes_exclude_regex else None
    )
    standalone: list[_NativeMesh] = []
    merge_groups: dict[str | None, list[_NativeMesh]] = {}

    for shape, mesh in _expanded_shapes(
        root, scene_path=scene_path, material_defs=material_defs
    ):
        shape_id = shape.attrib.get("id", mesh.name)
        excluded = exclude_regex is not None and exclude_regex.search(shape_id)
        if merge_shapes and not excluded and _xml_child_bool(shape, "face_normals"):
            merge_groups.setdefault(mesh.material_ref, []).append(mesh)
        else:
            standalone.append(mesh)

    native_meshes: list[_NativeMesh] = list(standalone)
    no_name_index = 1
    for material_ref in _ordered_merged_materials(merge_groups):
        meshes = merge_groups[material_ref]
        material_name = meshes[0].material_name
        if (
            material_name == "concrete"
            and len(meshes) == 1
            and meshes[0].name == "ground"
        ):
            native_meshes.append(_merge_meshes(meshes, "ground"))
            continue
        native_meshes.append(_merge_meshes(meshes, f"no-name-{no_name_index}"))
        no_name_index += 1

    structures: list[Structure] = []
    used: set[str] = set()
    for index, mesh in enumerate(native_meshes):
        material = _native_material(
            mesh.material_name,
            frequency,
            thickness_m=mesh.thickness_m,
            scattering_coefficient=mesh.scattering_coefficient,
            xpd_coefficient=mesh.xpd_coefficient,
        )
        name = _unique(_sanitize(mesh.name, f"structure-{index}"), used)
        structures.append(
            Structure(
                vertices=torch.from_numpy(mesh.vertices).to(dtype=torch.float32),
                faces=torch.from_numpy(mesh.faces).to(dtype=torch.int32),
                material=material,
                name=name,
                surface_id=index,
                uv=(
                    torch.from_numpy(mesh.uv).to(dtype=torch.float32)
                    if mesh.uv is not None
                    else None
                ),
                face_uv=(
                    torch.from_numpy(mesh.face_uv).to(dtype=torch.int32)
                    if mesh.face_uv is not None
                    else None
                ),
                metadata={
                    "sionna": {
                        "object_key": mesh.name,
                        "radio_material_name": mesh.material_name,
                        "material_ref": mesh.material_ref,
                        "thickness": mesh.thickness_m,
                        "scattering_coefficient": mesh.scattering_coefficient,
                        "xpd_coefficient": mesh.xpd_coefficient,
                        "source_shape_count": mesh.source_count,
                    }
                },
            )
        )

    resolved_metadata = dict(metadata or {})
    resolved_metadata["mitsuba"] = {
        "source_path": str(scene_path),
        "loader": "native_xml_ply",
        "compatible_loader": "sionna.rt.load_scene",
        "merge_shapes": bool(merge_shapes),
        "merge_shapes_exclude_regex": merge_shapes_exclude_regex,
        "frequency": float(frequency),
        "sionna_source_root": str(Path(source_root).resolve())
        if source_root is not None
        else None,
    }
    resolved_metadata["sionna_import_edge_policy"] = EdgePolicy(
        vertical_ratio=vertical_ratio,
        edge_selection_mode=edge_selection_mode,
        edge_diffraction=edge_diffraction,
        boundary_edge_policy=boundary_edge_policy,
    )

    return scene_cls(
        structures=structures,
        transmitters=[],
        receivers=[],
        frequency=float(frequency),
        metadata=resolved_metadata,
    )


def load_mitsuba(
    filename: str | Path,
    *,
    scene_cls,
    source_root: str | Path | None = None,
    merge_shapes: bool = True,
    merge_shapes_exclude_regex: str | None = None,
    frequency: float | None = None,
    metadata: dict[str, object] | None = None,
    native_loader: bool = True,
    vertical_ratio: float = 0.7,
    edge_selection_mode: str = "vertical_only",
    edge_diffraction: bool | None = True,
    boundary_edge_policy: str | None = None,
):
    scene_path = Path(filename).expanduser().resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Mitsuba scene file not found: {scene_path}")

    resolved_frequency = 3.5e9 if frequency is None else float(frequency)
    if not native_loader:
        raise ValueError("Scene.load_mitsuba requires the native loader")
    return _load_mitsuba_native_ply(
        scene_path,
        scene_cls=scene_cls,
        merge_shapes=bool(merge_shapes),
        merge_shapes_exclude_regex=merge_shapes_exclude_regex,
        frequency=resolved_frequency,
        metadata=metadata,
        vertical_ratio=vertical_ratio,
        edge_selection_mode=edge_selection_mode,
        edge_diffraction=edge_diffraction,
        boundary_edge_policy=boundary_edge_policy,
        source_root=source_root,
    )

# Keep the established compatibility path for introspection and pickle while
# this module owns the implementation.
_NativeMesh.__module__ = "witwin.channel.core.scene_loader"
itu_material_parameters.__module__ = "witwin.channel.core.scene_loader"
load_mitsuba.__module__ = "witwin.channel.core.scene_loader"
