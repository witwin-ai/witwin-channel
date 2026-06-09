from __future__ import annotations

from dataclasses import dataclass
import importlib
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .edge_policy import EdgePolicy
from .materials import Dielectric
from .objects import Structure


_REPO_ROOT = Path(__file__).resolve().parents[5]
_LOCAL_SIONNA_CANDIDATES = (
    _REPO_ROOT / "reference" / "sionna-rt-reference-2.0.1" / "src",
    _REPO_ROOT.parent / "channel" / "reference" / "sionna-rt-reference-2.0.1" / "src",
    _REPO_ROOT.parent / "channel" / "sionna-rt-reference-2.0.1" / "src",
)

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
    vertices: np.ndarray
    faces: np.ndarray
    source_count: int = 1


def _source_roots(source_root: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if source_root is not None:
        explicit = Path(source_root).expanduser().resolve()
        if (explicit / "sionna" / "rt").exists():
            roots.append(explicit)
        if (explicit / "src" / "sionna" / "rt").exists():
            roots.append(explicit / "src")
    for candidate in _LOCAL_SIONNA_CANDIDATES:
        resolved = candidate.resolve()
        if (resolved / "sionna" / "rt").exists():
            roots.append(resolved)
    seen: set[Path] = set()
    return [root for root in roots if not (root in seen or seen.add(root))]


@contextmanager
def _local_sionna_on_path(selected: Path | None, *, prefer_local: bool) -> Iterator[None]:
    entry = None if selected is None else str(selected)
    added = prefer_local and entry is not None and entry not in sys.path
    if added:
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        pass


def _sanitize(name: str | None, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_-]+", "-", "" if name is None else str(name)).strip("-_") or fallback
    return f"{fallback}-{text}" if text[0].isdigit() else text


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


def _xml_child_value(node: ET.Element, tag: str, name: str, default: str | None = None) -> str | None:
    child = node.find(f"{tag}[@name='{name}']")
    if child is None:
        return default
    return child.attrib.get("value", default)


def _xml_child_bool(node: ET.Element, name: str, default: bool = False) -> bool:
    value = _xml_child_value(node, "boolean", name)
    if value is None:
        return default
    return value.lower() in {"true", "1", "yes"}


def _itu_parameters(material_name: str, frequency_hz: float) -> tuple[float, float]:
    f_ghz = float(frequency_hz) / 1.0e9
    for (f_min, f_max), (a, b, c, d) in _ITU_MATERIALS_PROPERTIES[material_name]:
        if f_min <= f_ghz <= f_max:
            return a * (f_ghz**b), c * (f_ghz**d)
    raise ValueError(f"ITU material {material_name!r} is not defined for {frequency_hz} Hz")


def _native_material(material_name: str | None, frequency_hz: float) -> Dielectric:
    if material_name is None:
        return Dielectric(eps_r=1.0)
    if material_name not in _ITU_MATERIALS_PROPERTIES:
        raise ValueError(f"unsupported ITU radio material: {material_name!r}")
    eps_r, sigma = _itu_parameters(material_name, frequency_hz)
    return Dielectric(eps_r=eps_r, mu_r=1.0, sigma_e=sigma)


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

    if len(lines) < 3 or lines[0] != "ply" or lines[1] != "format binary_little_endian 1.0":
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
                raise ValueError(f"unsupported vertex PLY property: {line}")
            vertex_properties.append((parts[2], parts[1]))
        elif parts[0] == "property" and current_element == "face":
            if parts != ["property", "list", "uchar", "int", "vertex_indices"]:
                raise ValueError(f"unsupported face PLY property: {line}")

    if vertex_count <= 0 or face_count < 0:
        raise ValueError("invalid PLY element counts")
    if not {"x", "y", "z"}.issubset({name for name, _ in vertex_properties}):
        raise ValueError("PLY vertices must include x, y, and z properties")
    return lines, vertex_count, vertex_properties, face_count


def _load_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        _lines, vertex_count, vertex_properties, face_count = _read_ply_header(handle)
        vertex_dtype = np.dtype(
            [(name, _PLY_SCALAR_DTYPES[prop_type]) for name, prop_type in vertex_properties]
        )
        vertex_data = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
        if int(vertex_data.shape[0]) != vertex_count:
            raise ValueError(f"truncated PLY vertex data: {path}")
        vertices = np.column_stack(
            (vertex_data["x"], vertex_data["y"], vertex_data["z"])
        ).astype(np.float32, copy=False)

        face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
        face_data = np.fromfile(handle, dtype=face_dtype, count=face_count)
        if int(face_data.shape[0]) != face_count:
            raise ValueError(f"truncated PLY face data: {path}")
        if not np.all(face_data["count"] == 3):
            raise ValueError(f"only triangular PLY faces are supported: {path}")
        faces = face_data["indices"].astype(np.int32, copy=False)

    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def _merge_meshes(meshes: list[_NativeMesh], name: str) -> _NativeMesh:
    if not meshes:
        raise ValueError("cannot merge an empty mesh group")
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + vertex_offset)
        vertex_offset += int(mesh.vertices.shape[0])
    return _NativeMesh(
        name=name,
        material_name=meshes[0].material_name,
        vertices=np.ascontiguousarray(np.concatenate(vertices, axis=0)),
        faces=np.ascontiguousarray(np.concatenate(faces, axis=0)),
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


def _ordered_merged_materials(groups: dict[str | None, list[_NativeMesh]]) -> list[str | None]:
    ordered: list[str | None] = []
    for material_name in _MUNICH_MERGED_GROUP_ORDER:
        if material_name in groups:
            ordered.append(material_name)
    for material_name in groups:
        if material_name not in ordered and material_name != "concrete":
            ordered.append(material_name)
    if "concrete" in groups:
        ordered.append("concrete")
    return ordered


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
    exclude_regex = re.compile(merge_shapes_exclude_regex) if merge_shapes_exclude_regex else None
    standalone: list[_NativeMesh] = []
    merge_groups: dict[str | None, list[_NativeMesh]] = {}

    for shape in root.findall("shape"):
        shape_type = shape.attrib.get("type")
        if shape_type != "ply":
            raise ValueError(f"native Mitsuba loader only supports top-level PLY shapes, got {shape_type!r}")
        filename = _xml_child_value(shape, "string", "filename")
        if filename is None:
            raise ValueError("PLY shape is missing a filename")
        mesh_path = (scene_path.parent / filename).resolve()
        shape_id = shape.attrib.get("id", mesh_path.stem)
        object_key = shape_id[5:] if shape_id.startswith("mesh-") else shape_id
        material_id = _shape_ref_material(shape)
        material_name = str(material_defs.get(material_id, {}).get("type", material_id or ""))
        if material_id is None:
            material_name = None
        vertices, faces = _load_ply_mesh(mesh_path)
        mesh = _NativeMesh(
            name=object_key,
            material_name=material_name,
            vertices=vertices,
            faces=faces,
        )
        excluded = exclude_regex is not None and exclude_regex.search(shape_id)
        if merge_shapes and not excluded and _xml_child_bool(shape, "face_normals"):
            merge_groups.setdefault(material_name, []).append(mesh)
        else:
            standalone.append(mesh)

    native_meshes: list[_NativeMesh] = list(standalone)
    no_name_index = 1
    for material_name in _ordered_merged_materials(merge_groups):
        meshes = merge_groups[material_name]
        if material_name == "concrete" and len(meshes) == 1 and meshes[0].name == "ground":
            native_meshes.append(_merge_meshes(meshes, "ground"))
            continue
        native_meshes.append(_merge_meshes(meshes, f"no-name-{no_name_index}"))
        no_name_index += 1

    structures: list[Structure] = []
    used: set[str] = set()
    for index, mesh in enumerate(native_meshes):
        material = _native_material(mesh.material_name, frequency)
        material_info = material_defs.get(mesh.material_name or "", {})
        name = _unique(_sanitize(mesh.name, f"structure-{index}"), used)
        structures.append(
            Structure(
                vertices=torch.from_numpy(mesh.vertices).to(dtype=torch.float32),
                faces=torch.from_numpy(mesh.faces).to(dtype=torch.int32),
                material=material,
                name=name,
                surface_id=index,
                metadata={
                    "sionna": {
                        "object_key": mesh.name,
                        "radio_material_name": mesh.material_name,
                        "thickness": material_info.get("thickness"),
                        "scattering_coefficient": material_info.get("scattering_coefficient", 0.0),
                        "xpd_coefficient": material_info.get("xpd_coefficient", 0.0),
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
        "remove_duplicate_vertices": False,
        "frequency": float(frequency),
        "sionna_source_root": str(Path(source_root).resolve()) if source_root is not None else None,
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


def _to_np(value, *, dtype) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    if hasattr(value, "torch"):
        tensor = value.torch()
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _scalar(value) -> float:
    array = _to_np(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("expected a non-empty scalar-like material parameter")
    return float(array[0])


def _extract_mi_mesh(mi_mesh) -> tuple[np.ndarray, np.ndarray]:
    params = importlib.import_module("mitsuba").traverse(mi_mesh)
    vertices = _to_np(params["vertex_positions"], dtype=np.float32).reshape(-1, 3)
    faces = _to_np(params["faces"], dtype=np.int32).reshape(-1, 3)
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def load_rt(*, source_root: str | Path | None = None, prefer_local: bool = True) -> Any:
    roots = _source_roots(source_root)
    selected = roots[0] if roots else None
    with _local_sionna_on_path(selected, prefer_local=prefer_local):
        try:
            return importlib.import_module("sionna.rt")
        except Exception as exc:
            searched = ", ".join(str(root) for root in roots) if roots else "<none>"
            raise ImportError(
                f"Unable to import sionna.rt. Checked local roots: {searched}."
            ) from exc


def load_mitsuba(
    filename: str | Path,
    *,
    scene_cls,
    source_root: str | Path | None = None,
    prefer_local: bool = True,
    merge_shapes: bool = True,
    merge_shapes_exclude_regex: str | None = None,
    remove_duplicate_vertices: bool = False,
    frequency: float | None = None,
    metadata: dict[str, object] | None = None,
    native_loader: bool = True,
    allow_python_fallback: bool = False,
    vertical_ratio: float = 0.7,
    edge_selection_mode: str = "vertical_only",
    edge_diffraction: bool | None = None,
    boundary_edge_policy: str | None = None,
    **_ignored,
):
    scene_path = Path(filename).expanduser().resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Mitsuba scene file not found: {scene_path}")

    resolved_frequency = 3.5e9 if frequency is None else float(frequency)
    if native_loader:
        try:
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
        except Exception:
            if not allow_python_fallback:
                raise

    roots = _source_roots(source_root)
    rt = load_rt(source_root=source_root, prefer_local=prefer_local)
    try:
        sionna_scene = rt.load_scene(
            str(scene_path),
            merge_shapes=bool(merge_shapes),
            merge_shapes_exclude_regex=merge_shapes_exclude_regex,
            remove_duplicate_vertices=bool(remove_duplicate_vertices),
        )
    except TypeError:
        sionna_scene = rt.load_scene(str(scene_path))
    if frequency is not None:
        sionna_scene.frequency = float(frequency)

    structures: list[Structure] = []
    used: set[str] = set()
    mat_cache: dict[tuple[object, ...], Dielectric] = {}
    objects = getattr(sionna_scene, "objects", None)
    if not isinstance(objects, dict):
        raise TypeError("expected a sionna.rt.Scene with an object dictionary")

    for index, (key, obj) in enumerate(objects.items()):
        name = _unique(str(getattr(obj, "name", None) or key or f"structure-{index}"), used)
        vertices_np, faces_np = _extract_mi_mesh(obj.mi_mesh)
        radio_mat = getattr(obj, "radio_material", None)
        if radio_mat is None:
            material = Dielectric(eps_r=1.0)
            radio_meta = {
                "object_key": str(key),
                "radio_material_name": None,
                "thickness": None,
                "scattering_coefficient": None,
                "xpd_coefficient": None,
            }
        else:
            mat_key = (
                getattr(radio_mat, "name", None),
                _scalar(radio_mat.relative_permittivity),
                _scalar(radio_mat.conductivity),
                _scalar(radio_mat.thickness),
                _scalar(radio_mat.scattering_coefficient),
                _scalar(radio_mat.xpd_coefficient),
            )
            material = mat_cache.get(mat_key)
            if material is None:
                material = Dielectric(eps_r=float(mat_key[1]), mu_r=1.0, sigma_e=float(mat_key[2]))
                mat_cache[mat_key] = material
            radio_meta = {
                "object_key": str(key),
                "radio_material_name": mat_key[0],
                "thickness": mat_key[3],
                "scattering_coefficient": mat_key[4],
                "xpd_coefficient": mat_key[5],
            }

        structures.append(
            Structure(
                vertices=torch.from_numpy(vertices_np).to(dtype=torch.float32),
                faces=torch.from_numpy(faces_np).to(dtype=torch.int32),
                material=material,
                name=name,
                surface_id=index,
                metadata={"sionna": radio_meta},
            )
        )

    scene_frequency = getattr(sionna_scene, "frequency", None)
    resolved_frequency = frequency
    if resolved_frequency is None and scene_frequency is not None:
        resolved_frequency = _scalar(scene_frequency)
    if resolved_frequency is None:
        resolved_frequency = 3.5e9

    resolved_metadata = dict(metadata or {})
    resolved_metadata["mitsuba"] = {
        "source_path": str(scene_path),
        "loader": "sionna.rt.load_scene",
        "merge_shapes": bool(merge_shapes),
        "merge_shapes_exclude_regex": merge_shapes_exclude_regex,
        "remove_duplicate_vertices": bool(remove_duplicate_vertices),
        "frequency": float(resolved_frequency),
        "sionna_source_root": str(roots[0]) if roots else None,
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
        frequency=float(resolved_frequency),
        metadata=resolved_metadata,
    )
