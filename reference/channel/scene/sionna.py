from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from witwin.core import Material, Mesh, Structure

from .scene import Scene as ChannelScene


_LOCAL_SIONNA_CANDIDATES = (
    Path("sionna-rt-reference-2.0.0") / "src",
    Path("sionna-rt-reference") / "src",
)


@dataclass(frozen=True)
class SionnaImportResult:
    """Resolved Sionna RT import details."""

    rt: Any
    source: str
    source_root: Path | None


@dataclass(frozen=True)
class SionnaSceneConversionResult:
    """Converted in-memory Sionna RT scene plus conversion metadata."""

    scene: Any
    rt: Any
    source: str
    source_root: Path | None
    structure_name_map: dict[str, str]
    structure_material_map: dict[str, str]
    warnings: tuple[str, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _iter_sionna_source_roots(source_root: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if source_root is not None:
        explicit = Path(source_root).expanduser().resolve()
        if (explicit / "sionna" / "rt").exists():
            roots.append(explicit)
        if (explicit / "src" / "sionna" / "rt").exists():
            roots.append(explicit / "src")
    repo_root = _repo_root()
    for relative in _LOCAL_SIONNA_CANDIDATES:
        candidate = (repo_root / relative).resolve()
        if (candidate / "sionna" / "rt").exists():
            roots.append(candidate)

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique_roots.append(root)
    return unique_roots


def _path_is_within(path: Path | None, root: Path | None) -> bool:
    if path is None or root is None:
        return False
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_sionna_rt(
    *,
    source_root: str | Path | None = None,
    prefer_local: bool = True,
) -> SionnaImportResult:
    """Import Sionna RT, preferring the bundled local source tree when requested."""

    local_roots = _iter_sionna_source_roots(source_root)
    selected_root = local_roots[0] if local_roots else None

    if prefer_local and selected_root is not None and str(selected_root) not in sys.path:
        sys.path.insert(0, str(selected_root))

    try:
        rt = importlib.import_module("sionna.rt")
    except Exception as exc:
        searched = ", ".join(str(root) for root in local_roots) if local_roots else "<none>"
        raise ImportError(
            "Unable to import sionna.rt. "
            f"Checked local roots: {searched}. "
            "Install Sionna RT or place the source tree under "
            "`sionna-rt-reference-2.0.0/src`."
        ) from exc

    sionna_pkg = importlib.import_module("sionna")
    module_file = getattr(sionna_pkg, "__file__", None)
    module_path = Path(module_file).resolve() if module_file is not None else None

    if prefer_local and selected_root is not None and not _path_is_within(module_path, selected_root):
        raise RuntimeError(
            "Requested local Sionna RT import, but Python resolved a different "
            f"`sionna` package at '{module_path}'. Start from a fresh process or "
            "adjust `source_root` so the intended Sionna RT source is imported first."
        )

    if selected_root is not None and _path_is_within(module_path, selected_root):
        return SionnaImportResult(rt=rt, source="local_reference", source_root=selected_root)
    return SionnaImportResult(rt=rt, source="installed", source_root=None)


def _sanitize_identifier(name: str | None, *, prefix: str) -> str:
    text = "" if name is None else str(name)
    text = re.sub(r"[^0-9A-Za-z_-]+", "-", text).strip("-_")
    if not text:
        text = prefix
    if text[0].isdigit():
        text = f"{prefix}-{text}"
    return text


def _make_unique(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    index = 2
    while True:
        candidate = f"{base}-{index}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def _array_to_numpy(value, *, dtype):
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    if hasattr(value, "torch"):
        torch_value = value.torch()
        if isinstance(torch_value, torch.Tensor):
            return torch_value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _to_vertices_faces(geometry) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = geometry.to_mesh()
    vertices_np = _array_to_numpy(vertices, dtype=np.float32)
    faces_np = _array_to_numpy(faces, dtype=np.uint32)
    if vertices_np.ndim == 2 and vertices_np.shape[0] == 3 and vertices_np.shape[1] != 3:
        vertices_np = vertices_np.T
    if faces_np.ndim == 2 and faces_np.shape[0] == 3 and faces_np.shape[1] != 3:
        faces_np = faces_np.T
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise ValueError(
            "Geometry.to_mesh() must return vertices with shape [N, 3]. "
            f"Found {vertices_np.shape}."
        )
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(
            "Geometry.to_mesh() must return triangular faces with shape [M, 3]. "
            f"Found {faces_np.shape}."
        )
    return np.ascontiguousarray(vertices_np), np.ascontiguousarray(faces_np)


def _first_scalar(value, *, dtype) -> float:
    array = _array_to_numpy(value, dtype=dtype).reshape(-1)
    if array.size == 0:
        raise ValueError("Expected a non-empty scalar-like Sionna material parameter.")
    return float(array[0])


def _extract_sionna_mesh_buffers(mi_mesh) -> tuple[np.ndarray, np.ndarray]:
    mi = importlib.import_module("mitsuba")
    params = mi.traverse(mi_mesh)
    vertices = _array_to_numpy(params["vertex_positions"], dtype=np.float32).reshape(-1, 3)
    faces = _array_to_numpy(params["faces"], dtype=np.int32).reshape(-1, 3)
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def _material_signature(
    material: Material,
    *,
    default_thickness: float,
    scattering_coefficient: float,
    xpd_coefficient: float,
) -> tuple[Any, ...]:
    sample = material.evaluate_static()
    return (
        material.name,
        float(sample.eps_r),
        float(sample.mu_r),
        float(sample.sigma_e),
        float(default_thickness),
        float(scattering_coefficient),
        float(xpd_coefficient),
    )


def _build_mitsuba_mesh(mi, *, name: str, vertices: np.ndarray, faces: np.ndarray):
    mesh = mi.Mesh(
        name,
        int(vertices.shape[0]),
        int(faces.shape[0]),
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    params = mi.traverse(mesh)
    params["vertex_positions"] = vertices.reshape(-1)
    params["faces"] = faces.reshape(-1)
    params.update()
    return mesh


def scene_to_sionna_scene(
    scene: ChannelScene,
    *,
    source_root: str | Path | None = None,
    prefer_local: bool = True,
    include_disabled: bool = False,
    default_thickness: float = 0.1,
    scattering_coefficient: float = 0.0,
    xpd_coefficient: float = 0.0,
    strict_mu_r: bool = True,
    remove_duplicate_vertices: bool = False,
) -> SionnaSceneConversionResult:
    """Convert a declarative ``witwin.channel.Scene`` into an in-memory Sionna RT scene."""

    if not isinstance(scene, ChannelScene):
        raise TypeError("scene_to_sionna_scene expects a witwin.channel.Scene instance.")

    import_result = load_sionna_rt(source_root=source_root, prefer_local=prefer_local)
    rt = import_result.rt
    mi = importlib.import_module("mitsuba")

    sionna_scene = rt.Scene()
    material_cache: dict[tuple[Any, ...], Any] = {}
    used_structure_names: set[str] = set()
    used_material_names: set[str] = set()
    structure_name_map: dict[str, str] = {}
    structure_material_map: dict[str, str] = {}
    warnings: list[str] = []
    objects = []

    for index, structure in enumerate(scene.structures):
        if not include_disabled and not getattr(structure, "enabled", True):
            continue

        geometry = structure.geometry
        material = structure.material
        if not isinstance(material, Material):
            raise TypeError(
                "Sionna scene conversion currently expects witwin.core.Material instances. "
                f"Structure '{structure.name}' uses {type(material)}."
            )

        sample = material.evaluate_static()
        mu_r = float(sample.mu_r)
        if not np.isclose(mu_r, 1.0):
            message = (
                f"Structure '{structure.name}' uses mu_r={mu_r}, but Sionna RT materials "
                "assume mu_r=1.0. "
            )
            if strict_mu_r:
                raise ValueError(message + "Set strict_mu_r=False to convert while recording a warning.")
            warnings.append(message + "Proceeding with mu_r ignored.")

        material_key = _material_signature(
            material,
            default_thickness=default_thickness,
            scattering_coefficient=scattering_coefficient,
            xpd_coefficient=xpd_coefficient,
        )
        radio_material = material_cache.get(material_key)
        if radio_material is None:
            material_base = _sanitize_identifier(material.name, prefix="material")
            material_name = _make_unique(material_base, used_material_names)
            radio_material = rt.RadioMaterial(
                name=material_name,
                thickness=float(default_thickness),
                relative_permittivity=float(sample.eps_r),
                conductivity=float(sample.sigma_e),
                scattering_coefficient=float(scattering_coefficient),
                xpd_coefficient=float(xpd_coefficient),
            )
            material_cache[material_key] = radio_material

        vertices, faces = _to_vertices_faces(geometry)
        structure_base = _sanitize_identifier(structure.name, prefix=f"structure-{index}")
        structure_name = _make_unique(structure_base, used_structure_names)
        mi_mesh = _build_mitsuba_mesh(mi, name=structure_name, vertices=vertices, faces=faces)
        objects.append(
            rt.SceneObject(
                mi_mesh=mi_mesh,
                name=structure_name,
                radio_material=radio_material,
                remove_duplicate_vertices=remove_duplicate_vertices,
            )
        )
        source_name = structure.name if structure.name is not None else f"structure-{index}"
        structure_name_map[source_name] = structure_name
        structure_material_map[source_name] = radio_material.name

    if objects:
        sionna_scene.edit(add=objects)

    return SionnaSceneConversionResult(
        scene=sionna_scene,
        rt=rt,
        source=import_result.source,
        source_root=import_result.source_root,
        structure_name_map=structure_name_map,
        structure_material_map=structure_material_map,
        warnings=tuple(warnings),
    )


def scene_from_sionna_scene(
    sionna_scene,
    *,
    scene_cls: type[ChannelScene] = ChannelScene,
    config=None,
    monitors=None,
    metadata=None,
    device: str | None = "cuda",
    verbose: bool = False,
    vertical_ratio: float = 0.7,
    edge_selection_mode: str = "vertical_only",
    boundary_edge_policy: str = "exclude",
) -> ChannelScene:
    """Convert an in-memory ``sionna.rt.Scene`` into a declarative ``witwin.channel.Scene``."""

    objects = getattr(sionna_scene, "objects", None)
    if not isinstance(objects, dict):
        raise TypeError("scene_from_sionna_scene expects a sionna.rt.Scene with an object dictionary.")

    material_cache: dict[tuple[Any, ...], Material] = {}
    used_names: set[str] = set()
    structures: list[Structure] = []

    for index, (object_key, scene_object) in enumerate(objects.items()):
        structure_name = getattr(scene_object, "name", None) or str(object_key) or f"structure-{index}"
        structure_name = _make_unique(str(structure_name), used_names)

        vertices, faces = _extract_sionna_mesh_buffers(scene_object.mi_mesh)
        geometry = Mesh(vertices=vertices, faces=faces, position=(0.0, 0.0, 0.0), recenter=False, device="cpu")

        radio_material = getattr(scene_object, "radio_material", None)
        if radio_material is None:
            material = Material(name=f"{structure_name}-material")
        else:
            material_key = (
                getattr(radio_material, "name", None),
                _first_scalar(radio_material.relative_permittivity, dtype=np.float32),
                _first_scalar(radio_material.conductivity, dtype=np.float32),
                _first_scalar(radio_material.thickness, dtype=np.float32),
                _first_scalar(radio_material.scattering_coefficient, dtype=np.float32),
                _first_scalar(radio_material.xpd_coefficient, dtype=np.float32),
            )
            material = material_cache.get(material_key)
            if material is None:
                material = Material(
                    name=getattr(radio_material, "name", None),
                    eps_r=material_key[1],
                    mu_r=1.0,
                    sigma_e=material_key[2],
                )
                material_cache[material_key] = material

        structures.append(
            Structure(
                geometry=geometry,
                material=material,
                name=structure_name,
                metadata={
                    "sionna": {
                        "object_key": str(object_key),
                        "radio_material_name": getattr(radio_material, "name", None),
                        "thickness": None if radio_material is None else _first_scalar(radio_material.thickness, dtype=np.float32),
                        "scattering_coefficient": None if radio_material is None else _first_scalar(radio_material.scattering_coefficient, dtype=np.float32),
                        "xpd_coefficient": None if radio_material is None else _first_scalar(radio_material.xpd_coefficient, dtype=np.float32),
                    }
                },
            )
        )

    return scene_cls(
        config=config,
        structures=structures,
        monitors=monitors,
        metadata=metadata,
        device=device,
        verbose=verbose,
        vertical_ratio=vertical_ratio,
        edge_selection_mode=edge_selection_mode,
        boundary_edge_policy=boundary_edge_policy,
    )


__all__ = [
    "SionnaImportResult",
    "SionnaSceneConversionResult",
    "load_sionna_rt",
    "scene_to_sionna_scene",
    "scene_from_sionna_scene",
]
