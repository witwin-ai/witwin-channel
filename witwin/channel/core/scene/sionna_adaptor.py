from __future__ import annotations

import importlib
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch

from witwin.core import Material, Mesh, Structure
from .edge_policy import EdgePolicy

if TYPE_CHECKING:
    from .scene import Scene

_LOCAL_SIONNA_CANDIDATES = (Path("sionna-rt-reference-2.0.0") / "src", Path("sionna-rt-reference") / "src")
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _source_roots(source_root: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if source_root is not None:
        explicit = Path(source_root).expanduser().resolve()
        if (explicit / "sionna" / "rt").exists():
            roots.append(explicit)
        if (explicit / "src" / "sionna" / "rt").exists():
            roots.append(explicit / "src")
    for rel in _LOCAL_SIONNA_CANDIDATES:
        candidate = (_REPO_ROOT / rel).resolve()
        if (candidate / "sionna" / "rt").exists():
            roots.append(candidate)
    seen: set[Path] = set()
    return [r for r in roots if not (r in seen or seen.add(r))]


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


def _is_within(path: Path | None, root: Path | None) -> bool:
    if path is None or root is None:
        return False
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _sanitize(name: str | None, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_-]+", "-", "" if name is None else str(name)).strip("-_") or fallback
    return f"{fallback}-{text}" if text[0].isdigit() else text


def _unique(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while (c := f"{base}-{i}") in used:
        i += 1
    used.add(c)
    return c


def _to_np(value, *, dtype) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    if hasattr(value, "torch"):
        t = value.torch()
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _scalar(value) -> float:
    arr = _to_np(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Expected a non-empty scalar-like Sionna material parameter.")
    return float(arr[0])


def _to_mesh(geometry) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = geometry.to_mesh()
    v = _to_np(verts, dtype=np.float32)
    f = _to_np(faces, dtype=np.uint32)
    if v.ndim == 2 and v.shape[0] == 3 and v.shape[1] != 3:
        v = v.T
    if f.ndim == 2 and f.shape[0] == 3 and f.shape[1] != 3:
        f = f.T
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Expected vertices [N,3], got {v.shape}.")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"Expected faces [M,3], got {f.shape}.")
    return np.ascontiguousarray(v), np.ascontiguousarray(f)


def _extract_mi_mesh(mi_mesh) -> tuple[np.ndarray, np.ndarray]:
    params = importlib.import_module("mitsuba").traverse(mi_mesh)
    v = _to_np(params["vertex_positions"], dtype=np.float32).reshape(-1, 3)
    f = _to_np(params["faces"], dtype=np.int32).reshape(-1, 3)
    return np.ascontiguousarray(v), np.ascontiguousarray(f)


def _build_mi_mesh(mi, name: str, verts: np.ndarray, faces: np.ndarray):
    mesh = mi.Mesh(name, int(verts.shape[0]), int(faces.shape[0]),
                   has_vertex_normals=False, has_vertex_texcoords=False)
    params = mi.traverse(mesh)
    params["vertex_positions"] = verts.reshape(-1)
    params["faces"] = faces.reshape(-1)
    params.update()
    return mesh


def _resolve_scene_cls(scene_cls):
    if scene_cls is not None:
        return scene_cls
    from .scene import Scene as _Scene
    return _Scene


def _is_material_like(value) -> bool:
    return isinstance(value, Material) or (
        type(value).__name__ == "Material"
        and hasattr(value, "evaluate_static")
        and hasattr(value, "name")
    )


def _pattern_name(value) -> str:
    text = str(value or "iso").lower()
    if "tr38901" in text or "38.901" in text:
        return "tr38901"
    if "dipole" in text:
        return "dipole"
    return "iso"


def _import_sionna_array(value):
    if value is None:
        return None
    from .arrays import AntennaArray, PlanarArray

    pattern = _pattern_name(getattr(value, "antenna_pattern", getattr(value, "pattern", "iso")))
    polarization = str(getattr(value, "polarization", "V")).upper()
    if hasattr(value, "num_rows") and hasattr(value, "num_cols"):
        return PlanarArray(
            num_rows=int(getattr(value, "num_rows")),
            num_cols=int(getattr(value, "num_cols")),
            vertical_spacing=_scalar(getattr(value, "vertical_spacing", 0.5)),
            horizontal_spacing=_scalar(getattr(value, "horizontal_spacing", 0.5)),
            polarization=polarization,
            pattern=pattern,
        )
    positions = getattr(value, "element_positions", getattr(value, "positions", None))
    if positions is not None:
        return AntennaArray(
            element_positions=_to_np(positions, dtype=np.float32).reshape(-1, 3),
            polarization=polarization,
            pattern=pattern,
        )
    return None


class SionnaAdaptor:
    """Static adaptor between witwin.channel.core.scene.Scene and sionna.rt.Scene."""

    @staticmethod
    def load_rt(*, source_root: str | Path | None = None, prefer_local: bool = True) -> Any:
        roots = _source_roots(source_root)
        selected = roots[0] if roots else None
        with _local_sionna_on_path(selected, prefer_local=prefer_local):
            try:
                rt = importlib.import_module("sionna.rt")
            except Exception as exc:
                searched = ", ".join(str(r) for r in roots) if roots else "<none>"
                raise ImportError(
                    f"Unable to import sionna.rt. Checked local roots: {searched}. "
                    "Install Sionna RT or place the source tree under `sionna-rt-reference-2.0.0/src`."
                ) from exc
            sionna_pkg = importlib.import_module("sionna")

        mod_file = getattr(sionna_pkg, "__file__", None)
        mod_path = Path(mod_file).resolve() if mod_file else None
        if prefer_local and selected is not None and not _is_within(mod_path, selected):
            raise RuntimeError(
                "Requested local Sionna RT import, but Python resolved a different "
                f"`sionna` package at '{mod_path}'. Start from a fresh process or "
                "adjust `source_root` so the intended source is imported first."
            )
        return rt

    @staticmethod
    def load_mitsuba(
        filename: str | Path,
        *,
        scene_cls: type[Scene] | None = None,
        source_root: str | Path | None = None,
        prefer_local: bool = True,
        merge_shapes: bool = True,
        merge_shapes_exclude_regex: str | None = None,
        remove_duplicate_vertices: bool = False,
        frequency: float | None = None,
        metadata=None,
        device: str | None = "cuda",
        verbose: bool = False,
        vertical_ratio: float = 0.7,
        edge_selection_mode: str = "vertical_only",
        edge_diffraction: bool | None = None,
        boundary_edge_policy: str | None = None,
    ) -> Scene:
        scene_path = Path(filename).expanduser().resolve()
        if not scene_path.exists():
            raise FileNotFoundError(f"Mitsuba scene file not found: {scene_path}")

        roots = _source_roots(source_root)
        rt = SionnaAdaptor.load_rt(source_root=source_root, prefer_local=prefer_local)
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

        resolved_metadata = dict(metadata or {})
        resolved_metadata["mitsuba"] = {
            "source_path": str(scene_path),
            "loader": "sionna.rt.load_scene",
            "merge_shapes": bool(merge_shapes),
            "merge_shapes_exclude_regex": merge_shapes_exclude_regex,
            "remove_duplicate_vertices": bool(remove_duplicate_vertices),
            "frequency": None if frequency is None else float(frequency),
            "sionna_source_root": str(roots[0]) if roots else None,
        }
        scene = SionnaAdaptor.import_scene(
            sionna_scene, scene_cls=scene_cls, metadata=resolved_metadata, device=device,
            verbose=verbose,
            vertical_ratio=vertical_ratio,
            edge_selection_mode=edge_selection_mode,
            edge_diffraction=edge_diffraction,
            boundary_edge_policy=boundary_edge_policy,
        )
        return scene

    @staticmethod
    def export(
        scene: Scene,
        *,
        source_root: str | Path | None = None,
        prefer_local: bool = True,
        include_disabled: bool = False,
        default_thickness: float = 0.1,
        scattering_coefficient: float = 0.0,
        xpd_coefficient: float = 0.0,
        strict_mu_r: bool = True,
        remove_duplicates: bool = False,
    ) -> Any:
        rt = SionnaAdaptor.load_rt(source_root=source_root, prefer_local=prefer_local)
        mi = importlib.import_module("mitsuba")

        sionna_scene = rt.Scene()
        mat_cache: dict[tuple[Any, ...], Any] = {}
        used_structs: set[str] = set()
        used_mats: set[str] = set()
        objects = []

        for idx, structure in enumerate(scene.structures):
            if not include_disabled and not getattr(structure, "enabled", True):
                continue
            material = structure.material
            if not _is_material_like(material):
                raise TypeError(f"Structure '{structure.name}' uses {type(material)}, expected witwin.core.Material.")

            sample = material.evaluate_static()
            mu_r = float(sample.mu_r)
            if strict_mu_r and not np.isclose(mu_r, 1.0):
                raise ValueError(
                    f"Structure '{structure.name}' uses mu_r={mu_r}, but Sionna RT assumes mu_r=1.0. "
                    "Set strict_mu_r=False to skip this check."
                )

            mat_key = (
                material.name,
                float(sample.eps_r), float(sample.mu_r), float(sample.sigma_e),
                float(default_thickness), float(scattering_coefficient), float(xpd_coefficient),
            )
            radio_mat = mat_cache.get(mat_key)
            if radio_mat is None:
                radio_mat = rt.RadioMaterial(
                    name=_unique(_sanitize(material.name, "material"), used_mats),
                    thickness=float(default_thickness),
                    relative_permittivity=float(sample.eps_r),
                    conductivity=float(sample.sigma_e),
                    scattering_coefficient=float(scattering_coefficient),
                    xpd_coefficient=float(xpd_coefficient),
                )
                mat_cache[mat_key] = radio_mat

            verts, faces = _to_mesh(structure.geometry)
            struct_name = _unique(_sanitize(structure.name, f"structure-{idx}"), used_structs)
            objects.append(rt.SceneObject(
                mi_mesh=_build_mi_mesh(mi, struct_name, verts, faces),
                name=struct_name, radio_material=radio_mat,
                remove_duplicate_vertices=remove_duplicates,
            ))

        if objects:
            sionna_scene.edit(add=objects)
        return sionna_scene

    @staticmethod
    def import_scene(
        sionna_scene,
        *,
        scene_cls: type[Scene] | None = None,
        metadata=None,
        device: str | None = "cuda",
        verbose: bool = False,
        vertical_ratio: float = 0.7,
        edge_selection_mode: str = "vertical_only",
        edge_diffraction: bool | None = None,
        boundary_edge_policy: str | None = None,
    ) -> Scene:
        scene_cls = _resolve_scene_cls(scene_cls)
        objects = getattr(sionna_scene, "objects", None)
        if not isinstance(objects, dict):
            raise TypeError("Expected a sionna.rt.Scene with an object dictionary.")

        mat_cache: dict[tuple[Any, ...], Material] = {}
        used: set[str] = set()
        structures: list[Structure] = []

        for idx, (key, obj) in enumerate(objects.items()):
            name = _unique(str(getattr(obj, "name", None) or key or f"structure-{idx}"), used)
            verts, faces = _extract_mi_mesh(obj.mi_mesh)
            geometry = Mesh(vertices=verts, faces=faces, position=(0.0, 0.0, 0.0), recenter=False, device="cpu")

            radio_mat = getattr(obj, "radio_material", None)
            if radio_mat is None:
                material = Material(name=f"{name}-material")
                radio_meta = {"object_key": str(key), "radio_material_name": None,
                              "thickness": None, "scattering_coefficient": None, "xpd_coefficient": None}
            else:
                mat_key = (
                    getattr(radio_mat, "name", None),
                    _scalar(radio_mat.relative_permittivity), _scalar(radio_mat.conductivity),
                    _scalar(radio_mat.thickness), _scalar(radio_mat.scattering_coefficient),
                    _scalar(radio_mat.xpd_coefficient),
                )
                material = mat_cache.get(mat_key)
                if material is None:
                    material = Material(name=mat_key[0], eps_r=mat_key[1], mu_r=1.0, sigma_e=mat_key[2])
                    mat_cache[mat_key] = material
                radio_meta = {"object_key": str(key), "radio_material_name": mat_key[0],
                              "thickness": mat_key[3], "scattering_coefficient": mat_key[4], "xpd_coefficient": mat_key[5]}

            structures.append(Structure(geometry=geometry, material=material, name=name,
                                        metadata={"sionna": radio_meta}))

        resolved_metadata = dict(metadata or {})
        resolved_metadata.setdefault("sionna_import_edge_policy", EdgePolicy(
            vertical_ratio=vertical_ratio,
            edge_selection_mode=edge_selection_mode,
            edge_diffraction=edge_diffraction,
            boundary_edge_policy=boundary_edge_policy,
        ))
        scene_frequency = getattr(sionna_scene, "frequency", None)
        scene_frequency = None if scene_frequency is None else _scalar(scene_frequency)
        return scene_cls(
            structures=structures,
            metadata=resolved_metadata,
            frequency=scene_frequency,
            tx_array=_import_sionna_array(getattr(sionna_scene, "tx_array", None)),
            rx_array=_import_sionna_array(getattr(sionna_scene, "rx_array", None)),
            device=device,
            verbose=verbose,
        )


__all__ = ["SionnaAdaptor"]
