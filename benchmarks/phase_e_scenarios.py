from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import torch

from witwin.channel import (
    ReceiverGrid,
    ReceiverPoint,
    Scene,
    Structure,
    Transmitter,
)
from witwin.channel.core.materials import Dielectric


SCENARIO_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "scenarios" / "phase_e_scenarios.v1.json"
)
SCENE_ROOT_ENV = "WITWIN_CHANNEL_NATIVE_SCENE_ROOT"


class FullScenarioAssetError(RuntimeError):
    """Raised when a required full city asset cannot be resolved."""


@dataclass(frozen=True, slots=True)
class ScenarioRecord:
    name: str
    mode: str
    source_sha256: str
    scene_sha256: str
    triangle_count: int
    transmitter_count: int
    receiver_count: int
    receiver_container_count: int
    receiver_grid_cells: int
    source_path: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScenarioBundle:
    scene: Scene
    record: ScenarioRecord


def load_manifest() -> dict[str, Any]:
    return json.loads(SCENARIO_MANIFEST_PATH.read_text(encoding="utf-8"))


def scenario_names() -> tuple[str, ...]:
    return tuple(row["name"] for row in load_manifest()["scenarios"])


def _scenario_spec(name: str) -> dict[str, Any]:
    for row in load_manifest()["scenarios"]:
        if row["name"] == name:
            return row
    raise ValueError(f"unknown Phase E scenario {name!r}; expected one of {scenario_names()}")


def _point_endpoints(
    *, tx_count: int, receiver_count: int, tx_origin: tuple[float, float, float]
) -> tuple[list[Transmitter], list[ReceiverPoint]]:
    if tx_count <= 0 or receiver_count <= 0:
        raise ValueError("tx_count and receiver_count must be positive")
    tx = [
        Transmitter(
            position=torch.tensor(
                [tx_origin[0], tx_origin[1] + 0.25 * index, tx_origin[2]],
                dtype=torch.float32,
            )
        )
        for index in range(tx_count)
    ]
    rx = [
        ReceiverPoint(
            position=torch.tensor(
                [8.0, -2.0 + 4.0 * (index + 0.5) / receiver_count, 1.5],
                dtype=torch.float32,
            )
        )
        for index in range(receiver_count)
    ]
    return tx, rx


def _analytic_scene(*, tx_count: int, receiver_count: int) -> Scene:
    wall = Structure(
        vertices=torch.tensor(
            [[2.0, -4.0, 0.0], [2.0, 4.0, 0.0], [2.0, -4.0, 5.0], [2.0, 4.0, 5.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32),
        material=Dielectric(eps_r=4.0, sigma_e=0.01),
        name="analytic-wall",
        surface_id=1,
    )
    transmitters, receivers = _point_endpoints(
        tx_count=tx_count, receiver_count=receiver_count, tx_origin=(-2.0, 0.0, 1.5)
    )
    return Scene(
        structures=[wall],
        transmitters=transmitters,
        receivers=receivers,
        frequency=3.0e9,
        metadata={"phase_e_scenario": "analytic"},
    )


def _cube(center: tuple[float, float, float], *, index: int) -> Structure:
    cx, cy, cz = center
    vertices = torch.tensor(
        [
            [cx + x, cy + y, cz + z]
            for z in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for x in (-1.0, 1.0)
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1], [1, 2, 3],
            [4, 5, 6], [5, 7, 6],
            [0, 1, 4], [1, 5, 4],
            [2, 6, 3], [3, 6, 7],
            [0, 4, 2], [2, 4, 6],
            [1, 3, 5], [3, 7, 5],
        ],
        dtype=torch.int32,
    )
    return Structure(
        vertices=vertices,
        faces=faces,
        material=Dielectric(eps_r=3.5 + 0.5 * index, sigma_e=0.01),
        name=f"cube-{index}",
        surface_id=index,
    )


def _three_cube_scene(*, tx_count: int, receiver_count: int) -> Scene:
    structures = [
        _cube((0.0, -2.0, 1.0), index=1),
        _cube((3.0, 2.0, 1.0), index=2),
        _cube((6.0, -2.0, 1.0), index=3),
    ]
    transmitters, receivers = _point_endpoints(
        tx_count=tx_count, receiver_count=receiver_count, tx_origin=(-4.0, 0.0, 1.5)
    )
    return Scene(
        structures=structures,
        transmitters=transmitters,
        receivers=receivers,
        frequency=3.0e9,
        metadata={"phase_e_scenario": "three_cube"},
    )


def _terrain_mesh(size: int = 9) -> Structure:
    vertices = []
    for y in range(size):
        for x in range(size):
            px = -20.0 + 40.0 * x / (size - 1)
            py = -20.0 + 40.0 * y / (size - 1)
            pz = 0.8 * math.sin(px / 7.0) * math.cos(py / 9.0)
            vertices.append([px, py, pz])
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            faces.extend(([a, a + 1, a + size], [a + 1, a + size + 1, a + size]))
    return Structure(
        vertices=torch.tensor(vertices, dtype=torch.float32),
        faces=torch.tensor(faces, dtype=torch.int32),
        material=Dielectric(eps_r=5.0, sigma_e=0.02),
        name="generated-terrain",
        surface_id=1,
    )


def _receiver_grid(
    shape: tuple[int, int],
    *,
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    z: float,
) -> ReceiverGrid:
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape entries must be positive")
    spacing_x = (bounds_x[1] - bounds_x[0]) / rows
    spacing_y = (bounds_y[1] - bounds_y[0]) / cols
    return ReceiverGrid(
        origin=torch.tensor(
            [bounds_x[0] + 0.5 * spacing_x, bounds_y[0] + 0.5 * spacing_y, z],
            dtype=torch.float32,
        ),
        x_axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        y_axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        shape=shape,
        spacing=(spacing_x, spacing_y),
    )


def _terrain_scene(*, tx_count: int, grid_shape: tuple[int, int]) -> Scene:
    transmitters = [
        Transmitter(
            position=torch.tensor([-15.0, -10.0 + 0.5 * index, 8.0], dtype=torch.float32)
        )
        for index in range(tx_count)
    ]
    return Scene(
        structures=[_terrain_mesh()],
        transmitters=transmitters,
        receivers=[
            _receiver_grid(
                grid_shape, bounds_x=(-18.0, 18.0), bounds_y=(-18.0, 18.0), z=1.5
            )
        ],
        frequency=3.5e9,
        metadata={"phase_e_scenario": "terrain"},
    )


def _asset_root(explicit: str | Path | None) -> Path:
    value = explicit if explicit is not None else os.environ.get(SCENE_ROOT_ENV)
    if value is None or not str(value).strip():
        raise FullScenarioAssetError(
            "full Munich/SF scenarios require an explicit asset_root or "
            f"the {SCENE_ROOT_ENV} environment variable; no reduced fallback is used"
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FullScenarioAssetError(f"Phase E scene asset root is not a directory: {root}")
    return root


def _resolve_city_xml(root: Path, city: str) -> Path:
    candidates = (
        root / "sionna" / "rt" / "scenes" / city / f"{city}.xml",
        root / city / f"{city}.xml",
        root / f"{city}.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FullScenarioAssetError(
        f"full {city} scene is unavailable under {root}; searched: {searched}. "
        "No reduced or synthetic city scene is substituted."
    )


def _referenced_asset_files(scene_xml: Path) -> tuple[Path, ...]:
    pending = [scene_xml.resolve()]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FullScenarioAssetError(f"full scene references a missing asset: {path}")
        visited.add(path)
        if path.suffix.lower() != ".xml":
            continue
        root = ET.parse(path).getroot()
        references = [
            node.attrib["value"]
            for node in root.findall(".//string[@name='filename']")
            if "value" in node.attrib
        ]
        references.extend(
            node.attrib["filename"]
            for node in root.findall(".//include")
            if "filename" in node.attrib
        )
        pending.extend((path.parent / value).resolve() for value in references)
    return tuple(sorted(visited, key=lambda item: str(item).lower()))


def _files_sha256(scene_xml: Path) -> str:
    digest = hashlib.sha256()
    for path in _referenced_asset_files(scene_xml):
        try:
            label = path.relative_to(scene_xml.parent).as_posix()
        except ValueError:
            label = path.name
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _city_scene(
    name: str,
    *,
    asset_root: str | Path | None,
    tx_count: int,
    grid_shape: tuple[int, int],
) -> tuple[Scene, str, Path]:
    spec = _scenario_spec(name)
    root = _asset_root(asset_root)
    scene_xml = _resolve_city_xml(root, str(spec["asset_name"]))
    base = Scene.load_mitsuba(
        scene_xml,
        source_root=root,
        merge_shapes=True,
        frequency=float(spec["frequency_hz"]),
        edge_selection_mode="all_edges",
        boundary_edge_policy="half_plane",
    )
    tx_origin = tuple(float(value) for value in spec["transmitter"])
    transmitters = [
        Transmitter(
            position=torch.tensor(
                [tx_origin[0], tx_origin[1] + 0.5 * index, tx_origin[2]],
                dtype=torch.float32,
            )
        )
        for index in range(tx_count)
    ]
    grid = _receiver_grid(
        grid_shape,
        bounds_x=tuple(float(value) for value in spec["bounds_x"]),
        bounds_y=tuple(float(value) for value in spec["bounds_y"]),
        z=float(spec["plane_z"]),
    )
    scene = Scene(
        structures=base.structures,
        transmitters=transmitters,
        receivers=[grid],
        frequency=base.frequency,
        metadata={**base.metadata, "phase_e_scenario": name, "scenario_mode": "full"},
    )
    return scene, _files_sha256(scene_xml), scene_xml


def _update_tensor(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())


def _generated_source_sha256(scene: Scene) -> str:
    digest = hashlib.sha256()
    digest.update(str(scene.frequency).encode("ascii"))
    for structure in scene.structures:
        digest.update((structure.name or "").encode("utf-8"))
        digest.update(str(structure.surface_id).encode("ascii"))
        _update_tensor(digest, structure.vertices)
        _update_tensor(digest, structure.faces)
        digest.update(
            json.dumps(
                structure.material.parameters(scene.frequency),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _scene_sha256(scene: Scene, source_sha256: str) -> str:
    digest = hashlib.sha256(source_sha256.encode("ascii"))
    for transmitter in scene.transmitters:
        _update_tensor(digest, transmitter.position)
    for receiver in scene.receivers:
        _update_tensor(digest, receiver.origin if isinstance(receiver, ReceiverGrid) else receiver.position)
        if isinstance(receiver, ReceiverGrid):
            _update_tensor(digest, receiver.x_axis)
            _update_tensor(digest, receiver.y_axis)
            digest.update(json.dumps([*receiver.shape, *receiver.spacing]).encode("ascii"))
    return digest.hexdigest()


def _receiver_scale(scene: Scene) -> tuple[int, int]:
    grid_cells = sum(
        int(receiver.shape[0]) * int(receiver.shape[1])
        for receiver in scene.receivers
        if isinstance(receiver, ReceiverGrid)
    )
    points = sum(isinstance(receiver, ReceiverPoint) for receiver in scene.receivers)
    return points + grid_cells, grid_cells


def build_scenario(
    name: str,
    *,
    asset_root: str | Path | None = None,
    tx_count: int | None = None,
    receiver_count: int | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> ScenarioBundle:
    """Build one versioned Phase E scenario without hidden reduced fallbacks."""

    spec = _scenario_spec(name)
    tx = int(spec["default_tx_count"] if tx_count is None else tx_count)
    if tx <= 0:
        raise ValueError("tx_count must be positive")
    mode = str(spec["mode"])
    source_path: Path | None = None
    if mode == "generated":
        if asset_root is not None:
            raise ValueError(f"asset_root is not accepted for generated scenario {name!r}")
        if name in {"analytic", "three_cube"}:
            if grid_shape is not None:
                raise ValueError(f"grid_shape is not accepted for point scenario {name!r}")
            rx = int(
                spec["default_receiver_count"]
                if receiver_count is None
                else receiver_count
            )
            scene = (
                _analytic_scene(tx_count=tx, receiver_count=rx)
                if name == "analytic"
                else _three_cube_scene(tx_count=tx, receiver_count=rx)
            )
        elif name == "terrain":
            if receiver_count is not None:
                raise ValueError("terrain uses a receiver grid; receiver_count is not accepted")
            shape = tuple(spec["default_grid_shape"]) if grid_shape is None else grid_shape
            scene = _terrain_scene(tx_count=tx, grid_shape=(int(shape[0]), int(shape[1])))
        else:
            raise AssertionError(f"unhandled generated scenario: {name}")
        source_sha256 = _generated_source_sha256(scene)
    elif mode == "full_external":
        if receiver_count is not None:
            raise ValueError(f"{name} uses a receiver grid; receiver_count is not accepted")
        shape = tuple(spec["default_grid_shape"]) if grid_shape is None else grid_shape
        scene, source_sha256, source_path = _city_scene(
            name,
            asset_root=asset_root,
            tx_count=tx,
            grid_shape=(int(shape[0]), int(shape[1])),
        )
    else:
        raise ValueError(f"unsupported Phase E scenario mode: {mode!r}")

    receiver_total, grid_cells = _receiver_scale(scene)
    triangles = sum(int(structure.faces.shape[0]) for structure in scene.structures)
    record = ScenarioRecord(
        name=name,
        mode=mode,
        source_sha256=source_sha256,
        scene_sha256=_scene_sha256(scene, source_sha256),
        triangle_count=triangles,
        transmitter_count=len(scene.transmitters),
        receiver_count=receiver_total,
        receiver_container_count=len(scene.receivers),
        receiver_grid_cells=grid_cells,
        source_path=str(source_path) if source_path is not None else None,
    )
    expected_triangles = spec.get("expected_triangle_count")
    if expected_triangles is not None and triangles != int(expected_triangles):
        raise RuntimeError(
            f"{name} triangle count drifted: expected {expected_triangles}, got {triangles}"
        )
    expected_source = spec.get("source_sha256")
    if expected_source is not None and source_sha256 != expected_source:
        raise RuntimeError(
            f"{name} generated source SHA drifted: expected {expected_source}, got {source_sha256}"
        )
    return ScenarioBundle(scene=scene, record=record)


__all__ = [
    "FullScenarioAssetError",
    "SCENARIO_MANIFEST_PATH",
    "SCENE_ROOT_ENV",
    "ScenarioBundle",
    "ScenarioRecord",
    "build_scenario",
    "load_manifest",
    "scenario_names",
]
