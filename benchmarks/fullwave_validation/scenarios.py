# Copyright Xingyu Chen.
# Benchmarks scenarios.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from witwin.core import (
    AntennaState,
    MaterialLayer,
    Mesh,
    PhysicalMaterial,
    ReceiverGrid,
    Scene,
    Structure,
)
from witwin.core.identity import (
    new_antenna_id,
    new_assignment_id,
    new_material_id,
    new_structure_id,
)

from .models import CaseSpec, MaterialSpec


MANIFEST_PATH = Path(__file__).parents[1] / "scenarios" / "fullwave_validation.v1.json"
SCENARIOS = ("single_cube", "three_cube", "three_cube_320")
MATERIALS = ("metal", "dielectric")


def load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_case(scenario: str, material: str) -> CaseSpec:
    manifest = load_manifest()
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if material not in MATERIALS:
        raise ValueError(f"material must be one of {MATERIALS}")
    raw_scenario = manifest["scenarios"][scenario]  # type: ignore[index]
    raw_material = manifest["materials"][material]  # type: ignore[index]
    frequency_hz = manifest["electromagnetic_scaling"]["frequency_hz"]  # type: ignore[index]
    return CaseSpec(
        scenario=scenario,
        material=MaterialSpec(
            kind=material,
            eps_r=float(raw_material["eps_r"]),
            sigma_e=float(raw_material["sigma_e"]),
        ),
        frequency_hz=float(frequency_hz),
        analysis_bounds_xy=tuple(
            tuple(row) for row in raw_scenario["analysis_bounds_xy"]
        ),
        domain_bounds_xyz=tuple(
            tuple(row) for row in raw_scenario["domain_bounds_xyz"]
        ),
        plane_z=float(raw_scenario["plane_z"]),
        tx_position=tuple(raw_scenario["tx_position"]),
        cube_centers=tuple(tuple(row) for row in raw_scenario["cube_centers"]),
        cube_size_m=float(raw_scenario["cube_size_m"]),
        receiver_shape=tuple(raw_scenario["receiver_shape"]),
        fullwave_dl_m=float(raw_scenario["fullwave_dl_m"]),
        fullwave_pml_layers=int(raw_scenario["fullwave_pml_layers"]),
        max_depth=int(raw_scenario["max_depth"]),
        source_layout=str(raw_scenario["source_layout"]),
    )


def _cube_mesh(
    center: tuple[float, float, float], size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cx, cy, cz = center
    half = size / 2.0
    vertices = torch.tensor(
        [
            [cx - half, cy - half, cz - half],
            [cx + half, cy - half, cz - half],
            [cx + half, cy + half, cz - half],
            [cx - half, cy + half, cz - half],
            [cx - half, cy - half, cz + half],
            [cx + half, cy - half, cz + half],
            [cx + half, cy + half, cz + half],
            [cx - half, cy + half, cz + half],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.int32,
    )
    return vertices, faces


def build_channel_scene(spec: CaseSpec) -> Scene:
    if spec.material.kind == "metal":
        material = PhysicalMaterial.perfect_conductor(name="pec")
    else:
        material = PhysicalMaterial(
            layers=(
                MaterialLayer(
                    thickness_m=spec.cube_size_m / 2.0,
                    eps_r=spec.material.eps_r,
                    sigma_e=spec.material.sigma_e,
                ),
            ),
            name="dielectric-volume-interface-approximation",
        )
    structures = []
    for index, center in enumerate(spec.cube_centers, start=1):
        vertices, faces = _cube_mesh(center, spec.cube_size_m)
        structures.append(
            Structure(
                Mesh(
                    vertices,
                    faces,
                    recenter=False,
                    fill_mode="solid",
                    topology_diagnostics=False,
                ),
                material,
                name=f"cube-{index}",
                structure_id=new_structure_id(),
                material_id=new_material_id(),
                assignment_id=new_assignment_id(),
                surface_id=index,
            )
        )

    x = spec.x
    y = spec.y
    return Scene(
        structures=structures,
        endpoints=[
            AntennaState(
                new_antenna_id(),
                "tx",
                torch.tensor(spec.tx_position, dtype=torch.float32),
                polarization=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
            ),
            ReceiverGrid(
                new_antenna_id(),
                origin=torch.tensor(
                    [x[0], y[0], spec.plane_z], dtype=torch.float32
                ),
                x_axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
                y_axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
                shape=(x.size, y.size),
                spacing=(float(x[1] - x[0]), float(y[1] - y[0])),
                polarization=torch.tensor(
                    [0.0, 0.0, 1.0], dtype=torch.float32
                ),
            )
        ],
        metadata={
            "fullwave_validation_case": spec.case_id,
            "fullwave_validation_fingerprint": spec.fingerprint,
            "reference_frequency_hz": spec.frequency_hz,
        },
    )


def observation_valid_mask(spec: CaseSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return valid observation samples, excluding PEC volume intersections."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be one-dimensional")

    valid = np.ones((y_values.size, x_values.size), dtype=bool)
    if spec.material.kind != "metal":
        return valid

    half_size = spec.cube_size_m / 2.0
    for center_x, center_y, center_z in spec.cube_centers:
        if not center_z - half_size <= spec.plane_z <= center_z + half_size:
            continue
        footprint = (
            (np.abs(x_values[None, :] - center_x) <= half_size)
            & (np.abs(y_values[:, None] - center_y) <= half_size)
        )
        valid &= ~footprint
    return valid