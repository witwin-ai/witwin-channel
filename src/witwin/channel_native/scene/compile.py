"""Canonical scene compilation pipeline."""

from __future__ import annotations

from typing import Any
import hashlib
import json

import torch

from witwin.channel_native.materials.models import (
    GEOMETRY_MODE_IDS,
    MATERIAL_ABI_VERSION,
    PhaseScreen,
    SurfaceAssignment,
    effective_sigma_e,
)
from witwin.channel_native.propagation.topology.kernels import (
    primitives as topology_primitives,
)
from witwin.channel_native.runtime.native_buffers import bdpt_zero_matrix
from witwin.channel_native.scene.compiled import CompiledScene
from witwin.channel_native.scene.kernels.rayd_scene import RayDSceneResource
from witwin.channel_native.scene.models import Structure
from witwin.channel_native.scene.stores.assignments import AssignmentStore
from witwin.channel_native.scene.stores.geometry import GeometryStore
from witwin.channel_native.scene.stores.materials import MaterialStore


def compile_scene(scene: Any) -> CompiledScene:
    cached = scene._compiled_cache
    # Material records are always evaluated at the primal frequency:
    # dispersive material laws are frozen at compile time (fixed material
    # records), so frequency gradients flow only through the explicit
    # frequency dependence of the field kernels (plan 07 AD-1). The AD
    # seam therefore refuses frequency AD when any record is frequency
    # dependent (see materials.frequency_dependent).
    frequency_value = (
        float(scene.frequency.detach())
        if isinstance(scene.frequency, torch.Tensor)
        else float(scene.frequency)
    )
    (
        material_records,
        material_keys,
        material_cache_token,
        phase_screens,
    ) = _material_records(scene.structures, frequency_value)
    if (
        cached is not None
        and cached.geometry_version == scene._geometry_version
        and cached.material_version == scene._material_version
        and cached.assignment_version == scene._assignment_version
        and cached.materials.cache_token == material_cache_token
    ):
        return cached
    rayd = scene.rayd_scene()
    geometry = _compile_geometry(scene.structures, scene._geometry_version, rayd=rayd)
    materials = _compile_materials(
        material_records,
        material_keys,
        frequency_value,
        scene._material_version,
        material_cache_token,
        frequency_dependent=_frequency_dependent_material_keys(
            scene.structures, material_records, material_keys, frequency_value
        ),
    )
    assignments = _compile_assignments(
        scene.structures,
        num_faces=geometry.faces.shape[0],
        num_edges=geometry.edges.shape[0],
        version=scene._assignment_version,
        phase_screens=phase_screens,
    )
    (
        enumerated_penetration_scene_diagonal_m,
        montecarlo_penetration_scene_diagonal_m,
    ) = _compile_penetration_scene_diagonals(scene.structures, rayd=rayd)
    compiled = CompiledScene(
        geometry=geometry,
        materials=materials,
        assignments=assignments,
        rayd=rayd,
        geometry_version=geometry.version,
        material_version=materials.version,
        assignment_version=assignments.version,
        enumerated_penetration_scene_diagonal_m=(
            enumerated_penetration_scene_diagonal_m
        ),
        montecarlo_penetration_scene_diagonal_m=(
            montecarlo_penetration_scene_diagonal_m
        ),
    )
    object.__setattr__(scene, "_compiled_cache", compiled)
    return compiled


def _compile_penetration_scene_diagonals(
    structures: tuple[Structure, ...], *, rayd: RayDSceneResource
) -> tuple[float, float]:
    """Freeze the two distinct ADR-027 scale baselines at compile time.

    The scalar device reads are scene-static compile work. Solves consume only
    these host values and never reduce scene geometry or synchronize to recover
    them.
    """

    if not structures:
        return 0.0, 0.0

    records = rayd.edge_records()
    vertices = records.vertices
    enumerated = float((vertices.max(dim=0).values - vertices.min(dim=0).values).norm())

    minimum: torch.Tensor | None = None
    maximum: torch.Tensor | None = None
    for structure in structures:
        frozen_vertices = structure.vertices.detach()
        low = frozen_vertices.amin(dim=0)
        high = frozen_vertices.amax(dim=0)
        if minimum is None:
            minimum = low
            maximum = high
            continue
        assert maximum is not None
        low = low.to(device=minimum.device)
        high = high.to(device=minimum.device)
        minimum = torch.minimum(minimum, low)
        maximum = torch.maximum(maximum, high)
    assert minimum is not None and maximum is not None
    montecarlo = float((maximum - minimum).norm())
    return enumerated, montecarlo


def _compile_geometry(
    structures: tuple[Structure, ...], version: int, *, rayd: RayDSceneResource
) -> GeometryStore:
    if not structures:
        empty_vertices = torch.empty((0, 3), dtype=torch.float32)
        empty_faces = torch.empty((0, 3), dtype=torch.int32)
        empty_edges = torch.empty((0, 2), dtype=torch.int32)
        return GeometryStore(
            vertices=empty_vertices,
            faces=empty_faces,
            face_normals=empty_vertices,
            edges=empty_edges,
            edge_adj_faces=empty_edges,
            edge_param_range=torch.empty((0, 2), dtype=torch.float32),
            face_structure_id=torch.empty((0,), dtype=torch.int32),
            face_surface_id=torch.empty((0,), dtype=torch.int32),
            structure_uv_presence=(),
            version=version,
        )

    if not rayd.available:
        rayd.require_resource()
    records = rayd.edge_records()
    face_structure_id = []
    face_surface_id = []
    for structure_id, structure in enumerate(structures):
        face_structure_id.extend([structure_id] * structure.faces.shape[0])
        face_surface_id.extend([structure.surface_id] * structure.faces.shape[0])

    return GeometryStore(
        vertices=records.vertices,
        faces=records.faces,
        face_normals=records.face_normals,
        edges=topology_primitives.core_pack_int2(records.edge_v0, records.edge_v1),
        edge_adj_faces=topology_primitives.core_pack_int2(records.face0, records.face1),
        edge_param_range=bdpt_zero_matrix(
            records.vertices, rows=records.edge_v0.shape[0], cols=2
        ),
        face_structure_id=torch.tensor(face_structure_id, dtype=torch.int32),
        face_surface_id=torch.tensor(face_surface_id, dtype=torch.int32),
        structure_uv_presence=tuple(
            (structure.uv is not None, structure.face_uv is not None)
            for structure in structures
        ),
        version=version,
    )


def _abi_v3_layer_view(
    params: dict[str, object],
) -> dict[str, object]:
    """Derive the ABI v3 CSR/roughness view for any compiled material record.

    Legacy scalar materials become a single layer from
    ``(eps_r, sigma_e, mu_r, thickness_m)``; PEC keeps its effective-sigma
    encoding so the finite-sigma layer kernels see the conductor limit.
    """

    if "layers" in params:
        layers = [[float(v) for v in row] for row in params["layers"]]
        roughness = params.get("roughness")
        geometry_mode_id = GEOMETRY_MODE_IDS[str(params["geometry_mode"])]
    else:
        layers = [
            [
                float(params["thickness_m"]),
                float(params["eps_r"]),
                effective_sigma_e(params),
                float(params["mu_r"]),
            ]
        ]
        roughness = None
        geometry_mode_id = GEOMETRY_MODE_IDS["thin_sheet"]
    if roughness is None:
        roughness = [0.0, 0.0, 0.0, 0.0]
    else:
        roughness = [float(v) for v in roughness]
    return {
        "layers": layers,
        "roughness": roughness,
        "geometry_mode_id": geometry_mode_id,
        "scatter_model_id": 1 if roughness[0] > 0.0 else 0,
    }


def _phase_screen_descriptor(screen: PhaseScreen) -> dict[str, object]:
    """JSON-serializable cache-token descriptor (excludes bulk height data)."""

    correlation = screen.correlation
    return {
        "shape": list(screen.shape()),
        "height_scale_m": float(screen.height_scale_m),
        "height_offset_m": float(screen.height_offset_m),
        "realization_id": int(screen.realization_id),
        "mode": str(screen.mode),
        "quadrature_tolerance": float(screen.quadrature_tolerance),
        "correlation": None
        if correlation is None
        else [
            float(correlation.rms_height_m),
            float(correlation.corr_length_x_m),
            float(correlation.corr_length_y_m),
            float(correlation.principal_axis_rad),
            str(correlation.correlation),
        ],
    }


def _material_records(
    structures: tuple[Structure, ...], frequency_hz: float
) -> tuple[
    list[dict[str, object]],
    tuple[str, ...],
    str,
    dict[int, PhaseScreen],
]:
    params: list[dict[str, object]] = []
    phase_screens: dict[int, PhaseScreen] = {}
    for index, structure in enumerate(structures):
        material = structure.material
        if isinstance(material, SurfaceAssignment):
            if material.phase_screen is not None:
                phase_screens[index] = material.phase_screen
            material = material.material
        params.append(dict(material.parameters(frequency_hz)))
    keys = tuple(
        f"{index}:{structure.name or 'structure'}:{params[index].get('name', 'material')}"
        for index, structure in enumerate(structures)
    )
    if not params:
        params = [
            {
                "eps_r": 1.0,
                "mu_r": 1.0,
                "sigma_e": 0.0,
                "gain": 1.0,
                "thickness_m": 0.1,
                "scattering_coefficient": 0.0,
                "xpd_coefficient": 0.0,
                "model_id": 1,
                "name": "vacuum",
            }
        ]
        keys = ("0:vacuum:vacuum",)
    for record in params:
        record.update(_abi_v3_layer_view(record))
    payload = {
        "abi_version": MATERIAL_ABI_VERSION,
        "frequency_hz": float(frequency_hz),
        "materials": params,
        "keys": keys,
        "phase_screens": {
            str(index): _phase_screen_descriptor(screen)
            for index, screen in phase_screens.items()
        },
    }
    cache_token = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return params, keys, cache_token, phase_screens


def _frequency_dependent_material_keys(
    structures: tuple[Structure, ...],
    records: list[dict[str, object]],
    keys: tuple[str, ...],
    frequency_hz: float,
) -> tuple[str, ...]:
    """Keys of material records whose law changes with the carrier frequency.

    Probe each material law at a nearby frequency and compare the compiled
    record payload; a record that cannot even be evaluated at the probe
    (e.g. a tabulated permittivity at its range edge) is conservatively
    frequency-dependent. Backs the plan 07 AD-1 explicit-failure check:
    frequency AD refuses scenes with frequency-dependent records because
    compile() freezes records at the primal frequency.
    """

    probe_hz = float(frequency_hz) * (1.0 + 1.0e-3)
    dependent: list[str] = []
    for index, structure in enumerate(structures):
        material = structure.material
        if isinstance(material, SurfaceAssignment):
            material = material.material
        try:
            probe = dict(material.parameters(probe_hz))
            probe.update(_abi_v3_layer_view(probe))
        except ValueError:
            dependent.append(keys[index])
            continue
        if probe != records[index]:
            dependent.append(keys[index])
    return tuple(dependent)


def _compile_materials(
    params: list[dict[str, object]],
    material_keys: tuple[str, ...],
    frequency_hz: float,
    version: int,
    cache_token: str,
    *,
    frequency_dependent: tuple[str, ...] = (),
) -> MaterialStore:
    layer_offset: list[int] = []
    layer_count: list[int] = []
    layer_rows: list[list[float]] = []
    for p in params:
        rows = p["layers"]
        layer_offset.append(len(layer_rows))
        layer_count.append(len(rows))
        layer_rows.extend(rows)

    def layer_column(column: int) -> torch.Tensor:
        return torch.tensor([row[column] for row in layer_rows], dtype=torch.float32)

    def rough_column(column: int) -> torch.Tensor:
        return torch.tensor(
            [float(p["roughness"][column]) for p in params], dtype=torch.float32
        )

    return MaterialStore(
        material_id=torch.arange(len(params), dtype=torch.int32),
        eps_r=torch.tensor([float(p["eps_r"]) for p in params], dtype=torch.float32),
        mu_r=torch.tensor([float(p["mu_r"]) for p in params], dtype=torch.float32),
        sigma_e=torch.tensor(
            [float(p["sigma_e"]) for p in params], dtype=torch.float32
        ),
        gain=torch.tensor([float(p["gain"]) for p in params], dtype=torch.float32),
        model_id=torch.tensor([int(p["model_id"]) for p in params], dtype=torch.int32),
        thickness_m=torch.tensor(
            [float(p["thickness_m"]) for p in params], dtype=torch.float32
        ),
        scattering_coefficient=torch.tensor(
            [float(p["scattering_coefficient"]) for p in params], dtype=torch.float32
        ),
        xpd_coefficient=torch.tensor(
            [float(p["xpd_coefficient"]) for p in params], dtype=torch.float32
        ),
        layer_offset=torch.tensor(layer_offset, dtype=torch.int32),
        layer_count=torch.tensor(layer_count, dtype=torch.int32),
        layer_thickness_m=layer_column(0),
        layer_eps_r=layer_column(1),
        layer_sigma_e=layer_column(2),
        layer_mu_r=layer_column(3),
        rough_sigma_h_m=rough_column(0),
        rough_corr_x_m=rough_column(1),
        rough_corr_y_m=rough_column(2),
        rough_axis_rad=rough_column(3),
        geometry_mode_id=torch.tensor(
            [int(p["geometry_mode_id"]) for p in params], dtype=torch.int32
        ),
        scatter_model_id=torch.tensor(
            [int(p["scatter_model_id"]) for p in params], dtype=torch.int32
        ),
        material_keys=material_keys,
        frequency_hz=frequency_hz,
        abi_version=MATERIAL_ABI_VERSION,
        cache_token=cache_token,
        version=version,
        frequency_dependent=frequency_dependent,
    )


def _compile_assignments(
    structures: tuple[Structure, ...],
    *,
    num_faces: int,
    num_edges: int,
    version: int,
    phase_screens: dict[int, PhaseScreen] | None = None,
) -> AssignmentStore:
    face_material_ids = []
    for material_id, structure in enumerate(structures):
        face_material_ids.extend([material_id] * structure.faces.shape[0])
    return AssignmentStore(
        face_material_id=torch.tensor(face_material_ids, dtype=torch.int32),
        edge_material_id0=torch.tensor([0] * num_edges, dtype=torch.int32),
        edge_material_id1=torch.tensor([0] * num_edges, dtype=torch.int32),
        surface_material_id=torch.tensor(
            list(range(len(structures))), dtype=torch.int32
        ),
        structure_material_id=torch.tensor(
            list(range(len(structures))), dtype=torch.int32
        ),
        num_faces=num_faces,
        num_edges=num_edges,
        version=version,
        structure_phase_screens=dict(phase_screens or {}),
    )
