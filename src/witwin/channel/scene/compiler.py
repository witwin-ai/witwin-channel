"""Compile canonical Core world contracts into Channel runtime resources."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
from threading import RLock

import torch

from witwin.core import (
    MaterialLayer,
    Mesh,
    PhaseScreen,
    PhysicalMaterial,
    Scene,
    SceneSnapshot,
    Structure,
    quat_from_euler,
    quat_to_rotation_matrix,
)
from witwin.core.material import VACUUM_PERMITTIVITY

from witwin.channel.materials.abi import (
    DIELECTRIC_MODEL_ID,
    MATERIAL_ABI_VERSION,
    PEC_EFFECTIVE_SIGMA_E,
    PEC_MODEL_ID,
)
from witwin.channel.propagation.topology.kernels import (
    primitives as topology_primitives,
)
from witwin.channel.runtime.native_buffers import bdpt_zero_matrix
from witwin.channel.scene.compiled import CompiledScene
from witwin.channel.scene.kernels.rayd_scene import (
    RayDSceneResource,
    build_scene_from_structures,
)
from witwin.channel.scene.stores.assignments import AssignmentStore
from witwin.channel.scene.stores.geometry import GeometryStore
from witwin.channel.scene.stores.materials import MaterialStore


_GEOMETRY_MODE_IDS = {"thin_sheet": 0, "volumetric": 1}
_SCATTER_MODEL_SMOOTH = 0
_SCATTER_MODEL_KIRCHHOFF = 1
_CACHE_LIMIT = 32


@dataclass(frozen=True, slots=True)
class _RuntimeStructure:
    source: Structure
    vertices: torch.Tensor
    faces: torch.Tensor
    uv: torch.Tensor | None
    face_uv: torch.Tensor | None

    @property
    def name(self) -> str:
        return self.source.name or ""

    @property
    def material(self):
        return self.source.material

    @property
    def phase_screen(self) -> PhaseScreen | None:
        return self.source.phase_screen

    @property
    def structure_id(self) -> int:
        assert self.source.structure_id is not None
        return int(self.source.structure_id)

    @property
    def assignment_id(self) -> int:
        assert self.source.assignment_id is not None
        return int(self.source.assignment_id)

    @property
    def material_id(self) -> int:
        assert self.source.material_id is not None
        return int(self.source.material_id)

    def primitive_id(self, local_index: int) -> int:
        return int(self.source.primitive_id(local_index))


class _CompileRegistry:
    """Small Channel-owned LRU keyed only by logical versions/runtime identity."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._compiled: OrderedDict[tuple[object, ...], CompiledScene] = OrderedDict()

    def get(self, key: tuple[object, ...]) -> CompiledScene | None:
        with self._lock:
            value = self._compiled.get(key)
            if value is not None:
                self._compiled.move_to_end(key)
            return value

    def candidates(self) -> tuple[CompiledScene, ...]:
        with self._lock:
            return tuple(reversed(self._compiled.values()))

    def put(self, key: tuple[object, ...], compiled: CompiledScene) -> None:
        with self._lock:
            self._compiled[key] = compiled
            self._compiled.move_to_end(key)
            while len(self._compiled) > _CACHE_LIMIT:
                self._compiled.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._compiled.clear()


_REGISTRY = _CompileRegistry()


def clear_compile_cache() -> None:
    """Clear the bounded Channel runtime registry (primarily for tests)."""

    _REGISTRY.clear()


def _frequency_value(reference_frequency_hz: float | torch.Tensor) -> float:
    if isinstance(reference_frequency_hz, torch.Tensor):
        if (
            reference_frequency_hz.ndim != 0
            or not reference_frequency_hz.dtype.is_floating_point
        ):
            raise TypeError(
                "reference_frequency_hz must be a scalar floating-point tensor"
            )
        value = float(reference_frequency_hz.detach())
    else:
        value = float(reference_frequency_hz)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("reference_frequency_hz must be finite and positive")
    return value


def _frequency_identity(
    reference_frequency_hz: float | torch.Tensor,
    value: float,
) -> tuple[object, ...]:
    if isinstance(reference_frequency_hz, torch.Tensor):
        return (
            "tensor",
            id(reference_frequency_hz),
            int(reference_frequency_hz._version),
            reference_frequency_hz.device,
            reference_frequency_hz.dtype,
            value.hex(),
        )
    return ("float", value.hex())


def _versions(
    scene_or_snapshot: Scene | SceneSnapshot,
) -> tuple[int, int, int, int]:
    return (
        int(scene_or_snapshot.topology_version),
        int(scene_or_snapshot.geometry_version),
        int(scene_or_snapshot.material_version),
        int(scene_or_snapshot.assignment_version),
    )


def _source_structures(
    scene_or_snapshot: Scene | SceneSnapshot,
) -> tuple[tuple[Structure, object | None, object | None], ...]:
    if isinstance(scene_or_snapshot, Scene):
        return tuple((structure, None, None) for structure in scene_or_snapshot.structures)
    return tuple(
        (state.structure, state.rigid_motion, state.deformation)
        for state in scene_or_snapshot.structures
    )


def _runtime_structures(
    scene_or_snapshot: Scene | SceneSnapshot,
) -> tuple[_RuntimeStructure, ...]:
    runtime: list[_RuntimeStructure] = []
    for structure, rigid_motion, deformation in _source_structures(scene_or_snapshot):
        if not structure.enabled:
            continue
        geometry = structure.geometry
        vertices, faces = geometry.to_mesh()
        if not isinstance(vertices, torch.Tensor) or not isinstance(faces, torch.Tensor):
            raise TypeError("Core GeometrySpec.to_mesh() must return torch tensors")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Core mesh vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Core mesh faces must have shape (F, 3)")
        if deformation is not None:
            if not isinstance(geometry, Mesh):
                raise TypeError(
                    "Channel snapshot deformation currently requires "
                    "witwin.core.Mesh geometry"
                )
            local_vertices = (
                deformation.vertices
                if deformation.vertices is not None
                else geometry.vertices + deformation.offsets
            )
            if local_vertices.shape != geometry.vertices.shape:
                raise ValueError(
                    "snapshot deformation must match the authored local mesh"
                )
            if geometry.recenter:
                bounds_min = local_vertices.amin(dim=0)
                bounds_max = local_vertices.amax(dim=0)
                local_vertices = local_vertices - 0.5 * (bounds_min + bounds_max)
            local_vertices = local_vertices * geometry.scale.to(
                device=local_vertices.device, dtype=local_vertices.dtype
            )
            authored_rotation_value = geometry.rotation.to(
                device=local_vertices.device,
                dtype=local_vertices.dtype,
            )
            authored_rotation = quat_to_rotation_matrix(
                authored_rotation_value
                if authored_rotation_value.shape == (4,)
                else quat_from_euler(
                    authored_rotation_value[0],
                    authored_rotation_value[1],
                    authored_rotation_value[2],
                    device=authored_rotation_value.device,
                    dtype=authored_rotation_value.dtype,
                )
            )
            vertices = (
                local_vertices @ authored_rotation.T
                + geometry.position.to(
                    device=local_vertices.device,
                    dtype=local_vertices.dtype,
                )
            )
        if rigid_motion is not None:
            if rigid_motion.rotation is not None:
                rigid_rotation_value = rigid_motion.rotation
                if not isinstance(rigid_rotation_value, torch.Tensor):
                    rigid_rotation_value = torch.as_tensor(
                        rigid_rotation_value,
                        device=vertices.device,
                        dtype=vertices.dtype,
                    )
                rotation = quat_to_rotation_matrix(
                    rigid_rotation_value
                    if rigid_rotation_value.shape == (4,)
                    else quat_from_euler(
                        rigid_rotation_value[2],
                        rigid_rotation_value[1],
                        rigid_rotation_value[0],
                        device=rigid_rotation_value.device,
                        dtype=rigid_rotation_value.dtype,
                    )
                )
                vertices = vertices @ rotation.T
            if rigid_motion.translation is not None:
                vertices = vertices + rigid_motion.translation
        if (
            structure.primitive_ids is not None
            and len(structure.primitive_ids) != int(faces.shape[0])
        ):
            raise ValueError(
                "Core Structure.primitive_ids must contain one ID per mesh face"
            )
        uv = structure.uv
        face_uv = structure.face_uv
        if (uv is None) != (face_uv is None):
            raise ValueError("Core mesh uv and face_uv must be provided together")
        runtime.append(
            _RuntimeStructure(
                source=structure,
                vertices=vertices.to(dtype=torch.float32).contiguous(),
                faces=faces.to(dtype=torch.int32).contiguous(),
                uv=(
                    None
                    if uv is None
                    else uv.to(dtype=torch.float32).contiguous()
                ),
                face_uv=(
                    None
                    if face_uv is None
                    else face_uv.to(dtype=torch.int32).contiguous()
                ),
            )
        )
    return tuple(runtime)


def _compile_penetration_scene_diagonals(
    structures: tuple[_RuntimeStructure, ...],
    *,
    rayd: RayDSceneResource,
) -> tuple[float, float]:
    if not structures:
        return 0.0, 0.0
    records = rayd.edge_records()
    vertices = records.vertices
    enumerated = float(
        (vertices.max(dim=0).values - vertices.min(dim=0).values).norm().detach()
    )
    authored = torch.cat(
        tuple(
            structure.vertices.to(device=vertices.device, dtype=vertices.dtype)
            for structure in structures
        ),
        dim=0,
    )
    montecarlo = float(
        (authored.amax(dim=0) - authored.amin(dim=0)).norm().detach()
    )
    return enumerated, montecarlo


def _compile_geometry(
    structures: tuple[_RuntimeStructure, ...],
    version: int,
    *,
    rayd: RayDSceneResource,
) -> GeometryStore:
    if not structures:
        empty_vertices = torch.empty((0, 3), dtype=torch.float32)
        empty_faces = torch.empty((0, 3), dtype=torch.int32)
        empty_edges = torch.empty((0, 2), dtype=torch.int32)
        empty_ids = torch.empty((0,), dtype=torch.int64)
        return GeometryStore(
            vertices=empty_vertices,
            faces=empty_faces,
            face_normals=empty_vertices,
            edges=empty_edges,
            edge_adj_faces=empty_edges,
            edge_param_range=torch.empty((0, 2), dtype=torch.float32),
            face_structure_id=empty_ids,
            face_surface_id=empty_ids,
            face_primitive_id=empty_ids,
            structure_uv_presence=(),
            version=version,
        )

    records = rayd.edge_records()
    structure_ids: list[int] = []
    primitive_ids: list[int] = []
    for structure in structures:
        face_count = int(structure.faces.shape[0])
        structure_ids.extend((structure.structure_id,) * face_count)
        primitive_ids.extend(
            structure.primitive_id(index) for index in range(face_count)
        )
    stable_primitives = torch.tensor(primitive_ids, dtype=torch.int64)
    surface_ids = [
        int(structure.source.surface_id)
        for structure in structures
        for _ in range(int(structure.faces.shape[0]))
    ]
    return GeometryStore(
        vertices=records.vertices,
        faces=records.faces,
        face_normals=records.face_normals,
        edges=topology_primitives.core_pack_int2(records.edge_v0, records.edge_v1),
        edge_adj_faces=topology_primitives.core_pack_int2(records.face0, records.face1),
        edge_param_range=bdpt_zero_matrix(
            records.vertices, rows=records.edge_v0.shape[0], cols=2
        ),
        face_structure_id=torch.tensor(structure_ids, dtype=torch.int64),
        face_surface_id=torch.tensor(surface_ids, dtype=torch.int64),
        face_primitive_id=stable_primitives,
        structure_uv_presence=tuple(
            (structure.uv is not None, structure.face_uv is not None)
            for structure in structures
        ),
        version=version,
    )


def _real(value):
    if isinstance(value, complex):
        return value.real
    if isinstance(value, torch.Tensor) and value.is_complex():
        return value.real
    return value


def _sigma_from_eps(value, frequency_hz: float):
    if isinstance(value, complex):
        imaginary = value.imag
    elif isinstance(value, torch.Tensor) and value.is_complex():
        imaginary = value.imag
    else:
        imaginary = 0.0
    return -imaginary * (
        2.0 * math.pi * frequency_hz * VACUUM_PERMITTIVITY
    )


def _material_record(
    material: PhysicalMaterial,
    frequency_hz: float,
) -> dict[str, object]:
    if material.roughness_back is not None and (
        material.roughness_front is None
        or material.roughness_back is not material.roughness_front
    ):
        raise RuntimeError(
            "Channel ABI v3 cannot encode distinct back-side roughness"
        )
    model_id = (
        PEC_MODEL_ID
        if getattr(material, "conductor_model", None) == "perfect"
        else DIELECTRIC_MODEL_ID
    )
    layers = material.layers or (
        MaterialLayer(
            thickness_m=material.thickness_m,
            eps_r=material.eps_r,
            sigma_e=material.sigma_e,
            mu_r=material.mu_r,
            dispersion=material.dispersion,
        ),
    )
    layer_rows: list[tuple[object, object, object, object]] = []
    for layer in layers:
        layer_sample = layer.evaluate_at_frequency(frequency_hz)
        layer_rows.append(
            (
                layer.thickness_m,
                _real(layer_sample.eps_r),
                (
                    PEC_EFFECTIVE_SIGMA_E
                    if model_id == PEC_MODEL_ID
                    else _sigma_from_eps(layer_sample.eps_r, frequency_hz)
                ),
                layer_sample.mu_r,
            )
        )
    first_layer = layer_rows[0]
    roughness = material.roughness_front
    roughness_row: tuple[object, object, object, object]
    if roughness is None:
        roughness_row = (0.0, 0.0, 0.0, 0.0)
    else:
        roughness_row = (
            roughness.rms_height_m,
            roughness.correlation_length_x_m,
            roughness.correlation_length_y_m,
            roughness.principal_axis_rad,
        )
    return {
        # ABI v3 scalar Fresnel view is the first physical layer. The complete
        # stack remains in the CSR payload below.
        "eps_r": first_layer[1],
        "mu_r": first_layer[3],
        "sigma_e": first_layer[2],
        "gain": material.gain,
        "model_id": model_id,
        "thickness_m": first_layer[0],
        "scattering_coefficient": material.scattering_coefficient,
        "xpd_coefficient": material.xpd_coefficient,
        "layers": tuple(layer_rows),
        "roughness": roughness_row,
        "geometry_mode_id": _GEOMETRY_MODE_IDS[material.geometry_mode],
        "scatter_model_id": (
            _SCATTER_MODEL_KIRCHHOFF
            if roughness is not None
            else _SCATTER_MODEL_SMOOTH
        ),
    }


def _unique_materials(
    structures: tuple[_RuntimeStructure, ...],
) -> tuple[tuple[int, PhysicalMaterial], ...]:
    by_id: dict[int, PhysicalMaterial] = {}
    for structure in structures:
        material = structure.material
        if not isinstance(material, PhysicalMaterial):
            raise TypeError(
                "Channel compilation requires witwin.core.PhysicalMaterial"
            )
        existing = by_id.get(structure.material_id)
        if existing is not None and existing is not material:
            raise ValueError("one Core MaterialId must identify one material object")
        by_id[structure.material_id] = material
    return tuple((key, by_id[key]) for key in sorted(by_id))


def _material_cache_token(
    material_version: int,
    material_ids: tuple[int, ...],
    frequency_hz: float,
) -> str:
    payload = repr(
        (MATERIAL_ABI_VERSION, material_version, material_ids, frequency_hz.hex())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stack(values: list[object], *, dtype: torch.dtype) -> torch.Tensor:
    tensor_values = [value for value in values if isinstance(value, torch.Tensor)]
    if tensor_values:
        device = tensor_values[0].device
        if any(value.device != device for value in tensor_values[1:]):
            raise ValueError("compiled material tensor leaves must share one device")
        rows = [
            (
                value.to(dtype=dtype)
                if isinstance(value, torch.Tensor)
                else torch.tensor(value, dtype=dtype, device=device)
            )
            for value in values
        ]
        return torch.stack(rows).contiguous()
    return torch.tensor(values, dtype=dtype)


def _compile_materials(
    materials: tuple[tuple[int, PhysicalMaterial], ...],
    frequency_hz: float,
    version: int,
) -> MaterialStore:
    if not materials:
        materials = ((0, PhysicalMaterial(material_id=0, name="vacuum")),)
    material_ids = tuple(material_id for material_id, _ in materials)
    material_specs = tuple(material for _, material in materials)
    records = tuple(
        _material_record(material, frequency_hz) for material in material_specs
    )
    layer_rows = [row for record in records for row in record["layers"]]
    offsets: list[int] = []
    counts: list[int] = []
    offset = 0
    for record in records:
        rows = record["layers"]
        offsets.append(offset)
        counts.append(len(rows))
        offset += len(rows)

    def column(name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return _stack([record[name] for record in records], dtype=dtype)

    def layer_column(index: int) -> torch.Tensor:
        return _stack([row[index] for row in layer_rows], dtype=torch.float32)

    def rough_column(index: int) -> torch.Tensor:
        return _stack(
            [record["roughness"][index] for record in records],
            dtype=torch.float32,
        )

    return MaterialStore(
        material_id=torch.tensor(material_ids, dtype=torch.int64),
        eps_r=column("eps_r"),
        mu_r=column("mu_r"),
        sigma_e=column("sigma_e"),
        gain=column("gain"),
        model_id=column("model_id", torch.int32),
        thickness_m=column("thickness_m"),
        scattering_coefficient=column("scattering_coefficient"),
        xpd_coefficient=column("xpd_coefficient"),
        layer_offset=torch.tensor(offsets, dtype=torch.int32),
        layer_count=torch.tensor(counts, dtype=torch.int32),
        layer_thickness_m=layer_column(0),
        layer_eps_r=layer_column(1),
        layer_sigma_e=layer_column(2),
        layer_mu_r=layer_column(3),
        rough_sigma_h_m=rough_column(0),
        rough_corr_x_m=rough_column(1),
        rough_corr_y_m=rough_column(2),
        rough_axis_rad=rough_column(3),
        geometry_mode_id=column("geometry_mode_id", torch.int32),
        scatter_model_id=column("scatter_model_id", torch.int32),
        material_keys=tuple(
            f"material:{material_id}:{material.name or 'unnamed'}"
            for material_id, material in zip(material_ids, material_specs)
        ),
        frequency_hz=frequency_hz,
        abi_version=MATERIAL_ABI_VERSION,
        cache_token=_material_cache_token(version, material_ids, frequency_hz),
        version=version,
        frequency_dependent=tuple(
            f"material:{material_id}:{material.name or 'unnamed'}"
            for material_id, material in zip(material_ids, material_specs)
            if material.capabilities().dispersive
        ),
    )


def _compile_assignments(
    structures: tuple[_RuntimeStructure, ...],
    *,
    material_row_by_id: dict[int, int],
    geometry: GeometryStore,
    version: int,
) -> AssignmentStore:
    face_rows: list[int] = []
    phase_screens: dict[int, PhaseScreen] = {}
    for structure_index, structure in enumerate(structures):
        row = material_row_by_id[structure.material_id]
        face_rows.extend((row,) * int(structure.faces.shape[0]))
        if structure.phase_screen is not None:
            phase_screens[structure_index] = structure.phase_screen
    structure_rows = [
        material_row_by_id[structure.material_id] for structure in structures
    ]
    face_rows_tensor = torch.tensor(face_rows, dtype=torch.int32)
    adjacency = geometry.edge_adj_faces
    edge_face_rows = face_rows_tensor.to(device=adjacency.device)
    if adjacency.shape[0]:
        face0 = adjacency[:, 0].to(dtype=torch.int64)
        face1 = adjacency[:, 1].to(dtype=torch.int64)
        edge_material_id0 = edge_face_rows.index_select(0, face0.clamp_min(0))
        edge_material_id1 = torch.where(
            face1 >= 0,
            edge_face_rows.index_select(0, face1.clamp_min(0)),
            edge_material_id0,
        )
    else:
        edge_material_id0 = torch.empty(
            (0,), dtype=torch.int32, device=adjacency.device
        )
        edge_material_id1 = torch.empty(
            (0,), dtype=torch.int32, device=adjacency.device
        )
    return AssignmentStore(
        assignment_id=torch.tensor(
            [structure.assignment_id for structure in structures],
            dtype=torch.int64,
        ),
        material_id=torch.tensor(
            [structure.material_id for structure in structures],
            dtype=torch.int64,
        ),
        structure_id=torch.tensor(
            [structure.structure_id for structure in structures],
            dtype=torch.int64,
        ),
        surface_id=torch.tensor(
            [int(structure.source.surface_id) for structure in structures],
            dtype=torch.int64,
        ),
        face_material_id=face_rows_tensor,
        edge_material_id0=edge_material_id0,
        edge_material_id1=edge_material_id1,
        surface_material_id=torch.tensor(structure_rows, dtype=torch.int32),
        structure_material_id=torch.tensor(structure_rows, dtype=torch.int32),
        num_faces=int(geometry.faces.shape[0]),
        num_edges=int(geometry.edges.shape[0]),
        version=version,
        structure_phase_screens=phase_screens,
    )


def _cache_key(
    scene_or_snapshot: Scene | SceneSnapshot,
    versions: tuple[int, int, int, int],
    frequency_identity: tuple[object, ...],
) -> tuple[object, ...]:
    runtime_device = (
        torch.cuda.current_device() if torch.cuda.is_available() else "no-cuda"
    )
    return (
        id(scene_or_snapshot),
        *versions,
        frequency_identity,
        runtime_device,
        MATERIAL_ABI_VERSION,
    )


def _rayd_input_identity(
    scene_or_snapshot: Scene | SceneSnapshot,
) -> tuple[object, ...]:
    edge_policy = scene_or_snapshot.metadata.get("sionna_import_edge_policy")
    states = tuple(
        item
        for item in _source_structures(scene_or_snapshot)
        if item[0].enabled
    )
    if isinstance(scene_or_snapshot, SceneSnapshot) and any(
        motion is not None or deformation is not None
        for _, motion, deformation in states
    ):
        source_scene = getattr(scene_or_snapshot, "_source_scene", None)
        if source_scene is not None:
            return ("dynamic-source", id(source_scene), edge_policy)
    return (
        edge_policy,
        *(
            (
                id(structure.geometry),
                id(structure.uv),
                id(structure.face_uv),
                id(rigid_motion),
                id(deformation),
            )
            for structure, rigid_motion, deformation in states
        )
    )


def _geometry_mapping_identity(
    scene_or_snapshot: Scene | SceneSnapshot,
) -> tuple[object, ...]:
    return (
        _rayd_input_identity(scene_or_snapshot),
        tuple(
            (
                int(structure.structure_id),
                int(structure.surface_id),
                None
                if structure.primitive_ids is None
                else tuple(int(value) for value in structure.primitive_ids),
            )
            for structure, _, _ in _source_structures(scene_or_snapshot)
            if structure.enabled
        ),
    )


def _material_input_identity(
    scene_or_snapshot: Scene | SceneSnapshot,
) -> tuple[tuple[int, int], ...]:
    materials = {
        int(structure.material_id): structure.material
        for structure, _, _ in _source_structures(scene_or_snapshot)
        if structure.enabled
    }
    return tuple(
        (material_id, id(material))
        for material_id, material in sorted(materials.items())
    )


def _assignment_input_identity(
    scene_or_snapshot: Scene | SceneSnapshot,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            int(structure.structure_id),
            int(structure.surface_id),
            int(structure.assignment_id),
            int(structure.material_id),
            id(structure.phase_screen),
        )
        for structure, _, _ in _source_structures(scene_or_snapshot)
        if structure.enabled
    )


def compile(
    scene_or_snapshot: Scene | SceneSnapshot,
    *,
    reference_frequency_hz: float | torch.Tensor,
) -> CompiledScene:
    """Build or reuse the sole Channel runtime for a Core world contract."""

    if not isinstance(scene_or_snapshot, (Scene, SceneSnapshot)):
        raise TypeError("scene_or_snapshot must be a witwin.core Scene or SceneSnapshot")
    frequency_value = _frequency_value(reference_frequency_hz)
    versions = _versions(scene_or_snapshot)
    key = _cache_key(
        scene_or_snapshot,
        versions,
        _frequency_identity(reference_frequency_hz, frequency_value),
    )
    cached = _REGISTRY.get(key)
    if cached is not None and cached.source is scene_or_snapshot:
        return cached

    structures = _runtime_structures(scene_or_snapshot)
    topology_version, geometry_version, material_version, assignment_version = versions
    reusable = _REGISTRY.candidates()
    rayd_input_identity = _rayd_input_identity(scene_or_snapshot)
    rayd = next(
        (
            candidate.rayd
            for candidate in reusable
            if _rayd_input_identity(candidate.source) == rayd_input_identity
            and candidate.topology_version == topology_version
            and candidate.geometry_version == geometry_version
        ),
        None,
    )
    if rayd is None:
        rayd = build_scene_from_structures(structures)
        from witwin.channel.scene.edge_selection import resolve_scene_edge_policy

        rayd.runtime_cache["edge_policy"] = resolve_scene_edge_policy(
            scene_or_snapshot
        )
    geometry_mapping_identity = _geometry_mapping_identity(scene_or_snapshot)
    geometry = next(
        (
            candidate.geometry
            for candidate in reusable
            if candidate.rayd is rayd
            and _geometry_mapping_identity(candidate.source)
            == geometry_mapping_identity
            and candidate.topology_version == topology_version
            and candidate.geometry_version == geometry_version
        ),
        None,
    )
    if geometry is None:
        geometry = _compile_geometry(structures, geometry_version, rayd=rayd)

    logical_materials = _unique_materials(structures)
    material_input_identity = _material_input_identity(scene_or_snapshot)
    materials = next(
        (
            candidate.materials
            for candidate in reusable
            if _material_input_identity(candidate.source)
            == material_input_identity
            and candidate.material_version == material_version
            and candidate.materials.frequency_hz == frequency_value
        ),
        None,
    )
    if materials is None:
        materials = _compile_materials(
            logical_materials, frequency_value, material_version
        )
    material_row_by_id = {
        int(material_id): row
        for row, material_id in enumerate(materials.material_id.tolist())
    }
    assignment_input_identity = _assignment_input_identity(scene_or_snapshot)
    assignments = next(
        (
            candidate.assignments
            for candidate in reusable
            if candidate.geometry is geometry
            and _assignment_input_identity(candidate.source)
            == assignment_input_identity
            and candidate.topology_version == topology_version
            and candidate.assignment_version == assignment_version
        ),
        None,
    )
    if assignments is None:
        assignments = _compile_assignments(
            structures,
            material_row_by_id=material_row_by_id,
            geometry=geometry,
            version=assignment_version,
        )
    diagonals = next(
        (
            (
                candidate.enumerated_penetration_scene_diagonal_m,
                candidate.montecarlo_penetration_scene_diagonal_m,
            )
            for candidate in reusable
            if candidate.rayd is rayd
            and candidate.topology_version == topology_version
            and candidate.geometry_version == geometry_version
        ),
        None,
    )
    if diagonals is None:
        diagonals = _compile_penetration_scene_diagonals(structures, rayd=rayd)

    compiled = CompiledScene(
        source=scene_or_snapshot,
        structures=structures,
        geometry=geometry,
        materials=materials,
        assignments=assignments,
        rayd=rayd,
        reference_frequency_hz=(
            reference_frequency_hz
            if isinstance(reference_frequency_hz, torch.Tensor)
            else frequency_value
        ),
        reference_frequency_revision=(
            int(reference_frequency_hz._version)
            if isinstance(reference_frequency_hz, torch.Tensor)
            else None
        ),
        topology_version=topology_version,
        geometry_version=geometry_version,
        material_version=material_version,
        assignment_version=assignment_version,
        enumerated_penetration_scene_diagonal_m=diagonals[0],
        montecarlo_penetration_scene_diagonal_m=diagonals[1],
    )
    _REGISTRY.put(key, compiled)
    return compiled


__all__ = ["clear_compile_cache", "compile"]
