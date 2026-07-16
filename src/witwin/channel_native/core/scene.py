from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json

import torch

from .objects import ReceiverGrid, ReceiverPoint, Structure, Transmitter
from witwin.channel_native.scene.compiled import CompiledScene
from witwin.channel_native.scene.stores.assignments import AssignmentStore
from witwin.channel_native.scene.stores.geometry import GeometryStore
from witwin.channel_native.scene.stores.materials import MaterialStore
from witwin.channel_native.scene.kernels.rayd_scene import (
    RayDNScene,
    build_scene_from_structures,
)
from .edge_policy import DEFAULT_EDGE_POLICY, EdgePolicy
from .edge_selection import resolve_scene_edge_policy
from witwin.channel_native.runtime.native_buffers import bdpt_zero_matrix
from witwin.channel_native.materials.models import (
    GEOMETRY_MODE_IDS,
    MATERIAL_ABI_VERSION,
    PhaseScreen,
    SurfaceAssignment,
    effective_sigma_e,
)
from witwin.channel_native.propagation.geometry.kernels import (
    primitives as geometry_primitives,
)
from witwin.channel_native.propagation.topology.kernels import (
    primitives as topology_primitives,
)


Receiver = ReceiverPoint | ReceiverGrid

_RAYD_EDGE_INFO_PLANE_TOL = 1.34e-5


@dataclass(frozen=True, slots=True)
class Scene:
    structures: tuple[Structure, ...]
    transmitters: tuple[Transmitter, ...]
    receivers: tuple[Receiver, ...]
    frequency: float | torch.Tensor
    metadata: dict[str, object]
    _geometry_version: int = 0
    _material_version: int = 0
    _assignment_version: int = 0
    _compiled_cache: CompiledScene | None = field(
        default=None, init=False, compare=False, repr=False
    )
    _raydn_cache: RayDNScene | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def __init__(
        self,
        *,
        structures: list[Structure] | tuple[Structure, ...],
        transmitters: list[Transmitter] | tuple[Transmitter, ...],
        receivers: list[Receiver] | tuple[Receiver, ...],
        frequency: float | torch.Tensor,
        metadata: dict[str, object] | None = None,
        _geometry_version: int = 0,
        _material_version: int = 0,
        _assignment_version: int = 0,
    ) -> None:
        # The carrier frequency may be a 0-d torch tensor so it can carry
        # requires_grad / forward-mode tangents into the differentiable field
        # kernels (plan 07 AD-1). Every non-AD consumer reads it through
        # float(scene.frequency), which detaches by contract.
        if isinstance(frequency, torch.Tensor):
            if frequency.ndim != 0:
                raise ValueError("tensor frequency must be a 0-d tensor")
            if float(frequency.detach()) <= 0.0:
                raise ValueError("frequency must be positive")
        else:
            if frequency <= 0.0:
                raise ValueError("frequency must be positive")
            frequency = float(frequency)
        object.__setattr__(self, "structures", tuple(structures))
        object.__setattr__(self, "transmitters", tuple(transmitters))
        object.__setattr__(self, "receivers", tuple(receivers))
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "_geometry_version", _geometry_version)
        object.__setattr__(self, "_material_version", _material_version)
        object.__setattr__(self, "_assignment_version", _assignment_version)
        object.__setattr__(self, "_compiled_cache", None)
        object.__setattr__(self, "_raydn_cache", None)

    @classmethod
    def load_mitsuba(cls, filename: str, **kwargs) -> Scene:
        from .scene_loader import load_mitsuba

        return load_mitsuba(filename, scene_cls=cls, **kwargs)

    def add(self, obj: Transmitter | Receiver) -> Scene:
        if isinstance(obj, Transmitter):
            object.__setattr__(self, "transmitters", self.transmitters + (obj,))
            return self
        if isinstance(obj, (ReceiverPoint, ReceiverGrid)):
            object.__setattr__(self, "receivers", self.receivers + (obj,))
            return self
        raise TypeError(f"scene object type is not accepted: {type(obj).__name__}")

    def with_structure_vertices(self, index: int, vertices: torch.Tensor) -> Scene:
        structures = list(self.structures)
        structures[index] = structures[index].with_vertices(vertices)
        return replace(
            self,
            structures=tuple(structures),
            _geometry_version=self._geometry_version + 1,
        )

    def with_structure_material(self, index: int, material: object) -> Scene:
        structures = list(self.structures)
        structures[index] = structures[index].with_material(material)  # type: ignore[arg-type]
        return replace(
            self,
            structures=tuple(structures),
            _material_version=self._material_version + 1,
        )

    def with_frequency(self, frequency: float) -> Scene:
        return replace(
            self, frequency=frequency, _material_version=self._material_version + 1
        )

    def diffraction_edge_count(self, edge_policy: EdgePolicy | None = None) -> int:
        policy = DEFAULT_EDGE_POLICY if edge_policy is None else edge_policy
        if not policy.edge_diffraction:
            return 0
        if len(self.structures) == 0:
            return 0
        raydn_scene = self.raydn_scene()
        if not raydn_scene.available:
            raise RuntimeError(
                "diffraction edge counting requires RayDN native scene capability"
            )
        return _diffraction_edge_count_from_raydn_scene(raydn_scene, policy)

    @property
    def n_diffraction_edges(self) -> int:
        return self.diffraction_edge_count()

    def raydn_scene(self) -> RayDNScene:
        cached = self._raydn_cache
        if cached is not None:
            return cached
        raydn = build_scene_from_structures(self.structures)
        # Expose the scene's edge policy to the diffraction edge-geometry
        # builders so path generation honors it (audit DF-4).
        raydn.runtime_cache["edge_policy"] = resolve_scene_edge_policy(self)
        object.__setattr__(self, "_raydn_cache", raydn)
        return raydn

    def compile(self) -> CompiledScene:
        cached = self._compiled_cache
        # Material records are always evaluated at the primal frequency:
        # dispersive material laws are frozen at compile time (fixed material
        # records), so frequency gradients flow only through the explicit
        # frequency dependence of the field kernels (plan 07 AD-1). The AD
        # seam therefore refuses frequency AD when any record is frequency
        # dependent (see materials.frequency_dependent).
        frequency_value = (
            float(self.frequency.detach())
            if isinstance(self.frequency, torch.Tensor)
            else float(self.frequency)
        )
        (
            material_records,
            material_keys,
            material_cache_token,
            phase_screens,
        ) = _material_records(self.structures, frequency_value)
        if (
            cached is not None
            and cached.geometry_version == self._geometry_version
            and cached.material_version == self._material_version
            and cached.assignment_version == self._assignment_version
            and cached.materials.cache_token == material_cache_token
        ):
            return cached
        raydn = self.raydn_scene()
        geometry = _compile_geometry(
            self.structures, self._geometry_version, raydn=raydn
        )
        materials = _compile_materials(
            material_records,
            material_keys,
            frequency_value,
            self._material_version,
            material_cache_token,
            frequency_dependent=_frequency_dependent_material_keys(
                self.structures, material_records, material_keys, frequency_value
            ),
        )
        assignments = _compile_assignments(
            self.structures,
            num_faces=geometry.faces.shape[0],
            num_edges=geometry.edges.shape[0],
            version=self._assignment_version,
            phase_screens=phase_screens,
        )
        compiled = CompiledScene(
            geometry=geometry,
            materials=materials,
            assignments=assignments,
            raydn=raydn,
            workspace=None,
            geometry_version=geometry.version,
            material_version=materials.version,
            assignment_version=assignments.version,
        )
        object.__setattr__(self, "_compiled_cache", compiled)
        return compiled


def _diffraction_edge_count_from_raydn_scene(
    raydn_scene: RayDNScene, edge_policy: EdgePolicy
) -> int:
    records = raydn_scene.edge_records()
    return geometry_primitives.core_diffraction_edge_count(
        vertices=records.vertices,
        faces=records.faces,
        face_normals=records.face_normals,
        edge_v0=records.edge_v0,
        edge_v1=records.edge_v1,
        face0=records.face0,
        face1=records.face1,
        vertical_only=edge_policy.vertical_only,
        vertical_ratio=float(edge_policy.vertical_ratio),
        boundary_half_plane=edge_policy.boundary_edge_policy == "half_plane",
        plane_tol=_RAYD_EDGE_INFO_PLANE_TOL,
    )


def _compile_geometry(
    structures: tuple[Structure, ...], version: int, *, raydn: RayDNScene
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
            version=version,
        )

    if not raydn.available:
        raydn.require_handle()
    records = raydn.edge_records()
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
        edge_adj_faces=topology_primitives.core_pack_int2(
            records.face0, records.face1
        ),
        edge_param_range=bdpt_zero_matrix(
            records.vertices, rows=records.edge_v0.shape[0], cols=2
        ),
        face_structure_id=torch.tensor(face_structure_id, dtype=torch.int32),
        face_surface_id=torch.tensor(face_surface_id, dtype=torch.int32),
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
        except Exception:
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
        return torch.tensor(
            [row[column] for row in layer_rows], dtype=torch.float32
        )

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
