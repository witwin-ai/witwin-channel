from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import drjit as dr
import rayd
import torch
from witwin.channel import types as wt

from witwin.core import GeometryBase, SceneBase, Structure
from witwin.channel.core.numerics.arrays import broadcast, concat_points, scalar
from witwin.channel.core.numerics.constants import EPS, RAY_ORIGIN_BIAS
from witwin.channel.core.geometry import point_in_triangle_3d
from witwin.channel.core.physics.wave_math import material_angular_frequency
from witwin.channel.core.physics.materials import FaceMaterial
from witwin.channel.core.geometry.mesh_buffers import mesh_buffer_count, to_point3f
from witwin.channel.core.runtime import point_grad_enabled, scene_geometry_grad_enabled, scene_material_grad_enabled
from .arrays import default_array
from .builder import MATERIAL_EPS_R_DEFAULT, MATERIAL_SIGMA_E_DEFAULT, SceneBuilder
from .edge_policy import DEFAULT_EDGE_POLICY, EdgePolicy
from .endpoints import Receiver, ReceiverGrid, Transmitter
from .wedge import WedgeOps, WedgePack


_NATIVE_IGNORE_MAX_ENTRIES = 2_000_000_000


def _resolve_scene_device(device: str | None) -> str:
    resolved = torch.device("cuda" if device is None else device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Channel scenes default to CUDA, but torch.cuda.is_available() is False. "
            "Pass device='cpu' only for scene construction or non-rendering workflows."
        )
    return str(resolved)

def _normalized_constant_direction(direction):
    vector = wt.Vector3f(float(direction[0]), float(direction[1]), float(direction[2]))
    return vector / (dr.norm(vector) + EPS)


def _scalar_int(value) -> int:
    try:
        return int(value)
    except TypeError:
        return int(value[0])


def _scalar_bool(value) -> bool:
    try:
        return bool(value)
    except TypeError:
        return bool(value[0])


def _gather_scalar(buffer, idx: wt.UInt32):
    return dr.gather(type(buffer), buffer, idx)


def _grad_enabled_value(value) -> bool:
    if value is None:
        return False
    try:
        return bool(dr.grad_enabled(value))
    except (TypeError, RuntimeError):
        pass
    if isinstance(value, dict):
        return any(_grad_enabled_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_grad_enabled_value(item) for item in value)
    try:
        has_xyz = all(hasattr(value, axis) for axis in ("x", "y", "z"))
    except RuntimeError:
        has_xyz = False
    if has_xyz:
        return any(_grad_enabled_value(getattr(value, axis)) for axis in ("x", "y", "z"))
    try:
        has_complex_parts = hasattr(value, "real") and hasattr(value, "imag")
    except RuntimeError:
        has_complex_parts = False
    if has_complex_parts:
        real = value.real
        imag = value.imag
        return (
            (real is not value and _grad_enabled_value(real))
            or (imag is not value and _grad_enabled_value(imag))
        )
    return False


def _rayd_float(value, *, ad: bool):
    value = wt.Float(value)
    if ad:
        return value
    return dr.detached_t(wt.Float)(dr.detach(value))


def _rayd_int(value, *, ad: bool):
    value = wt.Int32(value)
    if ad:
        return value
    return dr.detached_t(wt.Int32)(dr.detach(value))


def _rayd_bool(value, *, ad: bool):
    value = wt.Bool(value)
    if ad:
        return value
    return dr.detached_t(wt.Bool)(dr.detach(value))


def _rayd_vector3(value, *, ad: bool):
    vector_type = wt.Vector3f if ad else dr.detached_t(wt.Vector3f)
    if all(hasattr(value, axis) for axis in ("x", "y", "z")):
        return vector_type(
            _rayd_float(value.x, ad=ad),
            _rayd_float(value.y, ad=ad),
            _rayd_float(value.z, ad=ad),
        )
    if len(value) != 3:
        raise ValueError("Expected a 3-vector.")
    return vector_type(
        _rayd_float(value[0], ad=ad),
        _rayd_float(value[1], ad=ad),
        _rayd_float(value[2], ad=ad),
    )


def _rayd_complex(value, *, ad: bool):
    value = wt.Complex2f(value)
    complex_type = wt.Complex2f if ad else dr.detached_t(wt.Complex2f)
    return complex_type(
        _rayd_float(value.real, ad=ad),
        _rayd_float(value.imag, ad=ad),
    )


def _rayd_call(trace, *args, ad: bool):
    if ad:
        return trace(*args)
    with dr.suspend_grad():
        return trace(*args)


_DFR_STATE_GRAD_ATTRS = (
    "edge_index",
    "edge_pos",
    "edge_dir",
    "edge_line_min",
    "edge_line_max",
    "n0",
    "n_face_n",
    "adjacent_face0",
    "adjacent_face1",
    "wedge_n",
    "source_pos",
    "source_power",
    "prefix_reflection_depth",
    "prefix_initial_ray_dir",
)


def _dfr_states_grad_enabled(states) -> bool:
    if isinstance(states, dict):
        return _grad_enabled_value(states)
    return any(
        _grad_enabled_value(getattr(states, name, None))
        for name in _DFR_STATE_GRAD_ATTRS
    )


@dataclass(frozen=True)
class _SceneEdgeView:
    vertex_indices: tuple[int, int]
    p0: wt.Point3f
    p1: wt.Point3f
    adjacent_faces: tuple[int, ...]
    is_boundary: bool
    edge_vector: wt.Vector3f
    length: wt.Float
    global_index: int
    local_index: int
    shape_id: int
    local_edge_id: int
    wedge_n: wt.Float | None = None
    face_normals_3d: tuple[wt.Vector3f, ...] = ()


class Scene(SceneBase):
    """Declarative channel scene backed by shared core structures."""

    @classmethod
    def from_sionna(cls, sionna_scene, **kwargs) -> Scene:
        from .sionna_adaptor import SionnaAdaptor
        return SionnaAdaptor.import_scene(sionna_scene, scene_cls=cls, **kwargs)

    @classmethod
    def load_mitsuba(cls, filename, **kwargs) -> Scene:
        from .sionna_adaptor import SionnaAdaptor
        return SionnaAdaptor.load_mitsuba(filename, scene_cls=cls, **kwargs)

    def __init__(
        self,
        *,
        structures=None,
        transmitters=None,
        receivers=None,
        metadata=None,
        frequency: float | None = None,
        tx_array=None,
        rx_array=None,
        device: str | None = "cuda",
        verbose: bool = False,
    ):
        normalized_structures = [self._normalize_structure(s) for s in (structures or ())]
        seen: set = set()
        for s in normalized_structures:
            if s.name in seen:
                raise ValueError(f"Structure '{s.name}' already exists.")
            seen.add(s.name)

        normalized_transmitters = self._normalize_named_endpoints(transmitters, (Transmitter,), role="Transmitter")
        normalized_receivers = self._normalize_named_endpoints(receivers, (Receiver, ReceiverGrid), role="Receiver")

        super().__init__(
            structures=normalized_structures,
            sources=normalized_transmitters,
            monitors=normalized_receivers,
            metadata=metadata,
            device=_resolve_scene_device(device),
            verbose=verbose,
        )
        self.transmitters = self.sources
        self.receivers = self.monitors
        self._frequency = None if frequency is None else float(frequency)
        self.tx_array = tx_array
        self.rx_array = rx_array

        self._rayd_scene = self.tri_data = self._triangle_surface_data = None
        self._triangle_material_data = self._face_normals = None
        self._wedge_geometry = self._wedge_selection = self._wedge_pack = self._wedge_tri_map = None
        self._edge_policy = DEFAULT_EDGE_POLICY
        self._edge_policy_cache_key = None
        self._geometry_cache_key = self._geometry_cache = None
        self._tri_data_cache_key = self._tri_data_cache = None
        self._edge_runtime_cache_key = self._edge_runtime_cache = None
        self._edge_view_cache_key = None
        self._structure_meshes: list[dict] = []
        self._edge_cache = {}
        self._triangle_surface_groups = ()
        self._triangle_surface_group_by_triangle = ()
        self._triangle_surface_edge_groups = ()
        self._edge_runtime_dirty = False
        self._mesh_version = 0
        self._edge_views_cache: tuple[_SceneEdgeView, ...] = ()
        self._edge_data_cache: dict[tuple[int, float], dict[str, object]] = {}
        self._rayd_visibility_ignore_pipeline_warmed = False

        SceneBuilder.rebuild(self)
        self._invalidate_runtime_view_cache()

    @property
    def frequency(self) -> float | None:
        return self._frequency

    @frequency.setter
    def frequency(self, value: float | None) -> None:
        resolved = None if value is None else float(value)
        if getattr(self, "_frequency", None) == resolved:
            return
        self._frequency = resolved
        SceneBuilder.rebuild(self)
        self._invalidate_runtime_view_cache()

    # ------------------------------------------------------------------
    # Structure / endpoint management
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_structure(structure) -> Structure:
        structure_like = (
            isinstance(structure, Structure)
            or (
                type(structure).__name__ == "Structure"
                and hasattr(structure, "geometry")
                and hasattr(structure, "material")
                and hasattr(structure, "enabled")
            )
        )
        if not structure_like:
            raise TypeError("Channel Scene structures must be witwin.core.Structure instances.")
        geometry_like = isinstance(structure.geometry, GeometryBase) or (
            hasattr(structure.geometry, "to_mesh") and hasattr(structure.geometry, "kind")
        )
        if not geometry_like:
            raise TypeError("Channel Scene structures must wrap a GeometryBase geometry.")
        if structure.name is None:
            raise ValueError("Channel Scene structures must define a unique name.")
        return structure

    @staticmethod
    def _normalize_named_endpoints(endpoints, endpoint_types: tuple[type, ...], *, role: str) -> list:
        normalized = list(endpoints or ())
        for endpoint in normalized:
            if not isinstance(endpoint, endpoint_types):
                type_names = ", ".join(cls.__name__ for cls in endpoint_types)
                raise TypeError(f"Channel Scene {role.lower()} endpoints must be {type_names} instances.")
        seen = set()
        for endpoint in normalized:
            if endpoint.name in seen:
                raise ValueError(f"{role} endpoint '{endpoint.name}' already exists.")
            seen.add(endpoint.name)
        return normalized

    @staticmethod
    def _resolve_named_endpoint(endpoints, endpoint, *, role: str):
        if not isinstance(endpoint, str):
            return endpoint
        for candidate in endpoints:
            if candidate.name == endpoint:
                return candidate
        raise ValueError(f"Unknown {role} endpoint '{endpoint}'.")

    def add(self, obj) -> Scene:
        """Append a Structure, Transmitter, Receiver, or ReceiverGrid to the scene."""
        if isinstance(obj, Structure) or (
            type(obj).__name__ == "Structure"
            and hasattr(obj, "geometry")
            and hasattr(obj, "material")
            and hasattr(obj, "enabled")
        ):
            normalized = self._normalize_structure(obj)
            if any(s.name == normalized.name for s in self.structures):
                raise ValueError(f"Structure '{normalized.name}' already exists.")
            self.structures.append(normalized)
            SceneBuilder.rebuild(self)
            self._invalidate_runtime_view_cache()
            return self
        if isinstance(obj, Transmitter):
            if any(e.name == obj.name for e in self.transmitters):
                raise ValueError(f"Transmitter endpoint '{obj.name}' already exists.")
            self.transmitters.append(obj)
            return self
        if isinstance(obj, (Receiver, ReceiverGrid)):
            if any(e.name == obj.name for e in self.receivers):
                raise ValueError(f"Receiver endpoint '{obj.name}' already exists.")
            self.receivers.append(obj)
            return self
        raise TypeError(
            f"Scene.add expects Structure, Transmitter, Receiver, or ReceiverGrid; got {type(obj).__name__}."
        )

    def structure(self, name: str) -> StructureBinding:
        for idx, s in enumerate(self.structures):
            if s.name == name:
                return StructureBinding(scene=self, structure=s, index=idx)
        raise ValueError(f"Unknown structure '{name}'.")

    def transmitter(self, endpoint="tx") -> Transmitter:
        resolved = self._resolve_named_endpoint(self.transmitters, endpoint, role="transmitter")
        if not isinstance(resolved, Transmitter):
            raise TypeError("transmitter endpoint must be a Transmitter instance.")
        return resolved

    def receiver(self, endpoint) -> Receiver | ReceiverGrid:
        resolved = self._resolve_named_endpoint(self.receivers, endpoint, role="receiver")
        if not isinstance(resolved, (Receiver, ReceiverGrid)):
            raise TypeError("receiver endpoint must be a Receiver or ReceiverGrid instance.")
        return resolved

    def transmitter_array(self, endpoint="tx"):
        tx = self.transmitter(endpoint)
        return tx.array if tx.array is not None else (self.tx_array if self.tx_array is not None else default_array())

    def receiver_array(self, endpoint):
        rx = self.receiver(endpoint)
        return rx.array if rx.array is not None else (self.rx_array if self.rx_array is not None else default_array())

    def clone(self, **overrides) -> Scene:
        unsupported = {
            "vertical_ratio",
            "edge_selection_mode",
            "edge_diffraction",
            "boundary_edge_policy",
        } & set(overrides)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"Scene.clone no longer accepts solver edge policy fields: {names}.")
        return Scene(
            structures=overrides.get("structures", list(self.structures)),
            transmitters=overrides.get("transmitters", list(self.transmitters)),
            receivers=overrides.get("receivers", list(self.receivers)),
            metadata=overrides.get("metadata", dict(self.metadata)),
            frequency=overrides.get("frequency", self.frequency),
            tx_array=overrides.get("tx_array", self.tx_array),
            rx_array=overrides.get("rx_array", self.rx_array),
            device=overrides.get("device", self.device),
            verbose=overrides.get("verbose", self.verbose),
        )

    def to_sionna(self, **kwargs):
        from .sionna_adaptor import SionnaAdaptor
        return SionnaAdaptor.export(self, **kwargs)

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(self, recompute_edges: bool = True) -> None:
        """Sync RayD BVH and recompute derived data after mesh vertex updates."""
        SceneBuilder.sync(self, recompute_edges=recompute_edges)
        self._invalidate_runtime_view_cache()

    def _invalidate_runtime_view_cache(self) -> None:
        self._geometry_cache_key = self._geometry_cache = None
        self._tri_data_cache_key = self._tri_data_cache = None
        self._edge_runtime_cache_key = self._edge_runtime_cache = None
        self._edge_view_cache_key = None
        self._edge_views_cache = ()
        self._edge_data_cache.clear()

    # ------------------------------------------------------------------
    # Edge / wedge queries
    # ------------------------------------------------------------------

    @property
    def n_diffraction_edges(self) -> int:
        return self.diffraction_edge_count()

    def diffraction_edge_count(self, *, edge_policy: EdgePolicy | None = None) -> int:
        SceneBuilder.ensure_edge_runtime(self, edge_policy=edge_policy)
        return self._wedge_pack.n_wedges if self._wedge_pack is not None else 0

    def _wedge_pack_at(self, calculation_height: float, *, edge_policy: EdgePolicy | None = None) -> WedgePack | None:
        SceneBuilder.ensure_edge_runtime(self, edge_policy=edge_policy)
        cache_key = (self._mesh_version, self._edge_policy_cache_key, calculation_height)
        if cache_key in self._edge_cache:
            return self._edge_cache[cache_key]

        selection = self._wedge_selection
        if selection is None or selection.size() == 0:
            self._edge_cache[cache_key] = None
            return None

        pack = WedgeOps.pack(selection, WedgeOps.build_anchors(selection, calculation_height))
        self._edge_cache[cache_key] = pack
        return pack

    def _merged_geometry(self) -> tuple[wt.Point3f, wt.Vector3u]:
        cache_key = (int(self._mesh_version), len(self._structure_meshes))
        if self._geometry_cache_key == cache_key and self._geometry_cache is not None:
            return self._geometry_cache

        geometry = (wt.Point3f(), wt.Vector3u()) if self._rayd_scene is None else \
            SceneBuilder._global_geometry_buffers(self._rayd_scene.global_geometry())
        self._geometry_cache_key, self._geometry_cache = cache_key, geometry
        return geometry

    @property
    def vertices(self):
        return self._merged_geometry()[0]

    @property
    def faces(self):
        return self._merged_geometry()[1]

    def _merged_vertices(self):
        return self.vertices

    def _merged_faces(self):
        return self.faces

    def _triangle_runtime(self):
        tri_data = self.tri_data
        if tri_data is None:
            return None

        cache_key = (
            int(self._mesh_version),
            id(tri_data),
            self._edge_policy_cache_key,
            int(tri_data.get("surface_max_edge_count", 0)),
        )
        if self._tri_data_cache_key == cache_key and self._tri_data_cache is not None:
            return self._tri_data_cache

        vertices, faces = self._merged_geometry()
        resolved = dict(tri_data)
        if int(resolved.get("n_triangles", 0)) > 0:
            resolved["v0"] = dr.gather(wt.Point3f, vertices, faces.x)
            resolved["v1"] = dr.gather(wt.Point3f, vertices, faces.y)
            resolved["v2"] = dr.gather(wt.Point3f, vertices, faces.z)

        self._tri_data_cache_key, self._tri_data_cache = cache_key, resolved
        return resolved

    def _selected_edge_runtime(self, *, edge_policy: EdgePolicy | None = None):
        SceneBuilder.ensure_edge_runtime(self, edge_policy=edge_policy)
        pack = self._wedge_pack
        tri_data = self._triangle_runtime()
        cache_key = (int(self._mesh_version), id(pack), None if tri_data is None else id(tri_data))
        if self._edge_runtime_cache_key == cache_key and self._edge_runtime_cache is not None:
            return self._edge_runtime_cache

        if pack is None or int(pack.n_wedges) <= 0:
            self._edge_runtime_cache_key, self._edge_runtime_cache = cache_key, None
            return None

        adj_group0 = adj_group1 = None
        if tri_data is not None and "surface_group_id" in tri_data:
            group_id = wt.UInt32(tri_data["surface_group_id"])
            n_tri = int(tri_data.get("n_triangles", 0))
            adj_group0 = self._gather_adjacent_group(pack.adjacent_face0, group_id, n_tri)
            adj_group1 = self._gather_adjacent_group(pack.adjacent_face1, group_id, n_tri)

        runtime = {
            "pos": pack.pos, "edge_dir": pack.edge_dir,
            "n0": pack.n0, "n_face_n": pack.nn, "wedge_n": pack.wedge_n,
            "length": pack.length, "line_min": pack.line_min, "line_max": pack.line_max,
            "adjacent_face0": wt.Int32(pack.adjacent_face0),
            "adjacent_face1": wt.Int32(pack.adjacent_face1),
            "adjacent_surface_group0": adj_group0,
            "adjacent_surface_group1": adj_group1,
            "global_idx": wt.Int32(pack.global_idx),
            "n_edges": int(pack.n_wedges),
        }
        self._edge_runtime_cache_key, self._edge_runtime_cache = cache_key, runtime
        return runtime

    @staticmethod
    def _gather_adjacent_group(face_buffer, group_id: wt.UInt32, n_triangles: int) -> wt.Int32:
        face = wt.Int32(face_buffer)
        valid = (face >= 0) & (face < wt.Int32(n_triangles))
        safe = wt.UInt32(dr.select(valid, face, wt.Int32(0)))
        return dr.select(valid, wt.Int32(dr.gather(wt.UInt32, group_id, safe)), wt.Int32(-1))

    def _refresh_edge_views(self) -> None:
        geometry = self._wedge_geometry
        selection = self._wedge_selection
        cache_key = (int(self._mesh_version), id(geometry), id(selection))
        if self._edge_view_cache_key == cache_key:
            return

        if geometry is None or selection is None or selection.size() == 0:
            self._edge_views_cache, self._edge_view_cache_key = (), cache_key
            return

        selected_idx = selection.selected_idx
        edges: list[_SceneEdgeView] = []
        for local_index in range(selection.size()):
            geom_idx = dr.gather(wt.UInt32, selected_idx, wt.UInt32(local_index))
            face0 = _scalar_int(_gather_scalar(geometry.face0, geom_idx))
            face1 = _scalar_int(_gather_scalar(geometry.face1, geom_idx))
            length = _gather_scalar(geometry.length, geom_idx)
            edges.append(_SceneEdgeView(
                vertex_indices=(_scalar_int(_gather_scalar(geometry.v0, geom_idx)),
                                _scalar_int(_gather_scalar(geometry.v1, geom_idx))),
                p0=_gather_scalar(geometry.start, geom_idx),
                p1=_gather_scalar(geometry.end, geom_idx),
                adjacent_faces=tuple(f for f in (face0, face1) if f >= 0),
                is_boundary=_scalar_bool(_gather_scalar(geometry.is_boundary, geom_idx)),
                edge_vector=_gather_scalar(geometry.edge_dir, geom_idx) * length,
                length=length,
                global_index=_scalar_int(_gather_scalar(geometry.global_edge_id, geom_idx)),
                local_index=local_index,
                shape_id=_scalar_int(_gather_scalar(geometry.shape_id, geom_idx)),
                local_edge_id=_scalar_int(_gather_scalar(geometry.local_edge_id, geom_idx)),
                wedge_n=_gather_scalar(geometry.wedge_n, geom_idx),
                face_normals_3d=(_gather_scalar(geometry.n0, geom_idx),
                                 _gather_scalar(geometry.nn, geom_idx)),
            ))

        self._edge_views_cache, self._edge_view_cache_key = tuple(edges), cache_key

    def _selected_edge_views(self, *, edge_policy: EdgePolicy | None = None):
        SceneBuilder.ensure_edge_runtime(self, edge_policy=edge_policy)
        self._refresh_edge_views()
        return self._edge_views_cache

    def get_triangle_surface_edge_candidates(self, prim_idx):
        tri_data = self._triangle_runtime()
        max_count = 0 if tri_data is None else int(tri_data.get("surface_max_edge_count", 0))
        if tri_data is None or max_count <= 0:
            return {"count": wt.UInt32(0), "slots": ()}

        n_tri = int(tri_data["n_triangles"])
        prim_i32 = wt.Int32(prim_idx)
        valid = (prim_i32 >= 0) & (prim_i32 < wt.Int32(n_tri))
        safe = wt.UInt32(dr.select(valid, prim_i32, wt.Int32(0)))
        count = dr.select(valid, dr.gather(wt.UInt32, tri_data["surface_edge_size"], safe), wt.UInt32(0))
        slots = tuple(
            dr.select(valid, dr.gather(wt.Int32, tri_data["surface_edge_indices"], safe * wt.UInt32(max_count) + wt.UInt32(s)), wt.Int32(-1))
            for s in range(max_count)
        )
        return {"count": count, "slots": slots}

    def get_edge_data(self, calculation_height, include_projection: bool = True, edge_policy: EdgePolicy | None = None):
        SceneBuilder.ensure_edge_runtime(self, edge_policy=edge_policy)
        policy = self._edge_policy
        cache_key = (
            int(self._mesh_version),
            self._edge_policy_cache_key,
            float(calculation_height),
            bool(include_projection),
        )
        if cache_key in self._edge_data_cache:
            return self._edge_data_cache[cache_key]

        pack = self._wedge_pack_at(float(calculation_height), edge_policy=edge_policy)
        if pack is None or int(pack.n_wedges) <= 0:
            entry = {
                "edge_data": None, "edge_pack": None, "edges_2d": None, "corners_2d": None,
                "diffraction_points": (),
                "edge_diffraction": policy.edge_diffraction,
                "boundary_edge_policy": policy.boundary_edge_policy,
            }
            self._edge_data_cache[cache_key] = entry
            return entry

        edge_runtime = self._selected_edge_runtime(edge_policy=edge_policy)
        diffraction_points = (
            tuple(self._make_diffraction_point(pack, i) for i in range(int(pack.n_wedges)))
            if bool(include_projection)
            else ()
        )

        entry = {
            "edge_data": {
                "pos": pack.pos, "edge_dir": pack.edge_dir,
                "n0": pack.n0, "n_face_n": pack.nn, "wedge_n": pack.wedge_n,
                "length": pack.length, "line_min": pack.line_min, "line_max": pack.line_max,
                "adjacent_face0": wt.Int32(pack.adjacent_face0),
                "adjacent_face1": wt.Int32(pack.adjacent_face1),
                "adjacent_surface_group0": None if edge_runtime is None else edge_runtime["adjacent_surface_group0"],
                "adjacent_surface_group1": None if edge_runtime is None else edge_runtime["adjacent_surface_group1"],
                "global_idx": wt.Int32(pack.global_idx),
                "n_edges": int(pack.n_wedges),
            },
            "edge_pack": pack,
            "edges_2d": None, "corners_2d": None,
            "diffraction_points": diffraction_points,
            "edge_diffraction": policy.edge_diffraction,
            "boundary_edge_policy": policy.boundary_edge_policy,
        }
        self._edge_data_cache[cache_key] = entry
        return entry

    @staticmethod
    def _make_diffraction_point(pack: WedgePack, edge_idx: int) -> SimpleNamespace:
        safe_idx = wt.UInt32(edge_idx)
        length = dr.gather(wt.Float, pack.length, safe_idx)
        return SimpleNamespace(
            position=dr.gather(wt.Point3f, pack.pos, safe_idx),
            edge_vector=dr.gather(wt.Vector3f, pack.edge_dir, safe_idx) * length,
            length=length,
            wedge_n=dr.gather(wt.Float, pack.wedge_n, safe_idx),
            face_normals_3d=(dr.gather(wt.Vector3f, pack.n0, safe_idx),
                             dr.gather(wt.Vector3f, pack.nn, safe_idx)),
            adjacent_faces=(_scalar_int(dr.gather(wt.Int32, pack.adjacent_face0, safe_idx)),
                            _scalar_int(dr.gather(wt.Int32, pack.adjacent_face1, safe_idx))),
            global_index=_scalar_int(dr.gather(wt.Int32, pack.global_idx, safe_idx)),
            line_min=dr.gather(wt.Float, pack.line_min, safe_idx),
            line_max=dr.gather(wt.Float, pack.line_max, safe_idx),
        )

    def _require_rayd_scene(self):
        if self._rayd_scene is None:
            raise RuntimeError("Scene ray queries require an active RayD runtime scene.")
        return self._rayd_scene

    # ------------------------------------------------------------------
    # Material queries
    # ------------------------------------------------------------------

    def triangle_material(self, prim_idx: wt.UInt32, valid_mask: wt.Bool | None = None) -> dict:
        tri_data = self._triangle_runtime()
        if tri_data is None:
            width = int(dr.width(wt.Int32(prim_idx)))
            return {
                "eps_r": dr.full(wt.Float, MATERIAL_EPS_R_DEFAULT, width),
                "mu_r": dr.full(wt.Float, 1.0, width),
                "sigma_e": dr.full(wt.Float, MATERIAL_SIGMA_E_DEFAULT, width),
                "specified": dr.full(wt.Bool, False, width),
                "structure_idx": dr.full(wt.Int32, -1, width),
                "valid": dr.full(wt.Bool, False, width),
            }

        prim_i32 = wt.Int32(prim_idx)
        if valid_mask is None:
            valid_mask = wt.Bool(True) if dr.width(prim_i32) == 1 else (prim_i32 >= 0)

        n_tri = int(tri_data["n_triangles"])
        valid = valid_mask & (prim_i32 >= 0) & (prim_i32 < wt.Int32(n_tri))
        safe = wt.UInt32(dr.select(valid, prim_i32, wt.Int32(0)))
        return {
            "eps_r": dr.select(valid, dr.gather(wt.Float, tri_data["material_eps_r"], safe), wt.Float(MATERIAL_EPS_R_DEFAULT)),
            "mu_r": dr.select(valid, dr.gather(wt.Float, tri_data["material_mu_r"], safe), wt.Float(1.0)),
            "sigma_e": dr.select(valid, dr.gather(wt.Float, tri_data["material_sigma_e"], safe), wt.Float(MATERIAL_SIGMA_E_DEFAULT)),
            "specified": dr.select(valid, dr.gather(wt.Bool, tri_data["material_specified"], safe), wt.Bool(False)),
            "structure_idx": dr.select(valid, dr.gather(wt.Int32, tri_data["material_structure_idx"], safe), wt.Int32(-1)),
            "valid": valid,
        }

    def gather_structure_indices(self, prim_idx: wt.UInt32, *, valid_mask: wt.Bool | None = None) -> wt.Int32:
        width = dr.width(prim_idx)
        tri_data = self._triangle_runtime()
        if tri_data is None or "material_structure_idx" not in tri_data:
            return dr.full(wt.Int32, -1, width)

        prim_i32 = wt.Int32(prim_idx)
        if valid_mask is None:
            valid_mask = prim_i32 >= 0
        n_tri = int(tri_data["n_triangles"])
        valid = valid_mask & (prim_i32 >= 0) & (prim_i32 < wt.Int32(n_tri))
        safe = wt.UInt32(dr.select(valid, prim_i32, wt.Int32(0)))
        return dr.select(valid, dr.gather(wt.Int32, tri_data["material_structure_idx"], safe), wt.Int32(-1))

    # ------------------------------------------------------------------
    # Ray queries
    # ------------------------------------------------------------------

    def ray_test(self, ray: object, active: bool = True) -> wt.Bool:
        if not isinstance(ray, (rayd.Ray, rayd.RayAD)):
            raise TypeError(f"Scene.ray_test expects rayd.Ray or rayd.RayAD, got {type(ray).__name__}.")
        return wt.Bool(self._require_rayd_scene().shadow_test(ray, active=active))

    def ray_intersect(self, ray: object, active: bool = True, flags: object = None) -> object:
        if not isinstance(ray, (rayd.Ray, rayd.RayAD)):
            raise TypeError(f"Scene.ray_intersect expects rayd.Ray or rayd.RayAD, got {type(ray).__name__}.")
        return self._require_rayd_scene().intersect(
            ray, active=active, flags=rayd.RayFlags.All if flags is None else flags,
        )

    def nearest_edge(self, query, active: bool = True):
        return self._require_rayd_scene().nearest_edge(query, active=active)

    @staticmethod
    def _make_rayd_dfr_states(diffraction_states, *, ad: bool = False):
        if isinstance(diffraction_states, dict):
            return Scene._make_rayd_dfr_states_from_arrays(diffraction_states, ad=ad)

        edge_index = wt.Int32(diffraction_states.edge_index)
        width = int(dr.width(edge_index))
        count = getattr(diffraction_states, "stored_count", None)
        count = width if count is None else int(count)
        if count < 0 or count > width:
            raise ValueError(
                "DiffractionStates.stored_count must be between 0 and the state buffer width."
            )

        source_to_edge = wt.Vector3f(diffraction_states.edge_pos - diffraction_states.source_pos)
        incident_direction = source_to_edge / (dr.norm(source_to_edge) + EPS)
        prefix_depth = wt.Int32(diffraction_states.prefix_reflection_depth)
        prefix_initial = getattr(diffraction_states, "prefix_initial_ray_dir", None)
        if prefix_initial is None:
            initial_direction = incident_direction
        else:
            initial_direction = dr.select(
                prefix_depth > wt.Int32(0),
                wt.Vector3f(prefix_initial),
                incident_direction,
            )

        table = rayd.DfrStatesAD() if ad else rayd.DfrStates()
        table.count = count
        table.edge_index = _rayd_int(edge_index, ad=ad)
        table.edge_pos = _rayd_vector3(diffraction_states.edge_pos, ad=ad)
        table.edge_dir = _rayd_vector3(diffraction_states.edge_dir, ad=ad)
        table.edge_t_min = _rayd_float(diffraction_states.edge_line_min, ad=ad)
        table.edge_t_max = _rayd_float(diffraction_states.edge_line_max, ad=ad)
        table.n0 = _rayd_vector3(diffraction_states.n0, ad=ad)
        table.n1 = _rayd_vector3(diffraction_states.n_face_n, ad=ad)
        table.prim0 = _rayd_int(diffraction_states.adjacent_face0, ad=ad)
        table.prim1 = _rayd_int(diffraction_states.adjacent_face1, ad=ad)
        table.exterior_angle = _rayd_float(diffraction_states.wedge_n * dr.pi, ad=ad)
        table.src = _rayd_vector3(diffraction_states.source_pos, ad=ad)
        table.src_power = _rayd_float(diffraction_states.source_power, ad=ad)
        table.wi = _rayd_vector3(incident_direction, ad=ad)
        table.d0 = _rayd_vector3(initial_direction, ad=ad)
        table.prefix_depth = _rayd_int(prefix_depth, ad=ad)
        return table

    @staticmethod
    def _make_rayd_dfr_states_from_arrays(state_arrays, *, ad: bool = False):
        if state_arrays is None:
            raise ValueError("trace_dfr_paths requires diffraction state arrays.")
        state_count = int(state_arrays.get("n_states", 0))
        if state_count <= 0:
            raise ValueError("trace_dfr_paths requires at least one diffraction state.")

        def required(name: str):
            if name not in state_arrays:
                raise KeyError(f"trace_dfr_paths state arrays missing {name!r}.")
            return state_arrays[name]

        edge_pos = required("edge_pos")
        edge_dir = required("edge_dir")
        source_pos = required("source_pos")
        source_to_edge = wt.Vector3f(edge_pos - source_pos)
        incident_direction = source_to_edge / (dr.norm(source_to_edge) + EPS)
        edge_index = state_arrays.get("edge_idx")
        if edge_index is None:
            edge_index = dr.arange(wt.UInt32, state_count)
        source_power = state_arrays.get("source_power")
        if source_power is None:
            incident_field = state_arrays.get("incident_field")
            if incident_field is None:
                source_power = dr.ones(wt.Float, state_count)
            else:
                incident_field = wt.Complex2f(incident_field)
                source_power = dr.square(incident_field.real) + dr.square(incident_field.imag)
        prefix_depth = wt.Int32(state_arrays.get("prefix_reflection_depth", dr.zeros(wt.UInt32, state_count)))

        table = rayd.DfrStatesAD() if ad else rayd.DfrStates()
        table.count = state_count
        table.edge_index = _rayd_int(edge_index, ad=ad)
        table.edge_pos = _rayd_vector3(edge_pos, ad=ad)
        table.edge_dir = _rayd_vector3(edge_dir, ad=ad)
        table.edge_t_min = _rayd_float(required("edge_line_min"), ad=ad)
        table.edge_t_max = _rayd_float(required("edge_line_max"), ad=ad)
        table.n0 = _rayd_vector3(required("n0"), ad=ad)
        table.n1 = _rayd_vector3(state_arrays.get("n_face_n", required("nn") if "nn" in state_arrays else required("n_face_n")), ad=ad)
        table.prim0 = _rayd_int(required("adjacent_face0"), ad=ad)
        table.prim1 = _rayd_int(required("adjacent_face1"), ad=ad)
        table.exterior_angle = _rayd_float(required("wedge_n") * dr.pi, ad=ad)
        table.src = _rayd_vector3(source_pos, ad=ad)
        table.src_power = _rayd_float(source_power, ad=ad)
        table.wi = _rayd_vector3(incident_direction, ad=ad)
        table.d0 = _rayd_vector3(incident_direction, ad=ad)
        table.prefix_depth = _rayd_int(prefix_depth, ad=ad)
        return table

    @staticmethod
    def _make_rayd_dfr_coherent_utd_states_from_arrays(state_arrays, *, ad: bool = False):
        if state_arrays is None:
            raise ValueError("accum_dfr_coherent_direct requires diffraction state arrays.")
        state_count = int(state_arrays.get("n_states", 0))
        if state_count <= 0:
            raise ValueError("accum_dfr_coherent_direct requires at least one diffraction state.")

        def required(name: str):
            if name not in state_arrays:
                raise KeyError(f"accum_dfr_coherent_direct state arrays missing {name!r}.")
            return state_arrays[name]

        table_type = rayd.DfrCoherentUtdStatesAD if ad else rayd.DfrCoherentUtdStates
        table = table_type()
        table.count = state_count
        table.edge_index = _rayd_int(
            state_arrays.get("edge_idx", dr.arange(wt.UInt32, state_count)),
            ad=ad,
        )
        table.edge_pos = _rayd_vector3(required("edge_pos"), ad=ad)
        table.edge_dir = _rayd_vector3(required("edge_dir"), ad=ad)
        table.n0 = _rayd_vector3(required("n0"), ad=ad)
        table.n_face_n = _rayd_vector3(
            state_arrays.get("n_face_n", required("nn") if "nn" in state_arrays else required("n_face_n")),
            ad=ad,
        )
        table.wedge_n = _rayd_float(required("wedge_n"), ad=ad)
        table.edge_line_min = _rayd_float(required("edge_line_min"), ad=ad)
        table.edge_line_max = _rayd_float(required("edge_line_max"), ad=ad)
        table.source_pos = _rayd_vector3(required("source_pos"), ad=ad)
        table.incident_field = _rayd_complex(required("incident_field"), ad=ad)
        table.incident_normal_derivative = _rayd_complex(
            required("incident_normal_derivative"), ad=ad
        )
        table.r_face0 = _rayd_complex(required("r_face0"), ad=ad)
        table.r_face_n = _rayd_complex(required("r_face_n"), ad=ad)
        table.incident_vector_x = _rayd_complex(required("incident_vector_x"), ad=ad)
        table.incident_vector_y = _rayd_complex(required("incident_vector_y"), ad=ad)
        table.incident_vector_z = _rayd_complex(required("incident_vector_z"), ad=ad)
        table.incident_normal_derivative_vector_x = _rayd_complex(
            required("incident_normal_derivative_vector_x"), ad=ad
        )
        table.incident_normal_derivative_vector_y = _rayd_complex(
            required("incident_normal_derivative_vector_y"), ad=ad
        )
        table.incident_normal_derivative_vector_z = _rayd_complex(
            required("incident_normal_derivative_vector_z"), ad=ad
        )
        table.incident_jones_u = _rayd_complex(required("incident_jones_u"), ad=ad)
        table.incident_jones_v = _rayd_complex(required("incident_jones_v"), ad=ad)
        table.incident_derivative_jones_u = _rayd_complex(
            required("incident_derivative_jones_u"), ad=ad
        )
        table.incident_derivative_jones_v = _rayd_complex(
            required("incident_derivative_jones_v"), ad=ad
        )
        table.incident_basis_u = _rayd_vector3(required("incident_basis_u"), ad=ad)
        table.incident_basis_v = _rayd_vector3(required("incident_basis_v"), ad=ad)
        table.incident_basis_k = _rayd_vector3(required("incident_basis_k"), ad=ad)
        for prefix in ("face0", "face1"):
            for key in ("m00", "m01", "m10", "m11"):
                setattr(
                    table,
                    f"{prefix}_operator_{key}",
                    _rayd_complex(required(f"{prefix}_operator_{key}"), ad=ad),
                )
            setattr(table, f"{prefix}_eta_r", _rayd_float(required(f"{prefix}_eta_r"), ad=ad))
            setattr(table, f"{prefix}_mu_r", _rayd_float(required(f"{prefix}_mu_r"), ad=ad))
            setattr(table, f"{prefix}_sigma", _rayd_float(required(f"{prefix}_sigma"), ad=ad))
            setattr(table, f"{prefix}_gain", _rayd_float(required(f"{prefix}_gain"), ad=ad))
            setattr(
                table,
                f"{prefix}_use_fresnel",
                _rayd_float(wt.Float(required(f"{prefix}_use_fresnel")), ad=ad),
            )

        source_type = wt.UInt32(state_arrays.get("source_type_code", dr.zeros(wt.UInt32, state_count)))
        order = wt.UInt32(state_arrays.get("order", dr.full(wt.UInt32, 1, state_count)))
        select_stationary = wt.Float((source_type == wt.UInt32(0)) & (order == wt.UInt32(1)))
        prefix_depth = wt.UInt32(state_arrays.get("prefix_reflection_depth", dr.zeros(wt.UInt32, state_count)))
        intermediate_depth = wt.UInt32(
            state_arrays.get("intermediate_reflection_depth", dr.zeros(wt.UInt32, state_count))
        )
        suffix_depth = wt.UInt32(state_arrays.get("suffix_reflection_depth", dr.zeros(wt.UInt32, state_count)))
        has_reflection = (
            (prefix_depth > wt.UInt32(0))
            | (intermediate_depth > wt.UInt32(0))
            | (suffix_depth > wt.UInt32(0))
        )
        table.select_stationary_point = _rayd_float(select_stationary, ad=ad)
        table.owner_code = _rayd_int(dr.select(has_reflection, wt.Int32(1), wt.Int32(0)), ad=ad)
        table.adjacent_face0 = _rayd_int(required("adjacent_face0"), ad=ad)
        table.adjacent_face1 = _rayd_int(required("adjacent_face1"), ad=ad)
        return table

    @staticmethod
    def _make_rayd_dfr_coherent_edge_table(edge_data, *, ad: bool = False):
        if edge_data is None:
            raise ValueError("build_dfr_coherent_tx_states requires edge data.")
        edge_count = int(edge_data.get("n_edges", 0))
        table_type = rayd.DfrCoherentEdgeAD if ad else rayd.DfrCoherentEdge
        table = table_type()
        table.count = edge_count
        if edge_count <= 0:
            return table
        table.edge_index = _rayd_int(dr.arange(wt.UInt32, edge_count), ad=ad)
        table.edge_pos = _rayd_vector3(edge_data["pos"], ad=ad)
        table.edge_dir = _rayd_vector3(edge_data["edge_dir"], ad=ad)
        table.n0 = _rayd_vector3(edge_data["n0"], ad=ad)
        table.n_face_n = _rayd_vector3(edge_data["n_face_n"], ad=ad)
        table.wedge_n = _rayd_float(edge_data["wedge_n"], ad=ad)
        table.edge_line_min = _rayd_float(edge_data["line_min"], ad=ad)
        table.edge_line_max = _rayd_float(edge_data["line_max"], ad=ad)
        table.adjacent_face0 = _rayd_int(edge_data["adjacent_face0"], ad=ad)
        table.adjacent_face1 = _rayd_int(edge_data["adjacent_face1"], ad=ad)
        return table

    def build_dfr_coherent_tx_states(
        self,
        *,
        edge_data,
        tx_position,
        tx_polarization,
        wave,
        material_context,
        active=True,
    ):
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "build_dfr_coherent_tx_states", None)
        if trace is None:
            raise RuntimeError("RayD build_dfr_coherent_tx_states is required for native deterministic state prep.")
        if self._symbolic_recording_active():
            raise RuntimeError(
                "RayD build_dfr_coherent_tx_states uses native visibility and cannot be recorded inside a Dr.Jit symbolic scope."
            )
        ad = (
            point_grad_enabled(tx_position)
            or scene_geometry_grad_enabled(self)
            or scene_material_grad_enabled(self)
            or _grad_enabled_value(tx_polarization)
            or _grad_enabled_value(active)
        )
        if ad:
            raise RuntimeError("RayD coherent Tx state preparation does not support AD inputs yet.")
        edge_table = self._make_rayd_dfr_coherent_edge_table(edge_data, ad=False)
        edge_count = int(edge_table.count)
        if edge_count <= 0:
            return edge_table
        ignore_prim_ids = self._native_segment_ignore_ids(
            width=edge_count,
            ignore_prim_idx=(edge_data["adjacent_face0"], edge_data["adjacent_face1"]),
            ignore_surface_group_idx=None,
        )
        edge_table.ignore_prim_ids = _rayd_int(ignore_prim_ids, ad=False)
        edge_table.ignore_k = (
            0 if edge_count <= 0 else int(dr.width(ignore_prim_ids)) // edge_count
        )
        if edge_table.ignore_k > 0:
            self._warm_rayd_visibility_ignore_pipeline()

        tri_data = self._triangle_runtime()
        if tri_data is None:
            raise RuntimeError("RayD coherent Tx state preparation requires scene triangle material runtime data.")
        n_triangles = int(tri_data.get("n_triangles", 0))
        if n_triangles <= 0:
            raise RuntimeError("RayD coherent Tx state preparation requires at least one triangle.")

        rayd_material = rayd.DfrMaterial()
        rayd_material.eta_r = _rayd_float(tri_data["material_eps_r"], ad=False)
        rayd_material.sigma = _rayd_float(tri_data["material_sigma_e"], ad=False)
        rayd_material.mu_r = _rayd_float(tri_data["material_mu_r"], ad=False)
        rayd_material.gain = _rayd_float(
            dr.full(wt.Float, float(material_context.gain_scalar), n_triangles),
            ad=False,
        )
        rayd_material.valid = _rayd_bool(
            tri_data.get("material_specified", dr.full(wt.Bool, True, n_triangles)),
            ad=False,
        )

        options = rayd.DfrCoherentOptions()
        options.wavelength = float(scalar(wave.wavelength))
        options.k = float(scalar(wave.k))
        options.max_order = 1
        options.receiver_model = int(rayd.RAYD_DFR_MATCHED_ISO)
        options.select_diffraction_point = True
        options.prefilter_visibility = True
        options.collect_debug_counts = False
        if hasattr(options, "omega"):
            options.omega = float(material_angular_frequency(options.wavelength)[0])
        if hasattr(options, "tx_pol_x"):
            options.tx_pol_x = float(
                scalar(tx_polarization.x if hasattr(tx_polarization, "x") else tx_polarization[0])
            )
            options.tx_pol_y = float(
                scalar(tx_polarization.y if hasattr(tx_polarization, "y") else tx_polarization[1])
            )
            options.tx_pol_z = float(
                scalar(tx_polarization.z if hasattr(tx_polarization, "z") else tx_polarization[2])
            )

        active_mask = _rayd_bool(self._broadcast_bool(active, edge_count), ad=False)
        return _rayd_call(
            trace,
            edge_table,
            _rayd_vector3(tx_position, ad=False),
            rayd_material,
            options,
            active_mask,
            ad=False,
        )

    def build_dfr_coherent_higher_candidates(
        self,
        *,
        prev_state_arrays,
        edge_data,
        global_to_local_edge_index,
        prev_start: int,
        chunk_n_prev: int,
        filter_visibility: bool = False,
        active=True,
    ):
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "build_dfr_coherent_higher_candidates", None)
        if trace is None:
            raise RuntimeError(
                "RayD build_dfr_coherent_higher_candidates is required for native deterministic higher-order candidate prep."
            )
        if self._symbolic_recording_active():
            raise RuntimeError(
                "RayD build_dfr_coherent_higher_candidates uses native edge queries and cannot be recorded inside a Dr.Jit symbolic scope."
            )
        if (
            _dfr_states_grad_enabled(prev_state_arrays)
            or scene_geometry_grad_enabled(self)
            or _grad_enabled_value(global_to_local_edge_index)
            or _grad_enabled_value(active)
        ):
            raise RuntimeError(
                "RayD coherent higher-order candidate preparation does not support AD inputs yet."
            )

        chunk_n_prev = int(chunk_n_prev)
        prev_start = int(prev_start)
        if chunk_n_prev <= 0:
            pairs = rayd.DfrCoherentCandidatePairs()
            pairs.count = 0
            return pairs

        state_idx = dr.arange(wt.UInt32, chunk_n_prev) + wt.UInt32(prev_start)
        state_table = rayd.DfrCoherentUtdStates()
        state_table.count = chunk_n_prev
        state_table.edge_index = _rayd_int(
            dr.gather(wt.UInt32, prev_state_arrays["edge_idx"], state_idx),
            ad=False,
        )
        state_table.edge_pos = _rayd_vector3(
            dr.gather(wt.Point3f, prev_state_arrays["edge_pos"], state_idx),
            ad=False,
        )
        state_table.source_pos = _rayd_vector3(
            dr.gather(wt.Point3f, prev_state_arrays["source_pos"], state_idx),
            ad=False,
        )
        if filter_visibility:
            state_table.adjacent_face0 = _rayd_int(
                dr.gather(wt.Int32, prev_state_arrays["adjacent_face0"], state_idx),
                ad=False,
            )
            state_table.adjacent_face1 = _rayd_int(
                dr.gather(wt.Int32, prev_state_arrays["adjacent_face1"], state_idx),
                ad=False,
            )
        state_table.incident_basis_u = _rayd_vector3(
            dr.gather(wt.Vector3f, prev_state_arrays["incident_basis_u"], state_idx),
            ad=False,
        )
        state_table.incident_basis_v = _rayd_vector3(
            dr.gather(wt.Vector3f, prev_state_arrays["incident_basis_v"], state_idx),
            ad=False,
        )
        state_table.incident_basis_k = _rayd_vector3(
            dr.gather(wt.Vector3f, prev_state_arrays["incident_basis_k"], state_idx),
            ad=False,
        )

        edge_table = self._make_rayd_dfr_coherent_edge_table(edge_data, ad=False)
        options = rayd.DfrCoherentOptions()
        if hasattr(options, "higher_probe_radius_scale"):
            options.higher_probe_radius_scale = 0.6
            options.higher_probe_radius_min = 0.5
            options.higher_probe_radius_max = 4.0
        if hasattr(options, "higher_filter_visibility"):
            options.higher_filter_visibility = bool(filter_visibility)
        active_mask = _rayd_bool(self._broadcast_bool(active, chunk_n_prev), ad=False)
        dr.eval(
            state_table.edge_index,
            state_table.edge_pos,
            state_table.source_pos,
            state_table.incident_basis_u,
            state_table.incident_basis_v,
            state_table.incident_basis_k,
            state_table.adjacent_face0,
            state_table.adjacent_face1,
            edge_table.edge_index,
            global_to_local_edge_index,
            active_mask,
        )
        return trace(
            state_table,
            edge_table,
            _rayd_int(global_to_local_edge_index, ad=False),
            options,
            active_mask,
        )

    def accumulate_reflections(
        self,
        *,
        ray_origin,
        ray_dir,
        tx_pos,
        grid,
        config,
        max_bounces: int,
        seed: int,
        rr_depth: int,
        rr_prob: float,
        stop_threshold_linear: float,
        solid_angle_per_ray: float,
        cell_area: float,
        collect_wedges: bool,
        wedge_capacity: int,
        collect_wedge_prefixes: bool = False,
        wedge_sample_stride: int = 1,
        tx_polarization=None,
        active=True,
    ):
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "accumulate_reflections", None)
        if trace is None:
            raise RuntimeError(
                "RayD accumulate_reflections is required for "
                "accumulation_backend='rayd_reflection_accumulation'."
            )
        if self._symbolic_recording_active():
            raise RuntimeError(
                "RayD accumulate_reflections uses native optixLaunch and "
                "cannot be recorded inside a Dr.Jit symbolic scope."
            )

        width = int(dr.width(ray_dir.x))
        if width <= 0:
            raise ValueError("accumulate_reflections requires at least one ray.")

        ad = (
            point_grad_enabled(ray_origin)
            or point_grad_enabled(ray_dir)
            or point_grad_enabled(tx_pos)
            or scene_geometry_grad_enabled(self)
            or scene_material_grad_enabled(self)
            or _grad_enabled_value(tx_polarization)
            or _grad_enabled_value(active)
        )
        tx_polarization_value = (
            getattr(config, "tx_polarization", (1.0, 0.0, 0.0))
            if tx_polarization is None
            else tx_polarization
        )
        ray_cls = rayd.RayAD if ad else rayd.Ray
        ray = ray_cls(_rayd_vector3(ray_origin, ad=ad), _rayd_vector3(ray_dir, ad=ad))
        tx_position = _rayd_vector3(tx_pos, ad=ad)
        tx_polarization_rayd = _rayd_vector3(tx_polarization_value, ad=ad)

        grid_desc = rayd.AccumGrid()
        grid_desc.axis = {"x": 0, "y": 1, "z": 2}[str(grid.axis)]
        grid_desc.position = float(grid.position)
        grid_desc.coord0_min = float(grid.bounds[0][0])
        grid_desc.coord0_max = float(grid.bounds[0][1])
        grid_desc.coord1_min = float(grid.bounds[1][0])
        grid_desc.coord1_max = float(grid.bounds[1][1])
        grid_desc.resolution0 = int(grid.grid_shape[0])
        grid_desc.resolution1 = int(grid.grid_shape[1])

        tri_data = self._triangle_runtime()
        if tri_data is None:
            raise RuntimeError(
                "RayD reflection accumulation requires scene triangle material runtime data."
            )
        n_triangles = int(tri_data.get("n_triangles", 0))
        if n_triangles <= 0:
            raise RuntimeError("RayD reflection accumulation requires at least one triangle.")

        material = rayd.MaterialAD() if ad else rayd.Material()
        material.eta_r = _rayd_float(tri_data["material_eps_r"], ad=ad)
        material.sigma = _rayd_float(tri_data["material_sigma_e"], ad=ad)
        material.mu_r = _rayd_float(tri_data["material_mu_r"], ad=ad)
        material.gain = _rayd_float(dr.full(wt.Float, 1.0, n_triangles), ad=ad)
        material.valid = _rayd_bool(
            tri_data.get(
                "material_specified",
                dr.full(wt.Bool, True, n_triangles),
            ),
            ad=ad,
        )

        options = rayd.AccumOptions()
        options.wavelength = float(config.wavelength)
        options.k = float(config.k)
        options.solid_angle_per_ray = float(solid_angle_per_ray)
        options.cell_area = float(cell_area)
        options.seed = int(seed)
        options.rr_depth = int(rr_depth)
        options.rr_prob = float(rr_prob)
        options.stop_threshold = float(stop_threshold_linear)
        options.collect_wedges = bool(collect_wedges)
        options.collect_wedge_prefixes = bool(collect_wedge_prefixes)
        options.wedge_capacity = int(wedge_capacity)
        options.wedge_sample_stride = int(wedge_sample_stride)

        active_mask = _rayd_bool(self._broadcast_bool(active, width), ad=ad)
        return _rayd_call(
            trace,
            ray,
            tx_position,
            grid_desc,
            material,
            int(max_bounces),
            options,
            active_mask,
            tx_polarization_rayd,
            ad=ad,
        )

    def accum_dfr_direct(
        self,
        *,
        diffraction_states,
        grid,
        config,
        seed: int,
        samples: int,
        direct_samples: int,
        keller_samples: int,
        suffix_samples: int = 0,
        sample_sequence: str = "hash",
        active=True,
    ):
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "accum_dfr_direct", None)
        if trace is None:
            raise RuntimeError(
                "RayD accum_dfr_direct is required for "
                "diffraction_execution.accumulate_primal='rayd_optix'."
            )
        if self._symbolic_recording_active():
            raise RuntimeError(
                "RayD accum_dfr_direct uses native optixLaunch and "
                "cannot be recorded inside a Dr.Jit symbolic scope."
            )

        ad = (
            _dfr_states_grad_enabled(diffraction_states)
            or scene_geometry_grad_enabled(self)
            or scene_material_grad_enabled(self)
            or _grad_enabled_value(active)
        )
        state_table = self._make_rayd_dfr_states(diffraction_states, ad=ad)
        state_count = int(state_table.count)
        if state_count <= 0:
            raise ValueError("accum_dfr_direct requires at least one diffraction state.")

        grid_desc = rayd.DfrGrid()
        grid_desc.axis = {"x": 0, "y": 1, "z": 2}[str(grid.axis)]
        grid_desc.position = float(grid.position)
        grid_desc.coord0_min = float(grid.bounds[0][0])
        grid_desc.coord0_max = float(grid.bounds[0][1])
        grid_desc.coord1_min = float(grid.bounds[1][0])
        grid_desc.coord1_max = float(grid.bounds[1][1])
        grid_desc.resolution0 = int(grid.grid_shape[0])
        grid_desc.resolution1 = int(grid.grid_shape[1])
        grid_desc.cell_area = float(grid.cell_size[0] * grid.cell_size[1])

        tri_data = self._triangle_runtime()
        if tri_data is None:
            raise RuntimeError(
                "RayD diffraction accumulation requires scene triangle material runtime data."
            )
        n_triangles = int(tri_data.get("n_triangles", 0))
        if n_triangles <= 0:
            raise RuntimeError("RayD diffraction accumulation requires at least one triangle.")

        material = rayd.DfrMaterialAD() if ad else rayd.DfrMaterial()
        material.eta_r = _rayd_float(tri_data["material_eps_r"], ad=ad)
        material.sigma = _rayd_float(tri_data["material_sigma_e"], ad=ad)
        material.mu_r = _rayd_float(tri_data["material_mu_r"], ad=ad)
        material.gain = _rayd_float(dr.full(wt.Float, 1.0, n_triangles), ad=ad)
        material.valid = _rayd_bool(
            tri_data.get(
                "material_specified",
                dr.full(wt.Bool, True, n_triangles),
            ),
            ad=ad,
        )

        direct_count = max(0, int(direct_samples))
        keller_count = max(0, int(keller_samples))
        suffix_count = max(0, int(suffix_samples))
        strategy_mask = 0
        if direct_count > 0:
            strategy_mask |= int(rayd.RAYD_DFR_DIRECT)
        if keller_count > 0:
            strategy_mask |= int(rayd.RAYD_DFR_KELLER)
        if suffix_count > 0:
            strategy_mask |= int(rayd.RAYD_DFR_SUFFIX_REFL)
        if strategy_mask == 0:
            raise ValueError("accum_dfr_direct requires direct, Keller, or suffix samples.")

        sequence = str(sample_sequence)
        if sequence == "hash":
            sample_sequence_id = int(rayd.RAYD_DFR_HASH)
        elif sequence == "sobol":
            sample_sequence_id = int(rayd.RAYD_DFR_SOBOL)
        else:
            raise ValueError("sample_sequence must be 'hash' or 'sobol'.")

        options = rayd.DfrOptions()
        options.wavelength = float(config.wavelength)
        options.k = float(config.k)
        options.seed = int(seed)
        options.samples = int(samples)
        options.max_order = 1
        options.direct_samples = direct_count
        options.keller_samples = keller_count
        options.suffix_samples = suffix_count
        options.strategy_mask = strategy_mask
        options.sample_sequence = sample_sequence_id
        options.receiver_model = int(rayd.RAYD_DFR_MATCHED_ISO)
        options.collect_edge_use = True
        options.collect_debug_counts = True

        active_mask = _rayd_bool(self._broadcast_bool(active, state_count), ad=ad)
        dr.eval(
            state_table.edge_index,
            state_table.edge_pos,
            state_table.edge_dir,
            state_table.edge_t_min,
            state_table.edge_t_max,
            state_table.n0,
            state_table.n1,
            state_table.prim0,
            state_table.prim1,
            state_table.exterior_angle,
            state_table.src,
            state_table.src_power,
            state_table.wi,
            state_table.d0,
            state_table.prefix_depth,
            material.eta_r,
            material.sigma,
            material.mu_r,
            material.gain,
            material.valid,
            active_mask,
        )
        return _rayd_call(
            trace,
            state_table,
            grid_desc,
            material,
            options,
            active_mask,
            ad=ad,
        )

    def accum_dfr_coherent_direct(
        self,
        *,
        diffraction_states,
        grid,
        config,
        active=True,
        select_diffraction_point: bool = True,
        prefilter_visibility: bool = False,
        tx_polarization=None,
    ):
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "accum_dfr_coherent_direct", None)
        if trace is None:
            raise RuntimeError(
                "RayD accum_dfr_coherent_direct is required for "
                "diffraction_execution.accumulate_primal='rayd_exact_coherent'."
            )
        if self._symbolic_recording_active():
            raise RuntimeError(
                "RayD accum_dfr_coherent_direct uses native optixLaunch and "
                "cannot be recorded inside a Dr.Jit symbolic scope."
            )

        ad = (
            _dfr_states_grad_enabled(diffraction_states)
            or scene_geometry_grad_enabled(self)
            or scene_material_grad_enabled(self)
            or _grad_enabled_value(active)
        )
        if ad:
            raise RuntimeError(
                "RayD exact coherent diffraction accumulation does not support AD inputs yet."
            )

        state_table = self._make_rayd_dfr_coherent_utd_states_from_arrays(
            diffraction_states,
            ad=False,
        )
        state_count = int(state_table.count)
        if state_count <= 0:
            raise ValueError(
                "accum_dfr_coherent_direct requires at least one diffraction state."
            )

        grid_desc = rayd.DfrGrid()
        grid_desc.axis = {"x": 0, "y": 1, "z": 2}[str(grid.axis)]
        grid_desc.position = float(grid.position)
        grid_desc.coord0_min = float(grid.bounds[0][0])
        grid_desc.coord0_max = float(grid.bounds[0][1])
        grid_desc.coord1_min = float(grid.bounds[1][0])
        grid_desc.coord1_max = float(grid.bounds[1][1])
        grid_shape = getattr(grid, "grid_shape", getattr(grid, "size", None))
        if grid_shape is None:
            raise ValueError(
                "accum_dfr_coherent_direct grid must expose grid_shape or size."
            )
        grid_desc.resolution0 = int(grid_shape[0])
        grid_desc.resolution1 = int(grid_shape[1])
        grid_desc.cell_area = float(grid.cell_size[0] * grid.cell_size[1])

        tri_data = self._triangle_runtime()
        if tri_data is None:
            raise RuntimeError(
                "RayD exact coherent diffraction accumulation requires scene triangle material runtime data."
            )
        n_triangles = int(tri_data.get("n_triangles", 0))
        if n_triangles <= 0:
            raise RuntimeError(
                "RayD exact coherent diffraction accumulation requires at least one triangle."
            )

        material = rayd.DfrMaterial()
        material.eta_r = _rayd_float(tri_data["material_eps_r"], ad=False)
        material.sigma = _rayd_float(tri_data["material_sigma_e"], ad=False)
        material.mu_r = _rayd_float(tri_data["material_mu_r"], ad=False)
        material.gain = _rayd_float(dr.full(wt.Float, 1.0, n_triangles), ad=False)
        material.valid = _rayd_bool(
            tri_data.get(
                "material_specified",
                dr.full(wt.Bool, True, n_triangles),
            ),
            ad=False,
        )

        options = rayd.DfrCoherentOptions()
        options.wavelength = float(scalar(config.wavelength))
        options.k = float(scalar(config.k))
        options.max_order = 1
        options.receiver_model = int(rayd.RAYD_DFR_MATCHED_ISO)
        options.select_diffraction_point = bool(select_diffraction_point)
        options.prefilter_visibility = bool(prefilter_visibility)
        options.collect_debug_counts = True
        if hasattr(options, "omega"):
            options.omega = float(material_angular_frequency(options.wavelength)[0])
        if hasattr(options, "tx_pol_x"):
            active_tx_pol = (
                getattr(config, "tx_polarization", (1.0, 0.0, 0.0))
                if tx_polarization is None
                else tx_polarization
            )
            options.tx_pol_x = float(
                scalar(active_tx_pol.x if hasattr(active_tx_pol, "x") else active_tx_pol[0])
            )
            options.tx_pol_y = float(
                scalar(active_tx_pol.y if hasattr(active_tx_pol, "y") else active_tx_pol[1])
            )
            options.tx_pol_z = float(
                scalar(active_tx_pol.z if hasattr(active_tx_pol, "z") else active_tx_pol[2])
            )

        active_mask = _rayd_bool(self._broadcast_bool(active, state_count), ad=False)
        dr.eval(
            state_table.edge_pos,
            state_table.edge_dir,
            state_table.n0,
            state_table.n_face_n,
            state_table.wedge_n,
            state_table.edge_line_min,
            state_table.edge_line_max,
            state_table.source_pos,
            state_table.incident_field,
            state_table.incident_normal_derivative,
            state_table.r_face0,
            state_table.r_face_n,
            state_table.incident_vector_x,
            state_table.incident_vector_y,
            state_table.incident_vector_z,
            state_table.incident_normal_derivative_vector_x,
            state_table.incident_normal_derivative_vector_y,
            state_table.incident_normal_derivative_vector_z,
            state_table.incident_jones_u,
            state_table.incident_jones_v,
            state_table.incident_derivative_jones_u,
            state_table.incident_derivative_jones_v,
            state_table.incident_basis_u,
            state_table.incident_basis_v,
            state_table.incident_basis_k,
            state_table.face0_operator_m00,
            state_table.face0_operator_m01,
            state_table.face0_operator_m10,
            state_table.face0_operator_m11,
            state_table.face1_operator_m00,
            state_table.face1_operator_m01,
            state_table.face1_operator_m10,
            state_table.face1_operator_m11,
            state_table.face0_eta_r,
            state_table.face0_mu_r,
            state_table.face0_sigma,
            state_table.face0_gain,
            state_table.face0_use_fresnel,
            state_table.face1_eta_r,
            state_table.face1_mu_r,
            state_table.face1_sigma,
            state_table.face1_gain,
            state_table.face1_use_fresnel,
            state_table.select_stationary_point,
            state_table.owner_code,
            state_table.adjacent_face0,
            state_table.adjacent_face1,
            material.eta_r,
            material.sigma,
            material.mu_r,
            material.gain,
            material.valid,
            active_mask,
        )
        return _rayd_call(
            trace,
            state_table,
            grid_desc,
            material,
            options,
            active_mask,
            ad=False,
        )

    def accum_dfr(
        self,
        *,
        initial_states,
        recursive_states,
        grid,
        config,
        seed: int,
        samples: int,
        direct_samples: int,
        max_order: int,
        keller_samples: int = 0,
        suffix_samples: int = 0,
        sample_sequence: str = "hash",
        active=True,
    ):
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "accum_dfr", None)
        if trace is None:
            raise RuntimeError(
                "RayD accum_dfr is required for "
                "BDPT chain diffraction with diffraction_execution.accumulate_primal='rayd_optix'."
            )
        if self._symbolic_recording_active():
            raise RuntimeError(
                "RayD accum_dfr uses native optixLaunch and "
                "cannot be recorded inside a Dr.Jit symbolic scope."
            )
        order = int(max_order)
        if order not in (2, 3):
            raise ValueError("accum_dfr currently supports max_order 2 or 3 only.")

        ad = (
            _dfr_states_grad_enabled(initial_states)
            or _dfr_states_grad_enabled(recursive_states)
            or scene_geometry_grad_enabled(self)
            or scene_material_grad_enabled(self)
            or _grad_enabled_value(active)
        )
        initial_table = self._make_rayd_dfr_states(initial_states, ad=ad)
        recursive_table = self._make_rayd_dfr_states(recursive_states, ad=ad)
        initial_count = int(initial_table.count)
        recursive_count = int(recursive_table.count)
        if initial_count <= 0:
            raise ValueError("accum_dfr requires at least one initial state.")
        if recursive_count <= 0:
            raise ValueError("accum_dfr requires at least one recursive state.")

        direct_count = max(0, int(direct_samples))
        keller_count = max(0, int(keller_samples))
        suffix_count = max(0, int(suffix_samples))
        if direct_count <= 0 and keller_count <= 0 and suffix_count <= 0:
            raise ValueError("accum_dfr requires direct, Keller, or suffix samples.")

        grid_desc = rayd.DfrGrid()
        grid_desc.axis = {"x": 0, "y": 1, "z": 2}[str(grid.axis)]
        grid_desc.position = float(grid.position)
        grid_desc.coord0_min = float(grid.bounds[0][0])
        grid_desc.coord0_max = float(grid.bounds[0][1])
        grid_desc.coord1_min = float(grid.bounds[1][0])
        grid_desc.coord1_max = float(grid.bounds[1][1])
        grid_desc.resolution0 = int(grid.grid_shape[0])
        grid_desc.resolution1 = int(grid.grid_shape[1])
        grid_desc.cell_area = float(grid.cell_size[0] * grid.cell_size[1])

        tri_data = self._triangle_runtime()
        if tri_data is None:
            raise RuntimeError(
                "RayD diffraction chain accumulation requires scene triangle material runtime data."
            )
        n_triangles = int(tri_data.get("n_triangles", 0))
        if n_triangles <= 0:
            raise RuntimeError("RayD diffraction chain accumulation requires at least one triangle.")

        material = rayd.DfrMaterialAD() if ad else rayd.DfrMaterial()
        material.eta_r = _rayd_float(tri_data["material_eps_r"], ad=ad)
        material.sigma = _rayd_float(tri_data["material_sigma_e"], ad=ad)
        material.mu_r = _rayd_float(tri_data["material_mu_r"], ad=ad)
        material.gain = _rayd_float(dr.full(wt.Float, 1.0, n_triangles), ad=ad)
        material.valid = _rayd_bool(
            tri_data.get(
                "material_specified",
                dr.full(wt.Bool, True, n_triangles),
            ),
            ad=ad,
        )

        sequence = str(sample_sequence)
        if sequence == "hash":
            sample_sequence_id = int(rayd.RAYD_DFR_HASH)
        elif sequence == "sobol":
            sample_sequence_id = int(rayd.RAYD_DFR_SOBOL)
        else:
            raise ValueError("sample_sequence must be 'hash' or 'sobol'.")

        options = rayd.DfrOptions()
        options.wavelength = float(config.wavelength)
        options.k = float(config.k)
        options.seed = int(seed)
        options.samples = int(samples)
        options.max_order = order
        options.direct_samples = direct_count
        options.keller_samples = keller_count
        options.suffix_samples = suffix_count
        options.strategy_mask = 0
        if direct_count > 0:
            options.strategy_mask |= int(rayd.RAYD_DFR_DIRECT)
        if keller_count > 0:
            options.strategy_mask |= int(rayd.RAYD_DFR_KELLER)
        if suffix_count > 0:
            options.strategy_mask |= int(rayd.RAYD_DFR_SUFFIX_REFL)
        options.sample_sequence = sample_sequence_id
        options.receiver_model = int(rayd.RAYD_DFR_MATCHED_ISO)
        options.collect_edge_use = True
        options.collect_debug_counts = True

        active_mask = _rayd_bool(self._broadcast_bool(active, initial_count), ad=ad)
        dr.eval(
            initial_table.edge_index,
            initial_table.edge_pos,
            initial_table.edge_dir,
            initial_table.edge_t_min,
            initial_table.edge_t_max,
            initial_table.n0,
            initial_table.n1,
            initial_table.prim0,
            initial_table.prim1,
            initial_table.exterior_angle,
            initial_table.src,
            initial_table.src_power,
            initial_table.wi,
            initial_table.d0,
            initial_table.prefix_depth,
            recursive_table.edge_index,
            recursive_table.edge_pos,
            recursive_table.edge_dir,
            recursive_table.edge_t_min,
            recursive_table.edge_t_max,
            recursive_table.n0,
            recursive_table.n1,
            recursive_table.prim0,
            recursive_table.prim1,
            recursive_table.exterior_angle,
            recursive_table.src,
            recursive_table.src_power,
            recursive_table.wi,
            recursive_table.d0,
            recursive_table.prefix_depth,
            material.eta_r,
            material.sigma,
            material.mu_r,
            material.gain,
            material.valid,
            active_mask,
        )
        return _rayd_call(
            trace,
            initial_table,
            recursive_table,
            grid_desc,
            material,
            options,
            active_mask,
            ad=ad,
        )

    def trace_dfr_paths(
        self,
        *,
        tx_positions,
        rx_positions,
        state_arrays,
        config,
        max_order: int,
        max_paths: int,
        seed: int,
        return_geometry: bool,
        active=True,
    ):
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "trace_dfr_paths", None)
        if trace is None:
            raise RuntimeError(
                "RayD trace_dfr_paths is required for "
                "path diffraction with diffraction_execution.accumulate_primal='rayd_optix'."
            )
        if self._symbolic_recording_active():
            raise RuntimeError(
                "RayD trace_dfr_paths uses native optixLaunch and "
                "cannot be recorded inside a Dr.Jit symbolic scope."
            )
        if int(max_order) != 1:
            raise RuntimeError(
                "RayD trace_dfr_paths currently supports first-order diffraction only."
            )
        ad = (
            point_grad_enabled(tx_positions)
            or point_grad_enabled(rx_positions)
            or _dfr_states_grad_enabled(state_arrays)
            or scene_geometry_grad_enabled(self)
            or scene_material_grad_enabled(self)
            or _grad_enabled_value(active)
        )
        state_table = self._make_rayd_dfr_states(state_arrays, ad=ad)
        state_count = int(state_table.count)
        if state_count <= 0:
            raise ValueError("trace_dfr_paths requires at least one diffraction state.")
        if int(max_paths) <= 0:
            raise ValueError("trace_dfr_paths requires max_paths > 0.")

        tri_data = self._triangle_runtime()
        if tri_data is None:
            raise RuntimeError(
                "RayD diffraction path export requires scene triangle material runtime data."
            )
        n_triangles = int(tri_data.get("n_triangles", 0))
        if n_triangles <= 0:
            raise RuntimeError("RayD diffraction path export requires at least one triangle.")

        material = rayd.DfrMaterialAD() if ad else rayd.DfrMaterial()
        material.eta_r = _rayd_float(tri_data["material_eps_r"], ad=ad)
        material.sigma = _rayd_float(tri_data["material_sigma_e"], ad=ad)
        material.mu_r = _rayd_float(tri_data["material_mu_r"], ad=ad)
        material.gain = _rayd_float(dr.full(wt.Float, 1.0, n_triangles), ad=ad)
        material.valid = _rayd_bool(
            tri_data.get(
                "material_specified",
                dr.full(wt.Bool, True, n_triangles),
            ),
            ad=ad,
        )

        options = rayd.DfrPathOptions()
        options.wavelength = float(config.wavelength)
        options.k = float(config.k)
        options.seed = int(seed)
        options.max_order = 1
        options.max_paths = int(max_paths)
        options.max_rx = int(dr.width(rx_positions.x))
        options.strategy_mask = int(rayd.RAYD_DFR_DIRECT)
        options.sample_count = int(max_paths)
        options.return_geom = 1 if bool(return_geometry) else 0
        options.receiver_model = int(rayd.RAYD_DFR_MATCHED_ISO)

        active_mask = _rayd_bool(self._broadcast_bool(active, state_count), ad=ad)
        tx_positions_rayd = _rayd_vector3(tx_positions, ad=ad)
        rx_positions_rayd = _rayd_vector3(rx_positions, ad=ad)
        dr.eval(
            tx_positions_rayd,
            rx_positions_rayd,
            state_table.edge_index,
            state_table.edge_pos,
            state_table.edge_dir,
            state_table.edge_t_min,
            state_table.edge_t_max,
            state_table.n0,
            state_table.n1,
            state_table.prim0,
            state_table.prim1,
            state_table.exterior_angle,
            state_table.src,
            state_table.src_power,
            state_table.wi,
            state_table.d0,
            state_table.prefix_depth,
            material.eta_r,
            material.sigma,
            material.mu_r,
            material.gain,
            material.valid,
            active_mask,
        )
        return _rayd_call(
            trace,
            tx_positions_rayd,
            rx_positions_rayd,
            state_table,
            material,
            options,
            active_mask,
            ad=ad,
        )

    # ------------------------------------------------------------------
    # Diffraction-path queries: triangle / ray / visibility / edge helpers
    # ------------------------------------------------------------------

    def triangle_group_id(self, prim_idx_i32) -> wt.Int32:
        tri_data = self._triangle_runtime()
        valid_prim = prim_idx_i32 >= 0
        if tri_data is None or "surface_group_id" not in tri_data:
            return dr.select(valid_prim, prim_idx_i32, wt.Int32(-1))
        safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))
        return dr.select(
            valid_prim,
            dr.gather(wt.Int32, tri_data["surface_group_id"], safe_prim_idx),
            wt.Int32(-1),
        )

    def triangle_canonical_prim(self, prim_idx_i32) -> wt.Int32:
        tri_data = self._triangle_runtime()
        valid_prim = prim_idx_i32 >= 0
        if tri_data is None or "surface_canonical_prim" not in tri_data:
            return dr.select(valid_prim, prim_idx_i32, wt.Int32(-1))
        safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))
        return dr.select(
            valid_prim,
            dr.gather(wt.Int32, tri_data["surface_canonical_prim"], safe_prim_idx),
            wt.Int32(-1),
        )

    def triangle_contains_point(self, point, prim_idx_i32):
        tri_data = self._triangle_runtime()
        valid_prim = prim_idx_i32 >= 0
        if tri_data is None:
            return valid_prim, dr.select(valid_prim, prim_idx_i32, wt.Int32(-1))

        safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))
        max_group_size = int(tri_data.get("surface_max_group_size", 0))
        if max_group_size <= 0 or "surface_group_members" not in tri_data:
            v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_prim_idx)
            v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_prim_idx)
            v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_prim_idx)
            hit = valid_prim & point_in_triangle_3d(point, v0, v1, v2)
            return hit, self.triangle_canonical_prim(prim_idx_i32)

        group_size = dr.gather(wt.UInt32, tri_data["surface_group_size"], safe_prim_idx)
        surface_hit = dr.zeros(wt.Bool, dr.width(point.x))
        for slot in range(max_group_size):
            slot_active = valid_prim & (group_size > wt.UInt32(slot))
            flat_idx = safe_prim_idx * wt.UInt32(max_group_size) + wt.UInt32(slot)
            member_idx_i32 = dr.gather(wt.Int32, tri_data["surface_group_members"], flat_idx)
            slot_active = slot_active & (member_idx_i32 >= 0)
            safe_member_idx = wt.UInt32(dr.select(slot_active, member_idx_i32, wt.Int32(0)))
            v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_member_idx)
            v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_member_idx)
            v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_member_idx)
            surface_hit = surface_hit | (slot_active & point_in_triangle_3d(point, v0, v1, v2))

        return surface_hit, self.triangle_canonical_prim(prim_idx_i32)

    def intersect_rays_raw_with_prim(self, ray_origin, ray_dir, active, *, tmax=None):
        rayd_scene = self._require_rayd_scene()
        ray = rayd.RayAD(ray_origin, ray_dir)
        if tmax is not None:
            ray.tmax = tmax
        with dr.suspend_grad():
            raw = rayd_scene.intersect(ray, active=active, flags=getattr(rayd.RayFlags, "None"))
            hit = raw.is_valid() & active
            blocker_dist = dr.select(hit, wt.Float(raw.t), wt.Float(1e10))
            prim_idx_i32 = wt.Int32(raw.global_prim_id)
        return hit, blocker_dist, wt.UInt32(dr.select(hit, prim_idx_i32, wt.Int32(-1)))

    def intersect_rays_with_prim(self, ray_origin, ray_dir, active):
        tri_data = self._triangle_runtime()
        ray = rayd.RayAD(ray_origin, ray_dir)
        with dr.suspend_grad():
            flags = rayd.RayFlags.All if tri_data is None else getattr(rayd.RayFlags, "None")
            si = self.ray_intersect(ray, active=active, flags=flags)
            hit = si.is_valid() & active
            blocker_dist = dr.select(hit, si.t, wt.Float(1e10))
        prim_idx_i32 = wt.Int32(si.global_prim_id)
        resolved_prim_idx_i32 = dr.select(hit, prim_idx_i32, wt.Int32(-1))
        triangle_count = max(1, int(dr.width(tri_data["v0"].x))) if tri_data is not None else 1
        tri_count_i32 = wt.Int32(triangle_count)
        max_prim_i32 = wt.Int32(triangle_count - 1)
        prim_idx_valid = hit & (prim_idx_i32 >= 0) & (prim_idx_i32 < tri_count_i32)
        clamped_prim_idx_i32 = dr.minimum(dr.maximum(prim_idx_i32, wt.Int32(0)), max_prim_i32)
        safe_prim_idx = wt.UInt32(clamped_prim_idx_i32)

        if tri_data is None:
            return hit, blocker_dist, si.p, si.n, wt.UInt32(resolved_prim_idx_i32)

        v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_prim_idx)
        v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_prim_idx)
        v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_prim_idx)
        recon_n = dr.cross(v1 - v0, v2 - v0)
        recon_n = recon_n / (dr.norm(recon_n) + EPS)
        denom = dr.dot(ray_dir, recon_n)
        t_hit = dr.dot(v0 - ray_origin, recon_n) / (denom + EPS)
        blocker_dist = dr.select(prim_idx_valid, t_hit, blocker_dist)
        hit_p = dr.select(prim_idx_valid, ray_origin + t_hit * ray_dir, si.p)
        geom_n = dr.select(prim_idx_valid, recon_n, si.n)

        surface_hit, resolved_prim_idx = self.triangle_contains_point(hit_p, resolved_prim_idx_i32)
        hit = hit & surface_hit
        blocker_dist = dr.select(hit, blocker_dist, wt.Float(1e10))
        return hit, blocker_dist, hit_p, geom_n, wt.UInt32(dr.select(hit, resolved_prim_idx, wt.Int32(-1)))

    def _broadcast_ignore_i32(self, value, width: int) -> wt.Int32:
        value_i32 = wt.Int32(value)
        value_width = int(dr.width(value_i32))
        if value_width == width:
            return value_i32
        if value_width == 1:
            return broadcast(value_i32, width)
        raise ValueError(f"Expected scalar or width {width} Int32 input, got width {value_width}.")

    @staticmethod
    def _broadcast_bool(value, width: int):
        value_bool = wt.Bool(value)
        value_width = int(dr.width(value_bool))
        if value_width == width:
            return value_bool
        if value_width == 1:
            return dr.repeat(value_bool, width)
        raise ValueError(f"Expected scalar or width {width} Bool input, got width {value_width}.")

    @staticmethod
    def _symbolic_recording_active() -> bool:
        return bool(dr.flag(dr.JitFlag.Recording)) or bool(dr.flag(dr.JitFlag.SymbolicScope))

    @staticmethod
    def _detached_int32(value):
        int_type = dr.detached_t(wt.Int32)
        return int_type(dr.detach(wt.Int32(value)))

    @staticmethod
    def _ignore_candidates(value):
        return value if isinstance(value, (tuple, list)) else (value,)

    def _native_segment_ignore_slot_count(self, ignore_prim_idx, ignore_surface_group_idx) -> int:
        tri_data = self._triangle_runtime()
        max_group_size = 0 if tri_data is None else int(tri_data.get("surface_max_group_size", 0))
        group_count = 0 if tri_data is None else int(tri_data.get("surface_group_count", 0))
        slot_count = 0
        for surface_group_idx in self._ignore_candidates(ignore_surface_group_idx):
            if surface_group_idx is None:
                continue
            slot_count += max_group_size if max_group_size > 1 and group_count > 0 else 1
        for prim_idx in self._ignore_candidates(ignore_prim_idx):
            if prim_idx is None:
                continue
            slot_count += max_group_size if max_group_size > 1 else 1
        return slot_count

    @classmethod
    def _interleave_native_ignore_slots(cls, slots: list[wt.Int32], width: int):
        if len(slots) == 0 or width <= 0:
            return cls._detached_int32(dr.zeros(wt.Int32, 0))
        if len(slots) == 1:
            return cls._detached_int32(slots[0])

        slot_count = len(slots)
        slot_major = dr.concat(slots)
        dst_idx = dr.arange(wt.UInt32, int(width) * slot_count)
        ray_idx = dst_idx // wt.UInt32(slot_count)
        slot_idx = dst_idx % wt.UInt32(slot_count)
        src_idx = slot_idx * wt.UInt32(width) + ray_idx
        return cls._detached_int32(dr.gather(wt.Int32, slot_major, src_idx))

    @staticmethod
    def _interleave_native_chain_points(point_slots: list[wt.Point3f], chain_count: int):
        max_points = len(point_slots)
        slot_major = concat_points(point_slots)
        dst_idx = dr.arange(wt.UInt32, int(chain_count) * max_points)
        chain_idx = dst_idx // wt.UInt32(max_points)
        point_idx = dst_idx % wt.UInt32(max_points)
        src_idx = point_idx * wt.UInt32(chain_count) + chain_idx
        return wt.Point3f(
            dr.gather(wt.Float, slot_major.x, src_idx),
            dr.gather(wt.Float, slot_major.y, src_idx),
            dr.gather(wt.Float, slot_major.z, src_idx),
        )

    @staticmethod
    def _append_native_prim_ignore_slots(slots: list[wt.Int32], tri_data, prim_idx_i32, has_ignore) -> None:
        n_triangles = 0 if tri_data is None else int(tri_data.get("n_triangles", 0))
        valid_prim = has_ignore & (prim_idx_i32 >= 0) & (prim_idx_i32 < wt.Int32(n_triangles))
        max_group_size = 0 if tri_data is None else int(tri_data.get("surface_max_group_size", 0))
        if (
            tri_data is None
            or max_group_size <= 1
            or "surface_group_size" not in tri_data
            or "surface_group_members" not in tri_data
        ):
            slots.append(dr.select(valid_prim, prim_idx_i32, wt.Int32(-1)))
            return

        safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))
        group_size = dr.gather(wt.UInt32, tri_data["surface_group_size"], safe_prim_idx)
        for slot in range(max_group_size):
            slot_active = valid_prim & (group_size > wt.UInt32(slot))
            flat_idx = safe_prim_idx * wt.UInt32(max_group_size) + wt.UInt32(slot)
            member_idx = dr.gather(wt.Int32, tri_data["surface_group_members"], flat_idx)
            slots.append(dr.select(slot_active & (member_idx >= 0), member_idx, wt.Int32(-1)))

    @staticmethod
    def _append_native_group_ignore_slots(slots: list[wt.Int32], tri_data, group_idx_i32, has_ignore) -> None:
        group_count = 0 if tri_data is None else int(tri_data.get("surface_group_count", 0))
        max_group_size = 0 if tri_data is None else int(tri_data.get("surface_max_group_size", 0))
        if (
            tri_data is None
            or max_group_size <= 1
            or group_count <= 0
            or "surface_group_size_by_group" not in tri_data
            or "surface_group_members_by_group" not in tri_data
        ):
            slots.append(dr.select(has_ignore, group_idx_i32, wt.Int32(-1)))
            return

        valid_group = has_ignore & (group_idx_i32 >= 0) & (group_idx_i32 < wt.Int32(group_count))
        safe_group_idx = wt.UInt32(dr.select(valid_group, group_idx_i32, wt.Int32(0)))
        group_size = dr.gather(wt.UInt32, tri_data["surface_group_size_by_group"], safe_group_idx)
        for slot in range(max_group_size):
            slot_active = valid_group & (group_size > wt.UInt32(slot))
            flat_idx = safe_group_idx * wt.UInt32(max_group_size) + wt.UInt32(slot)
            member_idx = dr.gather(wt.Int32, tri_data["surface_group_members_by_group"], flat_idx)
            slots.append(dr.select(slot_active & (member_idx >= 0), member_idx, wt.Int32(-1)))

    def _native_segment_ignore_ids(
        self,
        *,
        width: int,
        ignore_prim_idx,
        ignore_surface_group_idx,
    ):
        tri_data = self._triangle_runtime()
        slots: list[wt.Int32] = []
        for surface_group_idx in self._ignore_candidates(ignore_surface_group_idx):
            if surface_group_idx is None:
                continue
            group_i32 = self._broadcast_ignore_i32(surface_group_idx, width)
            self._append_native_group_ignore_slots(slots, tri_data, group_i32, group_i32 >= 0)

        for prim_idx in self._ignore_candidates(ignore_prim_idx):
            if prim_idx is None:
                continue
            prim_idx_i32 = self._broadcast_ignore_i32(prim_idx, width)
            self._append_native_prim_ignore_slots(slots, tri_data, prim_idx_i32, prim_idx_i32 >= 0)

        return self._interleave_native_ignore_slots(slots, width)

    @staticmethod
    def _slice_point(point, idx):
        return wt.Point3f(
            dr.gather(wt.Float, point.x, idx),
            dr.gather(wt.Float, point.y, idx),
            dr.gather(wt.Float, point.z, idx),
        )

    @staticmethod
    def _slice_bool(value, idx):
        return dr.gather(wt.Bool, wt.Bool(value), idx)

    def _slice_ignore_value(self, value, idx, width: int):
        if isinstance(value, (tuple, list)):
            return tuple(self._slice_ignore_value(item, idx, width) for item in value)
        if value is None:
            return None
        value_i32 = wt.Int32(value)
        value_width = int(dr.width(value_i32))
        if value_width == width:
            return dr.gather(wt.Int32, value_i32, idx)
        if value_width == 1:
            return value_i32
        raise ValueError(f"Expected scalar or width {width} Int32 input, got width {value_width}.")

    def _segment_visible_chunked(
        self,
        start_pos,
        end_pos,
        active,
        *,
        ignore_prim_idx,
        ignore_surface_group_idx,
        slot_count: int,
        width: int,
    ) -> wt.Bool:
        chunk_width = max(1, int(_NATIVE_IGNORE_MAX_ENTRIES) // max(1, int(slot_count)))
        chunks = []
        for offset in range(0, int(width), chunk_width):
            count = min(chunk_width, int(width) - offset)
            idx = dr.arange(wt.UInt32, count) + wt.UInt32(offset)
            native_ignore_ids = self._native_segment_ignore_ids(
                width=count,
                ignore_prim_idx=self._slice_ignore_value(ignore_prim_idx, idx, width),
                ignore_surface_group_idx=self._slice_ignore_value(ignore_surface_group_idx, idx, width),
            )
            chunks.append(
                self._visible_rayd(
                    self._slice_point(start_pos, idx),
                    self._slice_point(end_pos, idx),
                    self._slice_bool(active, idx),
                    native_ignore_ids,
                )
            )
        return dr.concat(chunks) if len(chunks) > 1 else chunks[0]

    @staticmethod
    def _per_segment_ignore_value(value, segment: int, max_segments: int, name: str):
        if value is None:
            return None
        if not isinstance(value, (tuple, list)) or len(value) != max_segments:
            raise ValueError(f"{name} must provide exactly {max_segments} per-segment entries.")
        return value[segment]

    def _native_segment_chain_ignore_ids(
        self,
        *,
        chain_count: int,
        max_segments: int,
        ignore_prim_idx_per_segment,
        ignore_surface_group_idx_per_segment,
    ):
        tri_data = self._triangle_runtime()
        segment_slots: list[list[wt.Int32]] = []
        max_slot_count = 0
        for segment in range(max_segments):
            slots: list[wt.Int32] = []
            surface_value = self._per_segment_ignore_value(
                ignore_surface_group_idx_per_segment,
                segment,
                max_segments,
                "ignore_surface_group_idx_per_segment",
            )
            for surface_group_idx in self._ignore_candidates(surface_value):
                if surface_group_idx is None:
                    continue
                group_i32 = self._broadcast_ignore_i32(surface_group_idx, chain_count)
                self._append_native_group_ignore_slots(slots, tri_data, group_i32, group_i32 >= 0)

            prim_value = self._per_segment_ignore_value(
                ignore_prim_idx_per_segment,
                segment,
                max_segments,
                "ignore_prim_idx_per_segment",
            )
            for prim_idx in self._ignore_candidates(prim_value):
                if prim_idx is None:
                    continue
                prim_idx_i32 = self._broadcast_ignore_i32(prim_idx, chain_count)
                self._append_native_prim_ignore_slots(slots, tri_data, prim_idx_i32, prim_idx_i32 >= 0)

            segment_slots.append(slots)
            max_slot_count = max(max_slot_count, len(slots))

        if max_slot_count == 0 or chain_count <= 0:
            return self._detached_int32(dr.zeros(wt.Int32, 0))

        invalid = dr.full(wt.Int32, -1, chain_count)
        segment_slot_major: list[wt.Int32] = []
        for slots in segment_slots:
            segment_slot_major.extend(slots)
            segment_slot_major.extend(invalid for _ in range(max_slot_count - len(slots)))

        slot_major = dr.concat(segment_slot_major)
        dst_idx = dr.arange(wt.UInt32, int(chain_count) * max_segments * max_slot_count)
        segment_stride = wt.UInt32(max_segments * max_slot_count)
        chain_idx = dst_idx // segment_stride
        segment_idx = (dst_idx // wt.UInt32(max_slot_count)) % wt.UInt32(max_segments)
        slot_idx = dst_idx % wt.UInt32(max_slot_count)
        src_idx = (
            segment_idx * wt.UInt32(max_slot_count * chain_count)
            + slot_idx * wt.UInt32(chain_count)
            + chain_idx
        )
        return self._detached_int32(dr.gather(wt.Int32, slot_major, src_idx))

    @staticmethod
    def _coerce_chain_point_slots(point_slots) -> list[wt.Point3f]:
        if not isinstance(point_slots, (tuple, list)):
            raise TypeError("segment_chain_visible expects a tuple/list of chain point slots.")
        if len(point_slots) < 2:
            raise ValueError("segment_chain_visible requires at least two point slots.")
        coerced = []
        for point in point_slots:
            if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
                coerced.append(point)
            else:
                coerced.append(wt.Point3f(point))
        return coerced

    def _normalize_chain_point_slots(self, point_slots) -> tuple[list[wt.Point3f], int]:
        coerced = self._coerce_chain_point_slots(point_slots)
        chain_count = max(int(dr.width(point.x)) for point in coerced)
        if chain_count <= 0:
            return coerced, 0

        normalized = []
        for point in coerced:
            point_width = int(dr.width(point.x))
            if point_width == chain_count:
                normalized.append(point)
            elif point_width == 1:
                normalized.append(broadcast(point, chain_count))
            else:
                raise ValueError(f"Expected point slots to have width 1 or {chain_count}, got {point_width}.")
        return normalized, chain_count

    def _visible_rayd(self, start_pos, end_pos, active, ignore_prim_ids=None):
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "visible", None)
        if trace is None:
            raise RuntimeError("RayD visible is required for segment_visible().")
        has_ignores = ignore_prim_ids is not None and int(dr.width(ignore_prim_ids)) > 0
        if has_ignores and self._symbolic_recording_active():
            raise RuntimeError(
                "RayD visible with primitive ignores uses native optixLaunch and cannot be "
                "recorded inside a Dr.Jit symbolic scope."
            )
        with dr.suspend_grad():
            if not has_ignores:
                result = trace(start_pos, end_pos, active=active)
            else:
                result = trace(start_pos, end_pos, ignore_prim_ids, active)
        return active & wt.Bool(result.visible)

    def _warm_rayd_visibility_ignore_pipeline(self) -> None:
        if self._rayd_visibility_ignore_pipeline_warmed:
            return
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "visible", None)
        if trace is None:
            raise RuntimeError("RayD visible is required for native visibility ignores.")

        float_type = dr.detached_t(wt.Float)
        bool_type = dr.detached_t(wt.Bool)
        int_type = dr.detached_t(wt.Int32)
        point_type = dr.detached_t(wt.Point3f)
        start = point_type(float_type([0.0]), float_type([0.0]), float_type([0.0]))
        end = point_type(float_type([0.0]), float_type([0.0]), float_type([1.0]))
        with dr.suspend_grad():
            result = trace(start, end, int_type([-1]), bool_type([False]))
            dr.eval(result.visible)
        self._rayd_visibility_ignore_pipeline_warmed = True

    def _visible_chain_rayd(self, points, chain_length, active, ignore_prim_ids=None):
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "visible_chain", None)
        if trace is None:
            raise RuntimeError("RayD visible_chain is required for segment_chain_visible().")
        has_ignores = ignore_prim_ids is not None and int(dr.width(ignore_prim_ids)) > 0
        if has_ignores and self._symbolic_recording_active():
            raise RuntimeError(
                "RayD visible_chain with primitive ignores uses native optixLaunch and cannot be "
                "recorded inside a Dr.Jit symbolic scope."
            )
        with dr.suspend_grad():
            if not has_ignores:
                result = trace(points, chain_length, active=active)
            else:
                result = trace(points, chain_length, ignore_prim_ids, active)
        return active & wt.Bool(result.all_visible)

    def axial_edge_visible(
        self,
        source_pos,
        edge_pos,
        edge_dir,
        edge_line_min,
        edge_line_max,
        sample_fractions,
        *,
        active=True,
    ) -> wt.Bool:
        width = int(dr.width(edge_pos.x))
        if width <= 0:
            return dr.zeros(wt.Bool, 0)
        source_pos_b = broadcast(source_pos, width)
        edge_dir_b = broadcast(edge_dir, width)
        edge_line_min_b = broadcast(wt.Float(edge_line_min), width)
        edge_line_max_b = broadcast(wt.Float(edge_line_max), width)
        active_mask = (
            self._broadcast_bool(active, width)
            & (edge_line_max_b >= edge_line_min_b)
            & (dr.norm(edge_dir_b) > wt.Float(EPS))
        )
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "visible_edge", None)
        if trace is None:
            raise RuntimeError("RayD visible_edge is required for axial_edge_visible().")
        with dr.suspend_grad():
            result = trace(
                source_pos_b,
                edge_pos,
                edge_dir_b,
                edge_line_min_b,
                edge_line_max_b,
                tuple(float(v) for v in sample_fractions),
                active_mask,
            )
        return active_mask & wt.Bool(result.any_visible)

    def segment_visible(self, start_pos, end_pos, *, ignore_prim_idx=None,
                        ignore_surface_group_idx=None, ignore_structure_idx=None,
                        max_ignored_hits: int = 4) -> wt.Bool:
        del max_ignored_hits
        width = dr.width(end_pos.x)
        start_pos_b = broadcast(start_pos, width)
        seg_vec = end_pos - start_pos_b
        seg_len = dr.norm(seg_vec)
        min_seg_len = wt.Float(2.0 * RAY_ORIGIN_BIAS + EPS)
        active = seg_len > min_seg_len

        if ignore_prim_idx is None and ignore_surface_group_idx is None and ignore_structure_idx is None:
            return self._visible_rayd(start_pos_b, end_pos, active)

        ignore_structure_candidates = self._ignore_candidates(ignore_structure_idx)
        for structure_idx in ignore_structure_candidates:
            if structure_idx is not None:
                raise ValueError(
                    "segment_visible(ignore_structure_idx=...) is not supported by RayD visible; "
                    "pass primitive or surface-group ignores instead."
                )

        self._warm_rayd_visibility_ignore_pipeline()
        slot_count = self._native_segment_ignore_slot_count(
            ignore_prim_idx=ignore_prim_idx,
            ignore_surface_group_idx=ignore_surface_group_idx,
        )
        if slot_count > 0 and int(width) * int(slot_count) > _NATIVE_IGNORE_MAX_ENTRIES:
            return self._segment_visible_chunked(
                start_pos_b,
                end_pos,
                active,
                ignore_prim_idx=ignore_prim_idx,
                ignore_surface_group_idx=ignore_surface_group_idx,
                slot_count=slot_count,
                width=width,
            )
        native_ignore_ids = self._native_segment_ignore_ids(
            width=width,
            ignore_prim_idx=ignore_prim_idx,
            ignore_surface_group_idx=ignore_surface_group_idx,
        )
        return self._visible_rayd(start_pos_b, end_pos, active, native_ignore_ids)

    def segment_chain_visible(
        self,
        point_slots,
        *,
        chain_length=None,
        ignore_prim_idx_per_segment=None,
        ignore_surface_group_idx_per_segment=None,
        active=True,
    ) -> wt.Bool:
        normalized_points, chain_count = self._normalize_chain_point_slots(point_slots)
        if chain_count <= 0:
            return dr.zeros(wt.Bool, 0)
        max_segments = len(normalized_points) - 1
        chain_length_i32 = (
            dr.full(wt.Int32, max_segments, chain_count)
            if chain_length is None
            else self._broadcast_ignore_i32(chain_length, chain_count)
        )
        active_mask = self._broadcast_bool(active, chain_count) & (chain_length_i32 >= 0)

        min_seg_len = wt.Float(2.0 * RAY_ORIGIN_BIAS + EPS)
        length_active = active_mask
        for segment in range(max_segments):
            segment_active = chain_length_i32 > wt.Int32(segment)
            segment_len = dr.norm(normalized_points[segment + 1] - normalized_points[segment])
            length_active &= (~segment_active) | (segment_len > min_seg_len)

        native_ignore_ids = self._native_segment_chain_ignore_ids(
            chain_count=chain_count,
            max_segments=max_segments,
            ignore_prim_idx_per_segment=ignore_prim_idx_per_segment,
            ignore_surface_group_idx_per_segment=ignore_surface_group_idx_per_segment,
        )
        if int(dr.width(native_ignore_ids)) > 0:
            self._warm_rayd_visibility_ignore_pipeline()
        points = self._interleave_native_chain_points(normalized_points, chain_count)
        return self._visible_chain_rayd(points, chain_length_i32, length_active, native_ignore_ids)

    def segment_pair_visible(self, start_pos, end_pos, end_pos_offset, *, active=True) -> wt.Bool:
        width = dr.width(end_pos.x)
        if width <= 0:
            return dr.zeros(wt.Bool, 0)
        start_pos_b = broadcast(start_pos, width)
        end_pos_b = broadcast(end_pos, width)
        end_pos_offset_b = broadcast(end_pos_offset, width)
        min_seg_len = wt.Float(2.0 * RAY_ORIGIN_BIAS + EPS)
        active_mask = (
            wt.Bool(active)
            & (dr.norm(end_pos_b - start_pos_b) > min_seg_len)
            & (dr.norm(end_pos_offset_b - start_pos_b) > min_seg_len)
        )
        rayd_scene = self._require_rayd_scene()
        trace = getattr(rayd_scene, "visible_pair", None)
        if trace is None:
            raise RuntimeError("RayD visible_pair is required for segment_pair_visible().")
        with dr.suspend_grad():
            result = trace(start_pos_b, end_pos_b, end_pos_offset_b, active=active_mask)
        return active_mask & wt.Bool(result.visible_a) & wt.Bool(result.visible_b)

    def segment_visible_batched(self, segment_starts, segment_ends) -> tuple:
        if len(segment_starts) != len(segment_ends):
            raise ValueError("segment_starts and segment_ends must have the same length")
        if len(segment_starts) == 0:
            return tuple()
        if len(segment_starts) == 1:
            return (self.segment_visible(segment_starts[0], segment_ends[0]),)

        widths = [dr.width(end_pos.x) for end_pos in segment_ends]
        if sum(widths) <= 0:
            return tuple(dr.zeros(wt.Bool, width) for width in widths)

        batched_start = concat_points([
            broadcast(start_pos, width)
            for start_pos, width in zip(segment_starts, widths, strict=True)
        ])
        batched_end = concat_points(segment_ends)
        batched_visible = self.segment_visible(batched_start, batched_end)

        masks = []
        offset = 0
        for width in widths:
            if width <= 0:
                masks.append(dr.zeros(wt.Bool, 0))
            else:
                gather_idx = dr.arange(wt.UInt32, width) + wt.UInt32(offset)
                masks.append(dr.gather(wt.Bool, batched_visible, gather_idx))
            offset += width
        return tuple(masks)

    def segment_visible_fused(self, *, source_pos, diff_point, diff_point_offset,
                              target_pos, target_valid) -> tuple:
        safe_target_diff_point = dr.select(target_valid, diff_point, target_pos)
        safe_target_diff_point_offset = dr.select(target_valid, diff_point_offset, target_pos)
        return self.segment_visible_batched(
            (source_pos, source_pos, target_pos, target_pos),
            (diff_point, diff_point_offset, safe_target_diff_point, safe_target_diff_point_offset),
        )

    def reflected_path_visible(self, image_source, target_pos, prim_idx) -> wt.Bool:
        width = dr.width(target_pos.x)
        tri_data = self._triangle_runtime()
        if tri_data is None:
            return dr.full(wt.Bool, True, width)

        image_source_b = broadcast(image_source, width)
        prim_idx = broadcast(wt.Int32(prim_idx), width)
        valid_prim = prim_idx >= 0
        safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx, wt.Int32(0)))

        v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_prim_idx)
        v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_prim_idx)
        v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_prim_idx)
        geom_n = dr.cross(v1 - v0, v2 - v0)
        geom_n = geom_n / (dr.norm(geom_n) + EPS)

        segment = target_pos - image_source_b
        denom = dr.dot(segment, geom_n)
        valid_denom = dr.abs(denom) > EPS
        t_hit = dr.dot(v0 - image_source_b, geom_n) / (denom + EPS)
        hit_p = image_source_b + t_hit * segment
        surface_hit, _ = self.triangle_contains_point(hit_p, prim_idx)
        return valid_prim & valid_denom & (t_hit > EPS) & (t_hit < (1.0 - EPS)) & surface_hit

    def triangle_surface_intersection(self, image_source, target_pos, prim_idx):
        width = dr.width(target_pos.x)
        tri_data = self._triangle_runtime()
        if tri_data is None:
            zero_point = wt.Point3f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))
            zero_normal = wt.Vector3f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))
            return dr.zeros(wt.Bool, width), zero_point, zero_normal, wt.Int32(prim_idx)

        image_source_b = broadcast(image_source, width)
        prim_idx_i32 = broadcast(wt.Int32(prim_idx), width)
        valid_prim = prim_idx_i32 >= 0
        safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))

        v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_prim_idx)
        v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_prim_idx)
        v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_prim_idx)
        geom_n = dr.cross(v1 - v0, v2 - v0)
        geom_n = geom_n / (dr.norm(geom_n) + EPS)

        segment = target_pos - image_source_b
        denom = dr.dot(segment, geom_n)
        valid_denom = dr.abs(denom) > EPS
        t_hit = dr.dot(v0 - image_source_b, geom_n) / (denom + EPS)
        hit_p = image_source_b + t_hit * segment
        surface_hit, resolved_prim_idx = self.triangle_contains_point(hit_p, prim_idx_i32)
        valid = valid_prim & valid_denom & (t_hit > EPS) & (t_hit < (1.0 - EPS)) & surface_hit
        return valid, hit_p, geom_n, resolved_prim_idx

    def _point_inside_one_ray(self, point, ray_dir, active) -> wt.Bool:
        width = dr.width(point.x)
        tri_data = self._triangle_runtime()
        if tri_data is None or int(tri_data.get("n_triangles", 0)) <= 0:
            return dr.zeros(wt.Bool, width)

        if (not point_grad_enabled(point) and not scene_geometry_grad_enabled(self)):
            ray_dir_b = wt.Vector3f(
                dr.full(wt.Float, scalar(ray_dir.x), width),
                dr.full(wt.Float, scalar(ray_dir.y), width),
                dr.full(wt.Float, scalar(ray_dir.z), width),
            )
            ray_origin = point + ray_dir_b * RAY_ORIGIN_BIAS
            with dr.suspend_grad():
                si = self.ray_intersect(
                    rayd.RayAD(ray_origin, ray_dir_b),
                    active=active,
                    flags=rayd.RayFlags.Geometric,
                )
                hit = si.is_valid() & active
            return active & hit & (dr.dot(si.geo_n, ray_dir_b) > wt.Float(0.0))
        ray_dir_b = broadcast(ray_dir, width)
        ray_origin = point + ray_dir_b * RAY_ORIGIN_BIAS
        hit, _, _, geom_n, _ = self.intersect_rays_with_prim(ray_origin, ray_dir_b, active)
        return active & hit & (dr.dot(geom_n, ray_dir_b) > wt.Float(0.0))

    def point_inside_closed_mesh(self, point, *, robust: bool = False, ray_dir=None,
                                  active=None) -> wt.Bool:
        width = dr.width(point.x)
        active_mask = dr.full(wt.Bool, True, width) if active is None else active
        tri_data = self._triangle_runtime()
        if tri_data is None or int(tri_data.get("n_triangles", 0)) <= 0:
            return dr.zeros(wt.Bool, width)

        if robust:
            directions = (
                _normalized_constant_direction((0.81234133, 0.52311241, 0.25843197)),
                _normalized_constant_direction((-0.37139068, 0.60114462, 0.70757474)),
            )
            inside = active_mask
            for direction in directions:
                inside = inside & self._point_inside_one_ray(point, direction, active_mask)
            return inside
        if ray_dir is None:
            raise ValueError("point_inside_closed_mesh requires ray_dir when robust=False.")
        return self._point_inside_one_ray(point, ray_dir, active_mask)

    def gather_edge_subset(self, edge_idx, *, valid_mask=None) -> dict:
        width = int(dr.width(edge_idx))
        edge_gpu = self._selected_edge_runtime()
        if edge_gpu is None:
            return {
                "pos": wt.Point3f(0.0, 0.0, 0.0),
                "edge_dir": wt.Vector3f(0.0, 0.0, 1.0),
                "n0": wt.Vector3f(0.0, 0.0, 1.0),
                "n_face_n": wt.Vector3f(0.0, 0.0, -1.0),
                "wedge_n": wt.Float(1.5),
                "length": wt.Float(0.0),
                "line_min": wt.Float(0.0),
                "line_max": wt.Float(0.0),
                "adjacent_face0": wt.Int32(-1),
                "adjacent_face1": wt.Int32(-1),
                "valid": dr.zeros(wt.Bool, width),
            }
        edge_idx_i32 = wt.Int32(edge_idx)
        if valid_mask is None:
            valid_mask = edge_idx_i32 >= 0
        safe_idx = wt.UInt32(dr.select(valid_mask, edge_idx_i32, wt.Int32(0)))
        return {
            "pos": dr.gather(wt.Point3f, edge_gpu["pos"], safe_idx),
            "edge_dir": dr.gather(wt.Vector3f, edge_gpu["edge_dir"], safe_idx),
            "n0": dr.gather(wt.Vector3f, edge_gpu["n0"], safe_idx),
            "n_face_n": dr.gather(wt.Vector3f, edge_gpu["n_face_n"], safe_idx),
            "wedge_n": dr.gather(wt.Float, edge_gpu["wedge_n"], safe_idx),
            "length": dr.gather(wt.Float, edge_gpu["length"], safe_idx),
            "line_min": dr.gather(wt.Float, edge_gpu["line_min"], safe_idx),
            "line_max": dr.gather(wt.Float, edge_gpu["line_max"], safe_idx),
            "adjacent_face0": dr.gather(wt.Int32, edge_gpu["adjacent_face0"], safe_idx),
            "adjacent_face1": dr.gather(wt.Int32, edge_gpu["adjacent_face1"], safe_idx),
            "valid": wt.Bool(valid_mask),
        }

    def edge_face_materials(self, face0_idx, face1_idx, *, valid_mask=None,
                            default_gain: float = 1.0) -> tuple[FaceMaterial, FaceMaterial]:
        def resolve(face_idx) -> FaceMaterial:
            face_mask = valid_mask
            if face_mask is None:
                face_mask = face_idx >= 0
            mat = self.triangle_material(face_idx, valid_mask=face_mask)
            width = int(dr.width(wt.Int32(face_idx)))
            return FaceMaterial(
                eta_r=mat["eps_r"],
                mu_r=mat["mu_r"],
                sigma=mat["sigma_e"],
                gain=dr.full(wt.Float, float(default_gain), width),
                use_fresnel=mat["valid"],
            )
        return resolve(face0_idx), resolve(face1_idx)

    def update_vertices(self, vertices, recompute_edges: bool = True):
        if vertices is None:
            raise ValueError("vertices must not be None.")
        vertices = to_point3f(vertices)

        mesh_structures = [s for s in self.structures if s.enabled]
        if len(mesh_structures) != len(self._structure_meshes):
            raise RuntimeError("Scene structure/runtime mesh counts are inconsistent.")

        offset = 0
        for structure, mesh in zip(mesh_structures, self._structure_meshes):
            count = mesh_buffer_count(mesh["vertices"])
            idx = dr.arange(wt.UInt32, count) + wt.UInt32(offset)
            mesh_vertices = wt.Point3f(
                dr.gather(wt.Float, vertices.x, idx),
                dr.gather(wt.Float, vertices.y, idx),
                dr.gather(wt.Float, vertices.z, idx),
            )
            mesh["vertices"] = mesh_vertices
            geometry = structure.geometry
            if hasattr(geometry, "update_vertices"):
                geometry.update_vertices(mesh_vertices, sync=False)
            offset += count

        SceneBuilder.configure_runtime_backends(self)
        self.sync(recompute_edges=recompute_edges)

    def _set_triangle_material_runtime(self, *, eps_r, mu_r, sigma_e, specified, structure_idx) -> None:
        if self.tri_data is None:
            return
        n_tri = int(self.tri_data.get("n_triangles", 0))
        if n_tri <= 0:
            return
        if int(dr.width(eps_r)) != n_tri or int(dr.width(mu_r)) != n_tri or int(dr.width(sigma_e)) != n_tri:
            raise ValueError("Material runtime arrays must match the scene triangle count.")

        has_specified = bool(dr.any(specified))
        n_specified = int(dr.width(dr.compress(specified)))
        self.tri_data.update({
            "material_eps_r": eps_r,
            "material_mu_r": mu_r,
            "material_sigma_e": sigma_e,
            "material_specified": specified,
            "material_structure_idx": structure_idx,
            "material_has_specified_materials": has_specified,
            "material_n_specified_triangles": n_specified,
            "material_n_default_material_triangles": int(n_tri - n_specified),
        })
        self._triangle_material_data = {
            "eps_r": eps_r, "mu_r": mu_r, "sigma_e": sigma_e, "specified": specified, "structure_idx": structure_idx,
            "has_specified_materials": has_specified,
            "n_specified_triangles": n_specified,
            "n_default_material_triangles": int(n_tri - n_specified),
        }
        self._tri_data_cache_key = self._tri_data_cache = None


@dataclass(frozen=True)
class StructureBinding:
    """Scene-bound handle for a Structure; exposes runtime mutations."""

    scene: Scene
    structure: Structure
    index: int

    @property
    def name(self) -> str:
        return self.structure.name

    def set_material_parameters(self, *, eps_r=None, mu_r=None, sigma_e=None, specified: bool | None = None) -> StructureBinding:
        """Update this structure's runtime material arrays without rebuilding geometry."""
        scene = self.scene
        if scene.tri_data is None:
            raise RuntimeError("Scene has no triangle material runtime data.")

        n_tri = int(scene.tri_data.get("n_triangles", 0))
        if n_tri <= 0:
            raise RuntimeError("Scene has no triangles to update.")

        material_structure_idx = scene.tri_data.get("material_structure_idx")
        if material_structure_idx is None:
            raise RuntimeError("Scene material runtime data has no structure index array.")

        target_mask = wt.Int32(material_structure_idx) == wt.Int32(int(self.index))
        if not bool(dr.any(target_mask)):
            raise ValueError(f"Structure '{self.name}' has no enabled runtime triangles.")

        cur_eps_r = scene.tri_data["material_eps_r"]
        cur_mu_r = scene.tri_data["material_mu_r"]
        cur_sigma_e = scene.tri_data["material_sigma_e"]
        cur_specified = scene.tri_data.get("material_specified", dr.full(wt.Bool, True, n_tri))

        next_eps_r = cur_eps_r if eps_r is None else dr.select(
            target_mask, SceneBuilder._material_array(eps_r, n_tri, name=f"{self.name}.material.eps_r"), cur_eps_r,
        )
        next_sigma_e = cur_sigma_e if sigma_e is None else dr.select(
            target_mask, SceneBuilder._material_array(sigma_e, n_tri, name=f"{self.name}.material.sigma_e"), cur_sigma_e,
        )
        next_mu_r = cur_mu_r if mu_r is None else dr.select(
            target_mask, SceneBuilder._material_array(mu_r, n_tri, name=f"{self.name}.material.mu_r"), cur_mu_r,
        )
        next_specified = cur_specified if specified is None else dr.select(
            target_mask, wt.Bool(bool(specified)), cur_specified,
        )

        scene._set_triangle_material_runtime(
            eps_r=next_eps_r,
            mu_r=next_mu_r,
            sigma_e=next_sigma_e,
            specified=next_specified,
            structure_idx=material_structure_idx,
        )
        return self
