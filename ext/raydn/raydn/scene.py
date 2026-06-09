from __future__ import annotations

import torch

from . import _C
from .autograd import accum_dfr_chain_native as _accum_dfr_chain_native
from .autograd import accum_dfr_coherent_direct_native as _accum_dfr_coherent_direct_native
from .autograd import accum_dfr_direct_native as _accum_dfr_direct_native
from .autograd import intersect as _intersect
from .autograd import nearest_edge as _nearest_edge
from .autograd import nearest_edge_ray as _nearest_edge_ray
from .autograd import trace_refl_epc_field as _trace_refl_epc_field
from .autograd import trace_dfr_paths_order1_native as _trace_dfr_paths_order1_native
from .autograd import trace_reflections as _trace_reflections
from .autograd import visible as _visible
from .mesh import Mesh
from .types import DfrGrid, DfrMaterial, DfrStates, Intersection, Ray, RayFlags, _LazyIntersection, _ReducedIntersection


def _native_scene_tensor(value: torch.Tensor) -> torch.Tensor:
    value = torch.autograd.forward_ad.unpack_dual(value).primal
    if torch._C._functorch.is_functorch_wrapped_tensor(value) or torch._C._functorch.is_gradtrackingtensor(value):
        value = torch._C._functorch.get_unwrapped(value)
    try:
        value.data_ptr()
    except RuntimeError:
        value = value.detach().clone()
    return value


def _has_reverse_or_forward_ad(*values: torch.Tensor) -> bool:
    if torch.autograd.forward_ad._current_level < 0:
        return any(value.requires_grad for value in values)
    for value in values:
        unpacked = torch.autograd.forward_ad.unpack_dual(value)
        if unpacked.primal.requires_grad or unpacked.tangent is not None:
            return True
    return False


def _has_forward_ad(*values: torch.Tensor) -> bool:
    if torch.autograd.forward_ad._current_level < 0:
        return False
    for value in values:
        if torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            return True
    return False


class Scene:
    def __init__(self) -> None:
        self._meshes: list[tuple[Mesh, bool]] = []
        self._native_scene = None
        self._ready = False
        self._pending_updates = False

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        if not isinstance(mesh, Mesh):
            raise TypeError("Scene.add_mesh() expects raydn.Mesh.")
        if self._native_scene is not None:
            self._native_scene = None
        self._meshes.append((mesh, bool(dynamic)))
        self._ready = False
        self._pending_updates = False
        return len(self._meshes) - 1

    def _mesh_spec(self, mesh: Mesh, dynamic: bool) -> dict[str, object]:
        return {
            "vertices": _native_scene_tensor(mesh.vertices),
            "faces": _native_scene_tensor(mesh.faces),
            "uv": _native_scene_tensor(mesh.uv),
            "face_uv": _native_scene_tensor(mesh.face_uv),
            "to_world_left": _native_scene_tensor(mesh.to_world_left),
            "to_world_right": _native_scene_tensor(mesh.to_world_right),
            "use_face_normals": mesh.use_face_normals,
            "edges_enabled": mesh.edges_enabled,
            "dynamic": dynamic,
        }

    def build(self) -> None:
        if _C is None:
            raise RuntimeError("RayDN extension is not built yet.")
        specs = [self._mesh_spec(mesh, dynamic) for mesh, dynamic in self._meshes]
        mesh_flags = []
        for mesh, dynamic in self._meshes:
            flags = 0
            if mesh.use_face_normals:
                flags |= 1
            if mesh.edges_enabled:
                flags |= 2
            if dynamic:
                flags |= 4
            mesh_flags.append(flags)
        with torch._C._DisableFuncTorch():
            native_scene = torch.classes.raydn.Scene(
                [spec["vertices"] for spec in specs],
                [spec["faces"] for spec in specs],
                [spec["uv"] for spec in specs],
                [spec["face_uv"] for spec in specs],
                [spec["to_world_left"] for spec in specs],
                [spec["to_world_right"] for spec in specs],
                mesh_flags,
            )
        self._native_scene = native_scene
        self._ready = True
        self._pending_updates = False

    def _require_native_scene(self):
        if not self._ready or self._native_scene is None:
            raise RuntimeError("Scene is not ready. Call build() before querying.")
        return self._native_scene

    def _mesh_vertex_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(mesh.vertices for mesh, _dynamic in self._meshes)

    def is_ready(self) -> bool:
        return self._ready

    @property
    def num_meshes(self) -> int:
        scene = self._require_native_scene()
        return int(scene.num_meshes())

    @property
    def version(self) -> int:
        scene = self._require_native_scene()
        return int(scene.version())

    def intersect(self, ray: Ray, active=None, flags: RayFlags = RayFlags.All):
        scene = self._require_native_scene()
        active_arg = active
        flags_value = int(flags)
        if len(self._meshes) == 1 and torch.autograd.forward_ad._current_level < 0:
            vertices = self._meshes[0][0].vertices
            if not (vertices.requires_grad or ray.o.requires_grad or ray.d.requires_grad or ray.tmax.requires_grad):
                if flags_value == 0:
                    t = torch.ops.raydn.intersect_forward_t(
                        scene,
                        ray.o,
                        ray.d,
                        ray.tmax,
                        active_arg,
                    )
                    return _ReducedIntersection(scene, t)
                values = torch.ops.raydn.intersect_forward_flags(
                    scene,
                    ray.o,
                    ray.d,
                    ray.tmax,
                    active_arg,
                    flags_value,
                )
                return Intersection(*values)
            if flags_value != 0:
                def load_t():
                    return torch.ops.raydn.intersect_ad_t(scene, vertices, ray.o, ray.d, ray.tmax, active_arg)

                def load_full():
                    values = torch.ops.raydn.intersect_ad_flags(
                        scene,
                        vertices,
                        ray.o,
                        ray.d,
                        ray.tmax,
                        active_arg,
                        flags_value,
                    )
                    return Intersection(*values)

                return _LazyIntersection(load_t, load_full)
            t = torch.ops.raydn.intersect_ad_t(scene, vertices, ray.o, ray.d, ray.tmax, active_arg)
            return _ReducedIntersection(scene, t)
        mesh_vertices = self._mesh_vertex_tensors()
        if not _has_reverse_or_forward_ad(*mesh_vertices, ray.o, ray.d, ray.tmax):
            if flags_value == 0:
                t = torch.ops.raydn.intersect_forward_t(
                    scene,
                    ray.o,
                    ray.d,
                    ray.tmax,
                    active_arg,
                )
                return _ReducedIntersection(scene, t)
            values = torch.ops.raydn.intersect_forward_flags(
                scene,
                ray.o,
                ray.d,
                ray.tmax,
                active_arg,
                flags_value,
            )
            return Intersection(*values)
        if len(mesh_vertices) == 1:
            vertices = mesh_vertices[0]
            if not _has_forward_ad(vertices, ray.o, ray.d, ray.tmax):
                if flags_value != 0:
                    def load_t():
                        return torch.ops.raydn.intersect_ad_t(scene, vertices, ray.o, ray.d, ray.tmax, active_arg)

                    def load_full():
                        values = torch.ops.raydn.intersect_ad_flags(
                            scene,
                            vertices,
                            ray.o,
                            ray.d,
                            ray.tmax,
                            active_arg,
                            flags_value,
                        )
                        return Intersection(*values)

                    return _LazyIntersection(load_t, load_full)
                t = torch.ops.raydn.intersect_ad_t(scene, vertices, ray.o, ray.d, ray.tmax, active_arg)
                return _ReducedIntersection(scene, t)
            return _intersect(scene, vertices, ray.o, ray.d, ray.tmax, active_arg, flags_value)
        return _intersect(
            scene,
            mesh_vertices[0],
            ray.o,
            ray.d,
            ray.tmax,
            active_arg,
            flags_value,
            mesh_vertices=mesh_vertices,
        )

    def nearest_edge(self, point: torch.Tensor | Ray):
        scene = self._require_native_scene()
        mesh_vertices = self._mesh_vertex_tensors()
        if isinstance(point, Ray):
            return _nearest_edge_ray(
                scene,
                mesh_vertices[0],
                point.o,
                point.d,
                point.tmax,
                None,
            )
        return _nearest_edge(scene, mesh_vertices[0], point, mesh_vertices=mesh_vertices)

    def visible(self, start: torch.Tensor, end: torch.Tensor, active=None):
        scene = self._require_native_scene()
        return _visible(scene, start, end, active)

    def trace_reflections(self, ray: Ray, max_bounces: int, active=None):
        scene = self._require_native_scene()
        mesh_vertices = self._mesh_vertex_tensors()
        return _trace_reflections(
            scene,
            mesh_vertices[0],
            ray.o,
            ray.d,
            ray.tmax,
            active,
            int(max_bounces),
            mesh_vertices=mesh_vertices,
        )

    def trace_refl_epc_field(self, source: torch.Tensor, receiver: torch.Tensor, max_bounces: int, active=None):
        scene = self._require_native_scene()
        mesh_vertices = self._mesh_vertex_tensors()
        return _trace_refl_epc_field(
            scene,
            mesh_vertices[0],
            source,
            receiver,
            active,
            int(max_bounces),
            mesh_vertices=mesh_vertices,
        )

    def _default_dfr_material(self, *, like: torch.Tensor) -> DfrMaterial:
        face_count = sum(int(mesh.faces.shape[0]) for mesh, _dynamic in self._meshes)
        return DfrMaterial(*torch.ops.raydn.default_dfr_material(face_count, like))

    def trace_dfr_paths(
        self,
        *,
        tx_positions: torch.Tensor,
        rx_positions: torch.Tensor,
        states: DfrStates,
        material: DfrMaterial | None = None,
        active: torch.Tensor | None = None,
        max_paths: int | None = None,
        wavelength: float = 1.0,
    ):
        scene = self._require_native_scene()
        if material is None:
            material = self._default_dfr_material(like=states.edge_pos)
        if max_paths is None:
            max_paths = states.state_count
        return _trace_dfr_paths_order1_native(
            scene,
            tx_positions,
            rx_positions,
            states,
            material,
            active=active,
            max_paths=int(max_paths),
            wavelength=float(wavelength),
        )

    def accum_dfr_direct(
        self,
        *,
        states: DfrStates | None = None,
        grid: DfrGrid | None = None,
        material: DfrMaterial | None = None,
        active: torch.Tensor | None = None,
        wavelength: float = 1.0,
        direct_samples: int = 0,
        keller_samples: int = 0,
        suffix_samples: int = 0,
        seed: int = 0,
    ):
        scene = self._require_native_scene()
        if states is None:
            raise TypeError("Scene.accum_dfr_direct() requires RayD-style DfrStates and DfrGrid.")
        if grid is None:
            raise TypeError("Scene.accum_dfr_direct() requires DfrGrid when states are provided.")
        if material is None:
            material = self._default_dfr_material(like=states.edge_pos)
        return _accum_dfr_direct_native(
            scene,
            states,
            grid,
            material,
            active=active,
            wavelength=float(wavelength),
            direct_samples=int(direct_samples),
            keller_samples=int(keller_samples),
            suffix_samples=int(suffix_samples),
            seed=int(seed),
        )

    def accum_dfr(
        self,
        initial_states: DfrStates | None = None,
        recursive_states: DfrStates | None = None,
        grid: DfrGrid | None = None,
        material: DfrMaterial | None = None,
        active: torch.Tensor | None = None,
        recursive_active: torch.Tensor | None = None,
        wavelength: float = 1.0,
        direct_samples: int = 0,
        keller_samples: int = 0,
        suffix_samples: int = 0,
        seed: int = 0,
        max_order: int = 2,
        **kwargs,
    ):
        if initial_states is None and recursive_states is None:
            return self.accum_dfr_direct(**kwargs)
        scene = self._require_native_scene()
        if initial_states is None or recursive_states is None or grid is None:
            raise TypeError("Scene.accum_dfr() requires initial_states, recursive_states, and grid.")
        if material is None:
            material = self._default_dfr_material(like=initial_states.edge_pos)
        return _accum_dfr_chain_native(
            scene,
            initial_states,
            recursive_states,
            grid,
            material,
            active=active,
            recursive_active=recursive_active,
            wavelength=float(wavelength),
            direct_samples=int(direct_samples),
            keller_samples=int(keller_samples),
            suffix_samples=int(suffix_samples),
            seed=int(seed),
            max_order=int(max_order),
        )

    def accum_dfr_coherent_direct(
        self,
        *,
        states: DfrStates,
        grid: DfrGrid,
        material: DfrMaterial | None = None,
        active: torch.Tensor | None = None,
        wavelength: float = 1.0,
        select_diffraction_point: bool = True,
        prefilter_visibility: bool = True,
    ):
        scene = self._require_native_scene()
        if material is None:
            material = self._default_dfr_material(like=states.edge_pos)
        return _accum_dfr_coherent_direct_native(
            scene,
            states,
            grid,
            material,
            active=active,
            wavelength=float(wavelength),
            select_diffraction_point=bool(select_diffraction_point),
            prefilter_visibility=bool(prefilter_visibility),
        )

    def update_mesh_vertices(self, mesh_id: int, positions):
        scene = self._require_native_scene()
        mesh, dynamic = self._meshes[mesh_id]
        if not dynamic:
            raise RuntimeError("Scene.update_mesh_vertices(): target mesh is not dynamic.")
        mesh.vertices = positions.contiguous()
        with torch._C._DisableFuncTorch():
            scene.update_vertices(int(mesh_id), _native_scene_tensor(mesh.vertices))
        self._pending_updates = True

    def sync(self) -> None:
        scene = self._require_native_scene()
        with torch._C._DisableFuncTorch():
            scene.sync()
        self._pending_updates = False

    def has_pending_updates(self) -> bool:
        return bool(self._pending_updates)
