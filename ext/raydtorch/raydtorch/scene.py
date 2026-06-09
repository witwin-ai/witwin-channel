from __future__ import annotations

import weakref

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
from .types import DfrGrid, DfrMaterial, DfrStates, Intersection, Ray, RayFlags


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
    for value in values:
        unpacked = torch.autograd.forward_ad.unpack_dual(value)
        if unpacked.primal.requires_grad or unpacked.tangent is not None:
            return True
    return False


class Scene:
    def __init__(self) -> None:
        self._meshes: list[tuple[Mesh, bool]] = []
        self._native_handle: int | None = None
        self._finalizer: weakref.finalize | None = None
        self._ready = False
        self._pending_updates = False

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        if not isinstance(mesh, Mesh):
            raise TypeError("Scene.add_mesh() expects raydtorch.Mesh.")
        if self._native_handle is not None:
            _C.destroy_scene(self._native_handle)
            self._native_handle = None
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
            raise RuntimeError("RayDTorch extension is not built yet.")
        specs = [self._mesh_spec(mesh, dynamic) for mesh, dynamic in self._meshes]
        with torch._C._DisableFuncTorch():
            handle = int(_C.create_scene(specs))
        self._native_handle = handle
        self._finalizer = weakref.finalize(self, _C.destroy_scene, handle)
        self._ready = True
        self._pending_updates = False

    def _require_ready(self) -> int:
        if not self._ready or self._native_handle is None:
            raise RuntimeError("Scene is not ready. Call build() before querying.")
        return self._native_handle

    def _vertices_for_ad(self) -> torch.Tensor:
        if len(self._meshes) == 1:
            return self._meshes[0][0].vertices
        return torch.cat([mesh.vertices for mesh, _dynamic in self._meshes], dim=0).contiguous()

    def is_ready(self) -> bool:
        return self._ready

    @property
    def num_meshes(self) -> int:
        handle = self._require_ready()
        return int(_C.scene_num_meshes(handle))

    @property
    def version(self) -> int:
        handle = self._require_ready()
        return int(_C.scene_version(handle))

    def intersect(self, ray: Ray, active=None, flags: RayFlags = RayFlags.All):
        handle = self._require_ready()
        if active is None:
            active = torch.ones((ray.o.shape[0],), device=ray.o.device, dtype=torch.bool)
        vertices = self._vertices_for_ad()
        flags = RayFlags(flags)
        if not _has_reverse_or_forward_ad(vertices, ray.o, ray.d, ray.tmax):
            values = _C.intersect_forward_flags(
                handle,
                ray.o,
                ray.d,
                ray.tmax,
                active.contiguous(),
                int(flags),
            )
            return Intersection(*values)
        return _intersect(handle, vertices, ray.o, ray.d, ray.tmax, active.contiguous(), int(flags))

    def nearest_edge(self, point: torch.Tensor | Ray):
        handle = self._require_ready()
        vertices = self._vertices_for_ad()
        if isinstance(point, Ray):
            active = torch.ones((point.o.shape[0],), device=point.o.device, dtype=torch.bool)
            return _nearest_edge_ray(
                handle,
                vertices,
                point.o,
                point.d,
                point.tmax,
                active,
            )
        return _nearest_edge(handle, vertices, point.contiguous())

    def visible(self, start: torch.Tensor, end: torch.Tensor, active=None):
        handle = self._require_ready()
        start = start.contiguous()
        end = end.contiguous()
        if active is None:
            active = torch.ones((start.shape[0],), device=start.device, dtype=torch.bool)
        return _visible(handle, start, end, active.contiguous())

    def trace_reflections(self, ray: Ray, max_bounces: int, active=None):
        handle = self._require_ready()
        if active is None:
            active = torch.ones((ray.o.shape[0],), device=ray.o.device, dtype=torch.bool)
        vertices = self._vertices_for_ad()
        return _trace_reflections(
            handle,
            vertices,
            ray.o,
            ray.d,
            ray.tmax,
            active.contiguous(),
            int(max_bounces),
        )

    def trace_refl_epc_field(self, source: torch.Tensor, receiver: torch.Tensor, max_bounces: int, active=None):
        handle = self._require_ready()
        source = source.contiguous()
        receiver = receiver.contiguous()
        if active is None:
            active = torch.ones((source.shape[0],), device=source.device, dtype=torch.bool)
        vertices = self._vertices_for_ad()
        return _trace_refl_epc_field(
            handle,
            vertices,
            source,
            receiver,
            active.contiguous(),
            int(max_bounces),
        )

    def _default_dfr_material(self, *, device: torch.device, dtype: torch.dtype) -> DfrMaterial:
        face_count = sum(int(mesh.faces.shape[0]) for mesh, _dynamic in self._meshes)
        return DfrMaterial.default(face_count, device=device, dtype=dtype)

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
        handle = self._require_ready()
        if material is None:
            material = self._default_dfr_material(device=states.edge_pos.device, dtype=states.edge_pos.dtype)
        if active is None:
            active = torch.ones((states.state_count,), device=states.edge_pos.device, dtype=torch.bool)
        if max_paths is None:
            max_paths = states.state_count
        return _trace_dfr_paths_order1_native(
            handle,
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
        handle = self._require_ready()
        if states is None:
            raise TypeError("Scene.accum_dfr_direct() requires RayD-style DfrStates and DfrGrid.")
        if grid is None:
            raise TypeError("Scene.accum_dfr_direct() requires DfrGrid when states are provided.")
        if material is None:
            material = self._default_dfr_material(device=states.edge_pos.device, dtype=states.edge_pos.dtype)
        if active is None:
            active = torch.ones((states.state_count,), device=states.edge_pos.device, dtype=torch.bool)
        return _accum_dfr_direct_native(
            handle,
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
        handle = self._require_ready()
        if initial_states is None or recursive_states is None or grid is None:
            raise TypeError("Scene.accum_dfr() requires initial_states, recursive_states, and grid.")
        if material is None:
            material = self._default_dfr_material(device=initial_states.edge_pos.device, dtype=initial_states.edge_pos.dtype)
        if active is None:
            active = torch.ones((initial_states.state_count,), device=initial_states.edge_pos.device, dtype=torch.bool)
        if recursive_active is None:
            recursive_active = torch.ones((recursive_states.state_count,), device=recursive_states.edge_pos.device, dtype=torch.bool)
        return _accum_dfr_chain_native(
            handle,
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
        handle = self._require_ready()
        if material is None:
            material = self._default_dfr_material(device=states.edge_pos.device, dtype=states.edge_pos.dtype)
        if active is None:
            active = torch.ones((states.state_count,), device=states.edge_pos.device, dtype=torch.bool)
        return _accum_dfr_coherent_direct_native(
            handle,
            states,
            grid,
            material,
            active=active,
            wavelength=float(wavelength),
            select_diffraction_point=bool(select_diffraction_point),
            prefilter_visibility=bool(prefilter_visibility),
        )

    def update_mesh_vertices(self, mesh_id: int, positions):
        handle = self._require_ready()
        mesh, dynamic = self._meshes[mesh_id]
        if not dynamic:
            raise RuntimeError("Scene.update_mesh_vertices(): target mesh is not dynamic.")
        mesh.vertices = positions.contiguous()
        with torch._C._DisableFuncTorch():
            _C.update_mesh_vertices(handle, int(mesh_id), _native_scene_tensor(mesh.vertices))
        self._pending_updates = True

    def sync(self) -> None:
        handle = self._require_ready()
        with torch._C._DisableFuncTorch():
            _C.sync_scene(handle)
        self._pending_updates = False

    def has_pending_updates(self) -> bool:
        return bool(self._pending_updates)
