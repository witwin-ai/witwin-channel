from __future__ import annotations

import torch

from . import _C
from .types import (
    DfrAccum,
    DfrCoherentAccum,
    DfrGrid,
    DfrMaterial,
    DfrPaths,
    DfrStates,
    Intersection,
    NearestPointEdge,
    NearestRayEdge,
    ReflEpcField,
    ReflectionChain,
)


def _native_tensor(value: torch.Tensor) -> torch.Tensor:
    value = torch.autograd.forward_ad.unpack_dual(value).primal
    if torch._C._functorch.is_functorch_wrapped_tensor(value) or torch._C._functorch.is_gradtrackingtensor(value):
        value = torch._C._functorch.get_unwrapped(value)
    return value


def _needs_reverse_or_forward_ad(*values: torch.Tensor) -> bool:
    for value in values:
        unpacked = torch.autograd.forward_ad.unpack_dual(value)
        if unpacked.primal.requires_grad or unpacked.tangent is not None:
            return True
    return False


_RAY_FLAG_GEOMETRIC = 0x01
_RAY_FLAG_SHADING_N = 0x02
_RAY_FLAG_UV = 0x04
_RAY_FLAG_ALL = _RAY_FLAG_GEOMETRIC | _RAY_FLAG_SHADING_N | _RAY_FLAG_UV


def _grad_or_zeros(value: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if value is not None and value.numel() != 0:
        return value.contiguous()
    return torch.zeros_like(like)


class _IntersectFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        scene_handle: int,
        vertices: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ray_tmax: torch.Tensor,
        active: torch.Tensor,
        flags: int,
    ):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        outputs = _C.intersect_forward_ad_flags(int(scene_handle), ray_o, ray_d, ray_tmax, active, int(flags))
        return outputs[:12]

    @staticmethod
    def setup_context(ctx, inputs, output):
        scene_handle, vertices, ray_o, ray_d, ray_tmax, active, flags = inputs
        (
            t,
            _p,
            _n,
            _geo_n,
            _uv,
            _barycentric,
            shape_id,
            prim_id,
            local_prim_id,
            global_prim_id,
            tape_prim_id,
            tape_barycentric,
        ) = output
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ray_o = torch.autograd.forward_ad.unpack_dual(ray_o).primal
        ray_d = torch.autograd.forward_ad.unpack_dual(ray_d).primal
        ray_tmax = torch.autograd.forward_ad.unpack_dual(ray_tmax).primal
        ctx.scene_handle = int(scene_handle)
        ctx.flags = int(flags)
        ctx.save_for_backward(ray_o, ray_d, ray_tmax, active, tape_prim_id, tape_barycentric, t)
        ctx.save_for_forward(vertices, ray_o, ray_d, active, tape_prim_id, tape_barycentric, t)
        ctx.mark_non_differentiable(shape_id, prim_id, local_prim_id, global_prim_id, tape_prim_id, tape_barycentric)

    @staticmethod
    def backward(ctx, *grad_outputs):
        ray_o, ray_d, ray_tmax, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
        grad_t = _grad_or_zeros(grad_outputs[0], tape_t)
        grad_p = _grad_or_zeros(grad_outputs[1], ray_o)
        grad_n = _grad_or_zeros(grad_outputs[2], ray_o)
        grad_geo_n = _grad_or_zeros(grad_outputs[3], ray_o)
        grad_uv = _grad_or_zeros(
            grad_outputs[4],
            torch.empty((ray_o.shape[0], 2), device=ray_o.device, dtype=ray_o.dtype),
        )
        grad_barycentric = _grad_or_zeros(
            grad_outputs[5],
            torch.empty((ray_o.shape[0], 3), device=ray_o.device, dtype=ray_o.dtype),
        )
        grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax = _C.intersect_backward(
            ctx.scene_handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            tape_prim_id,
            tape_barycentric,
            grad_t,
            grad_p,
            grad_n,
            grad_geo_n,
            grad_uv,
            grad_barycentric,
        )
        return None, grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax, None, None

    @staticmethod
    def jvp(ctx, grad_scene_handle, grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax, grad_active, grad_flags):
        vertices, ray_o, ray_d, active, tape_prim_id, tape_barycentric, _tape_t = ctx.saved_tensors
        if grad_vertices is None:
            grad_vertices = torch.zeros_like(vertices)
        if grad_ray_o is None:
            grad_ray_o = torch.zeros_like(ray_o)
        if grad_ray_d is None:
            grad_ray_d = torch.zeros_like(ray_d)
        with torch._C._DisableFuncTorch():
            values = _C.intersect_jvp(
                ctx.scene_handle,
                _native_tensor(ray_o),
                _native_tensor(ray_d),
                _native_tensor(active),
                _native_tensor(tape_prim_id),
                _native_tensor(tape_barycentric),
                _native_tensor(grad_vertices),
                _native_tensor(grad_ray_o),
                _native_tensor(grad_ray_d),
            )
        tangent_t, tangent_p, tangent_n, tangent_geo_n, tangent_uv, tangent_barycentric = values
        empty_vec3 = tangent_p.new_empty((0, 3))
        empty_uv = tangent_uv.new_empty((0, 2))
        flags = ctx.flags
        return (
            tangent_t,
            tangent_p if flags & _RAY_FLAG_GEOMETRIC else empty_vec3,
            tangent_n if flags & _RAY_FLAG_SHADING_N else empty_vec3,
            tangent_geo_n if flags & _RAY_FLAG_GEOMETRIC else empty_vec3,
            tangent_uv if flags & _RAY_FLAG_UV else empty_uv,
            tangent_barycentric if flags & _RAY_FLAG_GEOMETRIC else empty_vec3,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def intersect(
    scene_handle: int,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor,
    flags: int = _RAY_FLAG_ALL,
) -> Intersection:
    values = _IntersectFunction.apply(scene_handle, vertices, ray_o, ray_d, ray_tmax, active, int(flags))
    return Intersection(*values[:10])


class _NearestEdgeFunction(torch.autograd.Function):
    @staticmethod
    def forward(scene_handle: int, vertices: torch.Tensor, point: torch.Tensor):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        outputs = _C.nearest_edge_forward(int(scene_handle), point)
        return outputs

    @staticmethod
    def setup_context(ctx, inputs, output):
        scene_handle, vertices, point = inputs
        distance, edge_point, edge_t, shape_id, edge_id, global_edge_id, tape_edge_id, tape_s, tape_d = output
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        point = torch.autograd.forward_ad.unpack_dual(point).primal
        ctx.scene_handle = int(scene_handle)
        ctx.save_for_backward(point, tape_edge_id, tape_s, tape_d, distance)
        ctx.save_for_forward(vertices, point, tape_edge_id, tape_s, tape_d)
        ctx.mark_non_differentiable(shape_id, edge_id, global_edge_id, tape_edge_id)

    @staticmethod
    def backward(ctx, *grad_outputs):
        point, tape_edge_id, tape_s, tape_d, distance = ctx.saved_tensors
        grad_distance = grad_outputs[0].contiguous() if grad_outputs[0] is not None else torch.zeros_like(distance)
        grad_edge_point = grad_outputs[1].contiguous() if grad_outputs[1] is not None else torch.zeros_like(point)
        grad_edge_t = torch.zeros_like(distance)
        if grad_outputs[2] is not None:
            grad_edge_t = grad_edge_t + grad_outputs[2].contiguous()
        if len(grad_outputs) > 7 and grad_outputs[7] is not None:
            grad_edge_t = grad_edge_t + grad_outputs[7].contiguous()
        grad_vertices, grad_point = _C.nearest_edge_backward(
            ctx.scene_handle,
            point,
            tape_edge_id,
            tape_s,
            tape_d,
            grad_distance,
            grad_edge_point,
            grad_edge_t,
        )
        return None, grad_vertices, grad_point

    @staticmethod
    def jvp(ctx, grad_scene_handle, grad_vertices, grad_point):
        vertices, point, tape_edge_id, tape_s, tape_d = ctx.saved_tensors
        if grad_vertices is None:
            grad_vertices = torch.zeros_like(vertices)
        if grad_point is None:
            grad_point = torch.zeros_like(point)
        with torch._C._DisableFuncTorch():
            tangent_distance, tangent_edge_point, tangent_edge_t = _C.nearest_edge_jvp(
                ctx.scene_handle,
                _native_tensor(point),
                _native_tensor(tape_edge_id),
                _native_tensor(tape_s),
                _native_tensor(tape_d),
                _native_tensor(grad_vertices),
                _native_tensor(grad_point),
            )
        return (
            tangent_distance,
            tangent_edge_point,
            tangent_edge_t,
            None,
            None,
            None,
            None,
            torch.zeros_like(tape_s),
            torch.zeros_like(tape_d),
        )


def _needs_nearest_edge_ad(vertices: torch.Tensor, point: torch.Tensor) -> bool:
    for value in (vertices, point):
        if value.requires_grad:
            return True
        if torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            return True
    return False


def nearest_edge(scene_handle: int, vertices: torch.Tensor, point: torch.Tensor) -> NearestPointEdge:
    if _C is None:
        raise RuntimeError("RayDTorch extension is not built yet.")
    if not _needs_nearest_edge_ad(vertices, point):
        values = _C.nearest_edge_forward_noad(int(scene_handle), point)
        return NearestPointEdge(*values)
    values = _NearestEdgeFunction.apply(scene_handle, vertices, point)
    return NearestPointEdge(*values[:6])


class _NearestEdgeRayFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        scene_handle: int,
        vertices: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ray_tmax: torch.Tensor,
        active: torch.Tensor,
    ):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        return _C.nearest_edge_ray_forward(int(scene_handle), ray_o, ray_d, ray_tmax, active)

    @staticmethod
    def setup_context(ctx, inputs, output):
        distance, ray_t, point, edge_t, edge_point, shape_id, edge_id, global_edge_id, tape_edge_id = output
        ctx.mark_non_differentiable(shape_id, edge_id, global_edge_id, tape_edge_id)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None, None, None, None


def nearest_edge_ray(
    scene_handle: int,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor,
) -> NearestRayEdge:
    values = _NearestEdgeRayFunction.apply(scene_handle, vertices, ray_o, ray_d, ray_tmax, active)
    return NearestRayEdge(*values[:8])


def visible(scene_handle: int, start: torch.Tensor, end: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    if _C is None:
        raise RuntimeError("RayDTorch extension is not built yet.")
    values = _C.visibility_forward(int(scene_handle), start, end, active)
    return values[0]


def _needs_trace_reflection_ad(*values: torch.Tensor) -> bool:
    for value in values:
        if value.requires_grad:
            return True
        if torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            return True
    return False


class _TraceReflectionsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        scene_handle: int,
        vertices: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ray_tmax: torch.Tensor,
        active: torch.Tensor,
        max_bounces: int,
    ):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        outputs = _C.trace_reflections_forward(
            int(scene_handle),
            ray_o,
            ray_d,
            ray_tmax,
            active,
            int(max_bounces),
        )
        return outputs

    @staticmethod
    def setup_context(ctx, inputs, output):
        scene_handle, vertices, ray_o, ray_d, ray_tmax, active, max_bounces = inputs
        valid, t, image_sources, prim_ids, tape_prim_id, tape_barycentric, tape_t, tape_hits, tape_normals = output
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        ray_o = torch.autograd.forward_ad.unpack_dual(ray_o).primal
        ray_d = torch.autograd.forward_ad.unpack_dual(ray_d).primal
        ray_tmax = torch.autograd.forward_ad.unpack_dual(ray_tmax).primal
        ctx.scene_handle = int(scene_handle)
        ctx.max_bounces = int(max_bounces)
        ctx.save_for_backward(
            ray_o,
            ray_d,
            ray_tmax,
            active,
            tape_prim_id,
            tape_barycentric,
            tape_t,
            tape_hits,
            tape_normals,
            image_sources,
        )
        ctx.save_for_forward(
            vertices,
            ray_o,
            ray_d,
            active,
            tape_prim_id,
            tape_barycentric,
            tape_hits,
            tape_normals,
            image_sources,
        )
        ctx.mark_non_differentiable(
            valid,
            prim_ids,
            tape_prim_id,
            tape_barycentric,
            tape_t,
            tape_hits,
            tape_normals,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            ray_o,
            ray_d,
            ray_tmax,
            active,
            tape_prim_id,
            tape_barycentric,
            tape_t,
            tape_hits,
            tape_normals,
            image_sources,
        ) = ctx.saved_tensors
        grad_t = grad_outputs[1].contiguous() if grad_outputs[1] is not None else torch.zeros_like(tape_t)
        grad_image_sources = (
            grad_outputs[2].contiguous() if grad_outputs[2] is not None else torch.zeros_like(image_sources)
        )
        grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax = _C.trace_reflections_backward(
            ctx.scene_handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            tape_prim_id,
            tape_barycentric,
            tape_hits,
            tape_normals,
            image_sources,
            grad_t,
            grad_image_sources,
        )
        return None, grad_vertices, grad_ray_o, grad_ray_d, grad_ray_tmax, None, None

    @staticmethod
    def jvp(
        ctx,
        grad_scene_handle,
        grad_vertices,
        grad_ray_o,
        grad_ray_d,
        grad_ray_tmax,
        grad_active,
        grad_max_bounces,
    ):
        (
            vertices,
            ray_o,
            ray_d,
            active,
            tape_prim_id,
            tape_barycentric,
            tape_hits,
            tape_normals,
            image_sources,
        ) = ctx.saved_tensors
        if grad_vertices is None:
            grad_vertices = torch.zeros_like(vertices)
        if grad_ray_o is None:
            grad_ray_o = torch.zeros_like(ray_o)
        if grad_ray_d is None:
            grad_ray_d = torch.zeros_like(ray_d)
        with torch._C._DisableFuncTorch():
            tangent_t, tangent_image_sources = _C.trace_reflections_jvp(
                ctx.scene_handle,
                _native_tensor(ray_o),
                _native_tensor(ray_d),
                _native_tensor(active),
                _native_tensor(tape_prim_id),
                _native_tensor(tape_barycentric),
                _native_tensor(tape_hits),
                _native_tensor(tape_normals),
                _native_tensor(grad_vertices),
                _native_tensor(grad_ray_o),
                _native_tensor(grad_ray_d),
                _native_tensor(image_sources),
            )
        return None, tangent_t, tangent_image_sources, None, None, None, None, None, None


def trace_reflections(
    scene_handle: int,
    vertices: torch.Tensor,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor,
    max_bounces: int,
) -> ReflectionChain:
    if _C is None:
        raise RuntimeError("RayDTorch extension is not built yet.")
    if not _needs_trace_reflection_ad(vertices, ray_o, ray_d, ray_tmax):
        values = _C.trace_reflections_forward_noad(
            int(scene_handle),
            ray_o,
            ray_d,
            ray_tmax,
            active,
            int(max_bounces),
        )
        return ReflectionChain(*values)
    values = _TraceReflectionsFunction.apply(
        scene_handle,
        vertices,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        int(max_bounces),
    )
    return ReflectionChain(*values[:4])


class _TraceReflEpcFieldFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        scene_handle: int,
        vertices: torch.Tensor,
        source: torch.Tensor,
        receiver: torch.Tensor,
        active: torch.Tensor,
        max_bounces: int,
    ):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        return _C.trace_refl_epc_field_forward(
            int(scene_handle),
            source,
            receiver,
            active,
            int(max_bounces),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        scene_handle, vertices, source, receiver, active, max_bounces = inputs
        field_real, field_imag, path_length, valid, resolved_prim_ids, tape_prim_id, tape_barycentric, tape_t = output
        vertices = torch.autograd.forward_ad.unpack_dual(vertices).primal
        source = torch.autograd.forward_ad.unpack_dual(source).primal
        receiver = torch.autograd.forward_ad.unpack_dual(receiver).primal
        ctx.scene_handle = int(scene_handle)
        ctx.max_bounces = int(max_bounces)
        ctx.save_for_backward(source, receiver, active, tape_prim_id, tape_barycentric, tape_t)
        ctx.save_for_forward(vertices, source, receiver, active, tape_prim_id, tape_barycentric, tape_t)
        ctx.mark_non_differentiable(valid, resolved_prim_ids, tape_prim_id)

    @staticmethod
    def backward(ctx, *grad_outputs):
        source, receiver, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
        grad_field_real = grad_outputs[0].contiguous() if grad_outputs[0] is not None else torch.zeros_like(tape_t)
        grad_field_imag = grad_outputs[1].contiguous() if grad_outputs[1] is not None else torch.zeros_like(tape_t)
        grad_path_length = grad_outputs[2].contiguous() if grad_outputs[2] is not None else torch.zeros_like(tape_t)
        grad_vertices, grad_source, grad_receiver = _C.trace_refl_epc_field_backward(
            ctx.scene_handle,
            source,
            receiver,
            active,
            tape_prim_id,
            tape_barycentric,
            tape_t,
            grad_field_real,
            grad_field_imag,
            grad_path_length,
        )
        return None, grad_vertices, grad_source, grad_receiver, None, None

    @staticmethod
    def jvp(ctx, grad_scene_handle, grad_vertices, grad_source, grad_receiver, grad_active, grad_max_bounces):
        vertices, source, receiver, active, tape_prim_id, tape_barycentric, tape_t = ctx.saved_tensors
        if grad_vertices is None:
            grad_vertices = torch.zeros_like(vertices)
        if grad_source is None:
            grad_source = torch.zeros_like(source)
        if grad_receiver is None:
            grad_receiver = torch.zeros_like(receiver)
        with torch._C._DisableFuncTorch():
            tangent_field_real, tangent_field_imag, tangent_path_length = _C.trace_refl_epc_field_jvp(
                ctx.scene_handle,
                _native_tensor(source),
                _native_tensor(receiver),
                _native_tensor(active),
                _native_tensor(tape_prim_id),
                _native_tensor(tape_barycentric),
                _native_tensor(tape_t),
                _native_tensor(grad_vertices),
                _native_tensor(grad_source),
                _native_tensor(grad_receiver),
            )
        return tangent_field_real, tangent_field_imag, tangent_path_length, None, None, None, None, None


def trace_refl_epc_field(
    scene_handle: int,
    vertices: torch.Tensor,
    source: torch.Tensor,
    receiver: torch.Tensor,
    active: torch.Tensor,
    max_bounces: int,
) -> ReflEpcField:
    values = _TraceReflEpcFieldFunction.apply(
        scene_handle,
        vertices,
        source,
        receiver,
        active,
        int(max_bounces),
    )
    return ReflEpcField(*values[:5])


def _contig_states(states: DfrStates) -> DfrStates:
    states = states.with_default_vectors()
    n = states.state_count
    return DfrStates(
        edge_index=states.edge_index[:n].contiguous(),
        edge_pos=states.edge_pos[:n].contiguous(),
        edge_dir=states.edge_dir[:n].contiguous(),
        edge_t_min=states.edge_t_min[:n].contiguous(),
        edge_t_max=states.edge_t_max[:n].contiguous(),
        n0=states.n0[:n].contiguous(),
        n1=states.n1[:n].contiguous(),
        prim0=states.prim0[:n].contiguous(),
        prim1=states.prim1[:n].contiguous(),
        exterior_angle=states.exterior_angle[:n].contiguous(),
        src=states.src[:n].contiguous(),
        src_power=states.src_power[:n].contiguous(),
        wi=states.wi[:n].contiguous(),
        d0=states.d0[:n].contiguous(),
        count=states.count,
    )


def _contig_material(material: DfrMaterial) -> DfrMaterial:
    return DfrMaterial(
        eta_r=material.eta_r.contiguous(),
        sigma=material.sigma.contiguous(),
        mu_r=material.mu_r.contiguous(),
        gain=material.gain.contiguous(),
        valid=material.valid.contiguous(),
    )


def trace_dfr_paths_order1_native(
    scene_handle: int,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    states: DfrStates,
    material: DfrMaterial,
    *,
    active: torch.Tensor,
    max_paths: int,
    wavelength: float,
) -> DfrPaths:
    if _C is None:
        raise RuntimeError("RayDTorch extension is not built yet.")
    states = _contig_states(states)
    material = _contig_material(material)
    state_limit = min(states.state_count, int(max_paths))
    capacity = int(tx_positions.shape[0]) * int(rx_positions.shape[0]) * state_limit
    values = _C.diffraction_paths_order1_forward(
        int(scene_handle),
        tx_positions.contiguous(),
        rx_positions.contiguous(),
        active.contiguous(),
        states.edge_index,
        states.edge_pos,
        states.edge_dir,
        states.edge_t_min,
        states.edge_t_max,
        states.n0,
        states.n1,
        states.prim0,
        states.prim1,
        states.exterior_angle,
        states.src,
        states.src_power,
        material.gain,
        material.valid,
        capacity,
        float(wavelength),
    )
    return DfrPaths(capacity, *values)


class _DfrDirectAccumFunction(torch.autograd.Function):
    @staticmethod
    def forward(*args):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        active = args[1]
        state_edge_index = args[2]
        state_edge_pos = args[3]
        state_edge_t_min = args[5]
        empty_b = active.new_empty((0,))
        empty_i = state_edge_index.new_empty((0,))
        empty_f = state_edge_t_min.new_empty((0,))
        empty_v = state_edge_pos.new_empty((0, 3))
        return _C.diffraction_accumulation_forward(
            *args,
            1,
            empty_b,
            empty_i,
            empty_v,
            empty_v,
            empty_f,
            empty_f,
            empty_v,
            empty_v,
            empty_i,
            empty_i,
            empty_f,
            1,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.scene_handle = int(inputs[0])
        ctx.grid_axis = int(inputs[21])
        ctx.grid_position = float(inputs[22])
        ctx.grid_coord0_min = float(inputs[23])
        ctx.grid_coord0_max = float(inputs[24])
        ctx.grid_coord1_min = float(inputs[25])
        ctx.grid_coord1_max = float(inputs[26])
        ctx.grid_resolution0 = int(inputs[27])
        ctx.grid_resolution1 = int(inputs[28])
        ctx.grid_cell_area = float(inputs[29])
        ctx.wavelength = float(inputs[30])
        ctx.direct_samples = int(inputs[31])
        ctx.keller_samples = int(inputs[32])
        ctx.suffix_samples = int(inputs[33])
        ctx.seed = int(inputs[34])
        saved = (
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[9],
            inputs[10],
            inputs[11],
            inputs[12],
            inputs[13],
            inputs[14],
            inputs[19],
            inputs[20],
            output[14],
            output[15],
            output[16],
            output[17],
            output[18],
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[7:19])

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            state_wi,
            material_gain,
            material_valid,
            tape_active,
            tape_state_idx,
            tape_cell,
            tape_material_idx,
            tape_edge_u,
        ) = ctx.saved_tensors
        grad_power = grad_outputs[0]
        if grad_power is None:
            grad_power = torch.zeros(
                (ctx.grid_resolution1, ctx.grid_resolution0),
                device=state_edge_pos.device,
                dtype=state_edge_pos.dtype,
            )
        grad_field_x_re = grad_outputs[1]
        if grad_field_x_re is None:
            grad_field_x_re = torch.zeros(
                (ctx.grid_resolution1, ctx.grid_resolution0),
                device=state_edge_pos.device,
                dtype=state_edge_pos.dtype,
            )
        (
            grad_state_edge_pos,
            grad_state_edge_dir,
            grad_state_edge_t_min,
            grad_state_edge_t_max,
            grad_state_src,
            grad_state_wi,
            grad_state_src_power,
            grad_state_exterior_angle,
            grad_material_gain,
        ) = _C.diffraction_accumulation_direct_backward(
            ctx.scene_handle,
            tape_active,
            tape_state_idx,
            tape_cell,
            tape_material_idx,
            tape_edge_u,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            state_wi,
            material_gain,
            material_valid,
            ctx.grid_axis,
            ctx.grid_position,
            ctx.grid_coord0_min,
            ctx.grid_coord0_max,
            ctx.grid_coord1_min,
            ctx.grid_coord1_max,
            ctx.grid_resolution0,
            ctx.grid_resolution1,
            ctx.grid_cell_area,
            ctx.wavelength,
            ctx.direct_samples,
            ctx.keller_samples,
            ctx.suffix_samples,
            ctx.seed,
            grad_power.contiguous(),
            grad_field_x_re.contiguous(),
        )
        grads = [None] * 35
        grads[3] = grad_state_edge_pos
        grads[4] = grad_state_edge_dir
        grads[5] = grad_state_edge_t_min
        grads[6] = grad_state_edge_t_max
        grads[11] = grad_state_exterior_angle
        grads[12] = grad_state_src
        grads[13] = grad_state_src_power
        grads[14] = grad_state_wi
        grads[19] = grad_material_gain
        return tuple(grads)

    @staticmethod
    def jvp(ctx, *grad_inputs):
        (
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            state_wi,
            material_gain,
            material_valid,
            tape_active,
            tape_state_idx,
            tape_cell,
            tape_material_idx,
            tape_edge_u,
        ) = ctx.saved_tensors

        def tangent_at(index: int, primal: torch.Tensor) -> torch.Tensor:
            tangent = grad_inputs[index] if index < len(grad_inputs) else None
            return torch.zeros_like(primal) if tangent is None else tangent

        with torch._C._DisableFuncTorch():
            dot_power, dot_field_x_re = _C.diffraction_accumulation_direct_jvp(
                ctx.scene_handle,
                _native_tensor(tape_active),
                _native_tensor(tape_state_idx),
                _native_tensor(tape_cell),
                _native_tensor(tape_material_idx),
                _native_tensor(tape_edge_u),
                _native_tensor(state_edge_pos),
                _native_tensor(state_edge_dir),
                _native_tensor(state_edge_t_min),
                _native_tensor(state_edge_t_max),
                _native_tensor(state_prim0),
                _native_tensor(state_prim1),
                _native_tensor(state_exterior_angle),
                _native_tensor(state_src),
                _native_tensor(state_src_power),
                _native_tensor(state_wi),
                _native_tensor(material_gain),
                _native_tensor(material_valid),
                ctx.grid_axis,
                ctx.grid_position,
                ctx.grid_coord0_min,
                ctx.grid_coord0_max,
                ctx.grid_coord1_min,
                ctx.grid_coord1_max,
                ctx.grid_resolution0,
                ctx.grid_resolution1,
                ctx.grid_cell_area,
                ctx.wavelength,
                ctx.direct_samples,
                ctx.keller_samples,
                ctx.suffix_samples,
                ctx.seed,
                _native_tensor(tangent_at(3, state_edge_pos)),
                _native_tensor(tangent_at(4, state_edge_dir)),
                _native_tensor(tangent_at(5, state_edge_t_min)),
                _native_tensor(tangent_at(6, state_edge_t_max)),
                _native_tensor(tangent_at(11, state_exterior_angle)),
                _native_tensor(tangent_at(12, state_src)),
                _native_tensor(tangent_at(13, state_src_power)),
                _native_tensor(tangent_at(14, state_wi)),
                _native_tensor(tangent_at(19, material_gain)),
            )
        zero = torch.zeros_like(dot_power)
        return (
            dot_power,
            dot_field_x_re,
            zero,
            zero,
            zero,
            zero,
            zero,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def accum_dfr_direct_native(
    scene_handle: int,
    states: DfrStates,
    grid: DfrGrid,
    material: DfrMaterial,
    *,
    active: torch.Tensor,
    wavelength: float,
    direct_samples: int,
    keller_samples: int = 0,
    suffix_samples: int = 0,
    seed: int = 0,
) -> DfrAccum:
    states = _contig_states(states)
    material = _contig_material(material)
    if not _needs_reverse_or_forward_ad(
        states.edge_pos,
        states.edge_dir,
        states.edge_t_min,
        states.edge_t_max,
        states.exterior_angle,
        states.src,
        states.src_power,
        states.wi,
        material.gain,
    ):
        empty_b = active.new_empty((0,))
        empty_i = states.edge_index.new_empty((0,))
        empty_f = states.edge_t_min.new_empty((0,))
        empty_v = states.edge_pos.new_empty((0, 3))
        values = _C.diffraction_accumulation_forward(
            int(scene_handle),
            active.contiguous(),
            states.edge_index,
            states.edge_pos,
            states.edge_dir,
            states.edge_t_min,
            states.edge_t_max,
            states.n0,
            states.n1,
            states.prim0,
            states.prim1,
            states.exterior_angle,
            states.src,
            states.src_power,
            states.wi,
            states.d0,
            material.eta_r,
            material.sigma,
            material.mu_r,
            material.gain,
            material.valid,
            int(grid.axis),
            float(grid.position),
            float(grid.coord0_min),
            float(grid.coord0_max),
            float(grid.coord1_min),
            float(grid.coord1_max),
            int(grid.resolution0),
            int(grid.resolution1),
            grid.resolved_cell_area(),
            float(wavelength),
            int(direct_samples),
            int(keller_samples),
            int(suffix_samples),
            int(seed),
            1,
            empty_b,
            empty_i,
            empty_v,
            empty_v,
            empty_f,
            empty_f,
            empty_v,
            empty_v,
            empty_i,
            empty_i,
            empty_f,
            0,
        )
        grid_cell_count = int(grid.resolution0) * int(grid.resolution1)
        return DfrAccum(grid_cell_count, *values[:14])
    values = _DfrDirectAccumFunction.apply(
        int(scene_handle),
        active.contiguous(),
        states.edge_index,
        states.edge_pos,
        states.edge_dir,
        states.edge_t_min,
        states.edge_t_max,
        states.n0,
        states.n1,
        states.prim0,
        states.prim1,
        states.exterior_angle,
        states.src,
        states.src_power,
        states.wi,
        states.d0,
        material.eta_r,
        material.sigma,
        material.mu_r,
        material.gain,
        material.valid,
        int(grid.axis),
        float(grid.position),
        float(grid.coord0_min),
        float(grid.coord0_max),
        float(grid.coord1_min),
        float(grid.coord1_max),
        int(grid.resolution0),
        int(grid.resolution1),
        grid.resolved_cell_area(),
        float(wavelength),
        int(direct_samples),
        int(keller_samples),
        int(suffix_samples),
        int(seed),
    )
    grid_cell_count = int(grid.resolution0) * int(grid.resolution1)
    return DfrAccum(grid_cell_count, *values[:14])


class _DfrChainAccumFunction(torch.autograd.Function):
    @staticmethod
    def forward(*args):
        if _C is None:
            raise RuntimeError("RayDTorch extension is not built yet.")
        return _C.diffraction_accumulation_forward(*args)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.scene_handle = int(inputs[0])
        ctx.grid_axis = int(inputs[21])
        ctx.grid_position = float(inputs[22])
        ctx.grid_coord0_min = float(inputs[23])
        ctx.grid_coord0_max = float(inputs[24])
        ctx.grid_coord1_min = float(inputs[25])
        ctx.grid_coord1_max = float(inputs[26])
        ctx.grid_resolution0 = int(inputs[27])
        ctx.grid_resolution1 = int(inputs[28])
        ctx.grid_cell_area = float(inputs[29])
        ctx.wavelength = float(inputs[30])
        ctx.direct_samples = int(inputs[31])
        ctx.keller_samples = int(inputs[32])
        ctx.suffix_samples = int(inputs[33])
        ctx.seed = int(inputs[34])
        ctx.max_order = int(inputs[35])
        saved = (
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[9],
            inputs[10],
            inputs[11],
            inputs[12],
            inputs[13],
            inputs[19],
            inputs[20],
            inputs[37],
            inputs[38],
            inputs[39],
            inputs[40],
            inputs[41],
            inputs[44],
            inputs[45],
            inputs[46],
            output[14],
            output[16],
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[7:19])

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            material_gain,
            material_valid,
            recursive_state_edge_index,
            recursive_state_edge_pos,
            recursive_state_edge_dir,
            recursive_state_edge_t_min,
            recursive_state_edge_t_max,
            recursive_state_prim0,
            recursive_state_prim1,
            recursive_state_exterior_angle,
            tape_active,
            tape_cell,
        ) = ctx.saved_tensors
        grad_power = grad_outputs[0]
        if grad_power is None:
            grad_power = torch.zeros(
                (ctx.grid_resolution1, ctx.grid_resolution0),
                device=state_edge_pos.device,
                dtype=state_edge_pos.dtype,
            )
        grad_field_x_re = grad_outputs[1]
        if grad_field_x_re is None:
            grad_field_x_re = torch.zeros(
                (ctx.grid_resolution1, ctx.grid_resolution0),
                device=state_edge_pos.device,
                dtype=state_edge_pos.dtype,
            )
        (
            grad_state_edge_pos,
            grad_state_edge_dir,
            grad_state_edge_t_min,
            grad_state_edge_t_max,
            grad_state_src,
            grad_state_src_power,
            grad_state_exterior_angle,
            grad_recursive_state_edge_pos,
            grad_recursive_state_edge_dir,
            grad_recursive_state_edge_t_min,
            grad_recursive_state_edge_t_max,
            grad_recursive_state_exterior_angle,
            grad_material_gain,
        ) = _C.diffraction_accumulation_chain_backward(
            ctx.scene_handle,
            tape_active,
            tape_cell,
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            recursive_state_edge_index,
            recursive_state_edge_pos,
            recursive_state_edge_dir,
            recursive_state_edge_t_min,
            recursive_state_edge_t_max,
            recursive_state_prim0,
            recursive_state_prim1,
            recursive_state_exterior_angle,
            material_gain,
            material_valid,
            ctx.grid_axis,
            ctx.grid_position,
            ctx.grid_coord0_min,
            ctx.grid_coord0_max,
            ctx.grid_coord1_min,
            ctx.grid_coord1_max,
            ctx.grid_resolution0,
            ctx.grid_resolution1,
            ctx.grid_cell_area,
            ctx.wavelength,
            ctx.direct_samples,
            ctx.keller_samples,
            ctx.suffix_samples,
            ctx.seed,
            ctx.max_order,
            grad_power.contiguous(),
            grad_field_x_re.contiguous(),
        )
        grads = [None] * 48
        grads[3] = grad_state_edge_pos
        grads[4] = grad_state_edge_dir
        grads[5] = grad_state_edge_t_min
        grads[6] = grad_state_edge_t_max
        grads[11] = grad_state_exterior_angle
        grads[12] = grad_state_src
        grads[13] = grad_state_src_power
        grads[19] = grad_material_gain
        grads[38] = grad_recursive_state_edge_pos
        grads[39] = grad_recursive_state_edge_dir
        grads[40] = grad_recursive_state_edge_t_min
        grads[41] = grad_recursive_state_edge_t_max
        grads[46] = grad_recursive_state_exterior_angle
        return tuple(grads)

    @staticmethod
    def jvp(ctx, *grad_inputs):
        (
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            material_gain,
            material_valid,
            recursive_state_edge_index,
            recursive_state_edge_pos,
            recursive_state_edge_dir,
            recursive_state_edge_t_min,
            recursive_state_edge_t_max,
            recursive_state_prim0,
            recursive_state_prim1,
            recursive_state_exterior_angle,
            tape_active,
            tape_cell,
        ) = ctx.saved_tensors

        def tangent_at(index: int, primal: torch.Tensor) -> torch.Tensor:
            tangent = grad_inputs[index] if index < len(grad_inputs) else None
            return torch.zeros_like(primal) if tangent is None else tangent

        with torch._C._DisableFuncTorch():
            dot_power, dot_field_x_re = _C.diffraction_accumulation_chain_jvp(
                ctx.scene_handle,
                _native_tensor(tape_active),
                _native_tensor(tape_cell),
                _native_tensor(state_edge_index),
                _native_tensor(state_edge_pos),
                _native_tensor(state_edge_dir),
                _native_tensor(state_edge_t_min),
                _native_tensor(state_edge_t_max),
                _native_tensor(state_prim0),
                _native_tensor(state_prim1),
                _native_tensor(state_exterior_angle),
                _native_tensor(state_src),
                _native_tensor(state_src_power),
                _native_tensor(recursive_state_edge_index),
                _native_tensor(recursive_state_edge_pos),
                _native_tensor(recursive_state_edge_dir),
                _native_tensor(recursive_state_edge_t_min),
                _native_tensor(recursive_state_edge_t_max),
                _native_tensor(recursive_state_prim0),
                _native_tensor(recursive_state_prim1),
                _native_tensor(recursive_state_exterior_angle),
                _native_tensor(material_gain),
                _native_tensor(material_valid),
                ctx.grid_axis,
                ctx.grid_position,
                ctx.grid_coord0_min,
                ctx.grid_coord0_max,
                ctx.grid_coord1_min,
                ctx.grid_coord1_max,
                ctx.grid_resolution0,
                ctx.grid_resolution1,
                ctx.grid_cell_area,
                ctx.wavelength,
                ctx.direct_samples,
                ctx.keller_samples,
                ctx.suffix_samples,
                ctx.seed,
                ctx.max_order,
                _native_tensor(tangent_at(3, state_edge_pos)),
                _native_tensor(tangent_at(4, state_edge_dir)),
                _native_tensor(tangent_at(5, state_edge_t_min)),
                _native_tensor(tangent_at(6, state_edge_t_max)),
                _native_tensor(tangent_at(11, state_exterior_angle)),
                _native_tensor(tangent_at(12, state_src)),
                _native_tensor(tangent_at(13, state_src_power)),
                _native_tensor(tangent_at(38, recursive_state_edge_pos)),
                _native_tensor(tangent_at(39, recursive_state_edge_dir)),
                _native_tensor(tangent_at(40, recursive_state_edge_t_min)),
                _native_tensor(tangent_at(41, recursive_state_edge_t_max)),
                _native_tensor(tangent_at(46, recursive_state_exterior_angle)),
                _native_tensor(tangent_at(19, material_gain)),
            )
        zero = torch.zeros_like(dot_power)
        return (
            dot_power,
            dot_field_x_re,
            zero,
            zero,
            zero,
            zero,
            zero,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def accum_dfr_chain_native(
    scene_handle: int,
    initial_states: DfrStates,
    recursive_states: DfrStates,
    grid: DfrGrid,
    material: DfrMaterial,
    *,
    active: torch.Tensor,
    recursive_active: torch.Tensor,
    wavelength: float,
    direct_samples: int,
    keller_samples: int = 0,
    suffix_samples: int = 0,
    seed: int = 0,
    max_order: int = 2,
) -> DfrAccum:
    if _C is None:
        raise RuntimeError("RayDTorch extension is not built yet.")
    initial_states = _contig_states(initial_states)
    recursive_states = _contig_states(recursive_states)
    material = _contig_material(material)
    values = _DfrChainAccumFunction.apply(
        int(scene_handle),
        active.contiguous(),
        initial_states.edge_index,
        initial_states.edge_pos,
        initial_states.edge_dir,
        initial_states.edge_t_min,
        initial_states.edge_t_max,
        initial_states.n0,
        initial_states.n1,
        initial_states.prim0,
        initial_states.prim1,
        initial_states.exterior_angle,
        initial_states.src,
        initial_states.src_power,
        initial_states.wi,
        initial_states.d0,
        material.eta_r,
        material.sigma,
        material.mu_r,
        material.gain,
        material.valid,
        int(grid.axis),
        float(grid.position),
        float(grid.coord0_min),
        float(grid.coord0_max),
        float(grid.coord1_min),
        float(grid.coord1_max),
        int(grid.resolution0),
        int(grid.resolution1),
        grid.resolved_cell_area(),
        float(wavelength),
        int(direct_samples),
        int(keller_samples),
        int(suffix_samples),
        int(seed),
        int(max_order),
        recursive_active.contiguous(),
        recursive_states.edge_index,
        recursive_states.edge_pos,
        recursive_states.edge_dir,
        recursive_states.edge_t_min,
        recursive_states.edge_t_max,
        recursive_states.n0,
        recursive_states.n1,
        recursive_states.prim0,
        recursive_states.prim1,
        recursive_states.exterior_angle,
        1,
    )
    grid_cell_count = int(grid.resolution0) * int(grid.resolution1)
    return DfrAccum(grid_cell_count, *values[:14])


def accum_dfr_coherent_direct_native(
    scene_handle: int,
    states: DfrStates,
    grid: DfrGrid,
    material: DfrMaterial,
    *,
    active: torch.Tensor,
    wavelength: float,
    select_diffraction_point: bool = True,
    prefilter_visibility: bool = True,
) -> DfrCoherentAccum:
    if _C is None:
        raise RuntimeError("RayDTorch extension is not built yet.")
    states = _contig_states(states)
    material = _contig_material(material)
    values = _C.diffraction_coherent_accumulation_forward(
        int(scene_handle),
        active.contiguous(),
        states.edge_index,
        states.edge_pos,
        states.edge_dir,
        states.edge_t_min,
        states.edge_t_max,
        states.n0,
        states.n1,
        states.prim0,
        states.prim1,
        states.exterior_angle,
        states.src,
        states.src_power,
        states.wi,
        states.d0,
        material.eta_r,
        material.sigma,
        material.mu_r,
        material.gain,
        material.valid,
        int(grid.axis),
        float(grid.position),
        float(grid.coord0_min),
        float(grid.coord0_max),
        float(grid.coord1_min),
        float(grid.coord1_max),
        int(grid.resolution0),
        int(grid.resolution1),
        grid.resolved_cell_area(),
        float(wavelength),
        bool(select_diffraction_point),
        bool(prefilter_visibility),
    )
    grid_cell_count = int(grid.resolution0) * int(grid.resolution1)
    return DfrCoherentAccum(grid_cell_count, *values)


class NativeOpUnavailable(RuntimeError):
    pass
