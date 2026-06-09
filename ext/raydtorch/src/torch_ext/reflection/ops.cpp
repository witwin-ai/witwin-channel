#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <raydtorch/diffraction/accum_params.h>
#include <raydtorch/diffraction/accum_ad.h>
#include <raydtorch/diffraction/paths_params.h>
#include <raydtorch/diffraction/pipeline.h>
#include <raydtorch/scene/geometry_kernels.h>
#include <raydtorch/common/optix_pipeline.h>
#include <raydtorch/reflection/kernels.h>
#include <raydtorch/reflection/pipeline.h>
#include <raydtorch/common/optix_context.h>
#include <raydtorch/reflection/accum_reduce.h>
#include <raydtorch/reflection/accum_params.h>
#include <raydtorch/reflection/dedup.h>
#include <raydtorch/reflection/epc_field.h>
#include <raydtorch/reflection/epc_params.h>
#include <raydtorch/reflection/trace_params.h>
#include <raydtorch/reflection/visibility_params.h>
#include <raydtorch/scene/cache.h>
#include <raydtorch/common/tensor_check.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace raydtorch {

namespace {

constexpr int64_t kStagedReflAccumMinSamples = 2048;
constexpr int64_t kStagedReflAccumMinSamplesPerCell = 4;

void require_same_batch(const at::Tensor &a, const at::Tensor &b, const char *name) {
    if (a.size(0) != b.size(0))
        throw std::runtime_error(std::string(name) + " tensors must have the same batch size.");
}

void require_flat_i32(const at::Tensor &tensor, const char *name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kInt, name);
    require_rank(tensor, 1, name);
}

void require_flat_f32(const at::Tensor &tensor, const char *name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 1, name);
}

void require_state_width(const at::Tensor &tensor, int64_t state_count, const char *name) {
    if (tensor.size(0) < state_count)
        throw std::runtime_error(std::string(name) + " must cover state_count.");
}

int32_t checked_i32(int64_t value, const char *name) {
    if (value < 0 || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error(std::string(name) + " does not fit in int32.");
    return static_cast<int32_t>(value);
}

at::Tensor active_mask_for_states(const at::Tensor &active, int64_t state_count, const char *name) {
    if (active.size(0) == state_count)
        return active.contiguous();
    if (active.size(0) == 1)
        return active.expand({state_count}).contiguous();
    throw std::runtime_error(std::string(name) + " active width must be 1 or match state_count.");
}

at::Tensor first_bounce_column(const at::Tensor &value, int64_t ray_count) {
    if (value.dim() == 1)
        return value.reshape({ray_count}).contiguous();
    return value.slice(1, 0, 1).reshape({ray_count}).contiguous();
}

struct Vec3SoA {
    at::Tensor x;
    at::Tensor y;
    at::Tensor z;
};

Vec3SoA split_vec3(const at::Tensor &value) {
    return {
        value.select(1, 0).contiguous(),
        value.select(1, 1).contiguous(),
        value.select(1, 2).contiguous(),
    };
}

struct TriangleSoA {
    at::Tensor p0_x;
    at::Tensor p0_y;
    at::Tensor p0_z;
    at::Tensor e1_x;
    at::Tensor e1_y;
    at::Tensor e1_z;
    at::Tensor e2_x;
    at::Tensor e2_y;
    at::Tensor e2_z;
    at::Tensor fn_x;
    at::Tensor fn_y;
    at::Tensor fn_z;
    at::Tensor p0_packed;
    at::Tensor e1_packed;
    at::Tensor e2_packed;
    at::Tensor fn_packed;
    at::Tensor face_offsets;
    int32_t n_triangles = 0;
};

TriangleSoA make_triangle_soa(const MeshRecord &mesh) {
    at::Tensor faces_i64 = mesh.faces.to(at::kLong);
    at::Tensor v0 = mesh.vertices.index_select(0, faces_i64.select(1, 0)).contiguous();
    at::Tensor v1 = mesh.vertices.index_select(0, faces_i64.select(1, 1)).contiguous();
    at::Tensor v2 = mesh.vertices.index_select(0, faces_i64.select(1, 2)).contiguous();
    at::Tensor e1 = (v1 - v0).contiguous();
    at::Tensor e2 = (v2 - v0).contiguous();
    at::Tensor fn = at::cross(e1, e2, 1).contiguous();
    return {
        v0.select(1, 0).contiguous(),
        v0.select(1, 1).contiguous(),
        v0.select(1, 2).contiguous(),
        e1.select(1, 0).contiguous(),
        e1.select(1, 1).contiguous(),
        e1.select(1, 2).contiguous(),
        e2.select(1, 0).contiguous(),
        e2.select(1, 1).contiguous(),
        e2.select(1, 2).contiguous(),
        fn.select(1, 0).contiguous(),
        fn.select(1, 1).contiguous(),
        fn.select(1, 2).contiguous(),
        at::empty({0, 4}, mesh.vertices.options()),
        at::empty({0, 4}, mesh.vertices.options()),
        at::empty({0, 4}, mesh.vertices.options()),
        at::empty({0, 4}, mesh.vertices.options()),
        at::zeros({1}, mesh.faces.options()),
        static_cast<int32_t>(mesh.faces.size(0)),
    };
}

TriangleSoA make_scene_triangle_soa(const SceneCache &scene) {
    return {
        scene.tri_p0_x,
        scene.tri_p0_y,
        scene.tri_p0_z,
        scene.tri_e1_x,
        scene.tri_e1_y,
        scene.tri_e1_z,
        scene.tri_e2_x,
        scene.tri_e2_y,
        scene.tri_e2_z,
        scene.tri_fn_x,
        scene.tri_fn_y,
        scene.tri_fn_z,
        scene.tri_p0_packed,
        scene.tri_e1_packed,
        scene.tri_e2_packed,
        scene.tri_fn_packed,
        scene.face_offsets.contiguous(),
        static_cast<int32_t>(scene.global_faces.size(0)),
    };
}

const uint8_t *mask_ptr(const at::Tensor &mask) {
    return reinterpret_cast<const uint8_t *>(mask.data_ptr<bool>());
}

uint8_t *mutable_mask_ptr(const at::Tensor &mask) {
    return reinterpret_cast<uint8_t *>(mask.data_ptr<bool>());
}

at::Tensor stack_vec3(const at::Tensor &x, const at::Tensor &y, const at::Tensor &z) {
    return at::stack({x, y, z}, 1).contiguous();
}

std::shared_ptr<OptixLaunchPipeline> optix_pipeline_for_scene(
    const SceneCache &scene,
    const OptixPipelineConfig &config) {
    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    return shared_optix_launch_pipeline(
        optix_entry.optix_context,
        static_cast<int>(scene.device_index),
        1,
        config);
}

} // namespace

py::tuple visibility_forward_op(
    int64_t scene_handle,
    at::Tensor start,
    at::Tensor end,
    at::Tensor active) {
    require_vec3f(start, "start");
    require_vec3f(end, "end");
    require_mask(active, "active");
    require_same_batch(start, end, "visibility");
    if (active.size(0) != start.size(0))
        throw std::runtime_error("active must match the visibility batch size.");

    SceneCache &scene = get_scene(scene_handle);
    const int64_t ray_count = start.size(0);
    at::Tensor visible = at::empty({ray_count}, active.options());
    at::Tensor blocker_prim = at::full({ray_count}, -1, scene.global_faces.options());
    at::Tensor tape_t = at::full(
        {ray_count},
        std::numeric_limits<float>::infinity(),
        start.options());
    if (ray_count == 0)
        return py::make_tuple(visible, blocker_prim, tape_t);

    Vec3SoA start_soa = split_vec3(start);
    Vec3SoA end_soa = split_vec3(end);
    at::Tensor active_contig = active.contiguous();

    SegmentVisibilityParams params = {};
    params.handle = scene.triangle_ias.traversable;
    params.face_offsets = scene.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.start_x = start_soa.x.data_ptr<float>();
    params.start_y = start_soa.y.data_ptr<float>();
    params.start_z = start_soa.z.data_ptr<float>();
    params.end_x = end_soa.x.data_ptr<float>();
    params.end_y = end_soa.y.data_ptr<float>();
    params.end_z = end_soa.z.data_ptr<float>();
    params.active_mask = mask_ptr(active_contig);
    params.n_rays = static_cast<int32_t>(ray_count);
    params.out_visible = mutable_mask_ptr(visible);
    params.out_first_blocked_prim = blocker_prim.data_ptr<int>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    optix_pipeline_for_scene(scene, segment_visibility_pipeline_config())
        ->launch(0, params, static_cast<unsigned int>(ray_count), torch_ctx.stream);
    return py::make_tuple(visible, blocker_prim, tape_t);
}

py::tuple trace_reflections_forward_impl(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t max_bounces,
    bool export_tape) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    require_same_batch(ray_o, ray_d, "trace_reflections");
    if (ray_tmax.size(0) != ray_o.size(0) || active.size(0) != ray_o.size(0))
        throw std::runtime_error("ray_tmax and active must match the ray batch size.");
    if (max_bounces < 1)
        throw std::runtime_error("max_bounces must be at least 1.");

    SceneCache &scene = get_scene(scene_handle);
    const int64_t ray_count = ray_o.size(0);
    auto fopts = ray_o.options();
    auto iopts = scene.global_faces.options();
    const bool bounce_major_outputs = ray_count > 0 && max_bounces > 1;
    const int64_t trace_rows = bounce_major_outputs ? max_bounces : ray_count;
    const int64_t trace_cols = bounce_major_outputs ? ray_count : max_bounces;
    at::Tensor t_storage = at::full(
        {trace_rows, trace_cols},
        std::numeric_limits<float>::infinity(),
        fopts);
    at::Tensor prim_ids_storage = at::full({trace_rows, trace_cols}, -1, iopts);
    at::Tensor bounce_count = at::zeros({ray_count}, iopts);
    at::Tensor img_x_storage = at::zeros({trace_rows, trace_cols}, fopts);
    at::Tensor img_y_storage = at::zeros({trace_rows, trace_cols}, fopts);
    at::Tensor img_z_storage = at::zeros({trace_rows, trace_cols}, fopts);
    at::Tensor local_prim_ids;
    at::Tensor shape_ids;
    at::Tensor bary_u;
    at::Tensor bary_v;
    at::Tensor hit_x;
    at::Tensor hit_y;
    at::Tensor hit_z;
    at::Tensor norm_x;
    at::Tensor norm_y;
    at::Tensor norm_z;
    if (export_tape) {
        local_prim_ids = at::full({trace_rows, trace_cols}, -1, iopts);
        shape_ids = at::zeros({trace_rows, trace_cols}, iopts);
        bary_u = at::zeros({trace_rows, trace_cols}, fopts);
        bary_v = at::zeros({trace_rows, trace_cols}, fopts);
        hit_x = at::zeros({trace_rows, trace_cols}, fopts);
        hit_y = at::zeros({trace_rows, trace_cols}, fopts);
        hit_z = at::zeros({trace_rows, trace_cols}, fopts);
        norm_x = at::zeros({trace_rows, trace_cols}, fopts);
        norm_y = at::zeros({trace_rows, trace_cols}, fopts);
        norm_z = at::zeros({trace_rows, trace_cols}, fopts);
    }
    if (ray_count == 0) {
        at::Tensor valid = at::zeros({ray_count, max_bounces}, active.options());
        at::Tensor image_sources = at::zeros({ray_count, max_bounces, 3}, fopts);
        if (!export_tape)
            return py::make_tuple(valid, t_storage, image_sources, prim_ids_storage);
        at::Tensor tape_prim_id = at::full({ray_count, max_bounces}, -1, iopts);
        at::Tensor tape_barycentric = at::zeros({ray_count, max_bounces, 3}, fopts);
        at::Tensor tape_t = at::full(
            {ray_count, max_bounces},
            std::numeric_limits<float>::infinity(),
            fopts);
        at::Tensor tape_hit_points = at::zeros({ray_count, max_bounces, 3}, fopts);
        at::Tensor tape_normals = at::zeros({ray_count, max_bounces, 3}, fopts);
        return py::make_tuple(
            valid,
            t_storage,
            image_sources,
            prim_ids_storage,
            tape_prim_id,
            tape_barycentric,
            tape_t,
            tape_hit_points,
            tape_normals);
    }

    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA ray_o_soa = split_vec3(ray_o);
    Vec3SoA ray_d_soa = split_vec3(ray_d);
    at::Tensor ray_tmax_contig = ray_tmax.contiguous();
    at::Tensor active_contig = active.contiguous();

    TorchCudaContext torch_ctx = current_torch_cuda_context();

    ReflectionTraceParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.tri_p0_packed = reinterpret_cast<const float4 *>(tri.p0_packed.data_ptr<float>());
    params.tri_e1_packed = reinterpret_cast<const float4 *>(tri.e1_packed.data_ptr<float>());
    params.tri_e2_packed = reinterpret_cast<const float4 *>(tri.e2_packed.data_ptr<float>());
    params.tri_fn_packed = reinterpret_cast<const float4 *>(tri.fn_packed.data_ptr<float>());
    params.face_offsets = tri.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.n_triangles = tri.n_triangles;
    params.ray_ox = ray_o_soa.x.data_ptr<float>();
    params.ray_oy = ray_o_soa.y.data_ptr<float>();
    params.ray_oz = ray_o_soa.z.data_ptr<float>();
    params.ray_dx = ray_d_soa.x.data_ptr<float>();
    params.ray_dy = ray_d_soa.y.data_ptr<float>();
    params.ray_dz = ray_d_soa.z.data_ptr<float>();
    params.ray_tmax = ray_tmax_contig.data_ptr<float>();
    params.active_mask = mask_ptr(active_contig);
    params.n_rays = static_cast<int32_t>(ray_count);
    params.max_bounces = static_cast<int32_t>(max_bounces);
    params.export_mode = 0;
    params.return_trailing = 0;
    params.output_layout = bounce_major_outputs ? 1 : 0;
    params.out_bounce_count = bounce_count.data_ptr<int>();
    params.out_shape_ids = export_tape ? shape_ids.data_ptr<int>() : nullptr;
    params.out_prim_ids = export_tape ? local_prim_ids.data_ptr<int>() : nullptr;
    params.out_global_prim_ids = prim_ids_storage.data_ptr<int>();
    params.out_t = t_storage.data_ptr<float>();
    params.out_bary_u = export_tape ? bary_u.data_ptr<float>() : nullptr;
    params.out_bary_v = export_tape ? bary_v.data_ptr<float>() : nullptr;
    params.out_hit_x = export_tape ? hit_x.data_ptr<float>() : nullptr;
    params.out_hit_y = export_tape ? hit_y.data_ptr<float>() : nullptr;
    params.out_hit_z = export_tape ? hit_z.data_ptr<float>() : nullptr;
    params.out_norm_x = export_tape ? norm_x.data_ptr<float>() : nullptr;
    params.out_norm_y = export_tape ? norm_y.data_ptr<float>() : nullptr;
    params.out_norm_z = export_tape ? norm_z.data_ptr<float>() : nullptr;
    params.out_img_x = img_x_storage.data_ptr<float>();
    params.out_img_y = img_y_storage.data_ptr<float>();
    params.out_img_z = img_z_storage.data_ptr<float>();

    optix_pipeline_for_scene(scene, reflection_trace_pipeline_config())
        ->launch(0, params, static_cast<unsigned int>(ray_count), torch_ctx.stream);

    at::Tensor bounce_index =
        at::arange(max_bounces, at::TensorOptions().device(ray_o.device()).dtype(at::kLong))
            .reshape({1, max_bounces});
    at::Tensor valid = bounce_index.lt(bounce_count.to(at::kLong).reshape({ray_count, 1}));
    auto ray_major = [&](const at::Tensor &tensor) {
        return bounce_major_outputs ? tensor.transpose(0, 1).contiguous() : tensor;
    };
    at::Tensor t = ray_major(t_storage);
    at::Tensor prim_ids = ray_major(prim_ids_storage);
    at::Tensor img_x = ray_major(img_x_storage);
    at::Tensor img_y = ray_major(img_y_storage);
    at::Tensor img_z = ray_major(img_z_storage);
    at::Tensor image_sources = at::stack({img_x, img_y, img_z}, 2).contiguous();
    if (!export_tape)
        return py::make_tuple(valid, t, image_sources, prim_ids);
    local_prim_ids = ray_major(local_prim_ids);
    shape_ids = ray_major(shape_ids);
    bary_u = ray_major(bary_u);
    bary_v = ray_major(bary_v);
    hit_x = ray_major(hit_x);
    hit_y = ray_major(hit_y);
    hit_z = ray_major(hit_z);
    norm_x = ray_major(norm_x);
    norm_y = ray_major(norm_y);
    norm_z = ray_major(norm_z);
    at::Tensor tape_prim_id = prim_ids.contiguous();
    at::Tensor tape_barycentric = at::stack(
        {
            (1.0f - bary_u - bary_v),
            bary_u,
            bary_v,
        },
        2).contiguous();
    at::Tensor tape_t = t.clone();
    at::Tensor tape_hit_points = at::stack({hit_x, hit_y, hit_z}, 2).contiguous();
    at::Tensor tape_normals = at::stack({norm_x, norm_y, norm_z}, 2).contiguous();

    return py::make_tuple(
        valid,
        t,
        image_sources,
        prim_ids,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        tape_hit_points,
        tape_normals);
}

py::tuple trace_reflections_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t max_bounces) {
    return trace_reflections_forward_impl(
        scene_handle,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        max_bounces,
        true);
}

py::tuple trace_reflections_forward_noad_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t max_bounces) {
    return trace_reflections_forward_impl(
        scene_handle,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        max_bounces,
        false);
}

py::tuple trace_reflections_backward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_hit_points,
    at::Tensor tape_normals,
    at::Tensor image_sources,
    at::Tensor grad_t,
    at::Tensor grad_image_sources) {
    SceneCache &scene = get_scene(scene_handle);
    ReflectionBackwardOutputs out = reflection_chain_backward_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        image_sources,
        grad_t.contiguous(),
        grad_image_sources.contiguous());
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple trace_reflections_jvp_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_hit_points,
    at::Tensor tape_normals,
    at::Tensor tangent_vertices,
    at::Tensor tangent_ray_o,
    at::Tensor tangent_ray_d,
    at::Tensor image_sources) {
    SceneCache &scene = get_scene(scene_handle);
    ReflectionJvpOutputs out = reflection_chain_jvp_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        tangent_vertices.contiguous(),
        tangent_ray_o.contiguous(),
        tangent_ray_d.contiguous(),
        image_sources);
    return py::make_tuple(out.tangent_t, out.tangent_image_sources);
}

py::tuple trace_refl_epc_field_forward_op(
    int64_t scene_handle,
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor active,
    int64_t max_bounces) {
    require_vec3f(source, "source");
    require_vec3f(receiver, "receiver");
    require_mask(active, "active");
    require_same_batch(source, receiver, "trace_refl_epc_field");
    if (active.size(0) != source.size(0))
        throw std::runtime_error("active must match the EPC batch size.");
    if (max_bounces < 1)
        throw std::runtime_error("max_bounces must be at least 1.");

    SceneCache &scene = get_scene(scene_handle);
    const int64_t ray_count = source.size(0);
    const int64_t slot_count = ray_count * max_bounces;
    auto fopts = source.options();
    auto iopts = scene.global_faces.options();
    at::Tensor field_real = at::zeros({ray_count}, fopts);
    at::Tensor field_imag = at::zeros({ray_count}, fopts);
    at::Tensor path_length = at::full(
        {ray_count},
        std::numeric_limits<float>::infinity(),
        fopts);
    at::Tensor valid = at::zeros({ray_count}, active.options());
    at::Tensor resolved_first = at::full({ray_count}, -1, iopts);
    at::Tensor tape_prim_id = at::full({ray_count}, -1, iopts);
    at::Tensor tape_barycentric = at::zeros({ray_count, 3}, fopts);
    at::Tensor tape_t = at::full(
        {ray_count},
        std::numeric_limits<float>::infinity(),
        fopts);
    if (ray_count == 0) {
        return py::make_tuple(
            field_real,
            field_imag,
            path_length,
            valid,
            resolved_first,
            tape_prim_id,
            tape_barycentric,
            tape_t);
    }

    at::Tensor ray_d = (receiver - source).contiguous();
    at::Tensor ray_tmax = at::sqrt(at::sum(ray_d * ray_d, {1})).contiguous();
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA source_soa = split_vec3(source);
    Vec3SoA ray_d_soa = split_vec3(ray_d);
    Vec3SoA receiver_soa = split_vec3(receiver);
    at::Tensor active_contig = active.contiguous();

    at::Tensor epc_valid = at::zeros({ray_count}, active.options());
    at::Tensor epc_bounce_count = at::zeros({ray_count}, iopts);
    at::Tensor epc_path_length = at::full(
        {ray_count},
        std::numeric_limits<float>::infinity(),
        fopts);
    at::Tensor point_x = at::zeros({slot_count}, fopts);
    at::Tensor point_y = at::zeros({slot_count}, fopts);
    at::Tensor point_z = at::zeros({slot_count}, fopts);
    at::Tensor trace_prim_ids = at::full({slot_count}, -1, iopts);
    at::Tensor resolved_prim_ids = at::full({slot_count}, -1, iopts);
    at::Tensor surface_group_ids = at::full({slot_count}, -1, iopts);
    at::Tensor plane_normal_x = at::zeros({slot_count}, fopts);
    at::Tensor plane_normal_y = at::zeros({slot_count}, fopts);
    at::Tensor plane_normal_z = at::zeros({slot_count}, fopts);
    at::Tensor first_blocked_segment = at::full({ray_count}, -1, iopts);
    at::Tensor first_blocked_prim = at::full({ray_count}, -1, iopts);
    at::Tensor first_blocked_group = at::full({ray_count}, -1, iopts);

    ReflEpcParams epc_params = {};
    epc_params.primary_handle = scene.triangle_ias.traversable;
    epc_params.secondary_handle = 0;
    epc_params.split_mode = 0;
    epc_params.tri_p0_x = tri.p0_x.data_ptr<float>();
    epc_params.tri_p0_y = tri.p0_y.data_ptr<float>();
    epc_params.tri_p0_z = tri.p0_z.data_ptr<float>();
    epc_params.tri_e1_x = tri.e1_x.data_ptr<float>();
    epc_params.tri_e1_y = tri.e1_y.data_ptr<float>();
    epc_params.tri_e1_z = tri.e1_z.data_ptr<float>();
    epc_params.tri_e2_x = tri.e2_x.data_ptr<float>();
    epc_params.tri_e2_y = tri.e2_y.data_ptr<float>();
    epc_params.tri_e2_z = tri.e2_z.data_ptr<float>();
    epc_params.tri_fn_x = tri.fn_x.data_ptr<float>();
    epc_params.tri_fn_y = tri.fn_y.data_ptr<float>();
    epc_params.tri_fn_z = tri.fn_z.data_ptr<float>();
    epc_params.face_offsets = tri.face_offsets.data_ptr<int>();
    epc_params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    epc_params.n_triangles = tri.n_triangles;
    epc_params.visibility_ignore_mode = ReflEpcVisibilityIgnorePrimitive;
    epc_params.ray_ox = source_soa.x.data_ptr<float>();
    epc_params.ray_oy = source_soa.y.data_ptr<float>();
    epc_params.ray_oz = source_soa.z.data_ptr<float>();
    epc_params.ray_dx = ray_d_soa.x.data_ptr<float>();
    epc_params.ray_dy = ray_d_soa.y.data_ptr<float>();
    epc_params.ray_dz = ray_d_soa.z.data_ptr<float>();
    epc_params.ray_tmax = ray_tmax.data_ptr<float>();
    epc_params.rx_x = receiver_soa.x.data_ptr<float>();
    epc_params.rx_y = receiver_soa.y.data_ptr<float>();
    epc_params.rx_z = receiver_soa.z.data_ptr<float>();
    epc_params.rx_count = static_cast<int32_t>(ray_count);
    epc_params.active_mask = mask_ptr(active_contig);
    epc_params.n_rays = static_cast<int32_t>(ray_count);
    epc_params.max_bounces = static_cast<int32_t>(max_bounces);
    epc_params.out_valid = mutable_mask_ptr(epc_valid);
    epc_params.out_bounce_count = epc_bounce_count.data_ptr<int>();
    epc_params.out_path_length = epc_path_length.data_ptr<float>();
    epc_params.out_point_x = point_x.data_ptr<float>();
    epc_params.out_point_y = point_y.data_ptr<float>();
    epc_params.out_point_z = point_z.data_ptr<float>();
    epc_params.out_trace_prim_ids = trace_prim_ids.data_ptr<int>();
    epc_params.out_resolved_prim_ids = resolved_prim_ids.data_ptr<int>();
    epc_params.out_surface_group_ids = surface_group_ids.data_ptr<int>();
    epc_params.out_plane_normal_x = plane_normal_x.data_ptr<float>();
    epc_params.out_plane_normal_y = plane_normal_y.data_ptr<float>();
    epc_params.out_plane_normal_z = plane_normal_z.data_ptr<float>();
    epc_params.out_first_blocked_segment = first_blocked_segment.data_ptr<int>();
    epc_params.out_first_blocked_prim = first_blocked_prim.data_ptr<int>();
    epc_params.out_first_blocked_group = first_blocked_group.data_ptr<int>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    optix_pipeline_for_scene(scene, reflection_epc_pipeline_config())
        ->launch(0, epc_params, static_cast<unsigned int>(ray_count), torch_ctx.stream);

    at::Tensor slot_eta_r = at::ones({slot_count}, fopts);
    at::Tensor slot_mu_r = at::ones({slot_count}, fopts);
    at::Tensor slot_sigma = at::zeros({slot_count}, fopts);
    at::Tensor slot_gain = at::ones({slot_count}, fopts);
    at::Tensor tx_pol_x = at::ones({1}, fopts);
    at::Tensor tx_pol_y = at::zeros({1}, fopts);
    at::Tensor tx_pol_z = at::zeros({1}, fopts);
    at::Tensor field_y_re = at::zeros({ray_count}, fopts);
    at::Tensor field_y_im = at::zeros({ray_count}, fopts);
    at::Tensor field_z_re = at::zeros({ray_count}, fopts);
    at::Tensor field_z_im = at::zeros({ray_count}, fopts);

    ReflEpcFieldParams field_params = {};
    field_params.n_rays = static_cast<int32_t>(ray_count);
    field_params.max_bounces = static_cast<int32_t>(max_bounces);
    field_params.epc_valid = mask_ptr(epc_valid);
    field_params.epc_bounce_count = epc_bounce_count.data_ptr<int>();
    field_params.epc_path_length = epc_path_length.data_ptr<float>();
    field_params.ray_ox = source_soa.x.data_ptr<float>();
    field_params.ray_oy = source_soa.y.data_ptr<float>();
    field_params.ray_oz = source_soa.z.data_ptr<float>();
    field_params.rx_x = receiver_soa.x.data_ptr<float>();
    field_params.rx_y = receiver_soa.y.data_ptr<float>();
    field_params.rx_z = receiver_soa.z.data_ptr<float>();
    field_params.rx_count = static_cast<int32_t>(ray_count);
    field_params.hit_x = point_x.data_ptr<float>();
    field_params.hit_y = point_y.data_ptr<float>();
    field_params.hit_z = point_z.data_ptr<float>();
    field_params.epc_normal_x = plane_normal_x.data_ptr<float>();
    field_params.epc_normal_y = plane_normal_y.data_ptr<float>();
    field_params.epc_normal_z = plane_normal_z.data_ptr<float>();
    field_params.resolved_prim_ids = resolved_prim_ids.data_ptr<int>();
    field_params.surface_group_ids = surface_group_ids.data_ptr<int>();
    field_params.slot_normal_x = plane_normal_x.data_ptr<float>();
    field_params.slot_normal_y = plane_normal_y.data_ptr<float>();
    field_params.slot_normal_z = plane_normal_z.data_ptr<float>();
    field_params.slot_eta_r = slot_eta_r.data_ptr<float>();
    field_params.slot_mu_r = slot_mu_r.data_ptr<float>();
    field_params.slot_sigma = slot_sigma.data_ptr<float>();
    field_params.slot_gain = slot_gain.data_ptr<float>();
    field_params.tx_pol_x = tx_pol_x.data_ptr<float>();
    field_params.tx_pol_y = tx_pol_y.data_ptr<float>();
    field_params.tx_pol_z = tx_pol_z.data_ptr<float>();
    field_params.tx_pol_count = 1;
    field_params.omega = 2.0f * 3.14159265358979323846f * 299792458.0f;
    field_params.wavelength = 1.0f;
    field_params.out_valid = mutable_mask_ptr(valid);
    field_params.out_bounce_count = epc_bounce_count.data_ptr<int>();
    field_params.out_path_length = path_length.data_ptr<float>();
    field_params.out_field_x_re = field_real.data_ptr<float>();
    field_params.out_field_x_im = field_imag.data_ptr<float>();
    field_params.out_field_y_re = field_y_re.data_ptr<float>();
    field_params.out_field_y_im = field_y_im.data_ptr<float>();
    field_params.out_field_z_re = field_z_re.data_ptr<float>();
    field_params.out_field_z_im = field_z_im.data_ptr<float>();
    reflection_epc_field_gpu(field_params);

    resolved_first = resolved_prim_ids.reshape({ray_count, max_bounces})
                         .slice(1, 0, 1)
                         .reshape({ray_count})
                         .contiguous();
    tape_prim_id = trace_prim_ids.reshape({ray_count, max_bounces})
                       .slice(1, 0, 1)
                       .reshape({ray_count})
                       .contiguous();
    tape_t = path_length.contiguous();

    return py::make_tuple(
        field_real,
        field_imag,
        path_length,
        valid,
        resolved_first,
        tape_prim_id,
        tape_barycentric,
        tape_t);
}

py::tuple trace_refl_epc_field_backward_op(
    int64_t scene_handle,
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_t,
    at::Tensor grad_field_real,
    at::Tensor grad_field_imag,
    at::Tensor grad_path_length) {
    SceneCache &scene = get_scene(scene_handle);
    ReflEpcBackwardOutputs out = refl_epc_backward_cuda(
        scene.global_vertices,
        scene.global_faces,
        source,
        receiver,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        grad_field_real.contiguous(),
        grad_field_imag.contiguous(),
        grad_path_length.contiguous());
    return py::make_tuple(out.grad_vertices, out.grad_source, out.grad_receiver);
}

py::tuple trace_refl_epc_field_jvp_op(
    int64_t scene_handle,
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_t,
    at::Tensor tangent_vertices,
    at::Tensor tangent_source,
    at::Tensor tangent_receiver) {
    SceneCache &scene = get_scene(scene_handle);
    ReflEpcJvpOutputs out = refl_epc_jvp_cuda(
        scene.global_vertices,
        scene.global_faces,
        source,
        receiver,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        tangent_vertices.contiguous(),
        tangent_source.contiguous(),
        tangent_receiver.contiguous());
    return py::make_tuple(out.tangent_field_real, out.tangent_field_imag, out.tangent_path_length);
}

py::tuple reflection_dedup_forward_op(
    at::Tensor bounce_count,
    at::Tensor shape_ids,
    at::Tensor prim_ids,
    at::Tensor t,
    at::Tensor bary_u,
    at::Tensor bary_v,
    at::Tensor hit_x,
    at::Tensor hit_y,
    at::Tensor hit_z,
    at::Tensor norm_x,
    at::Tensor norm_y,
    at::Tensor norm_z,
    at::Tensor img_x,
    at::Tensor img_y,
    at::Tensor img_z,
    int64_t max_bounces,
    double image_source_tolerance) {
    if (bounce_count.dim() != 1)
        throw std::runtime_error("bounce_count must be flat.");
    if (max_bounces <= 0)
        throw std::runtime_error("max_bounces must be positive.");
    const int64_t ray_count = bounce_count.size(0);
    const int64_t slot_count = ray_count * max_bounces;
    auto iopts = bounce_count.options();
    auto fopts = t.options();
    at::Tensor out_bounce_count = at::zeros({ray_count}, iopts);
    at::Tensor out_shape_ids = at::full({slot_count}, -1, iopts);
    at::Tensor out_prim_ids = at::full({slot_count}, -1, iopts);
    at::Tensor out_t = at::full(
        {slot_count},
        std::numeric_limits<float>::infinity(),
        fopts);
    at::Tensor out_bary_u = at::zeros({slot_count}, fopts);
    at::Tensor out_bary_v = at::zeros({slot_count}, fopts);
    at::Tensor out_hit_x = at::zeros({slot_count}, fopts);
    at::Tensor out_hit_y = at::zeros({slot_count}, fopts);
    at::Tensor out_hit_z = at::zeros({slot_count}, fopts);
    at::Tensor out_norm_x = at::zeros({slot_count}, fopts);
    at::Tensor out_norm_y = at::zeros({slot_count}, fopts);
    at::Tensor out_norm_z = at::zeros({slot_count}, fopts);
    at::Tensor out_img_x = at::zeros({slot_count}, fopts);
    at::Tensor out_img_y = at::zeros({slot_count}, fopts);
    at::Tensor out_img_z = at::zeros({slot_count}, fopts);
    at::Tensor out_discovery_count = at::zeros({ray_count}, iopts);
    at::Tensor out_representative = at::full({ray_count}, -1, iopts);
    at::Tensor face_offsets = at::zeros({1}, iopts);

    int unique_count = 0;
    if (ray_count > 0) {
        at::Tensor bounce_count_c = bounce_count.contiguous();
        at::Tensor shape_ids_c = shape_ids.contiguous();
        at::Tensor prim_ids_c = prim_ids.contiguous();
        at::Tensor t_c = t.contiguous();
        at::Tensor bary_u_c = bary_u.contiguous();
        at::Tensor bary_v_c = bary_v.contiguous();
        at::Tensor hit_x_c = hit_x.contiguous();
        at::Tensor hit_y_c = hit_y.contiguous();
        at::Tensor hit_z_c = hit_z.contiguous();
        at::Tensor norm_x_c = norm_x.contiguous();
        at::Tensor norm_y_c = norm_y.contiguous();
        at::Tensor norm_z_c = norm_z.contiguous();
        at::Tensor img_x_c = img_x.contiguous();
        at::Tensor img_y_c = img_y.contiguous();
        at::Tensor img_z_c = img_z.contiguous();
        unique_count = reflection_dedup_gpu(
            static_cast<int32_t>(ray_count),
            static_cast<int32_t>(max_bounces),
            bounce_count_c.data_ptr<int>(),
            shape_ids_c.data_ptr<int>(),
            prim_ids_c.data_ptr<int>(),
            t_c.data_ptr<float>(),
            bary_u_c.data_ptr<float>(),
            bary_v_c.data_ptr<float>(),
            hit_x_c.data_ptr<float>(),
            hit_y_c.data_ptr<float>(),
            hit_z_c.data_ptr<float>(),
            norm_x_c.data_ptr<float>(),
            norm_y_c.data_ptr<float>(),
            norm_z_c.data_ptr<float>(),
            img_x_c.data_ptr<float>(),
            img_y_c.data_ptr<float>(),
            img_z_c.data_ptr<float>(),
            face_offsets.data_ptr<int>(),
            1,
            nullptr,
            0,
            static_cast<float>(image_source_tolerance),
            out_bounce_count.data_ptr<int>(),
            out_shape_ids.data_ptr<int>(),
            out_prim_ids.data_ptr<int>(),
            out_t.data_ptr<float>(),
            out_bary_u.data_ptr<float>(),
            out_bary_v.data_ptr<float>(),
            out_hit_x.data_ptr<float>(),
            out_hit_y.data_ptr<float>(),
            out_hit_z.data_ptr<float>(),
            out_norm_x.data_ptr<float>(),
            out_norm_y.data_ptr<float>(),
            out_norm_z.data_ptr<float>(),
            out_img_x.data_ptr<float>(),
            out_img_y.data_ptr<float>(),
            out_img_z.data_ptr<float>(),
            out_discovery_count.data_ptr<int>(),
            out_representative.data_ptr<int>());
    }

    return py::make_tuple(
        unique_count,
        out_bounce_count,
        out_shape_ids,
        out_prim_ids,
        out_t,
        out_bary_u,
        out_bary_v,
        out_hit_x,
        out_hit_y,
        out_hit_z,
        out_norm_x,
        out_norm_y,
        out_norm_z,
        out_img_x,
        out_img_y,
        out_img_z,
        out_discovery_count,
        out_representative);
}

py::tuple reflection_accumulation_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tx,
    at::Tensor tx_pol,
    int64_t max_bounces,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double wavelength) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    require_vec3f(tx, "tx");
    require_vec3f(tx_pol, "tx_pol");
    require_same_batch(ray_o, ray_d, "reflection_accumulation");
    require_same_batch(ray_o, tx, "reflection_accumulation");
    require_same_batch(ray_o, tx_pol, "reflection_accumulation");
    if (ray_tmax.size(0) != ray_o.size(0) || active.size(0) != ray_o.size(0))
        throw std::runtime_error("ray_tmax and active must match the ray batch size.");
    if (max_bounces < 0)
        throw std::runtime_error("max_bounces must be non-negative.");
    if (grid_axis < 0 || grid_axis > 2)
        throw std::runtime_error("grid_axis must be 0, 1, or 2.");
    if (grid_resolution0 <= 0 || grid_resolution1 <= 0)
        throw std::runtime_error("grid resolutions must be positive.");
    if (!(wavelength > 0.0))
        throw std::runtime_error("wavelength must be positive.");

    SceneCache &scene = get_scene(scene_handle);
    const int64_t ray_count = ray_o.size(0);
    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    const int32_t max_bounces_i = checked_i32(max_bounces, "max_bounces");
    const int64_t stage_depth_count = max_bounces + 1;
    const bool stage_sample_count_fits =
        ray_count <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()) /
                         std::max<int64_t>(stage_depth_count, 1);
    const int64_t stage_sample_count =
        stage_sample_count_fits ? ray_count * stage_depth_count : 0;
    const bool staged_accum =
        stage_sample_count_fits &&
        stage_sample_count >= kStagedReflAccumMinSamples &&
        stage_sample_count >= cell_count * kStagedReflAccumMinSamplesPerCell;
    auto fopts = ray_o.options();
    auto iopts = scene.global_faces.options();
    at::Tensor power = at::zeros({cell_count}, fopts);
    at::Tensor field_x_re = at::zeros({cell_count}, fopts);
    at::Tensor field_x_im = at::zeros({cell_count}, fopts);
    at::Tensor field_y_re = at::zeros({cell_count}, fopts);
    at::Tensor field_y_im = at::zeros({cell_count}, fopts);
    at::Tensor field_z_re = at::zeros({cell_count}, fopts);
    at::Tensor field_z_im = at::zeros({cell_count}, fopts);
    at::Tensor reflection_count = at::zeros({1}, iopts);
    if (ray_count == 0) {
        return py::make_tuple(
            power.reshape({grid_resolution1, grid_resolution0}),
            field_x_re.reshape({grid_resolution1, grid_resolution0}),
            field_x_im.reshape({grid_resolution1, grid_resolution0}),
            field_y_re.reshape({grid_resolution1, grid_resolution0}),
            field_y_im.reshape({grid_resolution1, grid_resolution0}),
            field_z_re.reshape({grid_resolution1, grid_resolution0}),
            field_z_im.reshape({grid_resolution1, grid_resolution0}),
            reflection_count);
    }

    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA ray_o_soa = split_vec3(ray_o);
    Vec3SoA ray_d_soa = split_vec3(ray_d);
    Vec3SoA tx_soa = split_vec3(tx);
    Vec3SoA tx_pol_soa = split_vec3(tx_pol);
    at::Tensor ray_tmax_contig = ray_tmax.contiguous();
    at::Tensor active_contig = active.contiguous();
    at::Tensor material_eta_r = at::ones({tri.n_triangles}, fopts);
    at::Tensor material_sigma = at::zeros({tri.n_triangles}, fopts);
    at::Tensor material_gain = at::ones({tri.n_triangles}, fopts);
    at::Tensor material_mu_r = at::ones({tri.n_triangles}, fopts);
    at::Tensor material_valid = at::ones({tri.n_triangles}, active.options());
    at::Tensor stage_cell = staged_accum
        ? at::full({stage_sample_count}, -1, iopts)
        : at::Tensor();
    at::Tensor stage_value = staged_accum
        ? at::zeros({stage_sample_count, 8}, fopts)
        : at::Tensor();

    AccumParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.face_offsets = tri.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.n_triangles = tri.n_triangles;
    params.ray_ox = ray_o_soa.x.data_ptr<float>();
    params.ray_oy = ray_o_soa.y.data_ptr<float>();
    params.ray_oz = ray_o_soa.z.data_ptr<float>();
    params.ray_dx = ray_d_soa.x.data_ptr<float>();
    params.ray_dy = ray_d_soa.y.data_ptr<float>();
    params.ray_dz = ray_d_soa.z.data_ptr<float>();
    params.ray_tmax = ray_tmax_contig.data_ptr<float>();
    params.active_mask = mask_ptr(active_contig);
    params.n_rays = static_cast<int32_t>(ray_count);
    params.tx_x = tx_soa.x.data_ptr<float>();
    params.tx_y = tx_soa.y.data_ptr<float>();
    params.tx_z = tx_soa.z.data_ptr<float>();
    params.tx_pol_x = tx_pol_soa.x.data_ptr<float>();
    params.tx_pol_y = tx_pol_soa.y.data_ptr<float>();
    params.tx_pol_z = tx_pol_soa.z.data_ptr<float>();
    params.max_bounces = max_bounces_i;
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.solid_angle_per_ray = 1.0f;
    const double span0 = grid_coord0_max - grid_coord0_min;
    const double span1 = grid_coord1_max - grid_coord1_min;
    params.cell_area = static_cast<float>(
        std::abs(span0 * span1) /
        static_cast<double>(grid_resolution0 * grid_resolution1));
    params.seed = 0;
    params.rr_depth = 0;
    params.rr_prob = 1.0f;
    params.stop_threshold = 0.0f;
    params.grid_axis = static_cast<int32_t>(grid_axis);
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = static_cast<int32_t>(grid_resolution0);
    params.grid_resolution1 = static_cast<int32_t>(grid_resolution1);
    params.material_eta_r = material_eta_r.data_ptr<float>();
    params.material_sigma = material_sigma.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_mu_r = material_mu_r.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.material_count = tri.n_triangles;
    params.collect_wedges = 0;
    params.collect_wedge_prefixes = 0;
    params.wedge_capacity = 0;
    params.wedge_sample_stride = 1;
    params.out_reflection_power = power.data_ptr<float>();
    params.out_field_x_re = field_x_re.data_ptr<float>();
    params.out_field_x_im = field_x_im.data_ptr<float>();
    params.out_field_y_re = field_y_re.data_ptr<float>();
    params.out_field_y_im = field_y_im.data_ptr<float>();
    params.out_field_z_re = field_z_re.data_ptr<float>();
    params.out_field_z_im = field_z_im.data_ptr<float>();
    params.out_reflection_count = reflection_count.data_ptr<int>();
    params.stage_cell = staged_accum ? stage_cell.data_ptr<int>() : nullptr;
    params.stage_value = staged_accum
        ? reinterpret_cast<ReflAccumStagedValue *>(stage_value.data_ptr<float>())
        : nullptr;

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    optix_pipeline_for_scene(scene, reflection_accumulation_pipeline_config())
        ->launch(0, params, static_cast<unsigned int>(ray_count), torch_ctx.stream);
    if (staged_accum) {
        reduce_refl_accum_staged_cuda(
            stage_sample_count,
            stage_cell,
            stage_value,
            power,
            field_x_re,
            field_x_im,
            field_y_re,
            field_y_im,
            field_z_re,
            field_z_im,
            reflection_count);
    }
    return py::make_tuple(
        power.reshape({grid_resolution1, grid_resolution0}),
        field_x_re.reshape({grid_resolution1, grid_resolution0}),
        field_x_im.reshape({grid_resolution1, grid_resolution0}),
        field_y_re.reshape({grid_resolution1, grid_resolution0}),
        field_y_im.reshape({grid_resolution1, grid_resolution0}),
        field_z_re.reshape({grid_resolution1, grid_resolution0}),
        field_z_im.reshape({grid_resolution1, grid_resolution0}),
        reflection_count);
}

void bind_reflection_ops(py::module_ &m) {
    m.def("visibility_forward", &visibility_forward_op);
    m.def("trace_reflections_forward", &trace_reflections_forward_op);
    m.def("trace_reflections_forward_noad", &trace_reflections_forward_noad_op);
    m.def("trace_reflections_backward", &trace_reflections_backward_op);
    m.def("trace_reflections_jvp", &trace_reflections_jvp_op);
    m.def("trace_refl_epc_field_forward", &trace_refl_epc_field_forward_op);
    m.def("trace_refl_epc_field_backward", &trace_refl_epc_field_backward_op);
    m.def("trace_refl_epc_field_jvp", &trace_refl_epc_field_jvp_op);
    m.def("reflection_dedup_forward", &reflection_dedup_forward_op);
    m.def("reflection_accumulation_forward", &reflection_accumulation_forward_op);
}

} // namespace raydtorch
