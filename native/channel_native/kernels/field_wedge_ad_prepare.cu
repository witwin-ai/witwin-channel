#include "field_wedge_ad_common.cuh"

namespace {

// ---------------------------------------------------------------------------
// Coupled stationary-geometry companions (fixed winner): the interaction
// points of a coupled path move with the endpoints. The primal re-solve is
// cn_coupled_rd_prepare_cuda (coupled_topology.cu); these duals mirror its
// row math with the wall plane and edge line frozen.
// ---------------------------------------------------------------------------

struct PrepareRowInputs {
    field::float3a source;
    field::float3a receiver;
    field::float3a plane_point;
    field::float3a plane_normal_unit;  // normalized like the primal kernel
    field::float3a edge_origin;
    field::float3a edge_dir_unit;
};

__device__ __forceinline__ PrepareRowInputs load_prepare_row(
    int64_t index,
    const float* source,
    const float* receiver,
    const float* plane_point,
    const float* plane_normal,
    const float* edge_pos,
    const float* edge_dir,
    const float* edge_t_min) {
    PrepareRowInputs in;
    in.source = load3f(source, index);
    in.receiver = load3f(receiver, index);
    in.plane_point = load3f(plane_point, index);
    // Primal normalize3: v / |v| when |v| > 1e-6, else zero.
    const field::float3a raw_n = load3f(plane_normal, index);
    const float n_len = sqrtf(field::f3_dot(raw_n, raw_n));
    in.plane_normal_unit = n_len > 1.0e-6f
                               ? field::f3_mul(raw_n, 1.0f / n_len)
                               : field::f3_zero();
    const field::float3a raw_d = load3f(edge_dir, index);
    const float d_len = sqrtf(field::f3_dot(raw_d, raw_d));
    in.edge_dir_unit = d_len > 1.0e-6f ? field::f3_mul(raw_d, 1.0f / d_len)
                                       : field::f3_zero();
    in.edge_origin = field::f3_add(
        load3f(edge_pos, index),
        field::f3_mul(in.edge_dir_unit, edge_t_min[index]));
    return in;
}

// Dual of one prepare row: edge stationary point and predicted reflection
// point as functions of (source, receiver) with the plane and edge frozen.
__device__ void prepare_row_dual(
    const PrepareRowInputs& in,
    field::float3a seed_source,
    field::float3a seed_receiver,
    DualV3& edge_point,
    DualV3& reflection_point) {
    const DualV3 src = field::dual_seed(in.source, seed_source);
    const DualV3 rcv = field::dual_seed(in.receiver, seed_receiver);
    const DualV3 normal = field::dual_const3(in.plane_normal_unit);
    const DualV3 p0 = field::dual_const3(in.plane_point);
    const DualV3 direction = field::dual_const3(in.edge_dir_unit);
    const DualV3 origin = field::dual_const3(in.edge_origin);
    const Dual signed_distance = field::f3_dot(field::f3_sub(src, p0), normal);
    const DualV3 image = field::f3_sub(
        src, field::f3_mul(normal, 2.0f * signed_distance));
    const Dual parameter = field::first_order_diffraction_parameter(
        image, rcv, origin, direction);
    edge_point = field::f3_add(origin, field::f3_mul(direction, parameter));
    const DualV3 image_to_edge = field::f3_sub(edge_point, image);
    const Dual plane_denominator = field::f3_dot(image_to_edge, normal);
    const Dual plane_parameter =
        field::f3_dot(field::f3_sub(p0, image), normal) / plane_denominator;
    reflection_point = field::f3_add(
        image, field::f3_mul(image_to_edge, plane_parameter));
}

__global__ void coupled_prepare_backward_kernel(
    int64_t count,
    const float* source,
    const float* receiver,
    const float* plane_point,
    const float* plane_normal,
    const float* edge_pos,
    const float* edge_dir,
    const float* edge_t_min,
    const float* grad_edge_point,
    const float* grad_reflection_point,
    float* grad_source,
    float* grad_receiver) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const PrepareRowInputs in = load_prepare_row(
            index, source, receiver, plane_point, plane_normal, edge_pos,
            edge_dir, edge_t_min);
        const int64_t base = index * 3;
        const field::float3a g_ep = grad_edge_point != nullptr
                                        ? load3f(grad_edge_point, index)
                                        : field::f3_zero();
        const field::float3a g_rp = grad_reflection_point != nullptr
                                        ? load3f(grad_reflection_point, index)
                                        : field::f3_zero();
        for (int slot = 0; slot < 2; ++slot) {
            float* out = slot == 0 ? grad_source : grad_receiver;
            if (out == nullptr)
                continue;
            for (int axis = 0; axis < 3; ++axis) {
                field::float3a seed_src = field::f3_zero();
                field::float3a seed_rcv = field::f3_zero();
                float* seed = slot == 0
                                  ? (axis == 0 ? &seed_src.x
                                               : axis == 1 ? &seed_src.y : &seed_src.z)
                                  : (axis == 0 ? &seed_rcv.x
                                               : axis == 1 ? &seed_rcv.y : &seed_rcv.z);
                *seed = 1.f;
                DualV3 edge_point;
                DualV3 reflection_point;
                prepare_row_dual(in, seed_src, seed_rcv, edge_point, reflection_point);
                out[base + axis] =
                    g_ep.x * edge_point.x.d + g_ep.y * edge_point.y.d +
                    g_ep.z * edge_point.z.d + g_rp.x * reflection_point.x.d +
                    g_rp.y * reflection_point.y.d + g_rp.z * reflection_point.z.d;
            }
        }
    }
}

__global__ void coupled_prepare_jvp_kernel(
    int64_t count,
    const float* source,
    const float* receiver,
    const float* plane_point,
    const float* plane_normal,
    const float* edge_pos,
    const float* edge_dir,
    const float* edge_t_min,
    const float* tangent_source,
    const float* tangent_receiver,
    float* tangent_edge_point,
    float* tangent_reflection_point) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const PrepareRowInputs in = load_prepare_row(
            index, source, receiver, plane_point, plane_normal, edge_pos,
            edge_dir, edge_t_min);
        const field::float3a seed_src = tangent_source != nullptr
                                            ? load3f(tangent_source, index)
                                            : field::f3_zero();
        const field::float3a seed_rcv = tangent_receiver != nullptr
                                            ? load3f(tangent_receiver, index)
                                            : field::f3_zero();
        DualV3 edge_point;
        DualV3 reflection_point;
        prepare_row_dual(in, seed_src, seed_rcv, edge_point, reflection_point);
        const int64_t base = index * 3;
        tangent_edge_point[base] = edge_point.x.d;
        tangent_edge_point[base + 1] = edge_point.y.d;
        tangent_edge_point[base + 2] = edge_point.z.d;
        tangent_reflection_point[base] = reflection_point.x.d;
        tangent_reflection_point[base + 1] = reflection_point.y.d;
        tangent_reflection_point[base + 2] = reflection_point.z.d;
    }
}

}  // namespace

namespace {

void check_prepare_rows(
    const at::Tensor& source,
    const at::Tensor& receiver,
    const at::Tensor& plane_point,
    const at::Tensor& plane_normal,
    const at::Tensor& edge_pos,
    const at::Tensor& edge_dir,
    const at::Tensor& edge_t_min) {
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    const int64_t count = source.size(0);
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {source, "source"},
             {receiver, "receiver"},
             {plane_point, "plane_point"},
             {plane_normal, "plane_normal"},
             {edge_pos, "edge_pos"},
             {edge_dir, "edge_dir"}}) {
        check_vec3_table(named.first, named.second);
        TORCH_CHECK(named.first.size(0) == count,
                    named.second, " must match source rows");
    }
    check_flat_tensor(edge_t_min, "edge_t_min", at::kFloat);
    TORCH_CHECK(edge_t_min.size(0) == count, "edge_t_min must match source rows");
}

}  // namespace

pybind11::dict cn_coupled_rd_prepare_backward(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor plane_point,
    at::Tensor plane_normal,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_t_min,
    pybind11::object grad_edge_point,
    pybind11::object grad_reflection_point,
    bool need_grad_source,
    bool need_grad_receiver) {
    check_prepare_rows(
        source, receiver, plane_point, plane_normal, edge_pos, edge_dir,
        edge_t_min);
    const int64_t count = source.size(0);
    at::Tensor grad_storage[2];
    const at::Tensor* g_edge_point = optional_tensor_arg(
        std::move(grad_edge_point), grad_storage[0], "grad_edge_point",
        at::kFloat, {count, 3}, source);
    const at::Tensor* g_reflection_point = optional_tensor_arg(
        std::move(grad_reflection_point), grad_storage[1],
        "grad_reflection_point", at::kFloat, {count, 3}, source);
    at::Tensor grad_source;
    at::Tensor grad_receiver;
    at::Tensor* grad_source_ptr = nullptr;
    at::Tensor* grad_receiver_ptr = nullptr;
    if (need_grad_source) {
        grad_source = at::empty({count, 3}, source.options());
        grad_source_ptr = &grad_source;
    }
    if (need_grad_receiver) {
        grad_receiver = at::empty({count, 3}, source.options());
        grad_receiver_ptr = &grad_receiver;
    }
    if (count > 0 && (g_edge_point != nullptr || g_reflection_point != nullptr)) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_prepare_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<float>(),
            receiver.data_ptr<float>(),
            plane_point.data_ptr<float>(),
            plane_normal.data_ptr<float>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            edge_t_min.data_ptr<float>(),
            opt_ptr<float>(g_edge_point),
            opt_ptr<float>(g_reflection_point),
            opt_mut_ptr<float>(grad_source_ptr),
            opt_mut_ptr<float>(grad_receiver_ptr));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        if (grad_source_ptr != nullptr)
            grad_source_ptr->zero_();
        if (grad_receiver_ptr != nullptr)
            grad_receiver_ptr->zero_();
    }
    pybind11::dict out;
    out["grad_source"] = grad_source_ptr != nullptr
                             ? pybind11::cast(grad_source)
                             : pybind11::object(pybind11::none());
    out["grad_receiver"] = grad_receiver_ptr != nullptr
                               ? pybind11::cast(grad_receiver)
                               : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_coupled_rd_prepare_jvp(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor plane_point,
    at::Tensor plane_normal,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_t_min,
    pybind11::object tangent_source,
    pybind11::object tangent_receiver) {
    check_prepare_rows(
        source, receiver, plane_point, plane_normal, edge_pos, edge_dir,
        edge_t_min);
    const int64_t count = source.size(0);
    at::Tensor storage[2];
    const at::Tensor* t_source = optional_tensor_arg(
        std::move(tangent_source), storage[0], "tangent_source", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_receiver = optional_tensor_arg(
        std::move(tangent_receiver), storage[1], "tangent_receiver", at::kFloat,
        {count, 3}, source);
    auto tangent_edge_point = at::empty({count, 3}, source.options());
    auto tangent_reflection_point = at::empty({count, 3}, source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_prepare_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<float>(),
            receiver.data_ptr<float>(),
            plane_point.data_ptr<float>(),
            plane_normal.data_ptr<float>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            edge_t_min.data_ptr<float>(),
            opt_ptr<float>(t_source),
            opt_ptr<float>(t_receiver),
            tangent_edge_point.data_ptr<float>(),
            tangent_reflection_point.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_edge_point"] = tangent_edge_point;
    out["tangent_reflection_point"] = tangent_reflection_point;
    return out;
}
