#include "field_wedge_ad_common.cuh"

namespace {

// ---------------------------------------------------------------------------
// field_project_complex3 companions: coefficient = <field, axis(direction)>
// with axis = project_to_wedge_plane(rx_pol, direction) (F1 unnormalized
// transverse of p_rx); path_gain = |coeff|^2.
// Linear in the field vector; direction feeds the axis.
// ---------------------------------------------------------------------------

__global__ void project_complex3_backward_kernel(
    int64_t count,
    const c10::complex<float>* field_vector,
    const float* direction,
    const float* rx_polarization,
    const c10::complex<float>* grad_coefficient,
    const float* grad_path_gain,
    c10::complex<float>* grad_field_vector,
    float* grad_direction) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        const field::Complex3 value = {
            from_c10(field_vector[base]),
            from_c10(field_vector[base + 1]),
            from_c10(field_vector[base + 2]),
        };
        const field::float3a dir = load3f(direction, index);
        const field::float3a pol = load3f(rx_polarization, index);
        // F1: coefficient = p_rx . E via the unnormalized transverse of p_rx.
        const field::float3a axis = field::project_to_wedge_plane(pol, dir);
        const field::Complex coefficient = transport::complex3_dot_real(value, axis);
        field::Complex g_coeff = field::cplx_zero();
        if (grad_coefficient != nullptr)
            g_coeff = from_c10(grad_coefficient[index]);
        if (grad_path_gain != nullptr) {
            const float g_gain = grad_path_gain[index];
            g_coeff.re += 2.0f * coefficient.re * g_gain;
            g_coeff.im += 2.0f * coefficient.im * g_gain;
        }
        field::Complex3 g_value = field::c3_zero();
        field::float3a g_axis = field::f3_zero();
        field::adj_cplx_dot_real(value, axis, g_coeff, g_value, g_axis);
        if (grad_field_vector != nullptr) {
            grad_field_vector[base] = to_c10(g_value.x);
            grad_field_vector[base + 1] = to_c10(g_value.y);
            grad_field_vector[base + 2] = to_c10(g_value.z);
        }
        if (grad_direction != nullptr) {
            field::float3a g_dir = field::f3_zero();
            field::float3a g_pol = field::f3_zero();
            ad::adj_transverse_project(dir, pol, g_axis, g_dir, g_pol);
            grad_direction[base] = g_dir.x;
            grad_direction[base + 1] = g_dir.y;
            grad_direction[base + 2] = g_dir.z;
        }
    }
}

__global__ void project_complex3_jvp_kernel(
    int64_t count,
    const c10::complex<float>* field_vector,
    const float* direction,
    const float* rx_polarization,
    const c10::complex<float>* tangent_field_vector,
    const float* tangent_direction,
    c10::complex<float>* tangent_coefficient,
    float* tangent_path_gain) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        const field::Complex3 value = {
            from_c10(field_vector[base]),
            from_c10(field_vector[base + 1]),
            from_c10(field_vector[base + 2]),
        };
        const field::float3a dir = load3f(direction, index);
        const field::float3a pol = load3f(rx_polarization, index);
        ad::DualF3 dir_dual = {
            dir,
            tangent_direction != nullptr ? load3f(tangent_direction, index)
                                         : field::f3_zero()};
        const ad::DualF3 axis = ad::dual_transverse_project(
            dir_dual, ad::df3_const(pol));
        const field::Complex3 t_value = {
            tangent_field_vector != nullptr ? from_c10(tangent_field_vector[base])
                                            : field::cplx_zero(),
            tangent_field_vector != nullptr
                ? from_c10(tangent_field_vector[base + 1])
                : field::cplx_zero(),
            tangent_field_vector != nullptr
                ? from_c10(tangent_field_vector[base + 2])
                : field::cplx_zero(),
        };
        const field::Complex coefficient = transport::complex3_dot_real(value, axis.v);
        const field::Complex t_coefficient = field::cplx_add(
            transport::complex3_dot_real(t_value, axis.v),
            transport::complex3_dot_real(value, axis.d));
        tangent_coefficient[index] = to_c10(t_coefficient);
        tangent_path_gain[index] = 2.0f * (coefficient.re * t_coefficient.re +
                                           coefficient.im * t_coefficient.im);
    }
}


}  // namespace

pybind11::dict cn_field_project_complex3_backward(
    at::Tensor field_vector,
    at::Tensor direction,
    at::Tensor rx_polarization,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_gain,
    bool need_grad_field_vector,
    bool need_grad_direction) {
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    check_tensor(field_vector, "field_vector", at::kComplexFloat, 2);
    TORCH_CHECK(field_vector.size(1) == 3, "field_vector must have shape (N, 3)");
    check_vec3_table(direction, "direction");
    check_vec3_table(rx_polarization, "rx_polarization");
    const int64_t count = field_vector.size(0);
    TORCH_CHECK(direction.size(0) == count && rx_polarization.size(0) == count,
                "projection tensors must match field_vector rows");
    at::Tensor grad_storage[2];
    const at::Tensor* g_coefficient = optional_tensor_arg(
        std::move(grad_coefficient), grad_storage[0], "grad_coefficient",
        at::kComplexFloat, {count}, field_vector);
    const at::Tensor* g_path_gain = optional_tensor_arg(
        std::move(grad_path_gain), grad_storage[1], "grad_path_gain",
        at::kFloat, {count}, field_vector);
    at::Tensor grad_field_vector;
    at::Tensor grad_direction;
    at::Tensor* grad_field_ptr = nullptr;
    at::Tensor* grad_direction_ptr = nullptr;
    if (need_grad_field_vector) {
        grad_field_vector = at::empty(
            {count, 3}, field_vector.options());
        grad_field_ptr = &grad_field_vector;
    }
    if (need_grad_direction) {
        grad_direction = at::empty({count, 3}, direction.options());
        grad_direction_ptr = &grad_direction;
    }
    if (count > 0 && (g_coefficient != nullptr || g_path_gain != nullptr)) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(direction.get_device()).stream();
        project_complex3_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            field_vector.data_ptr<c10::complex<float>>(),
            direction.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            opt_ptr<c10::complex<float>>(g_coefficient),
            opt_ptr<float>(g_path_gain),
            opt_mut_ptr<c10::complex<float>>(grad_field_ptr),
            opt_mut_ptr<float>(grad_direction_ptr));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        if (grad_field_ptr != nullptr)
            grad_field_ptr->zero_();
        if (grad_direction_ptr != nullptr)
            grad_direction_ptr->zero_();
    }
    pybind11::dict out;
    out["grad_field_vector"] = grad_field_ptr != nullptr
                                   ? pybind11::cast(grad_field_vector)
                                   : pybind11::object(pybind11::none());
    out["grad_direction"] = grad_direction_ptr != nullptr
                                ? pybind11::cast(grad_direction)
                                : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_field_project_complex3_jvp(
    at::Tensor field_vector,
    at::Tensor direction,
    at::Tensor rx_polarization,
    pybind11::object tangent_field_vector,
    pybind11::object tangent_direction) {
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    check_tensor(field_vector, "field_vector", at::kComplexFloat, 2);
    TORCH_CHECK(field_vector.size(1) == 3, "field_vector must have shape (N, 3)");
    check_vec3_table(direction, "direction");
    check_vec3_table(rx_polarization, "rx_polarization");
    const int64_t count = field_vector.size(0);
    TORCH_CHECK(direction.size(0) == count && rx_polarization.size(0) == count,
                "projection tensors must match field_vector rows");
    at::Tensor storage[2];
    const at::Tensor* t_field = optional_tensor_arg(
        std::move(tangent_field_vector), storage[0], "tangent_field_vector",
        at::kComplexFloat, {count, 3}, field_vector);
    const at::Tensor* t_direction = optional_tensor_arg(
        std::move(tangent_direction), storage[1], "tangent_direction",
        at::kFloat, {count, 3}, field_vector);
    auto tangent_coefficient = at::empty({count}, field_vector.options());
    auto tangent_path_gain = at::empty({count}, direction.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(direction.get_device()).stream();
        project_complex3_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            field_vector.data_ptr<c10::complex<float>>(),
            direction.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            opt_ptr<c10::complex<float>>(t_field),
            opt_ptr<float>(t_direction),
            tangent_coefficient.data_ptr<c10::complex<float>>(),
            tangent_path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_coefficient"] = tangent_coefficient;
    out["tangent_path_gain"] = tangent_path_gain;
    return out;
}
