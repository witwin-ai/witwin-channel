#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include "../field_transport_ad.cuh"
#include "../tensor_checks.h"

// Backward / JVP companion kernels for the field transport forwards
// (plan 07 AD-1 materials/frequency, AD-2 geometry). Fixed-topology contract:
// the discrete winner (face sequence, validity, normal flips, polarizations,
// tx_power, material ids) is constant; the differentiable inputs are
// eps_r / sigma_e / gain / thickness (per bounce or CSR layer), the carrier
// frequency, and the continuous hit geometry (source, target,
// interaction_positions, interaction_normals) behind need_grad_geometry.
// path_length_m / delay_s are differentiable outputs of the geometry alone
// (their material/frequency cotangent is exactly zero).

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxAdDepth = 8;
namespace field = witwin::channel::native_ext;
namespace em = channel_native::em;
namespace transport = channel_native::field_transport;
namespace ad = channel_native::field_transport_ad;

using ad::DualC;
using ad::adj_dot;

__device__ __forceinline__ field::float3a load3f(const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ field::float3a load_sequence3f(
    const float* values, int64_t index, int64_t bounce, int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ ad::DualF3 load_dual3f(
    const float* values, const float* tangents, int64_t index) {
    return {
        load3f(values, index),
        tangents != nullptr ? load3f(tangents, index) : field::f3_zero()};
}

__device__ __forceinline__ ad::DualF3 load_dual_sequence3f(
    const float* values,
    const float* tangents,
    int64_t index,
    int64_t bounce,
    int64_t depth) {
    return {
        load_sequence3f(values, index, bounce, depth),
        tangents != nullptr ? load_sequence3f(tangents, index, bounce, depth)
                            : field::f3_zero()};
}

__device__ __forceinline__ field::Complex complex_of(c10::complex<float> value) {
    return field::cplx(value.real(), value.imag());
}

__device__ __forceinline__ c10::complex<float> to_c10(field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

// Cotangent of the final complex3 field value, folded from the output
// cotangents (field_vector, coefficient, path_field, path_gain). The scalar
// chain is coefficient = <value, rx_axis>, path_field = coefficient * a,
// path_gain = |path_field|^2 with a = sqrt(max(tx_power, 0)); all real-linear
// maps except path_gain, whose real-pair adjoint at pf is 2*g*(pf.re, pf.im).
// The folded scalar cotangent is also emitted: it is the coefficient-level
// cotangent the geometry adjoint needs for the rx_axis (plan 07 AD-2).
__device__ __forceinline__ field::Complex3 fold_output_cotangents(
    const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient,
    const c10::complex<float>* grad_path_field,
    const float* grad_path_gain,
    int64_t index,
    field::float3a rx_axis,
    field::Complex path_field_value,
    float amplitude_scale,
    field::Complex& g_scalar_out) {
    field::Complex g_scalar = field::cplx_zero();
    if (grad_coefficient != nullptr)
        g_scalar = field::cplx_add(g_scalar, complex_of(grad_coefficient[index]));
    if (grad_path_field != nullptr)
        g_scalar = field::cplx_add(
            g_scalar,
            field::cplx_mul_real(
                complex_of(grad_path_field[index]), amplitude_scale));
    if (grad_path_gain != nullptr) {
        const float g_gain = grad_path_gain[index];
        g_scalar = field::cplx_add(
            g_scalar,
            field::cplx_mul_real(
                path_field_value, 2.0f * g_gain * amplitude_scale));
    }
    g_scalar_out = g_scalar;
    field::Complex3 g_value = field::c3_zero();
    g_value.x = field::cplx_mul_real(g_scalar, rx_axis.x);
    g_value.y = field::cplx_mul_real(g_scalar, rx_axis.y);
    g_value.z = field::cplx_mul_real(g_scalar, rx_axis.z);
    if (grad_field_vector != nullptr) {
        const int64_t base = index * 3;
        g_value.x = field::cplx_add(g_value.x, complex_of(grad_field_vector[base]));
        g_value.y = field::cplx_add(g_value.y, complex_of(grad_field_vector[base + 1]));
        g_value.z = field::cplx_add(g_value.z, complex_of(grad_field_vector[base + 2]));
    }
    return g_value;
}

// Forward-mode dual of the shared scalar output chain. Writes the six
// differentiable output tangents; d_rx_axis and d_length carry the geometry
// tangents (zero under material/frequency-only seeds).
__device__ __forceinline__ void write_output_tangents(
    int64_t index,
    field::Complex3 value,
    field::Complex3 d_value,
    field::float3a rx_axis,
    field::float3a d_rx_axis,
    float amplitude_scale,
    float d_length,
    c10::complex<float>* t_field_vector,
    c10::complex<float>* t_coefficient,
    c10::complex<float>* t_path_field,
    float* t_path_gain,
    float* t_path_length,
    float* t_delay) {
    const int64_t base = index * 3;
    t_field_vector[base] = to_c10(d_value.x);
    t_field_vector[base + 1] = to_c10(d_value.y);
    t_field_vector[base + 2] = to_c10(d_value.z);
    const field::Complex scalar = transport::complex3_dot_real(value, rx_axis);
    const field::Complex d_scalar = field::cplx_add(
        transport::complex3_dot_real(d_value, rx_axis),
        transport::complex3_dot_real(value, d_rx_axis));
    t_coefficient[index] = to_c10(d_scalar);
    const field::Complex path_field = field::cplx_mul_real(scalar, amplitude_scale);
    const field::Complex d_path_field = field::cplx_mul_real(d_scalar, amplitude_scale);
    t_path_field[index] = to_c10(d_path_field);
    t_path_gain[index] =
        2.0f * (path_field.re * d_path_field.re + path_field.im * d_path_field.im);
    t_path_length[index] = d_length;
    t_delay[index] = d_length / transport::kSpeedOfLight;
}

// ---------------------------------------------------------------------------
// Host entries.
// ---------------------------------------------------------------------------

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

// Gradient accumulators must start at zero; allocate raw and memset on the
// current stream (same pattern as los.cu) instead of ATen zero-fill.
at::Tensor zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(),
            stream));
    }
    return tensor;
}

const at::Tensor* optional_grad(
    pybind11::object value,
    at::Tensor& storage,
    const char* name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (value.is_none())
        return nullptr;
    storage = value.cast<at::Tensor>().contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(
        storage.get_device() == reference.get_device(),
        name, " must share the primal device");
    return &storage;
}

template <typename T>
const T* grad_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

}  // namespace
