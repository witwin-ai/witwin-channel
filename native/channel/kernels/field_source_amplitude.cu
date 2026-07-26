// ADR-039: source-amplitude application onto a transported complex3 field.
//
// The field transport kernels publish two families on one launch: the
// unit-excitation pair (``field_vector``, ``coefficient``) and the excited
// pair (``path_field = coefficient * sqrt(tx_power)``, ``path_gain``). There
// is no excited complex3 vector, so the public consumer complex3 response had
// no power-carrying quantity to publish. This owner supplies exactly that
// missing output:
//
//   path_field_vector = field_vector * sqrt(max(tx_power, 0))
//
// with the identical ``sqrtf(fmaxf(tx_power, 0))`` amplitude expression the
// transport kernels use, so ``<path_field_vector, rx_axis>`` and
// ``path_field`` are the same quantity. They are not required to be
// bit-identical: this owner scales and the caller then projects, while the
// transport kernel projects and then scales, so the two differ by float
// rounding order. The map is linear in the field vector and the
// amplitude is real, so the VJP and JVP are the same scale.
//
// ``tx_power`` is a frozen primal here exactly as it is in every field
// transport companion: no gradient or tangent is produced for it, and the
// Python wrappers reject a request for one.
//
// Elementwise over rows, one launch, no reduction and no atomics.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"

namespace {

constexpr int kBlockSize = 128;

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

using cfloat = c10::complex<float>;

__global__ void source_amplitude_scale_kernel(
    int64_t count,
    const cfloat* __restrict__ field_vector,
    const float* __restrict__ tx_power,
    cfloat* __restrict__ out_field_vector) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float amplitude = sqrtf(fmaxf(tx_power[row], 0.0f));
        for (int c = 0; c < 3; ++c) {
            const cfloat value = field_vector[row * 3 + c];
            out_field_vector[row * 3 + c] =
                cfloat(value.real() * amplitude, value.imag() * amplitude);
        }
    }
}

void check_power(const at::Tensor& tx_power, int64_t count, const at::Tensor& reference) {
    channel::check_tensor(tx_power, "tx_power", at::kFloat, 1);
    TORCH_CHECK(
        tx_power.size(0) == count,
        "tx_power must have one row per complex3 field row");
    TORCH_CHECK(
        tx_power.get_device() == reference.get_device(),
        "tx_power must share the field device");
}

int64_t check_field(const at::Tensor& field_vector, const char* name) {
    channel::check_tensor(field_vector, name, at::kComplexFloat, 2);
    TORCH_CHECK(field_vector.size(1) == 3, name, " must have shape (N, 3)");
    return field_vector.size(0);
}

at::Tensor scaled(const at::Tensor& field_vector, const at::Tensor& tx_power) {
    const int64_t count = field_vector.size(0);
    auto out = at::empty_like(field_vector);
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(field_vector.get_device()).stream();
        source_amplitude_scale_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            field_vector.data_ptr<cfloat>(),
            tx_power.data_ptr<float>(),
            out.data_ptr<cfloat>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

}  // namespace

pybind11::dict channel_field_source_amplitude_scale(
    at::Tensor field_vector,
    at::Tensor tx_power) {
    // Autograd hands cotangents and tangents in as strided views; the scale is
    // elementwise, so a canonical contiguous view is the whole staging cost.
    field_vector = field_vector.contiguous();
    tx_power = tx_power.contiguous();
    const int64_t count = check_field(field_vector, "field_vector");
    check_power(tx_power, count, field_vector);
    pybind11::dict out;
    out["path_field_vector"] = scaled(field_vector, tx_power);
    return out;
}

pybind11::dict channel_field_source_amplitude_scale_backward(
    at::Tensor tx_power,
    at::Tensor grad_path_field_vector) {
    grad_path_field_vector = grad_path_field_vector.contiguous();
    tx_power = tx_power.contiguous();
    const int64_t count = check_field(grad_path_field_vector, "grad_path_field_vector");
    check_power(tx_power, count, grad_path_field_vector);
    pybind11::dict out;
    out["grad_field_vector"] = scaled(grad_path_field_vector, tx_power);
    return out;
}

pybind11::dict channel_field_source_amplitude_scale_jvp(
    at::Tensor tx_power,
    at::Tensor tangent_field_vector) {
    tangent_field_vector = tangent_field_vector.contiguous();
    tx_power = tx_power.contiguous();
    const int64_t count = check_field(tangent_field_vector, "tangent_field_vector");
    check_power(tx_power, count, tangent_field_vector);
    pybind11::dict out;
    out["tangent_path_field_vector"] = scaled(tangent_field_vector, tx_power);
    return out;
}
