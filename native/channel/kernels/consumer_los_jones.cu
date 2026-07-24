#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda_minimal.h"

#include <rayd/shared/rf/field_transport.cuh>

#include "../tensor_checks.h"

#include <cmath>
#include <cstdint>

namespace {

constexpr int kJonesBlockSize = 256;
namespace field = rayd::shared::utd;
namespace transport = rayd::shared::rf::field_transport;

__device__ __forceinline__ field::float3a load3(
    const float *values,
    int64_t row) {
    const int64_t base = row * 3;
    return field::make_f3(
        values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ field::Basis3 endpoint_basis(
    const float *values,
    int64_t endpoint,
    field::float3a direction) {
    const int64_t base = endpoint * 6;
    const field::float3a first = field::make_f3(
        values[base], values[base + 1], values[base + 2]);
    const field::float3a second = field::make_f3(
        values[base + 3], values[base + 4], values[base + 5]);
    field::Basis3 basis = field::basis_from_first_vector(
        direction,
        first,
        field::stable_perp_basis(direction, second));
    if (field::f3_dot(basis.v, second) < 0.0f) {
        basis.v = field::f3_neg(basis.v);
    }
    return basis;
}

__device__ __forceinline__ void store_basis(
    float *values,
    int64_t row,
    field::Basis3 basis) {
    const int64_t base = row * 6;
    values[base] = basis.u.x;
    values[base + 1] = basis.u.y;
    values[base + 2] = basis.u.z;
    values[base + 3] = basis.v.x;
    values[base + 4] = basis.v.y;
    values[base + 5] = basis.v.z;
}

__device__ __forceinline__ c10::complex<float> to_complex(
    field::Complex value) {
    return {value.re, value.im};
}

__global__ void consumer_los_jones_kernel(
    const int64_t *__restrict__ pair_index,
    const float *__restrict__ source_positions,
    const float *__restrict__ sink_positions,
    const float *__restrict__ source_reference_basis,
    const float *__restrict__ sink_reference_basis,
    int64_t row_count,
    int64_t num_tx,
    int64_t pair_count,
    float frequency_hz,
    c10::complex<float> *__restrict__ matrix,
    float *__restrict__ source_basis,
    float *__restrict__ sink_basis) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int64_t pair = pair_index[row];
        CUDA_KERNEL_ASSERT(pair >= 0 && pair < pair_count);
        const int64_t tx = pair % num_tx;
        const int64_t rx = pair / num_tx;
        const field::float3a source = load3(source_positions, tx);
        const field::float3a sink = load3(sink_positions, rx);
        const field::float3a direction = field::safe_normalize(
            field::f3_sub(sink, source),
            field::make_f3(0.0f, 0.0f, 1.0f));
        const field::Basis3 source_frame = endpoint_basis(
            source_reference_basis, tx, direction);
        const field::Basis3 sink_frame = endpoint_basis(
            sink_reference_basis, rx, direction);
        store_basis(source_basis, row, source_frame);
        store_basis(sink_basis, row, sink_frame);

        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz /
            transport::kSpeedOfLight;
        const field::Complex3 source_u =
            transport::free_space_complex3(
                source, sink, wave_number, source_frame.u);
        const field::Complex3 source_v =
            transport::free_space_complex3(
                source, sink, wave_number, source_frame.v);
        const int64_t base = row * 4;
        matrix[base] = to_complex(transport::project_receiver(
            source_u, direction, sink_frame.u));
        matrix[base + 1] = to_complex(transport::project_receiver(
            source_v, direction, sink_frame.u));
        matrix[base + 2] = to_complex(transport::project_receiver(
            source_u, direction, sink_frame.v));
        matrix[base + 3] = to_complex(transport::project_receiver(
            source_v, direction, sink_frame.v));
    }
}

void check_positions(
    const at::Tensor& values,
    const char *name,
    int device) {
    channel::check_tensor(values, name, at::kFloat, 2);
    TORCH_CHECK(values.size(1) == 3, name, " must have shape (N, 3)");
    TORCH_CHECK(values.get_device() == device, name, " must share pair device");
}

void check_basis(
    const at::Tensor& values,
    const char *name,
    int64_t rows,
    int device) {
    channel::check_tensor(values, name, at::kFloat, 3);
    TORCH_CHECK(
        values.sizes() == at::IntArrayRef({rows, 2, 3}),
        name,
        " must have shape (N, 2, 3)");
    TORCH_CHECK(values.get_device() == device, name, " must share pair device");
}

}  // namespace

pybind11::dict channel_consumer_los_jones(
    at::Tensor pair_index,
    at::Tensor source_positions,
    at::Tensor sink_positions,
    at::Tensor source_reference_basis,
    at::Tensor sink_reference_basis,
    double frequency_hz) {
    channel::check_tensor(pair_index, "pair_index", at::kLong, 1);
    const int device = pair_index.get_device();
    check_positions(source_positions, "source_positions", device);
    check_positions(sink_positions, "sink_positions", device);
    const int64_t num_tx = source_positions.size(0);
    const int64_t num_rx = sink_positions.size(0);
    check_basis(
        source_reference_basis,
        "source_reference_basis",
        num_tx,
        device);
    check_basis(
        sink_reference_basis,
        "sink_reference_basis",
        num_rx,
        device);
    TORCH_CHECK(
        std::isfinite(frequency_hz) && frequency_hz > 0.0,
        "frequency_hz must be finite and positive");
    TORCH_CHECK(
        pair_index.size(0) == 0 || num_tx > 0,
        "non-empty pair_index requires sources");

    const int64_t row_count = pair_index.size(0);
    auto complex_options = pair_index.options().dtype(at::kComplexFloat);
    auto float_options = pair_index.options().dtype(at::kFloat);
    auto matrix = at::empty({row_count, 2, 2}, complex_options);
    auto out_source_basis = at::empty({row_count, 2, 3}, float_options);
    auto out_sink_basis = at::empty({row_count, 2, 3}, float_options);
    if (row_count > 0) {
        const int64_t pair_count = num_tx * num_rx;
        TORCH_CHECK(pair_count > 0, "non-empty pair_index requires endpoint pairs");
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device).stream();
        const int blocks = static_cast<int>(
            (row_count + kJonesBlockSize - 1) / kJonesBlockSize);
        consumer_los_jones_kernel<<<
            blocks, kJonesBlockSize, 0, stream>>>(
            pair_index.data_ptr<int64_t>(),
            source_positions.data_ptr<float>(),
            sink_positions.data_ptr<float>(),
            source_reference_basis.data_ptr<float>(),
            sink_reference_basis.data_ptr<float>(),
            row_count,
            num_tx,
            pair_count,
            static_cast<float>(frequency_hz),
            matrix.data_ptr<c10::complex<float>>(),
            out_source_basis.data_ptr<float>(),
            out_sink_basis.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["matrix"] = matrix;
    result["source_basis"] = out_source_basis;
    result["sink_basis"] = out_sink_basis;
    return result;
}
