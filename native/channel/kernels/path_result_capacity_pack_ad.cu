#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"

#include <cfloat>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

namespace {

constexpr int kPathResultPackAdBlockSize = 256;
constexpr int32_t kDiffractionInteraction = 2;
using cfloat = c10::complex<float>;
using channel::check_tensor;

struct Float3 {
    float x;
    float y;
    float z;
};

template <typename T>
struct OptionalView {
    const T *data;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
    bool present;
};

struct OutputCotangents {
    OptionalView<cfloat> a;
    OptionalView<float> tau;
    OptionalView<float> theta_t;
    OptionalView<float> phi_t;
    OptionalView<float> theta_r;
    OptionalView<float> phi_r;
    OptionalView<float> position;
    OptionalView<float> normal;
    OptionalView<cfloat> field_xyz;
    OptionalView<float> field_direction;
};

struct InputTangents {
    OptionalView<float> delay_s;
    OptionalView<float> field_direction;
    OptionalView<float> interaction_positions;
    OptionalView<float> interaction_normals;
    OptionalView<cfloat> coefficient;
    OptionalView<cfloat> field_xyz;
    OptionalView<float> tx_positions;
    OptionalView<float> rx_positions;
};

struct InputGradients {
    float *delay_s;
    float *field_direction;
    float *interaction_positions;
    float *interaction_normals;
    cfloat *coefficient;
    cfloat *field_xyz;
    float *tx_positions;
    float *rx_positions;
};

struct OutputTangents {
    cfloat *a;
    float *tau;
    float *theta_t;
    float *phi_t;
    float *theta_r;
    float *phi_r;
    float *position;
    float *normal;
    cfloat *field_xyz;
    float *field_direction;
};

__device__ __forceinline__ Float3 make_vec3(float x, float y, float z) {
    return {x, y, z};
}

__device__ __forceinline__ Float3 add(Float3 lhs, Float3 rhs) {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

__device__ __forceinline__ Float3 subtract(Float3 lhs, Float3 rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

__device__ __forceinline__ Float3 negate(Float3 value) {
    return {
        __uint_as_float(__float_as_uint(value.x) ^ 0x80000000u),
        __uint_as_float(__float_as_uint(value.y) ^ 0x80000000u),
        __uint_as_float(__float_as_uint(value.z) ^ 0x80000000u)};
}

__device__ __forceinline__ float negate(float value) {
    return __uint_as_float(__float_as_uint(value) ^ 0x80000000u);
}

__device__ __forceinline__ Float3 load_float3(const float *data, int64_t row) {
    const int64_t offset = row * 3;
    return {data[offset], data[offset + 1], data[offset + 2]};
}

template <typename T>
__device__ __forceinline__ T read_scalar(
    const OptionalView<T>& view,
    int64_t row) {
    return view.present ? view.data[row * view.stride0] : T(0);
}

template <typename T>
__device__ __forceinline__ T read_vector(
    const OptionalView<T>& view,
    int64_t row,
    int64_t component) {
    return view.present
        ? view.data[row * view.stride0 + component * view.stride1]
        : T(0);
}

template <typename T>
__device__ __forceinline__ T read_sequence_vector(
    const OptionalView<T>& view,
    int64_t row,
    int64_t slot,
    int64_t component) {
    return view.present
        ? view.data[
              row * view.stride0 + slot * view.stride1 + component * view.stride2]
        : T(0);
}

__device__ __forceinline__ Float3 optional_float3(
    const OptionalView<float>& view,
    int64_t row) {
    return {
        read_vector(view, row, 0),
        read_vector(view, row, 1),
        read_vector(view, row, 2)};
}

__device__ __forceinline__ Float3 optional_sequence_float3(
    const OptionalView<float>& view,
    int64_t row,
    int64_t slot) {
    return {
        read_sequence_vector(view, row, slot, 0),
        read_sequence_vector(view, row, slot, 1),
        read_sequence_vector(view, row, slot, 2)};
}

__device__ __forceinline__ float clamp_preserve_nan(
    float value,
    float lower,
    float upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

__device__ __forceinline__ Float3 endpoint_angle_vjp(
    Float3 direction,
    float grad_theta,
    bool has_theta,
    float grad_phi,
    bool has_phi) {
    const float squared =
        direction.x * direction.x + direction.y * direction.y +
        direction.z * direction.z;
    const float norm = sqrtf(squared);
    const float safe = norm < FLT_MIN ? FLT_MIN : norm;
    const float unclamped_cosine = direction.z / safe;
    const float cosine =
        clamp_preserve_nan(unclamped_cosine, -1.0f, 1.0f);
    Float3 grad = make_vec3(0.0f, 0.0f, 0.0f);
    if (has_theta) {
        // Mirror the eager Torch backward graph operation by operation:
        // acos -> clamp -> div -> clamp_min -> vector_norm.
        float grad_cosine =
            grad_theta * -rsqrtf(-cosine * cosine + 1.0f);
        if (!(unclamped_cosine >= -1.0f && unclamped_cosine <= 1.0f)) {
            grad_cosine = 0.0f;
        }
        grad.z += grad_cosine / safe;
        const float grad_safe =
            negate(grad_cosine) * ((direction.z / safe) / safe);
        const float grad_norm = norm >= FLT_MIN ? grad_safe : 0.0f;
        const float norm_x = norm == 0.0f ? 0.0f : direction.x / norm;
        const float norm_y = norm == 0.0f ? 0.0f : direction.y / norm;
        const float norm_z = norm == 0.0f ? 0.0f : direction.z / norm;
        grad.x += grad_norm * norm_x;
        grad.y += grad_norm * norm_y;
        grad.z += grad_norm * norm_z;
    }
    if (has_phi) {
        const float xy_squared =
            direction.y * direction.y + direction.x * direction.x;
        const float numerator_x = grad_phi * negate(direction.y);
        const float numerator_y = grad_phi * direction.x;
        const float reciprocal =
            xy_squared == 0.0f ? 0.0f : 1.0f / xy_squared;
        grad.x += numerator_x * reciprocal;
        grad.y += numerator_y * reciprocal;
    }
    return grad;
}

__device__ __forceinline__ void endpoint_angle_jvp(
    Float3 direction,
    Float3 tangent,
    float& tangent_theta,
    float& tangent_phi) {
    const float squared =
        direction.x * direction.x + direction.y * direction.y +
        direction.z * direction.z;
    const float norm = sqrtf(squared);
    const float safe = norm < FLT_MIN ? FLT_MIN : norm;
    const float dot =
        direction.x * tangent.x + direction.y * tangent.y +
        direction.z * tangent.z;
    float tangent_norm = dot / norm;
    if (norm == 0.0f) {
        tangent_norm = 0.0f;
    }
    const float tangent_safe = norm >= FLT_MIN ? tangent_norm : 0.0f;
    const float unclamped_cosine = direction.z / safe;
    const float cosine =
        clamp_preserve_nan(unclamped_cosine, -1.0f, 1.0f);
    float tangent_cosine =
        (tangent.z - tangent_safe * unclamped_cosine) / safe;
    if (!(unclamped_cosine >= -1.0f && unclamped_cosine <= 1.0f)) {
        tangent_cosine = 0.0f;
    }
    tangent_theta =
        tangent_cosine * -rsqrtf(-cosine * cosine + 1.0f);
    const float xy_squared =
        direction.y * direction.y + direction.x * direction.x;
    tangent_phi =
        (-direction.y * tangent.x + direction.x * tangent.y) / xy_squared;
}

__device__ __forceinline__ int64_t bounded_last_slot(
    int32_t depth,
    int64_t sequence_width) {
    (void)sequence_width;
    return static_cast<int64_t>(depth) - 1;
}

__device__ __forceinline__ void row_directions(
    int64_t row,
    int64_t sequence_width,
    const int32_t *depth,
    const int32_t *tx_id,
    const int32_t *rx_id,
    const float *interaction_positions,
    const float *tx_positions,
    const float *rx_positions,
    Float3& departure,
    Float3& receiver_direction) {
    const int32_t tx = tx_id[row];
    const int32_t rx = rx_id[row];
    const Float3 tx_position = load_float3(tx_positions, tx);
    const Float3 rx_position = load_float3(rx_positions, rx);
    const Float3 direct = subtract(rx_position, tx_position);
    departure = direct;
    receiver_direction = negate(direct);
    if (sequence_width > 0 && depth[row] > 0) {
        const int64_t base = row * sequence_width;
        const Float3 first = load_float3(interaction_positions, base);
        const Float3 last = load_float3(
            interaction_positions,
            base + bounded_last_slot(depth[row], sequence_width));
        departure = subtract(first, tx_position);
        receiver_direction = subtract(last, rx_position);
    }
}

__device__ __forceinline__ void row_angle_vjps(
    int64_t row,
    int64_t sequence_width,
    const int32_t *depth,
    const int32_t *tx_id,
    const int32_t *rx_id,
    const float *interaction_positions,
    const float *tx_positions,
    const float *rx_positions,
    const OutputCotangents& grad_output,
    Float3& departure_grad,
    Float3& receiver_grad) {
    Float3 departure;
    Float3 receiver_direction;
    row_directions(
        row,
        sequence_width,
        depth,
        tx_id,
        rx_id,
        interaction_positions,
        tx_positions,
        rx_positions,
        departure,
        receiver_direction);
    departure_grad = endpoint_angle_vjp(
        departure,
        read_scalar(grad_output.theta_t, row),
        grad_output.theta_t.present,
        read_scalar(grad_output.phi_t, row),
        grad_output.phi_t.present);
    receiver_grad = endpoint_angle_vjp(
        receiver_direction,
        read_scalar(grad_output.theta_r, row),
        grad_output.theta_r.present,
        read_scalar(grad_output.phi_r, row),
        grad_output.phi_r.present);
}

__global__ void path_result_capacity_backward_rows_kernel(
    const bool *__restrict__ valid,
    const int32_t *__restrict__ tx_id,
    const int32_t *__restrict__ rx_id,
    const int32_t *__restrict__ depth,
    const int32_t *__restrict__ interaction_type,
    const float *__restrict__ interaction_positions,
    const float *__restrict__ interaction_normals,
    const float *__restrict__ tx_positions,
    const float *__restrict__ rx_positions,
    OutputCotangents grad_output,
    InputGradients grad_input,
    int64_t row_capacity,
    int64_t sequence_width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_capacity;
         row += stride) {
        grad_input.delay_s[row] = 0.0f;
        grad_input.coefficient[row] = cfloat(0.0f, 0.0f);
        const int64_t row_vec = row * 3;
        for (int component = 0; component < 3; ++component) {
            grad_input.field_direction[row_vec + component] = 0.0f;
            grad_input.field_xyz[row_vec + component] = cfloat(0.0f, 0.0f);
        }
        const int64_t sequence_vec = row * sequence_width * 3;
        for (int64_t item = 0; item < sequence_width * 3; ++item) {
            grad_input.interaction_positions[sequence_vec + item] = 0.0f;
            grad_input.interaction_normals[sequence_vec + item] = 0.0f;
        }
        if (!valid[row]) {
            continue;
        }

        grad_input.delay_s[row] = read_scalar(grad_output.tau, row);
        grad_input.coefficient[row] = read_scalar(grad_output.a, row);
        for (int component = 0; component < 3; ++component) {
            grad_input.field_direction[row_vec + component] =
                read_vector(grad_output.field_direction, row, component);
            grad_input.field_xyz[row_vec + component] =
                read_vector(grad_output.field_xyz, row, component);
        }
        for (int64_t slot = 0; slot < sequence_width; ++slot) {
            const int64_t sequence_index = row * sequence_width + slot;
            for (int component = 0; component < 3; ++component) {
                const int64_t vector_index = sequence_vec + slot * 3 + component;
                grad_input.interaction_positions[vector_index] =
                    read_sequence_vector(grad_output.position, row, slot, component);
                const float primal_normal = interaction_normals[vector_index];
                grad_input.interaction_normals[vector_index] =
                    interaction_type[sequence_index] == kDiffractionInteraction &&
                            !isfinite(primal_normal)
                    ? 0.0f
                    : read_sequence_vector(
                          grad_output.normal, row, slot, component);
            }
        }
        if (sequence_width > 0 && depth[row] > 0) {
            Float3 departure_grad;
            Float3 receiver_grad;
            row_angle_vjps(
                row,
                sequence_width,
                depth,
                tx_id,
                rx_id,
                interaction_positions,
                tx_positions,
                rx_positions,
                grad_output,
                departure_grad,
                receiver_grad);
            const int64_t first = sequence_vec;
            grad_input.interaction_positions[first] += departure_grad.x;
            grad_input.interaction_positions[first + 1] += departure_grad.y;
            grad_input.interaction_positions[first + 2] += departure_grad.z;
            const int64_t last =
                sequence_vec + bounded_last_slot(depth[row], sequence_width) * 3;
            grad_input.interaction_positions[last] += receiver_grad.x;
            grad_input.interaction_positions[last + 1] += receiver_grad.y;
            grad_input.interaction_positions[last + 2] += receiver_grad.z;
        }
    }
}

__global__ void path_result_capacity_backward_endpoints_kernel(
    const bool *__restrict__ valid,
    const int32_t *__restrict__ tx_id,
    const int32_t *__restrict__ rx_id,
    const int32_t *__restrict__ depth,
    const float *__restrict__ interaction_positions,
    const float *__restrict__ tx_positions,
    const float *__restrict__ rx_positions,
    OutputCotangents grad_output,
    float *__restrict__ grad_tx_positions,
    float *__restrict__ grad_rx_positions,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair,
    int64_t sequence_width) {
    const int64_t endpoint_count = num_tx + num_rx;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t endpoint =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         endpoint < endpoint_count;
         endpoint += stride) {
        Float3 total = make_vec3(0.0f, 0.0f, 0.0f);
        bool has_total = false;
        if (endpoint < num_tx) {
            const int64_t tx = endpoint;
            for (int64_t rx = 0; rx < num_rx; ++rx) {
                const int64_t pair = rx * num_tx + tx;
                for (int64_t slot = 0; slot < path_capacity_per_pair; ++slot) {
                    const int64_t row = pair * path_capacity_per_pair + slot;
                    if (!valid[row]) {
                        continue;
                    }
                    Float3 departure_grad;
                    Float3 receiver_grad;
                    row_angle_vjps(
                        row,
                        sequence_width,
                        depth,
                        tx_id,
                        rx_id,
                        interaction_positions,
                        tx_positions,
                        rx_positions,
                        grad_output,
                        departure_grad,
                        receiver_grad);
                    Float3 contribution;
                    if (sequence_width > 0 && depth[row] > 0) {
                        contribution = negate(departure_grad);
                    } else {
                        // Eager Torch accumulates both angle branches on the
                        // shared direct tensor before the subtraction VJP.
                        const Float3 direct_grad =
                            subtract(departure_grad, receiver_grad);
                        contribution = negate(direct_grad);
                    }
                    total = has_total ? add(total, contribution) : contribution;
                    has_total = true;
                }
            }
            const int64_t base = tx * 3;
            grad_tx_positions[base] = total.x;
            grad_tx_positions[base + 1] = total.y;
            grad_tx_positions[base + 2] = total.z;
        } else {
            const int64_t rx = endpoint - num_tx;
            for (int64_t tx = 0; tx < num_tx; ++tx) {
                const int64_t pair = rx * num_tx + tx;
                for (int64_t slot = 0; slot < path_capacity_per_pair; ++slot) {
                    const int64_t row = pair * path_capacity_per_pair + slot;
                    if (!valid[row]) {
                        continue;
                    }
                    Float3 departure_grad;
                    Float3 receiver_grad;
                    row_angle_vjps(
                        row,
                        sequence_width,
                        depth,
                        tx_id,
                        rx_id,
                        interaction_positions,
                        tx_positions,
                        rx_positions,
                        grad_output,
                        departure_grad,
                        receiver_grad);
                    Float3 contribution;
                    if (sequence_width > 0 && depth[row] > 0) {
                        contribution = negate(receiver_grad);
                    } else {
                        const Float3 direct_grad =
                            subtract(departure_grad, receiver_grad);
                        contribution = direct_grad;
                    }
                    total = has_total ? add(total, contribution) : contribution;
                    has_total = true;
                }
            }
            const int64_t base = rx * 3;
            grad_rx_positions[base] = total.x;
            grad_rx_positions[base + 1] = total.y;
            grad_rx_positions[base + 2] = total.z;
        }
    }
}

__global__ void path_result_capacity_jvp_kernel(
    const bool *__restrict__ valid,
    const int32_t *__restrict__ tx_id,
    const int32_t *__restrict__ rx_id,
    const int32_t *__restrict__ depth,
    const int32_t *__restrict__ interaction_type,
    const float *__restrict__ interaction_positions,
    const float *__restrict__ interaction_normals,
    const float *__restrict__ tx_positions,
    const float *__restrict__ rx_positions,
    InputTangents tangent_input,
    OutputTangents tangent_output,
    int64_t row_capacity,
    int64_t sequence_width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_capacity;
         row += stride) {
        tangent_output.a[row] = cfloat(0.0f, 0.0f);
        tangent_output.tau[row] = 0.0f;
        tangent_output.theta_t[row] = 0.0f;
        tangent_output.phi_t[row] = 0.0f;
        tangent_output.theta_r[row] = 0.0f;
        tangent_output.phi_r[row] = 0.0f;
        const int64_t row_vec = row * 3;
        for (int component = 0; component < 3; ++component) {
            tangent_output.field_xyz[row_vec + component] = cfloat(0.0f, 0.0f);
            tangent_output.field_direction[row_vec + component] = 0.0f;
        }
        const int64_t sequence_vec = row * sequence_width * 3;
        for (int64_t item = 0; item < sequence_width * 3; ++item) {
            tangent_output.position[sequence_vec + item] = 0.0f;
            tangent_output.normal[sequence_vec + item] = 0.0f;
        }
        if (!valid[row]) {
            continue;
        }

        tangent_output.a[row] = read_scalar(tangent_input.coefficient, row);
        tangent_output.tau[row] = read_scalar(tangent_input.delay_s, row);
        for (int component = 0; component < 3; ++component) {
            tangent_output.field_xyz[row_vec + component] =
                read_vector(tangent_input.field_xyz, row, component);
            tangent_output.field_direction[row_vec + component] =
                read_vector(tangent_input.field_direction, row, component);
        }
        for (int64_t slot = 0; slot < sequence_width; ++slot) {
            const int64_t sequence_index = row * sequence_width + slot;
            for (int component = 0; component < 3; ++component) {
                const int64_t vector_index = sequence_vec + slot * 3 + component;
                tangent_output.position[vector_index] = read_sequence_vector(
                    tangent_input.interaction_positions, row, slot, component);
                const float primal_normal = interaction_normals[vector_index];
                tangent_output.normal[vector_index] =
                    interaction_type[sequence_index] == kDiffractionInteraction &&
                            !isfinite(primal_normal)
                    ? 0.0f
                    : read_sequence_vector(
                          tangent_input.interaction_normals, row, slot, component);
            }
        }

        Float3 departure;
        Float3 receiver_direction;
        row_directions(
            row,
            sequence_width,
            depth,
            tx_id,
            rx_id,
            interaction_positions,
            tx_positions,
            rx_positions,
            departure,
            receiver_direction);
        const int32_t tx = tx_id[row];
        const int32_t rx = rx_id[row];
        const Float3 tangent_tx = optional_float3(tangent_input.tx_positions, tx);
        const Float3 tangent_rx = optional_float3(tangent_input.rx_positions, rx);
        Float3 tangent_departure = subtract(tangent_rx, tangent_tx);
        Float3 tangent_receiver = negate(tangent_departure);
        if (sequence_width > 0 && depth[row] > 0) {
            const Float3 tangent_first =
                optional_sequence_float3(tangent_input.interaction_positions, row, 0);
            const Float3 tangent_last = optional_sequence_float3(
                tangent_input.interaction_positions,
                row,
                bounded_last_slot(depth[row], sequence_width));
            tangent_departure = subtract(tangent_first, tangent_tx);
            tangent_receiver = subtract(tangent_last, tangent_rx);
        }
        if (tangent_input.tx_positions.present ||
            tangent_input.rx_positions.present ||
            tangent_input.interaction_positions.present) {
            endpoint_angle_jvp(
                departure,
                tangent_departure,
                tangent_output.theta_t[row],
                tangent_output.phi_t[row]);
            endpoint_angle_jvp(
                receiver_direction,
                tangent_receiver,
                tangent_output.theta_r[row],
                tangent_output.phi_r[row]);
        }
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>(
        (count + kPathResultPackAdBlockSize - 1) / kPathResultPackAdBlockSize);
}

void check_same_device(
    const at::Tensor& value,
    const at::Tensor& reference,
    const char *name) {
    TORCH_CHECK(
        value.get_device() == reference.get_device(),
        name,
        " must share valid device");
}

void check_primal(
    const at::Tensor& valid,
    const at::Tensor& tx_id,
    const at::Tensor& rx_id,
    const at::Tensor& depth,
    const at::Tensor& interaction_type,
    const at::Tensor& interaction_positions,
    const at::Tensor& interaction_normals,
    const at::Tensor& tx_positions,
    const at::Tensor& rx_positions,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair) {
    check_tensor(valid, "valid", at::kBool, 1);
    TORCH_CHECK(num_tx >= 0 && num_rx >= 0, "endpoint counts must be non-negative");
    TORCH_CHECK(
        path_capacity_per_pair >= 0,
        "path_capacity_per_pair must be non-negative");
    constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
    TORCH_CHECK(
        num_rx <= kInt64Max - num_tx,
        "num_tx + num_rx overflows int64");
    TORCH_CHECK(
        num_tx == 0 || num_rx <= kInt64Max / num_tx,
        "num_tx * num_rx overflows int64");
    const int64_t pair_count = num_tx * num_rx;
    TORCH_CHECK(
        pair_count == 0 || path_capacity_per_pair <= kInt64Max / pair_count,
        "pair_count * path_capacity_per_pair overflows int64");
    const int64_t rows = pair_count * path_capacity_per_pair;
    TORCH_CHECK(valid.numel() == rows, "valid has wrong capacity shape");
    for (const auto& named : {
             std::pair<const char *, at::Tensor>{"tx_id", tx_id},
             {"rx_id", rx_id},
             {"depth", depth}}) {
        check_tensor(named.second, named.first, at::kInt, 1);
        TORCH_CHECK(named.second.numel() == rows, named.first, " has wrong shape");
        check_same_device(named.second, valid, named.first);
    }
    check_tensor(interaction_type, "interaction_type", at::kInt, 2);
    const int64_t width = interaction_type.size(1);
    TORCH_CHECK(
        interaction_type.sizes() == at::IntArrayRef({rows, width}),
        "interaction_type has wrong shape");
    check_same_device(interaction_type, valid, "interaction_type");
    for (const auto& named : {
             std::pair<const char *, at::Tensor>{
                 "interaction_positions", interaction_positions},
             {"interaction_normals", interaction_normals}}) {
        check_tensor(named.second, named.first, at::kFloat, 3);
        TORCH_CHECK(
            named.second.sizes() == at::IntArrayRef({rows, width, 3}),
            named.first,
            " has wrong shape");
        check_same_device(named.second, valid, named.first);
    }
    for (const auto& named : {
             std::tuple<const char *, at::Tensor, int64_t>{
                 "tx_positions", tx_positions, num_tx},
             {"rx_positions", rx_positions, num_rx}}) {
        const char *name = std::get<0>(named);
        const at::Tensor& tensor = std::get<1>(named);
        const int64_t count = std::get<2>(named);
        check_tensor(tensor, name, at::kFloat, 2);
        TORCH_CHECK(
            tensor.sizes() == at::IntArrayRef({count, 3}),
            name,
            " has wrong shape");
        check_same_device(tensor, valid, name);
    }
}

template <typename T>
OptionalView<T> optional_view(
    const std::optional<at::Tensor>& value,
    const char *name,
    c10::ScalarType dtype,
    at::IntArrayRef shape,
    int device) {
    if (!value.has_value()) {
        return {nullptr, 0, 0, 0, false};
    }
    const at::Tensor& tensor = *value;
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(tensor.sizes() == shape, name, " has wrong shape");
    TORCH_CHECK(tensor.get_device() == device, name, " must share valid device");
    return {
        tensor.data_ptr<T>(),
        tensor.stride(0),
        tensor.dim() > 1 ? tensor.stride(1) : 0,
        tensor.dim() > 2 ? tensor.stride(2) : 0,
        true};
}

struct AllocatedInputGradients {
    at::Tensor delay_s;
    at::Tensor field_direction;
    at::Tensor interaction_positions;
    at::Tensor interaction_normals;
    at::Tensor coefficient;
    at::Tensor field_xyz;
    at::Tensor tx_positions;
    at::Tensor rx_positions;

    InputGradients view() const {
        return {
            delay_s.data_ptr<float>(),
            field_direction.data_ptr<float>(),
            interaction_positions.data_ptr<float>(),
            interaction_normals.data_ptr<float>(),
            coefficient.data_ptr<cfloat>(),
            field_xyz.data_ptr<cfloat>(),
            tx_positions.data_ptr<float>(),
            rx_positions.data_ptr<float>()};
    }

    pybind11::dict dict() const {
        pybind11::dict result;
        result["delay_s"] = delay_s;
        result["field_direction"] = field_direction;
        result["interaction_positions"] = interaction_positions;
        result["interaction_normals"] = interaction_normals;
        result["coefficient"] = coefficient;
        result["field_xyz"] = field_xyz;
        result["tx_positions"] = tx_positions;
        result["rx_positions"] = rx_positions;
        return result;
    }
};

struct AllocatedOutputTangents {
    at::Tensor a;
    at::Tensor tau;
    at::Tensor theta_t;
    at::Tensor phi_t;
    at::Tensor theta_r;
    at::Tensor phi_r;
    at::Tensor position;
    at::Tensor normal;
    at::Tensor field_xyz;
    at::Tensor field_direction;

    OutputTangents view() const {
        return {
            a.data_ptr<cfloat>(),
            tau.data_ptr<float>(),
            theta_t.data_ptr<float>(),
            phi_t.data_ptr<float>(),
            theta_r.data_ptr<float>(),
            phi_r.data_ptr<float>(),
            position.data_ptr<float>(),
            normal.data_ptr<float>(),
            field_xyz.data_ptr<cfloat>(),
            field_direction.data_ptr<float>()};
    }

    pybind11::dict dict() const {
        pybind11::dict result;
        result["a"] = a;
        result["tau"] = tau;
        result["theta_t"] = theta_t;
        result["phi_t"] = phi_t;
        result["theta_r"] = theta_r;
        result["phi_r"] = phi_r;
        result["position"] = position;
        result["normal"] = normal;
        result["field_xyz"] = field_xyz;
        result["field_direction"] = field_direction;
        return result;
    }
};

}  // namespace

pybind11::dict channel_path_result_capacity_pack_backward(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor interaction_type,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor tx_positions,
    at::Tensor rx_positions,
    std::optional<at::Tensor> grad_a,
    std::optional<at::Tensor> grad_tau,
    std::optional<at::Tensor> grad_theta_t,
    std::optional<at::Tensor> grad_phi_t,
    std::optional<at::Tensor> grad_theta_r,
    std::optional<at::Tensor> grad_phi_r,
    std::optional<at::Tensor> grad_position,
    std::optional<at::Tensor> grad_normal,
    std::optional<at::Tensor> grad_field_xyz,
    std::optional<at::Tensor> grad_field_direction,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair) {
    check_primal(
        valid,
        tx_id,
        rx_id,
        depth,
        interaction_type,
        interaction_positions,
        interaction_normals,
        tx_positions,
        rx_positions,
        num_tx,
        num_rx,
        path_capacity_per_pair);
    const int device = valid.get_device();
    const int64_t rows = valid.numel();
    const int64_t width = interaction_type.size(1);
    OutputCotangents grad_output{
        optional_view<cfloat>(grad_a, "grad_a", at::kComplexFloat, {rows}, device),
        optional_view<float>(grad_tau, "grad_tau", at::kFloat, {rows}, device),
        optional_view<float>(
            grad_theta_t, "grad_theta_t", at::kFloat, {rows}, device),
        optional_view<float>(grad_phi_t, "grad_phi_t", at::kFloat, {rows}, device),
        optional_view<float>(
            grad_theta_r, "grad_theta_r", at::kFloat, {rows}, device),
        optional_view<float>(grad_phi_r, "grad_phi_r", at::kFloat, {rows}, device),
        optional_view<float>(
            grad_position, "grad_position", at::kFloat, {rows, width, 3}, device),
        optional_view<float>(
            grad_normal, "grad_normal", at::kFloat, {rows, width, 3}, device),
        optional_view<cfloat>(
            grad_field_xyz, "grad_field_xyz", at::kComplexFloat, {rows, 3}, device),
        optional_view<float>(grad_field_direction,
                             "grad_field_direction",
                             at::kFloat,
                             {rows, 3},
                             device)};
    const auto float_options = valid.options().dtype(at::kFloat);
    const auto complex_options = valid.options().dtype(at::kComplexFloat);
    AllocatedInputGradients grad_input{
        at::empty({rows}, float_options),
        at::empty({rows, 3}, float_options),
        at::empty({rows, width, 3}, float_options),
        at::empty({rows, width, 3}, float_options),
        at::empty({rows}, complex_options),
        at::empty({rows, 3}, complex_options),
        at::empty({num_tx, 3}, float_options),
        at::empty({num_rx, 3}, float_options)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    if (rows > 0) {
        path_result_capacity_backward_rows_kernel<<<
            launch_blocks(rows), kPathResultPackAdBlockSize, 0, stream>>>(
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int32_t>(),
            rx_id.data_ptr<int32_t>(),
            depth.data_ptr<int32_t>(),
            interaction_type.data_ptr<int32_t>(),
            interaction_positions.data_ptr<float>(),
            interaction_normals.data_ptr<float>(),
            tx_positions.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            grad_output,
            grad_input.view(),
            rows,
            width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int64_t endpoint_count = num_tx + num_rx;
    if (endpoint_count > 0) {
        path_result_capacity_backward_endpoints_kernel<<<
            launch_blocks(endpoint_count), kPathResultPackAdBlockSize, 0, stream>>>(
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int32_t>(),
            rx_id.data_ptr<int32_t>(),
            depth.data_ptr<int32_t>(),
            interaction_positions.data_ptr<float>(),
            tx_positions.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            grad_output,
            grad_input.tx_positions.data_ptr<float>(),
            grad_input.rx_positions.data_ptr<float>(),
            num_tx,
            num_rx,
            path_capacity_per_pair,
            width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return grad_input.dict();
}

pybind11::dict channel_path_result_capacity_pack_jvp(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor interaction_type,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor tx_positions,
    at::Tensor rx_positions,
    std::optional<at::Tensor> tangent_delay_s,
    std::optional<at::Tensor> tangent_field_direction,
    std::optional<at::Tensor> tangent_interaction_positions,
    std::optional<at::Tensor> tangent_interaction_normals,
    std::optional<at::Tensor> tangent_coefficient,
    std::optional<at::Tensor> tangent_field_xyz,
    std::optional<at::Tensor> tangent_tx_positions,
    std::optional<at::Tensor> tangent_rx_positions,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair) {
    check_primal(
        valid,
        tx_id,
        rx_id,
        depth,
        interaction_type,
        interaction_positions,
        interaction_normals,
        tx_positions,
        rx_positions,
        num_tx,
        num_rx,
        path_capacity_per_pair);
    const int device = valid.get_device();
    const int64_t rows = valid.numel();
    const int64_t width = interaction_type.size(1);
    InputTangents tangent_input{
        optional_view<float>(
            tangent_delay_s, "tangent_delay_s", at::kFloat, {rows}, device),
        optional_view<float>(tangent_field_direction,
                             "tangent_field_direction",
                             at::kFloat,
                             {rows, 3},
                             device),
        optional_view<float>(tangent_interaction_positions,
                             "tangent_interaction_positions",
                             at::kFloat,
                             {rows, width, 3},
                             device),
        optional_view<float>(tangent_interaction_normals,
                             "tangent_interaction_normals",
                             at::kFloat,
                             {rows, width, 3},
                             device),
        optional_view<cfloat>(tangent_coefficient,
                              "tangent_coefficient",
                              at::kComplexFloat,
                              {rows},
                              device),
        optional_view<cfloat>(tangent_field_xyz,
                              "tangent_field_xyz",
                              at::kComplexFloat,
                              {rows, 3},
                              device),
        optional_view<float>(tangent_tx_positions,
                             "tangent_tx_positions",
                             at::kFloat,
                             {num_tx, 3},
                             device),
        optional_view<float>(tangent_rx_positions,
                             "tangent_rx_positions",
                             at::kFloat,
                             {num_rx, 3},
                             device)};
    const auto float_options = valid.options().dtype(at::kFloat);
    const auto complex_options = valid.options().dtype(at::kComplexFloat);
    AllocatedOutputTangents tangent_output{
        at::empty({rows}, complex_options),
        at::empty({rows}, float_options),
        at::empty({rows}, float_options),
        at::empty({rows}, float_options),
        at::empty({rows}, float_options),
        at::empty({rows}, float_options),
        at::empty({rows, width, 3}, float_options),
        at::empty({rows, width, 3}, float_options),
        at::empty({rows, 3}, complex_options),
        at::empty({rows, 3}, float_options)};
    if (rows > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        path_result_capacity_jvp_kernel<<<
            launch_blocks(rows), kPathResultPackAdBlockSize, 0, stream>>>(
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int32_t>(),
            rx_id.data_ptr<int32_t>(),
            depth.data_ptr<int32_t>(),
            interaction_type.data_ptr<int32_t>(),
            interaction_positions.data_ptr<float>(),
            interaction_normals.data_ptr<float>(),
            tx_positions.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            tangent_input,
            tangent_output.view(),
            rows,
            width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return tangent_output.dict();
}
