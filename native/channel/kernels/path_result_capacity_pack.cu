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
#include <utility>

namespace {

constexpr int kPathResultPackBlockSize = 256;
constexpr int32_t kDiffractionComponent = 2;
constexpr int32_t kDiffractionInteraction = 2;
using cfloat = c10::complex<float>;
using channel::check_tensor;

struct PackInputs {
    const int32_t *tx_id;
    const int32_t *rx_id;
    const int32_t *depth;
    const int32_t *component_id;
    const int32_t *edge_id;
    const int32_t *primitive_sequence;
    const int32_t *material_sequence;
    const int32_t *interaction_type;
    const float *delay_s;
    const float *field_direction;
    const float *interaction_positions;
    const float *interaction_normals;
    const cfloat *coefficient;
    const cfloat *field_xyz;
    const float *tx_positions;
    const float *rx_positions;
};

struct PackOutputs {
    cfloat *a;
    float *tau;
    float *theta_t;
    float *phi_t;
    float *theta_r;
    float *phi_r;
    bool *valid;
    int32_t *interaction_type;
    int32_t *primitive_id;
    int32_t *material_id;
    float *position;
    float *normal;
    int32_t *num_paths;
    cfloat *field_xyz;
    float *field_direction;
};

struct Float3 {
    float x;
    float y;
    float z;
};

__device__ __forceinline__ Float3 load_float3(const float *data, int64_t row) {
    const int64_t offset = row * 3;
    return {data[offset], data[offset + 1], data[offset + 2]};
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

__device__ __forceinline__ void endpoint_angles(
    Float3 direction,
    float& theta,
    float& phi) {
    const float squared =
        direction.x * direction.x + direction.y * direction.y +
        direction.z * direction.z;
    const float norm = sqrtf(squared);
    const float safe = norm < FLT_MIN ? FLT_MIN : norm;
    const float cosine = clamp_preserve_nan(direction.z / safe, -1.0f, 1.0f);
    theta = acosf(cosine);
    phi = atan2f(direction.y, direction.x);
}

__global__ void path_result_capacity_pack_kernel(
    const int32_t *__restrict__ failure_bits,
    const bool *__restrict__ overflow,
    const bool *__restrict__ input_valid,
    const int32_t *__restrict__ input_num_paths,
    PackInputs input,
    PackOutputs output,
    int64_t row_capacity,
    int64_t pair_count,
    int64_t sequence_width) {
    const int64_t item_count = row_capacity > pair_count ? row_capacity : pair_count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < item_count;
         item += stride) {
        if (item < pair_count) {
            output.num_paths[item] = 0;
            if (failure_bits[0] == 0 && !overflow[0]) {
                output.num_paths[item] = input_num_paths[item];
            }
        }
        if (item >= row_capacity) {
            continue;
        }

        const int64_t row = item;
        output.a[row] = cfloat(0.0f, 0.0f);
        output.tau[row] = -1.0f;
        output.theta_t[row] = 0.0f;
        output.phi_t[row] = 0.0f;
        output.theta_r[row] = 0.0f;
        output.phi_r[row] = 0.0f;
        output.valid[row] = false;
        const int64_t row_vec = row * 3;
        for (int component = 0; component < 3; ++component) {
            output.field_xyz[row_vec + component] = cfloat(0.0f, 0.0f);
            output.field_direction[row_vec + component] = 0.0f;
        }
        const int64_t sequence_base = row * sequence_width;
        const int64_t sequence_vec_base = sequence_base * 3;
        for (int64_t slot = 0; slot < sequence_width; ++slot) {
            const int64_t sequence_index = sequence_base + slot;
            output.interaction_type[sequence_index] = 0;
            output.primitive_id[sequence_index] = -1;
            output.material_id[sequence_index] = -1;
            for (int component = 0; component < 3; ++component) {
                const int64_t vector_index = sequence_vec_base + slot * 3 + component;
                output.position[vector_index] = 0.0f;
                output.normal[vector_index] = 0.0f;
            }
        }

        if (failure_bits[0] != 0 || overflow[0] || !input_valid[row]) {
            continue;
        }

        // Endpoint ids are read only after validity and failure-state gates.
        const int32_t tx = input.tx_id[row];
        const int32_t rx = input.rx_id[row];
        const Float3 tx_position = load_float3(input.tx_positions, tx);
        const Float3 rx_position = load_float3(input.rx_positions, rx);
        const Float3 direct = subtract(rx_position, tx_position);
        const int32_t depth = input.depth[row];
        Float3 departure = direct;
        Float3 receiver_direction = negate(direct);
        if (sequence_width > 0 && depth > 0) {
            const Float3 first = load_float3(input.interaction_positions, sequence_base);
            const int64_t last_slot = static_cast<int64_t>(depth) - 1;
            const Float3 last =
                load_float3(input.interaction_positions, sequence_base + last_slot);
            departure = subtract(first, tx_position);
            const Float3 arrival = subtract(rx_position, last);
            receiver_direction = negate(arrival);
            // The zero-length endpoint branch is canonicalized by the fixed-
            // capacity contract to the direct last-rx subtraction. For every
            // nonzero direction, retain the former eager -(rx-last) bits.
            if (arrival.x == 0.0f && arrival.y == 0.0f && arrival.z == 0.0f) {
                receiver_direction = subtract(last, rx_position);
            }
        }
        endpoint_angles(
            departure, output.theta_t[row], output.phi_t[row]);
        endpoint_angles(
            receiver_direction, output.theta_r[row], output.phi_r[row]);

        output.a[row] = input.coefficient[row];
        output.tau[row] = input.delay_s[row];
        output.valid[row] = true;
        for (int component = 0; component < 3; ++component) {
            output.field_xyz[row_vec + component] =
                input.field_xyz[row_vec + component];
            output.field_direction[row_vec + component] =
                input.field_direction[row_vec + component];
        }
        for (int64_t slot = 0; slot < sequence_width; ++slot) {
            const int64_t sequence_index = sequence_base + slot;
            const int32_t interaction = input.interaction_type[sequence_index];
            output.interaction_type[sequence_index] = interaction;
            output.material_id[sequence_index] = input.material_sequence[sequence_index];
            output.primitive_id[sequence_index] =
                slot == 0 && input.component_id[row] == kDiffractionComponent &&
                        depth > 0
                    ? input.edge_id[row]
                    : input.primitive_sequence[sequence_index];
            for (int component = 0; component < 3; ++component) {
                const int64_t vector_index = sequence_vec_base + slot * 3 + component;
                output.position[vector_index] = input.interaction_positions[vector_index];
                const float normal = input.interaction_normals[vector_index];
                output.normal[vector_index] =
                    interaction == kDiffractionInteraction && !isfinite(normal)
                    ? 0.0f
                    : normal;
            }
        }
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>(
        (count + kPathResultPackBlockSize - 1) / kPathResultPackBlockSize);
}

void check_same_device(
    const at::Tensor& value,
    const at::Tensor& reference,
    const char *name) {
    TORCH_CHECK(
        value.get_device() == reference.get_device(),
        name,
        " must share failure_bits device");
}

void check_vec3(
    const at::Tensor& value,
    const at::Tensor& reference,
    const char *name,
    int64_t rows) {
    check_tensor(value, name, at::kFloat, 2);
    TORCH_CHECK(value.sizes() == at::IntArrayRef({rows, 3}), name, " has wrong shape");
    check_same_device(value, reference, name);
}

void check_sequence(
    const at::Tensor& value,
    const at::Tensor& reference,
    const char *name,
    c10::ScalarType dtype,
    int64_t rows,
    int64_t width,
    bool vector_tail) {
    check_tensor(value, name, dtype, vector_tail ? 3 : 2);
    if (vector_tail) {
        TORCH_CHECK(
            value.sizes() == at::IntArrayRef({rows, width, 3}),
            name,
            " has wrong shape");
    } else {
        TORCH_CHECK(
            value.sizes() == at::IntArrayRef({rows, width}),
            name,
            " has wrong shape");
    }
    check_same_device(value, reference, name);
}

}  // namespace

pybind11::dict channel_path_result_capacity_pack(
    at::Tensor failure_state,
    at::Tensor overflow,
    at::Tensor valid,
    at::Tensor num_paths,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor component_id,
    at::Tensor edge_id,
    at::Tensor primitive_sequence,
    at::Tensor material_sequence,
    at::Tensor interaction_type,
    at::Tensor delay_s,
    at::Tensor field_direction,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor coefficient,
    at::Tensor field_xyz,
    at::Tensor tx_positions,
    at::Tensor rx_positions,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair) {
    check_tensor(failure_state, "failure_state", at::kInt, 1);
    TORCH_CHECK(failure_state.numel() == 1, "failure_state must have shape (1,)");
    check_tensor(overflow, "overflow", at::kBool, 1);
    TORCH_CHECK(overflow.numel() == 1, "overflow must have shape (1,)");
    check_same_device(overflow, failure_state, "overflow");
    check_tensor(valid, "valid", at::kBool, 1);
    check_same_device(valid, failure_state, "valid");
    check_tensor(num_paths, "num_paths", at::kInt, 1);
    check_same_device(num_paths, failure_state, "num_paths");
    TORCH_CHECK(num_tx >= 0, "num_tx must be non-negative");
    TORCH_CHECK(num_rx >= 0, "num_rx must be non-negative");
    TORCH_CHECK(
        path_capacity_per_pair >= 0,
        "path_capacity_per_pair must be non-negative");
    constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
    TORCH_CHECK(
        num_tx == 0 || num_rx <= kInt64Max / num_tx,
        "num_tx * num_rx overflows int64");
    const int64_t pair_count = num_tx * num_rx;
    TORCH_CHECK(
        pair_count == 0 || path_capacity_per_pair <= kInt64Max / pair_count,
        "pair_count * path_capacity_per_pair overflows int64");
    const int64_t row_capacity = pair_count * path_capacity_per_pair;
    TORCH_CHECK(valid.numel() == row_capacity, "valid has wrong capacity shape");
    TORCH_CHECK(num_paths.numel() == pair_count, "num_paths has wrong pair shape");

    for (const auto& named : {
             std::pair<const char *, at::Tensor>{"tx_id", tx_id},
             {"rx_id", rx_id},
             {"depth", depth},
             {"component_id", component_id},
             {"edge_id", edge_id}}) {
        check_tensor(named.second, named.first, at::kInt, 1);
        TORCH_CHECK(named.second.numel() == row_capacity, named.first, " has wrong shape");
        check_same_device(named.second, failure_state, named.first);
    }
    check_tensor(delay_s, "delay_s", at::kFloat, 1);
    TORCH_CHECK(delay_s.numel() == row_capacity, "delay_s has wrong shape");
    check_same_device(delay_s, failure_state, "delay_s");
    const int64_t sequence_width = primitive_sequence.size(1);
    check_sequence(
        primitive_sequence,
        failure_state,
        "primitive_sequence",
        at::kInt,
        row_capacity,
        sequence_width,
        false);
    check_sequence(
        material_sequence,
        failure_state,
        "material_sequence",
        at::kInt,
        row_capacity,
        sequence_width,
        false);
    check_sequence(
        interaction_type,
        failure_state,
        "interaction_type",
        at::kInt,
        row_capacity,
        sequence_width,
        false);
    check_vec3(
        field_direction, failure_state, "field_direction", row_capacity);
    check_sequence(
        interaction_positions,
        failure_state,
        "interaction_positions",
        at::kFloat,
        row_capacity,
        sequence_width,
        true);
    check_sequence(
        interaction_normals,
        failure_state,
        "interaction_normals",
        at::kFloat,
        row_capacity,
        sequence_width,
        true);
    check_tensor(coefficient, "coefficient", at::kComplexFloat, 1);
    TORCH_CHECK(coefficient.numel() == row_capacity, "coefficient has wrong shape");
    check_same_device(coefficient, failure_state, "coefficient");
    check_tensor(field_xyz, "field_xyz", at::kComplexFloat, 2);
    TORCH_CHECK(
        field_xyz.sizes() == at::IntArrayRef({row_capacity, 3}),
        "field_xyz has wrong shape");
    check_same_device(field_xyz, failure_state, "field_xyz");
    check_vec3(tx_positions, failure_state, "tx_positions", num_tx);
    check_vec3(rx_positions, failure_state, "rx_positions", num_rx);

    const auto float_options = valid.options().dtype(at::kFloat);
    const auto int_options = valid.options().dtype(at::kInt);
    const auto bool_options = valid.options().dtype(at::kBool);
    const auto complex_options = valid.options().dtype(at::kComplexFloat);
    auto out_a = at::empty({row_capacity}, complex_options);
    auto out_tau = at::empty({row_capacity}, float_options);
    auto out_theta_t = at::empty({row_capacity}, float_options);
    auto out_phi_t = at::empty({row_capacity}, float_options);
    auto out_theta_r = at::empty({row_capacity}, float_options);
    auto out_phi_r = at::empty({row_capacity}, float_options);
    auto out_valid = at::empty({row_capacity}, bool_options);
    auto out_interaction_type =
        at::empty({row_capacity, sequence_width}, int_options);
    auto out_primitive_id = at::empty({row_capacity, sequence_width}, int_options);
    auto out_material_id = at::empty({row_capacity, sequence_width}, int_options);
    auto out_position =
        at::empty({row_capacity, sequence_width, 3}, float_options);
    auto out_normal = at::empty({row_capacity, sequence_width, 3}, float_options);
    auto out_num_paths = at::empty({pair_count}, int_options);
    auto out_field_xyz = at::empty({row_capacity, 3}, complex_options);
    auto out_field_direction = at::empty({row_capacity, 3}, float_options);

    const int64_t item_count = row_capacity > pair_count ? row_capacity : pair_count;
    if (item_count > 0) {
        const int device = valid.get_device();
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        path_result_capacity_pack_kernel<<<
            launch_blocks(item_count), kPathResultPackBlockSize, 0, stream>>>(
            failure_state.data_ptr<int32_t>(),
            overflow.data_ptr<bool>(),
            valid.data_ptr<bool>(),
            num_paths.data_ptr<int32_t>(),
            {tx_id.data_ptr<int32_t>(),
             rx_id.data_ptr<int32_t>(),
             depth.data_ptr<int32_t>(),
             component_id.data_ptr<int32_t>(),
             edge_id.data_ptr<int32_t>(),
             primitive_sequence.data_ptr<int32_t>(),
             material_sequence.data_ptr<int32_t>(),
             interaction_type.data_ptr<int32_t>(),
             delay_s.data_ptr<float>(),
             field_direction.data_ptr<float>(),
             interaction_positions.data_ptr<float>(),
             interaction_normals.data_ptr<float>(),
             coefficient.data_ptr<cfloat>(),
             field_xyz.data_ptr<cfloat>(),
             tx_positions.data_ptr<float>(),
             rx_positions.data_ptr<float>()},
            {out_a.data_ptr<cfloat>(),
             out_tau.data_ptr<float>(),
             out_theta_t.data_ptr<float>(),
             out_phi_t.data_ptr<float>(),
             out_theta_r.data_ptr<float>(),
             out_phi_r.data_ptr<float>(),
             out_valid.data_ptr<bool>(),
             out_interaction_type.data_ptr<int32_t>(),
             out_primitive_id.data_ptr<int32_t>(),
             out_material_id.data_ptr<int32_t>(),
             out_position.data_ptr<float>(),
             out_normal.data_ptr<float>(),
             out_num_paths.data_ptr<int32_t>(),
             out_field_xyz.data_ptr<cfloat>(),
             out_field_direction.data_ptr<float>()},
            row_capacity,
            pair_count,
            sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict result;
    result["a"] = out_a;
    result["tau"] = out_tau;
    result["theta_t"] = out_theta_t;
    result["phi_t"] = out_phi_t;
    result["theta_r"] = out_theta_r;
    result["phi_r"] = out_phi_r;
    result["valid"] = out_valid;
    result["interaction_type"] = out_interaction_type;
    result["primitive_id"] = out_primitive_id;
    result["material_id"] = out_material_id;
    result["position"] = out_position;
    result["normal"] = out_normal;
    result["num_paths"] = out_num_paths;
    result["field_xyz"] = out_field_xyz;
    result["field_direction"] = out_field_direction;
    return result;
}
