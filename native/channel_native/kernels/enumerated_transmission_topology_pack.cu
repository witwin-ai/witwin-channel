#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "../tensor_checks.h"
#include "capacity_failure_state.h"

#include <cstdint>
#include <limits>

namespace {

constexpr int kBlockSize = 256;
constexpr float kLightSpeedMetersPerSecond = 299792458.0f;

__device__ __forceinline__ float canonical_quiet_nan() {
    return __uint_as_float(0x7fc00000u);
}

struct TopologyOutput {
    bool *valid;
    int *tx_id;
    int *rx_id;
    int *depth;
    int *component_id;
    int *primitive_id;
    int *edge_id;
    float *path_length_m;
    float *delay_s;
    float *path_gain;
    c10::complex<float> *path_field;
    float *interaction_position;
    float *interaction_normal;
    int *material_id;
    int *primitive_sequence;
    int *material_sequence;
    float *interaction_positions;
    float *interaction_normals;
};

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

__global__ void transmission_topology_init_kernel(
    TopologyOutput output,
    int *candidate_count,
    int *guardrail_count,
    int64_t pair_count,
    int64_t hit_capacity) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < pair_count;
         row += stride) {
        output.valid[row] = false;
        output.tx_id[row] = -1;
        output.rx_id[row] = -1;
        output.depth[row] = 0;
        output.component_id[row] = -1;
        output.primitive_id[row] = -1;
        output.edge_id[row] = -1;
        output.path_length_m[row] = -1.0f;
        output.delay_s[row] = -1.0f;
        output.path_gain[row] = 0.0f;
        output.path_field[row] = c10::complex<float>(0.0f, 0.0f);
        output.material_id[row] = -1;
        for (int axis = 0; axis < 3; ++axis) {
            output.interaction_position[row * 3 + axis] = 0.0f;
            output.interaction_normal[row * 3 + axis] = 0.0f;
        }
        for (int64_t slot = 0; slot < hit_capacity; ++slot) {
            const int64_t sequence = row * hit_capacity + slot;
            output.primitive_sequence[sequence] = -1;
            output.material_sequence[sequence] = -1;
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t vector = sequence * 3 + axis;
                output.interaction_positions[vector] = 0.0f;
                output.interaction_normals[vector] = 0.0f;
            }
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        candidate_count[0] = 0;
        guardrail_count[0] = 0;
    }
}

__global__ void transmission_topology_preflight_kernel(
    int *failure_state,
    const bool *valid,
    const int *num_hits,
    const bool *reached_target,
    const bool *overflow,
    const int *global_primitive_id,
    int64_t pair_count,
    int64_t hit_capacity,
    int64_t face_count) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < pair_count;
         row += stride) {
        if (overflow[row]) {
            atomicOr(
                failure_state,
                channel_native::capacity::kSegmentPenetrationFailure);
            continue;
        }
        const int hits = num_hits[row];
        bool contract_ok =
            hits >= 0 && static_cast<int64_t>(hits) <= hit_capacity &&
            (reached_target[row] || hits == 0);
        for (int64_t slot = 0; slot < hit_capacity; ++slot) {
            const bool expected_valid = slot < hits;
            const int64_t index = row * hit_capacity + slot;
            if (valid[index] != expected_valid) {
                contract_ok = false;
            }
        }
        if (!contract_ok) {
            atomicOr(
                failure_state,
                channel_native::capacity::kSegmentPenetrationFailure);
            continue;
        }
        for (int slot = 0; slot < hits; ++slot) {
            const int primitive = global_primitive_id[row * hit_capacity + slot];
            if (primitive < 0 || static_cast<int64_t>(primitive) >= face_count) {
                atomicOr(
                    failure_state,
                    channel_native::capacity::kSegmentPenetrationFailure);
                break;
            }
        }
    }
}

__global__ void transmission_topology_pack_kernel(
    const int *failure_state,
    const bool *reached_target,
    const int *num_hits,
    const float *distance,
    const float *position,
    const float *normal,
    const int *global_primitive_id,
    const int *face_material_id,
    const int *geometry_mode_id,
    TopologyOutput output,
    int *candidate_count,
    int *guardrail_count,
    int64_t pair_count,
    int64_t hit_capacity,
    int64_t material_count,
    int64_t rx_count) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < pair_count;
         row += stride) {
        const int hits = num_hits[row];
        const bool penetrated = reached_target[row] && hits >= 1;
        if (!penetrated) {
            continue;
        }
        atomicAdd(candidate_count, 1);
        bool bad_material = false;
        for (int slot = 0; slot < hits; ++slot) {
            const int64_t sequence = row * hit_capacity + slot;
            const int primitive = global_primitive_id[sequence];
            const int material = face_material_id[primitive];
            if (material < 0 || static_cast<int64_t>(material) >= material_count ||
                geometry_mode_id[material] != 0) {
                bad_material = true;
            }
        }
        if (bad_material) {
            atomicAdd(guardrail_count, 1);
            continue;
        }
        output.valid[row] = true;
        output.tx_id[row] = static_cast<int>(row / rx_count);
        output.rx_id[row] = static_cast<int>(row % rx_count);
        output.depth[row] = hits;
        output.component_id[row] = 5;
        output.edge_id[row] = -1;
        output.path_length_m[row] = distance[row];
        output.delay_s[row] = distance[row] / kLightSpeedMetersPerSecond;
        const float nan = canonical_quiet_nan();
        output.path_gain[row] = nan;
        output.path_field[row] = c10::complex<float>(nan, nan);
        for (int64_t slot = 0; slot < hits; ++slot) {
            const int64_t sequence = row * hit_capacity + slot;
            const int primitive = global_primitive_id[sequence];
            const int material = face_material_id[primitive];
            output.primitive_sequence[sequence] = primitive;
            output.material_sequence[sequence] = material;
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t vector = sequence * 3 + axis;
                output.interaction_positions[vector] = position[vector];
                output.interaction_normals[vector] = normal[vector];
                if (slot == 0) {
                    output.interaction_position[row * 3 + axis] = position[vector];
                    output.interaction_normal[row * 3 + axis] = normal[vector];
                }
            }
            if (slot == 0) {
                output.primitive_id[row] = primitive;
                output.material_id[row] = material;
            }
        }
    }
}

__global__ void transmission_topology_sanitize_kernel(
    const int *failure_state,
    TopologyOutput output,
    int *candidate_count,
    int *guardrail_count,
    int64_t pair_count,
    int64_t hit_capacity) {
    if (failure_state[0] == 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < pair_count;
         row += stride) {
        output.valid[row] = false;
        output.tx_id[row] = -1;
        output.rx_id[row] = -1;
        output.depth[row] = 0;
        output.component_id[row] = -1;
        output.primitive_id[row] = -1;
        output.edge_id[row] = -1;
        output.path_length_m[row] = -1.0f;
        output.delay_s[row] = -1.0f;
        output.path_gain[row] = 0.0f;
        output.path_field[row] = c10::complex<float>(0.0f, 0.0f);
        output.material_id[row] = -1;
        for (int axis = 0; axis < 3; ++axis) {
            output.interaction_position[row * 3 + axis] = 0.0f;
            output.interaction_normal[row * 3 + axis] = 0.0f;
        }
        for (int64_t slot = 0; slot < hit_capacity; ++slot) {
            const int64_t sequence = row * hit_capacity + slot;
            output.primitive_sequence[sequence] = -1;
            output.material_sequence[sequence] = -1;
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t vector = sequence * 3 + axis;
                output.interaction_positions[vector] = 0.0f;
                output.interaction_normals[vector] = 0.0f;
            }
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        candidate_count[0] = 0;
        guardrail_count[0] = 0;
    }
}

void check_rows(
    const at::Tensor& tensor,
    const char *name,
    int64_t pair_count,
    int device) {
    TORCH_CHECK(tensor.size(0) == pair_count, name, " must match segment rows");
    TORCH_CHECK(tensor.get_device() == device, name, " must share segment device");
}

}  // namespace

pybind11::dict cn_enumerated_transmission_topology_pack(
    at::Tensor failure_state,
    at::Tensor valid,
    at::Tensor num_hits,
    at::Tensor reached_target,
    at::Tensor overflow,
    at::Tensor distance,
    at::Tensor position,
    at::Tensor normal,
    at::Tensor global_primitive_id,
    at::Tensor face_material_id,
    at::Tensor geometry_mode_id,
    int64_t tx_count,
    int64_t rx_count) {
    channel_native::check_tensor(valid, "valid", at::kBool, 2);
    channel_native::capacity::validate_failure_state(failure_state, valid);
    channel_native::check_tensor(num_hits, "num_hits", at::kInt, 1);
    channel_native::check_tensor(reached_target, "reached_target", at::kBool, 1);
    channel_native::check_tensor(overflow, "overflow", at::kBool, 1);
    channel_native::check_tensor(distance, "distance", at::kFloat, 1);
    channel_native::check_tensor(position, "position", at::kFloat, 3);
    channel_native::check_tensor(normal, "normal", at::kFloat, 3);
    channel_native::check_tensor(
        global_primitive_id, "global_primitive_id", at::kInt, 2);
    channel_native::check_tensor(
        face_material_id, "face_material_id", at::kInt, 1);
    channel_native::check_tensor(
        geometry_mode_id, "geometry_mode_id", at::kInt, 1);
    TORCH_CHECK(tx_count >= 0 && rx_count >= 0, "tx_count and rx_count must be non-negative");
    TORCH_CHECK(
        tx_count == 0 || rx_count <= std::numeric_limits<int64_t>::max() / tx_count,
        "endpoint pair count overflows int64");
    const int64_t pair_count = tx_count * rx_count;
    const int64_t hit_capacity = valid.size(1);
    TORCH_CHECK(valid.size(0) == pair_count, "valid rows must equal tx_count * rx_count");
    TORCH_CHECK(pair_count <= std::numeric_limits<int>::max(), "pair count exceeds int32");
    TORCH_CHECK(hit_capacity <= std::numeric_limits<int>::max(), "hit capacity exceeds int32");
    TORCH_CHECK(
        position.sizes() == at::IntArrayRef({pair_count, hit_capacity, 3}),
        "position must have shape (P, D, 3)");
    TORCH_CHECK(normal.sizes() == position.sizes(), "normal must match position");
    TORCH_CHECK(global_primitive_id.sizes() == valid.sizes(), "global_primitive_id must match valid");
    const int device = valid.get_device();
    check_rows(num_hits, "num_hits", pair_count, device);
    check_rows(reached_target, "reached_target", pair_count, device);
    check_rows(overflow, "overflow", pair_count, device);
    check_rows(distance, "distance", pair_count, device);
    check_rows(position, "position", pair_count, device);
    check_rows(normal, "normal", pair_count, device);
    check_rows(
        global_primitive_id, "global_primitive_id", pair_count, device);
    TORCH_CHECK(face_material_id.get_device() == device, "face_material_id must share segment device");
    TORCH_CHECK(geometry_mode_id.get_device() == device, "geometry_mode_id must share segment device");
    const c10::cuda::CUDAGuard device_guard(valid.device());

    auto bool_options = valid.options().dtype(at::kBool);
    auto int_options = valid.options().dtype(at::kInt);
    auto float_options = valid.options().dtype(at::kFloat);
    auto complex_options = valid.options().dtype(at::kComplexFloat);
    auto out_valid = at::empty({pair_count}, bool_options);
    auto tx_id = at::empty({pair_count}, int_options);
    auto rx_id = at::empty({pair_count}, int_options);
    auto depth = at::empty({pair_count}, int_options);
    auto component_id = at::empty({pair_count}, int_options);
    auto primitive_id = at::empty({pair_count}, int_options);
    auto edge_id = at::empty({pair_count}, int_options);
    auto path_length_m = at::empty({pair_count}, float_options);
    auto delay_s = at::empty({pair_count}, float_options);
    auto path_gain = at::empty({pair_count}, float_options);
    auto path_field = at::empty({pair_count}, complex_options);
    auto interaction_position = at::empty({pair_count, 3}, float_options);
    auto interaction_normal = at::empty({pair_count, 3}, float_options);
    auto material_id = at::empty({pair_count}, int_options);
    auto primitive_sequence = at::empty({pair_count, hit_capacity}, int_options);
    auto material_sequence = at::empty({pair_count, hit_capacity}, int_options);
    auto interaction_positions = at::empty({pair_count, hit_capacity, 3}, float_options);
    auto interaction_normals = at::empty({pair_count, hit_capacity, 3}, float_options);
    auto candidate_count = at::empty({1}, int_options);
    auto guardrail_count = at::empty({1}, int_options);
    const TopologyOutput output{
        out_valid.data_ptr<bool>(), tx_id.data_ptr<int>(), rx_id.data_ptr<int>(),
        depth.data_ptr<int>(), component_id.data_ptr<int>(), primitive_id.data_ptr<int>(),
        edge_id.data_ptr<int>(), path_length_m.data_ptr<float>(), delay_s.data_ptr<float>(),
        path_gain.data_ptr<float>(), path_field.data_ptr<c10::complex<float>>(),
        interaction_position.data_ptr<float>(), interaction_normal.data_ptr<float>(),
        material_id.data_ptr<int>(), primitive_sequence.data_ptr<int>(),
        material_sequence.data_ptr<int>(), interaction_positions.data_ptr<float>(),
        interaction_normals.data_ptr<float>()};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const int64_t init_count = pair_count > 0 ? pair_count : 1;
    transmission_topology_init_kernel<<<
        launch_blocks(init_count), kBlockSize, 0, stream>>>(
        output, candidate_count.data_ptr<int>(), guardrail_count.data_ptr<int>(),
        pair_count, hit_capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (pair_count > 0) {
        transmission_topology_preflight_kernel<<<
            launch_blocks(pair_count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), valid.data_ptr<bool>(), num_hits.data_ptr<int>(),
            reached_target.data_ptr<bool>(), overflow.data_ptr<bool>(),
            global_primitive_id.data_ptr<int>(), pair_count, hit_capacity,
            face_material_id.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        transmission_topology_pack_kernel<<<
            launch_blocks(pair_count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), reached_target.data_ptr<bool>(),
            num_hits.data_ptr<int>(),
            distance.data_ptr<float>(), position.data_ptr<float>(),
            normal.data_ptr<float>(), global_primitive_id.data_ptr<int>(),
            face_material_id.data_ptr<int>(), geometry_mode_id.data_ptr<int>(), output,
            candidate_count.data_ptr<int>(), guardrail_count.data_ptr<int>(), pair_count,
            hit_capacity, geometry_mode_id.numel(), rx_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        transmission_topology_sanitize_kernel<<<
            launch_blocks(pair_count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), output, candidate_count.data_ptr<int>(),
            guardrail_count.data_ptr<int>(), pair_count, hit_capacity);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict result;
    result["valid"] = out_valid;
    result["tx_id"] = tx_id;
    result["rx_id"] = rx_id;
    result["depth"] = depth;
    result["component_id"] = component_id;
    result["primitive_id"] = primitive_id;
    result["edge_id"] = edge_id;
    result["path_length_m"] = path_length_m;
    result["delay_s"] = delay_s;
    result["path_gain"] = path_gain;
    result["path_field"] = path_field;
    result["interaction_position"] = interaction_position;
    result["interaction_normal"] = interaction_normal;
    result["material_id"] = material_id;
    result["primitive_sequence"] = primitive_sequence;
    result["material_sequence"] = material_sequence;
    result["interaction_positions"] = interaction_positions;
    result["interaction_normals"] = interaction_normals;
    result["device_candidate_count"] = candidate_count;
    result["device_guardrail_count"] = guardrail_count;
    return result;
}
