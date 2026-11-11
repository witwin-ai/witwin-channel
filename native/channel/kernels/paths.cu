// Copyright Xingyu Chen.
// Implements path topology and compaction CUDA operations.

// ==== Section: Path tracing ====
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include "torch_cuda.h"
#include "math.cuh"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "path_compaction.cuh"
#include "constants.cuh"

namespace {

using channel::kPiF;
using channel::kSpeedOfLight;
constexpr float kReflectionBaryEps = 1.0e-4f;
constexpr float kReflectionVisEps = 1.0e-4f;

__device__ float channel_neg_kd_phase(float k, float d) {
    // Reduce k*d mod 2*pi in double: the f32 product loses ~k*d*2^-24 of
    // phase, which shifts coherent nulls at mmWave ranges.
    const double kd = fmod(static_cast<double>(k) * static_cast<double>(d), channel::kTwoPi);
    return -static_cast<float>(kd);
}

__global__ void path_concat_vec3_kernel(
    int64_t count,
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t dst_offset) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int64_t src_base = index * 3;
        const int64_t dst_base = (dst_offset + index) * 3;
        dst[dst_base + 0] = src[src_base + 0];
        dst[dst_base + 1] = src[src_base + 1];
        dst[dst_base + 2] = src[src_base + 2];
    }
}

__global__ void path_los_visibility_inputs_kernel(
    int64_t count,
    const float* __restrict__ tx_positions,
    const float* __restrict__ rx_positions,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    float* __restrict__ start,
    float* __restrict__ end,
    bool* __restrict__ active) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int tx = tx_id[index];
        const int rx = rx_id[index];
        const float* tx_p = tx_positions + static_cast<int64_t>(tx) * 3;
        const float* rx_p = rx_positions + static_cast<int64_t>(rx) * 3;
        float* s = start + index * 3;
        float* e = end + index * 3;
        s[0] = tx_p[0];
        s[1] = tx_p[1];
        s[2] = tx_p[2];
        e[0] = rx_p[0];
        e[1] = rx_p[1];
        e[2] = rx_p[2];
        active[index] = true;
    }
}

__global__ void deterministic_los_topology_all_kernel(
    int64_t count,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    const float* __restrict__ path_length,
    const float* __restrict__ delay,
    const float* __restrict__ path_gain,
    float frequency_hz,
    int64_t sequence_width,
    bool* __restrict__ out_valid,
    int* __restrict__ out_tx_id,
    int* __restrict__ out_rx_id,
    int* __restrict__ out_depth,
    int* __restrict__ out_component_id,
    int* __restrict__ out_primitive_id,
    int* __restrict__ out_edge_id,
    float* __restrict__ out_path_length,
    float* __restrict__ out_delay,
    float* __restrict__ out_path_gain,
    c10::complex<float>* __restrict__ out_path_field,
    float* __restrict__ out_interaction_position,
    float* __restrict__ out_interaction_normal,
    int* __restrict__ out_material_id,
    int* __restrict__ out_primitive_sequence,
    int* __restrict__ out_material_sequence,
    float* __restrict__ out_interaction_positions,
    float* __restrict__ out_interaction_normals) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const float length = path_length[index];
        const float gain = path_gain[index];
        const float amplitude = sqrtf(fmaxf(gain, 0.0f));
        const float wavelength = static_cast<float>(kSpeedOfLight) / frequency_hz;
        const float phase = channel_neg_kd_phase(2.0f * kPiF / wavelength, length);

        out_valid[index] = true;
        out_tx_id[index] = tx_id[index];
        out_rx_id[index] = rx_id[index];
        out_depth[index] = 0;
        out_component_id[index] = 0;
        out_primitive_id[index] = -1;
        out_edge_id[index] = -1;
        out_path_length[index] = length;
        out_delay[index] = delay[index];
        out_path_gain[index] = gain;
        out_path_field[index] = c10::complex<float>(amplitude * cosf(phase), amplitude * sinf(phase));
        out_material_id[index] = -1;

        const int64_t vec_base = index * 3;
        for (int axis = 0; axis < 3; ++axis) {
            out_interaction_position[vec_base + axis] = 0.0f;
            out_interaction_normal[vec_base + axis] = 0.0f;
        }
        for (int64_t column = 0; column < sequence_width; ++column) {
            const int64_t seq_index = index * sequence_width + column;
            out_primitive_sequence[seq_index] = -1;
            out_material_sequence[seq_index] = -1;
            const int64_t seq_vec_base = seq_index * 3;
            for (int axis = 0; axis < 3; ++axis) {
                out_interaction_positions[seq_vec_base + axis] = 0.0f;
                out_interaction_normals[seq_vec_base + axis] = 0.0f;
            }
        }
    }
}

__global__ void deterministic_topology_default_fields_kernel(
    int64_t count,
    float* __restrict__ interaction_position,
    float* __restrict__ interaction_normal,
    int* __restrict__ material_id,
    c10::complex<float>* __restrict__ path_field) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        material_id[index] = -1;
        path_field[index] = c10::complex<float>(0.0f, 0.0f);
        const int64_t vec_base = index * 3;
        for (int axis = 0; axis < 3; ++axis) {
            interaction_position[vec_base + axis] = 0.0f;
            interaction_normal[vec_base + axis] = 0.0f;
        }
    }
}

__global__ void deterministic_pad_topology_sequences_kernel(
    int64_t count,
    int64_t width,
    int64_t primitive_source_width,
    int64_t material_source_width,
    int64_t position_source_width,
    int64_t normal_source_width,
    const int* __restrict__ depth,
    const int* __restrict__ primitive_id,
    const int* __restrict__ material_id,
    const float* __restrict__ interaction_position,
    const float* __restrict__ interaction_normal,
    const int* __restrict__ primitive_source,
    const int* __restrict__ material_source,
    const float* __restrict__ position_source,
    const float* __restrict__ normal_source,
    int* __restrict__ primitive_out,
    int* __restrict__ material_out,
    float* __restrict__ position_out,
    float* __restrict__ normal_out) {
    const int64_t total = count * width;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += stride) {
        const int64_t index = flat / width;
        const int64_t column = flat - index * width;
        const bool has_depth = depth[index] > 0;

        int primitive_value = -1;
        if (column < primitive_source_width) {
            primitive_value = primitive_source[index * primitive_source_width + column];
        } else if (primitive_source_width == 0 && column == 0 && has_depth) {
            primitive_value = primitive_id[index];
        }
        primitive_out[flat] = primitive_value;

        int material_value = -1;
        if (column < material_source_width) {
            material_value = material_source[index * material_source_width + column];
        } else if (material_source_width == 0 && column == 0 && has_depth) {
            material_value = material_id[index];
        }
        material_out[flat] = material_value;

        const int64_t vec_base = flat * 3;
        for (int axis = 0; axis < 3; ++axis) {
            float position_value = 0.0f;
            if (column < position_source_width) {
                position_value = position_source[(index * position_source_width + column) * 3 + axis];
            } else if (position_source_width == 0 && column == 0 && has_depth) {
                position_value = interaction_position[index * 3 + axis];
            }
            position_out[vec_base + axis] = position_value;

            float normal_value = 0.0f;
            if (column < normal_source_width) {
                normal_value = normal_source[(index * normal_source_width + column) * 3 + axis];
            } else if (normal_source_width == 0 && column == 0 && has_depth) {
                normal_value = interaction_normal[index * 3 + axis];
            }
            normal_out[vec_base + axis] = normal_value;
        }
    }
}

__global__ void deterministic_face_anchor_points_kernel(
    int64_t face_count,
    const float* __restrict__ vertices,
    const int* __restrict__ faces,
    float* __restrict__ out) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t face = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         face < face_count;
         face += stride) {
        const int vertex_index = faces[face * 3];
        const int64_t vertex_base = static_cast<int64_t>(vertex_index) * 3;
        const int64_t out_base = face * 3;
        out[out_base + 0] = vertices[vertex_base + 0];
        out[out_base + 1] = vertices[vertex_base + 1];
        out[out_base + 2] = vertices[vertex_base + 2];
    }
}

// Candidate c of [candidate_start, candidate_start + candidate_count) maps to
// (transmitter, receiver, sequence). With shared sequences (sequence_tx null)
// c = (tx * rx_count + rx) * sequence_count + sequence; with per-transmitter
// sequences c = sequence * rx_count + rx and sequence_tx names the transmitter.
// With explicit candidate lists, candidate c is (candidate_sequence[c],
// candidate_receiver[c]) and sequence_tx names the transmitter.
template <typename index_t>
__global__ void deterministic_reflection_epc_input_batch_kernel(
    int64_t candidate_count,
    int64_t candidate_start,
    int64_t sequence_count,
    int64_t depth,
    int64_t rx_count,
    const int* __restrict__ sequence_tx,
    const int* __restrict__ candidate_sequence,
    const int* __restrict__ candidate_receiver,
    const float* __restrict__ tx_positions,
    const float* __restrict__ rx_positions,
    const index_t* __restrict__ sequences,
    const float* __restrict__ tri_a,
    const float* __restrict__ normals,
    float* __restrict__ tx_batch,
    float* __restrict__ rx_batch,
    int* __restrict__ tx_indices,
    int* __restrict__ rx_indices,
    int* __restrict__ sequence_batch,
    float* __restrict__ direct_plane_points,
    float* __restrict__ direct_plane_normals) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < candidate_count;
         pair += stride) {
        const int64_t candidate = candidate_start + pair;
        int64_t sequence_index = 0;
        int64_t rx_id = 0;
        int64_t tx_id = 0;
        if (candidate_sequence != nullptr) {
            sequence_index = candidate_sequence[candidate];
            rx_id = candidate_receiver[candidate];
            tx_id = sequence_tx[sequence_index];
        } else if (sequence_tx == nullptr) {
            const int64_t endpoint_pair = candidate / sequence_count;
            sequence_index = candidate - endpoint_pair * sequence_count;
            tx_id = endpoint_pair / rx_count;
            rx_id = endpoint_pair - tx_id * rx_count;
        } else {
            sequence_index = candidate / rx_count;
            rx_id = candidate - sequence_index * rx_count;
            tx_id = sequence_tx[sequence_index];
        }
        rx_indices[pair] = static_cast<int>(rx_id);
        tx_indices[pair] = static_cast<int>(tx_id);

        const int64_t vec_base = pair * 3;
        for (int axis = 0; axis < 3; ++axis) {
            tx_batch[vec_base + axis] = tx_positions[tx_id * 3 + axis];
            rx_batch[vec_base + axis] = rx_positions[rx_id * 3 + axis];
        }

        for (int64_t column = 0; column < depth; ++column) {
            const int64_t sequence_offset = sequence_index * depth + column;
            const int face = static_cast<int>(sequences[sequence_offset]);
            sequence_batch[pair * depth + column] = face;
            const int64_t face_base = static_cast<int64_t>(face) * 3;
            const int64_t out_base = (pair * depth + column) * 3;
            for (int axis = 0; axis < 3; ++axis) {
                direct_plane_points[out_base + axis] = tri_a[face_base + axis];
                direct_plane_normals[out_base + axis] = normals[face_base + axis];
            }
        }
    }
}

__global__ void deterministic_normalize_vec3_kernel(
    int64_t count,
    const float* __restrict__ values,
    float eps,
    float* __restrict__ out) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int64_t base = index * 3;
        const float x = values[base + 0];
        const float y = values[base + 1];
        const float z = values[base + 2];
        const float length = fmaxf(sqrtf(x * x + y * y + z * z), eps);
        out[base + 0] = x / length;
        out[base + 1] = y / length;
        out[base + 2] = z / length;
    }
}

__device__ bool inside_triangle3(
    const float* p,
    const float* a,
    const float* b,
    const float* c) {
    const float v0x = b[0] - a[0];
    const float v0y = b[1] - a[1];
    const float v0z = b[2] - a[2];
    const float v1x = c[0] - a[0];
    const float v1y = c[1] - a[1];
    const float v1z = c[2] - a[2];
    const float v2x = p[0] - a[0];
    const float v2y = p[1] - a[1];
    const float v2z = p[2] - a[2];
    const float d00 = v0x * v0x + v0y * v0y + v0z * v0z;
    const float d01 = v0x * v1x + v0y * v1y + v0z * v1z;
    const float d11 = v1x * v1x + v1y * v1y + v1z * v1z;
    const float d20 = v2x * v0x + v2y * v0y + v2z * v0z;
    const float d21 = v2x * v1x + v2y * v1y + v2z * v1z;
    const float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) <= 1.0e-12f) {
        return false;
    }
    const float inv = 1.0f / denom;
    const float v = (d11 * d20 - d01 * d21) * inv;
    const float w = (d00 * d21 - d01 * d20) * inv;
    const float u = 1.0f - v - w;
    return u >= -kReflectionBaryEps && v >= -kReflectionBaryEps && w >= -kReflectionBaryEps &&
        u <= 1.0f + kReflectionBaryEps && v <= 1.0f + kReflectionBaryEps && w <= 1.0f + kReflectionBaryEps;
}

__global__ void path_reflection_candidates_kernel(
    int64_t count,
    int64_t face_count,
    int64_t rx_count,
    const float* __restrict__ vertices,
    const int* __restrict__ faces,
    const float* __restrict__ face_normals,
    const float* __restrict__ face_gain,
    const float* __restrict__ tx_positions,
    const float* __restrict__ tx_power,
    const float* __restrict__ rx_positions,
    float wavelength,
    bool* __restrict__ valid,
    int* __restrict__ tx_id,
    int* __restrict__ rx_id,
    int* __restrict__ depth,
    int* __restrict__ component_id,
    int* __restrict__ primitive_id,
    int* __restrict__ edge_id,
    float* __restrict__ path_length,
    float* __restrict__ delay,
    float* __restrict__ path_gain,
    float* __restrict__ seg0_start,
    float* __restrict__ seg0_end,
    float* __restrict__ seg1_start,
    float* __restrict__ seg1_end,
    bool* __restrict__ active) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int face = static_cast<int>(index % face_count);
        const int64_t pair = index / face_count;
        const int rx = static_cast<int>(pair % rx_count);
        const int tx = static_cast<int>(pair / rx_count);

        valid[index] = false;
        active[index] = false;
        tx_id[index] = tx;
        rx_id[index] = rx;
        depth[index] = 1;
        component_id[index] = 1;
        primitive_id[index] = face;
        edge_id[index] = -1;
        path_length[index] = 0.0f;
        delay[index] = 0.0f;
        path_gain[index] = 0.0f;

        const int i0 = faces[static_cast<int64_t>(face) * 3 + 0];
        const int i1 = faces[static_cast<int64_t>(face) * 3 + 1];
        const int i2 = faces[static_cast<int64_t>(face) * 3 + 2];
        const float* a = vertices + static_cast<int64_t>(i0) * 3;
        const float* b = vertices + static_cast<int64_t>(i1) * 3;
        const float* c = vertices + static_cast<int64_t>(i2) * 3;
        const float* tx_p = tx_positions + static_cast<int64_t>(tx) * 3;
        const float* rx_p = rx_positions + static_cast<int64_t>(rx) * 3;
        const float* raw_n = face_normals + static_cast<int64_t>(face) * 3;
        const float n_len = fmaxf(channel::math::length_values(raw_n[0], raw_n[1], raw_n[2]), 1.0e-12f);
        const float n[3] = {raw_n[0] / n_len, raw_n[1] / n_len, raw_n[2] / n_len};

        const float tx_vec[3] = {tx_p[0] - a[0], tx_p[1] - a[1], tx_p[2] - a[2]};
        const float tx_to_plane = channel::math::dot_values(tx_vec, n);
        const float image[3] = {
            tx_p[0] - 2.0f * tx_to_plane * n[0],
            tx_p[1] - 2.0f * tx_to_plane * n[1],
            tx_p[2] - 2.0f * tx_to_plane * n[2],
        };
        const float line[3] = {rx_p[0] - image[0], rx_p[1] - image[1], rx_p[2] - image[2]};
        const float denom = channel::math::dot_values(line, n);
        if (fabsf(denom) <= 1.0e-8f) {
            continue;
        }
        const float plane_vec[3] = {a[0] - image[0], a[1] - image[1], a[2] - image[2]};
        const float s = channel::math::dot_values(plane_vec, n) / denom;
        if (s <= 1.0e-6f || s >= 1.0f - 1.0e-6f) {
            continue;
        }
        const float p[3] = {
            image[0] + s * line[0],
            image[1] + s * line[1],
            image[2] + s * line[2],
        };
        if (!inside_triangle3(p, a, b, c)) {
            continue;
        }

        const float tx_dx = p[0] - tx_p[0];
        const float tx_dy = p[1] - tx_p[1];
        const float tx_dz = p[2] - tx_p[2];
        const float rx_dx = rx_p[0] - p[0];
        const float rx_dy = rx_p[1] - p[1];
        const float rx_dz = rx_p[2] - p[2];
        const float d0_len = channel::math::length_values(tx_dx, tx_dy, tx_dz);
        const float d1_len = channel::math::length_values(rx_dx, rx_dy, rx_dz);
        if (d0_len <= 1.0e-6f || d1_len <= 1.0e-6f) {
            continue;
        }
        const float inv0 = 1.0f / d0_len;
        const float inv1 = 1.0f / d1_len;
        const float d0[3] = {tx_dx * inv0, tx_dy * inv0, tx_dz * inv0};
        const float d1[3] = {rx_dx * inv1, rx_dy * inv1, rx_dz * inv1};
        float* s0 = seg0_start + index * 3;
        float* e0 = seg0_end + index * 3;
        float* s1 = seg1_start + index * 3;
        float* e1 = seg1_end + index * 3;
        for (int axis = 0; axis < 3; ++axis) {
            s0[axis] = tx_p[axis] + d0[axis] * kReflectionVisEps;
            e0[axis] = p[axis] - d0[axis] * kReflectionVisEps;
            s1[axis] = p[axis] + d1[axis] * kReflectionVisEps;
            e1[axis] = rx_p[axis] - d1[axis] * kReflectionVisEps;
        }

        const float total_length = fmaxf(d0_len + d1_len, 1.0e-6f);
        const float gain = fmaxf(face_gain[face], 0.0f);
        const float scale = wavelength / (12.566370614359172f * total_length);
        valid[index] = true;
        active[index] = true;
        path_length[index] = total_length;
        delay[index] = total_length / static_cast<float>(kSpeedOfLight);
        path_gain[index] = tx_power[tx] * gain * scale * scale;
    }
}

}  // namespace

at::Tensor channel_path_concat_vec3_cuda(std::vector<at::Tensor> blocks) {
    TORCH_CHECK(!blocks.empty(), "path_concat_vec3 requires at least one block");
    int64_t total = 0;
    int device = blocks[0].get_device();
    for (size_t i = 0; i < blocks.size(); ++i) {
        check_vec3_table(blocks[i], "vec3 block");
        TORCH_CHECK(blocks[i].get_device() == device, "all vec3 blocks must share a device");
        total += blocks[i].size(0);
    }
    auto out = at::empty({total, 3}, blocks[0].options().dtype(at::kFloat));
    if (total == 0) {
        return out;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    int64_t offset = 0;
    for (const auto& block : blocks) {
        const int64_t count = block.size(0);
        if (count > 0) {
            path_concat_vec3_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                      kPathBlockSize, 0, stream>>>(
                count,
                block.data_ptr<float>(),
                out.data_ptr<float>(),
                offset);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        offset += count;
    }
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_path_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    at::Tensor rx_positions,
    at::Tensor tx_id,
    at::Tensor rx_id) {
    check_vec3_table(tx_positions, "tx_positions");
    check_vec3_table(rx_positions, "rx_positions");
    check_tensor(tx_id, "tx_id", at::kInt, 1);
    check_tensor(rx_id, "rx_id", at::kInt, 1);
    TORCH_CHECK(tx_positions.get_device() == rx_positions.get_device(), "tx_positions and rx_positions must share a device");
    TORCH_CHECK(tx_id.get_device() == tx_positions.get_device(), "tx_id must share tx_positions device");
    TORCH_CHECK(rx_id.get_device() == tx_positions.get_device(), "rx_id must share tx_positions device");
    TORCH_CHECK(rx_id.size(0) == tx_id.size(0), "rx_id must match tx_id");

    const int64_t count = tx_id.size(0);
    auto start = at::empty({count, 3}, tx_positions.options().dtype(at::kFloat));
    auto end = at::empty({count, 3}, tx_positions.options().dtype(at::kFloat));
    auto active = at::empty({count}, tx_positions.options().dtype(at::kBool));
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        path_los_visibility_inputs_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                            kPathBlockSize, 0, stream>>>(
            count,
            tx_positions.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            start.data_ptr<float>(),
            end.data_ptr<float>(),
            active.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {start, end, active};
}

pybind11::dict channel_deterministic_los_topology_block_all_visible(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor path_length,
    at::Tensor delay,
    at::Tensor path_gain,
    double frequency_hz,
    int64_t sequence_width) {
    check_tensor(tx_id, "tx_id", at::kInt, 1);
    check_tensor(rx_id, "rx_id", at::kInt, 1);
    check_tensor(path_length, "path_length_m", at::kFloat, 1);
    check_tensor(delay, "delay_s", at::kFloat, 1);
    check_tensor(path_gain, "path_gain", at::kFloat, 1);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    TORCH_CHECK(sequence_width >= 0, "sequence_width must be non-negative");
    const int64_t count = tx_id.size(0);
    TORCH_CHECK(rx_id.size(0) == count, "rx_id must match tx_id");
    TORCH_CHECK(path_length.size(0) == count, "path_length_m must match tx_id");
    TORCH_CHECK(delay.size(0) == count, "delay_s must match tx_id");
    TORCH_CHECK(path_gain.size(0) == count, "path_gain must match tx_id");
    const int device = tx_id.get_device();
    TORCH_CHECK(rx_id.get_device() == device, "rx_id must share tx_id device");
    TORCH_CHECK(path_length.get_device() == device, "path_length_m must share tx_id device");
    TORCH_CHECK(delay.get_device() == device, "delay_s must share tx_id device");
    TORCH_CHECK(path_gain.get_device() == device, "path_gain must share tx_id device");
    if (count == 0) {
        return empty_deterministic_los_topology_block_from(tx_id, sequence_width);
    }

    auto bool_options = tx_id.options().dtype(at::kBool);
    auto int_options = tx_id.options().dtype(at::kInt);
    auto float_options = path_gain.options().dtype(at::kFloat);
    auto complex_options = path_gain.options().dtype(at::kComplexFloat);
    auto out_valid = at::empty({count}, bool_options);
    auto out_tx_id = at::empty({count}, int_options);
    auto out_rx_id = at::empty({count}, int_options);
    auto out_depth = at::empty({count}, int_options);
    auto out_component_id = at::empty({count}, int_options);
    auto out_primitive_id = at::empty({count}, int_options);
    auto out_edge_id = at::empty({count}, int_options);
    auto out_path_length = at::empty({count}, float_options);
    auto out_delay = at::empty({count}, float_options);
    auto out_path_gain = at::empty({count}, float_options);
    auto out_path_field = at::empty({count}, complex_options);
    auto out_interaction_position = at::empty({count, 3}, float_options);
    auto out_interaction_normal = at::empty({count, 3}, float_options);
    auto out_material_id = at::empty({count}, int_options);
    auto out_primitive_sequence = at::empty({count, sequence_width}, int_options);
    auto out_material_sequence = at::empty({count, sequence_width}, int_options);
    auto out_interaction_positions = at::empty({count, sequence_width, 3}, float_options);
    auto out_interaction_normals = at::empty({count, sequence_width, 3}, float_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    deterministic_los_topology_all_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                            kPathBlockSize, 0, stream>>>(
        count,
        tx_id.data_ptr<int>(),
        rx_id.data_ptr<int>(),
        path_length.data_ptr<float>(),
        delay.data_ptr<float>(),
        path_gain.data_ptr<float>(),
        static_cast<float>(frequency_hz),
        sequence_width,
        out_valid.data_ptr<bool>(),
        out_tx_id.data_ptr<int>(),
        out_rx_id.data_ptr<int>(),
        out_depth.data_ptr<int>(),
        out_component_id.data_ptr<int>(),
        out_primitive_id.data_ptr<int>(),
        out_edge_id.data_ptr<int>(),
        out_path_length.data_ptr<float>(),
        out_delay.data_ptr<float>(),
        out_path_gain.data_ptr<float>(),
        out_path_field.data_ptr<c10::complex<float>>(),
        out_interaction_position.data_ptr<float>(),
        out_interaction_normal.data_ptr<float>(),
        out_material_id.data_ptr<int>(),
        out_primitive_sequence.data_ptr<int>(),
        out_material_sequence.data_ptr<int>(),
        out_interaction_positions.data_ptr<float>(),
        out_interaction_normals.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    pybind11::dict out;
    out["valid"] = out_valid;
    out["tx_id"] = out_tx_id;
    out["rx_id"] = out_rx_id;
    out["depth"] = out_depth;
    out["component_id"] = out_component_id;
    out["primitive_id"] = out_primitive_id;
    out["edge_id"] = out_edge_id;
    out["path_length_m"] = out_path_length;
    out["delay_s"] = out_delay;
    out["path_gain"] = out_path_gain;
    out["path_field"] = out_path_field;
    out["interaction_position"] = out_interaction_position;
    out["interaction_normal"] = out_interaction_normal;
    out["material_id"] = out_material_id;
    out["primitive_sequence"] = out_primitive_sequence;
    out["material_sequence"] = out_material_sequence;
    out["interaction_positions"] = out_interaction_positions;
    out["interaction_normals"] = out_interaction_normals;
    return out;
}

pybind11::dict channel_deterministic_topology_default_fields(at::Tensor reference) {
    check_tensor(reference, "reference", at::kFloat, 1);
    const int64_t count = reference.size(0);
    auto float_options = reference.options().dtype(at::kFloat);
    auto int_options = reference.options().dtype(at::kInt);
    auto complex_options = reference.options().dtype(at::kComplexFloat);
    auto interaction_position = at::empty({count, 3}, float_options);
    auto interaction_normal = at::empty({count, 3}, float_options);
    auto material_id = at::empty({count}, int_options);
    auto path_field = at::empty({count}, complex_options);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        deterministic_topology_default_fields_kernel<<<
            ::channel::launch_blocks(count, kPathBlockSize), kPathBlockSize, 0, stream>>>(
            count,
            interaction_position.data_ptr<float>(),
            interaction_normal.data_ptr<float>(),
            material_id.data_ptr<int>(),
            path_field.data_ptr<c10::complex<float>>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["interaction_position"] = interaction_position;
    out["interaction_normal"] = interaction_normal;
    out["material_id"] = material_id;
    out["path_field"] = path_field;
    return out;
}

pybind11::dict channel_deterministic_pad_topology_sequences(
    at::Tensor depth,
    at::Tensor primitive_id,
    at::Tensor material_id,
    at::Tensor interaction_position,
    at::Tensor interaction_normal,
    at::Tensor primitive_sequence,
    at::Tensor material_sequence,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    int64_t width) {
    check_tensor(depth, "depth", at::kInt, 1);
    check_tensor(primitive_id, "primitive_id", at::kInt, 1);
    check_tensor(material_id, "material_id", at::kInt, 1);
    check_vec3_table(interaction_position, "interaction_position");
    check_vec3_table(interaction_normal, "interaction_normal");
    check_tensor(primitive_sequence, "primitive_sequence", at::kInt, 2);
    check_tensor(material_sequence, "material_sequence", at::kInt, 2);
    check_tensor(interaction_positions, "interaction_positions", at::kFloat, 3);
    check_tensor(interaction_normals, "interaction_normals", at::kFloat, 3);
    TORCH_CHECK(width >= 0, "width must be non-negative");
    const int64_t count = depth.size(0);
    const int device = depth.get_device();
    TORCH_CHECK(primitive_id.size(0) == count, "primitive_id must match depth");
    TORCH_CHECK(material_id.size(0) == count, "material_id must match depth");
    TORCH_CHECK(interaction_position.size(0) == count, "interaction_position must match depth");
    TORCH_CHECK(interaction_normal.size(0) == count, "interaction_normal must match depth");
    TORCH_CHECK(primitive_sequence.size(0) == count, "primitive_sequence must match depth");
    TORCH_CHECK(material_sequence.size(0) == count, "material_sequence must match depth");
    TORCH_CHECK(interaction_positions.size(0) == count, "interaction_positions must match depth");
    TORCH_CHECK(interaction_normals.size(0) == count, "interaction_normals must match depth");
    TORCH_CHECK(interaction_positions.size(2) == 3, "interaction_positions must have shape (N, D, 3)");
    TORCH_CHECK(interaction_normals.size(2) == 3, "interaction_normals must have shape (N, D, 3)");
    TORCH_CHECK(primitive_id.get_device() == device, "primitive_id must share depth device");
    TORCH_CHECK(material_id.get_device() == device, "material_id must share depth device");
    TORCH_CHECK(interaction_position.get_device() == device, "interaction_position must share depth device");
    TORCH_CHECK(interaction_normal.get_device() == device, "interaction_normal must share depth device");
    TORCH_CHECK(primitive_sequence.get_device() == device, "primitive_sequence must share depth device");
    TORCH_CHECK(material_sequence.get_device() == device, "material_sequence must share depth device");
    TORCH_CHECK(interaction_positions.get_device() == device, "interaction_positions must share depth device");
    TORCH_CHECK(interaction_normals.get_device() == device, "interaction_normals must share depth device");

    auto int_options = depth.options().dtype(at::kInt);
    auto float_options = interaction_position.options().dtype(at::kFloat);
    auto primitive_out = at::empty({count, width}, int_options);
    auto material_out = at::empty({count, width}, int_options);
    auto position_out = at::empty({count, width, 3}, float_options);
    auto normal_out = at::empty({count, width, 3}, float_options);
    const int64_t total = count * width;
    if (total > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        deterministic_pad_topology_sequences_kernel<<<
            ::channel::launch_blocks(total, kPathBlockSize), kPathBlockSize, 0, stream>>>(
            count,
            width,
            primitive_sequence.size(1),
            material_sequence.size(1),
            interaction_positions.size(1),
            interaction_normals.size(1),
            depth.data_ptr<int>(),
            primitive_id.data_ptr<int>(),
            material_id.data_ptr<int>(),
            interaction_position.data_ptr<float>(),
            interaction_normal.data_ptr<float>(),
            primitive_sequence.data_ptr<int>(),
            material_sequence.data_ptr<int>(),
            interaction_positions.data_ptr<float>(),
            interaction_normals.data_ptr<float>(),
            primitive_out.data_ptr<int>(),
            material_out.data_ptr<int>(),
            position_out.data_ptr<float>(),
            normal_out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict out;
    out["primitive_sequence"] = primitive_out;
    out["material_sequence"] = material_out;
    out["interaction_positions"] = position_out;
    out["interaction_normals"] = normal_out;
    return out;
}

at::Tensor channel_deterministic_face_anchor_points(at::Tensor vertices, at::Tensor faces) {
    check_vec3_table(vertices, "vertices");
    check_tensor(faces, "faces", at::kInt, 2);
    TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
    TORCH_CHECK(faces.get_device() == vertices.get_device(), "faces must share vertices device");
    const int64_t face_count = faces.size(0);
    auto out = at::empty({face_count, 3}, vertices.options().dtype(at::kFloat));
    if (face_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
        deterministic_face_anchor_points_kernel<<<
            ::channel::launch_blocks(face_count, kPathBlockSize), kPathBlockSize, 0, stream>>>(
            face_count,
            vertices.data_ptr<float>(),
            faces.data_ptr<int>(),
            out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

pybind11::dict channel_deterministic_reflection_epc_input_batch(
    at::Tensor tx_positions,
    at::Tensor rx_positions,
    at::Tensor sequences,
    at::Tensor sequence_tx,
    at::Tensor candidate_sequence,
    at::Tensor candidate_receiver,
    at::Tensor tri_a,
    at::Tensor normals,
    int64_t candidate_start,
    int64_t candidate_end) {
    check_vec3_table(tx_positions, "tx_positions");
    check_vec3_table(rx_positions, "rx_positions");
    check_vec3_table(tri_a, "tri_a");
    check_vec3_table(normals, "normals");
    TORCH_CHECK(normals.sizes() == tri_a.sizes(), "normals must match tri_a");
    TORCH_CHECK(sequences.is_cuda(), "sequences must be a CUDA tensor");
    TORCH_CHECK(sequences.is_contiguous(), "sequences must be contiguous");
    TORCH_CHECK(sequences.dim() == 2, "sequences must have shape (S, depth)");
    TORCH_CHECK(
        sequences.scalar_type() == at::kLong || sequences.scalar_type() == at::kInt,
        "sequences must be int64 or int32");
    check_tensor(sequence_tx, "sequence_tx", at::kInt, 1);
    const bool shared = sequence_tx.size(0) == 0;
    TORCH_CHECK(
        shared || sequence_tx.size(0) == sequences.size(0),
        "sequence_tx must be empty or match sequences");
    check_tensor(candidate_sequence, "candidate_sequence", at::kInt, 1);
    check_tensor(candidate_receiver, "candidate_receiver", at::kInt, 1);
    const bool listed = candidate_sequence.size(0) > 0;
    TORCH_CHECK(
        candidate_receiver.size(0) == candidate_sequence.size(0),
        "candidate_receiver must match candidate_sequence");
    TORCH_CHECK(!listed || !shared, "explicit candidates require per-transmitter sequences");
    TORCH_CHECK(candidate_sequence.get_device() == tx_positions.get_device() &&
                    candidate_receiver.get_device() == tx_positions.get_device(),
                "candidate lists must share tx device");
    const int device = tx_positions.get_device();
    TORCH_CHECK(rx_positions.get_device() == device, "rx_positions must share tx device");
    TORCH_CHECK(sequences.get_device() == device, "sequences must share tx device");
    TORCH_CHECK(sequence_tx.get_device() == device, "sequence_tx must share tx device");
    TORCH_CHECK(tri_a.get_device() == device, "tri_a must share tx device");
    TORCH_CHECK(normals.get_device() == device, "normals must share tx device");
    const int64_t tx_count = tx_positions.size(0);
    const int64_t rx_count = rx_positions.size(0);
    const int64_t sequence_count = sequences.size(0);
    const int64_t depth = sequences.size(1);
    const int64_t total = listed ? candidate_sequence.size(0)
        : (shared ? tx_count * rx_count * sequence_count : sequence_count * rx_count);
    TORCH_CHECK(candidate_start >= 0, "candidate_start must be non-negative");
    TORCH_CHECK(candidate_end >= candidate_start, "candidate_end must not precede candidate_start");
    TORCH_CHECK(candidate_end <= total, "candidate_end is out of range");
    const int64_t pair_count = candidate_end - candidate_start;
    auto float_options = tx_positions.options().dtype(at::kFloat);
    auto int_options = tx_positions.options().dtype(at::kInt);

    pybind11::dict out;
    out["tx_batch"] = at::empty({pair_count, 3}, float_options);
    out["rx_batch"] = at::empty({pair_count, 3}, float_options);
    out["tx_indices"] = at::empty({pair_count}, int_options);
    out["rx_indices"] = at::empty({pair_count}, int_options);
    out["sequence_batch"] = at::empty({pair_count, depth}, int_options);
    out["direct_plane_points"] = at::empty({pair_count, depth, 3}, float_options);
    out["direct_plane_normals"] = at::empty({pair_count, depth, 3}, float_options);

    if (pair_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        auto launch = [&](auto* sequence_data) {
            deterministic_reflection_epc_input_batch_kernel<<<
                ::channel::launch_blocks(pair_count, kPathBlockSize),
                kPathBlockSize,
                0,
                stream>>>(
                pair_count,
                candidate_start,
                sequence_count,
                depth,
                rx_count,
                shared ? nullptr : sequence_tx.data_ptr<int>(),
                listed ? candidate_sequence.data_ptr<int>() : nullptr,
                listed ? candidate_receiver.data_ptr<int>() : nullptr,
                tx_positions.data_ptr<float>(),
                rx_positions.data_ptr<float>(),
                sequence_data,
                tri_a.data_ptr<float>(),
                normals.data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["tx_batch"]).data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["rx_batch"]).data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["tx_indices"]).data_ptr<int>(),
                pybind11::cast<at::Tensor>(out["rx_indices"]).data_ptr<int>(),
                pybind11::cast<at::Tensor>(out["sequence_batch"]).data_ptr<int>(),
                pybind11::cast<at::Tensor>(out["direct_plane_points"]).data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["direct_plane_normals"]).data_ptr<float>());
        };
        if (sequences.scalar_type() == at::kLong) {
            launch(sequences.data_ptr<int64_t>());
        } else {
            launch(sequences.data_ptr<int>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor channel_deterministic_normalize_vec3(at::Tensor values, double eps) {
    check_vec3_table(values, "values");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
    const int64_t count = values.size(0);
    auto out = at::empty_like(values);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(values.get_device()).stream();
        deterministic_normalize_vec3_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                              kPathBlockSize, 0, stream>>>(
            count,
            values.data_ptr<float>(),
            static_cast<float>(eps),
            out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
channel_path_reflection_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor face_gain,
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz) {
    check_vec3_table(vertices, "vertices");
    check_tensor(faces, "faces", at::kInt, 2);
    check_vec3_table(face_normals, "face_normals");
    check_tensor(face_gain, "face_gain", at::kFloat, 1);
    check_vec3_table(tx_positions, "tx_positions");
    check_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_vec3_table(rx_positions, "rx_positions");
    TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
    TORCH_CHECK(face_normals.size(0) == faces.size(0), "face_normals must match faces");
    TORCH_CHECK(face_gain.size(0) == faces.size(0), "face_gain must match faces");
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    const int64_t tx_count = tx_positions.size(0);
    const int64_t rx_count = rx_positions.size(0);
    const int64_t face_count = faces.size(0);
    const int64_t count = tx_count * rx_count * face_count;
    auto bool_options = tx_positions.options().dtype(at::kBool);
    auto int_options = tx_positions.options().dtype(at::kInt);
    auto float_options = tx_positions.options().dtype(at::kFloat);
    auto valid = at::empty({count}, bool_options);
    auto tx_id = at::empty({count}, int_options);
    auto rx_id = at::empty({count}, int_options);
    auto depth = at::empty({count}, int_options);
    auto component_id = at::empty({count}, int_options);
    auto primitive_id = at::empty({count}, int_options);
    auto edge_id = at::empty({count}, int_options);
    auto path_length = at::empty({count}, float_options);
    auto delay = at::empty({count}, float_options);
    auto path_gain = at::empty({count}, float_options);
    auto seg0_start = at::empty({count, 3}, float_options);
    auto seg0_end = at::empty({count, 3}, float_options);
    auto seg1_start = at::empty({count, 3}, float_options);
    auto seg1_end = at::empty({count, 3}, float_options);
    auto active = at::empty({count}, bool_options);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        path_reflection_candidates_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                            kPathBlockSize, 0, stream>>>(
            count,
            face_count,
            rx_count,
            vertices.data_ptr<float>(),
            faces.data_ptr<int>(),
            face_normals.data_ptr<float>(),
            face_gain.data_ptr<float>(),
            tx_positions.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            static_cast<float>(kSpeedOfLight / frequency_hz),
            valid.data_ptr<bool>(),
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            depth.data_ptr<int>(),
            component_id.data_ptr<int>(),
            primitive_id.data_ptr<int>(),
            edge_id.data_ptr<int>(),
            path_length.data_ptr<float>(),
            delay.data_ptr<float>(),
            path_gain.data_ptr<float>(),
            seg0_start.data_ptr<float>(),
            seg0_end.data_ptr<float>(),
            seg1_start.data_ptr<float>(),
            seg1_end.data_ptr<float>(),
            active.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        valid,
        tx_id,
        rx_id,
        depth,
        component_id,
        primitive_id,
        edge_id,
        path_length,
        delay,
        path_gain,
        seg0_start,
        seg0_end,
        seg1_start,
        seg1_end,
        active};
}

// ==== Section: Path compaction ====
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include "torch_cuda.h"
#include "math.cuh"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include "path_compaction.cuh"

#define channel_neg_kd_phase channel_neg_kd_phase_compaction

namespace {

using channel::kPiF;
using channel::kSpeedOfLight;

__device__ float channel_neg_kd_phase(float k, float d) {
    // Reduce k*d mod 2*pi in double: the f32 product loses ~k*d*2^-24 of
    // phase, which shifts coherent nulls at mmWave ranges.
    const double kd = fmod(static_cast<double>(k) * static_cast<double>(d), channel::kTwoPi);
    return -static_cast<float>(kd);
}

__global__ void path_visibility_flags_kernel(
    int64_t count,
    const bool* __restrict__ visible,
    int* __restrict__ flags) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        flags[index] = visible[index] ? 1 : 0;
    }
}

__global__ void deterministic_los_topology_compact_kernel(
    int64_t count,
    const int* __restrict__ flags,
    const int* __restrict__ offsets,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    const float* __restrict__ path_length,
    const float* __restrict__ delay,
    const float* __restrict__ path_gain,
    float frequency_hz,
    int64_t sequence_width,
    bool* __restrict__ out_valid,
    int* __restrict__ out_tx_id,
    int* __restrict__ out_rx_id,
    int* __restrict__ out_depth,
    int* __restrict__ out_component_id,
    int* __restrict__ out_primitive_id,
    int* __restrict__ out_edge_id,
    float* __restrict__ out_path_length,
    float* __restrict__ out_delay,
    float* __restrict__ out_path_gain,
    c10::complex<float>* __restrict__ out_path_field,
    float* __restrict__ out_interaction_position,
    float* __restrict__ out_interaction_normal,
    int* __restrict__ out_material_id,
    int* __restrict__ out_primitive_sequence,
    int* __restrict__ out_material_sequence,
    float* __restrict__ out_interaction_positions,
    float* __restrict__ out_interaction_normals) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        if (flags[index] == 0) {
            continue;
        }
        const int dst = offsets[index];
        const float length = path_length[index];
        const float gain = path_gain[index];
        const float amplitude = sqrtf(fmaxf(gain, 0.0f));
        const float wavelength = static_cast<float>(kSpeedOfLight) / frequency_hz;
        const float phase = channel_neg_kd_phase(2.0f * kPiF / wavelength, length);

        out_valid[dst] = true;
        out_tx_id[dst] = tx_id[index];
        out_rx_id[dst] = rx_id[index];
        out_depth[dst] = 0;
        out_component_id[dst] = 0;
        out_primitive_id[dst] = -1;
        out_edge_id[dst] = -1;
        out_path_length[dst] = length;
        out_delay[dst] = delay[index];
        out_path_gain[dst] = gain;
        out_path_field[dst] = c10::complex<float>(amplitude * cosf(phase), amplitude * sinf(phase));
        out_material_id[dst] = -1;

        const int64_t vec_base = static_cast<int64_t>(dst) * 3;
        for (int axis = 0; axis < 3; ++axis) {
            out_interaction_position[vec_base + axis] = 0.0f;
            out_interaction_normal[vec_base + axis] = 0.0f;
        }
        for (int64_t column = 0; column < sequence_width; ++column) {
            const int64_t seq_index = static_cast<int64_t>(dst) * sequence_width + column;
            out_primitive_sequence[seq_index] = -1;
            out_material_sequence[seq_index] = -1;
            const int64_t seq_vec_base = seq_index * 3;
            for (int axis = 0; axis < 3; ++axis) {
                out_interaction_positions[seq_vec_base + axis] = 0.0f;
                out_interaction_normals[seq_vec_base + axis] = 0.0f;
            }
        }
    }
}

// Writes one visible reflection EPC candidate as a complete topology row. The
// field columns stay zero: the field stage evaluates every reflection row.
template <typename sequence_t>
__global__ void deterministic_reflection_topology_compact_kernel(
    int64_t count,
    int64_t depth,
    const int* __restrict__ flags,
    const int* __restrict__ offsets,
    const sequence_t* __restrict__ epc_sequences,
    const float* __restrict__ epc_hits,
    const float* __restrict__ epc_normals,
    const float* __restrict__ epc_path_length,
    const int* __restrict__ tx_indices,
    const int* __restrict__ rx_indices,
    const int* __restrict__ face_material_id,
    bool* __restrict__ out_valid,
    int* __restrict__ out_tx_id,
    int* __restrict__ out_rx_id,
    int* __restrict__ out_depth,
    int* __restrict__ out_component_id,
    int* __restrict__ out_primitive_id,
    int* __restrict__ out_edge_id,
    float* __restrict__ out_path_length,
    float* __restrict__ out_delay,
    float* __restrict__ out_path_gain,
    c10::complex<float>* __restrict__ out_path_field,
    c10::complex<float>* __restrict__ out_field_xyz,
    float* __restrict__ out_interaction_position,
    float* __restrict__ out_interaction_normal,
    int* __restrict__ out_material_id,
    int* __restrict__ out_primitive_sequence,
    int* __restrict__ out_material_sequence,
    float* __restrict__ out_interaction_positions,
    float* __restrict__ out_interaction_normals) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        if (flags[index] == 0) {
            continue;
        }
        const int64_t dst = offsets[index];
        const float length = epc_path_length[index];
        out_valid[dst] = true;
        out_tx_id[dst] = tx_indices[index];
        out_rx_id[dst] = rx_indices[index];
        out_depth[dst] = static_cast<int>(depth);
        out_component_id[dst] = 1;
        out_edge_id[dst] = -1;
        out_path_length[dst] = length;
        out_delay[dst] = length / channel::kSpeedOfLightF;
        out_path_gain[dst] = 0.0f;
        out_path_field[dst] = c10::complex<float>(0.0f, 0.0f);
        for (int axis = 0; axis < 3; ++axis) {
            out_field_xyz[dst * 3 + axis] = c10::complex<float>(0.0f, 0.0f);
        }
        for (int64_t column = 0; column < depth; ++column) {
            const int64_t src_slot = index * depth + column;
            const int64_t dst_slot = dst * depth + column;
            const int face = static_cast<int>(epc_sequences[src_slot]);
            const int material = face_material_id[face];
            out_primitive_sequence[dst_slot] = face;
            out_material_sequence[dst_slot] = material;
            for (int axis = 0; axis < 3; ++axis) {
                const float hit = epc_hits[src_slot * 3 + axis];
                const float normal = epc_normals[src_slot * 3 + axis];
                out_interaction_positions[dst_slot * 3 + axis] = hit;
                out_interaction_normals[dst_slot * 3 + axis] = normal;
                if (column == 0) {
                    out_interaction_position[dst * 3 + axis] = hit;
                    out_interaction_normal[dst * 3 + axis] = normal;
                }
            }
            if (column == 0) {
                out_primitive_id[dst] = face;
                out_material_id[dst] = material;
            }
        }
    }
}

template <typename face_id_t>
__global__ void deterministic_reflection_order1_compact_kernel(
    int64_t count,
    const int* __restrict__ flags,
    const int* __restrict__ offsets,
    bool grouped_export,
    const face_id_t* __restrict__ epc_faces,
    int64_t epc_face_width,
    const float* __restrict__ epc_hits,
    const float* __restrict__ epc_normals,
    int64_t epc_hit_width,
    const int* __restrict__ sequence_batch,
    int64_t sequence_width,
    const int* __restrict__ rx_indices,
    const float* __restrict__ tx,
    const float* __restrict__ rx_positions,
    const float* __restrict__ tx_power,
    int tx_index,
    const float* __restrict__ face_eps_r,
    const float* __restrict__ face_sigma_e,
    const float* __restrict__ face_mu_r,
    const float* __restrict__ face_gain,
    const int* __restrict__ face_material_id,
    int* __restrict__ selected_faces,
    float* __restrict__ selected_points,
    float* __restrict__ selected_normals,
    int* __restrict__ selected_rx_id,
    float* __restrict__ tx_keep,
    float* __restrict__ rx_keep,
    float* __restrict__ selected_tx_power,
    float* __restrict__ selected_eps_r,
    float* __restrict__ selected_sigma_e,
    float* __restrict__ selected_mu_r,
    float* __restrict__ selected_gain,
    int* __restrict__ selected_material_id) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        if (flags[index] == 0) {
            continue;
        }
        const int dst = offsets[index];
        const int face = grouped_export
            ? static_cast<int>(epc_faces[index * epc_face_width])
            : sequence_batch[index * sequence_width];
        const int rx_id = rx_indices[index];
        selected_faces[dst] = face;
        selected_rx_id[dst] = rx_id;
        selected_tx_power[dst] = tx_power[tx_index];
        selected_eps_r[dst] = face_eps_r[face];
        selected_sigma_e[dst] = face_sigma_e[face];
        selected_mu_r[dst] = face_mu_r[face];
        selected_gain[dst] = face_gain[face];
        selected_material_id[dst] = face_material_id[face];
        const int64_t hit_base = index * epc_hit_width * 3;
        for (int axis = 0; axis < 3; ++axis) {
            selected_points[dst * 3 + axis] = epc_hits[hit_base + axis];
            selected_normals[dst * 3 + axis] = epc_normals[hit_base + axis];
            tx_keep[dst * 3 + axis] = tx[axis];
            rx_keep[dst * 3 + axis] = rx_positions[static_cast<int64_t>(rx_id) * 3 + axis];
        }
    }
}

template <typename sequence_t>
__global__ void deterministic_reflection_sequence_compact_kernel(
    int64_t count,
    int64_t depth,
    int64_t out_count,
    const int* __restrict__ flags,
    const int* __restrict__ offsets,
    const sequence_t* __restrict__ epc_sequences,
    const float* __restrict__ epc_hits,
    const float* __restrict__ epc_normals,
    const int* __restrict__ rx_indices,
    const float* __restrict__ tx,
    const float* __restrict__ rx_positions,
    const float* __restrict__ tx_power,
    int tx_index,
    const float* __restrict__ face_eps_r,
    const float* __restrict__ face_sigma_e,
    const float* __restrict__ face_mu_r,
    const float* __restrict__ face_gain,
    const int* __restrict__ face_material_id,
    int* __restrict__ selected_sequences,
    float* __restrict__ selected_hits,
    float* __restrict__ selected_normals,
    int* __restrict__ selected_rx_id,
    float* __restrict__ selected_tx,
    float* __restrict__ selected_rx,
    float* __restrict__ selected_tx_power,
    float* __restrict__ selected_eps_r,
    float* __restrict__ selected_sigma_e,
    float* __restrict__ selected_mu_r,
    float* __restrict__ selected_gain,
    int* __restrict__ first_face,
    int* __restrict__ material_id,
    int* __restrict__ material_sequence,
    float* __restrict__ first_hit,
    float* __restrict__ first_normal) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        if (flags[index] == 0) {
            continue;
        }
        const int dst = offsets[index];
        if (dst >= out_count) {
            continue;
        }
        const int rx_id = rx_indices[index];
        selected_rx_id[dst] = rx_id;
        selected_tx_power[dst] = tx_power[tx_index];
        for (int axis = 0; axis < 3; ++axis) {
            selected_tx[dst * 3 + axis] = tx[axis];
            selected_rx[dst * 3 + axis] = rx_positions[static_cast<int64_t>(rx_id) * 3 + axis];
        }
        for (int64_t column = 0; column < depth; ++column) {
            const int64_t seq_index = static_cast<int64_t>(dst) * depth + column;
            const int64_t src_seq_index = index * depth + column;
            const int face = static_cast<int>(epc_sequences[src_seq_index]);
            selected_sequences[seq_index] = face;
            selected_eps_r[seq_index] = face_eps_r[face];
            selected_sigma_e[seq_index] = face_sigma_e[face];
            selected_mu_r[seq_index] = face_mu_r[face];
            selected_gain[seq_index] = face_gain[face];
            material_sequence[seq_index] = face_material_id[face];
            const int64_t vec_base = seq_index * 3;
            const int64_t src_vec_base = src_seq_index * 3;
            for (int axis = 0; axis < 3; ++axis) {
                const float hit_value = epc_hits[src_vec_base + axis];
                const float normal_value = epc_normals[src_vec_base + axis];
                selected_hits[vec_base + axis] = hit_value;
                selected_normals[vec_base + axis] = normal_value;
                if (column == 0) {
                    first_hit[dst * 3 + axis] = hit_value;
                    first_normal[dst * 3 + axis] = normal_value;
                }
            }
        }
        const int first = selected_sequences[static_cast<int64_t>(dst) * depth];
        first_face[dst] = first;
        material_id[dst] = face_material_id[first];
    }
}

// Writes one valid order-1 diffraction export row as a topology row: the exporter's UTD vector
// field becomes field_xyz, the scalar field columns stay zero for the field stage.
__global__ void deterministic_diffraction_topology_compact_kernel(
    int64_t count,
    int rx_offset,
    const int* __restrict__ flags,
    const int* __restrict__ offsets,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    const int* __restrict__ depth,
    const int* __restrict__ edge_id,
    const float* __restrict__ delay,
    const float* __restrict__ x_re,
    const float* __restrict__ x_im,
    const float* __restrict__ y_re,
    const float* __restrict__ y_im,
    const float* __restrict__ z_re,
    const float* __restrict__ z_im,
    const float* __restrict__ interaction_position,
    bool* __restrict__ out_valid,
    int* __restrict__ out_tx_id,
    int* __restrict__ out_rx_id,
    int* __restrict__ out_depth,
    int* __restrict__ out_component_id,
    int* __restrict__ out_primitive_id,
    int* __restrict__ out_edge_id,
    float* __restrict__ out_path_length,
    float* __restrict__ out_delay,
    float* __restrict__ out_path_gain,
    c10::complex<float>* __restrict__ out_path_field,
    c10::complex<float>* __restrict__ out_field_xyz,
    float* __restrict__ out_interaction_position,
    float* __restrict__ out_interaction_normal,
    int* __restrict__ out_material_id) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        if (flags[index] == 0) {
            continue;
        }
        const int64_t dst = offsets[index];
        const float row_delay = delay[index];
        out_valid[dst] = true;
        out_tx_id[dst] = tx_id[index];
        out_rx_id[dst] = rx_id[index] + rx_offset;
        out_depth[dst] = depth[index];
        out_component_id[dst] = 2;
        out_primitive_id[dst] = -1;
        out_edge_id[dst] = edge_id[index];
        out_path_length[dst] = row_delay * channel::kSpeedOfLightF;
        out_delay[dst] = row_delay;
        out_path_gain[dst] = 0.0f;
        out_path_field[dst] = c10::complex<float>(0.0f, 0.0f);
        out_field_xyz[dst * 3 + 0] = c10::complex<float>(x_re[index], x_im[index]);
        out_field_xyz[dst * 3 + 1] = c10::complex<float>(y_re[index], y_im[index]);
        out_field_xyz[dst * 3 + 2] = c10::complex<float>(z_re[index], z_im[index]);
        out_material_id[dst] = -1;
        for (int axis = 0; axis < 3; ++axis) {
            out_interaction_position[dst * 3 + axis] = interaction_position[index * 3 + axis];
            out_interaction_normal[dst * 3 + axis] = 0.0f;
        }
    }
}

}  // namespace

pybind11::dict channel_deterministic_los_topology_block(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor path_length,
    at::Tensor delay,
    at::Tensor path_gain,
    at::Tensor visible,
    double frequency_hz,
    int64_t sequence_width) {
    check_tensor(tx_id, "tx_id", at::kInt, 1);
    check_tensor(rx_id, "rx_id", at::kInt, 1);
    check_tensor(path_length, "path_length_m", at::kFloat, 1);
    check_tensor(delay, "delay_s", at::kFloat, 1);
    check_tensor(path_gain, "path_gain", at::kFloat, 1);
    check_tensor(visible, "visible", at::kBool, 1);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    TORCH_CHECK(sequence_width >= 0, "sequence_width must be non-negative");
    const int64_t count = tx_id.size(0);
    TORCH_CHECK(rx_id.size(0) == count, "rx_id must match tx_id");
    TORCH_CHECK(path_length.size(0) == count, "path_length_m must match tx_id");
    TORCH_CHECK(delay.size(0) == count, "delay_s must match tx_id");
    TORCH_CHECK(path_gain.size(0) == count, "path_gain must match tx_id");
    TORCH_CHECK(visible.size(0) == count, "visible must match tx_id");
    const int device = tx_id.get_device();
    TORCH_CHECK(rx_id.get_device() == device, "rx_id must share tx_id device");
    TORCH_CHECK(path_length.get_device() == device, "path_length_m must share tx_id device");
    TORCH_CHECK(delay.get_device() == device, "delay_s must share tx_id device");
    TORCH_CHECK(path_gain.get_device() == device, "path_gain must share tx_id device");
    TORCH_CHECK(visible.get_device() == device, "visible must share tx_id device");
    if (count == 0) {
        return empty_deterministic_los_topology_block_from(tx_id, sequence_width);
    }

    auto int_options = tx_id.options().dtype(at::kInt);
    auto flags = at::empty({count}, int_options);
    auto offsets = at::empty({count + 1}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    path_visibility_flags_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                   kPathBlockSize, 0, stream>>>(
        count,
        visible.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    channel::scan::exclusive_flag_offsets(flags, offsets, count, stream);

    const int64_t out_count =
        observe_compact_count(offsets, count, stream).count;
    if (out_count == 0) {
        return empty_deterministic_los_topology_block_from(tx_id, sequence_width);
    }

    auto bool_options = visible.options().dtype(at::kBool);
    auto float_options = path_gain.options().dtype(at::kFloat);
    auto complex_options = path_gain.options().dtype(at::kComplexFloat);
    auto out_valid = at::empty({out_count}, bool_options);
    auto out_tx_id = at::empty({out_count}, int_options);
    auto out_rx_id = at::empty({out_count}, int_options);
    auto out_depth = at::empty({out_count}, int_options);
    auto out_component_id = at::empty({out_count}, int_options);
    auto out_primitive_id = at::empty({out_count}, int_options);
    auto out_edge_id = at::empty({out_count}, int_options);
    auto out_path_length = at::empty({out_count}, float_options);
    auto out_delay = at::empty({out_count}, float_options);
    auto out_path_gain = at::empty({out_count}, float_options);
    auto out_path_field = at::empty({out_count}, complex_options);
    auto out_interaction_position = at::empty({out_count, 3}, float_options);
    auto out_interaction_normal = at::empty({out_count, 3}, float_options);
    auto out_material_id = at::empty({out_count}, int_options);
    auto out_primitive_sequence = at::empty({out_count, sequence_width}, int_options);
    auto out_material_sequence = at::empty({out_count, sequence_width}, int_options);
    auto out_interaction_positions = at::empty({out_count, sequence_width, 3}, float_options);
    auto out_interaction_normals = at::empty({out_count, sequence_width, 3}, float_options);
    deterministic_los_topology_compact_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                                kPathBlockSize, 0, stream>>>(
        count,
        flags.data_ptr<int>(),
        offsets.data_ptr<int>(),
        tx_id.data_ptr<int>(),
        rx_id.data_ptr<int>(),
        path_length.data_ptr<float>(),
        delay.data_ptr<float>(),
        path_gain.data_ptr<float>(),
        static_cast<float>(frequency_hz),
        sequence_width,
        out_valid.data_ptr<bool>(),
        out_tx_id.data_ptr<int>(),
        out_rx_id.data_ptr<int>(),
        out_depth.data_ptr<int>(),
        out_component_id.data_ptr<int>(),
        out_primitive_id.data_ptr<int>(),
        out_edge_id.data_ptr<int>(),
        out_path_length.data_ptr<float>(),
        out_delay.data_ptr<float>(),
        out_path_gain.data_ptr<float>(),
        out_path_field.data_ptr<c10::complex<float>>(),
        out_interaction_position.data_ptr<float>(),
        out_interaction_normal.data_ptr<float>(),
        out_material_id.data_ptr<int>(),
        out_primitive_sequence.data_ptr<int>(),
        out_material_sequence.data_ptr<int>(),
        out_interaction_positions.data_ptr<float>(),
        out_interaction_normals.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    pybind11::dict out;
    out["valid"] = out_valid;
    out["tx_id"] = out_tx_id;
    out["rx_id"] = out_rx_id;
    out["depth"] = out_depth;
    out["component_id"] = out_component_id;
    out["primitive_id"] = out_primitive_id;
    out["edge_id"] = out_edge_id;
    out["path_length_m"] = out_path_length;
    out["delay_s"] = out_delay;
    out["path_gain"] = out_path_gain;
    out["path_field"] = out_path_field;
    out["interaction_position"] = out_interaction_position;
    out["interaction_normal"] = out_interaction_normal;
    out["material_id"] = out_material_id;
    out["primitive_sequence"] = out_primitive_sequence;
    out["material_sequence"] = out_material_sequence;
    out["interaction_positions"] = out_interaction_positions;
    out["interaction_normals"] = out_interaction_normals;
    return out;
}

static void check_reflection_compact_tensors(
    const at::Tensor& rx_indices, const at::Tensor& tx, const at::Tensor& rx_positions,
    const at::Tensor& tx_power, const at::Tensor& face_eps_r, const at::Tensor& face_sigma_e,
    const at::Tensor& face_mu_r, const at::Tensor& face_gain, const at::Tensor& face_material_id) {
    check_tensor(rx_indices, "rx_indices", at::kInt, 1);
    check_tensor(tx, "tx", at::kFloat, 1);
    TORCH_CHECK(tx.size(0) == 3, "tx must have shape (3,)");
    check_vec3_table(rx_positions, "rx_positions");
    check_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_tensor(face_eps_r, "face_eps_r", at::kFloat, 1);
    check_tensor(face_sigma_e, "face_sigma_e", at::kFloat, 1);
    check_tensor(face_mu_r, "face_mu_r", at::kFloat, 1);
    check_tensor(face_gain, "face_gain", at::kFloat, 1);
    check_tensor(face_material_id, "face_material_id", at::kInt, 1);
}

static void check_reflection_compact_shapes(
    const at::Tensor& rx_indices, const at::Tensor& tx_power, int64_t tx_index,
    const at::Tensor& face_eps_r, const at::Tensor& face_sigma_e, const at::Tensor& face_mu_r,
    const at::Tensor& face_gain, const at::Tensor& face_material_id, int64_t count) {
    TORCH_CHECK(rx_indices.size(0) == count, "rx_indices must match visible");
    TORCH_CHECK(tx_index >= 0 && tx_index < tx_power.size(0), "tx_index is out of range");
    TORCH_CHECK(face_sigma_e.size(0) == face_eps_r.size(0), "face_sigma_e must match face_eps_r");
    TORCH_CHECK(face_mu_r.size(0) == face_eps_r.size(0), "face_mu_r must match face_eps_r");
    TORCH_CHECK(face_gain.size(0) == face_eps_r.size(0), "face_gain must match face_eps_r");
    TORCH_CHECK(face_material_id.size(0) == face_eps_r.size(0),
                "face_material_id must match face_eps_r");
}

static void
check_reflection_compact_device(const at::Tensor& rx_indices, const at::Tensor& tx,
                                const at::Tensor& rx_positions, const at::Tensor& tx_power,
                                const at::Tensor& face_eps_r, const at::Tensor& face_sigma_e,
                                const at::Tensor& face_mu_r, const at::Tensor& face_gain,
                                const at::Tensor& face_material_id, int device) {
    TORCH_CHECK(rx_indices.get_device() == device, "rx_indices must share visible device");
    TORCH_CHECK(tx.get_device() == device, "tx must share visible device");
    TORCH_CHECK(rx_positions.get_device() == device, "rx_positions must share visible device");
    TORCH_CHECK(tx_power.get_device() == device, "tx_power must share visible device");
    TORCH_CHECK(face_eps_r.get_device() == device, "face_eps_r must share visible device");
    TORCH_CHECK(face_sigma_e.get_device() == device, "face_sigma_e must share visible device");
    TORCH_CHECK(face_mu_r.get_device() == device, "face_mu_r must share visible device");
    TORCH_CHECK(face_gain.get_device() == device, "face_gain must share visible device");
    TORCH_CHECK(face_material_id.get_device() == device,
                "face_material_id must share visible device");
}

pybind11::dict channel_deterministic_reflection_topology_compact(
    at::Tensor visible, at::Tensor epc_sequences, at::Tensor epc_hits, at::Tensor epc_normals,
    at::Tensor epc_path_length, at::Tensor tx_indices, at::Tensor rx_indices,
    at::Tensor face_material_id) {
    check_tensor(visible, "visible", at::kBool, 1);
    TORCH_CHECK(epc_sequences.is_cuda(), "epc_sequences must be a CUDA tensor");
    TORCH_CHECK(epc_sequences.is_contiguous(), "epc_sequences must be contiguous");
    TORCH_CHECK(epc_sequences.dim() == 2, "epc_sequences must have shape (N, depth)");
    TORCH_CHECK(epc_sequences.scalar_type() == at::kInt || epc_sequences.scalar_type() == at::kLong,
                "epc_sequences must be int32 or int64");
    check_tensor(epc_hits, "epc_hits", at::kFloat, 3);
    check_tensor(epc_normals, "epc_normals", at::kFloat, 3);
    check_tensor(epc_path_length, "epc_path_length", at::kFloat, 1);
    check_tensor(tx_indices, "tx_indices", at::kInt, 1);
    check_tensor(rx_indices, "rx_indices", at::kInt, 1);
    check_tensor(face_material_id, "face_material_id", at::kInt, 1);
    const int64_t count = visible.size(0);
    const int64_t depth = epc_sequences.size(1);
    TORCH_CHECK(depth > 0, "epc_sequences must have positive depth");
    TORCH_CHECK(epc_sequences.size(0) == count, "epc_sequences must match visible");
    TORCH_CHECK(epc_hits.size(0) == count && epc_hits.size(1) == depth && epc_hits.size(2) == 3,
                "epc_hits must have shape (N, depth, 3)");
    TORCH_CHECK(epc_normals.sizes() == epc_hits.sizes(), "epc_normals must match epc_hits");
    TORCH_CHECK(epc_path_length.size(0) == count, "epc_path_length must match visible");
    TORCH_CHECK(tx_indices.size(0) == count, "tx_indices must match visible");
    TORCH_CHECK(rx_indices.size(0) == count, "rx_indices must match visible");
    const int device = visible.get_device();
    for (const at::Tensor* tensor : {&epc_sequences, &epc_hits, &epc_normals, &epc_path_length,
                                     &tx_indices, &rx_indices, &face_material_id}) {
        TORCH_CHECK(tensor->get_device() == device, "reflection compact inputs must share one device");
    }

    auto bool_options = visible.options().dtype(at::kBool);
    auto int_options = visible.options().dtype(at::kInt);
    auto float_options = visible.options().dtype(at::kFloat);
    auto complex_options = visible.options().dtype(at::kComplexFloat);
    auto allocate = [&](int64_t rows) {
        pybind11::dict out;
        out["valid"] = at::empty({rows}, bool_options);
        out["tx_id"] = at::empty({rows}, int_options);
        out["rx_id"] = at::empty({rows}, int_options);
        out["depth"] = at::empty({rows}, int_options);
        out["component_id"] = at::empty({rows}, int_options);
        out["primitive_id"] = at::empty({rows}, int_options);
        out["edge_id"] = at::empty({rows}, int_options);
        out["path_length_m"] = at::empty({rows}, float_options);
        out["delay_s"] = at::empty({rows}, float_options);
        out["path_gain"] = at::empty({rows}, float_options);
        out["path_field"] = at::empty({rows}, complex_options);
        out["field_xyz"] = at::empty({rows, 3}, complex_options);
        out["interaction_position"] = at::empty({rows, 3}, float_options);
        out["interaction_normal"] = at::empty({rows, 3}, float_options);
        out["material_id"] = at::empty({rows}, int_options);
        out["primitive_sequence"] = at::empty({rows, depth}, int_options);
        out["material_sequence"] = at::empty({rows, depth}, int_options);
        out["interaction_positions"] = at::empty({rows, depth, 3}, float_options);
        out["interaction_normals"] = at::empty({rows, depth, 3}, float_options);
        return out;
    };
    if (count == 0) {
        return allocate(0);
    }

    auto flags = at::empty({count}, int_options);
    auto offsets = at::empty({count + 1}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    path_visibility_flags_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                   kPathBlockSize, 0, stream>>>(
        count,
        visible.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    channel::scan::exclusive_flag_offsets(flags, offsets, count, stream);
    const int64_t out_count = observe_compact_count(offsets, count, stream).count;
    pybind11::dict out = allocate(out_count);
    if (out_count == 0) {
        return out;
    }
    auto tensor = [&](const char* name) { return pybind11::cast<at::Tensor>(out[name]); };
    auto launch = [&](auto* sequence_data) {
        deterministic_reflection_topology_compact_kernel<<<
            ::channel::launch_blocks(count, kPathBlockSize), kPathBlockSize, 0, stream>>>(
            count,
            depth,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            sequence_data,
            epc_hits.data_ptr<float>(),
            epc_normals.data_ptr<float>(),
            epc_path_length.data_ptr<float>(),
            tx_indices.data_ptr<int>(),
            rx_indices.data_ptr<int>(),
            face_material_id.data_ptr<int>(),
            tensor("valid").data_ptr<bool>(),
            tensor("tx_id").data_ptr<int>(),
            tensor("rx_id").data_ptr<int>(),
            tensor("depth").data_ptr<int>(),
            tensor("component_id").data_ptr<int>(),
            tensor("primitive_id").data_ptr<int>(),
            tensor("edge_id").data_ptr<int>(),
            tensor("path_length_m").data_ptr<float>(),
            tensor("delay_s").data_ptr<float>(),
            tensor("path_gain").data_ptr<float>(),
            tensor("path_field").data_ptr<c10::complex<float>>(),
            tensor("field_xyz").data_ptr<c10::complex<float>>(),
            tensor("interaction_position").data_ptr<float>(),
            tensor("interaction_normal").data_ptr<float>(),
            tensor("material_id").data_ptr<int>(),
            tensor("primitive_sequence").data_ptr<int>(),
            tensor("material_sequence").data_ptr<int>(),
            tensor("interaction_positions").data_ptr<float>(),
            tensor("interaction_normals").data_ptr<float>());
    };
    if (epc_sequences.scalar_type() == at::kLong) {
        launch(epc_sequences.data_ptr<int64_t>());
    } else {
        launch(epc_sequences.data_ptr<int>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

pybind11::dict channel_deterministic_reflection_order1_compact(
    at::Tensor visible, at::Tensor epc_faces, at::Tensor epc_hits, at::Tensor epc_normals,
    at::Tensor sequence_batch, at::Tensor rx_indices, at::Tensor tx, at::Tensor rx_positions,
    at::Tensor tx_power, int64_t tx_index, at::Tensor face_eps_r, at::Tensor face_sigma_e,
    at::Tensor face_mu_r, at::Tensor face_gain, at::Tensor face_material_id, bool grouped_export) {
    check_tensor(visible, "visible", at::kBool, 1);
    TORCH_CHECK(epc_faces.is_cuda(), "epc_faces must be a CUDA tensor");
    TORCH_CHECK(epc_faces.is_contiguous(), "epc_faces must be contiguous");
    TORCH_CHECK(epc_faces.dim() == 2, "epc_faces must have shape (N, depth)");
    TORCH_CHECK(epc_faces.scalar_type() == at::kInt || epc_faces.scalar_type() == at::kLong,
                "epc_faces must be int32 or int64");
    check_tensor(epc_hits, "epc_hits", at::kFloat, 3);
    check_tensor(epc_normals, "epc_normals", at::kFloat, 3);
    check_tensor(sequence_batch, "sequence_batch", at::kInt, 2);
    check_reflection_compact_tensors(rx_indices, tx, rx_positions, tx_power, face_eps_r,
                                     face_sigma_e, face_mu_r, face_gain, face_material_id);
    const int64_t count = visible.size(0);
    TORCH_CHECK(epc_faces.size(0) == count, "epc_faces must match visible");
    TORCH_CHECK(epc_faces.size(1) >= 1, "epc_faces must include the first bounce");
    TORCH_CHECK(epc_hits.size(0) == count, "epc_hits must match visible");
    TORCH_CHECK(epc_normals.sizes() == epc_hits.sizes(), "epc_normals must match epc_hits");
    TORCH_CHECK(epc_hits.size(1) >= 1 && epc_hits.size(2) == 3,
                "epc_hits must have shape (N, depth, 3)");
    TORCH_CHECK(sequence_batch.size(0) == count, "sequence_batch must match visible");
    TORCH_CHECK(sequence_batch.size(1) >= 1, "sequence_batch must include the first bounce");
    check_reflection_compact_shapes(rx_indices, tx_power, tx_index, face_eps_r, face_sigma_e,
                                    face_mu_r, face_gain, face_material_id, count);
    const int device = visible.get_device();
    TORCH_CHECK(epc_faces.get_device() == device, "epc_faces must share visible device");
    TORCH_CHECK(epc_hits.get_device() == device, "epc_hits must share visible device");
    TORCH_CHECK(epc_normals.get_device() == device, "epc_normals must share visible device");
    TORCH_CHECK(sequence_batch.get_device() == device, "sequence_batch must share visible device");
    check_reflection_compact_device(rx_indices, tx, rx_positions, tx_power, face_eps_r,
                                    face_sigma_e, face_mu_r, face_gain, face_material_id, device);

    auto int_options = visible.options().dtype(at::kInt);
    auto float_options = visible.options().dtype(at::kFloat);
    auto empty_out = [&]() {
        pybind11::dict out;
        out["selected_faces"] = at::empty({0}, int_options);
        out["selected_points"] = at::empty({0, 3}, float_options);
        out["selected_normals"] = at::empty({0, 3}, float_options);
        out["selected_rx_id"] = at::empty({0}, int_options);
        out["tx_keep"] = at::empty({0, 3}, float_options);
        out["rx_keep"] = at::empty({0, 3}, float_options);
        out["tx_power"] = at::empty({0}, float_options);
        out["eps_r"] = at::empty({0}, float_options);
        out["sigma_e"] = at::empty({0}, float_options);
        out["mu_r"] = at::empty({0}, float_options);
        out["gain"] = at::empty({0}, float_options);
        out["material_id"] = at::empty({0}, int_options);
        return out;
    };
    if (count == 0) {
        return empty_out();
    }

    auto flags = at::empty({count}, int_options);
    auto offsets = at::empty({count + 1}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    path_visibility_flags_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                   kPathBlockSize, 0, stream>>>(
        count,
        visible.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    channel::scan::exclusive_flag_offsets(flags, offsets, count, stream);
    const int64_t out_count =
        observe_compact_count(offsets, count, stream).count;
    if (out_count == 0) {
        return empty_out();
    }

    pybind11::dict out;
    out["selected_faces"] = at::empty({out_count}, int_options);
    out["selected_points"] = at::empty({out_count, 3}, float_options);
    out["selected_normals"] = at::empty({out_count, 3}, float_options);
    out["selected_rx_id"] = at::empty({out_count}, int_options);
    out["tx_keep"] = at::empty({out_count, 3}, float_options);
    out["rx_keep"] = at::empty({out_count, 3}, float_options);
    out["tx_power"] = at::empty({out_count}, float_options);
    out["eps_r"] = at::empty({out_count}, float_options);
    out["sigma_e"] = at::empty({out_count}, float_options);
    out["mu_r"] = at::empty({out_count}, float_options);
    out["gain"] = at::empty({out_count}, float_options);
    out["material_id"] = at::empty({out_count}, int_options);

    if (epc_faces.scalar_type() == at::kLong) {
        deterministic_reflection_order1_compact_kernel<int64_t><<<
            ::channel::launch_blocks(count, kPathBlockSize),
            kPathBlockSize,
            0,
            stream>>>(
            count,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            grouped_export,
            epc_faces.data_ptr<int64_t>(),
            epc_faces.size(1),
            epc_hits.data_ptr<float>(),
            epc_normals.data_ptr<float>(),
            epc_hits.size(1),
            sequence_batch.data_ptr<int>(),
            sequence_batch.size(1),
            rx_indices.data_ptr<int>(),
            tx.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            static_cast<int>(tx_index),
            face_eps_r.data_ptr<float>(),
            face_sigma_e.data_ptr<float>(),
            face_mu_r.data_ptr<float>(),
            face_gain.data_ptr<float>(),
            face_material_id.data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["selected_faces"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["selected_points"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["selected_normals"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["selected_rx_id"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["tx_keep"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["rx_keep"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["tx_power"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["eps_r"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["sigma_e"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["mu_r"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["gain"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["material_id"]).data_ptr<int>());
    } else {
        deterministic_reflection_order1_compact_kernel<int><<<
            ::channel::launch_blocks(count, kPathBlockSize),
            kPathBlockSize,
            0,
            stream>>>(
            count,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            grouped_export,
            epc_faces.data_ptr<int>(),
            epc_faces.size(1),
            epc_hits.data_ptr<float>(),
            epc_normals.data_ptr<float>(),
            epc_hits.size(1),
            sequence_batch.data_ptr<int>(),
            sequence_batch.size(1),
            rx_indices.data_ptr<int>(),
            tx.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            static_cast<int>(tx_index),
            face_eps_r.data_ptr<float>(),
            face_sigma_e.data_ptr<float>(),
            face_mu_r.data_ptr<float>(),
            face_gain.data_ptr<float>(),
            face_material_id.data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["selected_faces"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["selected_points"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["selected_normals"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["selected_rx_id"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["tx_keep"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["rx_keep"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["tx_power"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["eps_r"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["sigma_e"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["mu_r"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["gain"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["material_id"]).data_ptr<int>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

pybind11::dict channel_deterministic_reflection_sequence_compact(
    at::Tensor visible, at::Tensor epc_sequences, at::Tensor epc_hits, at::Tensor epc_normals,
    at::Tensor rx_indices, at::Tensor tx, at::Tensor rx_positions, at::Tensor tx_power,
    int64_t tx_index, at::Tensor face_eps_r, at::Tensor face_sigma_e, at::Tensor face_mu_r,
    at::Tensor face_gain, at::Tensor face_material_id, int64_t max_count) {
    check_tensor(visible, "visible", at::kBool, 1);
    TORCH_CHECK(epc_sequences.is_cuda(), "epc_sequences must be a CUDA tensor");
    TORCH_CHECK(epc_sequences.is_contiguous(), "epc_sequences must be contiguous");
    TORCH_CHECK(epc_sequences.dim() == 2, "epc_sequences must have shape (N, depth)");
    TORCH_CHECK(epc_sequences.scalar_type() == at::kInt || epc_sequences.scalar_type() == at::kLong,
                "epc_sequences must be int32 or int64");
    check_tensor(epc_hits, "epc_hits", at::kFloat, 3);
    check_tensor(epc_normals, "epc_normals", at::kFloat, 3);
    check_reflection_compact_tensors(rx_indices, tx, rx_positions, tx_power, face_eps_r,
                                     face_sigma_e, face_mu_r, face_gain, face_material_id);
    TORCH_CHECK(max_count >= -1, "max_count must be -1 or non-negative");
    const int64_t count = visible.size(0);
    const int64_t depth = epc_sequences.size(1);
    TORCH_CHECK(depth > 0, "epc_sequences must have positive depth");
    TORCH_CHECK(epc_sequences.size(0) == count, "epc_sequences must match visible");
    TORCH_CHECK(epc_hits.size(0) == count, "epc_hits must match visible");
    TORCH_CHECK(epc_hits.size(1) == depth && epc_hits.size(2) == 3,
                "epc_hits must have shape (N, depth, 3)");
    TORCH_CHECK(epc_normals.sizes() == epc_hits.sizes(), "epc_normals must match epc_hits");
    check_reflection_compact_shapes(rx_indices, tx_power, tx_index, face_eps_r, face_sigma_e,
                                    face_mu_r, face_gain, face_material_id, count);
    const int device = visible.get_device();
    TORCH_CHECK(epc_sequences.get_device() == device, "epc_sequences must share visible device");
    TORCH_CHECK(epc_hits.get_device() == device, "epc_hits must share visible device");
    TORCH_CHECK(epc_normals.get_device() == device, "epc_normals must share visible device");
    check_reflection_compact_device(rx_indices, tx, rx_positions, tx_power, face_eps_r,
                                    face_sigma_e, face_mu_r, face_gain, face_material_id, device);

    auto int_options = visible.options().dtype(at::kInt);
    auto float_options = visible.options().dtype(at::kFloat);
    auto empty_out = [&]() {
        pybind11::dict out;
        out["selected_sequences"] = at::empty({0, depth}, int_options);
        out["selected_hits"] = at::empty({0, depth, 3}, float_options);
        out["selected_normals"] = at::empty({0, depth, 3}, float_options);
        out["selected_rx_id"] = at::empty({0}, int_options);
        out["selected_tx"] = at::empty({0, 3}, float_options);
        out["selected_rx"] = at::empty({0, 3}, float_options);
        out["tx_power"] = at::empty({0}, float_options);
        out["eps_r"] = at::empty({0, depth}, float_options);
        out["sigma_e"] = at::empty({0, depth}, float_options);
        out["mu_r"] = at::empty({0, depth}, float_options);
        out["gain"] = at::empty({0, depth}, float_options);
        out["first_face"] = at::empty({0}, int_options);
        out["material_id"] = at::empty({0}, int_options);
        out["material_sequence"] = at::empty({0, depth}, int_options);
        out["first_hit"] = at::empty({0, 3}, float_options);
        out["first_normal"] = at::empty({0, 3}, float_options);
        return out;
    };
    if (count == 0 || max_count == 0) {
        return empty_out();
    }

    auto flags = at::empty({count}, int_options);
    auto offsets = at::empty({count + 1}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    path_visibility_flags_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                   kPathBlockSize, 0, stream>>>(
        count,
        visible.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    channel::scan::exclusive_flag_offsets(flags, offsets, count, stream);
    const int64_t visible_count =
        observe_compact_count(offsets, count, stream).count;
    const int64_t out_count = max_count < 0 ? visible_count : std::min<int64_t>(visible_count, max_count);
    if (out_count == 0) {
        return empty_out();
    }

    pybind11::dict out;
    out["selected_sequences"] = at::empty({out_count, depth}, int_options);
    out["selected_hits"] = at::empty({out_count, depth, 3}, float_options);
    out["selected_normals"] = at::empty({out_count, depth, 3}, float_options);
    out["selected_rx_id"] = at::empty({out_count}, int_options);
    out["selected_tx"] = at::empty({out_count, 3}, float_options);
    out["selected_rx"] = at::empty({out_count, 3}, float_options);
    out["tx_power"] = at::empty({out_count}, float_options);
    out["eps_r"] = at::empty({out_count, depth}, float_options);
    out["sigma_e"] = at::empty({out_count, depth}, float_options);
    out["mu_r"] = at::empty({out_count, depth}, float_options);
    out["gain"] = at::empty({out_count, depth}, float_options);
    out["first_face"] = at::empty({out_count}, int_options);
    out["material_id"] = at::empty({out_count}, int_options);
    out["material_sequence"] = at::empty({out_count, depth}, int_options);
    out["first_hit"] = at::empty({out_count, 3}, float_options);
    out["first_normal"] = at::empty({out_count, 3}, float_options);

    if (epc_sequences.scalar_type() == at::kLong) {
        deterministic_reflection_sequence_compact_kernel<int64_t><<<
            ::channel::launch_blocks(count, kPathBlockSize),
            kPathBlockSize,
            0,
            stream>>>(
            count,
            depth,
            out_count,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            epc_sequences.data_ptr<int64_t>(),
            epc_hits.data_ptr<float>(),
            epc_normals.data_ptr<float>(),
            rx_indices.data_ptr<int>(),
            tx.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            static_cast<int>(tx_index),
            face_eps_r.data_ptr<float>(),
            face_sigma_e.data_ptr<float>(),
            face_mu_r.data_ptr<float>(),
            face_gain.data_ptr<float>(),
            face_material_id.data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["selected_sequences"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["selected_hits"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["selected_normals"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["selected_rx_id"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["selected_tx"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["selected_rx"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["tx_power"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["eps_r"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["sigma_e"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["mu_r"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["gain"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["first_face"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["material_id"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["material_sequence"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["first_hit"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["first_normal"]).data_ptr<float>());
    } else {
        deterministic_reflection_sequence_compact_kernel<int><<<
            ::channel::launch_blocks(count, kPathBlockSize),
            kPathBlockSize,
            0,
            stream>>>(
            count,
            depth,
            out_count,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            epc_sequences.data_ptr<int>(),
            epc_hits.data_ptr<float>(),
            epc_normals.data_ptr<float>(),
            rx_indices.data_ptr<int>(),
            tx.data_ptr<float>(),
            rx_positions.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            static_cast<int>(tx_index),
            face_eps_r.data_ptr<float>(),
            face_sigma_e.data_ptr<float>(),
            face_mu_r.data_ptr<float>(),
            face_gain.data_ptr<float>(),
            face_material_id.data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["selected_sequences"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["selected_hits"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["selected_normals"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["selected_rx_id"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["selected_tx"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["selected_rx"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["tx_power"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["eps_r"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["sigma_e"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["mu_r"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["gain"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["first_face"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["material_id"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["material_sequence"]).data_ptr<int>(),
            pybind11::cast<at::Tensor>(out["first_hit"]).data_ptr<float>(),
            pybind11::cast<at::Tensor>(out["first_normal"]).data_ptr<float>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

pybind11::dict channel_deterministic_diffraction_topology_compact(
    at::Tensor valid, at::Tensor tx_id, at::Tensor rx_id, int64_t rx_offset, at::Tensor depth,
    at::Tensor edge_id, at::Tensor delay, at::Tensor x_re, at::Tensor x_im, at::Tensor y_re,
    at::Tensor y_im, at::Tensor z_re, at::Tensor z_im, at::Tensor interaction_position) {
    check_tensor(valid, "valid", at::kBool, 1);
    const int64_t count = valid.size(0);
    const std::pair<const char*, const at::Tensor*> int_inputs[] = {
        {"tx_id", &tx_id}, {"rx_id", &rx_id}, {"depth", &depth}, {"edge_id", &edge_id}};
    for (const auto& item : int_inputs) {
        check_tensor(*item.second, item.first, at::kInt, 1);
        TORCH_CHECK(item.second->size(0) == count, item.first, " must match valid");
    }
    const std::pair<const char*, const at::Tensor*> float_inputs[] = {
        {"delay_s", &delay}, {"x_re", &x_re}, {"x_im", &x_im}, {"y_re", &y_re},
        {"y_im", &y_im}, {"z_re", &z_re}, {"z_im", &z_im}};
    for (const auto& item : float_inputs) {
        check_tensor(*item.second, item.first, at::kFloat, 1);
        TORCH_CHECK(item.second->size(0) == count, item.first, " must match valid");
    }
    check_vec3_table(interaction_position, "interaction_position");
    TORCH_CHECK(interaction_position.size(0) == count, "interaction_position must match valid");
    TORCH_CHECK(rx_offset >= 0 && rx_offset <= std::numeric_limits<int>::max(), "rx_offset is out of range");
    const int device = valid.get_device();
    for (const at::Tensor* tensor : {&tx_id, &rx_id, &depth, &edge_id, &delay, &x_re, &x_im, &y_re, &y_im,
                                     &z_re, &z_im, &interaction_position}) {
        TORCH_CHECK(tensor->get_device() == device, "diffraction compact inputs must share one device");
    }
    auto bool_options = valid.options().dtype(at::kBool);
    auto int_options = valid.options().dtype(at::kInt);
    auto float_options = valid.options().dtype(at::kFloat);
    auto complex_options = valid.options().dtype(at::kComplexFloat);
    auto allocate = [&](int64_t rows) {
        pybind11::dict out;
        out["valid"] = at::empty({rows}, bool_options);
        out["tx_id"] = at::empty({rows}, int_options);
        out["rx_id"] = at::empty({rows}, int_options);
        out["depth"] = at::empty({rows}, int_options);
        out["component_id"] = at::empty({rows}, int_options);
        out["primitive_id"] = at::empty({rows}, int_options);
        out["edge_id"] = at::empty({rows}, int_options);
        out["path_length_m"] = at::empty({rows}, float_options);
        out["delay_s"] = at::empty({rows}, float_options);
        out["path_gain"] = at::empty({rows}, float_options);
        out["path_field"] = at::empty({rows}, complex_options);
        out["field_xyz"] = at::empty({rows, 3}, complex_options);
        out["interaction_position"] = at::empty({rows, 3}, float_options);
        out["interaction_normal"] = at::empty({rows, 3}, float_options);
        out["material_id"] = at::empty({rows}, int_options);
        return out;
    };
    if (count == 0) {
        return allocate(0);
    }
    auto flags = at::empty({count}, int_options);
    auto offsets = at::empty({count + 1}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    path_visibility_flags_kernel<<<::channel::launch_blocks(count, kPathBlockSize),
                                   kPathBlockSize, 0, stream>>>(
        count,
        valid.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    channel::scan::exclusive_flag_offsets(flags, offsets, count, stream);
    const int64_t out_count = observe_compact_count(offsets, count, stream).count;
    pybind11::dict out = allocate(out_count);
    if (out_count == 0) {
        return out;
    }
    auto tensor = [&](const char* name) { return pybind11::cast<at::Tensor>(out[name]); };
    deterministic_diffraction_topology_compact_kernel<<<
        ::channel::launch_blocks(count, kPathBlockSize), kPathBlockSize, 0, stream>>>(
        count,
        static_cast<int>(rx_offset),
        flags.data_ptr<int>(),
        offsets.data_ptr<int>(),
        tx_id.data_ptr<int>(),
        rx_id.data_ptr<int>(),
        depth.data_ptr<int>(),
        edge_id.data_ptr<int>(),
        delay.data_ptr<float>(),
        x_re.data_ptr<float>(),
        x_im.data_ptr<float>(),
        y_re.data_ptr<float>(),
        y_im.data_ptr<float>(),
        z_re.data_ptr<float>(),
        z_im.data_ptr<float>(),
        interaction_position.data_ptr<float>(),
        tensor("valid").data_ptr<bool>(),
        tensor("tx_id").data_ptr<int>(),
        tensor("rx_id").data_ptr<int>(),
        tensor("depth").data_ptr<int>(),
        tensor("component_id").data_ptr<int>(),
        tensor("primitive_id").data_ptr<int>(),
        tensor("edge_id").data_ptr<int>(),
        tensor("path_length_m").data_ptr<float>(),
        tensor("delay_s").data_ptr<float>(),
        tensor("path_gain").data_ptr<float>(),
        tensor("path_field").data_ptr<c10::complex<float>>(),
        tensor("field_xyz").data_ptr<c10::complex<float>>(),
        tensor("interaction_position").data_ptr<float>(),
        tensor("interaction_normal").data_ptr<float>(),
        tensor("material_id").data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

#undef channel_neg_kd_phase

// ==== Section: Deterministic topology ====
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include "torch_cuda.h"
#include "math.cuh"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "path_compaction.cuh"

namespace {

constexpr int64_t kFaceGroupKeyWidth = 5;

__global__ void deterministic_order_init_kernel(
    int64_t count,
    int64_t* __restrict__ order) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        order[index] = index;
    }
}

__global__ void deterministic_face_group_keys_kernel(
    int64_t count,
    const float* __restrict__ tri_a,
    const float* __restrict__ normals,
    const int64_t* __restrict__ surface_ids,
    float scale,
    int64_t* __restrict__ key_matrix) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t face = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         face < count;
         face += stride) {
        const int64_t vec_base = face * 3;
        float nx = normals[vec_base + 0];
        float ny = normals[vec_base + 1];
        float nz = normals[vec_base + 2];
        const float norm = sqrtf(nx * nx + ny * ny + nz * nz);
        const float inv_norm = norm > 1.0e-6f ? 1.0f / norm : 0.0f;
        nx *= inv_norm;
        ny *= inv_norm;
        nz *= inv_norm;

        float major_abs = fabsf(nx);
        float major_component = nx;
        const float ay = fabsf(ny);
        if (ay > major_abs) {
            major_abs = ay;
            major_component = ny;
        }
        const float az = fabsf(nz);
        if (az > major_abs) {
            major_component = nz;
        }

        const float sign = major_component < 0.0f ? -1.0f : 1.0f;
        nx *= sign;
        ny *= sign;
        nz *= sign;
        const float px = tri_a[vec_base + 0];
        const float py = tri_a[vec_base + 1];
        const float pz = tri_a[vec_base + 2];
        const float offset = nx * px + ny * py + nz * pz;

        const int64_t key_base = face * kFaceGroupKeyWidth;
        key_matrix[key_base + 0] = surface_ids[face];
        key_matrix[key_base + 1] = static_cast<int64_t>(llroundf(nx * scale));
        key_matrix[key_base + 2] = static_cast<int64_t>(llroundf(ny * scale));
        key_matrix[key_base + 3] = static_cast<int64_t>(llroundf(nz * scale));
        key_matrix[key_base + 4] = static_cast<int64_t>(llroundf(offset * scale));
    }
}

__global__ void deterministic_face_group_sort_key_kernel(
    int64_t count,
    int64_t column,
    const int64_t* __restrict__ key_matrix,
    const int64_t* __restrict__ order,
    int64_t* __restrict__ keys) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        keys[index] = key_matrix[order[index] * kFaceGroupKeyWidth + column];
    }
}

__global__ void deterministic_face_group_flags_kernel(
    int64_t count,
    const int64_t* __restrict__ key_matrix,
    const int64_t* __restrict__ order,
    int* __restrict__ flags) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        bool starts_group = index == 0;
        if (!starts_group) {
            const int64_t face = order[index];
            const int64_t prev = order[index - 1];
            for (int column = 0; column < kFaceGroupKeyWidth; ++column) {
                if (key_matrix[face * kFaceGroupKeyWidth + column] !=
                    key_matrix[prev * kFaceGroupKeyWidth + column]) {
                    starts_group = true;
                    break;
                }
            }
        }
        flags[index] = starts_group ? 1 : 0;
    }
}

__global__ void deterministic_face_group_assign_kernel(
    int64_t count,
    const int64_t* __restrict__ order,
    const int* __restrict__ flags,
    const int* __restrict__ group_offsets,
    int* __restrict__ face_group_id,
    int64_t* __restrict__ representative_faces,
    int* __restrict__ surface_group_size) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int group = group_offsets[index] + flags[index] - 1;
        const int64_t face = order[index];
        face_group_id[face] = group;
        if (flags[index] != 0) {
            representative_faces[group] = face;
        }
        atomicAdd(surface_group_size + group, 1);
    }
}

__global__ void deterministic_face_group_members_kernel(
    int64_t count,
    int max_group_size,
    const int64_t* __restrict__ order,
    const int* __restrict__ flags,
    const int* __restrict__ group_offsets,
    int* __restrict__ member_offsets,
    int* __restrict__ surface_group_members) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int group = group_offsets[index] + flags[index] - 1;
        const int slot = atomicAdd(member_offsets + group, 1);
        surface_group_members[static_cast<int64_t>(group) * max_group_size + slot] =
            static_cast<int>(order[index]);
    }
}

at::Tensor block_tensor(const pybind11::dict& block, const char* key) {
    TORCH_CHECK(block.contains(key), "topology block is missing field ", key);
    return pybind11::cast<at::Tensor>(block[key]);
}

bool block_has_field(const pybind11::dict& block, const char* key) {
    return block.contains(key);
}

void check_optional_field_presence(
    const pybind11::dict& block,
    const char* key,
    bool expected) {
    const bool actual = block_has_field(block, key);
    TORCH_CHECK(actual == expected, "all topology blocks must share field presence for ", key);
}

void check_topology_concat_schema(
    const pybind11::dict& block,
    int64_t sequence_width,
    bool has_path_field,
    bool has_field_xyz,
    bool has_coefficient,
    bool has_interaction_position,
    bool has_interaction_normal,
    bool has_material_id,
    bool has_primitive_sequence,
    bool has_material_sequence,
    bool has_interaction_positions,
    bool has_interaction_normals) {
    at::Tensor valid = block_tensor(block, "valid");
    at::Tensor tx_id = block_tensor(block, "tx_id");
    at::Tensor rx_id = block_tensor(block, "rx_id");
    at::Tensor depth = block_tensor(block, "depth");
    at::Tensor component_id = block_tensor(block, "component_id");
    at::Tensor primitive_id = block_tensor(block, "primitive_id");
    at::Tensor edge_id = block_tensor(block, "edge_id");
    at::Tensor path_length = block_tensor(block, "path_length_m");
    at::Tensor delay = block_tensor(block, "delay_s");
    at::Tensor path_gain = block_tensor(block, "path_gain");
    check_path_block_shapes(valid, tx_id, rx_id, depth, component_id, primitive_id, edge_id, path_length, delay, path_gain);
    const int64_t count = valid.size(0);
    const int device = valid.get_device();

    check_optional_field_presence(block, "path_field", has_path_field);
    if (has_path_field) {
        at::Tensor path_field = block_tensor(block, "path_field");
        check_tensor(path_field, "path_field", at::kComplexFloat, 1);
        TORCH_CHECK(path_field.size(0) == count, "path_field must match valid");
        TORCH_CHECK(path_field.get_device() == device, "path_field must share valid device");
    }

    check_optional_field_presence(block, "field_xyz", has_field_xyz);
    if (has_field_xyz) {
        at::Tensor field_xyz = block_tensor(block, "field_xyz");
        check_tensor(field_xyz, "field_xyz", at::kComplexFloat, 2);
        TORCH_CHECK(field_xyz.size(0) == count && field_xyz.size(1) == 3,
                    "field_xyz must have shape (N, 3)");
        TORCH_CHECK(field_xyz.get_device() == device, "field_xyz must share valid device");
    }

    check_optional_field_presence(block, "coefficient", has_coefficient);
    if (has_coefficient) {
        at::Tensor coefficient = block_tensor(block, "coefficient");
        check_tensor(coefficient, "coefficient", at::kComplexFloat, 1);
        TORCH_CHECK(coefficient.size(0) == count, "coefficient must match valid");
        TORCH_CHECK(coefficient.get_device() == device, "coefficient must share valid device");
    }

    check_optional_field_presence(block, "interaction_position", has_interaction_position);
    if (has_interaction_position) {
        at::Tensor interaction_position = block_tensor(block, "interaction_position");
        check_vec3_table(interaction_position, "interaction_position");
        TORCH_CHECK(interaction_position.size(0) == count, "interaction_position must match valid");
        TORCH_CHECK(interaction_position.get_device() == device, "interaction_position must share valid device");
    }

    check_optional_field_presence(block, "interaction_normal", has_interaction_normal);
    if (has_interaction_normal) {
        at::Tensor interaction_normal = block_tensor(block, "interaction_normal");
        check_vec3_table(interaction_normal, "interaction_normal");
        TORCH_CHECK(interaction_normal.size(0) == count, "interaction_normal must match valid");
        TORCH_CHECK(interaction_normal.get_device() == device, "interaction_normal must share valid device");
    }

    check_optional_field_presence(block, "material_id", has_material_id);
    if (has_material_id) {
        at::Tensor material_id = block_tensor(block, "material_id");
        check_tensor(material_id, "material_id", at::kInt, 1);
        TORCH_CHECK(material_id.size(0) == count, "material_id must match valid");
        TORCH_CHECK(material_id.get_device() == device, "material_id must share valid device");
    }

    check_optional_field_presence(block, "primitive_sequence", has_primitive_sequence);
    if (has_primitive_sequence) {
        at::Tensor primitive_sequence = block_tensor(block, "primitive_sequence");
        check_tensor(primitive_sequence, "primitive_sequence", at::kInt, 2);
        TORCH_CHECK(primitive_sequence.size(0) == count, "primitive_sequence must match valid");
        TORCH_CHECK(primitive_sequence.size(1) == sequence_width, "primitive_sequence has wrong width");
        TORCH_CHECK(primitive_sequence.get_device() == device, "primitive_sequence must share valid device");
    }

    check_optional_field_presence(block, "material_sequence", has_material_sequence);
    if (has_material_sequence) {
        at::Tensor material_sequence = block_tensor(block, "material_sequence");
        check_tensor(material_sequence, "material_sequence", at::kInt, 2);
        TORCH_CHECK(material_sequence.size(0) == count, "material_sequence must match valid");
        TORCH_CHECK(material_sequence.size(1) == sequence_width, "material_sequence has wrong width");
        TORCH_CHECK(material_sequence.get_device() == device, "material_sequence must share valid device");
    }

    check_optional_field_presence(block, "interaction_positions", has_interaction_positions);
    if (has_interaction_positions) {
        at::Tensor interaction_positions = block_tensor(block, "interaction_positions");
        check_tensor(interaction_positions, "interaction_positions", at::kFloat, 3);
        TORCH_CHECK(interaction_positions.size(0) == count, "interaction_positions must match valid");
        TORCH_CHECK(interaction_positions.size(1) == sequence_width, "interaction_positions has wrong width");
        TORCH_CHECK(interaction_positions.size(2) == 3, "interaction_positions must have vec3 rows");
        TORCH_CHECK(interaction_positions.get_device() == device, "interaction_positions must share valid device");
    }

    check_optional_field_presence(block, "interaction_normals", has_interaction_normals);
    if (has_interaction_normals) {
        at::Tensor interaction_normals = block_tensor(block, "interaction_normals");
        check_tensor(interaction_normals, "interaction_normals", at::kFloat, 3);
        TORCH_CHECK(interaction_normals.size(0) == count, "interaction_normals must match valid");
        TORCH_CHECK(interaction_normals.size(1) == sequence_width, "interaction_normals has wrong width");
        TORCH_CHECK(interaction_normals.size(2) == 3, "interaction_normals must have vec3 rows");
        TORCH_CHECK(interaction_normals.get_device() == device, "interaction_normals must share valid device");
    }
}

void copy_tensor_rows(
    const at::Tensor& src,
    const at::Tensor& dst,
    int64_t dst_row_offset,
    int64_t row_stride,
    cudaStream_t stream) {
    if (src.numel() == 0) {
        return;
    }
    const size_t element_size = src.element_size();
    char* dst_ptr = static_cast<char*>(dst.data_ptr()) + dst_row_offset * row_stride * element_size;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        dst_ptr,
        src.data_ptr(),
        src.numel() * element_size,
        cudaMemcpyDeviceToDevice,
        stream));
}

template <typename scalar_t>
__global__ void deterministic_gather_rows_kernel(
    int64_t out_count,
    int64_t row_stride,
    const int64_t* __restrict__ order,
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ dst) {
    const int64_t total = out_count * row_stride;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += stride) {
        const int64_t row = linear / row_stride;
        const int64_t col = linear - row * row_stride;
        const int64_t src_row = order[row];
        dst[linear] = src[src_row * row_stride + col];
    }
}

template <typename scalar_t>
void gather_tensor_rows(
    const at::Tensor& src,
    const at::Tensor& dst,
    const at::Tensor& order,
    int64_t out_count,
    int64_t row_stride,
    cudaStream_t stream) {
    if (out_count == 0 || row_stride == 0) {
        return;
    }
    deterministic_gather_rows_kernel<scalar_t><<<
        ::channel::launch_blocks(out_count * row_stride, kPathBlockSize),
        kPathBlockSize, 0, stream>>>(
        out_count,
        row_stride,
        order.data_ptr<int64_t>(),
        src.data_ptr<scalar_t>(),
        dst.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

pybind11::dict channel_deterministic_concat_topology_blocks(pybind11::sequence blocks, int64_t sequence_width) {
    TORCH_CHECK(sequence_width >= 0, "sequence_width must be non-negative");
    TORCH_CHECK(pybind11::len(blocks) > 0, "topology blocks must not be empty");

    std::vector<pybind11::dict> block_dicts;
    block_dicts.reserve(pybind11::len(blocks));
    int64_t total = 0;
    int device = -1;
    bool schema_initialized = false;
    bool has_path_field = false;
    bool has_field_xyz = false;
    bool has_coefficient = false;
    bool has_interaction_position = false;
    bool has_interaction_normal = false;
    bool has_material_id = false;
    bool has_primitive_sequence = false;
    bool has_material_sequence = false;
    bool has_interaction_positions = false;
    bool has_interaction_normals = false;
    for (auto item : blocks) {
        pybind11::dict block = pybind11::cast<pybind11::dict>(item);
        if (!schema_initialized) {
            has_path_field = block_has_field(block, "path_field");
            has_field_xyz = block_has_field(block, "field_xyz");
            has_coefficient = block_has_field(block, "coefficient");
            has_interaction_position = block_has_field(block, "interaction_position");
            has_interaction_normal = block_has_field(block, "interaction_normal");
            has_material_id = block_has_field(block, "material_id");
            has_primitive_sequence = block_has_field(block, "primitive_sequence");
            has_material_sequence = block_has_field(block, "material_sequence");
            has_interaction_positions = block_has_field(block, "interaction_positions");
            has_interaction_normals = block_has_field(block, "interaction_normals");
            schema_initialized = true;
        }
        check_topology_concat_schema(
            block,
            sequence_width,
            has_path_field,
            has_field_xyz,
            has_coefficient,
            has_interaction_position,
            has_interaction_normal,
            has_material_id,
            has_primitive_sequence,
            has_material_sequence,
            has_interaction_positions,
            has_interaction_normals);
        at::Tensor valid = block_tensor(block, "valid");
        if (device < 0) {
            device = valid.get_device();
        }
        TORCH_CHECK(valid.get_device() == device, "all topology blocks must share a device");
        total += valid.size(0);
        block_dicts.push_back(block);
    }

    at::Tensor reference = block_tensor(block_dicts[0], "valid");
    auto bool_options = reference.options().dtype(at::kBool);
    auto int_options = reference.options().dtype(at::kInt);
    auto float_options = reference.options().dtype(at::kFloat);
    auto complex_options = reference.options().dtype(at::kComplexFloat);

    pybind11::dict out;
    out["valid"] = at::empty({total}, bool_options);
    out["tx_id"] = at::empty({total}, int_options);
    out["rx_id"] = at::empty({total}, int_options);
    out["depth"] = at::empty({total}, int_options);
    out["component_id"] = at::empty({total}, int_options);
    out["primitive_id"] = at::empty({total}, int_options);
    out["edge_id"] = at::empty({total}, int_options);
    out["path_length_m"] = at::empty({total}, float_options);
    out["delay_s"] = at::empty({total}, float_options);
    out["path_gain"] = at::empty({total}, float_options);
    if (has_path_field) {
        out["path_field"] = at::empty({total}, complex_options);
    }
    if (has_field_xyz) {
        out["field_xyz"] = at::empty({total, 3}, complex_options);
    }
    if (has_coefficient) {
        out["coefficient"] = at::empty({total}, complex_options);
    }
    if (has_interaction_position) {
        out["interaction_position"] = at::empty({total, 3}, float_options);
    }
    if (has_interaction_normal) {
        out["interaction_normal"] = at::empty({total, 3}, float_options);
    }
    if (has_material_id) {
        out["material_id"] = at::empty({total}, int_options);
    }
    if (has_primitive_sequence) {
        out["primitive_sequence"] = at::empty({total, sequence_width}, int_options);
    }
    if (has_material_sequence) {
        out["material_sequence"] = at::empty({total, sequence_width}, int_options);
    }
    if (has_interaction_positions) {
        out["interaction_positions"] = at::empty({total, sequence_width, 3}, float_options);
    }
    if (has_interaction_normals) {
        out["interaction_normals"] = at::empty({total, sequence_width, 3}, float_options);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    int64_t offset = 0;
    for (const auto& block : block_dicts) {
        at::Tensor valid = block_tensor(block, "valid");
        const int64_t count = valid.size(0);
        if (count == 0) {
            continue;
        }
        copy_tensor_rows(valid, pybind11::cast<at::Tensor>(out["valid"]), offset, 1, stream);
        copy_tensor_rows(block_tensor(block, "tx_id"), pybind11::cast<at::Tensor>(out["tx_id"]), offset, 1, stream);
        copy_tensor_rows(block_tensor(block, "rx_id"), pybind11::cast<at::Tensor>(out["rx_id"]), offset, 1, stream);
        copy_tensor_rows(block_tensor(block, "depth"), pybind11::cast<at::Tensor>(out["depth"]), offset, 1, stream);
        copy_tensor_rows(block_tensor(block, "component_id"), pybind11::cast<at::Tensor>(out["component_id"]), offset, 1, stream);
        copy_tensor_rows(block_tensor(block, "primitive_id"), pybind11::cast<at::Tensor>(out["primitive_id"]), offset, 1, stream);
        copy_tensor_rows(block_tensor(block, "edge_id"), pybind11::cast<at::Tensor>(out["edge_id"]), offset, 1, stream);
        copy_tensor_rows(block_tensor(block, "path_length_m"), pybind11::cast<at::Tensor>(out["path_length_m"]), offset, 1, stream);
        copy_tensor_rows(block_tensor(block, "delay_s"), pybind11::cast<at::Tensor>(out["delay_s"]), offset, 1, stream);
        copy_tensor_rows(block_tensor(block, "path_gain"), pybind11::cast<at::Tensor>(out["path_gain"]), offset, 1, stream);
        if (has_path_field) {
            copy_tensor_rows(block_tensor(block, "path_field"), pybind11::cast<at::Tensor>(out["path_field"]), offset, 1, stream);
        }
        if (has_field_xyz) {
            copy_tensor_rows(block_tensor(block, "field_xyz"), pybind11::cast<at::Tensor>(out["field_xyz"]), offset, 3, stream);
        }
        if (has_coefficient) {
            copy_tensor_rows(block_tensor(block, "coefficient"), pybind11::cast<at::Tensor>(out["coefficient"]), offset, 1, stream);
        }
        if (has_interaction_position) {
            copy_tensor_rows(block_tensor(block, "interaction_position"), pybind11::cast<at::Tensor>(out["interaction_position"]), offset, 3, stream);
        }
        if (has_interaction_normal) {
            copy_tensor_rows(block_tensor(block, "interaction_normal"), pybind11::cast<at::Tensor>(out["interaction_normal"]), offset, 3, stream);
        }
        if (has_material_id) {
            copy_tensor_rows(block_tensor(block, "material_id"), pybind11::cast<at::Tensor>(out["material_id"]), offset, 1, stream);
        }
        if (has_primitive_sequence) {
            copy_tensor_rows(block_tensor(block, "primitive_sequence"), pybind11::cast<at::Tensor>(out["primitive_sequence"]), offset, sequence_width, stream);
        }
        if (has_material_sequence) {
            copy_tensor_rows(block_tensor(block, "material_sequence"), pybind11::cast<at::Tensor>(out["material_sequence"]), offset, sequence_width, stream);
        }
        if (has_interaction_positions) {
            copy_tensor_rows(block_tensor(block, "interaction_positions"), pybind11::cast<at::Tensor>(out["interaction_positions"]), offset, sequence_width * 3, stream);
        }
        if (has_interaction_normals) {
            copy_tensor_rows(block_tensor(block, "interaction_normals"), pybind11::cast<at::Tensor>(out["interaction_normals"]), offset, sequence_width * 3, stream);
        }
        offset += count;
    }
    return out;
}

pybind11::dict channel_deterministic_gather_topology_block(
    pybind11::dict block,
    at::Tensor order,
    int64_t max_count,
    int64_t sequence_width) {
    TORCH_CHECK(sequence_width >= 0, "sequence_width must be non-negative");
    TORCH_CHECK(max_count >= -1, "max_count must be -1 or non-negative");

    const bool has_path_field = block_has_field(block, "path_field");
    const bool has_field_xyz = block_has_field(block, "field_xyz");
    const bool has_coefficient = block_has_field(block, "coefficient");
    const bool has_interaction_position = block_has_field(block, "interaction_position");
    const bool has_interaction_normal = block_has_field(block, "interaction_normal");
    const bool has_material_id = block_has_field(block, "material_id");
    const bool has_primitive_sequence = block_has_field(block, "primitive_sequence");
    const bool has_material_sequence = block_has_field(block, "material_sequence");
    const bool has_interaction_positions = block_has_field(block, "interaction_positions");
    const bool has_interaction_normals = block_has_field(block, "interaction_normals");
    check_topology_concat_schema(
        block,
        sequence_width,
        has_path_field,
        has_field_xyz,
        has_coefficient,
        has_interaction_position,
        has_interaction_normal,
        has_material_id,
        has_primitive_sequence,
        has_material_sequence,
        has_interaction_positions,
        has_interaction_normals);
    check_tensor(order, "order", at::kLong, 1);

    at::Tensor valid = block_tensor(block, "valid");
    TORCH_CHECK(order.get_device() == valid.get_device(), "order must share topology block device");
    const int64_t order_count = order.size(0);
    const int64_t out_count = max_count < 0 ? order_count : std::min<int64_t>(order_count, max_count);
    const int device = valid.get_device();
    auto bool_options = valid.options().dtype(at::kBool);
    auto int_options = valid.options().dtype(at::kInt);
    auto float_options = valid.options().dtype(at::kFloat);
    auto complex_options = valid.options().dtype(at::kComplexFloat);

    pybind11::dict out;
    out["valid"] = at::empty({out_count}, bool_options);
    out["tx_id"] = at::empty({out_count}, int_options);
    out["rx_id"] = at::empty({out_count}, int_options);
    out["depth"] = at::empty({out_count}, int_options);
    out["component_id"] = at::empty({out_count}, int_options);
    out["primitive_id"] = at::empty({out_count}, int_options);
    out["edge_id"] = at::empty({out_count}, int_options);
    out["path_length_m"] = at::empty({out_count}, float_options);
    out["delay_s"] = at::empty({out_count}, float_options);
    out["path_gain"] = at::empty({out_count}, float_options);
    if (has_path_field) {
        out["path_field"] = at::empty({out_count}, complex_options);
    }
    if (has_field_xyz) {
        out["field_xyz"] = at::empty({out_count, 3}, complex_options);
    }
    if (has_coefficient) {
        out["coefficient"] = at::empty({out_count}, complex_options);
    }
    if (has_interaction_position) {
        out["interaction_position"] = at::empty({out_count, 3}, float_options);
    }
    if (has_interaction_normal) {
        out["interaction_normal"] = at::empty({out_count, 3}, float_options);
    }
    if (has_material_id) {
        out["material_id"] = at::empty({out_count}, int_options);
    }
    if (has_primitive_sequence) {
        out["primitive_sequence"] = at::empty({out_count, sequence_width}, int_options);
    }
    if (has_material_sequence) {
        out["material_sequence"] = at::empty({out_count, sequence_width}, int_options);
    }
    if (has_interaction_positions) {
        out["interaction_positions"] = at::empty({out_count, sequence_width, 3}, float_options);
    }
    if (has_interaction_normals) {
        out["interaction_normals"] = at::empty({out_count, sequence_width, 3}, float_options);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    gather_tensor_rows<bool>(valid, pybind11::cast<at::Tensor>(out["valid"]), order, out_count, 1, stream);
    gather_tensor_rows<int>(block_tensor(block, "tx_id"), pybind11::cast<at::Tensor>(out["tx_id"]), order, out_count, 1, stream);
    gather_tensor_rows<int>(block_tensor(block, "rx_id"), pybind11::cast<at::Tensor>(out["rx_id"]), order, out_count, 1, stream);
    gather_tensor_rows<int>(block_tensor(block, "depth"), pybind11::cast<at::Tensor>(out["depth"]), order, out_count, 1, stream);
    gather_tensor_rows<int>(block_tensor(block, "component_id"), pybind11::cast<at::Tensor>(out["component_id"]), order, out_count, 1, stream);
    gather_tensor_rows<int>(block_tensor(block, "primitive_id"), pybind11::cast<at::Tensor>(out["primitive_id"]), order, out_count, 1, stream);
    gather_tensor_rows<int>(block_tensor(block, "edge_id"), pybind11::cast<at::Tensor>(out["edge_id"]), order, out_count, 1, stream);
    gather_tensor_rows<float>(block_tensor(block, "path_length_m"), pybind11::cast<at::Tensor>(out["path_length_m"]), order, out_count, 1, stream);
    gather_tensor_rows<float>(block_tensor(block, "delay_s"), pybind11::cast<at::Tensor>(out["delay_s"]), order, out_count, 1, stream);
    gather_tensor_rows<float>(block_tensor(block, "path_gain"), pybind11::cast<at::Tensor>(out["path_gain"]), order, out_count, 1, stream);
    if (has_path_field) {
        gather_tensor_rows<c10::complex<float>>(block_tensor(block, "path_field"), pybind11::cast<at::Tensor>(out["path_field"]), order, out_count, 1, stream);
    }
    if (has_field_xyz) {
        gather_tensor_rows<c10::complex<float>>(block_tensor(block, "field_xyz"), pybind11::cast<at::Tensor>(out["field_xyz"]), order, out_count, 3, stream);
    }
    if (has_coefficient) {
        gather_tensor_rows<c10::complex<float>>(block_tensor(block, "coefficient"), pybind11::cast<at::Tensor>(out["coefficient"]), order, out_count, 1, stream);
    }
    if (has_interaction_position) {
        gather_tensor_rows<float>(block_tensor(block, "interaction_position"), pybind11::cast<at::Tensor>(out["interaction_position"]), order, out_count, 3, stream);
    }
    if (has_interaction_normal) {
        gather_tensor_rows<float>(block_tensor(block, "interaction_normal"), pybind11::cast<at::Tensor>(out["interaction_normal"]), order, out_count, 3, stream);
    }
    if (has_material_id) {
        gather_tensor_rows<int>(block_tensor(block, "material_id"), pybind11::cast<at::Tensor>(out["material_id"]), order, out_count, 1, stream);
    }
    if (has_primitive_sequence) {
        gather_tensor_rows<int>(block_tensor(block, "primitive_sequence"), pybind11::cast<at::Tensor>(out["primitive_sequence"]), order, out_count, sequence_width, stream);
    }
    if (has_material_sequence) {
        gather_tensor_rows<int>(block_tensor(block, "material_sequence"), pybind11::cast<at::Tensor>(out["material_sequence"]), order, out_count, sequence_width, stream);
    }
    if (has_interaction_positions) {
        gather_tensor_rows<float>(block_tensor(block, "interaction_positions"), pybind11::cast<at::Tensor>(out["interaction_positions"]), order, out_count, sequence_width * 3, stream);
    }
    if (has_interaction_normals) {
        gather_tensor_rows<float>(block_tensor(block, "interaction_normals"), pybind11::cast<at::Tensor>(out["interaction_normals"]), order, out_count, sequence_width * 3, stream);
    }
    return out;
}

pybind11::dict channel_deterministic_face_groups(
    at::Tensor tri_a,
    at::Tensor normals,
    at::Tensor surface_ids,
    double quantization) {
    check_vec3_table(tri_a, "tri_a");
    check_vec3_table(normals, "normals");
    check_tensor(surface_ids, "surface_ids", at::kLong, 1);
    TORCH_CHECK(normals.sizes() == tri_a.sizes(), "normals must match tri_a");
    TORCH_CHECK(surface_ids.size(0) == tri_a.size(0), "surface_ids must match tri_a");
    TORCH_CHECK(normals.get_device() == tri_a.get_device(), "normals must share tri_a device");
    TORCH_CHECK(surface_ids.get_device() == tri_a.get_device(), "surface_ids must share tri_a device");
    TORCH_CHECK(quantization > 0.0, "quantization must be positive");

    const int64_t face_count = tri_a.size(0);
    auto int_options = tri_a.options().dtype(at::kInt);
    auto long_options = tri_a.options().dtype(at::kLong);
    pybind11::dict out;
    if (face_count == 0) {
        out["face_group_id"] = at::empty({0}, int_options);
        out["representative_faces"] = at::empty({0}, long_options);
        out["surface_group_id"] = at::empty({0}, int_options);
        out["surface_group_size"] = at::empty({0}, int_options);
        out["surface_group_members"] = at::empty({0}, int_options);
        out["group_count"] = static_cast<int64_t>(0);
        return out;
    }

    const int device = tri_a.get_device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    auto key_matrix = at::empty({face_count, kFaceGroupKeyWidth}, long_options);
    auto order = at::empty({face_count}, long_options);
    auto sort_keys = at::empty({face_count}, long_options);
    deterministic_face_group_keys_kernel<<<::channel::launch_blocks(face_count, kPathBlockSize),
                                           kPathBlockSize, 0, stream>>>(
        face_count,
        tri_a.data_ptr<float>(),
        normals.data_ptr<float>(),
        surface_ids.data_ptr<int64_t>(),
        static_cast<float>(1.0 / quantization),
        key_matrix.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    deterministic_order_init_kernel<<<::channel::launch_blocks(face_count, kPathBlockSize),
                                      kPathBlockSize, 0, stream>>>(
        face_count,
        order.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int64_t column = kFaceGroupKeyWidth - 1; column >= 0; --column) {
        deterministic_face_group_sort_key_kernel<<<
            ::channel::launch_blocks(face_count, kPathBlockSize), kPathBlockSize, 0, stream>>>(
            face_count,
            column,
            key_matrix.data_ptr<int64_t>(),
            order.data_ptr<int64_t>(),
            sort_keys.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        thrust::stable_sort_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(sort_keys.data_ptr<int64_t>()),
            thrust::device_pointer_cast(sort_keys.data_ptr<int64_t>() + face_count),
            thrust::device_pointer_cast(order.data_ptr<int64_t>()));
    }

    auto flags = at::empty({face_count}, int_options);
    auto group_offsets = at::empty({face_count + 1}, int_options);
    deterministic_face_group_flags_kernel<<<::channel::launch_blocks(face_count, kPathBlockSize),
                                            kPathBlockSize, 0, stream>>>(
        face_count,
        key_matrix.data_ptr<int64_t>(),
        order.data_ptr<int64_t>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    channel::scan::exclusive_flag_offsets(flags, group_offsets, face_count, stream);

    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + face_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        group_offsets.data_ptr<int>() + face_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t group_count = static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset);

    auto face_group_id = at::empty({face_count}, int_options);
    auto representative_faces = at::empty({group_count}, long_options);
    auto surface_group_size = at::empty({group_count}, int_options);
    C10_CUDA_CHECK(cudaMemsetAsync(surface_group_size.data_ptr<int>(), 0, group_count * sizeof(int), stream));
    deterministic_face_group_assign_kernel<<<::channel::launch_blocks(face_count, kPathBlockSize),
                                             kPathBlockSize, 0, stream>>>(
        face_count,
        order.data_ptr<int64_t>(),
        flags.data_ptr<int>(),
        group_offsets.data_ptr<int>(),
        face_group_id.data_ptr<int>(),
        representative_faces.data_ptr<int64_t>(),
        surface_group_size.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto max_iter = thrust::max_element(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(surface_group_size.data_ptr<int>()),
        thrust::device_pointer_cast(surface_group_size.data_ptr<int>() + group_count));
    int max_group_size = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &max_group_size,
        max_iter.get(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto member_offsets = at::empty({group_count}, int_options);
    auto surface_group_members = at::empty({group_count * static_cast<int64_t>(max_group_size)}, int_options);
    C10_CUDA_CHECK(cudaMemsetAsync(member_offsets.data_ptr<int>(), 0, group_count * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        surface_group_members.data_ptr<int>(),
        0xff,
        surface_group_members.numel() * sizeof(int),
        stream));
    deterministic_face_group_members_kernel<<<::channel::launch_blocks(face_count, kPathBlockSize),
                                              kPathBlockSize, 0, stream>>>(
        face_count,
        max_group_size,
        order.data_ptr<int64_t>(),
        flags.data_ptr<int>(),
        group_offsets.data_ptr<int>(),
        member_offsets.data_ptr<int>(),
        surface_group_members.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    out["face_group_id"] = face_group_id;
    out["representative_faces"] = representative_faces;
    out["surface_group_id"] = face_group_id;
    out["surface_group_size"] = surface_group_size;
    out["surface_group_members"] = surface_group_members;
    out["group_count"] = group_count;
    return out;
}

// ==== Section: Path result padding ====
namespace {

constexpr unsigned long long kNoFirstRow = ~0ULL;

__global__ void path_pack_first_row_kernel(int64_t row_count, int64_t num_tx,
                                           const int* __restrict__ tx_id,
                                           const int* __restrict__ rx_id,
                                           const int64_t* __restrict__ pair_offsets,
                                           unsigned long long* __restrict__ first_row,
                                           unsigned long long* __restrict__ max_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < row_count;
         row += stride) {
        const int64_t pair = static_cast<int64_t>(rx_id[row]) * num_tx + tx_id[row];
        atomicMin(first_row + pair, static_cast<unsigned long long>(row));
        const int64_t count = pair_offsets[pair + 1] - pair_offsets[pair];
        atomicMax(max_count, static_cast<unsigned long long>(count));
    }
}

// Matches torch.linalg.vector_norm over a contiguous vec3 on CUDA, which sums
// (x*x + z*z) + y*y without contraction, then the clamp/acos/atan2 sequence of
// the endpoint angle helper, NaN propagating as torch.clamp does.
__device__ void path_endpoint_angles(float x, float y, float z, float* theta, float* phi) {
    const float norm =
        sqrtf(__fadd_rn(__fadd_rn(__fmul_rn(x, x), __fmul_rn(z, z)), __fmul_rn(y, y)));
    const float tiny = 1.17549435e-38f;
    const float safe = norm != norm ? norm : (norm < tiny ? tiny : norm);
    float cosine = __fdiv_rn(z, safe);
    if (cosine == cosine) {
        cosine = cosine < -1.0f ? -1.0f : (cosine > 1.0f ? 1.0f : cosine);
    }
    *theta = acosf(cosine);
    *phi = atan2f(y, x);
}

__global__ void path_pack_padded_kernel(
    int64_t slot_count, int64_t width, int64_t depth_width, int64_t num_tx,
    const unsigned long long* __restrict__ first_row, const int64_t* __restrict__ pair_offsets,
    const int* __restrict__ tx_id, const int* __restrict__ rx_id, const int* __restrict__ depth,
    const int* __restrict__ component_id, const int* __restrict__ edge_id,
    const int* __restrict__ interaction_type, const int* __restrict__ primitive_sequence,
    const int* __restrict__ material_sequence, const float* __restrict__ positions,
    const float* __restrict__ normals, const c10::complex<float>* __restrict__ coefficient,
    const float* __restrict__ delay, const c10::complex<float>* __restrict__ field_xyz,
    const float* __restrict__ field_direction, const float* __restrict__ tx_positions,
    const float* __restrict__ rx_positions, c10::complex<float>* __restrict__ out_a,
    float* __restrict__ out_tau, float* __restrict__ out_theta_t, float* __restrict__ out_phi_t,
    float* __restrict__ out_theta_r, float* __restrict__ out_phi_r, bool* __restrict__ out_valid,
    int* __restrict__ out_interaction_type, int* __restrict__ out_primitive_id,
    int* __restrict__ out_material_id, float* __restrict__ out_position,
    float* __restrict__ out_normal, c10::complex<float>* __restrict__ out_field_xyz,
    float* __restrict__ out_field_direction, int* __restrict__ out_num_paths) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t slot = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         slot < slot_count; slot += stride) {
        const int64_t pair = slot / width;
        const int64_t rank = slot - pair * width;
        const int64_t count = pair_offsets[pair + 1] - pair_offsets[pair];
        if (rank == 0) {
            out_num_paths[pair] = static_cast<int>(count);
        }
        const unsigned long long first = first_row[pair];
        if (rank >= count || first == kNoFirstRow) {
            out_a[slot] = c10::complex<float>(0.0f, 0.0f);
            out_tau[slot] = -1.0f;
            out_theta_t[slot] = 0.0f;
            out_phi_t[slot] = 0.0f;
            out_theta_r[slot] = 0.0f;
            out_phi_r[slot] = 0.0f;
            out_valid[slot] = false;
            for (int64_t k = 0; k < depth_width; ++k) {
                const int64_t at = slot * depth_width + k;
                out_interaction_type[at] = 0;
                out_primitive_id[at] = -1;
                out_material_id[at] = -1;
                for (int c = 0; c < 3; ++c) {
                    out_position[at * 3 + c] = 0.0f;
                    out_normal[at * 3 + c] = 0.0f;
                }
            }
            for (int c = 0; c < 3; ++c) {
                out_field_xyz[slot * 3 + c] = c10::complex<float>(0.0f, 0.0f);
                out_field_direction[slot * 3 + c] = 0.0f;
            }
            continue;
        }
        const int64_t row = static_cast<int64_t>(first) + rank;
        const float* tx = tx_positions + static_cast<int64_t>(tx_id[row]) * 3;
        const float* rx = rx_positions + static_cast<int64_t>(rx_id[row]) * 3;
        float departure[3];
        float arrival[3];
        for (int c = 0; c < 3; ++c) {
            departure[c] = __fsub_rn(rx[c], tx[c]);
            arrival[c] = departure[c];
        }
        const int row_depth = depth[row];
        if (depth_width > 0 && row_depth > 0) {
            int64_t last = row_depth - 1;
            last = last < depth_width ? last : depth_width - 1;
            const float* first_point = positions + row * depth_width * 3;
            const float* last_point = positions + (row * depth_width + last) * 3;
            for (int c = 0; c < 3; ++c) {
                departure[c] = __fsub_rn(first_point[c], tx[c]);
                arrival[c] = __fsub_rn(rx[c], last_point[c]);
            }
        }
        path_endpoint_angles(departure[0], departure[1], departure[2], out_theta_t + slot,
                             out_phi_t + slot);
        path_endpoint_angles(-arrival[0], -arrival[1], -arrival[2], out_theta_r + slot,
                             out_phi_r + slot);
        out_a[slot] = coefficient[row];
        out_tau[slot] = delay[row];
        out_valid[slot] = true;
        const bool diffraction_row =
            component_id[row] == channel::kComponentDiffraction && row_depth > 0;
        for (int64_t k = 0; k < depth_width; ++k) {
            const int64_t at = slot * depth_width + k;
            const int64_t source = row * depth_width + k;
            const int type = interaction_type[source];
            out_interaction_type[at] = type;
            out_primitive_id[at] =
                (k == 0 && diffraction_row) ? edge_id[row] : primitive_sequence[source];
            out_material_id[at] = material_sequence[source];
            for (int c = 0; c < 3; ++c) {
                const float normal = normals[source * 3 + c];
                out_position[at * 3 + c] = positions[source * 3 + c];
                out_normal[at * 3 + c] = (type == 2 && !isfinite(normal)) ? 0.0f : normal;
            }
        }
        for (int c = 0; c < 3; ++c) {
            out_field_xyz[slot * 3 + c] = field_xyz[row * 3 + c];
            out_field_direction[slot * 3 + c] = field_direction[row * 3 + c];
        }
    }
}

} // namespace

std::vector<at::Tensor> channel_path_pack_padded_cuda(
    at::Tensor tx_id, at::Tensor rx_id, at::Tensor depth, at::Tensor component_id,
    at::Tensor edge_id, at::Tensor interaction_type, at::Tensor primitive_sequence,
    at::Tensor material_sequence, at::Tensor positions, at::Tensor normals, at::Tensor coefficient,
    at::Tensor delay, at::Tensor field_xyz, at::Tensor field_direction, at::Tensor tx_positions,
    at::Tensor rx_positions, at::Tensor pair_offsets, int64_t width) {
    const int64_t row_count = tx_id.size(0);
    const int64_t depth_width = interaction_type.size(1);
    check_vec3_table(tx_positions, "tx_positions");
    check_vec3_table(rx_positions, "rx_positions");
    const int64_t num_tx = tx_positions.size(0);
    const int64_t num_rx = rx_positions.size(0);
    const int64_t pair_count = num_tx * num_rx;
    for (const auto& [value, name] :
         std::array<std::pair<const at::Tensor*, const char*>, 5>{{{&tx_id, "tx_id"},
                                                                   {&rx_id, "rx_id"},
                                                                   {&depth, "depth"},
                                                                   {&component_id, "component_id"},
                                                                   {&edge_id, "edge_id"}}}) {
        check_tensor(*value, name, at::kInt, 1);
        TORCH_CHECK(value->size(0) == row_count, name, " must have one entry per row");
    }
    for (const auto& [value, name] : std::array<std::pair<const at::Tensor*, const char*>, 3>{
             {{&interaction_type, "interaction_type"},
              {&primitive_sequence, "primitive_sequence"},
              {&material_sequence, "material_sequence"}}}) {
        check_tensor(*value, name, at::kInt, 2);
        TORCH_CHECK(value->size(0) == row_count && value->size(1) == depth_width, name,
                    " must have shape (row, depth)");
    }
    for (const auto& [value, name] : std::array<std::pair<const at::Tensor*, const char*>, 2>{
             {{&positions, "interaction_positions"}, {&normals, "interaction_normals"}}}) {
        check_tensor(*value, name, at::kFloat, 3);
        TORCH_CHECK(value->size(0) == row_count && value->size(1) == depth_width &&
                        value->size(2) == 3,
                    name, " must have shape (row, depth, 3)");
    }
    check_tensor(coefficient, "coefficient", at::kComplexFloat, 1);
    check_tensor(delay, "delay_s", at::kFloat, 1);
    check_tensor(field_xyz, "field_xyz", at::kComplexFloat, 2);
    check_tensor(field_direction, "field_direction", at::kFloat, 2);
    check_tensor(pair_offsets, "pair_offsets", at::kLong, 1);
    TORCH_CHECK(coefficient.size(0) == row_count && delay.size(0) == row_count,
                "row fields must match row count");
    TORCH_CHECK(field_xyz.size(0) == row_count && field_xyz.size(1) == 3 &&
                    field_direction.size(0) == row_count && field_direction.size(1) == 3,
                "field_xyz and field_direction must have shape (row, 3)");
    TORCH_CHECK(pair_offsets.size(0) == pair_count + 1,
                "pair_offsets must have pair_count + 1 entries");
    TORCH_CHECK(width >= -1, "width must be -1 (measure) or non-negative");
    const int device = tx_id.get_device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    auto long_options = pair_offsets.options();
    auto first_row = at::empty({std::max<int64_t>(pair_count, 1)}, long_options);
    auto max_count = at::empty({1}, long_options);
    C10_CUDA_CHECK(cudaMemsetAsync(first_row.data_ptr<int64_t>(), 0xFF,
                                   first_row.numel() * sizeof(int64_t), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(max_count.data_ptr<int64_t>(), 0, sizeof(int64_t), stream));
    if (row_count > 0) {
        path_pack_first_row_kernel<<<::channel::launch_blocks(row_count, kPathBlockSize),
                                     kPathBlockSize, 0,
                                     stream>>>(
            row_count, num_tx, tx_id.data_ptr<int>(), rx_id.data_ptr<int>(),
            pair_offsets.data_ptr<int64_t>(),
            reinterpret_cast<unsigned long long*>(first_row.data_ptr<int64_t>()),
            reinterpret_cast<unsigned long long*>(max_count.data_ptr<int64_t>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (width < 0) {
        // The padded width is the largest per-pair path count; reading it is the one host
        // synchronization of padded packing, taken only when the caller cannot derive it.
        int64_t host_width = 0;
        if (row_count > 0) {
            C10_CUDA_CHECK(cudaMemcpyAsync(&host_width, max_count.data_ptr<int64_t>(),
                                           sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        width = host_width;
    }
    auto float_options = tx_positions.options();
    auto int_options = tx_id.options();
    const int64_t slot_count = pair_count * width;
    auto out_a = at::empty({slot_count}, coefficient.options());
    auto out_tau = at::empty({slot_count}, float_options);
    auto out_theta_t = at::empty({slot_count}, float_options);
    auto out_phi_t = at::empty({slot_count}, float_options);
    auto out_theta_r = at::empty({slot_count}, float_options);
    auto out_phi_r = at::empty({slot_count}, float_options);
    auto out_valid = at::empty({slot_count}, float_options.dtype(at::kBool));
    auto out_interaction_type = at::empty({slot_count, depth_width}, int_options);
    auto out_primitive_id = at::empty({slot_count, depth_width}, int_options);
    auto out_material_id = at::empty({slot_count, depth_width}, int_options);
    auto out_position = at::empty({slot_count, depth_width, 3}, float_options);
    auto out_normal = at::empty({slot_count, depth_width, 3}, float_options);
    auto out_field_xyz = at::empty({slot_count, 3}, coefficient.options());
    auto out_field_direction = at::empty({slot_count, 3}, float_options);
    auto out_num_paths = channel::empty_zero_cuda({pair_count}, int_options, stream);
    if (slot_count > 0) {
        path_pack_padded_kernel<<<::channel::launch_blocks(slot_count, kPathBlockSize),
                                  kPathBlockSize, 0,
                                  stream>>>(
            slot_count, width, depth_width, num_tx,
            reinterpret_cast<const unsigned long long*>(first_row.data_ptr<int64_t>()),
            pair_offsets.data_ptr<int64_t>(), tx_id.data_ptr<int>(), rx_id.data_ptr<int>(),
            depth.data_ptr<int>(), component_id.data_ptr<int>(), edge_id.data_ptr<int>(),
            interaction_type.data_ptr<int>(), primitive_sequence.data_ptr<int>(),
            material_sequence.data_ptr<int>(), positions.data_ptr<float>(),
            normals.data_ptr<float>(), coefficient.data_ptr<c10::complex<float>>(),
            delay.data_ptr<float>(), field_xyz.data_ptr<c10::complex<float>>(),
            field_direction.data_ptr<float>(), tx_positions.data_ptr<float>(),
            rx_positions.data_ptr<float>(), out_a.data_ptr<c10::complex<float>>(),
            out_tau.data_ptr<float>(), out_theta_t.data_ptr<float>(), out_phi_t.data_ptr<float>(),
            out_theta_r.data_ptr<float>(), out_phi_r.data_ptr<float>(), out_valid.data_ptr<bool>(),
            out_interaction_type.data_ptr<int>(), out_primitive_id.data_ptr<int>(),
            out_material_id.data_ptr<int>(), out_position.data_ptr<float>(),
            out_normal.data_ptr<float>(), out_field_xyz.data_ptr<c10::complex<float>>(),
            out_field_direction.data_ptr<float>(), out_num_paths.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {out_a,        out_tau,    out_theta_t,          out_phi_t,           out_theta_r,
            out_phi_r,    out_valid,  out_interaction_type, out_primitive_id,    out_material_id,
            out_position, out_normal, out_field_xyz,        out_field_direction, out_num_paths};
}
