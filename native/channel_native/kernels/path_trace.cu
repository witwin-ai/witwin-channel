#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
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

#include "path_compaction_common.cuh"

namespace {

constexpr double kLightSpeedMetersPerSecond = 299792458.0;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kReflectionBaryEps = 1.0e-4f;
constexpr float kReflectionVisEps = 1.0e-4f;

__device__ float cn_neg_kd_phase(float k, float d) {
    // Reduce k*d mod 2*pi in double: the f32 product loses ~k*d*2^-24 of
    // phase, which shifts coherent nulls at mmWave ranges.
    const double kd = fmod(static_cast<double>(k) * static_cast<double>(d), 6.283185307179586476925287);
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
        const float wavelength = static_cast<float>(kLightSpeedMetersPerSecond) / frequency_hz;
        const float phase = cn_neg_kd_phase(2.0f * kPi / wavelength, length);

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

__global__ void deterministic_topology_base_fields_kernel(
    int64_t count,
    const int* __restrict__ rx_id,
    const float* __restrict__ path_length,
    const float* __restrict__ delay,
    const float* __restrict__ path_gain,
    int tx_index,
    int component_id,
    const int* __restrict__ depth_source,
    int depth_source_count,
    int depth_value,
    const int* __restrict__ primitive_source,
    int primitive_source_count,
    int primitive_value,
    const int* __restrict__ edge_source,
    int edge_source_count,
    int edge_value,
    bool* __restrict__ out_valid,
    int* __restrict__ out_tx_id,
    int* __restrict__ out_rx_id,
    int* __restrict__ out_depth,
    int* __restrict__ out_component_id,
    int* __restrict__ out_primitive_id,
    int* __restrict__ out_edge_id,
    float* __restrict__ out_path_length,
    float* __restrict__ out_delay,
    float* __restrict__ out_path_gain) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        out_valid[index] = true;
        out_tx_id[index] = tx_index;
        out_rx_id[index] = rx_id[index];
        out_depth[index] = depth_source_count == count ? depth_source[index] : depth_value;
        out_component_id[index] = component_id;
        out_primitive_id[index] = primitive_source_count == count ? primitive_source[index] : primitive_value;
        out_edge_id[index] = edge_source_count == count ? edge_source[index] : edge_value;
        out_path_length[index] = path_length[index];
        out_delay[index] = delay[index];
        out_path_gain[index] = path_gain[index];
    }
}

__global__ void deterministic_repeat_range_kernel(
    int64_t count,
    int start,
    int repeats,
    int* __restrict__ out) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        out[index] = start + static_cast<int>(index / repeats);
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

template <typename index_t>
__global__ void deterministic_reflection_epc_input_batch_kernel(
    int64_t pair_count,
    int64_t sequence_count,
    int64_t depth,
    int rx_start,
    const float* __restrict__ tx,
    const float* __restrict__ rx_positions,
    const index_t* __restrict__ sequences,
    const float* __restrict__ tri_a,
    const float* __restrict__ normals,
    float* __restrict__ tx_batch,
    float* __restrict__ rx_batch,
    int* __restrict__ rx_indices,
    int* __restrict__ sequence_batch,
    float* __restrict__ direct_plane_points,
    float* __restrict__ direct_plane_normals) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pair_count;
         pair += stride) {
        const int64_t rx_local = pair / sequence_count;
        const int64_t sequence_index = pair - rx_local * sequence_count;
        const int rx_id = rx_start + static_cast<int>(rx_local);
        rx_indices[pair] = rx_id;

        const int64_t vec_base = pair * 3;
        const int64_t rx_base = static_cast<int64_t>(rx_id) * 3;
        for (int axis = 0; axis < 3; ++axis) {
            tx_batch[vec_base + axis] = tx[axis];
            rx_batch[vec_base + axis] = rx_positions[rx_base + axis];
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

template <typename face_id_t>
__global__ void deterministic_face_sequence_chunk_kernel(
    int64_t row_count,
    int64_t face_count,
    int64_t depth,
    int64_t start,
    bool adjacent_distinct,
    const face_id_t* __restrict__ face_ids,
    int* __restrict__ out) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        int64_t value = start + row;
        int64_t divisor = 1;
        const int64_t base = adjacent_distinct && depth > 1 ? face_count - 1 : face_count;
        for (int64_t power = 1; power < depth; ++power) {
            divisor *= base;
        }
        int64_t previous = -1;
        for (int64_t column = 0; column < depth; ++column) {
            int64_t digit = 0;
            if (adjacent_distinct && depth > 1) {
                if (column == 0) {
                    digit = value / divisor;
                    value -= digit * divisor;
                    divisor = divisor > 1 ? divisor / base : 1;
                } else {
                    const int64_t raw_digit = value / divisor;
                    value -= raw_digit * divisor;
                    divisor = divisor > 1 ? divisor / base : 1;
                    digit = raw_digit < previous ? raw_digit : raw_digit + 1;
                }
                previous = digit;
            } else {
                digit = value / divisor;
                value -= digit * divisor;
                divisor = divisor > 1 ? divisor / face_count : 1;
            }
            const int mapped = face_ids == nullptr ? static_cast<int>(digit) : static_cast<int>(face_ids[digit]);
            out[row * depth + column] = mapped;
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

__global__ void deterministic_reflect_points_kernel(
    int64_t count,
    const float* __restrict__ points,
    const float* __restrict__ plane_points,
    const float* __restrict__ normals,
    float* __restrict__ out) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int64_t base = index * 3;
        const float dx = points[base + 0] - plane_points[base + 0];
        const float dy = points[base + 1] - plane_points[base + 1];
        const float dz = points[base + 2] - plane_points[base + 2];
        const float nx = normals[base + 0];
        const float ny = normals[base + 1];
        const float nz = normals[base + 2];
        const float distance = dx * nx + dy * ny + dz * nz;
        out[base + 0] = points[base + 0] - 2.0f * distance * nx;
        out[base + 1] = points[base + 1] - 2.0f * distance * ny;
        out[base + 2] = points[base + 2] - 2.0f * distance * nz;
    }
}

__device__ float dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ float norm3(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
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
        const float n_len = fmaxf(norm3(raw_n[0], raw_n[1], raw_n[2]), 1.0e-12f);
        const float n[3] = {raw_n[0] / n_len, raw_n[1] / n_len, raw_n[2] / n_len};

        const float tx_vec[3] = {tx_p[0] - a[0], tx_p[1] - a[1], tx_p[2] - a[2]};
        const float tx_to_plane = dot3(tx_vec, n);
        const float image[3] = {
            tx_p[0] - 2.0f * tx_to_plane * n[0],
            tx_p[1] - 2.0f * tx_to_plane * n[1],
            tx_p[2] - 2.0f * tx_to_plane * n[2],
        };
        const float line[3] = {rx_p[0] - image[0], rx_p[1] - image[1], rx_p[2] - image[2]};
        const float denom = dot3(line, n);
        if (fabsf(denom) <= 1.0e-8f) {
            continue;
        }
        const float plane_vec[3] = {a[0] - image[0], a[1] - image[1], a[2] - image[2]};
        const float s = dot3(plane_vec, n) / denom;
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
        const float d0_len = norm3(tx_dx, tx_dy, tx_dz);
        const float d1_len = norm3(rx_dx, rx_dy, rx_dz);
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
        delay[index] = total_length / static_cast<float>(kLightSpeedMetersPerSecond);
        path_gain[index] = tx_power[tx] * gain * scale * scale;
    }
}

__global__ void path_copy_block_kernel(
    int64_t count,
    int64_t offset,
    const bool* __restrict__ valid,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    const int* __restrict__ depth,
    const int* __restrict__ component_id,
    const int* __restrict__ primitive_id,
    const int* __restrict__ edge_id,
    const float* __restrict__ path_length,
    const float* __restrict__ delay,
    const float* __restrict__ path_gain,
    bool* __restrict__ out_valid,
    int* __restrict__ out_tx_id,
    int* __restrict__ out_rx_id,
    int* __restrict__ out_depth,
    int* __restrict__ out_component_id,
    int* __restrict__ out_primitive_id,
    int* __restrict__ out_edge_id,
    float* __restrict__ out_path_length,
    float* __restrict__ out_delay,
    float* __restrict__ out_path_gain) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int64_t dst = offset + index;
        out_valid[dst] = valid[index];
        out_tx_id[dst] = tx_id[index];
        out_rx_id[dst] = rx_id[index];
        out_depth[dst] = depth[index];
        out_component_id[dst] = component_id[index];
        out_primitive_id[dst] = primitive_id[index];
        out_edge_id[dst] = edge_id[index];
        out_path_length[dst] = path_length[index];
        out_delay[dst] = delay[index];
        out_path_gain[dst] = path_gain[index];
    }
}

__global__ void path_sort_key_kernel(
    int64_t count,
    int64_t tx_count,
    int64_t max_depth,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    const int* __restrict__ depth,
    const int* __restrict__ component_id,
    int64_t* __restrict__ keys,
    int64_t* __restrict__ order) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t tx_stride = tx_count > 0 ? tx_count : 1;
    const int64_t depth_stride = (max_depth > 1 ? max_depth : 1) + 1;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int64_t key =
            (((static_cast<int64_t>(rx_id[index]) * tx_stride + static_cast<int64_t>(tx_id[index])) *
              depth_stride +
              static_cast<int64_t>(depth[index])) *
                 3 +
             static_cast<int64_t>(component_id[index]));
        keys[index] = key;
        order[index] = index;
    }
}

__global__ void path_gather_kernel(
    int64_t count,
    const int64_t* __restrict__ order,
    const bool* __restrict__ valid,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
    const int* __restrict__ depth,
    const int* __restrict__ component_id,
    const int* __restrict__ primitive_id,
    const int* __restrict__ edge_id,
    const float* __restrict__ path_length,
    const float* __restrict__ delay,
    const float* __restrict__ path_gain,
    bool* __restrict__ out_valid,
    int* __restrict__ out_tx_id,
    int* __restrict__ out_rx_id,
    int* __restrict__ out_depth,
    int* __restrict__ out_component_id,
    int* __restrict__ out_primitive_id,
    int* __restrict__ out_edge_id,
    float* __restrict__ out_path_length,
    float* __restrict__ out_delay,
    float* __restrict__ out_path_gain) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int64_t src = order[index];
        out_valid[index] = valid[src];
        out_tx_id[index] = tx_id[src];
        out_rx_id[index] = rx_id[src];
        out_depth[index] = depth[src];
        out_component_id[index] = component_id[src];
        out_primitive_id[index] = primitive_id[src];
        out_edge_id[index] = edge_id[src];
        out_path_length[index] = path_length[src];
        out_delay[index] = delay[src];
        out_path_gain[index] = path_gain[src];
    }
}

}  // namespace

at::Tensor cn_path_concat_vec3_cuda(std::vector<at::Tensor> blocks) {
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
            path_concat_vec3_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> cn_path_los_visibility_inputs_cuda(
    at::Tensor tx_positions,
    at::Tensor rx_positions,
    at::Tensor tx_id,
    at::Tensor rx_id) {
    check_vec3_table(tx_positions, "tx_positions");
    check_vec3_table(rx_positions, "rx_positions");
    check_cuda_tensor(tx_id, "tx_id", at::kInt, 1);
    check_cuda_tensor(rx_id, "rx_id", at::kInt, 1);
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
        path_los_visibility_inputs_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
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

pybind11::dict cn_deterministic_los_topology_block_all_visible(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor path_length,
    at::Tensor delay,
    at::Tensor path_gain,
    double frequency_hz,
    int64_t sequence_width) {
    check_cuda_tensor(tx_id, "tx_id", at::kInt, 1);
    check_cuda_tensor(rx_id, "rx_id", at::kInt, 1);
    check_cuda_tensor(path_length, "path_length_m", at::kFloat, 1);
    check_cuda_tensor(delay, "delay_s", at::kFloat, 1);
    check_cuda_tensor(path_gain, "path_gain", at::kFloat, 1);
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
    deterministic_los_topology_all_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
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

pybind11::dict cn_deterministic_topology_default_fields(at::Tensor reference) {
    check_cuda_tensor(reference, "reference", at::kFloat, 1);
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
        deterministic_topology_default_fields_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
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

pybind11::dict cn_deterministic_pad_topology_sequences(
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
    check_cuda_tensor(depth, "depth", at::kInt, 1);
    check_cuda_tensor(primitive_id, "primitive_id", at::kInt, 1);
    check_cuda_tensor(material_id, "material_id", at::kInt, 1);
    check_vec3_table(interaction_position, "interaction_position");
    check_vec3_table(interaction_normal, "interaction_normal");
    check_cuda_tensor(primitive_sequence, "primitive_sequence", at::kInt, 2);
    check_cuda_tensor(material_sequence, "material_sequence", at::kInt, 2);
    check_cuda_tensor(interaction_positions, "interaction_positions", at::kFloat, 3);
    check_cuda_tensor(interaction_normals, "interaction_normals", at::kFloat, 3);
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
        deterministic_pad_topology_sequences_kernel<<<static_cast<int>(launch_blocks(total)), kPathBlockSize, 0, stream>>>(
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

pybind11::dict cn_deterministic_topology_base_fields(
    at::Tensor rx_id,
    at::Tensor path_length,
    at::Tensor delay,
    at::Tensor path_gain,
    int64_t tx_index,
    int64_t component_id,
    at::Tensor depth_source,
    int64_t depth_value,
    at::Tensor primitive_source,
    int64_t primitive_value,
    at::Tensor edge_source,
    int64_t edge_value) {
    check_cuda_tensor(rx_id, "rx_id", at::kInt, 1);
    check_cuda_tensor(path_length, "path_length_m", at::kFloat, 1);
    check_cuda_tensor(delay, "delay_s", at::kFloat, 1);
    check_cuda_tensor(path_gain, "path_gain", at::kFloat, 1);
    check_cuda_tensor(depth_source, "depth_source", at::kInt, 1);
    check_cuda_tensor(primitive_source, "primitive_source", at::kInt, 1);
    check_cuda_tensor(edge_source, "edge_source", at::kInt, 1);
    const int64_t count = rx_id.size(0);
    const int device = rx_id.get_device();
    TORCH_CHECK(path_length.size(0) == count, "path_length_m must match rx_id");
    TORCH_CHECK(delay.size(0) == count, "delay_s must match rx_id");
    TORCH_CHECK(path_gain.size(0) == count, "path_gain must match rx_id");
    TORCH_CHECK(depth_source.size(0) == 0 || depth_source.size(0) == count, "depth_source must be empty or match rx_id");
    TORCH_CHECK(primitive_source.size(0) == 0 || primitive_source.size(0) == count, "primitive_source must be empty or match rx_id");
    TORCH_CHECK(edge_source.size(0) == 0 || edge_source.size(0) == count, "edge_source must be empty or match rx_id");
    TORCH_CHECK(path_length.get_device() == device, "path_length_m must share rx_id device");
    TORCH_CHECK(delay.get_device() == device, "delay_s must share rx_id device");
    TORCH_CHECK(path_gain.get_device() == device, "path_gain must share rx_id device");
    TORCH_CHECK(depth_source.get_device() == device, "depth_source must share rx_id device");
    TORCH_CHECK(primitive_source.get_device() == device, "primitive_source must share rx_id device");
    TORCH_CHECK(edge_source.get_device() == device, "edge_source must share rx_id device");

    auto bool_options = rx_id.options().dtype(at::kBool);
    auto int_options = rx_id.options().dtype(at::kInt);
    auto float_options = path_gain.options().dtype(at::kFloat);
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
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        deterministic_topology_base_fields_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
            count,
            rx_id.data_ptr<int>(),
            path_length.data_ptr<float>(),
            delay.data_ptr<float>(),
            path_gain.data_ptr<float>(),
            static_cast<int>(tx_index),
            static_cast<int>(component_id),
            depth_source.data_ptr<int>(),
            static_cast<int>(depth_source.size(0)),
            static_cast<int>(depth_value),
            primitive_source.data_ptr<int>(),
            static_cast<int>(primitive_source.size(0)),
            static_cast<int>(primitive_value),
            edge_source.data_ptr<int>(),
            static_cast<int>(edge_source.size(0)),
            static_cast<int>(edge_value),
            out_valid.data_ptr<bool>(),
            out_tx_id.data_ptr<int>(),
            out_rx_id.data_ptr<int>(),
            out_depth.data_ptr<int>(),
            out_component_id.data_ptr<int>(),
            out_primitive_id.data_ptr<int>(),
            out_edge_id.data_ptr<int>(),
            out_path_length.data_ptr<float>(),
            out_delay.data_ptr<float>(),
            out_path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

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
    return out;
}

at::Tensor cn_deterministic_repeat_range(at::Tensor reference, int64_t start, int64_t end, int64_t repeats) {
    check_cuda_tensor(reference, "reference", at::kFloat, 1);
    TORCH_CHECK(end >= start, "end must be greater than or equal to start");
    TORCH_CHECK(repeats > 0, "repeats must be positive");
    const int64_t range_count = end - start;
    const int64_t out_count = range_count * repeats;
    auto out = at::empty({out_count}, reference.options().dtype(at::kInt));
    if (out_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        deterministic_repeat_range_kernel<<<static_cast<int>(launch_blocks(out_count)), kPathBlockSize, 0, stream>>>(
            out_count,
            static_cast<int>(start),
            static_cast<int>(repeats),
            out.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor cn_deterministic_face_anchor_points(at::Tensor vertices, at::Tensor faces) {
    check_vec3_table(vertices, "vertices");
    check_cuda_tensor(faces, "faces", at::kInt, 2);
    TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
    TORCH_CHECK(faces.get_device() == vertices.get_device(), "faces must share vertices device");
    const int64_t face_count = faces.size(0);
    auto out = at::empty({face_count, 3}, vertices.options().dtype(at::kFloat));
    if (face_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
        deterministic_face_anchor_points_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
            face_count,
            vertices.data_ptr<float>(),
            faces.data_ptr<int>(),
            out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

pybind11::dict cn_deterministic_reflection_epc_input_batch(
    at::Tensor tx,
    at::Tensor rx_positions,
    at::Tensor sequences,
    at::Tensor tri_a,
    at::Tensor normals,
    int64_t rx_start,
    int64_t rx_end) {
    check_cuda_tensor(tx, "tx", at::kFloat, 1);
    TORCH_CHECK(tx.size(0) == 3, "tx must have shape (3,)");
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
    const int device = tx.get_device();
    TORCH_CHECK(rx_positions.get_device() == device, "rx_positions must share tx device");
    TORCH_CHECK(sequences.get_device() == device, "sequences must share tx device");
    TORCH_CHECK(tri_a.get_device() == device, "tri_a must share tx device");
    TORCH_CHECK(normals.get_device() == device, "normals must share tx device");
    TORCH_CHECK(rx_start >= 0, "rx_start must be non-negative");
    TORCH_CHECK(rx_end >= rx_start, "rx_end must be greater than or equal to rx_start");
    TORCH_CHECK(rx_end <= rx_positions.size(0), "rx_end is out of range");

    const int64_t rx_count = rx_end - rx_start;
    const int64_t sequence_count = sequences.size(0);
    const int64_t depth = sequences.size(1);
    const int64_t pair_count = rx_count * sequence_count;
    auto float_options = tx.options().dtype(at::kFloat);
    auto int_options = tx.options().dtype(at::kInt);

    pybind11::dict out;
    out["tx_batch"] = at::empty({pair_count, 3}, float_options);
    out["rx_batch"] = at::empty({pair_count, 3}, float_options);
    out["rx_indices"] = at::empty({pair_count}, int_options);
    out["sequence_batch"] = at::empty({pair_count, depth}, int_options);
    out["direct_plane_points"] = at::empty({pair_count, depth, 3}, float_options);
    out["direct_plane_normals"] = at::empty({pair_count, depth, 3}, float_options);

    if (pair_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        if (sequences.scalar_type() == at::kLong) {
            deterministic_reflection_epc_input_batch_kernel<int64_t><<<
                static_cast<int>(launch_blocks(pair_count)),
                kPathBlockSize,
                0,
                stream>>>(
                pair_count,
                sequence_count,
                depth,
                static_cast<int>(rx_start),
                tx.data_ptr<float>(),
                rx_positions.data_ptr<float>(),
                sequences.data_ptr<int64_t>(),
                tri_a.data_ptr<float>(),
                normals.data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["tx_batch"]).data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["rx_batch"]).data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["rx_indices"]).data_ptr<int>(),
                pybind11::cast<at::Tensor>(out["sequence_batch"]).data_ptr<int>(),
                pybind11::cast<at::Tensor>(out["direct_plane_points"]).data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["direct_plane_normals"]).data_ptr<float>());
        } else {
            deterministic_reflection_epc_input_batch_kernel<int><<<
                static_cast<int>(launch_blocks(pair_count)),
                kPathBlockSize,
                0,
                stream>>>(
                pair_count,
                sequence_count,
                depth,
                static_cast<int>(rx_start),
                tx.data_ptr<float>(),
                rx_positions.data_ptr<float>(),
                sequences.data_ptr<int>(),
                tri_a.data_ptr<float>(),
                normals.data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["tx_batch"]).data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["rx_batch"]).data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["rx_indices"]).data_ptr<int>(),
                pybind11::cast<at::Tensor>(out["sequence_batch"]).data_ptr<int>(),
                pybind11::cast<at::Tensor>(out["direct_plane_points"]).data_ptr<float>(),
                pybind11::cast<at::Tensor>(out["direct_plane_normals"]).data_ptr<float>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor cn_deterministic_face_sequence_chunk(
    at::Tensor reference,
    int64_t face_count,
    int64_t depth,
    int64_t start,
    int64_t end,
    bool adjacent_distinct) {
    TORCH_CHECK(reference.is_cuda(), "reference must be a CUDA tensor");
    TORCH_CHECK(face_count > 0, "face_count must be positive");
    TORCH_CHECK(depth > 0, "depth must be positive");
    TORCH_CHECK(start >= 0, "start must be non-negative");
    TORCH_CHECK(end >= start, "end must be greater than or equal to start");
    int64_t total = face_count;
    if (adjacent_distinct && depth > 1 && face_count == 1) {
        total = 0;
    } else {
        const int64_t base = adjacent_distinct && depth > 1 ? face_count - 1 : face_count;
        for (int64_t power = 1; power < depth; ++power) {
            TORCH_CHECK(total <= std::numeric_limits<int64_t>::max() / base, "face sequence count overflows int64");
            total *= base;
        }
    }
    TORCH_CHECK(end <= total, "end exceeds total face sequence count");
    const int64_t row_count = end - start;
    auto out = at::empty({row_count, depth}, reference.options().dtype(at::kInt));
    if (row_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        deterministic_face_sequence_chunk_kernel<int><<<
            static_cast<int>(launch_blocks(row_count)),
            kPathBlockSize,
            0,
            stream>>>(
            row_count,
            face_count,
            depth,
            start,
            adjacent_distinct,
            nullptr,
            out.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor cn_deterministic_mapped_face_sequence_chunk(
    at::Tensor face_ids,
    int64_t depth,
    int64_t start,
    int64_t end,
    bool adjacent_distinct) {
    TORCH_CHECK(face_ids.is_cuda(), "face_ids must be a CUDA tensor");
    TORCH_CHECK(face_ids.is_contiguous(), "face_ids must be contiguous");
    TORCH_CHECK(face_ids.dim() == 1, "face_ids must be one-dimensional");
    TORCH_CHECK(face_ids.scalar_type() == at::kLong || face_ids.scalar_type() == at::kInt, "face_ids must be int64 or int32");
    TORCH_CHECK(depth > 0, "depth must be positive");
    TORCH_CHECK(start >= 0, "start must be non-negative");
    TORCH_CHECK(end >= start, "end must be greater than or equal to start");
    const int64_t face_count = face_ids.size(0);
    TORCH_CHECK(face_count > 0, "face_ids must not be empty");
    int64_t total = face_count;
    if (adjacent_distinct && depth > 1 && face_count == 1) {
        total = 0;
    } else {
        const int64_t base = adjacent_distinct && depth > 1 ? face_count - 1 : face_count;
        for (int64_t power = 1; power < depth; ++power) {
            TORCH_CHECK(total <= std::numeric_limits<int64_t>::max() / base, "face sequence count overflows int64");
            total *= base;
        }
    }
    TORCH_CHECK(end <= total, "end exceeds total face sequence count");
    const int64_t row_count = end - start;
    auto out = at::empty({row_count, depth}, face_ids.options().dtype(at::kInt));
    if (row_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(face_ids.get_device()).stream();
        if (face_ids.scalar_type() == at::kLong) {
            deterministic_face_sequence_chunk_kernel<int64_t><<<
                static_cast<int>(launch_blocks(row_count)),
                kPathBlockSize,
                0,
                stream>>>(
                row_count,
                face_count,
                depth,
                start,
                adjacent_distinct,
                face_ids.data_ptr<int64_t>(),
                out.data_ptr<int>());
        } else {
            deterministic_face_sequence_chunk_kernel<int><<<
                static_cast<int>(launch_blocks(row_count)),
                kPathBlockSize,
                0,
                stream>>>(
                row_count,
                face_count,
                depth,
                start,
                adjacent_distinct,
                face_ids.data_ptr<int>(),
                out.data_ptr<int>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor cn_deterministic_normalize_vec3(at::Tensor values, double eps) {
    check_vec3_table(values, "values");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
    const int64_t count = values.size(0);
    auto out = at::empty_like(values);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(values.get_device()).stream();
        deterministic_normalize_vec3_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
            count,
            values.data_ptr<float>(),
            static_cast<float>(eps),
            out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor cn_deterministic_reflect_points(
    at::Tensor points,
    at::Tensor plane_points,
    at::Tensor normals) {
    check_vec3_table(points, "points");
    check_vec3_table(plane_points, "plane_points");
    check_vec3_table(normals, "normals");
    TORCH_CHECK(plane_points.sizes() == points.sizes(), "plane_points must match points");
    TORCH_CHECK(normals.sizes() == points.sizes(), "normals must match points");
    TORCH_CHECK(plane_points.get_device() == points.get_device(), "plane_points must share points device");
    TORCH_CHECK(normals.get_device() == points.get_device(), "normals must share points device");
    const int64_t count = points.size(0);
    auto out = at::empty_like(points);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(points.get_device()).stream();
        deterministic_reflect_points_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
            count,
            points.data_ptr<float>(),
            plane_points.data_ptr<float>(),
            normals.data_ptr<float>(),
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
cn_path_reflection_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor face_gain,
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor rx_positions,
    double frequency_hz) {
    check_vec3_table(vertices, "vertices");
    check_cuda_tensor(faces, "faces", at::kInt, 2);
    check_vec3_table(face_normals, "face_normals");
    check_cuda_tensor(face_gain, "face_gain", at::kFloat, 1);
    check_vec3_table(tx_positions, "tx_positions");
    check_cuda_tensor(tx_power, "tx_power", at::kFloat, 1);
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
        path_reflection_candidates_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
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
            static_cast<float>(kLightSpeedMetersPerSecond / frequency_hz),
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_path_finalize_blocks_cuda(
    std::vector<at::Tensor> valid_blocks,
    std::vector<at::Tensor> tx_id_blocks,
    std::vector<at::Tensor> rx_id_blocks,
    std::vector<at::Tensor> depth_blocks,
    std::vector<at::Tensor> component_id_blocks,
    std::vector<at::Tensor> primitive_id_blocks,
    std::vector<at::Tensor> edge_id_blocks,
    std::vector<at::Tensor> path_length_blocks,
    std::vector<at::Tensor> delay_blocks,
    std::vector<at::Tensor> path_gain_blocks,
    int64_t max_paths,
    int64_t tx_count,
    int64_t max_depth) {
    const size_t block_count = valid_blocks.size();
    TORCH_CHECK(block_count > 0, "path_finalize_blocks requires at least one block");
    TORCH_CHECK(tx_id_blocks.size() == block_count, "tx_id block count mismatch");
    TORCH_CHECK(rx_id_blocks.size() == block_count, "rx_id block count mismatch");
    TORCH_CHECK(depth_blocks.size() == block_count, "depth block count mismatch");
    TORCH_CHECK(component_id_blocks.size() == block_count, "component_id block count mismatch");
    TORCH_CHECK(primitive_id_blocks.size() == block_count, "primitive_id block count mismatch");
    TORCH_CHECK(edge_id_blocks.size() == block_count, "edge_id block count mismatch");
    TORCH_CHECK(path_length_blocks.size() == block_count, "path_length block count mismatch");
    TORCH_CHECK(delay_blocks.size() == block_count, "delay block count mismatch");
    TORCH_CHECK(path_gain_blocks.size() == block_count, "path_gain block count mismatch");
    TORCH_CHECK(max_paths >= -1, "max_paths must be -1 or non-negative");
    TORCH_CHECK(tx_count >= 0, "tx_count must be non-negative");
    TORCH_CHECK(max_depth >= 0, "max_depth must be non-negative");

    const int device = valid_blocks[0].get_device();
    int64_t total = 0;
    for (size_t i = 0; i < block_count; ++i) {
        check_path_block_shapes(
            valid_blocks[i],
            tx_id_blocks[i],
            rx_id_blocks[i],
            depth_blocks[i],
            component_id_blocks[i],
            primitive_id_blocks[i],
            edge_id_blocks[i],
            path_length_blocks[i],
            delay_blocks[i],
            path_gain_blocks[i]);
        TORCH_CHECK(valid_blocks[i].get_device() == device, "all path blocks must share a device");
        total += valid_blocks[i].size(0);
    }
    if (total == 0) {
        return empty_path_block_from(valid_blocks[0]);
    }

    auto bool_options = valid_blocks[0].options().dtype(at::kBool);
    auto int_options = tx_id_blocks[0].options().dtype(at::kInt);
    auto long_options = tx_id_blocks[0].options().dtype(at::kLong);
    auto float_options = path_gain_blocks[0].options().dtype(at::kFloat);
    auto valid = at::empty({total}, bool_options);
    auto tx_id = at::empty({total}, int_options);
    auto rx_id = at::empty({total}, int_options);
    auto depth = at::empty({total}, int_options);
    auto component_id = at::empty({total}, int_options);
    auto primitive_id = at::empty({total}, int_options);
    auto edge_id = at::empty({total}, int_options);
    auto path_length = at::empty({total}, float_options);
    auto delay = at::empty({total}, float_options);
    auto path_gain = at::empty({total}, float_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    int64_t offset = 0;
    for (size_t i = 0; i < block_count; ++i) {
        const int64_t count = valid_blocks[i].size(0);
        if (count > 0) {
            path_copy_block_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
                count,
                offset,
                valid_blocks[i].data_ptr<bool>(),
                tx_id_blocks[i].data_ptr<int>(),
                rx_id_blocks[i].data_ptr<int>(),
                depth_blocks[i].data_ptr<int>(),
                component_id_blocks[i].data_ptr<int>(),
                primitive_id_blocks[i].data_ptr<int>(),
                edge_id_blocks[i].data_ptr<int>(),
                path_length_blocks[i].data_ptr<float>(),
                delay_blocks[i].data_ptr<float>(),
                path_gain_blocks[i].data_ptr<float>(),
                valid.data_ptr<bool>(),
                tx_id.data_ptr<int>(),
                rx_id.data_ptr<int>(),
                depth.data_ptr<int>(),
                component_id.data_ptr<int>(),
                primitive_id.data_ptr<int>(),
                edge_id.data_ptr<int>(),
                path_length.data_ptr<float>(),
                delay.data_ptr<float>(),
                path_gain.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        offset += count;
    }

    auto keys = at::empty({total}, long_options);
    auto order = at::empty({total}, long_options);
    path_sort_key_kernel<<<static_cast<int>(launch_blocks(total)), kPathBlockSize, 0, stream>>>(
        total,
        tx_count,
        max_depth,
        tx_id.data_ptr<int>(),
        rx_id.data_ptr<int>(),
        depth.data_ptr<int>(),
        component_id.data_ptr<int>(),
        keys.data_ptr<int64_t>(),
        order.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(keys.data_ptr<int64_t>()),
        thrust::device_pointer_cast(keys.data_ptr<int64_t>() + total),
        thrust::device_pointer_cast(order.data_ptr<int64_t>()));

    const int64_t out_count = max_paths < 0 ? total : std::min<int64_t>(max_paths, total);
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
    if (out_count > 0) {
        path_gather_kernel<<<static_cast<int>(launch_blocks(out_count)), kPathBlockSize, 0, stream>>>(
            out_count,
            order.data_ptr<int64_t>(),
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
            out_valid.data_ptr<bool>(),
            out_tx_id.data_ptr<int>(),
            out_rx_id.data_ptr<int>(),
            out_depth.data_ptr<int>(),
            out_component_id.data_ptr<int>(),
            out_primitive_id.data_ptr<int>(),
            out_edge_id.data_ptr<int>(),
            out_path_length.data_ptr<float>(),
            out_delay.data_ptr<float>(),
            out_path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        out_valid,
        out_tx_id,
        out_rx_id,
        out_depth,
        out_component_id,
        out_primitive_id,
        out_edge_id,
        out_path_length,
        out_delay,
        out_path_gain};
}
