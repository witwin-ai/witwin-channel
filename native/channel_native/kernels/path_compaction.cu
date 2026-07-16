#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include "path_compaction_common.cuh"

namespace {

constexpr double kLightSpeedMetersPerSecond = 299792458.0;
constexpr float kPi = 3.14159265358979323846f;

__device__ float cn_neg_kd_phase(float k, float d) {
    // Reduce k*d mod 2*pi in double: the f32 product loses ~k*d*2^-24 of
    // phase, which shifts coherent nulls at mmWave ranges.
    const double kd = fmod(static_cast<double>(k) * static_cast<double>(d), 6.283185307179586476925287);
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

__global__ void path_los_compact_kernel(
    int64_t count,
    const int* __restrict__ flags,
    const int* __restrict__ offsets,
    const int* __restrict__ tx_id,
    const int* __restrict__ rx_id,
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
        if (flags[index] == 0) {
            continue;
        }
        const int dst = offsets[index];
        out_valid[dst] = true;
        out_tx_id[dst] = tx_id[index];
        out_rx_id[dst] = rx_id[index];
        out_depth[dst] = 0;
        out_component_id[dst] = 0;
        out_primitive_id[dst] = -1;
        out_edge_id[dst] = -1;
        out_path_length[dst] = path_length[index];
        out_delay[dst] = delay[index];
        out_path_gain[dst] = path_gain[index];
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
        const float wavelength = static_cast<float>(kLightSpeedMetersPerSecond) / frequency_hz;
        const float phase = cn_neg_kd_phase(2.0f * kPi / wavelength, length);

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

__global__ void deterministic_diffraction_order1_compact_kernel(
    int64_t count,
    const int* __restrict__ flags,
    const int* __restrict__ offsets,
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
    int* __restrict__ out_rx_id,
    int* __restrict__ out_depth,
    int* __restrict__ out_edge_id,
    float* __restrict__ out_delay,
    float* __restrict__ out_x_re,
    float* __restrict__ out_x_im,
    float* __restrict__ out_y_re,
    float* __restrict__ out_y_im,
    float* __restrict__ out_z_re,
    float* __restrict__ out_z_im,
    float* __restrict__ out_interaction_position) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        if (flags[index] == 0) {
            continue;
        }
        const int dst = offsets[index];
        out_rx_id[dst] = rx_id[index];
        out_depth[dst] = depth[index];
        out_edge_id[dst] = edge_id[index];
        out_delay[dst] = delay[index];
        out_x_re[dst] = x_re[index];
        out_x_im[dst] = x_im[index];
        out_y_re[dst] = y_re[index];
        out_y_im[dst] = y_im[index];
        out_z_re[dst] = z_re[index];
        out_z_im[dst] = z_im[index];
        const int64_t src_base = index * 3;
        const int64_t dst_base = static_cast<int64_t>(dst) * 3;
        out_interaction_position[dst_base + 0] = interaction_position[src_base + 0];
        out_interaction_position[dst_base + 1] = interaction_position[src_base + 1];
        out_interaction_position[dst_base + 2] = interaction_position[src_base + 2];
    }
}

__global__ void path_block_flags_kernel(
    int64_t count,
    const bool* __restrict__ valid,
    const bool* __restrict__ visible0,
    const bool* __restrict__ visible1,
    int* __restrict__ flags) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        flags[index] = (valid[index] && visible0[index] && visible1[index]) ? 1 : 0;
    }
}

__global__ void path_block_compact_kernel(
    int64_t count,
    const int* __restrict__ flags,
    const int* __restrict__ offsets,
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
        if (flags[index] == 0) {
            continue;
        }
        const int dst = offsets[index];
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

__global__ void path_diffraction_compact_kernel(
    int64_t count,
    const int* __restrict__ flags,
    const int* __restrict__ offsets,
    int tx_index,
    const int* __restrict__ rx_id,
    const int* __restrict__ depth,
    const int* __restrict__ edge_id,
    const float* __restrict__ delay,
    const float* __restrict__ field_x_re,
    const float* __restrict__ field_x_im,
    const float* __restrict__ field_y_re,
    const float* __restrict__ field_y_im,
    const float* __restrict__ field_z_re,
    const float* __restrict__ field_z_im,
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
        if (flags[index] == 0) {
            continue;
        }
        const int dst = offsets[index];
        const float x_re = field_x_re[index];
        const float x_im = field_x_im[index];
        const float y_re = field_y_re[index];
        const float y_im = field_y_im[index];
        const float z_re = field_z_re[index];
        const float z_im = field_z_im[index];
        const float sample_delay = delay[index];
        out_valid[dst] = true;
        out_tx_id[dst] = tx_index;
        out_rx_id[dst] = rx_id[index];
        out_depth[dst] = depth[index];
        out_component_id[dst] = 2;
        out_primitive_id[dst] = -1;
        out_edge_id[dst] = edge_id[index];
        out_path_length[dst] = sample_delay * static_cast<float>(kLightSpeedMetersPerSecond);
        out_delay[dst] = sample_delay;
        out_path_gain[dst] =
            x_re * x_re + x_im * x_im +
            y_re * y_re + y_im * y_im +
            z_re * z_re + z_im * z_im;
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_path_filter_los_cuda(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor path_length,
    at::Tensor delay,
    at::Tensor path_gain,
    at::Tensor visible) {
    check_cuda_tensor(tx_id, "tx_id", at::kInt, 1);
    check_cuda_tensor(rx_id, "rx_id", at::kInt, 1);
    check_cuda_tensor(path_length, "path_length_m", at::kFloat, 1);
    check_cuda_tensor(delay, "delay_s", at::kFloat, 1);
    check_cuda_tensor(path_gain, "path_gain", at::kFloat, 1);
    check_cuda_tensor(visible, "visible", at::kBool, 1);
    const int64_t count = tx_id.size(0);
    TORCH_CHECK(rx_id.size(0) == count, "rx_id must match tx_id");
    TORCH_CHECK(path_length.size(0) == count, "path_length_m must match tx_id");
    TORCH_CHECK(delay.size(0) == count, "delay_s must match tx_id");
    TORCH_CHECK(path_gain.size(0) == count, "path_gain must match tx_id");
    TORCH_CHECK(visible.size(0) == count, "visible must match tx_id");

    if (count == 0) {
        return empty_path_block_from(tx_id);
    }

    auto int_options = tx_id.options().dtype(at::kInt);
    auto flags = at::empty({count}, int_options);
    auto offsets = at::empty({count}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_id.get_device()).stream();
    path_visibility_flags_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        visible.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + count),
        thrust::device_pointer_cast(offsets.data_ptr<int>()));

    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        offsets.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t out_count = static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset);
    if (out_count == 0) {
        return empty_path_block_from(tx_id);
    }

    auto bool_options = visible.options().dtype(at::kBool);
    auto float_options = path_gain.options().dtype(at::kFloat);
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
    path_los_compact_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        flags.data_ptr<int>(),
        offsets.data_ptr<int>(),
        tx_id.data_ptr<int>(),
        rx_id.data_ptr<int>(),
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

pybind11::dict cn_deterministic_los_topology_block(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor path_length,
    at::Tensor delay,
    at::Tensor path_gain,
    at::Tensor visible,
    double frequency_hz,
    int64_t sequence_width) {
    check_cuda_tensor(tx_id, "tx_id", at::kInt, 1);
    check_cuda_tensor(rx_id, "rx_id", at::kInt, 1);
    check_cuda_tensor(path_length, "path_length_m", at::kFloat, 1);
    check_cuda_tensor(delay, "delay_s", at::kFloat, 1);
    check_cuda_tensor(path_gain, "path_gain", at::kFloat, 1);
    check_cuda_tensor(visible, "visible", at::kBool, 1);
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
    auto offsets = at::empty({count}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    path_visibility_flags_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        visible.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + count),
        thrust::device_pointer_cast(offsets.data_ptr<int>()));

    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        offsets.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t out_count = static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset);
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
    deterministic_los_topology_compact_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
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

pybind11::dict cn_deterministic_reflection_order1_compact(
    at::Tensor visible,
    at::Tensor epc_faces,
    at::Tensor epc_hits,
    at::Tensor epc_normals,
    at::Tensor sequence_batch,
    at::Tensor rx_indices,
    at::Tensor tx,
    at::Tensor rx_positions,
    at::Tensor tx_power,
    int64_t tx_index,
    at::Tensor face_eps_r,
    at::Tensor face_sigma_e,
    at::Tensor face_mu_r,
    at::Tensor face_gain,
    at::Tensor face_material_id,
    bool grouped_export) {
    check_cuda_tensor(visible, "visible", at::kBool, 1);
    TORCH_CHECK(epc_faces.is_cuda(), "epc_faces must be a CUDA tensor");
    TORCH_CHECK(epc_faces.is_contiguous(), "epc_faces must be contiguous");
    TORCH_CHECK(epc_faces.dim() == 2, "epc_faces must have shape (N, depth)");
    TORCH_CHECK(epc_faces.scalar_type() == at::kInt || epc_faces.scalar_type() == at::kLong, "epc_faces must be int32 or int64");
    check_cuda_tensor(epc_hits, "epc_hits", at::kFloat, 3);
    check_cuda_tensor(epc_normals, "epc_normals", at::kFloat, 3);
    check_cuda_tensor(sequence_batch, "sequence_batch", at::kInt, 2);
    check_cuda_tensor(rx_indices, "rx_indices", at::kInt, 1);
    check_cuda_tensor(tx, "tx", at::kFloat, 1);
    TORCH_CHECK(tx.size(0) == 3, "tx must have shape (3,)");
    check_vec3_table(rx_positions, "rx_positions");
    check_cuda_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_cuda_tensor(face_eps_r, "face_eps_r", at::kFloat, 1);
    check_cuda_tensor(face_sigma_e, "face_sigma_e", at::kFloat, 1);
    check_cuda_tensor(face_mu_r, "face_mu_r", at::kFloat, 1);
    check_cuda_tensor(face_gain, "face_gain", at::kFloat, 1);
    check_cuda_tensor(face_material_id, "face_material_id", at::kInt, 1);
    const int64_t count = visible.size(0);
    TORCH_CHECK(epc_faces.size(0) == count, "epc_faces must match visible");
    TORCH_CHECK(epc_faces.size(1) >= 1, "epc_faces must include the first bounce");
    TORCH_CHECK(epc_hits.size(0) == count, "epc_hits must match visible");
    TORCH_CHECK(epc_normals.sizes() == epc_hits.sizes(), "epc_normals must match epc_hits");
    TORCH_CHECK(epc_hits.size(1) >= 1 && epc_hits.size(2) == 3, "epc_hits must have shape (N, depth, 3)");
    TORCH_CHECK(sequence_batch.size(0) == count, "sequence_batch must match visible");
    TORCH_CHECK(sequence_batch.size(1) >= 1, "sequence_batch must include the first bounce");
    TORCH_CHECK(rx_indices.size(0) == count, "rx_indices must match visible");
    TORCH_CHECK(tx_index >= 0 && tx_index < tx_power.size(0), "tx_index is out of range");
    TORCH_CHECK(face_sigma_e.size(0) == face_eps_r.size(0), "face_sigma_e must match face_eps_r");
    TORCH_CHECK(face_mu_r.size(0) == face_eps_r.size(0), "face_mu_r must match face_eps_r");
    TORCH_CHECK(face_gain.size(0) == face_eps_r.size(0), "face_gain must match face_eps_r");
    TORCH_CHECK(face_material_id.size(0) == face_eps_r.size(0), "face_material_id must match face_eps_r");
    const int device = visible.get_device();
    TORCH_CHECK(epc_faces.get_device() == device, "epc_faces must share visible device");
    TORCH_CHECK(epc_hits.get_device() == device, "epc_hits must share visible device");
    TORCH_CHECK(epc_normals.get_device() == device, "epc_normals must share visible device");
    TORCH_CHECK(sequence_batch.get_device() == device, "sequence_batch must share visible device");
    TORCH_CHECK(rx_indices.get_device() == device, "rx_indices must share visible device");
    TORCH_CHECK(tx.get_device() == device, "tx must share visible device");
    TORCH_CHECK(rx_positions.get_device() == device, "rx_positions must share visible device");
    TORCH_CHECK(tx_power.get_device() == device, "tx_power must share visible device");
    TORCH_CHECK(face_eps_r.get_device() == device, "face_eps_r must share visible device");
    TORCH_CHECK(face_sigma_e.get_device() == device, "face_sigma_e must share visible device");
    TORCH_CHECK(face_mu_r.get_device() == device, "face_mu_r must share visible device");
    TORCH_CHECK(face_gain.get_device() == device, "face_gain must share visible device");
    TORCH_CHECK(face_material_id.get_device() == device, "face_material_id must share visible device");

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
    auto offsets = at::empty({count}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    path_visibility_flags_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        visible.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + count),
        thrust::device_pointer_cast(offsets.data_ptr<int>()));
    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        offsets.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t out_count = static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset);
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
            static_cast<int>(launch_blocks(count)),
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
            static_cast<int>(launch_blocks(count)),
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

pybind11::dict cn_deterministic_reflection_sequence_compact(
    at::Tensor visible,
    at::Tensor epc_sequences,
    at::Tensor epc_hits,
    at::Tensor epc_normals,
    at::Tensor rx_indices,
    at::Tensor tx,
    at::Tensor rx_positions,
    at::Tensor tx_power,
    int64_t tx_index,
    at::Tensor face_eps_r,
    at::Tensor face_sigma_e,
    at::Tensor face_mu_r,
    at::Tensor face_gain,
    at::Tensor face_material_id,
    int64_t max_count) {
    check_cuda_tensor(visible, "visible", at::kBool, 1);
    TORCH_CHECK(epc_sequences.is_cuda(), "epc_sequences must be a CUDA tensor");
    TORCH_CHECK(epc_sequences.is_contiguous(), "epc_sequences must be contiguous");
    TORCH_CHECK(epc_sequences.dim() == 2, "epc_sequences must have shape (N, depth)");
    TORCH_CHECK(epc_sequences.scalar_type() == at::kInt || epc_sequences.scalar_type() == at::kLong, "epc_sequences must be int32 or int64");
    check_cuda_tensor(epc_hits, "epc_hits", at::kFloat, 3);
    check_cuda_tensor(epc_normals, "epc_normals", at::kFloat, 3);
    check_cuda_tensor(rx_indices, "rx_indices", at::kInt, 1);
    check_cuda_tensor(tx, "tx", at::kFloat, 1);
    TORCH_CHECK(tx.size(0) == 3, "tx must have shape (3,)");
    check_vec3_table(rx_positions, "rx_positions");
    check_cuda_tensor(tx_power, "tx_power", at::kFloat, 1);
    check_cuda_tensor(face_eps_r, "face_eps_r", at::kFloat, 1);
    check_cuda_tensor(face_sigma_e, "face_sigma_e", at::kFloat, 1);
    check_cuda_tensor(face_mu_r, "face_mu_r", at::kFloat, 1);
    check_cuda_tensor(face_gain, "face_gain", at::kFloat, 1);
    check_cuda_tensor(face_material_id, "face_material_id", at::kInt, 1);
    TORCH_CHECK(max_count >= -1, "max_count must be -1 or non-negative");
    const int64_t count = visible.size(0);
    const int64_t depth = epc_sequences.size(1);
    TORCH_CHECK(depth > 0, "epc_sequences must have positive depth");
    TORCH_CHECK(epc_sequences.size(0) == count, "epc_sequences must match visible");
    TORCH_CHECK(epc_hits.size(0) == count, "epc_hits must match visible");
    TORCH_CHECK(epc_hits.size(1) == depth && epc_hits.size(2) == 3, "epc_hits must have shape (N, depth, 3)");
    TORCH_CHECK(epc_normals.sizes() == epc_hits.sizes(), "epc_normals must match epc_hits");
    TORCH_CHECK(rx_indices.size(0) == count, "rx_indices must match visible");
    TORCH_CHECK(tx_index >= 0 && tx_index < tx_power.size(0), "tx_index is out of range");
    TORCH_CHECK(face_sigma_e.size(0) == face_eps_r.size(0), "face_sigma_e must match face_eps_r");
    TORCH_CHECK(face_mu_r.size(0) == face_eps_r.size(0), "face_mu_r must match face_eps_r");
    TORCH_CHECK(face_gain.size(0) == face_eps_r.size(0), "face_gain must match face_eps_r");
    TORCH_CHECK(face_material_id.size(0) == face_eps_r.size(0), "face_material_id must match face_eps_r");
    const int device = visible.get_device();
    TORCH_CHECK(epc_sequences.get_device() == device, "epc_sequences must share visible device");
    TORCH_CHECK(epc_hits.get_device() == device, "epc_hits must share visible device");
    TORCH_CHECK(epc_normals.get_device() == device, "epc_normals must share visible device");
    TORCH_CHECK(rx_indices.get_device() == device, "rx_indices must share visible device");
    TORCH_CHECK(tx.get_device() == device, "tx must share visible device");
    TORCH_CHECK(rx_positions.get_device() == device, "rx_positions must share visible device");
    TORCH_CHECK(tx_power.get_device() == device, "tx_power must share visible device");
    TORCH_CHECK(face_eps_r.get_device() == device, "face_eps_r must share visible device");
    TORCH_CHECK(face_sigma_e.get_device() == device, "face_sigma_e must share visible device");
    TORCH_CHECK(face_mu_r.get_device() == device, "face_mu_r must share visible device");
    TORCH_CHECK(face_gain.get_device() == device, "face_gain must share visible device");
    TORCH_CHECK(face_material_id.get_device() == device, "face_material_id must share visible device");

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
    auto offsets = at::empty({count}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    path_visibility_flags_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        visible.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + count),
        thrust::device_pointer_cast(offsets.data_ptr<int>()));
    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        offsets.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t visible_count = static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset);
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
            static_cast<int>(launch_blocks(count)),
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
            static_cast<int>(launch_blocks(count)),
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

pybind11::dict cn_deterministic_diffraction_order1_compact(
    at::Tensor valid,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor edge_id,
    at::Tensor delay,
    at::Tensor x_re,
    at::Tensor x_im,
    at::Tensor y_re,
    at::Tensor y_im,
    at::Tensor z_re,
    at::Tensor z_im,
    at::Tensor interaction_position) {
    check_cuda_tensor(valid, "valid", at::kBool, 1);
    check_cuda_tensor(rx_id, "rx_id", at::kInt, 1);
    check_cuda_tensor(depth, "depth", at::kInt, 1);
    check_cuda_tensor(edge_id, "edge_id", at::kInt, 1);
    check_cuda_tensor(delay, "delay_s", at::kFloat, 1);
    check_cuda_tensor(x_re, "x_re", at::kFloat, 1);
    check_cuda_tensor(x_im, "x_im", at::kFloat, 1);
    check_cuda_tensor(y_re, "y_re", at::kFloat, 1);
    check_cuda_tensor(y_im, "y_im", at::kFloat, 1);
    check_cuda_tensor(z_re, "z_re", at::kFloat, 1);
    check_cuda_tensor(z_im, "z_im", at::kFloat, 1);
    check_vec3_table(interaction_position, "interaction_position");
    const int64_t count = valid.size(0);
    const int device = valid.get_device();
    for (const auto& tensor : {rx_id, depth, edge_id}) {
        TORCH_CHECK(tensor.size(0) == count, "diffraction int tensors must match valid");
        TORCH_CHECK(tensor.get_device() == device, "diffraction int tensors must share valid device");
    }
    for (const auto& tensor : {delay, x_re, x_im, y_re, y_im, z_re, z_im}) {
        TORCH_CHECK(tensor.size(0) == count, "diffraction float tensors must match valid");
        TORCH_CHECK(tensor.get_device() == device, "diffraction float tensors must share valid device");
    }
    TORCH_CHECK(interaction_position.size(0) == count, "interaction_position must match valid");
    TORCH_CHECK(interaction_position.get_device() == device, "interaction_position must share valid device");

    auto int_options = rx_id.options().dtype(at::kInt);
    auto float_options = delay.options().dtype(at::kFloat);
    auto empty_out = [&]() {
        pybind11::dict out;
        out["rx_id"] = at::empty({0}, int_options);
        out["depth"] = at::empty({0}, int_options);
        out["edge_id"] = at::empty({0}, int_options);
        out["delay_s"] = at::empty({0}, float_options);
        out["x_re"] = at::empty({0}, float_options);
        out["x_im"] = at::empty({0}, float_options);
        out["y_re"] = at::empty({0}, float_options);
        out["y_im"] = at::empty({0}, float_options);
        out["z_re"] = at::empty({0}, float_options);
        out["z_im"] = at::empty({0}, float_options);
        out["interaction_position"] = at::empty({0, 3}, float_options);
        return out;
    };
    if (count == 0) {
        return empty_out();
    }

    auto flags = at::empty({count}, int_options);
    auto offsets = at::empty({count}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    path_visibility_flags_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        valid.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + count),
        thrust::device_pointer_cast(offsets.data_ptr<int>()));
    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        offsets.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t out_count = static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset);
    if (out_count == 0) {
        return empty_out();
    }

    pybind11::dict out;
    out["rx_id"] = at::empty({out_count}, int_options);
    out["depth"] = at::empty({out_count}, int_options);
    out["edge_id"] = at::empty({out_count}, int_options);
    out["delay_s"] = at::empty({out_count}, float_options);
    out["x_re"] = at::empty({out_count}, float_options);
    out["x_im"] = at::empty({out_count}, float_options);
    out["y_re"] = at::empty({out_count}, float_options);
    out["y_im"] = at::empty({out_count}, float_options);
    out["z_re"] = at::empty({out_count}, float_options);
    out["z_im"] = at::empty({out_count}, float_options);
    out["interaction_position"] = at::empty({out_count, 3}, float_options);
    deterministic_diffraction_order1_compact_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        flags.data_ptr<int>(),
        offsets.data_ptr<int>(),
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
        pybind11::cast<at::Tensor>(out["rx_id"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["depth"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["edge_id"]).data_ptr<int>(),
        pybind11::cast<at::Tensor>(out["delay_s"]).data_ptr<float>(),
        pybind11::cast<at::Tensor>(out["x_re"]).data_ptr<float>(),
        pybind11::cast<at::Tensor>(out["x_im"]).data_ptr<float>(),
        pybind11::cast<at::Tensor>(out["y_re"]).data_ptr<float>(),
        pybind11::cast<at::Tensor>(out["y_im"]).data_ptr<float>(),
        pybind11::cast<at::Tensor>(out["z_re"]).data_ptr<float>(),
        pybind11::cast<at::Tensor>(out["z_im"]).data_ptr<float>(),
        pybind11::cast<at::Tensor>(out["interaction_position"]).data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_path_filter_block_cuda(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor component_id,
    at::Tensor primitive_id,
    at::Tensor edge_id,
    at::Tensor path_length,
    at::Tensor delay,
    at::Tensor path_gain,
    at::Tensor visible0,
    at::Tensor visible1) {
    check_path_block_shapes(valid, tx_id, rx_id, depth, component_id, primitive_id, edge_id, path_length, delay, path_gain);
    check_cuda_tensor(visible0, "visible0", at::kBool, 1);
    check_cuda_tensor(visible1, "visible1", at::kBool, 1);
    const int64_t count = valid.size(0);
    TORCH_CHECK(visible0.size(0) == count, "visible0 must match valid");
    TORCH_CHECK(visible1.size(0) == count, "visible1 must match valid");
    if (count == 0) {
        return empty_path_block_from(valid);
    }

    auto int_options = tx_id.options().dtype(at::kInt);
    auto flags = at::empty({count}, int_options);
    auto offsets = at::empty({count}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    path_block_flags_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        valid.data_ptr<bool>(),
        visible0.data_ptr<bool>(),
        visible1.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + count),
        thrust::device_pointer_cast(offsets.data_ptr<int>()));

    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        offsets.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t out_count = static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset);
    if (out_count == 0) {
        return empty_path_block_from(valid);
    }

    auto bool_options = valid.options().dtype(at::kBool);
    auto float_options = path_gain.options().dtype(at::kFloat);
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
    path_block_compact_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        flags.data_ptr<int>(),
        offsets.data_ptr<int>(),
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cn_path_diffraction_block_cuda(
    at::Tensor valid,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor edge_id,
    at::Tensor delay,
    at::Tensor field_x_re,
    at::Tensor field_x_im,
    at::Tensor field_y_re,
    at::Tensor field_y_im,
    at::Tensor field_z_re,
    at::Tensor field_z_im,
    int64_t tx_index) {
    check_cuda_tensor(valid, "valid", at::kBool, 1);
    check_cuda_tensor(rx_id, "rx_id", at::kInt, 1);
    check_cuda_tensor(depth, "depth", at::kInt, 1);
    check_cuda_tensor(edge_id, "edge_id", at::kInt, 1);
    check_cuda_tensor(delay, "delay_s", at::kFloat, 1);
    check_cuda_tensor(field_x_re, "field_x_re", at::kFloat, 1);
    check_cuda_tensor(field_x_im, "field_x_im", at::kFloat, 1);
    check_cuda_tensor(field_y_re, "field_y_re", at::kFloat, 1);
    check_cuda_tensor(field_y_im, "field_y_im", at::kFloat, 1);
    check_cuda_tensor(field_z_re, "field_z_re", at::kFloat, 1);
    check_cuda_tensor(field_z_im, "field_z_im", at::kFloat, 1);
    TORCH_CHECK(tx_index >= 0 && tx_index <= static_cast<int64_t>(std::numeric_limits<int>::max()), "tx_index out of range");
    const int64_t count = valid.size(0);
    for (const auto& tensor : {rx_id, depth, edge_id, delay, field_x_re, field_x_im, field_y_re, field_y_im, field_z_re, field_z_im}) {
        TORCH_CHECK(tensor.size(0) == count, "diffraction output tensors must share capacity");
        TORCH_CHECK(tensor.get_device() == valid.get_device(), "diffraction output tensors must share a device");
    }
    if (count == 0) {
        return empty_path_block_from(valid);
    }

    auto int_options = rx_id.options().dtype(at::kInt);
    auto flags = at::empty({count}, int_options);
    auto offsets = at::empty({count}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    path_visibility_flags_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        valid.data_ptr<bool>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + count),
        thrust::device_pointer_cast(offsets.data_ptr<int>()));

    int last_flag = 0;
    int last_offset = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_flag,
        flags.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        offsets.data_ptr<int>() + count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t out_count = static_cast<int64_t>(last_flag) + static_cast<int64_t>(last_offset);
    if (out_count == 0) {
        return empty_path_block_from(valid);
    }

    auto bool_options = valid.options().dtype(at::kBool);
    auto float_options = delay.options().dtype(at::kFloat);
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
    path_diffraction_compact_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        flags.data_ptr<int>(),
        offsets.data_ptr<int>(),
        static_cast<int>(tx_index),
        rx_id.data_ptr<int>(),
        depth.data_ptr<int>(),
        edge_id.data_ptr<int>(),
        delay.data_ptr<float>(),
        field_x_re.data_ptr<float>(),
        field_x_im.data_ptr<float>(),
        field_y_re.data_ptr<float>(),
        field_y_im.data_ptr<float>(),
        field_z_re.data_ptr<float>(),
        field_z_im.data_ptr<float>(),
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
