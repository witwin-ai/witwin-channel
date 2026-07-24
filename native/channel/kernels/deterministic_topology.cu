#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"
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

__global__ void deterministic_sort_key_1d_kernel(
    int64_t count,
    const int* __restrict__ values,
    const int64_t* __restrict__ order,
    int64_t* __restrict__ keys) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        keys[index] = static_cast<int64_t>(values[order[index]]);
    }
}

__global__ void deterministic_sort_key_sequence_kernel(
    int64_t count,
    int64_t width,
    int64_t column,
    const int* __restrict__ sequence,
    const int64_t* __restrict__ order,
    int64_t* __restrict__ keys) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        keys[index] = static_cast<int64_t>(sequence[order[index] * width + column]);
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

__global__ void deterministic_surface_group_keys_kernel(
    int64_t count,
    const int64_t* __restrict__ surface_ids,
    int64_t* __restrict__ key_matrix) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t face = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         face < count;
         face += stride) {
        const int64_t key_base = face * kFaceGroupKeyWidth;
        key_matrix[key_base + 0] = surface_ids[face];
        key_matrix[key_base + 1] = 0;
        key_matrix[key_base + 2] = 0;
        key_matrix[key_base + 3] = 0;
        key_matrix[key_base + 4] = 0;
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
        check_cuda_tensor(path_field, "path_field", at::kComplexFloat, 1);
        TORCH_CHECK(path_field.size(0) == count, "path_field must match valid");
        TORCH_CHECK(path_field.get_device() == device, "path_field must share valid device");
    }

    check_optional_field_presence(block, "field_xyz", has_field_xyz);
    if (has_field_xyz) {
        at::Tensor field_xyz = block_tensor(block, "field_xyz");
        check_cuda_tensor(field_xyz, "field_xyz", at::kComplexFloat, 2);
        TORCH_CHECK(field_xyz.size(0) == count && field_xyz.size(1) == 3,
                    "field_xyz must have shape (N, 3)");
        TORCH_CHECK(field_xyz.get_device() == device, "field_xyz must share valid device");
    }

    check_optional_field_presence(block, "coefficient", has_coefficient);
    if (has_coefficient) {
        at::Tensor coefficient = block_tensor(block, "coefficient");
        check_cuda_tensor(coefficient, "coefficient", at::kComplexFloat, 1);
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
        check_cuda_tensor(material_id, "material_id", at::kInt, 1);
        TORCH_CHECK(material_id.size(0) == count, "material_id must match valid");
        TORCH_CHECK(material_id.get_device() == device, "material_id must share valid device");
    }

    check_optional_field_presence(block, "primitive_sequence", has_primitive_sequence);
    if (has_primitive_sequence) {
        at::Tensor primitive_sequence = block_tensor(block, "primitive_sequence");
        check_cuda_tensor(primitive_sequence, "primitive_sequence", at::kInt, 2);
        TORCH_CHECK(primitive_sequence.size(0) == count, "primitive_sequence must match valid");
        TORCH_CHECK(primitive_sequence.size(1) == sequence_width, "primitive_sequence has wrong width");
        TORCH_CHECK(primitive_sequence.get_device() == device, "primitive_sequence must share valid device");
    }

    check_optional_field_presence(block, "material_sequence", has_material_sequence);
    if (has_material_sequence) {
        at::Tensor material_sequence = block_tensor(block, "material_sequence");
        check_cuda_tensor(material_sequence, "material_sequence", at::kInt, 2);
        TORCH_CHECK(material_sequence.size(0) == count, "material_sequence must match valid");
        TORCH_CHECK(material_sequence.size(1) == sequence_width, "material_sequence has wrong width");
        TORCH_CHECK(material_sequence.get_device() == device, "material_sequence must share valid device");
    }

    check_optional_field_presence(block, "interaction_positions", has_interaction_positions);
    if (has_interaction_positions) {
        at::Tensor interaction_positions = block_tensor(block, "interaction_positions");
        check_cuda_tensor(interaction_positions, "interaction_positions", at::kFloat, 3);
        TORCH_CHECK(interaction_positions.size(0) == count, "interaction_positions must match valid");
        TORCH_CHECK(interaction_positions.size(1) == sequence_width, "interaction_positions has wrong width");
        TORCH_CHECK(interaction_positions.size(2) == 3, "interaction_positions must have vec3 rows");
        TORCH_CHECK(interaction_positions.get_device() == device, "interaction_positions must share valid device");
    }

    check_optional_field_presence(block, "interaction_normals", has_interaction_normals);
    if (has_interaction_normals) {
        at::Tensor interaction_normals = block_tensor(block, "interaction_normals");
        check_cuda_tensor(interaction_normals, "interaction_normals", at::kFloat, 3);
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
    deterministic_gather_rows_kernel<scalar_t><<<static_cast<int>(launch_blocks(out_count * row_stride)), kPathBlockSize, 0, stream>>>(
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
    check_cuda_tensor(order, "order", at::kLong, 1);

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
    check_cuda_tensor(surface_ids, "surface_ids", at::kLong, 1);
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
    deterministic_face_group_keys_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
        face_count,
        tri_a.data_ptr<float>(),
        normals.data_ptr<float>(),
        surface_ids.data_ptr<int64_t>(),
        static_cast<float>(1.0 / quantization),
        key_matrix.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    deterministic_order_init_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
        face_count,
        order.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int64_t column = kFaceGroupKeyWidth - 1; column >= 0; --column) {
        deterministic_face_group_sort_key_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
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
    auto group_offsets = at::empty({face_count}, int_options);
    deterministic_face_group_flags_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
        face_count,
        key_matrix.data_ptr<int64_t>(),
        order.data_ptr<int64_t>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + face_count),
        thrust::device_pointer_cast(group_offsets.data_ptr<int>()));

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
    deterministic_face_group_assign_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
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
    deterministic_face_group_members_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
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

pybind11::dict channel_deterministic_surface_face_groups(at::Tensor surface_ids) {
    check_cuda_tensor(surface_ids, "surface_ids", at::kLong, 1);
    const int64_t face_count = surface_ids.size(0);
    auto int_options = surface_ids.options().dtype(at::kInt);
    auto long_options = surface_ids.options().dtype(at::kLong);
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

    const int device = surface_ids.get_device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    auto key_matrix = at::empty({face_count, kFaceGroupKeyWidth}, long_options);
    auto order = at::empty({face_count}, long_options);
    auto sort_keys = at::empty({face_count}, long_options);
    deterministic_surface_group_keys_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
        face_count,
        surface_ids.data_ptr<int64_t>(),
        key_matrix.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    deterministic_order_init_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
        face_count,
        order.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int64_t column = kFaceGroupKeyWidth - 1; column >= 0; --column) {
        deterministic_face_group_sort_key_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
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
    auto group_offsets = at::empty({face_count}, int_options);
    deterministic_face_group_flags_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
        face_count,
        key_matrix.data_ptr<int64_t>(),
        order.data_ptr<int64_t>(),
        flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(flags.data_ptr<int>()),
        thrust::device_pointer_cast(flags.data_ptr<int>() + face_count),
        thrust::device_pointer_cast(group_offsets.data_ptr<int>()));

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
    deterministic_face_group_assign_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
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
    deterministic_face_group_members_kernel<<<static_cast<int>(launch_blocks(face_count)), kPathBlockSize, 0, stream>>>(
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

at::Tensor channel_deterministic_sort_order(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor component_id,
    at::Tensor primitive_id,
    at::Tensor edge_id,
    at::Tensor primitive_sequence) {
    check_cuda_tensor(valid, "valid", at::kBool, 1);
    check_cuda_tensor(tx_id, "tx_id", at::kInt, 1);
    check_cuda_tensor(rx_id, "rx_id", at::kInt, 1);
    check_cuda_tensor(depth, "depth", at::kInt, 1);
    check_cuda_tensor(component_id, "component_id", at::kInt, 1);
    check_cuda_tensor(primitive_id, "primitive_id", at::kInt, 1);
    check_cuda_tensor(edge_id, "edge_id", at::kInt, 1);
    check_cuda_tensor(primitive_sequence, "primitive_sequence", at::kInt, 2);
    const int64_t count = valid.size(0);
    TORCH_CHECK(tx_id.size(0) == count, "tx_id must match valid");
    TORCH_CHECK(rx_id.size(0) == count, "rx_id must match valid");
    TORCH_CHECK(depth.size(0) == count, "depth must match valid");
    TORCH_CHECK(component_id.size(0) == count, "component_id must match valid");
    TORCH_CHECK(primitive_id.size(0) == count, "primitive_id must match valid");
    TORCH_CHECK(edge_id.size(0) == count, "edge_id must match valid");
    TORCH_CHECK(primitive_sequence.size(0) == count, "primitive_sequence must match valid");
    const int device = valid.get_device();
    TORCH_CHECK(tx_id.get_device() == device, "tx_id must share valid device");
    TORCH_CHECK(rx_id.get_device() == device, "rx_id must share valid device");
    TORCH_CHECK(depth.get_device() == device, "depth must share valid device");
    TORCH_CHECK(component_id.get_device() == device, "component_id must share valid device");
    TORCH_CHECK(primitive_id.get_device() == device, "primitive_id must share valid device");
    TORCH_CHECK(edge_id.get_device() == device, "edge_id must share valid device");
    TORCH_CHECK(primitive_sequence.get_device() == device, "primitive_sequence must share valid device");
    auto long_options = tx_id.options().dtype(at::kLong);
    auto order = at::empty({count}, long_options);
    if (count == 0) {
        return order;
    }
    auto keys = at::empty({count}, long_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    deterministic_order_init_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
        count,
        order.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto stable_sort_by_current_keys = [&]() {
        thrust::stable_sort_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(keys.data_ptr<int64_t>()),
            thrust::device_pointer_cast(keys.data_ptr<int64_t>() + count),
            thrust::device_pointer_cast(order.data_ptr<int64_t>()));
    };
    auto sort_by_1d = [&](const at::Tensor& values) {
        deterministic_sort_key_1d_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
            count,
            values.data_ptr<int>(),
            order.data_ptr<int64_t>(),
            keys.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        stable_sort_by_current_keys();
    };

    sort_by_1d(edge_id);
    sort_by_1d(primitive_id);
    const int64_t width = primitive_sequence.size(1);
    for (int64_t column = width - 1; column >= 0; --column) {
        deterministic_sort_key_sequence_kernel<<<static_cast<int>(launch_blocks(count)), kPathBlockSize, 0, stream>>>(
            count,
            width,
            column,
            primitive_sequence.data_ptr<int>(),
            order.data_ptr<int64_t>(),
            keys.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        stable_sort_by_current_keys();
    }
    sort_by_1d(component_id);
    sort_by_1d(depth);
    sort_by_1d(tx_id);
    sort_by_1d(rx_id);
    return order;
}
