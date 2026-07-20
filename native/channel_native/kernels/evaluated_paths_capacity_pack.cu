#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "../tensor_checks.h"
#include "deterministic_capacity_finalize.h"

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kPackBlockSize = 256;
using cfloat = c10::complex<float>;
using channel_native::check_tensor;

struct PackInput {
    const int *tx_id;
    const int *rx_id;
    const int *depth;
    const int *component_id;
    const int *primitive_id;
    const int *edge_id;
    const int *material_id;
    const int *primitive_sequence;
    const int *material_sequence;
    const int *interaction_type;
    const float *path_length_m;
    const float *delay_s;
    const float *field_direction;
    const float *interaction_position;
    const float *interaction_normal;
    const float *interaction_positions;
    const float *interaction_normals;
    const float *path_gain;
    const cfloat *path_field;
    const cfloat *field_xyz;
    const cfloat *coefficient;
};

struct PackOutput {
    int64_t *selected_row_index;
    bool *valid;
    int *num_paths;
    bool *overflow;
    int *tx_id;
    int *rx_id;
    int *depth;
    int *component_id;
    int *primitive_id;
    int *edge_id;
    int *material_id;
    int *primitive_sequence;
    int *material_sequence;
    int *interaction_type;
    float *path_length_m;
    float *delay_s;
    float *field_direction;
    float *interaction_position;
    float *interaction_normal;
    float *interaction_positions;
    float *interaction_normals;
    float *path_gain;
    cfloat *path_field;
    cfloat *field_xyz;
    cfloat *coefficient;
};

__global__ void evaluated_paths_capacity_init_kernel(
    PackOutput output,
    int64_t row_capacity,
    int64_t pair_count,
    int64_t sequence_width) {
    const int64_t count = row_capacity > pair_count ? row_capacity : pair_count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count;
         row += stride) {
        if (row < pair_count) {
            output.num_paths[row] = 0;
        }
        if (row >= row_capacity) {
            continue;
        }
        output.selected_row_index[row] = -1;
        output.valid[row] = false;
        output.tx_id[row] = -1;
        output.rx_id[row] = -1;
        output.depth[row] = 0;
        output.component_id[row] = -1;
        output.primitive_id[row] = -1;
        output.edge_id[row] = -1;
        output.material_id[row] = -1;
        output.path_length_m[row] = -1.0f;
        output.delay_s[row] = -1.0f;
        output.path_gain[row] = 0.0f;
        output.path_field[row] = cfloat(0.0f, 0.0f);
        output.coefficient[row] = cfloat(0.0f, 0.0f);
        const int64_t vec = row * 3;
        for (int component = 0; component < 3; ++component) {
            output.field_direction[vec + component] = 0.0f;
            output.interaction_position[vec + component] = 0.0f;
            output.interaction_normal[vec + component] = 0.0f;
            output.field_xyz[vec + component] = cfloat(0.0f, 0.0f);
        }
        const int64_t sequence = row * sequence_width;
        const int64_t sequence_vec = sequence * 3;
        for (int64_t slot = 0; slot < sequence_width; ++slot) {
            output.primitive_sequence[sequence + slot] = -1;
            output.material_sequence[sequence + slot] = -1;
            output.interaction_type[sequence + slot] = 0;
            const int64_t slot_vec = sequence_vec + slot * 3;
            for (int component = 0; component < 3; ++component) {
                output.interaction_positions[slot_vec + component] = 0.0f;
                output.interaction_normals[slot_vec + component] = 0.0f;
            }
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output.overflow[0] = false;
    }
}

__global__ void evaluated_paths_capacity_gather_kernel(
    PackInput input,
    PackOutput output,
    const int *__restrict__ overflow_flag,
    const int *__restrict__ contract_error,
    int64_t row_capacity,
    int64_t sequence_width) {
    if (overflow_flag[0] || contract_error[0]) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t destination =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         destination < row_capacity;
         destination += stride) {
        if (!output.valid[destination]) {
            continue;
        }
        const int64_t source = output.selected_row_index[destination];
        output.tx_id[destination] = input.tx_id[source];
        output.rx_id[destination] = input.rx_id[source];
        output.depth[destination] = input.depth[source];
        output.component_id[destination] = input.component_id[source];
        output.primitive_id[destination] = input.primitive_id[source];
        output.edge_id[destination] = input.edge_id[source];
        output.material_id[destination] = input.material_id[source];
        output.path_length_m[destination] = input.path_length_m[source];
        output.delay_s[destination] = input.delay_s[source];
        output.path_gain[destination] = input.path_gain[source];
        output.path_field[destination] = input.path_field[source];
        output.coefficient[destination] = input.coefficient[source];
        const int64_t destination_vec = destination * 3;
        const int64_t source_vec = source * 3;
        for (int component = 0; component < 3; ++component) {
            output.field_direction[destination_vec + component] =
                input.field_direction[source_vec + component];
            output.interaction_position[destination_vec + component] =
                input.interaction_position[source_vec + component];
            output.interaction_normal[destination_vec + component] =
                input.interaction_normal[source_vec + component];
            output.field_xyz[destination_vec + component] =
                input.field_xyz[source_vec + component];
        }
        const int64_t destination_sequence = destination * sequence_width;
        const int64_t source_sequence = source * sequence_width;
        for (int64_t slot = 0; slot < sequence_width; ++slot) {
            output.primitive_sequence[destination_sequence + slot] =
                input.primitive_sequence[source_sequence + slot];
            output.material_sequence[destination_sequence + slot] =
                input.material_sequence[source_sequence + slot];
            output.interaction_type[destination_sequence + slot] =
                input.interaction_type[source_sequence + slot];
            const int64_t destination_slot = (destination_sequence + slot) * 3;
            const int64_t source_slot = (source_sequence + slot) * 3;
            for (int component = 0; component < 3; ++component) {
                output.interaction_positions[destination_slot + component] =
                    input.interaction_positions[source_slot + component];
                output.interaction_normals[destination_slot + component] =
                    input.interaction_normals[source_slot + component];
            }
        }
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kPackBlockSize - 1) / kPackBlockSize);
}

void check_row_tensor(
    const at::Tensor& tensor,
    const char *name,
    c10::ScalarType dtype,
    int64_t rank,
    int64_t rows,
    int device) {
    check_tensor(tensor, name, dtype, rank);
    TORCH_CHECK(tensor.size(0) == rows, name, " must match candidate rows");
    TORCH_CHECK(tensor.get_device() == device, name, " must share candidate device");
}

}  // namespace

pybind11::dict cn_evaluated_paths_capacity_pack(
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor component_id,
    at::Tensor primitive_id,
    at::Tensor edge_id,
    at::Tensor material_id,
    at::Tensor primitive_sequence,
    at::Tensor material_sequence,
    at::Tensor interaction_type,
    at::Tensor path_length_m,
    at::Tensor delay_s,
    at::Tensor field_direction,
    at::Tensor interaction_position,
    at::Tensor interaction_normal,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor path_gain,
    at::Tensor path_field,
    at::Tensor field_xyz,
    at::Tensor coefficient,
    int64_t pair_count,
    int64_t num_tx,
    int64_t num_rx,
    int64_t path_capacity_per_pair) {
    check_tensor(valid, "valid", at::kBool, 1);
    const int64_t candidate_count = valid.size(0);
    const int device = valid.get_device();
    for (const auto& item : std::initializer_list<std::tuple<at::Tensor, const char *, c10::ScalarType>>{
             {tx_id, "tx_id", at::kInt},
             {rx_id, "rx_id", at::kInt},
             {depth, "depth", at::kInt},
             {component_id, "component_id", at::kInt},
             {primitive_id, "primitive_id", at::kInt},
             {edge_id, "edge_id", at::kInt},
             {material_id, "material_id", at::kInt},
             {path_length_m, "path_length_m", at::kFloat},
             {delay_s, "delay_s", at::kFloat},
             {path_gain, "path_gain", at::kFloat},
             {path_field, "path_field", at::kComplexFloat},
             {coefficient, "coefficient", at::kComplexFloat}}) {
        check_row_tensor(
            std::get<0>(item),
            std::get<1>(item),
            std::get<2>(item),
            1,
            candidate_count,
            device);
    }
    for (const auto& item : std::initializer_list<std::tuple<at::Tensor, const char *, c10::ScalarType>>{
             {field_direction, "field_direction", at::kFloat},
             {interaction_position, "interaction_position", at::kFloat},
             {interaction_normal, "interaction_normal", at::kFloat},
             {field_xyz, "field_xyz", at::kComplexFloat}}) {
        check_row_tensor(
            std::get<0>(item),
            std::get<1>(item),
            std::get<2>(item),
            2,
            candidate_count,
            device);
        TORCH_CHECK(std::get<0>(item).size(1) == 3, std::get<1>(item), " must be vec3 rows");
    }
    check_row_tensor(
        primitive_sequence, "primitive_sequence", at::kInt, 2, candidate_count, device);
    const int64_t sequence_width = primitive_sequence.size(1);
    for (const auto& item : std::initializer_list<std::pair<at::Tensor, const char *>>{
             {material_sequence, "material_sequence"},
             {interaction_type, "interaction_type"}}) {
        check_row_tensor(
            item.first, item.second, at::kInt, 2, candidate_count, device);
        TORCH_CHECK(item.first.size(1) == sequence_width, item.second, " has wrong width");
    }
    for (const auto& item : std::initializer_list<std::pair<at::Tensor, const char *>>{
             {interaction_positions, "interaction_positions"},
             {interaction_normals, "interaction_normals"}}) {
        check_row_tensor(
            item.first, item.second, at::kFloat, 3, candidate_count, device);
        TORCH_CHECK(
            item.first.size(1) == sequence_width && item.first.size(2) == 3,
            item.second,
            " must have shape (rows, width, 3)");
    }
    const int64_t row_capacity =
        channel_native::capacity::deterministic_capacity_validate(
            valid,
            tx_id,
            rx_id,
            pair_count,
            num_tx,
            num_rx,
            path_capacity_per_pair);

    auto bool_options = valid.options().dtype(at::kBool);
    auto int_options = valid.options().dtype(at::kInt);
    auto long_options = valid.options().dtype(at::kLong);
    auto float_options = valid.options().dtype(at::kFloat);
    auto complex_options = valid.options().dtype(at::kComplexFloat);
    auto selected_row_index = at::empty({row_capacity}, long_options);
    auto out_valid = at::empty({row_capacity}, bool_options);
    auto num_paths = at::empty({pair_count}, int_options);
    auto overflow = at::empty({1}, bool_options);
    auto out_tx_id = at::empty({row_capacity}, int_options);
    auto out_rx_id = at::empty({row_capacity}, int_options);
    auto out_depth = at::empty({row_capacity}, int_options);
    auto out_component_id = at::empty({row_capacity}, int_options);
    auto out_primitive_id = at::empty({row_capacity}, int_options);
    auto out_edge_id = at::empty({row_capacity}, int_options);
    auto out_material_id = at::empty({row_capacity}, int_options);
    auto out_primitive_sequence = at::empty({row_capacity, sequence_width}, int_options);
    auto out_material_sequence = at::empty({row_capacity, sequence_width}, int_options);
    auto out_interaction_type = at::empty({row_capacity, sequence_width}, int_options);
    auto out_path_length_m = at::empty({row_capacity}, float_options);
    auto out_delay_s = at::empty({row_capacity}, float_options);
    auto out_field_direction = at::empty({row_capacity, 3}, float_options);
    auto out_interaction_position = at::empty({row_capacity, 3}, float_options);
    auto out_interaction_normal = at::empty({row_capacity, 3}, float_options);
    auto out_interaction_positions =
        at::empty({row_capacity, sequence_width, 3}, float_options);
    auto out_interaction_normals =
        at::empty({row_capacity, sequence_width, 3}, float_options);
    auto out_path_gain = at::empty({row_capacity}, float_options);
    auto out_path_field = at::empty({row_capacity}, complex_options);
    auto out_field_xyz = at::empty({row_capacity, 3}, complex_options);
    auto out_coefficient = at::empty({row_capacity}, complex_options);

    const PackInput input{
        tx_id.data_ptr<int>(), rx_id.data_ptr<int>(), depth.data_ptr<int>(),
        component_id.data_ptr<int>(), primitive_id.data_ptr<int>(), edge_id.data_ptr<int>(),
        material_id.data_ptr<int>(), primitive_sequence.data_ptr<int>(),
        material_sequence.data_ptr<int>(), interaction_type.data_ptr<int>(),
        path_length_m.data_ptr<float>(), delay_s.data_ptr<float>(),
        field_direction.data_ptr<float>(), interaction_position.data_ptr<float>(),
        interaction_normal.data_ptr<float>(), interaction_positions.data_ptr<float>(),
        interaction_normals.data_ptr<float>(), path_gain.data_ptr<float>(),
        path_field.data_ptr<cfloat>(), field_xyz.data_ptr<cfloat>(),
        coefficient.data_ptr<cfloat>()};
    const PackOutput output{
        selected_row_index.data_ptr<int64_t>(), out_valid.data_ptr<bool>(),
        num_paths.data_ptr<int>(), overflow.data_ptr<bool>(), out_tx_id.data_ptr<int>(),
        out_rx_id.data_ptr<int>(), out_depth.data_ptr<int>(),
        out_component_id.data_ptr<int>(), out_primitive_id.data_ptr<int>(),
        out_edge_id.data_ptr<int>(), out_material_id.data_ptr<int>(),
        out_primitive_sequence.data_ptr<int>(), out_material_sequence.data_ptr<int>(),
        out_interaction_type.data_ptr<int>(), out_path_length_m.data_ptr<float>(),
        out_delay_s.data_ptr<float>(), out_field_direction.data_ptr<float>(),
        out_interaction_position.data_ptr<float>(), out_interaction_normal.data_ptr<float>(),
        out_interaction_positions.data_ptr<float>(), out_interaction_normals.data_ptr<float>(),
        out_path_gain.data_ptr<float>(), out_path_field.data_ptr<cfloat>(),
        out_field_xyz.data_ptr<cfloat>(), out_coefficient.data_ptr<cfloat>()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const int64_t init_count = std::max<int64_t>(1, std::max(row_capacity, pair_count));
    evaluated_paths_capacity_init_kernel<<<
        launch_blocks(init_count), kPackBlockSize, 0, stream>>>(
        output, row_capacity, pair_count, sequence_width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto state = channel_native::capacity::deterministic_capacity_finalize_no_trap(
        valid,
        tx_id,
        rx_id,
        pair_count,
        num_tx,
        num_rx,
        path_capacity_per_pair,
        selected_row_index,
        out_valid,
        num_paths,
        false);
    if (row_capacity > 0) {
        evaluated_paths_capacity_gather_kernel<<<
            launch_blocks(row_capacity), kPackBlockSize, 0, stream>>>(
            input,
            output,
            state.overflow_flag.data_ptr<int>(),
            state.contract_error.data_ptr<int>(),
            row_capacity,
            sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    channel_native::capacity::deterministic_capacity_publish_status(
        state, overflow, stream);
    channel_native::capacity::deterministic_capacity_trap(state, stream);

    pybind11::dict result;
    result["selected_row_index"] = selected_row_index;
    result["valid"] = out_valid;
    result["num_paths"] = num_paths;
    result["overflow"] = overflow;
    result["tx_id"] = out_tx_id;
    result["rx_id"] = out_rx_id;
    result["depth"] = out_depth;
    result["component_id"] = out_component_id;
    result["primitive_id"] = out_primitive_id;
    result["edge_id"] = out_edge_id;
    result["material_id"] = out_material_id;
    result["primitive_sequence"] = out_primitive_sequence;
    result["material_sequence"] = out_material_sequence;
    result["interaction_type"] = out_interaction_type;
    result["path_length_m"] = out_path_length_m;
    result["delay_s"] = out_delay_s;
    result["field_direction"] = out_field_direction;
    result["interaction_position"] = out_interaction_position;
    result["interaction_normal"] = out_interaction_normal;
    result["interaction_positions"] = out_interaction_positions;
    result["interaction_normals"] = out_interaction_normals;
    result["path_gain"] = out_path_gain;
    result["path_field"] = out_path_field;
    result["field_xyz"] = out_field_xyz;
    result["coefficient"] = out_coefficient;
    return result;
}
