// Copyright Xingyu Chen.
// Declares evaluated paths payload plumbing native contracts.

#pragma once

#include <ATen/ATen.h>
#include <c10/util/complex.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"

#include <cstdint>
#include <tuple>

namespace channel::evaluated_paths {

using cfloat = c10::complex<float>;

struct PayloadTensors {
    const at::Tensor& tx_id;
    const at::Tensor& rx_id;
    const at::Tensor& depth;
    const at::Tensor& component_id;
    const at::Tensor& primitive_id;
    const at::Tensor& edge_id;
    const at::Tensor& material_id;
    const at::Tensor& primitive_sequence;
    const at::Tensor& material_sequence;
    const at::Tensor& interaction_type;
    const at::Tensor& path_length_m;
    const at::Tensor& delay_s;
    const at::Tensor& field_direction;
    const at::Tensor& interaction_position;
    const at::Tensor& interaction_normal;
    const at::Tensor& interaction_positions;
    const at::Tensor& interaction_normals;
    const at::Tensor& path_gain;
    const at::Tensor& path_field;
    const at::Tensor& field_xyz;
    const at::Tensor& coefficient;
};

struct PayloadInputView {
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

inline PayloadInputView input_view(const PayloadTensors& payload) {
    return {
        payload.tx_id.data_ptr<int>(),
        payload.rx_id.data_ptr<int>(),
        payload.depth.data_ptr<int>(),
        payload.component_id.data_ptr<int>(),
        payload.primitive_id.data_ptr<int>(),
        payload.edge_id.data_ptr<int>(),
        payload.material_id.data_ptr<int>(),
        payload.primitive_sequence.data_ptr<int>(),
        payload.material_sequence.data_ptr<int>(),
        payload.interaction_type.data_ptr<int>(),
        payload.path_length_m.data_ptr<float>(),
        payload.delay_s.data_ptr<float>(),
        payload.field_direction.data_ptr<float>(),
        payload.interaction_position.data_ptr<float>(),
        payload.interaction_normal.data_ptr<float>(),
        payload.interaction_positions.data_ptr<float>(),
        payload.interaction_normals.data_ptr<float>(),
        payload.path_gain.data_ptr<float>(),
        payload.path_field.data_ptr<cfloat>(),
        payload.field_xyz.data_ptr<cfloat>(),
        payload.coefficient.data_ptr<cfloat>()};
}

inline void check_row_tensor(
    const at::Tensor& tensor,
    const char *name,
    c10::ScalarType dtype,
    int64_t rank,
    int64_t rows,
    int device) {
    channel::check_tensor(tensor, name, dtype, rank);
    TORCH_CHECK(tensor.size(0) == rows, name, " must match candidate rows");
    TORCH_CHECK(tensor.get_device() == device, name, " must share candidate device");
}

inline int64_t validate_payload(
    const PayloadTensors& payload,
    int64_t rows,
    int device) {
    for (const auto& item : std::initializer_list<std::tuple<at::Tensor, const char *, c10::ScalarType>>{
             {payload.tx_id, "tx_id", at::kInt},
             {payload.rx_id, "rx_id", at::kInt},
             {payload.depth, "depth", at::kInt},
             {payload.component_id, "component_id", at::kInt},
             {payload.primitive_id, "primitive_id", at::kInt},
             {payload.edge_id, "edge_id", at::kInt},
             {payload.material_id, "material_id", at::kInt},
             {payload.path_length_m, "path_length_m", at::kFloat},
             {payload.delay_s, "delay_s", at::kFloat},
             {payload.path_gain, "path_gain", at::kFloat},
             {payload.path_field, "path_field", at::kComplexFloat},
             {payload.coefficient, "coefficient", at::kComplexFloat}}) {
        check_row_tensor(
            std::get<0>(item), std::get<1>(item), std::get<2>(item), 1, rows, device);
    }
    for (const auto& item : std::initializer_list<std::tuple<at::Tensor, const char *, c10::ScalarType>>{
             {payload.field_direction, "field_direction", at::kFloat},
             {payload.interaction_position, "interaction_position", at::kFloat},
             {payload.interaction_normal, "interaction_normal", at::kFloat},
             {payload.field_xyz, "field_xyz", at::kComplexFloat}}) {
        check_row_tensor(
            std::get<0>(item), std::get<1>(item), std::get<2>(item), 2, rows, device);
        TORCH_CHECK(std::get<0>(item).size(1) == 3, std::get<1>(item), " must be vec3 rows");
    }
    check_row_tensor(payload.primitive_sequence, "primitive_sequence", at::kInt, 2, rows, device);
    const int64_t sequence_width = payload.primitive_sequence.size(1);
    for (const auto& item : std::initializer_list<std::pair<at::Tensor, const char *>>{
             {payload.material_sequence, "material_sequence"},
             {payload.interaction_type, "interaction_type"}}) {
        check_row_tensor(item.first, item.second, at::kInt, 2, rows, device);
        TORCH_CHECK(item.first.size(1) == sequence_width, item.second, " has wrong width");
    }
    for (const auto& item : std::initializer_list<std::pair<at::Tensor, const char *>>{
             {payload.interaction_positions, "interaction_positions"},
             {payload.interaction_normals, "interaction_normals"}}) {
        check_row_tensor(item.first, item.second, at::kFloat, 3, rows, device);
        TORCH_CHECK(
            item.first.size(1) == sequence_width && item.first.size(2) == 3,
            item.second,
            " must have shape (rows, width, 3)");
    }
    return sequence_width;
}

struct AllocatedPayload {
    at::Tensor tx_id;
    at::Tensor rx_id;
    at::Tensor depth;
    at::Tensor component_id;
    at::Tensor primitive_id;
    at::Tensor edge_id;
    at::Tensor material_id;
    at::Tensor primitive_sequence;
    at::Tensor material_sequence;
    at::Tensor interaction_type;
    at::Tensor path_length_m;
    at::Tensor delay_s;
    at::Tensor field_direction;
    at::Tensor interaction_position;
    at::Tensor interaction_normal;
    at::Tensor interaction_positions;
    at::Tensor interaction_normals;
    at::Tensor path_gain;
    at::Tensor path_field;
    at::Tensor field_xyz;
    at::Tensor coefficient;

    void append_to(pybind11::dict& result) const {
        result["tx_id"] = tx_id;
        result["rx_id"] = rx_id;
        result["depth"] = depth;
        result["component_id"] = component_id;
        result["primitive_id"] = primitive_id;
        result["edge_id"] = edge_id;
        result["material_id"] = material_id;
        result["primitive_sequence"] = primitive_sequence;
        result["material_sequence"] = material_sequence;
        result["interaction_type"] = interaction_type;
        result["path_length_m"] = path_length_m;
        result["delay_s"] = delay_s;
        result["field_direction"] = field_direction;
        result["interaction_position"] = interaction_position;
        result["interaction_normal"] = interaction_normal;
        result["interaction_positions"] = interaction_positions;
        result["interaction_normals"] = interaction_normals;
        result["path_gain"] = path_gain;
        result["path_field"] = path_field;
        result["field_xyz"] = field_xyz;
        result["coefficient"] = coefficient;
    }
};

struct PayloadOutputView {
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

inline PayloadOutputView output_view(const AllocatedPayload& payload) {
    return {
        payload.tx_id.data_ptr<int>(),
        payload.rx_id.data_ptr<int>(),
        payload.depth.data_ptr<int>(),
        payload.component_id.data_ptr<int>(),
        payload.primitive_id.data_ptr<int>(),
        payload.edge_id.data_ptr<int>(),
        payload.material_id.data_ptr<int>(),
        payload.primitive_sequence.data_ptr<int>(),
        payload.material_sequence.data_ptr<int>(),
        payload.interaction_type.data_ptr<int>(),
        payload.path_length_m.data_ptr<float>(),
        payload.delay_s.data_ptr<float>(),
        payload.field_direction.data_ptr<float>(),
        payload.interaction_position.data_ptr<float>(),
        payload.interaction_normal.data_ptr<float>(),
        payload.interaction_positions.data_ptr<float>(),
        payload.interaction_normals.data_ptr<float>(),
        payload.path_gain.data_ptr<float>(),
        payload.path_field.data_ptr<cfloat>(),
        payload.field_xyz.data_ptr<cfloat>(),
        payload.coefficient.data_ptr<cfloat>()};
}

__device__ inline void initialize_row(
    PayloadOutputView output,
    int64_t row,
    int64_t sequence_width) {
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
    const int64_t row_vector = row * 3;
    for (int component = 0; component < 3; ++component) {
        output.field_direction[row_vector + component] = 0.0f;
        output.interaction_position[row_vector + component] = 0.0f;
        output.interaction_normal[row_vector + component] = 0.0f;
        output.field_xyz[row_vector + component] = cfloat(0.0f, 0.0f);
    }
    const int64_t row_sequence = row * sequence_width;
    for (int64_t slot = 0; slot < sequence_width; ++slot) {
        const int64_t item = row_sequence + slot;
        output.primitive_sequence[item] = -1;
        output.material_sequence[item] = -1;
        output.interaction_type[item] = 0;
        for (int component = 0; component < 3; ++component) {
            const int64_t item_vector = item * 3 + component;
            output.interaction_positions[item_vector] = 0.0f;
            output.interaction_normals[item_vector] = 0.0f;
        }
    }
}

__device__ inline void copy_row(
    PayloadInputView input,
    PayloadOutputView output,
    int64_t source,
    int64_t destination,
    int64_t sequence_width) {
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
    const int64_t source_vector = source * 3;
    const int64_t destination_vector = destination * 3;
    for (int component = 0; component < 3; ++component) {
        output.field_direction[destination_vector + component] =
            input.field_direction[source_vector + component];
        output.interaction_position[destination_vector + component] =
            input.interaction_position[source_vector + component];
        output.interaction_normal[destination_vector + component] =
            input.interaction_normal[source_vector + component];
        output.field_xyz[destination_vector + component] =
            input.field_xyz[source_vector + component];
    }
    const int64_t source_sequence = source * sequence_width;
    const int64_t destination_sequence = destination * sequence_width;
    for (int64_t slot = 0; slot < sequence_width; ++slot) {
        const int64_t source_item = source_sequence + slot;
        const int64_t destination_item = destination_sequence + slot;
        output.primitive_sequence[destination_item] =
            input.primitive_sequence[source_item];
        output.material_sequence[destination_item] =
            input.material_sequence[source_item];
        output.interaction_type[destination_item] = input.interaction_type[source_item];
        for (int component = 0; component < 3; ++component) {
            const int64_t source_item_vector = source_item * 3 + component;
            const int64_t destination_item_vector = destination_item * 3 + component;
            output.interaction_positions[destination_item_vector] =
                input.interaction_positions[source_item_vector];
            output.interaction_normals[destination_item_vector] =
                input.interaction_normals[source_item_vector];
        }
    }
}

inline AllocatedPayload allocate_payload(
    const at::Tensor& reference,
    int64_t rows,
    int64_t sequence_width) {
    auto int_options = reference.options().dtype(at::kInt);
    auto float_options = reference.options().dtype(at::kFloat);
    auto complex_options = reference.options().dtype(at::kComplexFloat);
    return {
        at::empty({rows}, int_options),
        at::empty({rows}, int_options),
        at::empty({rows}, int_options),
        at::empty({rows}, int_options),
        at::empty({rows}, int_options),
        at::empty({rows}, int_options),
        at::empty({rows}, int_options),
        at::empty({rows, sequence_width}, int_options),
        at::empty({rows, sequence_width}, int_options),
        at::empty({rows, sequence_width}, int_options),
        at::empty({rows}, float_options),
        at::empty({rows}, float_options),
        at::empty({rows, 3}, float_options),
        at::empty({rows, 3}, float_options),
        at::empty({rows, 3}, float_options),
        at::empty({rows, sequence_width, 3}, float_options),
        at::empty({rows, sequence_width, 3}, float_options),
        at::empty({rows}, float_options),
        at::empty({rows}, complex_options),
        at::empty({rows, 3}, complex_options),
        at::empty({rows}, complex_options)};
}

}  // namespace channel::evaluated_paths
