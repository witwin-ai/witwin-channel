#pragma once

#include <ATen/ATen.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include "../tensor_checks.h"

#include <cstdint>
#include <tuple>

namespace channel_native::evaluated_paths {

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

inline void check_row_tensor(
    const at::Tensor& tensor,
    const char *name,
    c10::ScalarType dtype,
    int64_t rank,
    int64_t rows,
    int device) {
    channel_native::check_tensor(tensor, name, dtype, rank);
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

}  // namespace channel_native::evaluated_paths
