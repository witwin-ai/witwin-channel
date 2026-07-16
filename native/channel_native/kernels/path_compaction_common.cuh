#pragma once

#include <torch/extension.h>

#include <tuple>

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
empty_path_block_from(const at::Tensor& reference) {
    auto bool_options = reference.options().dtype(at::kBool);
    auto int_options = reference.options().dtype(at::kInt);
    auto float_options = reference.options().dtype(at::kFloat);
    return {
        at::empty({0}, bool_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, int_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
        at::empty({0}, float_options),
    };
}

pybind11::dict empty_deterministic_los_topology_block_from(const at::Tensor& reference, int64_t sequence_width) {
    auto bool_options = reference.options().dtype(at::kBool);
    auto int_options = reference.options().dtype(at::kInt);
    auto float_options = reference.options().dtype(at::kFloat);
    auto complex_options = reference.options().dtype(at::kComplexFloat);
    pybind11::dict out;
    out["valid"] = at::empty({0}, bool_options);
    out["tx_id"] = at::empty({0}, int_options);
    out["rx_id"] = at::empty({0}, int_options);
    out["depth"] = at::empty({0}, int_options);
    out["component_id"] = at::empty({0}, int_options);
    out["primitive_id"] = at::empty({0}, int_options);
    out["edge_id"] = at::empty({0}, int_options);
    out["path_length_m"] = at::empty({0}, float_options);
    out["delay_s"] = at::empty({0}, float_options);
    out["path_gain"] = at::empty({0}, float_options);
    out["path_field"] = at::empty({0}, complex_options);
    out["interaction_position"] = at::empty({0, 3}, float_options);
    out["interaction_normal"] = at::empty({0, 3}, float_options);
    out["material_id"] = at::empty({0}, int_options);
    out["primitive_sequence"] = at::empty({0, sequence_width}, int_options);
    out["material_sequence"] = at::empty({0, sequence_width}, int_options);
    out["interaction_positions"] = at::empty({0, sequence_width, 3}, float_options);
    out["interaction_normals"] = at::empty({0, sequence_width, 3}, float_options);
    return out;
}

}  // namespace
