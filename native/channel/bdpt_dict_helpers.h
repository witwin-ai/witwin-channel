// Copyright Xingyu Chen.
// Declares bdpt dict helpers native contracts.

#pragma once

#include <torch/extension.h>

// Inline dictionary accessor shared by both BDPT Torch bridges.
inline at::Tensor tensor_from_dict(const pybind11::dict& values, const char* field) {
    TORCH_CHECK(values.contains(field), "missing tensor field: ", field);
    return pybind11::cast<at::Tensor>(values[pybind11::str(field)]);
}
