#pragma once

#include <torch/extension.h>

// Shared dict-field accessor for the BDPT Torch bridges. Defined inline so
// bdpt.cpp and bdpt_ad_dispatch.cpp share a single owner without duplicating the
// definition (the ADR-022 AD dispatch split, native size budget).
inline at::Tensor tensor_from_dict(const pybind11::dict& values, const char* field) {
    TORCH_CHECK(values.contains(field), "missing tensor field: ", field);
    return pybind11::cast<at::Tensor>(values[pybind11::str(field)]);
}
