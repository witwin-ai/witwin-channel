#pragma once

// CUDA translation units need ATen and PyTorch's pybind tensor caster, not the
// full C++ frontend pulled in by torch/extension.h.
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
