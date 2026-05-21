#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void register_shadow_boundary_bindings(nb::module_ &m);
