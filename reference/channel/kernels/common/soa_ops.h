#pragma once

#include <common/primitives.h>

namespace witwin::channel::native_ext::common {

WITWIN_KERNEL_DINLINE Vec3f load_xyz(
    const float* x,
    const float* y,
    const float* z,
    int idx
) {
    return make_vec3(x[idx], y[idx], z[idx]);
}

WITWIN_KERNEL_DINLINE void store_xyz(
    float* x,
    float* y,
    float* z,
    int idx,
    Vec3f value
) {
    x[idx] = value.x;
    y[idx] = value.y;
    z[idx] = value.z;
}

} // namespace witwin::channel::native_ext::common

