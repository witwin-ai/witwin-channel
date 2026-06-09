#pragma once

#include <optix.h>

#include <cstdint>

namespace raydn {

struct OptixIntersectParams {
    OptixTraversableHandle traversable = 0;
    const float *ray_o = nullptr;
    const float *ray_d = nullptr;
    const float *ray_tmax = nullptr;
    const bool *active = nullptr;
    float *out_t = nullptr;
    int *out_shape_id = nullptr;
    int *out_local_prim_id = nullptr;
    int *out_global_prim_id = nullptr;
    float *out_bary_uv = nullptr;
    const int *face_offsets = nullptr;
    int32_t mesh_count = 0;
    int32_t ray_count = 0;
};

} // namespace raydn
