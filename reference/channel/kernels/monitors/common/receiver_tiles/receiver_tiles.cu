#include <monitors/common/receiver_tiles/receiver_tiles.h>

#include <common/cuda_check.h>
#include <common/grid_ops.h>

namespace witwin::channel::native_ext {

namespace {

using common::ceil_div_int;
using common::throw_cuda;

__global__ void build_receiver_tiles_kernel(
    int plane_axis,
    float plane_position,
    const float* coord_0,
    const float* coord_1,
    int n_coord_0,
    int n_coord_1,
    int tile_size_0,
    int tile_size_1,
    int n_tiles_0,
    int n_tiles,
    int* out_tile_i0,
    int* out_tile_i1,
    int* out_tile_extent_0,
    int* out_tile_extent_1,
    float* out_coord_0_min,
    float* out_coord_0_max,
    float* out_coord_1_min,
    float* out_coord_1_max,
    float* out_aabb_min_x,
    float* out_aabb_min_y,
    float* out_aabb_min_z,
    float* out_aabb_max_x,
    float* out_aabb_max_y,
    float* out_aabb_max_z
) {
    int tile_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tile_idx >= n_tiles) {
        return;
    }

    int tile_x = tile_idx % n_tiles_0;
    int tile_y = tile_idx / n_tiles_0;
    int start_0 = tile_x * tile_size_0;
    int start_1 = tile_y * tile_size_1;
    int extent_0 = min(tile_size_0, n_coord_0 - start_0);
    int extent_1 = min(tile_size_1, n_coord_1 - start_1);
    int end_0 = start_0 + extent_0 - 1;
    int end_1 = start_1 + extent_1 - 1;

    float coord_0_min = coord_0[start_0];
    float coord_0_max = coord_0[end_0];
    float coord_1_min = coord_1[start_1];
    float coord_1_max = coord_1[end_1];

    out_tile_i0[tile_idx] = start_0;
    out_tile_i1[tile_idx] = start_1;
    out_tile_extent_0[tile_idx] = extent_0;
    out_tile_extent_1[tile_idx] = extent_1;
    out_coord_0_min[tile_idx] = coord_0_min;
    out_coord_0_max[tile_idx] = coord_0_max;
    out_coord_1_min[tile_idx] = coord_1_min;
    out_coord_1_max[tile_idx] = coord_1_max;

    common::Vec3f aabb_min = common::point_on_plane(
        plane_axis,
        plane_position,
        coord_0_min,
        coord_1_min
    );
    common::Vec3f aabb_max = common::point_on_plane(
        plane_axis,
        plane_position,
        coord_0_max,
        coord_1_max
    );
    out_aabb_min_x[tile_idx] = aabb_min.x;
    out_aabb_min_y[tile_idx] = aabb_min.y;
    out_aabb_min_z[tile_idx] = aabb_min.z;
    out_aabb_max_x[tile_idx] = aabb_max.x;
    out_aabb_max_y[tile_idx] = aabb_max.y;
    out_aabb_max_z[tile_idx] = aabb_max.z;
}

} // namespace

void build_receiver_tiles(
    int plane_axis,
    float plane_position,
    const float* coord_0,
    const float* coord_1,
    int n_coord_0,
    int n_coord_1,
    int tile_size_0,
    int tile_size_1,
    int* out_tile_i0,
    int* out_tile_i1,
    int* out_tile_extent_0,
    int* out_tile_extent_1,
    float* out_coord_0_min,
    float* out_coord_0_max,
    float* out_coord_1_min,
    float* out_coord_1_max,
    float* out_aabb_min_x,
    float* out_aabb_min_y,
    float* out_aabb_min_z,
    float* out_aabb_max_x,
    float* out_aabb_max_y,
    float* out_aabb_max_z
) {
    if (n_coord_0 <= 0 || n_coord_1 <= 0 || tile_size_0 <= 0 || tile_size_1 <= 0) {
        return;
    }
    int n_tiles_0 = ceil_div_int(n_coord_0, tile_size_0);
    int n_tiles_1 = ceil_div_int(n_coord_1, tile_size_1);
    int n_tiles = n_tiles_0 * n_tiles_1;
    constexpr int threads = 128;
    int blocks = ceil_div_int(n_tiles, threads);
    build_receiver_tiles_kernel<<<blocks, threads>>>(
        plane_axis,
        plane_position,
        coord_0,
        coord_1,
        n_coord_0,
        n_coord_1,
        tile_size_0,
        tile_size_1,
        n_tiles_0,
        n_tiles,
        out_tile_i0,
        out_tile_i1,
        out_tile_extent_0,
        out_tile_extent_1,
        out_coord_0_min,
        out_coord_0_max,
        out_coord_1_min,
        out_coord_1_max,
        out_aabb_min_x,
        out_aabb_min_y,
        out_aabb_min_z,
        out_aabb_max_x,
        out_aabb_max_y,
        out_aabb_max_z
    );
    throw_cuda(cudaGetLastError(), "build_receiver_tiles_kernel launch");
}

} // namespace witwin::channel::native_ext
