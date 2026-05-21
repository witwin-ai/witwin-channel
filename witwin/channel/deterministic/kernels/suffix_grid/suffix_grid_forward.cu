#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <suffix_grid/suffix_grid.h>
#include <suffix_grid/suffix_grid_common.h>

namespace witwin::channel::native_ext {
namespace {

using namespace suffix_grid_detail;

using common::throw_cuda;

__global__ void suffix_grid_forward_kernel(
    int plane_axis,
    float plane_position,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int max_steps,
    const float* __restrict__ coord_0,
    const float* __restrict__ coord_1,
    const int* __restrict__ active,
    const int* __restrict__ task_segment_idx,
    const float* __restrict__ seg_origin_x,
    const float* __restrict__ seg_origin_y,
    const float* __restrict__ seg_origin_z,
    const float* __restrict__ seg_dir_x,
    const float* __restrict__ seg_dir_y,
    const float* __restrict__ seg_dir_z,
    const float* __restrict__ blocker_dist,
    const float* __restrict__ seg_field_re,
    const float* __restrict__ seg_field_im,
    const float* __restrict__ seg_vec_x_re,
    const float* __restrict__ seg_vec_x_im,
    const float* __restrict__ seg_vec_y_re,
    const float* __restrict__ seg_vec_y_im,
    const float* __restrict__ seg_vec_z_re,
    const float* __restrict__ seg_vec_z_im,
    float wavelength,
    float k,
    int n_rays,
    float* __restrict__ out_field_re,
    float* __restrict__ out_field_im,
    float* __restrict__ out_vec_x_re,
    float* __restrict__ out_vec_x_im,
    float* __restrict__ out_vec_y_re,
    float* __restrict__ out_vec_y_im,
    float* __restrict__ out_vec_z_re,
    float* __restrict__ out_vec_z_im
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rays || active[tid] == 0) {
        return;
    }

    int segment_idx = task_segment_idx[tid];
    Vec3f origin = make_vec3(seg_origin_x[segment_idx], seg_origin_y[segment_idx], seg_origin_z[segment_idx]);
    Vec3f dir = make_vec3(seg_dir_x[segment_idx], seg_dir_y[segment_idx], seg_dir_z[segment_idx]);
    float dir_0 = 0.0f;
    float dir_1 = 0.0f;
    float cur_coord_0 = 0.0f;
    float cur_coord_1 = 0.0f;
    tangential_components(plane_axis, dir, dir_0, dir_1);
    tangential_components(plane_axis, origin, cur_coord_0, cur_coord_1);

    float abs_dir_0 = fmaxf(fabsf(dir_0), SG_EPS);
    float abs_dir_1 = fmaxf(fabsf(dir_1), SG_EPS);
    float dt_0 = fabsf(cell_size_0 / abs_dir_0);
    float dt_1 = fabsf(cell_size_1 / abs_dir_1);
    float step_0 = dir_0 > 0.0f ? cell_size_0 : -cell_size_0;
    float step_1 = dir_1 > 0.0f ? cell_size_1 : -cell_size_1;

    float next_0 = dir_0 > 0.0f
        ? (floorf((cur_coord_0 - coord_0_min) / cell_size_0) + 1.0f) * cell_size_0 + coord_0_min
        : floorf((cur_coord_0 - coord_0_min) / cell_size_0) * cell_size_0 + coord_0_min;
    float next_1 = dir_1 > 0.0f
        ? (floorf((cur_coord_1 - coord_1_min) / cell_size_1) + 1.0f) * cell_size_1 + coord_1_min
        : floorf((cur_coord_1 - coord_1_min) / cell_size_1) * cell_size_1 + coord_1_min;

    float t = 0.0f;
    float t_max_0 = fabsf((next_0 - cur_coord_0) / abs_dir_0);
    float t_max_1 = fabsf((next_1 - cur_coord_1) / abs_dir_1);
    float ray_blocker = blocker_dist[segment_idx];

    Complexf field = make_complex(seg_field_re[segment_idx], seg_field_im[segment_idx]);
    Complexf vec_x = make_complex(seg_vec_x_re[segment_idx], seg_vec_x_im[segment_idx]);
    Complexf vec_y = make_complex(seg_vec_y_re[segment_idx], seg_vec_y_im[segment_idx]);
    Complexf vec_z = make_complex(seg_vec_z_re[segment_idx], seg_vec_z_im[segment_idx]);

    for (int step = 0; step < max_steps; ++step) {
        bool in_bounds =
            cur_coord_0 >= coord_0_min &&
            cur_coord_0 < coord_0_max &&
            cur_coord_1 >= coord_1_min &&
            cur_coord_1 < coord_1_max &&
            t < ray_blocker;
        if (!in_bounds) {
            break;
        }

        int cell_idx = grid_cell_index(
            cur_coord_0,
            cur_coord_1,
            coord_0_min,
            coord_1_min,
            cell_size_0,
            cell_size_1,
            n_coord_0,
            n_coord_1
        );
        int cell_i0 = cell_idx % n_coord_0;
        int cell_i1 = cell_idx / n_coord_0;
        Vec3f cell_pos = point_on_plane(
            plane_axis,
            plane_position,
            coord_0[cell_i0],
            coord_1[cell_i1]
        );
        float d = norm(sub(cell_pos, origin)) + SG_EPS;
        Complexf factor = distance_factor(wavelength, k, d);
        Complexf field_contrib = cmul(field, factor);
        Complexf vec_x_contrib = cmul(vec_x, factor);
        Complexf vec_y_contrib = cmul(vec_y, factor);
        Complexf vec_z_contrib = cmul(vec_z, factor);

        atomicAdd(out_field_re + cell_idx, field_contrib.re);
        atomicAdd(out_field_im + cell_idx, field_contrib.im);
        atomicAdd(out_vec_x_re + cell_idx, vec_x_contrib.re);
        atomicAdd(out_vec_x_im + cell_idx, vec_x_contrib.im);
        atomicAdd(out_vec_y_re + cell_idx, vec_y_contrib.re);
        atomicAdd(out_vec_y_im + cell_idx, vec_y_contrib.im);
        atomicAdd(out_vec_z_re + cell_idx, vec_z_contrib.re);
        atomicAdd(out_vec_z_im + cell_idx, vec_z_contrib.im);

        bool move_0 = t_max_0 < t_max_1;
        t = move_0 ? t_max_0 : t_max_1;
        if (move_0) {
            cur_coord_0 += step_0;
            t_max_0 += dt_0;
        } else {
            cur_coord_1 += step_1;
            t_max_1 += dt_1;
        }
    }
}

__global__ void suffix_grid_forward_resume_kernel(
    int plane_axis,
    float plane_position,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int max_steps,
    const float* __restrict__ coord_0,
    const float* __restrict__ coord_1,
    const int* __restrict__ active,
    const float* __restrict__ seg_origin_x,
    const float* __restrict__ seg_origin_y,
    const float* __restrict__ seg_origin_z,
    const float* __restrict__ seg_dir_x,
    const float* __restrict__ seg_dir_y,
    const float* __restrict__ seg_dir_z,
    const float* __restrict__ blocker_dist,
    const float* __restrict__ seg_field_re,
    const float* __restrict__ seg_field_im,
    const float* __restrict__ seg_vec_x_re,
    const float* __restrict__ seg_vec_x_im,
    const float* __restrict__ seg_vec_y_re,
    const float* __restrict__ seg_vec_y_im,
    const float* __restrict__ seg_vec_z_re,
    const float* __restrict__ seg_vec_z_im,
    const float* __restrict__ trace_coord_0,
    const float* __restrict__ trace_coord_1,
    const float* __restrict__ trace_t,
    const float* __restrict__ trace_t_max_0,
    const float* __restrict__ trace_t_max_1,
    float wavelength,
    float k,
    int n_rays,
    float* __restrict__ out_field_re,
    float* __restrict__ out_field_im,
    float* __restrict__ out_vec_x_re,
    float* __restrict__ out_vec_x_im,
    float* __restrict__ out_vec_y_re,
    float* __restrict__ out_vec_y_im,
    float* __restrict__ out_vec_z_re,
    float* __restrict__ out_vec_z_im
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rays || active[tid] == 0) {
        return;
    }

    Vec3f origin = make_vec3(seg_origin_x[tid], seg_origin_y[tid], seg_origin_z[tid]);
    Vec3f dir = make_vec3(seg_dir_x[tid], seg_dir_y[tid], seg_dir_z[tid]);
    float dir_0 = 0.0f;
    float dir_1 = 0.0f;
    tangential_components(plane_axis, dir, dir_0, dir_1);

    float cur_coord_0 = trace_coord_0[tid];
    float cur_coord_1 = trace_coord_1[tid];
    float t = trace_t[tid];

    float abs_dir_0 = fmaxf(fabsf(dir_0), SG_EPS);
    float abs_dir_1 = fmaxf(fabsf(dir_1), SG_EPS);
    float dt_0 = fabsf(cell_size_0 / abs_dir_0);
    float dt_1 = fabsf(cell_size_1 / abs_dir_1);
    float step_0 = dir_0 > 0.0f ? cell_size_0 : -cell_size_0;
    float step_1 = dir_1 > 0.0f ? cell_size_1 : -cell_size_1;
    float t_max_0 = trace_t_max_0[tid];
    float t_max_1 = trace_t_max_1[tid];
    float ray_blocker = blocker_dist[tid];

    Complexf field = make_complex(seg_field_re[tid], seg_field_im[tid]);
    Complexf vec_x = make_complex(seg_vec_x_re[tid], seg_vec_x_im[tid]);
    Complexf vec_y = make_complex(seg_vec_y_re[tid], seg_vec_y_im[tid]);
    Complexf vec_z = make_complex(seg_vec_z_re[tid], seg_vec_z_im[tid]);

    for (int step = 0; step < max_steps; ++step) {
        bool in_bounds =
            cur_coord_0 >= coord_0_min &&
            cur_coord_0 < coord_0_max &&
            cur_coord_1 >= coord_1_min &&
            cur_coord_1 < coord_1_max &&
            t < ray_blocker;
        if (!in_bounds) {
            break;
        }

        int cell_idx = grid_cell_index(
            cur_coord_0,
            cur_coord_1,
            coord_0_min,
            coord_1_min,
            cell_size_0,
            cell_size_1,
            n_coord_0,
            n_coord_1
        );
        int cell_i0 = cell_idx % n_coord_0;
        int cell_i1 = cell_idx / n_coord_0;
        Vec3f cell_pos = point_on_plane(
            plane_axis,
            plane_position,
            coord_0[cell_i0],
            coord_1[cell_i1]
        );
        float d = norm(sub(cell_pos, origin)) + SG_EPS;
        Complexf factor = distance_factor(wavelength, k, d);
        Complexf field_contrib = cmul(field, factor);
        Complexf vec_x_contrib = cmul(vec_x, factor);
        Complexf vec_y_contrib = cmul(vec_y, factor);
        Complexf vec_z_contrib = cmul(vec_z, factor);

        atomicAdd(out_field_re + cell_idx, field_contrib.re);
        atomicAdd(out_field_im + cell_idx, field_contrib.im);
        atomicAdd(out_vec_x_re + cell_idx, vec_x_contrib.re);
        atomicAdd(out_vec_x_im + cell_idx, vec_x_contrib.im);
        atomicAdd(out_vec_y_re + cell_idx, vec_y_contrib.re);
        atomicAdd(out_vec_y_im + cell_idx, vec_y_contrib.im);
        atomicAdd(out_vec_z_re + cell_idx, vec_z_contrib.re);
        atomicAdd(out_vec_z_im + cell_idx, vec_z_contrib.im);

        bool move_0 = t_max_0 < t_max_1;
        t = move_0 ? t_max_0 : t_max_1;
        if (move_0) {
            cur_coord_0 += step_0;
            t_max_0 += dt_0;
        } else {
            cur_coord_1 += step_1;
            t_max_1 += dt_1;
        }
    }
}

__global__ void suffix_grid_forward_resume_batched_kernel(
    int plane_axis,
    float plane_position,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int max_steps,
    const float* __restrict__ coord_0,
    const float* __restrict__ coord_1,
    const int* __restrict__ active,
    const int* __restrict__ task_segment_idx,
    const float* __restrict__ seg_origin_x,
    const float* __restrict__ seg_origin_y,
    const float* __restrict__ seg_origin_z,
    const float* __restrict__ seg_dir_x,
    const float* __restrict__ seg_dir_y,
    const float* __restrict__ seg_dir_z,
    const float* __restrict__ blocker_dist,
    const float* __restrict__ seg_field_re,
    const float* __restrict__ seg_field_im,
    const float* __restrict__ seg_vec_x_re,
    const float* __restrict__ seg_vec_x_im,
    const float* __restrict__ seg_vec_y_re,
    const float* __restrict__ seg_vec_y_im,
    const float* __restrict__ seg_vec_z_re,
    const float* __restrict__ seg_vec_z_im,
    const float* __restrict__ trace_coord_0,
    const float* __restrict__ trace_coord_1,
    const float* __restrict__ trace_t,
    const float* __restrict__ trace_t_max_0,
    const float* __restrict__ trace_t_max_1,
    const int* __restrict__ tile_i0,
    const int* __restrict__ tile_i1,
    const int* __restrict__ tile_extent_0,
    const int* __restrict__ tile_extent_1,
    float wavelength,
    float k,
    int n_rays,
    float* __restrict__ out_field_re,
    float* __restrict__ out_field_im,
    float* __restrict__ out_vec_x_re,
    float* __restrict__ out_vec_x_im,
    float* __restrict__ out_vec_y_re,
    float* __restrict__ out_vec_y_im,
    float* __restrict__ out_vec_z_re,
    float* __restrict__ out_vec_z_im
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rays || active[tid] == 0) {
        return;
    }

    int segment_idx = task_segment_idx[tid];
    Vec3f origin = make_vec3(seg_origin_x[segment_idx], seg_origin_y[segment_idx], seg_origin_z[segment_idx]);
    Vec3f dir = make_vec3(seg_dir_x[segment_idx], seg_dir_y[segment_idx], seg_dir_z[segment_idx]);
    float dir_0 = 0.0f;
    float dir_1 = 0.0f;
    tangential_components(plane_axis, dir, dir_0, dir_1);

    float cur_coord_0 = trace_coord_0[tid];
    float cur_coord_1 = trace_coord_1[tid];
    float t = trace_t[tid];

    float abs_dir_0 = fmaxf(fabsf(dir_0), SG_EPS);
    float abs_dir_1 = fmaxf(fabsf(dir_1), SG_EPS);
    float dt_0 = fabsf(cell_size_0 / abs_dir_0);
    float dt_1 = fabsf(cell_size_1 / abs_dir_1);
    float step_0 = dir_0 > 0.0f ? cell_size_0 : -cell_size_0;
    float step_1 = dir_1 > 0.0f ? cell_size_1 : -cell_size_1;
    float t_max_0 = trace_t_max_0[tid];
    float t_max_1 = trace_t_max_1[tid];
    float ray_blocker = blocker_dist[segment_idx];

    int task_tile_i0 = tile_i0[tid];
    int task_tile_i1 = tile_i1[tid];
    int task_tile_extent_0 = tile_extent_0[tid];
    int task_tile_extent_1 = tile_extent_1[tid];
    float task_coord_0_min = coord_0_min + static_cast<float>(task_tile_i0) * cell_size_0;
    float task_coord_0_max = fminf(
        coord_0_max,
        task_coord_0_min + static_cast<float>(task_tile_extent_0) * cell_size_0
    );
    float task_coord_1_min = coord_1_min + static_cast<float>(task_tile_i1) * cell_size_1;
    float task_coord_1_max = fminf(
        coord_1_max,
        task_coord_1_min + static_cast<float>(task_tile_extent_1) * cell_size_1
    );

    Complexf field = make_complex(seg_field_re[segment_idx], seg_field_im[segment_idx]);
    Complexf vec_x = make_complex(seg_vec_x_re[segment_idx], seg_vec_x_im[segment_idx]);
    Complexf vec_y = make_complex(seg_vec_y_re[segment_idx], seg_vec_y_im[segment_idx]);
    Complexf vec_z = make_complex(seg_vec_z_re[segment_idx], seg_vec_z_im[segment_idx]);

    for (int step = 0; step < max_steps; ++step) {
        bool in_bounds =
            cur_coord_0 >= task_coord_0_min &&
            cur_coord_0 < task_coord_0_max &&
            cur_coord_1 >= task_coord_1_min &&
            cur_coord_1 < task_coord_1_max &&
            t < ray_blocker;
        if (!in_bounds) {
            break;
        }

        int cell_idx = grid_cell_index(
            cur_coord_0,
            cur_coord_1,
            coord_0_min,
            coord_1_min,
            cell_size_0,
            cell_size_1,
            n_coord_0,
            n_coord_1
        );
        int cell_i0 = cell_idx % n_coord_0;
        int cell_i1 = cell_idx / n_coord_0;
        Vec3f cell_pos = point_on_plane(
            plane_axis,
            plane_position,
            coord_0[cell_i0],
            coord_1[cell_i1]
        );
        float d = norm(sub(cell_pos, origin)) + SG_EPS;
        Complexf factor = distance_factor(wavelength, k, d);
        Complexf field_contrib = cmul(field, factor);
        Complexf vec_x_contrib = cmul(vec_x, factor);
        Complexf vec_y_contrib = cmul(vec_y, factor);
        Complexf vec_z_contrib = cmul(vec_z, factor);

        atomicAdd(out_field_re + cell_idx, field_contrib.re);
        atomicAdd(out_field_im + cell_idx, field_contrib.im);
        atomicAdd(out_vec_x_re + cell_idx, vec_x_contrib.re);
        atomicAdd(out_vec_x_im + cell_idx, vec_x_contrib.im);
        atomicAdd(out_vec_y_re + cell_idx, vec_y_contrib.re);
        atomicAdd(out_vec_y_im + cell_idx, vec_y_contrib.im);
        atomicAdd(out_vec_z_re + cell_idx, vec_z_contrib.re);
        atomicAdd(out_vec_z_im + cell_idx, vec_z_contrib.im);

        bool move_0 = t_max_0 < t_max_1;
        t = move_0 ? t_max_0 : t_max_1;
        if (move_0) {
            cur_coord_0 += step_0;
            t_max_0 += dt_0;
        } else {
            cur_coord_1 += step_1;
            t_max_1 += dt_1;
        }
    }
}

void launch_suffix_grid_forward(
    int plane_axis,
    float plane_position,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int max_steps,
    const float* coord_0,
    const float* coord_1,
    const int* active,
    const int* task_segment_idx,
    const float* seg_origin_x,
    const float* seg_origin_y,
    const float* seg_origin_z,
    const float* seg_dir_x,
    const float* seg_dir_y,
    const float* seg_dir_z,
    const float* blocker_dist,
    const float* seg_field_re,
    const float* seg_field_im,
    const float* seg_vec_x_re,
    const float* seg_vec_x_im,
    const float* seg_vec_y_re,
    const float* seg_vec_y_im,
    const float* seg_vec_z_re,
    const float* seg_vec_z_im,
    float wavelength,
    float k,
    int n_rays,
    float* out_field_re,
    float* out_field_im,
    float* out_vec_x_re,
    float* out_vec_x_im,
    float* out_vec_y_re,
    float* out_vec_y_im,
    float* out_vec_z_re,
    float* out_vec_z_im
) {
    if (n_rays <= 0) {
        return;
    }
    constexpr int BLOCK = 256;
    int grid = (n_rays + BLOCK - 1) / BLOCK;
    suffix_grid_forward_kernel<<<grid, BLOCK>>>(
        plane_axis,
        plane_position,
        coord_0_min,
        coord_0_max,
        coord_1_min,
        coord_1_max,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1,
        max_steps,
        coord_0,
        coord_1,
        active,
        task_segment_idx,
        seg_origin_x,
        seg_origin_y,
        seg_origin_z,
        seg_dir_x,
        seg_dir_y,
        seg_dir_z,
        blocker_dist,
        seg_field_re,
        seg_field_im,
        seg_vec_x_re,
        seg_vec_x_im,
        seg_vec_y_re,
        seg_vec_y_im,
        seg_vec_z_re,
        seg_vec_z_im,
        wavelength,
        k,
        n_rays,
        out_field_re,
        out_field_im,
        out_vec_x_re,
        out_vec_x_im,
        out_vec_y_re,
        out_vec_y_im,
        out_vec_z_re,
        out_vec_z_im
    );
    throw_cuda(cudaGetLastError(), "suffix_grid_forward_kernel launch");
}

void launch_suffix_grid_forward_resume(
    int plane_axis,
    float plane_position,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int max_steps,
    const float* coord_0,
    const float* coord_1,
    const int* active,
    const float* seg_origin_x,
    const float* seg_origin_y,
    const float* seg_origin_z,
    const float* seg_dir_x,
    const float* seg_dir_y,
    const float* seg_dir_z,
    const float* blocker_dist,
    const float* seg_field_re,
    const float* seg_field_im,
    const float* seg_vec_x_re,
    const float* seg_vec_x_im,
    const float* seg_vec_y_re,
    const float* seg_vec_y_im,
    const float* seg_vec_z_re,
    const float* seg_vec_z_im,
    const float* trace_coord_0,
    const float* trace_coord_1,
    const float* trace_t,
    const float* trace_t_max_0,
    const float* trace_t_max_1,
    float wavelength,
    float k,
    int n_rays,
    float* out_field_re,
    float* out_field_im,
    float* out_vec_x_re,
    float* out_vec_x_im,
    float* out_vec_y_re,
    float* out_vec_y_im,
    float* out_vec_z_re,
    float* out_vec_z_im
) {
    if (n_rays <= 0) {
        return;
    }
    constexpr int BLOCK = 256;
    int grid = (n_rays + BLOCK - 1) / BLOCK;
    suffix_grid_forward_resume_kernel<<<grid, BLOCK>>>(
        plane_axis,
        plane_position,
        coord_0_min,
        coord_0_max,
        coord_1_min,
        coord_1_max,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1,
        max_steps,
        coord_0,
        coord_1,
        active,
        seg_origin_x,
        seg_origin_y,
        seg_origin_z,
        seg_dir_x,
        seg_dir_y,
        seg_dir_z,
        blocker_dist,
        seg_field_re,
        seg_field_im,
        seg_vec_x_re,
        seg_vec_x_im,
        seg_vec_y_re,
        seg_vec_y_im,
        seg_vec_z_re,
        seg_vec_z_im,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        wavelength,
        k,
        n_rays,
        out_field_re,
        out_field_im,
        out_vec_x_re,
        out_vec_x_im,
        out_vec_y_re,
        out_vec_y_im,
        out_vec_z_re,
        out_vec_z_im
    );
    throw_cuda(cudaGetLastError(), "suffix_grid_forward_resume_kernel launch");
}

void launch_suffix_grid_forward_resume_batched(
    int plane_axis,
    float plane_position,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int max_steps,
    const float* coord_0,
    const float* coord_1,
    const int* active,
    const int* task_segment_idx,
    const float* seg_origin_x,
    const float* seg_origin_y,
    const float* seg_origin_z,
    const float* seg_dir_x,
    const float* seg_dir_y,
    const float* seg_dir_z,
    const float* blocker_dist,
    const float* seg_field_re,
    const float* seg_field_im,
    const float* seg_vec_x_re,
    const float* seg_vec_x_im,
    const float* seg_vec_y_re,
    const float* seg_vec_y_im,
    const float* seg_vec_z_re,
    const float* seg_vec_z_im,
    const float* trace_coord_0,
    const float* trace_coord_1,
    const float* trace_t,
    const float* trace_t_max_0,
    const float* trace_t_max_1,
    const int* tile_i0,
    const int* tile_i1,
    const int* tile_extent_0,
    const int* tile_extent_1,
    float wavelength,
    float k,
    int n_rays,
    float* out_field_re,
    float* out_field_im,
    float* out_vec_x_re,
    float* out_vec_x_im,
    float* out_vec_y_re,
    float* out_vec_y_im,
    float* out_vec_z_re,
    float* out_vec_z_im
) {
    if (n_rays <= 0) {
        return;
    }
    constexpr int BLOCK = 256;
    int grid = (n_rays + BLOCK - 1) / BLOCK;
    suffix_grid_forward_resume_batched_kernel<<<grid, BLOCK>>>(
        plane_axis,
        plane_position,
        coord_0_min,
        coord_0_max,
        coord_1_min,
        coord_1_max,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1,
        max_steps,
        coord_0,
        coord_1,
        active,
        task_segment_idx,
        seg_origin_x,
        seg_origin_y,
        seg_origin_z,
        seg_dir_x,
        seg_dir_y,
        seg_dir_z,
        blocker_dist,
        seg_field_re,
        seg_field_im,
        seg_vec_x_re,
        seg_vec_x_im,
        seg_vec_y_re,
        seg_vec_y_im,
        seg_vec_z_re,
        seg_vec_z_im,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
        wavelength,
        k,
        n_rays,
        out_field_re,
        out_field_im,
        out_vec_x_re,
        out_vec_x_im,
        out_vec_y_re,
        out_vec_y_im,
        out_vec_z_re,
        out_vec_z_im
    );
    throw_cuda(cudaGetLastError(), "suffix_grid_forward_resume_batched_kernel launch");
}

} // namespace

void suffix_grid_forward(
    int plane_axis,
    float plane_position,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int max_steps,
    const float* coord_0,
    const float* coord_1,
    const int* active,
    const int* task_segment_idx,
    const float* seg_origin_x,
    const float* seg_origin_y,
    const float* seg_origin_z,
    const float* seg_dir_x,
    const float* seg_dir_y,
    const float* seg_dir_z,
    const float* blocker_dist,
    const float* seg_field_re,
    const float* seg_field_im,
    const float* seg_vec_x_re,
    const float* seg_vec_x_im,
    const float* seg_vec_y_re,
    const float* seg_vec_y_im,
    const float* seg_vec_z_re,
    const float* seg_vec_z_im,
    float wavelength,
    float k,
    int n_rays,
    float* out_field_re,
    float* out_field_im,
    float* out_vec_x_re,
    float* out_vec_x_im,
    float* out_vec_y_re,
    float* out_vec_y_im,
    float* out_vec_z_re,
    float* out_vec_z_im
) {
    launch_suffix_grid_forward(
        plane_axis,
        plane_position,
        coord_0_min,
        coord_0_max,
        coord_1_min,
        coord_1_max,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1,
        max_steps,
        coord_0,
        coord_1,
        active,
        task_segment_idx,
        seg_origin_x,
        seg_origin_y,
        seg_origin_z,
        seg_dir_x,
        seg_dir_y,
        seg_dir_z,
        blocker_dist,
        seg_field_re,
        seg_field_im,
        seg_vec_x_re,
        seg_vec_x_im,
        seg_vec_y_re,
        seg_vec_y_im,
        seg_vec_z_re,
        seg_vec_z_im,
        wavelength,
        k,
        n_rays,
        out_field_re,
        out_field_im,
        out_vec_x_re,
        out_vec_x_im,
        out_vec_y_re,
        out_vec_y_im,
        out_vec_z_re,
        out_vec_z_im
    );
}

void suffix_grid_forward_resume(
    int plane_axis,
    float plane_position,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int max_steps,
    const float* coord_0,
    const float* coord_1,
    const int* active,
    const float* seg_origin_x,
    const float* seg_origin_y,
    const float* seg_origin_z,
    const float* seg_dir_x,
    const float* seg_dir_y,
    const float* seg_dir_z,
    const float* blocker_dist,
    const float* seg_field_re,
    const float* seg_field_im,
    const float* seg_vec_x_re,
    const float* seg_vec_x_im,
    const float* seg_vec_y_re,
    const float* seg_vec_y_im,
    const float* seg_vec_z_re,
    const float* seg_vec_z_im,
    const float* trace_coord_0,
    const float* trace_coord_1,
    const float* trace_t,
    const float* trace_t_max_0,
    const float* trace_t_max_1,
    float wavelength,
    float k,
    int n_rays,
    float* out_field_re,
    float* out_field_im,
    float* out_vec_x_re,
    float* out_vec_x_im,
    float* out_vec_y_re,
    float* out_vec_y_im,
    float* out_vec_z_re,
    float* out_vec_z_im
) {
    launch_suffix_grid_forward_resume(
        plane_axis,
        plane_position,
        coord_0_min,
        coord_0_max,
        coord_1_min,
        coord_1_max,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1,
        max_steps,
        coord_0,
        coord_1,
        active,
        seg_origin_x,
        seg_origin_y,
        seg_origin_z,
        seg_dir_x,
        seg_dir_y,
        seg_dir_z,
        blocker_dist,
        seg_field_re,
        seg_field_im,
        seg_vec_x_re,
        seg_vec_x_im,
        seg_vec_y_re,
        seg_vec_y_im,
        seg_vec_z_re,
        seg_vec_z_im,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        wavelength,
        k,
        n_rays,
        out_field_re,
        out_field_im,
        out_vec_x_re,
        out_vec_x_im,
        out_vec_y_re,
        out_vec_y_im,
        out_vec_z_re,
        out_vec_z_im
    );
}

void suffix_grid_forward_resume_batched(
    int plane_axis,
    float plane_position,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int max_steps,
    const float* coord_0,
    const float* coord_1,
    const int* active,
    const int* task_segment_idx,
    const float* seg_origin_x,
    const float* seg_origin_y,
    const float* seg_origin_z,
    const float* seg_dir_x,
    const float* seg_dir_y,
    const float* seg_dir_z,
    const float* blocker_dist,
    const float* seg_field_re,
    const float* seg_field_im,
    const float* seg_vec_x_re,
    const float* seg_vec_x_im,
    const float* seg_vec_y_re,
    const float* seg_vec_y_im,
    const float* seg_vec_z_re,
    const float* seg_vec_z_im,
    const float* trace_coord_0,
    const float* trace_coord_1,
    const float* trace_t,
    const float* trace_t_max_0,
    const float* trace_t_max_1,
    const int* tile_i0,
    const int* tile_i1,
    const int* tile_extent_0,
    const int* tile_extent_1,
    float wavelength,
    float k,
    int n_rays,
    float* out_field_re,
    float* out_field_im,
    float* out_vec_x_re,
    float* out_vec_x_im,
    float* out_vec_y_re,
    float* out_vec_y_im,
    float* out_vec_z_re,
    float* out_vec_z_im
) {
    launch_suffix_grid_forward_resume_batched(
        plane_axis,
        plane_position,
        coord_0_min,
        coord_0_max,
        coord_1_min,
        coord_1_max,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1,
        max_steps,
        coord_0,
        coord_1,
        active,
        task_segment_idx,
        seg_origin_x,
        seg_origin_y,
        seg_origin_z,
        seg_dir_x,
        seg_dir_y,
        seg_dir_z,
        blocker_dist,
        seg_field_re,
        seg_field_im,
        seg_vec_x_re,
        seg_vec_x_im,
        seg_vec_y_re,
        seg_vec_y_im,
        seg_vec_z_re,
        seg_vec_z_im,
        trace_coord_0,
        trace_coord_1,
        trace_t,
        trace_t_max_0,
        trace_t_max_1,
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
        wavelength,
        k,
        n_rays,
        out_field_re,
        out_field_im,
        out_vec_x_re,
        out_vec_x_im,
        out_vec_y_re,
        out_vec_y_im,
        out_vec_z_re,
        out_vec_z_im
    );
}

} // namespace witwin::channel::native_ext
