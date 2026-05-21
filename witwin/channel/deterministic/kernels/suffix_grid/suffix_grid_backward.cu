#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <suffix_grid/suffix_grid.h>
#include <suffix_grid/suffix_grid_common.h>

namespace witwin::channel::native_ext {
namespace {

using namespace suffix_grid_detail;

using common::throw_cuda;

__global__ void suffix_grid_backward_kernel(
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
    const float* __restrict__ grad_field_re,
    const float* __restrict__ grad_field_im,
    const float* __restrict__ grad_vec_x_re,
    const float* __restrict__ grad_vec_x_im,
    const float* __restrict__ grad_vec_y_re,
    const float* __restrict__ grad_vec_y_im,
    const float* __restrict__ grad_vec_z_re,
    const float* __restrict__ grad_vec_z_im,
    float* __restrict__ grad_seg_origin_x,
    float* __restrict__ grad_seg_origin_y,
    float* __restrict__ grad_seg_origin_z,
    float* __restrict__ grad_seg_field_re,
    float* __restrict__ grad_seg_field_im,
    float* __restrict__ grad_seg_vec_x_re,
    float* __restrict__ grad_seg_vec_x_im,
    float* __restrict__ grad_seg_vec_y_re,
    float* __restrict__ grad_seg_vec_y_im,
    float* __restrict__ grad_seg_vec_z_re,
    float* __restrict__ grad_seg_vec_z_im
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

    Vec3f grad_origin = make_vec3(0.0f, 0.0f, 0.0f);
    Complexf grad_field = make_complex(0.0f, 0.0f);
    Complexf grad_vx = make_complex(0.0f, 0.0f);
    Complexf grad_vy = make_complex(0.0f, 0.0f);
    Complexf grad_vz = make_complex(0.0f, 0.0f);

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
        Vec3f diff = sub(cell_pos, origin);
        Vec3f unit = safe_unit(diff);
        float d = norm(diff) + SG_EPS;
        Complexf factor = distance_factor(wavelength, k, d);

        Complexf g_field = make_complex(grad_field_re[cell_idx], grad_field_im[cell_idx]);
        Complexf g_vec_x = make_complex(grad_vec_x_re[cell_idx], grad_vec_x_im[cell_idx]);
        Complexf g_vec_y = make_complex(grad_vec_y_re[cell_idx], grad_vec_y_im[cell_idx]);
        Complexf g_vec_z = make_complex(grad_vec_z_re[cell_idx], grad_vec_z_im[cell_idx]);
        Complexf grad_factor = make_complex(0.0f, 0.0f);

        suffix_grid_detail::accumulate_complex_backward(field, factor, g_field, grad_field, grad_factor);
        suffix_grid_detail::accumulate_complex_backward(vec_x, factor, g_vec_x, grad_vx, grad_factor);
        suffix_grid_detail::accumulate_complex_backward(vec_y, factor, g_vec_y, grad_vy, grad_factor);
        suffix_grid_detail::accumulate_complex_backward(vec_z, factor, g_vec_z, grad_vz, grad_factor);

        float grad_d = factor_distance_adjoint(wavelength, k, d, grad_factor);
        grad_origin = add(grad_origin, mul(unit, -grad_d));

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

    atomicAdd(grad_seg_origin_x + segment_idx, grad_origin.x);
    atomicAdd(grad_seg_origin_y + segment_idx, grad_origin.y);
    atomicAdd(grad_seg_origin_z + segment_idx, grad_origin.z);
    atomicAdd(grad_seg_field_re + segment_idx, grad_field.re);
    atomicAdd(grad_seg_field_im + segment_idx, grad_field.im);
    atomicAdd(grad_seg_vec_x_re + segment_idx, grad_vx.re);
    atomicAdd(grad_seg_vec_x_im + segment_idx, grad_vx.im);
    atomicAdd(grad_seg_vec_y_re + segment_idx, grad_vy.re);
    atomicAdd(grad_seg_vec_y_im + segment_idx, grad_vy.im);
    atomicAdd(grad_seg_vec_z_re + segment_idx, grad_vz.re);
    atomicAdd(grad_seg_vec_z_im + segment_idx, grad_vz.im);
}

__global__ void suffix_grid_backward_resume_kernel(
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
    const float* __restrict__ grad_field_re,
    const float* __restrict__ grad_field_im,
    const float* __restrict__ grad_vec_x_re,
    const float* __restrict__ grad_vec_x_im,
    const float* __restrict__ grad_vec_y_re,
    const float* __restrict__ grad_vec_y_im,
    const float* __restrict__ grad_vec_z_re,
    const float* __restrict__ grad_vec_z_im,
    float* __restrict__ grad_seg_origin_x,
    float* __restrict__ grad_seg_origin_y,
    float* __restrict__ grad_seg_origin_z,
    float* __restrict__ grad_seg_field_re,
    float* __restrict__ grad_seg_field_im,
    float* __restrict__ grad_seg_vec_x_re,
    float* __restrict__ grad_seg_vec_x_im,
    float* __restrict__ grad_seg_vec_y_re,
    float* __restrict__ grad_seg_vec_y_im,
    float* __restrict__ grad_seg_vec_z_re,
    float* __restrict__ grad_seg_vec_z_im
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

    Vec3f grad_origin = make_vec3(0.0f, 0.0f, 0.0f);
    Complexf grad_field = make_complex(0.0f, 0.0f);
    Complexf grad_vx = make_complex(0.0f, 0.0f);
    Complexf grad_vy = make_complex(0.0f, 0.0f);
    Complexf grad_vz = make_complex(0.0f, 0.0f);

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
        Vec3f diff = sub(cell_pos, origin);
        Vec3f unit = safe_unit(diff);
        float d = norm(diff) + SG_EPS;
        Complexf factor = distance_factor(wavelength, k, d);

        Complexf g_field = make_complex(grad_field_re[cell_idx], grad_field_im[cell_idx]);
        Complexf g_vec_x = make_complex(grad_vec_x_re[cell_idx], grad_vec_x_im[cell_idx]);
        Complexf g_vec_y = make_complex(grad_vec_y_re[cell_idx], grad_vec_y_im[cell_idx]);
        Complexf g_vec_z = make_complex(grad_vec_z_re[cell_idx], grad_vec_z_im[cell_idx]);
        Complexf grad_factor = make_complex(0.0f, 0.0f);

        suffix_grid_detail::accumulate_complex_backward(field, factor, g_field, grad_field, grad_factor);
        suffix_grid_detail::accumulate_complex_backward(vec_x, factor, g_vec_x, grad_vx, grad_factor);
        suffix_grid_detail::accumulate_complex_backward(vec_y, factor, g_vec_y, grad_vy, grad_factor);
        suffix_grid_detail::accumulate_complex_backward(vec_z, factor, g_vec_z, grad_vz, grad_factor);

        float grad_d = factor_distance_adjoint(wavelength, k, d, grad_factor);
        grad_origin = add(grad_origin, mul(unit, -grad_d));

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

    grad_seg_origin_x[tid] = grad_origin.x;
    grad_seg_origin_y[tid] = grad_origin.y;
    grad_seg_origin_z[tid] = grad_origin.z;
    grad_seg_field_re[tid] = grad_field.re;
    grad_seg_field_im[tid] = grad_field.im;
    grad_seg_vec_x_re[tid] = grad_vx.re;
    grad_seg_vec_x_im[tid] = grad_vx.im;
    grad_seg_vec_y_re[tid] = grad_vy.re;
    grad_seg_vec_y_im[tid] = grad_vy.im;
    grad_seg_vec_z_re[tid] = grad_vz.re;
    grad_seg_vec_z_im[tid] = grad_vz.im;
}

__global__ void suffix_grid_backward_resume_batched_kernel(
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
    const float* __restrict__ grad_field_re,
    const float* __restrict__ grad_field_im,
    const float* __restrict__ grad_vec_x_re,
    const float* __restrict__ grad_vec_x_im,
    const float* __restrict__ grad_vec_y_re,
    const float* __restrict__ grad_vec_y_im,
    const float* __restrict__ grad_vec_z_re,
    const float* __restrict__ grad_vec_z_im,
    float* __restrict__ grad_seg_origin_x,
    float* __restrict__ grad_seg_origin_y,
    float* __restrict__ grad_seg_origin_z,
    float* __restrict__ grad_seg_field_re,
    float* __restrict__ grad_seg_field_im,
    float* __restrict__ grad_seg_vec_x_re,
    float* __restrict__ grad_seg_vec_x_im,
    float* __restrict__ grad_seg_vec_y_re,
    float* __restrict__ grad_seg_vec_y_im,
    float* __restrict__ grad_seg_vec_z_re,
    float* __restrict__ grad_seg_vec_z_im
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

    Vec3f grad_origin = make_vec3(0.0f, 0.0f, 0.0f);
    Complexf grad_field = make_complex(0.0f, 0.0f);
    Complexf grad_vx = make_complex(0.0f, 0.0f);
    Complexf grad_vy = make_complex(0.0f, 0.0f);
    Complexf grad_vz = make_complex(0.0f, 0.0f);

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
        Vec3f diff = sub(cell_pos, origin);
        Vec3f unit = safe_unit(diff);
        float d = norm(diff) + SG_EPS;
        Complexf factor = distance_factor(wavelength, k, d);

        Complexf g_field = make_complex(grad_field_re[cell_idx], grad_field_im[cell_idx]);
        Complexf g_vec_x = make_complex(grad_vec_x_re[cell_idx], grad_vec_x_im[cell_idx]);
        Complexf g_vec_y = make_complex(grad_vec_y_re[cell_idx], grad_vec_y_im[cell_idx]);
        Complexf g_vec_z = make_complex(grad_vec_z_re[cell_idx], grad_vec_z_im[cell_idx]);
        Complexf grad_factor = make_complex(0.0f, 0.0f);

        suffix_grid_detail::accumulate_complex_backward(field, factor, g_field, grad_field, grad_factor);
        suffix_grid_detail::accumulate_complex_backward(vec_x, factor, g_vec_x, grad_vx, grad_factor);
        suffix_grid_detail::accumulate_complex_backward(vec_y, factor, g_vec_y, grad_vy, grad_factor);
        suffix_grid_detail::accumulate_complex_backward(vec_z, factor, g_vec_z, grad_vz, grad_factor);

        float grad_d = factor_distance_adjoint(wavelength, k, d, grad_factor);
        grad_origin = add(grad_origin, mul(unit, -grad_d));

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

    atomicAdd(grad_seg_origin_x + segment_idx, grad_origin.x);
    atomicAdd(grad_seg_origin_y + segment_idx, grad_origin.y);
    atomicAdd(grad_seg_origin_z + segment_idx, grad_origin.z);
    atomicAdd(grad_seg_field_re + segment_idx, grad_field.re);
    atomicAdd(grad_seg_field_im + segment_idx, grad_field.im);
    atomicAdd(grad_seg_vec_x_re + segment_idx, grad_vx.re);
    atomicAdd(grad_seg_vec_x_im + segment_idx, grad_vx.im);
    atomicAdd(grad_seg_vec_y_re + segment_idx, grad_vy.re);
    atomicAdd(grad_seg_vec_y_im + segment_idx, grad_vy.im);
    atomicAdd(grad_seg_vec_z_re + segment_idx, grad_vz.re);
    atomicAdd(grad_seg_vec_z_im + segment_idx, grad_vz.im);
}

} // namespace

void suffix_grid_backward(
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
    const float* grad_field_re,
    const float* grad_field_im,
    const float* grad_vec_x_re,
    const float* grad_vec_x_im,
    const float* grad_vec_y_re,
    const float* grad_vec_y_im,
    const float* grad_vec_z_re,
    const float* grad_vec_z_im,
    float* grad_seg_origin_x,
    float* grad_seg_origin_y,
    float* grad_seg_origin_z,
    float* grad_seg_field_re,
    float* grad_seg_field_im,
    float* grad_seg_vec_x_re,
    float* grad_seg_vec_x_im,
    float* grad_seg_vec_y_re,
    float* grad_seg_vec_y_im,
    float* grad_seg_vec_z_re,
    float* grad_seg_vec_z_im
) {
    if (n_rays <= 0) {
        return;
    }
    constexpr int BLOCK = 256;
    int grid = (n_rays + BLOCK - 1) / BLOCK;
    suffix_grid_backward_kernel<<<grid, BLOCK>>>(
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
        grad_field_re,
        grad_field_im,
        grad_vec_x_re,
        grad_vec_x_im,
        grad_vec_y_re,
        grad_vec_y_im,
        grad_vec_z_re,
        grad_vec_z_im,
        grad_seg_origin_x,
        grad_seg_origin_y,
        grad_seg_origin_z,
        grad_seg_field_re,
        grad_seg_field_im,
        grad_seg_vec_x_re,
        grad_seg_vec_x_im,
        grad_seg_vec_y_re,
        grad_seg_vec_y_im,
        grad_seg_vec_z_re,
        grad_seg_vec_z_im
    );
    throw_cuda(cudaGetLastError(), "suffix_grid_backward_kernel launch");
}

void suffix_grid_backward_resume(
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
    const float* grad_field_re,
    const float* grad_field_im,
    const float* grad_vec_x_re,
    const float* grad_vec_x_im,
    const float* grad_vec_y_re,
    const float* grad_vec_y_im,
    const float* grad_vec_z_re,
    const float* grad_vec_z_im,
    float* grad_seg_origin_x,
    float* grad_seg_origin_y,
    float* grad_seg_origin_z,
    float* grad_seg_field_re,
    float* grad_seg_field_im,
    float* grad_seg_vec_x_re,
    float* grad_seg_vec_x_im,
    float* grad_seg_vec_y_re,
    float* grad_seg_vec_y_im,
    float* grad_seg_vec_z_re,
    float* grad_seg_vec_z_im
) {
    if (n_rays <= 0) {
        return;
    }
    constexpr int BLOCK = 256;
    int grid = (n_rays + BLOCK - 1) / BLOCK;
    suffix_grid_backward_resume_kernel<<<grid, BLOCK>>>(
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
        grad_field_re,
        grad_field_im,
        grad_vec_x_re,
        grad_vec_x_im,
        grad_vec_y_re,
        grad_vec_y_im,
        grad_vec_z_re,
        grad_vec_z_im,
        grad_seg_origin_x,
        grad_seg_origin_y,
        grad_seg_origin_z,
        grad_seg_field_re,
        grad_seg_field_im,
        grad_seg_vec_x_re,
        grad_seg_vec_x_im,
        grad_seg_vec_y_re,
        grad_seg_vec_y_im,
        grad_seg_vec_z_re,
        grad_seg_vec_z_im
    );
    throw_cuda(cudaGetLastError(), "suffix_grid_backward_resume_kernel launch");
}

void suffix_grid_backward_resume_batched(
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
    const float* grad_field_re,
    const float* grad_field_im,
    const float* grad_vec_x_re,
    const float* grad_vec_x_im,
    const float* grad_vec_y_re,
    const float* grad_vec_y_im,
    const float* grad_vec_z_re,
    const float* grad_vec_z_im,
    float* grad_seg_origin_x,
    float* grad_seg_origin_y,
    float* grad_seg_origin_z,
    float* grad_seg_field_re,
    float* grad_seg_field_im,
    float* grad_seg_vec_x_re,
    float* grad_seg_vec_x_im,
    float* grad_seg_vec_y_re,
    float* grad_seg_vec_y_im,
    float* grad_seg_vec_z_re,
    float* grad_seg_vec_z_im
) {
    if (n_rays <= 0) {
        return;
    }
    constexpr int BLOCK = 256;
    int grid = (n_rays + BLOCK - 1) / BLOCK;
    suffix_grid_backward_resume_batched_kernel<<<grid, BLOCK>>>(
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
        grad_field_re,
        grad_field_im,
        grad_vec_x_re,
        grad_vec_x_im,
        grad_vec_y_re,
        grad_vec_y_im,
        grad_vec_z_re,
        grad_vec_z_im,
        grad_seg_origin_x,
        grad_seg_origin_y,
        grad_seg_origin_z,
        grad_seg_field_re,
        grad_seg_field_im,
        grad_seg_vec_x_re,
        grad_seg_vec_x_im,
        grad_seg_vec_y_re,
        grad_seg_vec_y_im,
        grad_seg_vec_z_re,
        grad_seg_vec_z_im
    );
    throw_cuda(cudaGetLastError(), "suffix_grid_backward_resume_batched_kernel launch");
}

} // namespace witwin::channel::native_ext
