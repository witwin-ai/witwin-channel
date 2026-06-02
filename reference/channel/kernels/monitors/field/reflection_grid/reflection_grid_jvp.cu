#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <monitors/field/reflection_grid/reflection_grid.h>
#include <monitors/field/reflection_grid/reflection_grid_common.h>

namespace witwin::channel::native_ext {
namespace {

using namespace reflection_grid_detail;

using common::throw_cuda;

__device__ __forceinline__ void accumulate_reflection_jvp_output(
    int cell_idx,
    Vec3f cell_pos,
    Vec3f prev_refl_p,
    Vec3f prev_refl_n,
    Vec3f prev_tx,
    Complexf prev_weight,
    Complexf prev_pol_x,
    Complexf prev_pol_y,
    Complexf prev_pol_z,
    Vec3f t_prev_refl_p,
    Vec3f t_prev_refl_n,
    Vec3f t_prev_tx,
    Complexf t_prev_weight,
    Complexf t_prev_pol_x,
    Complexf t_prev_pol_y,
    Complexf t_prev_pol_z,
    int prev_prim_idx,
    int validate_paths,
    int has_mesh_data,
    const float* tri_v0_x,
    const float* tri_v0_y,
    const float* tri_v0_z,
    const float* tri_v1_x,
    const float* tri_v1_y,
    const float* tri_v1_z,
    const float* tri_v2_x,
    const float* tri_v2_y,
    const float* tri_v2_z,
    const int* tri_group_size,
    const int* tri_group_members,
    int max_group_size,
    float wavelength,
    float k,
    float* out_field_re,
    float* out_field_im,
    float* out_pol_x_re,
    float* out_pol_x_im,
    float* out_pol_y_re,
    float* out_pol_y_im,
    float* out_pol_z_re,
    float* out_pol_z_im
) {
    Vec3f mirror = mirror_point(prev_tx, prev_refl_p, prev_refl_n);
    if (!validate_reflection_path(
            validate_paths,
            has_mesh_data,
            cell_pos,
            mirror,
            prev_refl_p,
            prev_refl_n,
            prev_prim_idx,
            tri_v0_x,
            tri_v0_y,
            tri_v0_z,
            tri_v1_x,
            tri_v1_y,
            tri_v1_z,
            tri_v2_x,
            tri_v2_y,
            tri_v2_z,
            tri_group_size,
            tri_group_members,
            max_group_size)) {
        return;
    }

    Vec3f mirror_t = mirror_tangent(
        prev_tx,
        prev_refl_p,
        prev_refl_n,
        t_prev_tx,
        t_prev_refl_p,
        t_prev_refl_n
    );
    Vec3f diff = sub(cell_pos, mirror);
    Vec3f unit = safe_unit(diff);
    float d = norm(diff);
    float d_tangent = -dot(unit, mirror_t);
    Complexf factor = distance_factor(wavelength, k, d);
    Complexf factor_t = distance_factor_tangent(wavelength, k, d, d_tangent);

    Complexf field = cadd(cmul(t_prev_weight, factor), cmul(prev_weight, factor_t));
    Complexf pol_x = cadd(cmul(t_prev_pol_x, factor), cmul(prev_pol_x, factor_t));
    Complexf pol_y = cadd(cmul(t_prev_pol_y, factor), cmul(prev_pol_y, factor_t));
    Complexf pol_z = cadd(cmul(t_prev_pol_z, factor), cmul(prev_pol_z, factor_t));

    atomicAdd(out_field_re + cell_idx, field.re);
    atomicAdd(out_field_im + cell_idx, field.im);
    atomicAdd(out_pol_x_re + cell_idx, pol_x.re);
    atomicAdd(out_pol_x_im + cell_idx, pol_x.im);
    atomicAdd(out_pol_y_re + cell_idx, pol_y.re);
    atomicAdd(out_pol_y_im + cell_idx, pol_y.im);
    atomicAdd(out_pol_z_re + cell_idx, pol_z.re);
    atomicAdd(out_pol_z_im + cell_idx, pol_z.im);
}

__global__ void reflection_grid_jvp_kernel(
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
    const float* __restrict__ ray_origin_x,
    const float* __restrict__ ray_origin_y,
    const float* __restrict__ ray_origin_z,
    const float* __restrict__ ray_dir_x,
    const float* __restrict__ ray_dir_y,
    const float* __restrict__ ray_dir_z,
    const float* __restrict__ blocker_dist,
    const float* __restrict__ prev_refl_p_x,
    const float* __restrict__ prev_refl_p_y,
    const float* __restrict__ prev_refl_p_z,
    const float* __restrict__ prev_refl_n_x,
    const float* __restrict__ prev_refl_n_y,
    const float* __restrict__ prev_refl_n_z,
    const float* __restrict__ prev_tx_x,
    const float* __restrict__ prev_tx_y,
    const float* __restrict__ prev_tx_z,
    const float* __restrict__ prev_weight_re,
    const float* __restrict__ prev_weight_im,
    const float* __restrict__ prev_pol_x_re,
    const float* __restrict__ prev_pol_x_im,
    const float* __restrict__ prev_pol_y_re,
    const float* __restrict__ prev_pol_y_im,
    const float* __restrict__ prev_pol_z_re,
    const float* __restrict__ prev_pol_z_im,
    const int* __restrict__ prev_prim_idx,
    int validate_paths,
    int has_mesh_data,
    const float* __restrict__ tri_v0_x,
    const float* __restrict__ tri_v0_y,
    const float* __restrict__ tri_v0_z,
    const float* __restrict__ tri_v1_x,
    const float* __restrict__ tri_v1_y,
    const float* __restrict__ tri_v1_z,
    const float* __restrict__ tri_v2_x,
    const float* __restrict__ tri_v2_y,
    const float* __restrict__ tri_v2_z,
    const int* __restrict__ tri_group_size,
    const int* __restrict__ tri_group_members,
    int max_group_size,
    const float* __restrict__ t_prev_refl_p_x,
    const float* __restrict__ t_prev_refl_p_y,
    const float* __restrict__ t_prev_refl_p_z,
    const float* __restrict__ t_prev_refl_n_x,
    const float* __restrict__ t_prev_refl_n_y,
    const float* __restrict__ t_prev_refl_n_z,
    const float* __restrict__ t_prev_tx_x,
    const float* __restrict__ t_prev_tx_y,
    const float* __restrict__ t_prev_tx_z,
    const float* __restrict__ t_prev_weight_re,
    const float* __restrict__ t_prev_weight_im,
    const float* __restrict__ t_prev_pol_x_re,
    const float* __restrict__ t_prev_pol_x_im,
    const float* __restrict__ t_prev_pol_y_re,
    const float* __restrict__ t_prev_pol_y_im,
    const float* __restrict__ t_prev_pol_z_re,
    const float* __restrict__ t_prev_pol_z_im,
    float wavelength,
    float k,
    int n_rays,
    float* __restrict__ out_field_re,
    float* __restrict__ out_field_im,
    float* __restrict__ out_pol_x_re,
    float* __restrict__ out_pol_x_im,
    float* __restrict__ out_pol_y_re,
    float* __restrict__ out_pol_y_im,
    float* __restrict__ out_pol_z_re,
    float* __restrict__ out_pol_z_im
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rays || active[tid] == 0) {
        return;
    }

    Vec3f ray_origin = make_vec3(ray_origin_x[tid], ray_origin_y[tid], ray_origin_z[tid]);
    Vec3f ray_dir = make_vec3(ray_dir_x[tid], ray_dir_y[tid], ray_dir_z[tid]);
    Vec3f prev_refl_p = make_vec3(prev_refl_p_x[tid], prev_refl_p_y[tid], prev_refl_p_z[tid]);
    Vec3f prev_refl_n = make_vec3(prev_refl_n_x[tid], prev_refl_n_y[tid], prev_refl_n_z[tid]);
    Vec3f prev_tx = make_vec3(prev_tx_x[tid], prev_tx_y[tid], prev_tx_z[tid]);
    Vec3f t_prev_refl_p = make_vec3(t_prev_refl_p_x[tid], t_prev_refl_p_y[tid], t_prev_refl_p_z[tid]);
    Vec3f t_prev_refl_n = make_vec3(t_prev_refl_n_x[tid], t_prev_refl_n_y[tid], t_prev_refl_n_z[tid]);
    Vec3f t_prev_tx = make_vec3(t_prev_tx_x[tid], t_prev_tx_y[tid], t_prev_tx_z[tid]);
    Complexf prev_weight = make_complex(prev_weight_re[tid], prev_weight_im[tid]);
    Complexf t_prev_weight = make_complex(t_prev_weight_re[tid], t_prev_weight_im[tid]);
    Complexf prev_pol_x = make_complex(prev_pol_x_re[tid], prev_pol_x_im[tid]);
    Complexf prev_pol_y = make_complex(prev_pol_y_re[tid], prev_pol_y_im[tid]);
    Complexf prev_pol_z = make_complex(prev_pol_z_re[tid], prev_pol_z_im[tid]);
    Complexf t_prev_pol_x = make_complex(t_prev_pol_x_re[tid], t_prev_pol_x_im[tid]);
    Complexf t_prev_pol_y = make_complex(t_prev_pol_y_re[tid], t_prev_pol_y_im[tid]);
    Complexf t_prev_pol_z = make_complex(t_prev_pol_z_re[tid], t_prev_pol_z_im[tid]);
    int prim_idx = prev_prim_idx[tid];

    if (plane_axis == 2) {
        float cur_coord_0 = ray_origin.x;
        float cur_coord_1 = ray_origin.y;
        float dir_0 = ray_dir.x;
        float dir_1 = ray_dir.y;
        float abs_dir_0 = fmaxf(fabsf(dir_0), RG_EPS);
        float abs_dir_1 = fmaxf(fabsf(dir_1), RG_EPS);
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
        float ray_blocker = blocker_dist[tid];

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
            Vec3f cell_pos = make_vec3(coord_0[cell_i0], coord_1[cell_i1], plane_position);
            accumulate_reflection_jvp_output(
                cell_idx,
                cell_pos,
                prev_refl_p,
                prev_refl_n,
                prev_tx,
                prev_weight,
                prev_pol_x,
                prev_pol_y,
                prev_pol_z,
                t_prev_refl_p,
                t_prev_refl_n,
                t_prev_tx,
                t_prev_weight,
                t_prev_pol_x,
                t_prev_pol_y,
                t_prev_pol_z,
                prim_idx,
                validate_paths,
                has_mesh_data,
                tri_v0_x,
                tri_v0_y,
                tri_v0_z,
                tri_v1_x,
                tri_v1_y,
                tri_v1_z,
                tri_v2_x,
                tri_v2_y,
                tri_v2_z,
                tri_group_size,
                tri_group_members,
                max_group_size,
                wavelength,
                k,
                out_field_re,
                out_field_im,
                out_pol_x_re,
                out_pol_x_im,
                out_pol_y_re,
                out_pol_y_im,
                out_pol_z_re,
                out_pol_z_im
            );

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
        return;
    }

    float ray_coord_0 = 0.0f;
    float ray_coord_1 = 0.0f;
    float ray_normal = 0.0f;
    float dir_coord_0 = 0.0f;
    float dir_coord_1 = 0.0f;
    float dir_normal = 0.0f;
    tangential_components(plane_axis, ray_origin, ray_coord_0, ray_coord_1, ray_normal);
    tangential_components(plane_axis, ray_dir, dir_coord_0, dir_coord_1, dir_normal);
    float distance_to_plane = plane_position - ray_normal;
    bool parallel = fabsf(dir_normal) <= RG_EPS;
    bool points_away = !parallel && (distance_to_plane * dir_normal < 0.0f);
    if (parallel || points_away) {
        return;
    }
    float t_plane = distance_to_plane / dir_normal;
    float hit_coord_0 = ray_coord_0 + t_plane * dir_coord_0;
    float hit_coord_1 = ray_coord_1 + t_plane * dir_coord_1;
    bool before_blocker = t_plane < blocker_dist[tid];
    bool in_bounds =
        hit_coord_0 >= coord_0_min &&
        hit_coord_0 < coord_0_max &&
        hit_coord_1 >= coord_1_min &&
        hit_coord_1 < coord_1_max;
    if (t_plane < 0.0f || !before_blocker || !in_bounds) {
        return;
    }

    int cell_idx = grid_cell_index(
        hit_coord_0,
        hit_coord_1,
        coord_0_min,
        coord_1_min,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1
    );
    int cell_i0 = cell_idx % n_coord_0;
    int cell_i1 = cell_idx / n_coord_0;
    Vec3f cell_pos = point_on_plane(plane_axis, plane_position, coord_0[cell_i0], coord_1[cell_i1]);
    accumulate_reflection_jvp_output(
        cell_idx,
        cell_pos,
        prev_refl_p,
        prev_refl_n,
        prev_tx,
        prev_weight,
        prev_pol_x,
        prev_pol_y,
        prev_pol_z,
        t_prev_refl_p,
        t_prev_refl_n,
        t_prev_tx,
        t_prev_weight,
        t_prev_pol_x,
        t_prev_pol_y,
        t_prev_pol_z,
        prim_idx,
        validate_paths,
        has_mesh_data,
        tri_v0_x,
        tri_v0_y,
        tri_v0_z,
        tri_v1_x,
        tri_v1_y,
        tri_v1_z,
        tri_v2_x,
        tri_v2_y,
        tri_v2_z,
        tri_group_size,
        tri_group_members,
        max_group_size,
        wavelength,
        k,
        out_field_re,
        out_field_im,
        out_pol_x_re,
        out_pol_x_im,
        out_pol_y_re,
        out_pol_y_im,
        out_pol_z_re,
        out_pol_z_im
    );
}

} // namespace

void reflection_grid_jvp(
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
    const float* ray_origin_x,
    const float* ray_origin_y,
    const float* ray_origin_z,
    const float* ray_dir_x,
    const float* ray_dir_y,
    const float* ray_dir_z,
    const float* blocker_dist,
    const float* prev_refl_p_x,
    const float* prev_refl_p_y,
    const float* prev_refl_p_z,
    const float* prev_refl_n_x,
    const float* prev_refl_n_y,
    const float* prev_refl_n_z,
    const float* prev_tx_x,
    const float* prev_tx_y,
    const float* prev_tx_z,
    const float* prev_weight_re,
    const float* prev_weight_im,
    const float* prev_pol_x_re,
    const float* prev_pol_x_im,
    const float* prev_pol_y_re,
    const float* prev_pol_y_im,
    const float* prev_pol_z_re,
    const float* prev_pol_z_im,
    const int* prev_prim_idx,
    int validate_paths,
    int has_mesh_data,
    const float* tri_v0_x,
    const float* tri_v0_y,
    const float* tri_v0_z,
    const float* tri_v1_x,
    const float* tri_v1_y,
    const float* tri_v1_z,
    const float* tri_v2_x,
    const float* tri_v2_y,
    const float* tri_v2_z,
    const int* tri_group_size,
    const int* tri_group_members,
    int max_group_size,
    const float* t_prev_refl_p_x,
    const float* t_prev_refl_p_y,
    const float* t_prev_refl_p_z,
    const float* t_prev_refl_n_x,
    const float* t_prev_refl_n_y,
    const float* t_prev_refl_n_z,
    const float* t_prev_tx_x,
    const float* t_prev_tx_y,
    const float* t_prev_tx_z,
    const float* t_prev_weight_re,
    const float* t_prev_weight_im,
    const float* t_prev_pol_x_re,
    const float* t_prev_pol_x_im,
    const float* t_prev_pol_y_re,
    const float* t_prev_pol_y_im,
    const float* t_prev_pol_z_re,
    const float* t_prev_pol_z_im,
    float wavelength,
    float k,
    int n_rays,
    float* out_field_re,
    float* out_field_im,
    float* out_pol_x_re,
    float* out_pol_x_im,
    float* out_pol_y_re,
    float* out_pol_y_im,
    float* out_pol_z_re,
    float* out_pol_z_im
) {
    if (n_rays <= 0) {
        return;
    }
    constexpr int BLOCK = 256;
    int grid = (n_rays + BLOCK - 1) / BLOCK;
    reflection_grid_jvp_kernel<<<grid, BLOCK>>>(
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
        ray_origin_x,
        ray_origin_y,
        ray_origin_z,
        ray_dir_x,
        ray_dir_y,
        ray_dir_z,
        blocker_dist,
        prev_refl_p_x,
        prev_refl_p_y,
        prev_refl_p_z,
        prev_refl_n_x,
        prev_refl_n_y,
        prev_refl_n_z,
        prev_tx_x,
        prev_tx_y,
        prev_tx_z,
        prev_weight_re,
        prev_weight_im,
        prev_pol_x_re,
        prev_pol_x_im,
        prev_pol_y_re,
        prev_pol_y_im,
        prev_pol_z_re,
        prev_pol_z_im,
        prev_prim_idx,
        validate_paths,
        has_mesh_data,
        tri_v0_x,
        tri_v0_y,
        tri_v0_z,
        tri_v1_x,
        tri_v1_y,
        tri_v1_z,
        tri_v2_x,
        tri_v2_y,
        tri_v2_z,
        tri_group_size,
        tri_group_members,
        max_group_size,
        t_prev_refl_p_x,
        t_prev_refl_p_y,
        t_prev_refl_p_z,
        t_prev_refl_n_x,
        t_prev_refl_n_y,
        t_prev_refl_n_z,
        t_prev_tx_x,
        t_prev_tx_y,
        t_prev_tx_z,
        t_prev_weight_re,
        t_prev_weight_im,
        t_prev_pol_x_re,
        t_prev_pol_x_im,
        t_prev_pol_y_re,
        t_prev_pol_y_im,
        t_prev_pol_z_re,
        t_prev_pol_z_im,
        wavelength,
        k,
        n_rays,
        out_field_re,
        out_field_im,
        out_pol_x_re,
        out_pol_x_im,
        out_pol_y_re,
        out_pol_y_im,
        out_pol_z_re,
        out_pol_z_im
    );
    throw_cuda(cudaGetLastError(), "reflection_grid_jvp_kernel launch");
}

} // namespace witwin::channel::native_ext
