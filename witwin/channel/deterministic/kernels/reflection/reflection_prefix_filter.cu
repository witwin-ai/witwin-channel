#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <utd/utd_types.h>
#include <utd/utd_math.h>
#include <reflection/reflection_prefix_filter.h>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

__global__ void reflection_prefix_filter_kernel(
    const int* __restrict__ has_reflected_support,
    const float* __restrict__ source_x,
    const float* __restrict__ source_y,
    const float* __restrict__ source_z,
    const float* __restrict__ edge_pos_x,
    const float* __restrict__ edge_pos_y,
    const float* __restrict__ edge_pos_z,
    const float* __restrict__ edge_dir_x,
    const float* __restrict__ edge_dir_y,
    const float* __restrict__ edge_dir_z,
    const float* __restrict__ n0_x,
    const float* __restrict__ n0_y,
    const float* __restrict__ n0_z,
    const float* __restrict__ nn_x,
    const float* __restrict__ nn_y,
    const float* __restrict__ nn_z,
    const float* __restrict__ vec_x_re,
    const float* __restrict__ vec_x_im,
    const float* __restrict__ vec_y_re,
    const float* __restrict__ vec_y_im,
    const float* __restrict__ vec_z_re,
    const float* __restrict__ vec_z_im,
    float wavelength,
    float field_power_threshold,
    int n_pairs,
    int* __restrict__ out_support_mask,
    int* __restrict__ out_keep_mask
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) {
        return;
    }

    bool reflected_support = has_reflected_support[tid] != 0;
    if (!reflected_support) {
        out_support_mask[tid] = 0;
        out_keep_mask[tid] = 0;
        return;
    }

    float3a source = make_f3(source_x[tid], source_y[tid], source_z[tid]);
    float3a edge_pos = make_f3(edge_pos_x[tid], edge_pos_y[tid], edge_pos_z[tid]);
    float3a edge_dir = make_f3(edge_dir_x[tid], edge_dir_y[tid], edge_dir_z[tid]);
    float3a n0 = make_f3(n0_x[tid], n0_y[tid], n0_z[tid]);
    float3a nn = make_f3(nn_x[tid], nn_y[tid], nn_z[tid]);

    float3a direction_from_edge = f3_sub(source, edge_pos);
    float edge_proj_dot = f3_dot(direction_from_edge, edge_dir);
    float3a direction_proj = f3_sub(direction_from_edge, f3_mul(edge_dir, edge_proj_dot));
    float signed_distance_0 = f3_dot(direction_proj, n0);
    float signed_distance_n = f3_dot(direction_proj, nn);
    bool source_exterior =
        (safe_length(direction_proj) > UTD_SMALL_EPS)
        && (signed_distance_0 >= -UTD_SMALL_EPS || signed_distance_n >= -UTD_SMALL_EPS);

    bool support_mask = reflected_support && source_exterior;
    out_support_mask[tid] = support_mask ? 1 : 0;
    if (!support_mask) {
        out_keep_mask[tid] = 0;
        return;
    }

    float dx = edge_pos.x - source.x;
    float dy = edge_pos.y - source.y;
    float dz = edge_pos.z - source.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz) + UTD_EPS;
    float fspl = wavelength / (4.f * UTD_PI * dist);
    float vec_power =
        vec_x_re[tid] * vec_x_re[tid] + vec_x_im[tid] * vec_x_im[tid]
        + vec_y_re[tid] * vec_y_re[tid] + vec_y_im[tid] * vec_y_im[tid]
        + vec_z_re[tid] * vec_z_re[tid] + vec_z_im[tid] * vec_z_im[tid];
    float field_power = vec_power * fspl * fspl;

    out_keep_mask[tid] = field_power > field_power_threshold ? 1 : 0;
}

} // namespace

void reflection_prefix_filter(
    const int* has_reflected_support,
    const float* source_x,
    const float* source_y,
    const float* source_z,
    const float* edge_pos_x,
    const float* edge_pos_y,
    const float* edge_pos_z,
    const float* edge_dir_x,
    const float* edge_dir_y,
    const float* edge_dir_z,
    const float* n0_x,
    const float* n0_y,
    const float* n0_z,
    const float* nn_x,
    const float* nn_y,
    const float* nn_z,
    const float* vec_x_re,
    const float* vec_x_im,
    const float* vec_y_re,
    const float* vec_y_im,
    const float* vec_z_re,
    const float* vec_z_im,
    float wavelength,
    float field_power_threshold,
    int n_pairs,
    int* out_support_mask,
    int* out_keep_mask
) {
    if (n_pairs <= 0) {
        return;
    }

    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    reflection_prefix_filter_kernel<<<grid, BLOCK>>>(
        has_reflected_support,
        source_x,
        source_y,
        source_z,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        n0_x,
        n0_y,
        n0_z,
        nn_x,
        nn_y,
        nn_z,
        vec_x_re,
        vec_x_im,
        vec_y_re,
        vec_y_im,
        vec_z_re,
        vec_z_im,
        wavelength,
        field_power_threshold,
        n_pairs,
        out_support_mask,
        out_keep_mask
    );

    throw_cuda(cudaGetLastError(), "reflection_prefix_filter_kernel launch");
}

} // namespace witwin::channel::native_ext
