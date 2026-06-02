#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <common/primitives.h>
#include <monitors/field/radio_map_monte_carlo/radio_map_monte_carlo.h>

namespace witwin::channel::native_ext {
namespace {

using common::ceil_div_int;
using common::throw_cuda;

__global__ void radiomap_monte_carlo_scatter_axis_aligned_kernel(
    const float* coord_0,
    const float* coord_1,
    const float* los_power,
    const float* reflection_power,
    const float* diffraction_power,
    float* out_los,
    float* out_reflection,
    float* out_diffraction,
    int n_samples,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_samples) {
        return;
    }

    float sample_coord_0 = coord_0[sample_idx];
    float sample_coord_1 = coord_1[sample_idx];
    if (
        sample_coord_0 < coord_0_min
        || sample_coord_0 > coord_0_max
        || sample_coord_1 < coord_1_min
        || sample_coord_1 > coord_1_max
    ) {
        return;
    }

    int ix = static_cast<int>(floorf((sample_coord_0 - coord_0_min) / cell_size_0));
    int iy = static_cast<int>(floorf((sample_coord_1 - coord_1_min) / cell_size_1));
    ix = max(0, min(ix, n_coord_0 - 1));
    iy = max(0, min(iy, n_coord_1 - 1));
    int cell_idx = iy * n_coord_0 + ix;

    float los = los_power[sample_idx];
    float reflection = reflection_power[sample_idx];
    float diffraction = diffraction_power[sample_idx];
    if (los != 0.0f) {
        atomicAdd(out_los + cell_idx, los);
    }
    if (reflection != 0.0f) {
        atomicAdd(out_reflection + cell_idx, reflection);
    }
    if (diffraction != 0.0f) {
        atomicAdd(out_diffraction + cell_idx, diffraction);
    }
}

} // namespace

void radiomap_monte_carlo_scatter_axis_aligned(
    const float* coord_0,
    const float* coord_1,
    const float* los_power,
    const float* reflection_power,
    const float* diffraction_power,
    float* out_los,
    float* out_reflection,
    float* out_diffraction,
    int n_samples,
    float coord_0_min,
    float coord_0_max,
    float coord_1_min,
    float coord_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    if (
        n_samples <= 0
        || n_coord_0 <= 0
        || n_coord_1 <= 0
        || cell_size_0 <= 0.0f
        || cell_size_1 <= 0.0f
    ) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = ceil_div_int(n_samples, block_size);
    radiomap_monte_carlo_scatter_axis_aligned_kernel<<<grid_size, block_size>>>(
        coord_0,
        coord_1,
        los_power,
        reflection_power,
        diffraction_power,
        out_los,
        out_reflection,
        out_diffraction,
        n_samples,
        coord_0_min,
        coord_0_max,
        coord_1_min,
        coord_1_max,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1
    );
    throw_cuda(cudaGetLastError(), "radiomap_monte_carlo_scatter_axis_aligned_kernel launch");
}

} // namespace witwin::channel::native_ext
