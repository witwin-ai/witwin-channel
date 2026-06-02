#pragma once

namespace witwin::channel::native_ext {

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
);

} // namespace witwin::channel::native_ext
