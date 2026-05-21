#pragma once

namespace witwin::channel::native_ext {

void monte_carlo_transport_grid_forward(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    float *out_grid,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
);

void monte_carlo_transport_grid_jvp(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const float *t_coord_0,
    const float *t_coord_1,
    const float *t_power,
    float *out_grid,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
);

void monte_carlo_transport_grid_backward(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const float *upstream_grid,
    float *grad_coord_0,
    float *grad_coord_1,
    float *grad_power,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
);

} // namespace witwin::channel::native_ext
