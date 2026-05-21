#pragma once

namespace witwin::channel::native_ext {

void monte_carlo_transport_vertex_jvp_into(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const int *vertex_indices,
    const float *coord_0_coeff_x,
    const float *coord_0_coeff_y,
    const float *coord_0_coeff_z,
    const float *coord_1_coeff_x,
    const float *coord_1_coeff_y,
    const float *coord_1_coeff_z,
    int vertex_slot_count,
    const float *vertex_tangent_x,
    const float *vertex_tangent_y,
    const float *vertex_tangent_z,
    float *out_grid,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
);

void monte_carlo_transport_vertex_vjp_into(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const int *vertex_indices,
    const float *coord_0_coeff_x,
    const float *coord_0_coeff_y,
    const float *coord_0_coeff_z,
    const float *coord_1_coeff_x,
    const float *coord_1_coeff_y,
    const float *coord_1_coeff_z,
    int vertex_slot_count,
    const float *upstream_grid,
    float *out_vertex_grad_x,
    float *out_vertex_grad_y,
    float *out_vertex_grad_z,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
);

} // namespace witwin::channel::native_ext
