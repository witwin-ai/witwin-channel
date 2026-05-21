#pragma once

namespace witwin::channel::native_ext {

void monte_carlo_sparse_coeff_jvp_into(
    const unsigned int* cell_idx,
    const float* tx_coeff_x,
    const float* tx_coeff_y,
    const float* tx_coeff_z,
    const int* vertex_indices,
    const float* vertex_coeff_x,
    const float* vertex_coeff_y,
    const float* vertex_coeff_z,
    int vertex_slot_count,
    const int* material_indices,
    const float* material_coeff_eps,
    const float* material_coeff_sigma,
    int material_slot_count,
    const float* tx_tangent_x,
    const float* tx_tangent_y,
    const float* tx_tangent_z,
    const float* vertex_tangent_x,
    const float* vertex_tangent_y,
    const float* vertex_tangent_z,
    const float* material_tangent_eps,
    const float* material_tangent_sigma,
    float* out_component,
    int n_samples
);

void monte_carlo_sparse_coeff_vjp_into(
    const unsigned int* cell_idx,
    const float* tx_coeff_x,
    const float* tx_coeff_y,
    const float* tx_coeff_z,
    const int* vertex_indices,
    const float* vertex_coeff_x,
    const float* vertex_coeff_y,
    const float* vertex_coeff_z,
    int vertex_slot_count,
    const int* material_indices,
    const float* material_coeff_eps,
    const float* material_coeff_sigma,
    int material_slot_count,
    const float* upstream_component,
    float* out_tx_grad_x,
    float* out_tx_grad_y,
    float* out_tx_grad_z,
    float* out_vertex_grad_x,
    float* out_vertex_grad_y,
    float* out_vertex_grad_z,
    float* out_material_grad_eps,
    float* out_material_grad_sigma,
    int n_samples
);

} // namespace witwin::channel::native_ext
