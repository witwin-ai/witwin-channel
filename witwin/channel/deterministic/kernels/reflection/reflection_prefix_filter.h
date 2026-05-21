#pragma once

namespace witwin::channel::native_ext {

void reflection_prefix_filter(
    const int* has_reflected_support,
    const float* source_x, const float* source_y, const float* source_z,
    const float* edge_pos_x, const float* edge_pos_y, const float* edge_pos_z,
    const float* edge_dir_x, const float* edge_dir_y, const float* edge_dir_z,
    const float* n0_x, const float* n0_y, const float* n0_z,
    const float* nn_x, const float* nn_y, const float* nn_z,
    const float* vec_x_re, const float* vec_x_im,
    const float* vec_y_re, const float* vec_y_im,
    const float* vec_z_re, const float* vec_z_im,
    float wavelength,
    float field_power_threshold,
    int n_pairs,
    int* out_support_mask,
    int* out_keep_mask
);

} // namespace witwin::channel::native_ext
