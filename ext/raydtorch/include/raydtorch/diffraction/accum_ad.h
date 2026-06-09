#pragma once

#include <cstdint>

namespace raydtorch {

struct DfrDirectAccumADParams {
    int n_rays;
    int state_count;
    int material_count;
    int grid_axis;
    float grid_position;
    float grid_coord0_min;
    float grid_coord0_max;
    float grid_coord1_min;
    float grid_coord1_max;
    int grid_resolution0;
    int grid_resolution1;
    float grid_cell_area;
    int direct_samples;
    int keller_samples;
    int suffix_samples;
    float wavelength;
    int seed;
    int n_triangles;

    const uint8_t *tape_active;
    const int *tape_state_idx;
    const int *tape_cell;
    const int *tape_material_idx;
    const float *tape_edge_u;

    const float *state_edge_pos_x;
    const float *state_edge_pos_y;
    const float *state_edge_pos_z;
    const float *state_edge_dir_x;
    const float *state_edge_dir_y;
    const float *state_edge_dir_z;
    const float *state_edge_t_min;
    const float *state_edge_t_max;
    const float *state_src_x;
    const float *state_src_y;
    const float *state_src_z;
    const float *state_wi_x;
    const float *state_wi_y;
    const float *state_wi_z;
    const float *state_src_power;
    const float *state_exterior_angle;
    const int *state_prim0;
    const int *state_prim1;
    const float *tri_p0_x;
    const float *tri_p0_y;
    const float *tri_p0_z;
    const float *tri_e1_x;
    const float *tri_e1_y;
    const float *tri_e1_z;
    const float *tri_e2_x;
    const float *tri_e2_y;
    const float *tri_e2_z;
    const float *tri_fn_x;
    const float *tri_fn_y;
    const float *tri_fn_z;
    const float *material_gain;
    const uint8_t *material_valid;

    const float *dot_state_edge_pos_x;
    const float *dot_state_edge_pos_y;
    const float *dot_state_edge_pos_z;
    const float *dot_state_edge_dir_x;
    const float *dot_state_edge_dir_y;
    const float *dot_state_edge_dir_z;
    const float *dot_state_edge_t_min;
    const float *dot_state_edge_t_max;
    const float *dot_state_src_x;
    const float *dot_state_src_y;
    const float *dot_state_src_z;
    const float *dot_state_wi_x;
    const float *dot_state_wi_y;
    const float *dot_state_wi_z;
    const float *dot_state_src_power;
    const float *dot_state_exterior_angle;
    const float *dot_material_gain;
    const float *dot_tri_p0_x;
    const float *dot_tri_p0_y;
    const float *dot_tri_p0_z;
    const float *dot_tri_fn_x;
    const float *dot_tri_fn_y;
    const float *dot_tri_fn_z;

    float *dot_out_power;
    float *dot_out_field_x_re;

    const float *grad_out_power;
    const float *grad_out_field_x_re;

    float *grad_state_edge_pos_x;
    float *grad_state_edge_pos_y;
    float *grad_state_edge_pos_z;
    float *grad_state_edge_dir_x;
    float *grad_state_edge_dir_y;
    float *grad_state_edge_dir_z;
    float *grad_state_edge_t_min;
    float *grad_state_edge_t_max;
    float *grad_state_src_x;
    float *grad_state_src_y;
    float *grad_state_src_z;
    float *grad_state_wi_x;
    float *grad_state_wi_y;
    float *grad_state_wi_z;
    float *grad_state_src_power;
    float *grad_state_exterior_angle;
    float *grad_material_gain;
    float *grad_tri_p0_x;
    float *grad_tri_p0_y;
    float *grad_tri_p0_z;
    float *grad_tri_fn_x;
    float *grad_tri_fn_y;
    float *grad_tri_fn_z;
};

void dfr_direct_accum_jvp_gpu(const DfrDirectAccumADParams &params);
void dfr_direct_accum_vjp_gpu(const DfrDirectAccumADParams &params);

struct DfrChainAccumADParams {
    int n_rays;
    int state_count;
    int recursive_state_count;
    int material_count;
    int grid_axis;
    float grid_position;
    float grid_coord0_min;
    float grid_coord0_max;
    float grid_coord1_min;
    float grid_coord1_max;
    int grid_resolution0;
    int grid_resolution1;
    float grid_cell_area;
    int direct_samples;
    int keller_samples;
    int suffix_samples;
    int max_order;
    float wavelength;
    int seed;
    int n_triangles;

    const uint8_t *tape_active;
    const int *tape_cell;

    const int *state_edge_index;
    const float *state_edge_pos_x;
    const float *state_edge_pos_y;
    const float *state_edge_pos_z;
    const float *state_edge_dir_x;
    const float *state_edge_dir_y;
    const float *state_edge_dir_z;
    const float *state_edge_t_min;
    const float *state_edge_t_max;
    const float *state_src_x;
    const float *state_src_y;
    const float *state_src_z;
    const float *state_src_power;
    const float *state_exterior_angle;
    const int *state_prim0;
    const int *state_prim1;

    const int *recursive_state_edge_index;
    const float *recursive_state_edge_pos_x;
    const float *recursive_state_edge_pos_y;
    const float *recursive_state_edge_pos_z;
    const float *recursive_state_edge_dir_x;
    const float *recursive_state_edge_dir_y;
    const float *recursive_state_edge_dir_z;
    const float *recursive_state_edge_t_min;
    const float *recursive_state_edge_t_max;
    const float *recursive_state_exterior_angle;
    const int *recursive_state_prim0;
    const int *recursive_state_prim1;

    const float *tri_p0_x;
    const float *tri_p0_y;
    const float *tri_p0_z;
    const float *tri_e1_x;
    const float *tri_e1_y;
    const float *tri_e1_z;
    const float *tri_e2_x;
    const float *tri_e2_y;
    const float *tri_e2_z;
    const float *tri_fn_x;
    const float *tri_fn_y;
    const float *tri_fn_z;

    const float *material_gain;
    const uint8_t *material_valid;

    const float *dot_state_edge_pos_x;
    const float *dot_state_edge_pos_y;
    const float *dot_state_edge_pos_z;
    const float *dot_state_edge_dir_x;
    const float *dot_state_edge_dir_y;
    const float *dot_state_edge_dir_z;
    const float *dot_state_edge_t_min;
    const float *dot_state_edge_t_max;
    const float *dot_state_src_x;
    const float *dot_state_src_y;
    const float *dot_state_src_z;
    const float *dot_state_src_power;
    const float *dot_state_exterior_angle;

    const float *dot_recursive_state_edge_pos_x;
    const float *dot_recursive_state_edge_pos_y;
    const float *dot_recursive_state_edge_pos_z;
    const float *dot_recursive_state_edge_dir_x;
    const float *dot_recursive_state_edge_dir_y;
    const float *dot_recursive_state_edge_dir_z;
    const float *dot_recursive_state_edge_t_min;
    const float *dot_recursive_state_edge_t_max;
    const float *dot_recursive_state_exterior_angle;
    const float *dot_material_gain;
    const float *dot_tri_p0_x;
    const float *dot_tri_p0_y;
    const float *dot_tri_p0_z;
    const float *dot_tri_fn_x;
    const float *dot_tri_fn_y;
    const float *dot_tri_fn_z;

    float *dot_out_power;
    float *dot_out_field_x_re;

    const float *grad_out_power;
    const float *grad_out_field_x_re;

    float *grad_state_edge_pos_x;
    float *grad_state_edge_pos_y;
    float *grad_state_edge_pos_z;
    float *grad_state_edge_dir_x;
    float *grad_state_edge_dir_y;
    float *grad_state_edge_dir_z;
    float *grad_state_edge_t_min;
    float *grad_state_edge_t_max;
    float *grad_state_src_x;
    float *grad_state_src_y;
    float *grad_state_src_z;
    float *grad_state_src_power;
    float *grad_state_exterior_angle;

    float *grad_recursive_state_edge_pos_x;
    float *grad_recursive_state_edge_pos_y;
    float *grad_recursive_state_edge_pos_z;
    float *grad_recursive_state_edge_dir_x;
    float *grad_recursive_state_edge_dir_y;
    float *grad_recursive_state_edge_dir_z;
    float *grad_recursive_state_edge_t_min;
    float *grad_recursive_state_edge_t_max;
    float *grad_recursive_state_exterior_angle;
    float *grad_material_gain;
    float *grad_tri_p0_x;
    float *grad_tri_p0_y;
    float *grad_tri_p0_z;
    float *grad_tri_fn_x;
    float *grad_tri_fn_y;
    float *grad_tri_fn_z;
};

void dfr_chain_accum_jvp_gpu(const DfrChainAccumADParams &params);
void dfr_chain_accum_vjp_gpu(const DfrChainAccumADParams &params);

} // namespace raydtorch

