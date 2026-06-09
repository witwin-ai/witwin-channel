#pragma once

#include <raydtorch/diffraction/common.h>

#include <cstdint>

#ifdef __CUDACC__
#  include <optix.h>
#else
#  include <optix.h>
#endif

namespace raydtorch {

/// Launch parameters for compact first-order diffraction path export.
struct DfrPathParams {
    OptixTraversableHandle primary_handle;
    OptixTraversableHandle secondary_handle;
    int split_mode;

    int n_rays;
    int capacity;

    const float *tx_pos_x;
    const float *tx_pos_y;
    const float *tx_pos_z;
    int tx_count;

    const float *rx_pos_x;
    const float *rx_pos_y;
    const float *rx_pos_z;
    int rx_count;

    const uint8_t *active_mask;
    int active_width;
    int state_count;
    int state_limit;
    const int *state_edge_index;
    const float *state_edge_pos_x;
    const float *state_edge_pos_y;
    const float *state_edge_pos_z;
    const float *state_edge_dir_x;
    const float *state_edge_dir_y;
    const float *state_edge_dir_z;
    const float *state_edge_t_min;
    const float *state_edge_t_max;
    const float *state_n0_x;
    const float *state_n0_y;
    const float *state_n0_z;
    const float *state_n1_x;
    const float *state_n1_y;
    const float *state_n1_z;
    const int *state_prim0;
    const int *state_prim1;
    const float *state_exterior_angle;
    const float *state_src_x;
    const float *state_src_y;
    const float *state_src_z;
    const float *state_src_power;

    const float *material_gain;
    const uint8_t *material_valid;
    int material_count;

    float wavelength;
    float k;
    int seed;
    int max_order;
    int strategy_mask;
    int sample_count;
    int return_geom;
    int receiver_model;

    uint8_t *temp_visibility;

    int *out_count;
    uint8_t *out_valid;
    int *out_tx_id;
    int *out_rx_id;
    int *out_order;
    int *out_edge0;
    int *out_edge1;
    int *out_edge2;
    float *out_delay;
    float *out_field_x_re;
    float *out_field_x_im;
    float *out_field_y_re;
    float *out_field_y_im;
    float *out_field_z_re;
    float *out_field_z_im;
    float *out_p0_x;
    float *out_p0_y;
    float *out_p0_z;
    float *out_p1_x;
    float *out_p1_y;
    float *out_p1_z;
    float *out_p2_x;
    float *out_p2_y;
    float *out_p2_z;
};

} // namespace raydtorch

