#pragma once

#include <cstdint>

namespace raydtorch {

/// Launch parameters for the EPC field-evaluation kernel. It consumes the geometry
/// produced by an EPC trace (epc_* and per-slot hit/normal arrays) and emits the
/// complex field per ray. All array fields are flat SoA device pointers.
struct ReflEpcFieldParams {
    int n_rays = 0;
    int max_bounces = 0;

    // Path validity/length from the EPC trace this field pass builds on.
    const uint8_t *epc_valid = nullptr;
    const int *epc_bounce_count = nullptr;
    const float *epc_path_length = nullptr;

    const float *ray_ox = nullptr;
    const float *ray_oy = nullptr;
    const float *ray_oz = nullptr;
    const float *rx_x = nullptr;
    const float *rx_y = nullptr;
    const float *rx_z = nullptr;
    int rx_count = 0;

    const float *hit_x = nullptr;
    const float *hit_y = nullptr;
    const float *hit_z = nullptr;
    const float *epc_normal_x = nullptr;
    const float *epc_normal_y = nullptr;
    const float *epc_normal_z = nullptr;
    const int *resolved_prim_ids = nullptr;
    const int *surface_group_ids = nullptr;

    const float *slot_normal_x = nullptr;
    const float *slot_normal_y = nullptr;
    const float *slot_normal_z = nullptr;
    const float *slot_eta_r = nullptr;
    const float *slot_mu_r = nullptr;
    const float *slot_sigma = nullptr;
    const float *slot_gain = nullptr;

    const float *tx_pol_x = nullptr;
    const float *tx_pol_y = nullptr;
    const float *tx_pol_z = nullptr;
    int tx_pol_count = 0;
    float omega = 0.f;
    float wavelength = 0.f;

    // Outputs: per-ray validity/length, the complex (x, y, z) field, optional endpoints,
    // and optional per-slot geometry mirroring ReflEpcFieldData.
    uint8_t *out_valid = nullptr;
    int *out_bounce_count = nullptr;
    float *out_path_length = nullptr;
    float *out_field_x_re = nullptr;
    float *out_field_x_im = nullptr;
    float *out_field_y_re = nullptr;
    float *out_field_y_im = nullptr;
    float *out_field_z_re = nullptr;
    float *out_field_z_im = nullptr;

    float *out_tx_x = nullptr;
    float *out_tx_y = nullptr;
    float *out_tx_z = nullptr;
    float *out_first_hit_x = nullptr;
    float *out_first_hit_y = nullptr;
    float *out_first_hit_z = nullptr;
    float *out_last_hit_x = nullptr;
    float *out_last_hit_y = nullptr;
    float *out_last_hit_z = nullptr;

    float *out_hit_x = nullptr;
    float *out_hit_y = nullptr;
    float *out_hit_z = nullptr;
    float *out_normal_x = nullptr;
    float *out_normal_y = nullptr;
    float *out_normal_z = nullptr;
    int *out_resolved_prim_ids = nullptr;
    int *out_surface_group_ids = nullptr;
};

/// Evaluate the complex reflected field for each ray from precomputed EPC geometry.
void reflection_epc_field_gpu(const ReflEpcFieldParams &params);

} // namespace raydtorch
